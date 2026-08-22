"""Completed market-day loader for an asset with an observation-only reference."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Dataset

from dataset import MarketDayDataset, market_context_window_size


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("REFERENCE_DATASET")


def trailing_validation_start(last_date: pd.Timestamp, weeks: int) -> pd.Timestamp:
    """Inclusive start date of the final ``weeks`` calendar weeks."""
    if weeks < 1:
        raise ValueError("validation_weeks must be positive")
    return pd.Timestamp(last_date).normalize() - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1)


class MarketReferenceDataset(Dataset):
    """Return one tradable asset and a time-matched, observation-only reference."""

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        reference_symbol: str,
        window_size: int,
        rollout_size: int,
        limit: int | None = None,
    ):
        self.reference_symbol = str(reference_symbol)
        if not self.reference_symbol:
            raise ValueError("reference_symbol must be non-empty")

        reference_days = days[days.sample_id.eq(self.reference_symbol)].copy()
        if "is_tradable" in reference_days:
            reference_days = reference_days[reference_days.is_tradable.astype(bool)]
        if not len(reference_days):
            raise ValueError(f"reference symbol {self.reference_symbol} has no tradable days")

        asset_days = days[~days.sample_id.eq(self.reference_symbol)].copy()
        if "is_tradable" in asset_days:
            asset_days = asset_days[asset_days.is_tradable.astype(bool)]
        shared_dates = set(reference_days.date)
        asset_days = asset_days[asset_days.date.isin(shared_dates)]

        self.assets = MarketDayDataset(
            asset_days,
            data_dir,
            window_size,
            rollout_size,
            require_full_session=True,
            calendar_days=days,
            limit=limit,
        )
        self.reference = MarketDayDataset(
            reference_days,
            data_dir,
            window_size,
            rollout_size,
            require_full_session=True,
            calendar_days=days[days.sample_id.eq(self.reference_symbol)],
        )
        self.reference_by_date = {
            pd.Timestamp(row.date): index
            for index, row in self.reference.days.iterrows()
        }
        missing_reference = set(pd.to_datetime(self.assets.days.date)).difference(
            self.reference_by_date
        )
        if missing_reference:
            raise ValueError("one or more asset days lack a usable reference session")

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> dict:
        asset = self.assets[index]
        row = self.assets.days.iloc[index]
        reference = self.reference[self.reference_by_date[pd.Timestamp(row.date)]]
        if not np.array_equal(asset["secs"], reference["secs"]):
            raise ValueError(
                f"{asset['_id']} and {self.reference_symbol} are not time-matched on {row.date}"
            )
        return {
            "_id": asset["_id"],
            "prices": asset["prices"],
            "reference_prices": reference["prices"],
            "secs": asset["secs"],
        }


def make_reference_dataloader(
    cfg_data: DictConfig, cfg_split: DictConfig, seed: int
) -> DataLoader:
    del seed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    days = pd.read_csv(str(cfg_data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    context_days = int(cfg_data.context_days)
    expected_window = market_context_window_size(context_days)
    if int(cfg_data.loader_window_size) != expected_window:
        raise ValueError(
            f"loader_window_size must be {expected_window} for context_days={context_days}"
        )

    last_date = days.date.max()
    validation_start = trailing_validation_start(last_date, int(cfg_data.validation_weeks))
    configured_start = cfg_data.get("date_val", None)
    if configured_start is not None and pd.Timestamp(str(configured_start)) != validation_start:
        raise ValueError(
            f"date_val must be {validation_start.date()} for the trailing "
            f"{int(cfg_data.validation_weeks)} weeks ending {last_date.date()}"
        )
    if cfg_split.split == "val":
        selected = days.date >= validation_start
    elif cfg_split.split == "train":
        selected = days.date < validation_start
    else:
        raise ValueError("split must be 'train' or 'val'")

    # Preserve all dates for schedule completion, while distributing only the
    # selected target rows across ranks.
    period_days = days[selected]
    reference_symbol = str(cfg_data.reference_symbol)
    asset_targets = period_days[~period_days.sample_id.eq(reference_symbol)]
    rank_indices = np.array_split(np.arange(len(asset_targets)), world_size)[rank]
    asset_targets = asset_targets.iloc[rank_indices]
    # Every rank needs the complete SPY date map even though asset targets are
    # partitioned. SPY is observation-only and never appears in ``assets``.
    reference_targets = period_days[period_days.sample_id.eq(reference_symbol)]
    target_days = pd.concat((asset_targets, reference_targets), ignore_index=True)
    calendar_and_targets = days.copy()
    selected_keys = set(zip(target_days.sample_id, target_days.date))
    calendar_and_targets["is_tradable"] = [
        bool(tradable) and (sample_id, date) in selected_keys
        for sample_id, date, tradable in zip(
            calendar_and_targets.sample_id,
            calendar_and_targets.date,
            calendar_and_targets.is_tradable,
        )
    ]
    dataset = MarketReferenceDataset(
        days=calendar_and_targets,
        data_dir=str(cfg_data.data_dir),
        reference_symbol=str(cfg_data.reference_symbol),
        window_size=int(cfg_data.loader_window_size),
        rollout_size=int(cfg_data.rollout_size),
        limit=cfg_split.get("samples_num", None),
    )
    logger.info(
        "Reference samples found, samples_num=%d, split=%s, reference=%s, "
        "validation_start=%s, last_date=%s",
        len(dataset),
        cfg_split.split,
        cfg_data.reference_symbol,
        validation_start.date(),
        last_date.date(),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg_split.batch_size),
        num_workers=int(cfg_split.workers_num),
        shuffle=bool(cfg_split.get("should_augment", False)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(cfg_split.workers_num) > 0,
    )
