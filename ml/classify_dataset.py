"""Anchored stock/reference samples for binary classification tasks.

One sample is a symbol, a session, and a random time of day inside regular
hours. The observation is the trailing window ending at that anchor and the
label tick can use either the same clock on a later session or an explicitly
configured target clock such as next-session 09:45.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Dataset

from .dataset import EXTENDED_SESSION_BARS
from .reference_dataset import trailing_validation_start
from .week_dataset import forward_filled_prices, read_universe


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("CLASSIFY_DATASET")

EXTENDED_OPEN_MINUTE = 4 * 60
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60


def session_tick_count(tick_minutes: int) -> int:
    if tick_minutes < 1 or EXTENDED_SESSION_BARS % tick_minutes:
        raise ValueError(f"tick_minutes must divide {EXTENDED_SESSION_BARS}")
    return EXTENDED_SESSION_BARS // tick_minutes


def regular_hours_offsets(tick_minutes: int) -> tuple[int, int]:
    """Inclusive tick offsets of 09:30 and 16:00 inside an extended session."""
    for minute in (RTH_OPEN_MINUTE, RTH_CLOSE_MINUTE):
        if (minute - EXTENDED_OPEN_MINUTE) % tick_minutes:
            raise ValueError(f"{tick_minutes}-minute grid does not land on {minute // 60:02d}:00")
    return (
        (RTH_OPEN_MINUTE - EXTENDED_OPEN_MINUTE) // tick_minutes,
        (RTH_CLOSE_MINUTE - EXTENDED_OPEN_MINUTE) // tick_minutes,
    )


def classification_split_mask(
    dates: pd.Series,
    split: str,
    validation_start: pd.Timestamp,
    horizon_days: int,
) -> pd.Series:
    """Select anchors while purging labels that would enter validation.

    If validation begins at exchange session ``V`` and the label is read at
    ``anchor + horizon_days``, the final training anchor must be strictly before
    ``V - horizon_days``. Thus every training label is still pre-validation;
    the intervening anchors are intentionally unused.
    """
    horizon = int(horizon_days)
    if horizon < 1:
        raise ValueError("horizon_days must be positive")
    normalized_dates = pd.to_datetime(dates)
    calendar = np.sort(normalized_dates.unique())
    validation_index = int(np.searchsorted(calendar, pd.Timestamp(validation_start), side="left"))
    if validation_index >= len(calendar):
        raise ValueError("validation starts after the final exchange session")

    if split == "val":
        return normalized_dates >= pd.Timestamp(validation_start)
    if split != "train":
        raise ValueError("split must be 'train' or 'val'")

    first_purged_index = validation_index - horizon
    if first_purged_index < 1:
        raise ValueError("validation starts too early to leave any training anchors")
    return normalized_dates < calendar[first_purged_index]


class RelativeDirectionDataset(Dataset):
    """Windows of the stock/reference log ratio, labelled by its future sign."""

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        reference_symbol: str,
        context_days: int = 10,
        tick_minutes: int = 10,
        window_size: int = 960,
        horizon_days: int = 2,
        should_augment: bool = False,
        targets: pd.DataFrame | None = None,
        limit: int | None = None,
        seed: int = 0,
        anchor_minute: int | None = None,
        target_minute: int | None = None,
    ):
        self.data_dir = str(data_dir)
        self.reference_symbol = str(reference_symbol)
        self.context_days = int(context_days)
        self.tick_minutes = int(tick_minutes)
        self.tick_seconds = self.tick_minutes * 60
        self.window_size = int(window_size)
        self.horizon_days = int(horizon_days)
        self.should_augment = bool(should_augment)
        self.seed = int(seed)
        self.day_ticks = session_tick_count(self.tick_minutes)
        self.first_offset, self.last_offset = regular_hours_offsets(self.tick_minutes)
        self.fixed_anchor_offset: int | None = None
        if anchor_minute is not None:
            minute = int(anchor_minute)
            if not RTH_OPEN_MINUTE <= minute <= RTH_CLOSE_MINUTE:
                raise ValueError("anchor minute must be inside 09:30..16:00 Eastern")
            if (minute - EXTENDED_OPEN_MINUTE) % self.tick_minutes:
                raise ValueError("anchor minute must lie on the configured tick grid")
            self.fixed_anchor_offset = (minute - EXTENDED_OPEN_MINUTE) // self.tick_minutes
        self.fixed_target_offset: int | None = None
        if target_minute is not None:
            minute = int(target_minute)
            if not RTH_OPEN_MINUTE <= minute <= RTH_CLOSE_MINUTE:
                raise ValueError("target minute must be inside 09:30..16:00 Eastern")
            if (minute - EXTENDED_OPEN_MINUTE) % self.tick_minutes:
                raise ValueError("target minute must lie on the configured tick grid")
            self.fixed_target_offset = (minute - EXTENDED_OPEN_MINUTE) // self.tick_minutes
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be positive")
        # The window ends at the anchor, which sits at least at 09:30 of the
        # current session, so the context must reach back far enough for the
        # earliest permitted anchor.
        available = self.context_days * self.day_ticks + self.first_offset + 1
        if self.window_size > available:
            raise ValueError(
                f"window_size={self.window_size} exceeds the {available} ticks available "
                f"at the earliest anchor for context_days={self.context_days}"
            )

        frame = days.loc[:, ["sample_id", "date", "context_sod_sec", "is_tradable"]].copy()
        frame["date"] = pd.to_datetime(frame.date)
        frame["is_tradable"] = frame.is_tradable.fillna(False).astype(bool)

        # Session boundaries are a property of the date, not the symbol, so one
        # global calendar indexes every symbol's sessions.
        calendar = frame.loc[:, ["date", "context_sod_sec"]].drop_duplicates()
        if calendar.date.duplicated().any():
            raise ValueError("symbols disagree about one or more session boundaries")
        calendar = calendar.sort_values("date").reset_index(drop=True)
        self.context_sod = calendar.context_sod_sec.to_numpy(dtype=np.int64)
        self.calendar_dates = calendar.date.to_numpy()
        position = pd.Series(np.arange(len(calendar)), index=calendar.date)
        frame["session"] = frame.date.map(position).to_numpy()

        reference_rows = frame[frame.sample_id.eq(self.reference_symbol)]
        if not len(reference_rows):
            raise ValueError(f"reference symbol {self.reference_symbol} is absent")
        self.reference_tradable = np.zeros(len(calendar), dtype=bool)
        self.reference_tradable[reference_rows.session.to_numpy()] = (
            reference_rows.is_tradable.to_numpy()
        )
        self.samples = self._build_samples(frame, targets)
        if limit is not None:
            self.samples = self.samples.iloc[: int(limit)].reset_index(drop=True)
        if not len(self.samples):
            raise ValueError("no anchored samples satisfy the context and horizon requirements")

    def _build_samples(
        self, frame: pd.DataFrame, targets: pd.DataFrame | None
    ) -> pd.DataFrame:
        assets = frame[~frame.sample_id.eq(self.reference_symbol)]
        pieces = []
        for sample_id, group in assets.groupby("sample_id", sort=False):
            ordered = group.sort_values("session")
            sessions = ordered.session.to_numpy()
            if sessions.size < self.context_days + self.horizon_days + 1:
                continue
            if np.any(np.diff(sessions) != 1):
                raise ValueError(f"{sample_id} has a gap in its calendar rows")
            tradable = ordered.is_tradable.to_numpy()
            # An anchor needs its own context behind it and a labelled session
            # ahead of it, and both sessions must be tradable for the symbol and
            # for the reference.
            count = sessions.size
            index = np.arange(count)
            usable = (
                (index >= self.context_days)
                & (index + self.horizon_days < count)
                & tradable
            )
            usable[: self.context_days] = False
            ahead = np.zeros(count, dtype=bool)
            ahead[: count - self.horizon_days] = tradable[self.horizon_days :]
            usable &= ahead
            anchors = sessions[usable]
            anchors = anchors[
                self.reference_tradable[anchors]
                & self.reference_tradable[anchors + self.horizon_days]
            ]
            if not anchors.size:
                continue
            pieces.append(pd.DataFrame({"sample_id": sample_id, "session": anchors}))
        if not pieces:
            return pd.DataFrame(columns=("sample_id", "session"))
        samples = pd.concat(pieces, ignore_index=True)
        samples["date"] = self.calendar_dates[samples.session.to_numpy()]
        if targets is not None:
            samples = samples.merge(
                targets.loc[:, ["sample_id", "date"]].drop_duplicates(),
                on=("sample_id", "date"),
                how="inner",
            )
        return samples.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def anchor_offset(self, index: int) -> int:
        """Tick offset of the anchor inside its session.

        Training draws a fresh time of day each visit; validation derives one
        from the row index so its number does not move between runs.
        """
        low, high = self.first_offset, self.last_offset
        if self.fixed_anchor_offset is not None:
            return self.fixed_anchor_offset
        if self.should_augment:
            return int(np.random.randint(low, high + 1))
        generator = np.random.default_rng(self.seed + int(index))
        return int(generator.integers(low, high + 1))

    def sample_seconds(
        self, session: int, offset: int, minute_shift: int = 0
    ) -> np.ndarray:
        """Window timestamps ending at the anchor, then the labelled tick.

        ``minute_shift`` supports evaluation between training-grid points while
        retaining the same 10-minute cadence and 960-value input shape.
        """
        shift = int(minute_shift)
        if not 0 <= shift < self.tick_minutes:
            raise ValueError("minute_shift must be in [0, tick_minutes)")
        first = session - self.context_days
        grid = np.concatenate(
            [
                self.context_sod[day]
                + shift * 60
                + np.arange(self.day_ticks, dtype=np.int64) * self.tick_seconds
                for day in range(first, session + 1)
            ]
        )
        anchor = self.context_days * self.day_ticks + offset
        window = grid[anchor - self.window_size + 1 : anchor + 1]
        if window.size != self.window_size:
            raise AssertionError("window construction produced the wrong length")
        target_offset = offset if self.fixed_target_offset is None else self.fixed_target_offset
        target = (
            self.context_sod[session + self.horizon_days]
            + shift * 60
            + target_offset * self.tick_seconds
        )
        return np.concatenate((window, [target]))

    def __getitem__(self, index: int) -> dict:
        row = self.samples.iloc[index]
        offset = self.anchor_offset(index)
        secs = self.sample_seconds(int(row.session), offset)
        prices = forward_filled_prices(self.data_dir, str(row.sample_id), secs)
        reference = forward_filled_prices(self.data_dir, self.reference_symbol, secs)
        # 0.0 at 09:30 and 1.0 at 16:00, so the model knows how far into the
        # session its anchor sits without seeing a clock in the window.
        progress = (offset - self.first_offset) / (self.last_offset - self.first_offset)
        target_offset = offset if self.fixed_target_offset is None else self.fixed_target_offset
        anchor_minute = EXTENDED_OPEN_MINUTE + offset * self.tick_minutes
        target_minute = EXTENDED_OPEN_MINUTE + target_offset * self.tick_minutes
        return {
            "_id": str(row.sample_id),
            "date": str(pd.Timestamp(row.date).date()),
            "target_date": str(
                pd.Timestamp(self.calendar_dates[int(row.session) + self.horizon_days]).date()
            ),
            "weekday": np.int64(pd.Timestamp(row.date).dayofweek),
            "anchor_time": f"{anchor_minute // 60:02d}:{anchor_minute % 60:02d}",
            "target_time": f"{target_minute // 60:02d}:{target_minute % 60:02d}",
            "prices": prices.astype(np.float32),
            "reference_prices": reference.astype(np.float32),
            "secs": secs,
            "anchor_progress": np.float32(progress),
        }


def make_classify_dataloader(
    cfg_data: DictConfig, cfg_split: DictConfig, seed: int
) -> DataLoader:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    days = pd.read_csv(str(cfg_data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    horizon = int(cfg_data.horizon_days)
    last_date = days.date.max()
    validation_start = trailing_validation_start(last_date, int(cfg_data.validation_weeks))
    configured_start = cfg_data.get("date_val", None)
    if configured_start is not None and pd.Timestamp(str(configured_start)) != validation_start:
        raise ValueError(
            f"date_val must be {validation_start.date()} for the trailing "
            f"{int(cfg_data.validation_weeks)} weeks ending {last_date.date()}"
        )

    selected = classification_split_mask(
        days.date, str(cfg_split.split), validation_start, horizon
    )

    targets = days.loc[selected, ["sample_id", "date"]].drop_duplicates()
    reference_symbol = str(cfg_data.reference_symbol)
    targets = targets[~targets.sample_id.eq(reference_symbol)]
    universe = cfg_data.get("val_universe", None)
    if cfg_split.split == "val" and universe is not None:
        symbols = read_universe(str(universe.path), universe.get("size", None), reference_symbol)
        targets = targets[targets.sample_id.isin(symbols)]
    targets = targets.reset_index(drop=True)
    rank_indices = np.array_split(np.arange(len(targets)), world_size)[rank]
    targets = targets.iloc[rank_indices]

    dataset = RelativeDirectionDataset(
        days=days,
        data_dir=str(cfg_data.data_dir),
        reference_symbol=reference_symbol,
        context_days=int(cfg_data.context_days),
        tick_minutes=int(cfg_data.tick_minutes),
        window_size=int(cfg_data.window_size),
        horizon_days=horizon,
        should_augment=bool(cfg_split.get("should_augment", False)),
        targets=targets,
        limit=cfg_split.get("samples_num", None),
        seed=seed,
        anchor_minute=cfg_data.get("anchor_minute", None),
        target_minute=cfg_data.get("target_minute", None),
    )
    if cfg_split.split == "train":
        # Defense in depth: the split mask and the dataset's session calendar
        # must agree that no training answer reaches the held-out period.
        target_sessions = dataset.samples.session.to_numpy(dtype=np.int64) + horizon
        target_dates = pd.to_datetime(dataset.calendar_dates[target_sessions])
        if (target_dates >= validation_start).any():
            raise AssertionError("a training label crosses into the validation period")
    logger.info(
        "Classification samples found, samples_num=%d, split=%s, horizon_days=%d, "
        "window_size=%d, validation_start=%s",
        len(dataset),
        cfg_split.split,
        horizon,
        int(cfg_data.window_size),
        validation_start.date(),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg_split.batch_size),
        num_workers=int(cfg_split.workers_num),
        shuffle=bool(
            cfg_split.get("shuffle", cfg_split.get("should_augment", False))
        ),
        pin_memory=True,
        drop_last=cfg_split.split != "val",
        persistent_workers=int(cfg_split.workers_num) > 0,
    )
