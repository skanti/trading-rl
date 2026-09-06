"""Mon--Fri week loader on a 10-minute grid with an observation-only reference.

The intraday policy decides once per minute and is liquidated at 16:00. This
loader instead returns one complete trading week: five consecutive regular
sessions concatenated end to end, sampled every ``tick_minutes``. Decisions may
therefore carry inventory across the four weeknights, while the final grid point
is Friday's 16:00 close, so no position can survive into a weekend.
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
from .price_utils import forward_fill_positions
from .reference_dataset import trailing_validation_start


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("WEEK_DATASET")

FULL_SESSION_INTERVALS = 390  # 09:30 through 16:00 Eastern.
SESSIONS_PER_WEEK = 5
SCHEDULE_COLUMNS = ("sod_sec", "eod_sec", "context_sod_sec", "context_eod_sec")


def read_universe(path: str, size: int | None, exclude: str | None = None) -> tuple[str, ...]:
    """First ``size`` symbols of a rank-ordered ticker list.

    ``tickers_all.txt`` is ordered by liquidity, so the head of the file is the
    most-traded names. The reference symbol leads that file and is dropped, so
    ``size=50`` means fifty tradable stocks rather than forty-nine plus SPY.
    Relative paths resolve against this module, not the working directory.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent / resolved
    symbols = [line.strip() for line in resolved.read_text().splitlines() if line.strip()]
    if exclude is not None:
        symbols = [symbol for symbol in symbols if symbol != exclude]
    if size is not None:
        if int(size) < 1:
            raise ValueError("universe size must be positive")
        if len(symbols) < int(size):
            raise ValueError(f"{resolved} holds {len(symbols)} tradable symbols, fewer than {size}")
        symbols = symbols[: int(size)]
    if not symbols:
        raise ValueError(f"{resolved} yielded no symbols")
    return tuple(symbols)


def forward_filled_prices(data_dir: str, sample_id: str, secs: np.ndarray) -> np.ndarray:
    """Last valid trade price at or before each timestamp.

    A missing bar means no new trade-derived observation, so the most recent
    print is carried forward. This is the single price convention every loader
    and baseline in this directory resolves prices with.
    """
    source = np.load(f"{data_dir}/{sample_id}.npy", mmap_mode="r")
    position = forward_fill_positions(source, secs, sample_id)
    prices = np.asarray(source[position, 1], dtype=np.float64) / 1000.0
    return prices


def ticks_per_context_day(tick_minutes: int) -> int:
    """Context bars kept from one 04:00--19:59 extended session."""
    if tick_minutes < 1 or EXTENDED_SESSION_BARS % tick_minutes:
        raise ValueError(f"tick_minutes must divide {EXTENDED_SESSION_BARS}")
    return EXTENDED_SESSION_BARS // tick_minutes


def ticks_per_session(tick_minutes: int) -> int:
    """Grid points in one regular session, counting both 09:30 and 16:00."""
    if tick_minutes < 1 or FULL_SESSION_INTERVALS % tick_minutes:
        raise ValueError(f"tick_minutes must divide {FULL_SESSION_INTERVALS}")
    return FULL_SESSION_INTERVALS // tick_minutes + 1


def week_context_ticks(context_days: int, tick_minutes: int) -> int:
    if context_days < 1:
        raise ValueError("context_days must be positive")
    return int(context_days) * ticks_per_context_day(tick_minutes)


def week_rollout_size(tick_minutes: int, sessions_per_week: int = SESSIONS_PER_WEEK) -> int:
    """Decisions in one week.

    The week grid holds ``sessions_per_week * ticks_per_session`` points. Every
    point except the last is a decision; the last is Friday's 16:00 close, which
    only prices the final interval and liquidates.
    """
    if sessions_per_week < 1:
        raise ValueError("sessions_per_week must be positive")
    return int(sessions_per_week) * ticks_per_session(tick_minutes) - 1


class MarketWeekDataset(Dataset):
    """Return a 10-day context followed by one complete Mon--Fri week.

    ``days`` is the per-symbol day index produced by ``scripts/split.py``. Rows
    with ``is_tradable=False`` still supply completed context, exactly as in the
    intraday loader, but a week is only offered when all five of its sessions
    are tradable and contiguous in the symbol's own session sequence.
    """

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        context_days: int = 10,
        tick_minutes: int = 10,
        sessions_per_week: int = SESSIONS_PER_WEEK,
        targets: pd.DataFrame | None = None,
        limit: int | None = None,
    ):
        missing = set(SCHEDULE_COLUMNS).difference(days.columns)
        if missing:
            raise ValueError(f"day metadata lacks schedule columns: {sorted(missing)}")
        self.data_dir = str(data_dir)
        self.context_days = int(context_days)
        self.tick_minutes = int(tick_minutes)
        self.tick_seconds = self.tick_minutes * 60
        self.sessions_per_week = int(sessions_per_week)
        self.context_day_ticks = ticks_per_context_day(self.tick_minutes)
        self.session_ticks = ticks_per_session(self.tick_minutes)
        self.context_ticks = week_context_ticks(self.context_days, self.tick_minutes)
        self.rollout_size = week_rollout_size(self.tick_minutes, self.sessions_per_week)

        frame = days.loc[:, ["sample_id", "date", "is_tradable", *SCHEDULE_COLUMNS]].copy()
        frame["date"] = pd.to_datetime(frame.date)
        frame = frame.sort_values(["sample_id", "date"], kind="stable").reset_index(drop=True)
        if frame.duplicated(("sample_id", "date")).any():
            raise ValueError("day metadata contains duplicate symbol-days")
        frame["is_tradable"] = frame.is_tradable.fillna(False).astype(bool)
        frame["session_index"] = frame.groupby("sample_id", sort=False).cumcount()

        # Sessions are indexed per symbol, so the schedule can be sliced by
        # position without another search once a week has been selected.
        self.schedule: dict[str, tuple[np.ndarray, ...]] = {}
        for sample_id, group in frame.groupby("sample_id", sort=False):
            self.schedule[str(sample_id)] = tuple(
                group[column].to_numpy(dtype=np.int64) for column in SCHEDULE_COLUMNS
            )

        self.weeks = self._build_weeks(frame, targets)
        if limit is not None:
            self.weeks = self.weeks.iloc[: int(limit)].reset_index(drop=True)
        if not len(self.weeks):
            raise ValueError("no complete market weeks satisfy the context and week requirements")

    def _build_weeks(
        self, frame: pd.DataFrame, targets: pd.DataFrame | None
    ) -> pd.DataFrame:
        weekday = frame.date.dt.dayofweek
        frame = frame.assign(
            week_start=frame.date - pd.to_timedelta(weekday, unit="D"), weekday=weekday
        )
        # A week counts as complete only when the exchange calendar itself holds
        # five sessions. Holiday weeks are dropped rather than padded, because a
        # shorter rollout would not share the fixed policy input shape.
        calendar = frame.loc[:, ["date", "week_start"]].drop_duplicates()
        calendar_sizes = calendar.groupby("week_start").size()
        complete = calendar_sizes.index[calendar_sizes == self.sessions_per_week]
        frame = frame[frame.week_start.isin(complete)]

        grouped = frame.groupby(["sample_id", "week_start"], sort=False)
        summary = grouped.agg(
            sessions=("is_tradable", "size"),
            tradable=("is_tradable", "sum"),
            first_index=("session_index", "min"),
            last_index=("session_index", "max"),
            first_weekday=("weekday", "min"),
            last_weekday=("weekday", "max"),
        ).reset_index()
        span = self.sessions_per_week - 1
        selected = summary[
            summary.sessions.eq(self.sessions_per_week)
            & summary.tradable.eq(self.sessions_per_week)
            # Contiguous in the symbol's own sequence, so the week has no gap
            # that a preceding-session slice would silently skip over.
            & summary.last_index.sub(summary.first_index).eq(span)
            & summary.first_weekday.eq(0)
            & summary.last_weekday.eq(span)
            # Ten preceding sessions must exist for this symbol to fill context.
            & summary.first_index.ge(self.context_days)
        ]
        weeks = selected.loc[:, ["sample_id", "week_start", "first_index"]]
        if targets is not None:
            weeks = weeks.merge(
                targets.loc[:, ["sample_id", "week_start"]].drop_duplicates(),
                on=("sample_id", "week_start"),
                how="inner",
            )
        return weeks.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.weeks)

    def expected_secs(self, sample_id: str, first_index: int) -> np.ndarray:
        """Context grid followed by the Mon--Fri regular-session grid."""
        sod, eod, context_sod, context_eod = self.schedule[sample_id]
        step = self.tick_seconds
        context = [
            np.arange(context_sod[index], context_eod[index] + 1, step, dtype=np.int64)
            for index in range(first_index - self.context_days, first_index)
        ]
        week = [
            np.arange(sod[index], eod[index] + 1, step, dtype=np.int64)
            for index in range(first_index, first_index + self.sessions_per_week)
        ]
        for grid in context:
            if grid.size != self.context_day_ticks:
                raise ValueError(f"{sample_id} has an unexpected extended-session length")
        for grid in week:
            if grid.size != self.session_ticks:
                raise ValueError(f"{sample_id} has an unexpected regular-session length")
        secs = np.concatenate(context + week)
        if secs.size != self.context_ticks + self.rollout_size + 1:
            raise AssertionError("week grid construction produced the wrong length")
        if not np.all(np.diff(secs) > 0):
            raise ValueError(f"{sample_id} week grid is not strictly increasing")
        return secs

    def __getitem__(self, index: int) -> dict:
        row = self.weeks.iloc[index]
        sample_id = str(row.sample_id)
        secs = self.expected_secs(sample_id, int(row.first_index))

        source = np.load(f"{self.data_dir}/{sample_id}.npy", mmap_mode="r")
        if source.ndim != 2 or source.shape[1] < 3:
            raise ValueError(f"{sample_id} must contain [seconds, price_mills, volume]")
        source_secs = np.ascontiguousarray(source[:, 0]).astype(np.int64)

        # A missing bar means no new trade-derived observation, so the last
        # observed price is carried forward. Volume is the sum over the bucket
        # that ends at the grid point, which is the aggregate a 10-minute bar
        # would report.
        stop = np.searchsorted(source_secs, secs, side="right")
        start = np.searchsorted(source_secs, secs - self.tick_seconds, side="right")
        if (stop < 1).any():
            raise ValueError(f"{sample_id} cannot forward-fill its first context price")
        prices = np.asarray(source[stop - 1, 1], dtype=np.float32) / 1000.0
        # Only the ~12 calendar days the grid spans are summed; cumulating a
        # decade of bars for every sample would dominate loader time.
        low, high = int(start.min()), int(stop.max())
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.asarray(source[low:high, 2], dtype=np.float64)))
        )
        volumes = (cumulative[stop - low] - cumulative[start - low]).astype(np.float32)

        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"{sample_id} contains non-positive or non-finite prices")
        if not np.isfinite(volumes).all() or (volumes < 0).any():
            raise ValueError(f"{sample_id} contains negative or non-finite volumes")
        return {
            "_id": sample_id,
            "week_start": str(pd.Timestamp(row.week_start).date()),
            "prices": prices,
            "volumes": volumes,
            "secs": secs,
        }


class WeekReferenceDataset(Dataset):
    """One tradable asset plus a time-matched, observation-only reference week."""

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        reference_symbol: str,
        context_days: int = 10,
        tick_minutes: int = 10,
        sessions_per_week: int = SESSIONS_PER_WEEK,
        targets: pd.DataFrame | None = None,
        limit: int | None = None,
    ):
        self.reference_symbol = str(reference_symbol)
        if not self.reference_symbol:
            raise ValueError("reference_symbol must be non-empty")
        is_reference = days.sample_id.eq(self.reference_symbol)
        if not is_reference.any():
            raise ValueError(f"reference symbol {self.reference_symbol} is absent from the day index")

        self.reference = MarketWeekDataset(
            days[is_reference],
            data_dir,
            context_days=context_days,
            tick_minutes=tick_minutes,
            sessions_per_week=sessions_per_week,
        )
        reference_weeks = self.reference.weeks.loc[:, ["week_start"]]
        asset_targets = targets
        if asset_targets is not None:
            asset_targets = asset_targets[~asset_targets.sample_id.eq(self.reference_symbol)]
            asset_targets = asset_targets.merge(reference_weeks, on="week_start", how="inner")
        self.assets = MarketWeekDataset(
            days[~is_reference],
            data_dir,
            context_days=context_days,
            tick_minutes=tick_minutes,
            sessions_per_week=sessions_per_week,
            targets=asset_targets,
            limit=limit,
        )
        if asset_targets is None:
            keep = self.assets.weeks.week_start.isin(set(reference_weeks.week_start))
            self.assets.weeks = self.assets.weeks[keep].reset_index(drop=True)
            if not len(self.assets.weeks):
                raise ValueError("no asset week overlaps a usable reference week")
        self.reference_by_week = {
            pd.Timestamp(week): index
            for index, week in enumerate(self.reference.weeks.week_start)
        }
        self.context_ticks = self.assets.context_ticks
        self.rollout_size = self.assets.rollout_size

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> dict:
        asset = self.assets[index]
        week_start = pd.Timestamp(self.assets.weeks.iloc[index].week_start)
        reference = self.reference[self.reference_by_week[week_start]]
        if not np.array_equal(asset["secs"], reference["secs"]):
            raise ValueError(
                f"{asset['_id']} and {self.reference_symbol} are not time-matched "
                f"in the week of {asset['week_start']}"
            )
        return {
            "_id": asset["_id"],
            "week_start": asset["week_start"],
            "prices": asset["prices"],
            "volumes": asset["volumes"],
            "reference_prices": reference["prices"],
            "secs": asset["secs"],
        }


def week_targets(days: pd.DataFrame) -> pd.DataFrame:
    """Symbol/week keys for every row, used to partition splits and ranks."""
    dates = pd.to_datetime(days.date)
    week_start = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    return pd.DataFrame({"sample_id": days.sample_id, "week_start": week_start}).drop_duplicates()


def make_week_dataloader(cfg_data: DictConfig, cfg_split: DictConfig, seed: int) -> DataLoader:
    del seed  # DataLoader sampling uses PyTorch's process seed.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    days = pd.read_csv(str(cfg_data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    tick_minutes = int(cfg_data.tick_minutes)
    context_days = int(cfg_data.context_days)
    expected_context = week_context_ticks(context_days, tick_minutes)
    expected_rollout = week_rollout_size(tick_minutes)
    if int(cfg_data.context_ticks) != expected_context:
        raise ValueError(
            f"context_ticks must be {expected_context} for context_days={context_days} "
            f"at {tick_minutes}-minute resolution"
        )
    if int(cfg_data.rollout_size) != expected_rollout:
        raise ValueError(
            f"rollout_size must be {expected_rollout} for a five-session week at "
            f"{tick_minutes}-minute resolution"
        )

    last_date = days.date.max()
    validation_start = trailing_validation_start(last_date, int(cfg_data.validation_weeks))
    configured_start = cfg_data.get("date_val", None)
    if configured_start is not None and pd.Timestamp(str(configured_start)) != validation_start:
        raise ValueError(
            f"date_val must be {validation_start.date()} for the trailing "
            f"{int(cfg_data.validation_weeks)} weeks ending {last_date.date()}"
        )

    targets = week_targets(days)
    # A week is assigned whole. Training weeks must finish before validation
    # begins, so no rollout ever spans the split boundary.
    week_end = targets.week_start + pd.Timedelta(days=SESSIONS_PER_WEEK - 1)
    if cfg_split.split == "val":
        targets = targets[targets.week_start >= validation_start]
    elif cfg_split.split == "train":
        targets = targets[week_end < validation_start]
    else:
        raise ValueError("split must be 'train' or 'val'")

    reference_symbol = str(cfg_data.reference_symbol)
    asset_targets = targets[~targets.sample_id.eq(reference_symbol)]
    # Validation is pinned to a fixed, liquid universe so its number means the
    # same thing at every step and can be compared with the rule-based
    # baselines, which are quoted on the same names.
    universe = cfg_data.get("val_universe", None)
    if cfg_split.split == "val" and universe is not None:
        symbols = read_universe(
            str(universe.path), universe.get("size", None), reference_symbol
        )
        asset_targets = asset_targets[asset_targets.sample_id.isin(symbols)]
        missing = set(symbols).difference(asset_targets.sample_id)
        if missing:
            logger.warning(
                "Validation universe symbols absent from the split, missing_num=%d, first=%s",
                len(missing),
                sorted(missing)[:5],
            )
    asset_targets = asset_targets.reset_index(drop=True)
    rank_indices = np.array_split(np.arange(len(asset_targets)), world_size)[rank]
    asset_targets = asset_targets.iloc[rank_indices]

    dataset = WeekReferenceDataset(
        days=days,
        data_dir=str(cfg_data.data_dir),
        reference_symbol=reference_symbol,
        context_days=context_days,
        tick_minutes=tick_minutes,
        targets=asset_targets,
        limit=cfg_split.get("samples_num", None),
    )
    logger.info(
        "Week samples found, samples_num=%d, split=%s, reference=%s, "
        "context_ticks=%d, rollout_size=%d, validation_start=%s, last_date=%s",
        len(dataset),
        cfg_split.split,
        reference_symbol,
        dataset.context_ticks,
        dataset.rollout_size,
        validation_start.date(),
        last_date.date(),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg_split.batch_size),
        num_workers=int(cfg_split.workers_num),
        shuffle=bool(cfg_split.get("should_augment", False)),
        pin_memory=True,
        # Validation must cover its whole universe, so its final partial batch
        # is kept; training samples indefinitely and drops it.
        drop_last=cfg_split.split != "val",
        persistent_workers=int(cfg_split.workers_num) > 0,
    )
