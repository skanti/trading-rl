"""Build day metadata for per-symbol minute-bar arrays."""

from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing as mp
from datetime import UTC, datetime, time, timedelta
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("SPLIT")

RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60
FULL_SESSION_INTERVALS = RTH_CLOSE_MINUTE - RTH_OPEN_MINUTE
EXTENDED_OPEN_MINUTE = 4 * 60
EXTENDED_CLOSE_MINUTE = 20 * 60
EXTENDED_SESSION_BARS = EXTENDED_CLOSE_MINUTE - EXTENDED_OPEN_MINUTE
EASTERN = ZoneInfo("US/Eastern")
OUTPUT_COLUMNS = (
    "sample_id",
    "date",
    "ctx_idx",
    "sod_idx",
    "eod_idx",
    "sod_sec",
    "eod_sec",
    "context_sod_sec",
    "context_eod_sec",
    "missing_ticks",
    "missing_context_ticks",
    "is_tradable",
)


def is_scheduled_half_day(day) -> bool:
    """Return whether the NYSE has its recurring 13:00 ET close on ``day``."""
    # The session after the fourth Thursday in November.
    day_after_thanksgiving = day.month == 11 and day.weekday() == 4 and 23 <= day.day <= 29
    # The exchange calendar schedules recurring pre-holiday early closes on
    # July 3 and December 24 when those dates are trading weekdays.
    pre_independence_day = day.month == 7 and day.day == 3 and day.weekday() < 5
    christmas_eve = day.month == 12 and day.day == 24 and day.weekday() < 5
    return day_after_thanksgiving or pre_independence_day or christmas_eve


def split_npy(
    npy_path: str,
    anno: str = "2010-01-01",
    rollout_size: int = FULL_SESSION_INTERVALS,
    min_session_ticks: int = 120,
) -> list[tuple]:
    """Return regular-session rows for one sorted, possibly sparse bar array.

    The output contains the expected 04:00--19:59 context grid and the expected
    09:30--16:00 rollout grid. The training loader completes missing bars in
    memory by carrying prices forward and assigning zero volume; this script
    never rewrites the source arrays.
    """
    path = Path(npy_path)
    if not path.exists():
        return []
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} must contain [seconds, price, volume]")
    # Keep timestamps contiguous: thousands of binary searches against a
    # strided mmap column are surprisingly expensive.
    secs = np.ascontiguousarray(data[:, 0])
    if secs.size < rollout_size + 1:
        return []
    if (secs < 0).any() or not np.all(secs[:-1] < secs[1:]):
        raise ValueError(f"timestamps must be non-negative and strictly sorted: {path}")

    origin = datetime.fromisoformat(anno)
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=UTC)
    else:
        origin = origin.astimezone(UTC)
    if origin.time() != time(0):
        raise ValueError("anno must identify midnight UTC")

    # Search the sorted timestamp column by exchange session instead of
    # converting millions of individual ticks through pandas. US regular
    # hours always fall on the same UTC calendar date; ZoneInfo supplies the
    # correct UTC offset for each date, including DST transitions.
    first_day = int(secs[0] // 86_400)
    last_day = int(secs[-1] // 86_400)
    schedule = []
    for day_offset in range(first_day, last_day + 1):
        session_day = (origin + timedelta(days=day_offset)).date()
        if is_scheduled_half_day(session_day):
            continue
        expected_open = datetime.combine(session_day, time(9, 30), tzinfo=EASTERN)
        sod_sec = int((expected_open.astimezone(UTC) - origin).total_seconds())
        context_open = datetime.combine(session_day, time(4), tzinfo=EASTERN)
        context_sod_sec = int((context_open.astimezone(UTC) - origin).total_seconds())
        context_eod_sec = context_sod_sec + (EXTENDED_SESSION_BARS - 1) * 60
        schedule.append(
            (
                session_day,
                sod_sec,
                sod_sec + FULL_SESSION_INTERVALS * 60,
                context_sod_sec,
                context_eod_sec,
            )
        )
    sod_secs = np.fromiter((item[1] for item in schedule), dtype=np.int64)
    eod_secs = np.fromiter((item[2] for item in schedule), dtype=np.int64)
    context_sod_secs = np.fromiter((item[3] for item in schedule), dtype=np.int64)
    context_eod_secs = np.fromiter((item[4] for item in schedule), dtype=np.int64)
    sod_indices = np.searchsorted(secs, sod_secs, side="left")
    stop_indices = np.searchsorted(secs, eod_secs, side="right")
    context_start_indices = np.searchsorted(secs, context_sod_secs, side="left")
    context_stop_indices = np.searchsorted(secs, context_eod_secs, side="right")

    rows: list[tuple] = []
    for (
        (session_day, sod_sec, eod_sec, context_sod_sec, context_eod_sec),
        sod_idx,
        stop_idx,
        context_start,
        context_stop,
    ) in zip(
        schedule,
        sod_indices,
        stop_indices,
        context_start_indices,
        context_stop_indices,
    ):
        sod_idx, stop_idx = int(sod_idx), int(stop_idx)
        # A completed context may carry a prior price forward, but it must not
        # backfill from a future observation. This mainly drops the first
        # partial day in a newly downloaded or newly listed symbol.
        if int(secs[0]) > context_sod_sec:
            continue
        if stop_idx <= sod_idx:
            continue
        session_secs = secs[sod_idx:stop_idx]
        aligned_secs = session_secs[session_secs % 60 == 0]
        if aligned_secs.size < min_session_ticks:
            continue

        if rollout_size == FULL_SESSION_INTERVALS:
            # Evidence near both boundaries distinguishes a sparse full day
            # from an exchange half-day, which must not be padded to 16:00.
            if aligned_secs[0] > sod_sec + 30 * 60 or aligned_secs[-1] < eod_sec - 30 * 60:
                continue
        eod_idx = stop_idx - 1
        expected_ticks = (eod_sec - sod_sec) // 60 + 1
        missing_ticks = int(expected_ticks - aligned_secs.size)
        if missing_ticks < 0:
            raise ValueError(f"too many regular-session minute bars for {path} on {session_day}")
        context_start, context_stop = int(context_start), int(context_stop)
        context_secs = secs[context_start:context_stop]
        aligned_context_secs = context_secs[context_secs % 60 == 0]
        missing_context_ticks = int(EXTENDED_SESSION_BARS - aligned_context_secs.size)
        if missing_context_ticks < 0:
            raise ValueError(f"too many extended-session minute bars for {path} on {session_day}")
        rows.append(
            (
                path.stem,
                session_day.isoformat(),
                0,
                sod_idx,
                eod_idx,
                sod_sec,
                eod_sec,
                context_sod_sec,
                context_eod_sec,
                missing_ticks,
                missing_context_ticks,
                True,
            )
        )
    return rows


def add_market_calendar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add open-market dates on which an individual symbol has no usable bars.

    The union across the supplied universe distinguishes exchange holidays from
    a completely missing symbol-day. Calendar-only rows may supply completed
    context, but ``is_tradable=False`` prevents them from becoming rollouts.
    """
    schedule_columns = (
        "date",
        "sod_sec",
        "eod_sec",
        "context_sod_sec",
        "context_eod_sec",
    )
    market_calendar = frame.loc[:, schedule_columns].drop_duplicates()
    if market_calendar.date.duplicated().any():
        raise ValueError("symbols disagree about one or more market-session boundaries")
    market_calendar = market_calendar.sort_values("date").reset_index(drop=True)

    payload_columns = (
        "date",
        "ctx_idx",
        "sod_idx",
        "eod_idx",
        "missing_ticks",
        "missing_context_ticks",
        "is_tradable",
    )
    pieces = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        first_date, last_date = group.date.min(), group.date.max()
        symbol_calendar = market_calendar[
            market_calendar.date.between(first_date, last_date)
        ].copy()
        symbol_calendar.insert(0, "sample_id", sample_id)
        symbol_calendar = symbol_calendar.merge(
            group.loc[:, payload_columns], on="date", how="left", validate="one_to_one"
        )
        symbol_calendar["is_tradable"] = symbol_calendar.is_tradable.fillna(False).astype(bool)
        for column in ("ctx_idx", "sod_idx", "eod_idx"):
            symbol_calendar[column] = symbol_calendar[column].fillna(-1).astype(np.int64)
        symbol_calendar["missing_ticks"] = (
            symbol_calendar.missing_ticks.fillna(FULL_SESSION_INTERVALS + 1).astype(np.int64)
        )
        symbol_calendar["missing_context_ticks"] = (
            symbol_calendar.missing_context_ticks.fillna(EXTENDED_SESSION_BARS).astype(np.int64)
        )
        pieces.append(symbol_calendar)
    return pd.concat(pieces, ignore_index=True).loc[:, OUTPUT_COLUMNS]


def read_sample_ids(path_or_id: str) -> list[str]:
    path = Path(path_or_id)
    if path.suffix == ".txt":
        sample_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        sample_ids = [path_or_id]
    if not sample_ids:
        raise ValueError("sample ID list is empty")
    bad = [sample_id for sample_id in sample_ids if sample_id.endswith(".npy")]
    if bad:
        raise ValueError("sample IDs must not include the .npy suffix")
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_ids", required=True)
    parser.add_argument("--npy_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--workers_num", type=int, default=16)
    parser.add_argument("--anno", default="2010-01-01")
    parser.add_argument("--rollout_size", type=int, default=FULL_SESSION_INTERVALS)
    parser.add_argument("--min_session_ticks", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.workers_num < 1:
        parser.error("--workers_num must be positive")
    if args.rollout_size < 1 or args.min_session_ticks < 1:
        parser.error("rollout and minimum session sizes must be positive")

    sample_ids = read_sample_ids(args.sample_ids)
    paths = [str(Path(args.npy_dir) / f"{sample_id}.npy") for sample_id in sample_ids]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} input files are missing; first: {preview}")
    logger.info("Samples found, samples_num=%d", len(paths))

    fn = partial(
        split_npy,
        anno=args.anno,
        rollout_size=args.rollout_size,
        min_session_ticks=args.min_session_ticks,
    )
    if args.workers_num == 1:
        nested_rows = list(tqdm(map(fn, paths), total=len(paths), desc="split symbols"))
    else:
        with mp.Pool(args.workers_num) as pool:
            nested_rows = list(
                tqdm(pool.imap(fn, paths), total=len(paths), desc="split symbols")
            )

    rows = list(itertools.chain.from_iterable(nested_rows))
    if not rows:
        raise ValueError("no complete sessions met the requested window and rollout sizes")
    rng = np.random.default_rng(args.seed)
    frame = add_market_calendar_rows(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
    rows = frame.iloc[rng.permutation(len(frame))]
    frame = rows.reset_index(drop=True)
    expected_intervals = (frame.eod_sec - frame.sod_sec) // 60
    if not (expected_intervals >= args.rollout_size).all():
        raise AssertionError("one or more expected sessions are shorter than the requested rollout")
    context_bars = (frame.context_eod_sec - frame.context_sod_sec) // 60 + 1
    if not (context_bars == EXTENDED_SESSION_BARS).all():
        raise AssertionError("one or more context sessions do not span 04:00--19:59")

    output = Path(args.out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info("Data split, seqs_num=%d, output=%s", len(frame), output)


if __name__ == "__main__":
    main()
