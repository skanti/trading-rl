"""Choose replay sessions from the market calendar, bounded by exit data."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..market_data.calendar import EASTERN, session_closes
from ..market_data.schema import validate_bar_columns
from ..market_data.session_calendar import DEFAULT_CALENDAR_PATH, load_calendar
from .price_archives import open_price_archive

ORIGIN = pd.Timestamp("2010-01-01", tz="UTC")


def bar_inventory_bounds(directory: Path, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Read file endpoints, without requiring any particular symbol or full session."""
    first, last = [], []
    for path in directory.glob("*.npy"):
        bars = np.load(path, mmap_mode="r")
        validate_bar_columns(bars, timeframe, str(path))
        if len(bars):
            first.append(int(bars[0, 0]))
            last.append(int(bars[-1, 0]))
    if not first:
        raise ValueError(f"no {timeframe} bars in {directory}")
    return tuple((ORIGIN + pd.Timedelta(seconds=value)).tz_convert(EASTERN) for value in (min(first), max(last)))


def latest_exit_date(args, now: pd.Timestamp) -> date:
    """Use execution-data coverage; individual missing prices remain engine errors/warnings."""
    if args.exit_price_source in {"opening-auction", "nbbo-bid"}:
        path = Path(args.auctions_path if args.exit_price_source == "opening-auction" else args.exit_nbbo_path)
        with open_price_archive(path) as archive:
            days = np.asarray(archive["date"], dtype="datetime64[D]")
            if args.exit_price_source == "opening-auction":
                valid = (archive["session"] == 0) & (archive["condition"].astype(str) == "O")
                prices = archive["price"]
            else:
                valid = np.ones(len(days), dtype=bool)
                prices = archive["bid_price"]
            valid &= np.isfinite(prices) & (prices > 0) & (days <= np.datetime64(now.date()))
            if not valid.any():
                raise ValueError(f"no available {args.exit_price_source} exit prices in {path}")
            return pd.Timestamp(days[valid].max()).date()
    return bar_inventory_bounds(Path(args.minute_bars_dir), "1Min")[1].date()


def backtest_sessions(args, *, now: pd.Timestamp | None = None):
    """Retain every scheduled day, including a final morning-only exit session."""
    now = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(EASTERN)
    start = bar_inventory_bounds(Path(args.daily_bars_dir), "1Day")[0].date()
    requested_end = args.end_date.date() if args.end_date is not None else latest_exit_date(args, now)
    elapsed_exit = args.exit_time + int(args.exit_price_source.startswith("minute-") and args.exit_price_source != "minute-open")
    latest_elapsed = now.date() if now.hour * 60 + now.minute >= elapsed_exit else (now - pd.Timedelta(days=1)).date()
    if args.end_date is not None and requested_end > latest_elapsed:
        raise ValueError("--end-date requests an exit time that has not elapsed yet")
    end = min(requested_end, latest_elapsed)
    snapshot = load_calendar(
        start, end, path=args.calendar_path or DEFAULT_CALENDAR_PATH,
        refresh=args.refresh_calendar, offline=args.calendar_path is not None,
    )
    closes = session_closes(snapshot.sessions, start, end)
    dates = pd.DatetimeIndex(sorted(closes))
    if len(dates) < 2:
        raise ValueError("calendar range must include an entry and a later exit session")
    context = (dates.tz_localize(EASTERN) + pd.Timedelta(hours=4)).tz_convert("UTC")
    seconds = np.asarray((context - ORIGIN).total_seconds(), dtype=np.int64)
    return dates, seconds, closes, snapshot.metadata
