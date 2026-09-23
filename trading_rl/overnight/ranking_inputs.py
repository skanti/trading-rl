"""Daily-bar inventory and official calendar inputs for standalone ranking."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..market_data.calendar import session_closes
from ..market_data.session_calendar import load_calendar
from .history import EASTERN, _security_symbol, daily_bar_dates


def daily_ranking_symbols(
    daily_dir: Path, symbols_path: Path | None = None
) -> np.ndarray:
    """Use the daily inventory, optionally restricted to an explicit shortlist."""
    available = {path.stem for path in daily_dir.glob("*.npy")}
    if symbols_path is None:
        selected = available
    else:
        selected = {
            _security_symbol(line.strip())
            for line in symbols_path.read_text().splitlines()
            if line.strip()
        }
        missing = selected - available
        if missing:
            raise ValueError(
                f"shortlist symbols have no daily bars: {', '.join(sorted(missing))}"
            )
    if not selected:
        raise ValueError(f"no daily-bar candidates in {daily_dir}")
    return np.asarray(sorted(selected), dtype=str)


def fetch_ranking_calendar(start: date, end: date) -> list[dict[str, object]]:
    return load_calendar(start, end).sessions


def daily_ranking_calendar(
    daily_dir: Path,
    symbols: Sequence[str],
    *,
    end_date: pd.Timestamp | None = None,
    calendar_path: Path | None = None,
    now: datetime | None = None,
) -> tuple[pd.DatetimeIndex, dict[date, int]]:
    """Bound replay by completed daily data, then retain every official session.

    Today's bar is always excluded, consistent with the daily downloader. A saved
    calendar declares its requested coverage, including non-trading days, so stale
    files cannot silently truncate replay. Missing observed dates fail explicitly.
    """
    today = (now or datetime.now(EASTERN)).astimezone(EASTERN).date()
    observed: set[date] = set()
    for symbol in symbols:
        dates = daily_bar_dates(daily_dir / f"{_security_symbol(symbol)}.npy")
        observed.update(
            stamp.date()
            for stamp in dates
            if stamp.date() < today and (end_date is None or stamp <= end_date)
        )
    if len(observed) < 2:
        raise ValueError("daily bars contain fewer than two completed session dates")
    start, end = min(observed), max(observed)
    if calendar_path is None:
        sessions = fetch_ranking_calendar(start, end)
    else:
        sessions = load_calendar(start, end, path=calendar_path, offline=True).sessions
    if not isinstance(sessions, list):
        raise ValueError("calendar sessions must be a list")  # noqa: TRY004 -- invalid external data
    closes = session_closes(sessions, start, end)
    missing = sorted(observed - closes.keys())
    if missing:
        labels = ", ".join(day.isoformat() for day in missing[:10])
        raise ValueError(f"calendar is missing daily-bar session dates: {labels}")
    return pd.DatetimeIndex(sorted(closes)), closes
