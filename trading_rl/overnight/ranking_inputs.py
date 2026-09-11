"""Daily-bar inventory and official calendar inputs for standalone ranking."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..market_data.calendar import _clock_minute
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
    # Reuse the application's authenticated client, including bounded retries.
    from .live import AlpacaClient, load_credentials

    key, secret = load_credentials()
    client = AlpacaClient(
        key,
        secret,
        trading_url=os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets/v2"),
    )
    return client.calendar(start, end)


def session_closes(
    sessions: Sequence[Mapping[str, object]], start: date, end: date
) -> dict[date, int]:
    """Validate official records without guessing holidays or shortened sessions."""
    closes: dict[date, int] = {}
    for session in sessions:
        if not isinstance(session, Mapping):
            raise ValueError(  # noqa: TRY004 -- invalid external data
                "calendar sessions must be objects with date, open, and close"
            )
        try:
            day = date.fromisoformat(str(session["date"]))
            opening = _clock_minute(session["open"])
            closing = _clock_minute(session["close"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid calendar session: {session!r}") from error
        if closing <= opening:
            raise ValueError(f"calendar close must follow open for {day}")
        if start <= day <= end:
            if day in closes:
                raise ValueError(f"duplicate calendar session: {day}")
            closes[day] = closing
    if not closes:
        raise ValueError(f"no calendar sessions from {start} through {end}")
    return closes


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
        payload = json.loads(calendar_path.read_text())
        try:
            covered_start = date.fromisoformat(payload["start"])
            covered_end = date.fromisoformat(payload["end"])
            sessions = payload["sessions"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "calendar JSON requires start, end, and sessions"
            ) from error
        if covered_start > start or covered_end < end:
            raise ValueError(f"calendar coverage must include {start} through {end}")
    if not isinstance(sessions, list):
        raise ValueError("calendar sessions must be a list")  # noqa: TRY004 -- invalid external data
    closes = session_closes(sessions, start, end)
    missing = sorted(observed - closes.keys())
    if missing:
        labels = ", ".join(day.isoformat() for day in missing[:10])
        raise ValueError(f"calendar is missing daily-bar session dates: {labels}")
    return pd.DatetimeIndex(sorted(closes)), closes
