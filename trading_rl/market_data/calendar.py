"""Shared US-equity session-close helpers."""

from __future__ import annotations

from collections import Counter
from datetime import date, time
from pathlib import Path
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


EASTERN = ZoneInfo("America/New_York")


def _clock_minute(value: object) -> int:
    """Parse an Alpaca calendar clock or timestamp into a New York wall minute."""
    text = str(value).strip()
    if not text:
        raise ValueError("market-calendar close time is blank")
    if "T" not in text and " " not in text:
        try:
            parsed = time.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"invalid market-calendar close time: {text!r}") from error
        return parsed.hour * 60 + parsed.minute
    try:
        parsed_stamp = pd.Timestamp(text)
    except ValueError as error:
        raise ValueError(f"invalid market-calendar close timestamp: {text!r}") from error
    if parsed_stamp.tzinfo is None:
        parsed_stamp = parsed_stamp.tz_localize(EASTERN)
    else:
        parsed_stamp = parsed_stamp.tz_convert(EASTERN)
    return int(parsed_stamp.hour * 60 + parsed_stamp.minute)


def calendar_session_supports_entry(
    session: Mapping[str, object], entry_time: time
) -> bool:
    """Return whether the official session remains open after an entry clock."""
    if "close" not in session:
        raise ValueError(f"Alpaca calendar session {session.get('date')} has no close time")
    entry_minute = entry_time.hour * 60 + entry_time.minute
    return _clock_minute(session["close"]) > entry_minute


def auction_close_minutes(
    path: Path,
    required_dates: Iterable[date] | None,
    reference_symbol: str = "SPY",
) -> dict[date, int]:
    """Read official close minutes from an auction NPZ.

    SPY supplies the normal fast path. If its official close is absent for a required
    date, the modal official close minute across all downloaded symbols fills that
    date. This retains exchange-calendar behavior without inferring the close from
    extended-hours minute bars.
    """
    required = None if required_dates is None else set(required_dates)
    if required is not None and not required:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"auction data required for session closes: {path}")
    with np.load(path, allow_pickle=False) as data:
        needed = {"symbol", "date", "session", "condition", "timestamp"}
        missing = needed.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing auction arrays: {sorted(missing)}")
        symbols = np.asarray(data["symbol"]).astype(str, copy=False)
        dates = np.asarray(data["date"], dtype="datetime64[D]")
        sessions = np.asarray(data["session"])
        conditions = np.asarray(data["condition"]).astype(str, copy=False)
        timestamps = np.asarray(data["timestamp"]).astype(str, copy=False)
        official = (sessions == 1) & (conditions == "6")
        if required is not None:
            required_values = np.asarray(sorted(required), dtype="datetime64[D]")
            official &= np.isin(dates, required_values)
        reference = official & (np.char.upper(symbols) == reference_symbol.upper())

        closes: dict[date, int] = {}
        for index in np.flatnonzero(reference):
            day = pd.Timestamp(dates[index]).date()
            closes[day] = _clock_minute(timestamps[index])

        if required is None:
            return closes

        unresolved = required.difference(closes)
        if unresolved:
            unresolved_values = np.asarray(sorted(unresolved), dtype="datetime64[D]")
            candidates: dict[date, list[int]] = {day: [] for day in unresolved}
            for index in np.flatnonzero(official & np.isin(dates, unresolved_values)):
                day = pd.Timestamp(dates[index]).date()
                candidates[day].append(_clock_minute(timestamps[index]))
            for day, minutes in candidates.items():
                if minutes:
                    # Closing prints can arrive seconds or minutes late. The modal
                    # wall minute across symbols recovers the scheduled close.
                    closes[day] = Counter(minutes).most_common(1)[0][0]

    missing_dates = sorted(required.difference(closes))
    if missing_dates:
        labels = ", ".join(day.isoformat() for day in missing_dates[:10])
        suffix = f" (+{len(missing_dates) - 10} more)" if len(missing_dates) > 10 else ""
        raise ValueError(f"official session close is unavailable for {labels}{suffix}")
    return closes


def short_entry_dates(
    close_minutes: Mapping[date, int], entry_minute: int
) -> set[date]:
    """Return sessions whose official close is not after the entry clock."""
    return {
        day for day, close_minute in close_minutes.items() if close_minute <= entry_minute
    }
