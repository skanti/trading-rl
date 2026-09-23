"""Validated, durable snapshots of Alpaca's US-equity session calendar."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from .calendar import session_closes

DEFAULT_CALENDAR_PATH = Path("/data/ppv1/live/market_calendar.json")


def fetch_calendar(start: date, end: date) -> list[dict]:
    key = os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("calendar cache needs coverage: set ALPACA_KEY and ALPACA_SECRET or provide --calendar-path")
    base = os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets/v2").rstrip("/")
    if not base.rsplit("/", 1)[-1].startswith("v"):
        base += "/v2"
    response = requests.get(
        f"{base}/calendar", params={"start": start.isoformat(), "end": end.isoformat()},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, timeout=(10, 30),
    )
    response.raise_for_status()
    sessions = response.json()
    if not isinstance(sessions, list):
        raise ValueError("calendar API must return a list of sessions")  # noqa: TRY004
    session_closes(sessions, start, end)
    return sessions


@dataclass(frozen=True)
class CalendarSnapshot:
    sessions: list[dict]
    metadata: dict


def load_calendar(
    start: date, end: date, *, path: Path = DEFAULT_CALENDAR_PATH,
    refresh: bool = False, offline: bool = False,
) -> CalendarSnapshot:
    """Never substitute observed bars for missing calendar coverage.

    Explicit offline snapshots fail if coverage is insufficient. The default
    cache is extended by whole years, retaining its existing coverage. A refresh
    replaces the complete range so removed sessions cannot linger after closures.
    """
    if end < start:
        raise ValueError("calendar end precedes start")
    if refresh and offline:
        raise ValueError("cannot refresh an offline calendar snapshot")
    payload = None
    covered_start, covered_end = start, end
    if path.exists():
        payload = json.loads(path.read_text())
        try:
            covered_start = date.fromisoformat(payload["start"])
            covered_end = date.fromisoformat(payload["end"])
            sessions = payload["sessions"]
            if not isinstance(sessions, list) or covered_start > covered_end:
                raise ValueError("invalid coverage or sessions")
            session_closes(sessions, covered_start, covered_end)
            if any(not covered_start <= date.fromisoformat(row["date"]) <= covered_end for row in sessions):
                raise ValueError("sessions outside declared coverage")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid calendar snapshot {path}: {error}") from error
    if payload is None or refresh or covered_start > start or covered_end < end:
        if offline:
            raise ValueError(f"calendar coverage must include {start} through {end}: {path}")
        covered_start = date(min(start, covered_start).year, 1, 1)
        covered_end = date(max(end, covered_end).year, 12, 31)
        sessions = fetch_calendar(covered_start, covered_end)
        payload = {
            "version": 1, "source": "Alpaca US market calendar",
            "start": covered_start.isoformat(), "end": covered_end.isoformat(),
            "fetched_at": datetime.now(UTC).isoformat(), "sessions": sessions,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    records = sorted(
        (row for row in payload["sessions"] if start <= date.fromisoformat(row["date"]) <= end),
        key=lambda row: row["date"],
    )
    session_closes(records, start, end)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CalendarSnapshot(records, {
        "path": str(path.resolve()), "source": payload.get("source", "provided calendar snapshot"),
        "fetched_at": payload.get("fetched_at"), "start": start.isoformat(), "end": end.isoformat(),
        "sessions_sha256": digest, "sessions": records,
    })
