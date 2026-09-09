"""Download split-adjusted Alpaca SIP NBBO snapshots at a scheduled time."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time as clock_time, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..market_data.calendar import auction_close_minutes, short_entry_dates
from ..market_data.download_mode import use_incremental_download
from ..market_data.download_output import current_report, download_output, info, track_download
from .download_auctions import (
    _download_splits,
    _ordered_splits,
    _request_headers,
    _security_symbol,
    symbols_from_file,
)


LOGGER = logging.getLogger(__name__)

QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
FORMAT_VERSION = 2
FIELDNAMES = (
    "symbol",
    "date",
    "target_timestamp",
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "bid_exchange",
    "ask_exchange",
)
EASTERN = ZoneInfo("America/New_York")
RECENT_SIP_SAFETY_DELAY = timedelta(minutes=20)
REFERENCE_SYMBOL = "SPY"
DEFAULT_AUCTIONS_PATH = "/data/ppv1/updates/alpaca_auctions_2022-01-01.npz"


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._next = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self._interval


def default_end(now: datetime | None = None) -> str:
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(EASTERN).date().isoformat()


def parse_target_time(value: str) -> clock_time:
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("target time must use HH:MM") from error
    return parsed


def target_timestamp(day: date, target: clock_time) -> datetime:
    return datetime.combine(day, target, tzinfo=EASTERN).astimezone(timezone.utc)


def eligible_sessions(
    sessions: Iterable[date],
    target: clock_time,
    now: datetime | None = None,
) -> tuple[list[date], int]:
    """Exclude target snapshots that remain inside the delayed-SIP window."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference.astimezone(timezone.utc) - RECENT_SIP_SAFETY_DELAY
    values = list(sessions)
    eligible = [day for day in values if target_timestamp(day, target) <= cutoff]
    return eligible, len(values) - len(eligible)


def _request_json(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict[str, object],
    limiter: RateLimiter,
    retries: int = 8,
) -> object:
    for attempt in range(retries):
        limiter.wait()
        response = session.get(url, headers=headers, params=params, timeout=(10, 90))
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt + 1 == retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else min(30.0, 2.0**attempt))
            continue
        if not 200 <= response.status_code < 300:
            try:
                message = str(response.json().get("message") or response.text)
            except (TypeError, ValueError):
                message = response.text
            raise RuntimeError(
                f"Alpaca request failed with HTTP {response.status_code}: {message.strip()}"
            )
        return response.json()
    raise AssertionError("request retry loop exited unexpectedly")


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def targets_from_trade_csv(
    path: Path, date_column: str = "entry_date"
) -> dict[date, set[str]]:
    """Read the exact daily basket from a simulator trade CSV.

    The union of symbols is deliberately not crossed with every date: membership may
    rotate, and each symbol is requested only for the sessions on which it was selected.
    """
    try:
        trades = pd.read_csv(path, usecols=[date_column, "sample_id"])
    except ValueError as error:
        raise ValueError(
            f"{path} must contain {date_column} and sample_id columns"
        ) from error
    if trades.empty:
        raise ValueError(f"NBBO target CSV is empty: {path}")
    if trades[[date_column, "sample_id"]].isna().any().any():
        raise ValueError(f"NBBO target CSV contains blank dates or symbols: {path}")

    targets: dict[date, set[str]] = {}
    for raw_day, raw_symbol in trades[[date_column, "sample_id"]].itertuples(
        index=False, name=None
    ):
        try:
            day = pd.Timestamp(raw_day).date()
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid NBBO target date {raw_day!r} in {path}") from error
        symbol = _security_symbol(str(raw_symbol).strip())
        if not symbol:
            raise ValueError(f"blank NBBO target symbol on {day} in {path}")
        targets.setdefault(day, set()).add(symbol)
    return targets


def merge_target_schedules(
    schedules: Iterable[dict[date, set[str]]],
) -> dict[date, set[str]]:
    merged: dict[date, set[str]] = {}
    for schedule in schedules:
        for day, symbols in schedule.items():
            merged.setdefault(day, set()).update(symbols)
    if not merged:
        raise ValueError("no NBBO symbol-date targets were provided")
    return merged


def targets_from_symbols(
    symbols: set[str], start: str, end: str, target: clock_time
) -> dict[date, set[str]]:
    """Schedule a reusable symbol universe using the official trading calendar."""
    from ..overnight.live import AlpacaClient, load_credentials
    from ..market_data.calendar import calendar_session_supports_entry

    start_day, end_day = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if end_day < start_day:
        raise ValueError(f"end {end} precedes start {start}")
    key, secret = load_credentials()
    client = AlpacaClient(
        key, secret,
        trading_url=os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets/v2"),
    )
    sessions = client.calendar(start_day, end_day)
    universe = {_security_symbol(symbol) for symbol in symbols} | {REFERENCE_SYMBOL}
    targets = {
        date.fromisoformat(str(session["date"])): set(universe)
        for session in sessions
        if calendar_session_supports_entry(session, target)
    }
    if not targets:
        raise ValueError(f"no eligible trading sessions from {start} through {end}")
    return targets


def _bounded_targets(
    targets: dict[date, set[str]], start: str, end: str
) -> dict[date, set[str]]:
    start_day = pd.Timestamp(start).date()
    end_day = pd.Timestamp(end).date()
    if end_day < start_day:
        raise ValueError(f"end {end} precedes start {start}")
    bounded = {
        day: set(symbols)
        for day, symbols in targets.items()
        if start_day <= day <= end_day
    }
    if not bounded:
        raise ValueError(f"target schedule has no sessions from {start} through {end}")
    return bounded


def _valid_quote(record: object, target: datetime) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        stamp = pd.Timestamp(record["t"])
        bid = float(record["bp"])
        ask = float(record["ap"])
        bid_size = float(record["bs"])
        ask_size = float(record["as"])
    except (KeyError, TypeError, ValueError):
        return False
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    return bool(
        stamp <= pd.Timestamp(target)
        and np.isfinite([bid, ask, bid_size, ask_size]).all()
        and bid > 0.0
        and ask >= bid
        and bid_size >= 0.0
        and ask_size >= 0.0
    )


def select_causal_quotes(
    payload: object,
    day: date,
    target: datetime,
) -> dict[str, dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("quotes"), dict):
        raise ValueError("Alpaca response has no quotes mapping")
    selected: dict[str, dict[str, object]] = {}
    for raw_symbol, records in payload["quotes"].items():
        symbol = _security_symbol(str(raw_symbol))
        if not isinstance(records, list):
            continue
        valid = [record for record in records if _valid_quote(record, target)]
        if not valid:
            continue
        quote = max(valid, key=lambda record: pd.Timestamp(record["t"]))
        selected[symbol] = {
            "symbol": symbol,
            "date": day.isoformat(),
            "target_timestamp": target.isoformat().replace("+00:00", "Z"),
            "timestamp": str(quote["t"]),
            "bid_price": float(quote["bp"]),
            "ask_price": float(quote["ap"]),
            "bid_size": float(quote["bs"]),
            "ask_size": float(quote["as"]),
            "bid_exchange": str(quote.get("bx", "")),
            "ask_exchange": str(quote.get("ax", "")),
        }
    return selected


def _quote_request(
    session: requests.Session,
    headers: dict[str, str],
    limiter: RateLimiter,
    symbols: list[str],
    start: datetime,
    target: datetime,
    limit: int,
    page_token: str | None = None,
) -> object:
    params = {
        "symbols": ",".join(symbols),
        "start": start.isoformat().replace("+00:00", "Z"),
        # Alpaca treats end as inclusive. Filtering again is the causal guard.
        "end": target.isoformat().replace("+00:00", "Z"),
        "feed": "sip",
        "sort": "desc",
        "limit": limit,
    }
    if page_token is not None:
        params["page_token"] = page_token
    return _request_json(
        session,
        QUOTES_URL,
        headers,
        params,
        limiter,
    )


def _latest_valid_quote(
    session: requests.Session,
    headers: dict[str, str],
    limiter: RateLimiter,
    symbol: str,
    day: date,
    target: datetime,
    lookback_seconds: int,
) -> dict[str, object] | None:
    """Search past invalid latest quotes without widening the allowed window."""
    page_token = None
    seen_tokens: set[str] = set()
    while True:
        kwargs = {} if page_token is None else {"page_token": page_token}
        payload = _quote_request(
            session, headers, limiter, [symbol],
            target - timedelta(seconds=lookback_seconds), target, 1000, **kwargs,
        )
        candidate = select_causal_quotes(payload, day, target).get(symbol)
        if candidate is not None:
            return candidate
        page_token = payload.get("next_page_token")
        if not page_token:
            return None
        if page_token in seen_tokens:
            raise ValueError(f"repeated NBBO page token for {day.isoformat()}:{symbol}")
        seen_tokens.add(page_token)


def _download_quote_rows(
    session: requests.Session,
    headers: dict[str, str],
    targets: dict[date, set[str]],
    target: clock_time,
    batch_size: int,
    lookback_seconds: int,
    limiter: RateLimiter,
    missing_quotes: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sessions = sorted(targets)
    for day in track_download(sessions, total=len(sessions), description="Fetching NBBO", unit="sessions"):
        target_at = target_timestamp(day, target)
        for batch in _batches(sorted(targets[day]), batch_size):
            # A tight batch query is efficient for liquid names. Any absent name gets
            # a single-symbol fallback over the full allowed staleness interval.
            payload = _quote_request(
                session,
                headers,
                limiter,
                batch,
                target_at - timedelta(seconds=min(2, lookback_seconds)),
                target_at,
                10_000,
            )
            selected = select_causal_quotes(payload, day, target_at)
            for symbol in batch:
                candidate = selected.get(symbol)
                if candidate is not None and _quote_staleness_seconds(
                    candidate, target_at
                ) <= lookback_seconds:
                    rows.append(candidate)
                    continue
                candidate = _latest_valid_quote(
                    session, headers, limiter, symbol, day, target_at, lookback_seconds
                )
                if candidate is None or _quote_staleness_seconds(
                    candidate, target_at
                ) > lookback_seconds:
                    reason = f"no valid causal SIP NBBO within {lookback_seconds}s"
                    LOGGER.warning("Skipping NBBO %s:%s: %s", day.isoformat(), symbol, reason)
                    if missing_quotes is not None:
                        missing_quotes.append({
                            "date": day.isoformat(), "symbol": symbol,
                            "target_timestamp": target_at.isoformat(), "reason": reason,
                        })
                    continue
                rows.append(candidate)
    return rows


def _quote_staleness_seconds(row: dict[str, object], target: datetime) -> float:
    stamp = pd.Timestamp(str(row["timestamp"]))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    return float((pd.Timestamp(target) - stamp).total_seconds())


def _ordered_unique_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique = {(str(row["symbol"]), str(row["date"])): row for row in rows}
    return sorted(unique.values(), key=lambda row: (str(row["symbol"]), str(row["date"])))


def merge_quote_rows(
    existing_rows: Iterable[dict[str, object]],
    refreshed_rows: Iterable[dict[str, object]],
    refreshed_targets: set[tuple[str, str]],
) -> list[dict[str, object]]:
    """Replace only attempted symbol/date pairs, including explicitly missing quotes."""
    kept = [
        row
        for row in existing_rows
        if (str(row["symbol"]), str(row["date"])) not in refreshed_targets
    ]
    return _ordered_unique_rows([*kept, *refreshed_rows])


def _dataset_arrays(
    rows: list[dict[str, object]], split_rows: list[dict[str, object]]
) -> dict[str, np.ndarray]:
    symbols = np.asarray([str(row["symbol"]) for row in rows])
    dates = np.asarray([str(row["date"]) for row in rows], dtype="datetime64[D]")
    raw_bid = np.asarray([row["bid_price"] for row in rows], dtype=np.float64)
    raw_ask = np.asarray([row["ask_price"] for row in rows], dtype=np.float64)
    raw_bid_size = np.asarray([row["bid_size"] for row in rows], dtype=np.float64)
    raw_ask_size = np.asarray([row["ask_size"] for row in rows], dtype=np.float64)
    bid = raw_bid.copy()
    ask = raw_ask.copy()
    bid_size = raw_bid_size.copy()
    ask_size = raw_ask_size.copy()
    split_symbols: list[str] = []
    split_dates: list[str] = []
    split_old_rates: list[float] = []
    split_new_rates: list[float] = []
    for split in split_rows:
        symbol = str(split["symbol"])
        ex_date = str(split["ex_date"])
        old_rate = float(split["old_rate"])
        new_rate = float(split["new_rate"])
        if not np.isfinite(old_rate) or not np.isfinite(new_rate) or min(old_rate, new_rate) <= 0:
            raise ValueError(f"invalid split rate for {symbol} on {ex_date}")
        ratio = new_rate / old_rate
        before = (symbols == symbol) & (dates < np.datetime64(ex_date, "D"))
        bid[before] /= ratio
        ask[before] /= ratio
        bid_size[before] *= ratio
        ask_size[before] *= ratio
        split_symbols.append(symbol)
        split_dates.append(ex_date)
        split_old_rates.append(old_rate)
        split_new_rates.append(new_rate)
    return {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int16),
        "split_adjusted": np.asarray(True),
        "symbol": symbols,
        "date": dates,
        "target_timestamp": np.asarray([str(row["target_timestamp"]) for row in rows]),
        "timestamp": np.asarray([str(row["timestamp"]) for row in rows]),
        "bid_price": bid,
        "ask_price": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "raw_bid_price": raw_bid,
        "raw_ask_price": raw_ask,
        "raw_bid_size": raw_bid_size,
        "raw_ask_size": raw_ask_size,
        "bid_exchange": np.asarray([str(row["bid_exchange"]) for row in rows]),
        "ask_exchange": np.asarray([str(row["ask_exchange"]) for row in rows]),
        "split_type": np.asarray([str(row["type"]) for row in split_rows]),
        "split_symbol": np.asarray(split_symbols),
        "split_ex_date": np.asarray(split_dates, dtype="datetime64[D]"),
        "split_old_rate": np.asarray(split_old_rates, dtype=np.float64),
        "split_new_rate": np.asarray(split_new_rates, dtype=np.float64),
        "split_id": np.asarray([str(row["id"]) for row in split_rows]),
    }


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o664
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
        np.savez_compressed(temporary, **arrays)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(mode)
    temporary_path.replace(path)


def _load_raw_rows(path: Path) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        found_version = int(np.asarray(data["format_version"]).item())
        if found_version != FORMAT_VERSION:
            raise ValueError(
                f"NBBO dataset format {found_version} cannot be updated as strict "
                f"symbol-date format {FORMAT_VERSION}; rebuild it with "
                "--targets-from-trades"
            )
        count = len(data["symbol"])
        return [
            {
                "symbol": str(data["symbol"][index]),
                "date": np.datetime_as_string(data["date"][index], unit="D"),
                "target_timestamp": str(data["target_timestamp"][index]),
                "timestamp": str(data["timestamp"][index]),
                "bid_price": float(data["raw_bid_price"][index]),
                "ask_price": float(data["raw_ask_price"][index]),
                "bid_size": float(data["raw_bid_size"][index]),
                "ask_size": float(data["raw_ask_size"][index]),
                "bid_exchange": str(data["bid_exchange"][index]),
                "ask_exchange": str(data["ask_exchange"][index]),
            }
            for index in range(count)
        ]


def _write_dataset(
    output: Path,
    rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
    symbols: list[str],
    start: str,
    end: str,
    target: clock_time,
    lookback_seconds: int,
    target_count: int,
    target_session_count: int,
    update_metadata: dict[str, object] | None = None,
    missing_quotes: list[dict[str, object]] | None = None,
) -> None:
    ordered = _ordered_unique_rows(rows)
    ordered_splits = _ordered_splits(split_rows)
    _atomic_write_npz(output, _dataset_arrays(ordered, ordered_splits))
    manifest: dict[str, object] = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": QUOTES_URL,
        "feed": "sip",
        "start": start,
        "end": end,
        "target_time": target.strftime("%H:%M"),
        "selection": "latest valid SIP NBBO at or before target",
        "max_staleness_seconds": lookback_seconds,
        "scheduled_target_count": target_count,
        "scheduled_session_count": target_session_count,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "quote_count": len(ordered),
        "missing_quote_count": len(missing_quotes or []),
        "missing_quotes": missing_quotes or [],
        "format": "numpy-npz-columnar-v2",
        "prices": "split-adjusted",
        "raw_prices_embedded": True,
        "split_ledger_embedded": True,
        "split_adjustment_count": len(ordered_splits),
    }
    if update_metadata:
        manifest["last_update"] = update_metadata
    manifest_path = output.with_suffix(".json")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.chmod(
        manifest_path.stat().st_mode & 0o777 if manifest_path.exists() else 0o664
    )
    temporary_path.replace(manifest_path)


def _fetch(
    targets: dict[date, set[str]],
    start: str,
    end: str,
    target: clock_time,
    batch_size: int,
    lookback_seconds: int,
    requests_per_minute: int,
) -> tuple[list[dict[str, object]], int, list[dict[str, object]], set[str]]:
    limiter = RateLimiter(requests_per_minute)
    headers = _request_headers()
    requested = _bounded_targets(targets, start, end)
    eligible, deferred = eligible_sessions(sorted(requested), target)
    eligible_targets = {day: requested[day] for day in eligible}
    if not eligible_targets:
        return [], deferred, [], set()
    info(
        f"requesting {sum(map(len, eligible_targets.values())):,} scheduled NBBO quotes "
        f"across {len(eligible_targets):,} sessions"
    )
    missing_quotes: list[dict[str, object]] = []
    with requests.Session() as session:
        rows = _download_quote_rows(
            session,
            headers,
            eligible_targets,
            target,
            batch_size,
            lookback_seconds,
            limiter,
            missing_quotes,
        )
    expected = sum(len(eligible_targets[day]) for day in eligible_targets)
    if len(rows) + len(missing_quotes) != expected:
        raise AssertionError(f"unaccounted scheduled NBBO targets: expected {expected:,}")
    return rows, deferred, missing_quotes, {day.isoformat() for day in eligible_targets}


def download_nbbo(
    targets: dict[date, set[str]],
    start: str,
    end: str,
    output: Path,
    target: clock_time,
    batch_size: int = 100,
    lookback_seconds: int = 60,
    requests_per_minute: int = 180,
) -> tuple[int, int, int]:
    targets = _bounded_targets(targets, start, end)
    report = current_report()
    if report:
        report.set("Targets requested", sum(map(len, targets.values())), "symbol/date pairs")
    rows, deferred, missing_quotes, _ = _fetch(
        targets, start, end, target, batch_size, lookback_seconds, requests_per_minute
    )
    symbols = sorted({symbol for values in targets.values() for symbol in values})
    headers = _request_headers()
    split_end = max(pd.Timestamp(end).date(), datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        splits = _download_splits(session, headers, symbols, start, split_end)
    _write_dataset(
        output,
        rows,
        splits,
        symbols,
        start,
        end,
        target,
        lookback_seconds,
        sum(len(values) for values in targets.values()),
        len(targets),
        missing_quotes=missing_quotes,
    )
    if report:
        report.set("Full redownloads" if report.mode == "Rebuild" else "New quotes", len(rows), "quotes")
        report.set("Missing this run", len(missing_quotes), "symbol/date pairs")
        report.set("Targets deferred", sum(map(len, targets.values())) - len(rows) - len(missing_quotes), "symbol/date pairs")
    return len(symbols), len(_ordered_unique_rows(rows)), deferred


def update_nbbo(
    output: Path,
    end: str,
    targets: dict[date, set[str]],
    overlap_days: int,
    batch_size: int,
    lookback_seconds: int,
    requests_per_minute: int,
    target_override: clock_time | None = None,
) -> tuple[int, int, int]:
    manifest_path = output.with_suffix(".json")
    if not output.exists() or not manifest_path.exists():
        raise ValueError("--update requires the existing NBBO NPZ and JSON manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = min(str(manifest["start"])[:10], min(targets).isoformat())
    previous_end = str(manifest["end"])
    target = parse_target_time(str(manifest["target_time"]))
    if target_override is not None and target_override != target:
        raise ValueError(
            f"--target-time {target_override:%H:%M} does not match existing "
            f"dataset target {target:%H:%M}"
        )
    if int(manifest["max_staleness_seconds"]) != lookback_seconds:
        raise ValueError("--lookback-seconds must match the existing NBBO dataset")
    if pd.Timestamp(end).date() < pd.Timestamp(previous_end).date():
        raise ValueError(f"update end {end} precedes existing end {previous_end}")
    refresh_start = max(
        pd.Timestamp(start).date(),
        pd.Timestamp(previous_end).date() - timedelta(days=overlap_days - 1),
    ).isoformat()
    existing_rows = _load_raw_rows(output)
    existing_symbols = {str(row["symbol"]) for row in existing_rows} | set(manifest["symbols"])
    scheduled_symbols = {symbol for values in targets.values() for symbol in values}
    all_symbols = sorted(existing_symbols | scheduled_symbols)
    # Previously missing quotes count as attempted. Retry those in the overlap,
    # while backfilling every never-attempted pair regardless of its age.
    known = {
        (str(row["symbol"]), str(row["date"]))
        for row in [*existing_rows, *manifest.get("missing_quotes", [])]
    }
    pending = {
        day: {
            symbol for symbol in symbols
            if day.isoformat() >= refresh_start or (symbol, day.isoformat()) not in known
        }
        for day, symbols in targets.items()
        if start <= day.isoformat() <= end
    }
    pending = {day: symbols for day, symbols in pending.items() if symbols}
    report = current_report()
    if report:
        report.set("Targets requested", sum(map(len, pending.values())), "symbol/date pairs")
    if pending:
        refreshed, deferred, missing_quotes, refreshed_days = _fetch(
            pending, min(pending).isoformat(), end, target, batch_size,
            lookback_seconds, requests_per_minute,
        )
    else:
        refreshed, deferred, missing_quotes, refreshed_days = [], 0, [], set()
    refreshed_targets = {
        (symbol, day.isoformat())
        for day, symbols in pending.items() if day.isoformat() in refreshed_days
        for symbol in symbols
    }
    missing_this_run = len(missing_quotes)
    # Refresh the complete split ledger once for the final universe.
    headers = _request_headers()
    split_end = max(pd.Timestamp(end).date(), datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        splits = _download_splits(session, headers, all_symbols, start, split_end)
    # Missing attempted pairs replace old quotes, but unrequested symbols and
    # deferred sessions retain their existing records.
    missing_quotes = [
        row for row in manifest.get("missing_quotes", [])
        if (str(row["symbol"]), str(row["date"])) not in refreshed_targets
    ] + missing_quotes
    merged = merge_quote_rows(existing_rows, refreshed, refreshed_targets)
    _write_dataset(
        output,
        merged,
        splits,
        all_symbols,
        start,
        end,
        target,
        lookback_seconds,
        sum(len(values) for values in targets.values()),
        len(targets),
        {
            "previous_end": previous_end,
            "refresh_start": refresh_start,
            "overlap_days": overlap_days,
            "scheduled_symbols": sorted(scheduled_symbols),
            "scheduled_target_count": sum(len(values) for values in targets.values()),
            "attempted_target_count": len(refreshed_targets),
        },
        missing_quotes=missing_quotes,
    )
    if report:
        old_keys = {(str(row["symbol"]), str(row["date"])) for row in existing_rows}
        updated = sum((str(row["symbol"]), str(row["date"])) in old_keys for row in refreshed)
        backfilled = sum(
            (str(row["symbol"]), str(row["date"])) not in old_keys and str(row["date"]) < refresh_start
            for row in refreshed
        )
        report.set("Updated quotes", updated, "quotes")
        report.set("New quotes", len(refreshed) - updated - backfilled, "quotes")
        report.set("Historical backfills", backfilled, "quotes")
        report.set("Missing this run", missing_this_run, "symbol/date pairs")
        report.set("Targets deferred", sum(map(len, pending.values())) - len(refreshed_targets), "symbol/date pairs")
        report.set("Existing quotes retained", len(old_keys - refreshed_targets), "quotes")
    return len(all_symbols), len(merged), deferred


@download_output("NBBO", LOGGER, exit_on_error=True)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--target-time", type=parse_target_time, default=None)
    parser.add_argument(
        "--auctions-path",
        default=DEFAULT_AUCTIONS_PATH,
        help="auction NPZ used for official session-close times",
    )
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true",
                      help="require an existing dataset (default: create or update automatically)")
    mode.add_argument("--rebuild", action="store_true", help="redownload the full target schedule")
    parser.add_argument("--symbols-file", action="append", default=[],
                        help="one symbol per line; query each trading session without a trade CSV")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--lookback-seconds", type=int, default=60)
    parser.add_argument(
        "--target-date-column",
        choices=("entry_date", "exit_date"),
        default="entry_date",
        help="trade CSV date to query: entry_date for entry quotes, exit_date for exit quotes",
    )
    parser.add_argument(
        "--targets-from-trades",
        "--symbols-from-trades",
        dest="targets_from_trades",
        action="append",
        default=[],
        metavar="CSV",
        help=(
            "simulator trade CSV containing the target date column and sample_id; may be repeated. "
            "The older --symbols-from-trades spelling remains an alias"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--requests-per-minute", type=int, default=180)
    args = parser.parse_args()
    report = current_report()
    report.output = args.output
    if min(args.overlap_days, args.lookback_seconds, args.batch_size, args.requests_per_minute) < 1:
        parser.error("overlap, lookback, batch size, and request rate must be positive")
    if bool(args.targets_from_trades) == bool(args.symbols_file):
        parser.error("provide either --symbols-file or --targets-from-trades")
    output = Path(args.output)
    try:
        incremental = use_incremental_download(output, update=args.update, rebuild=args.rebuild)
    except ValueError as error:
        parser.error(str(error))
    report.mode = "Rebuild" if args.rebuild else "Update" if incremental else "Initial download"
    for label in ("Updated quotes", "New quotes", "Historical backfills", "Full redownloads"):
        report.set(label, 0, "quotes")
    target = args.target_time
    manifest_path = output.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if target is None and manifest:
        target = parse_target_time(str(manifest["target_time"]))
    target = target or parse_target_time("15:45")
    shortened = set()
    if args.symbols_file:
        start = args.start or manifest.get("start")
        if not start:
            parser.error("--start is required for an initial symbol-file download")
        if incremental:
            start = min(start, str(manifest["start"]))
        symbols = set().union(*(symbols_from_file(Path(path)) for path in args.symbols_file))
        if not symbols:
            parser.error("symbol files contain no symbols")
        end = args.end or default_end()
        targets = targets_from_symbols(symbols, start, end, target)
    else:
        targets = merge_target_schedules(
            targets_from_trade_csv(Path(value), args.target_date_column)
            for value in args.targets_from_trades
        )
        close_minutes = auction_close_minutes(Path(args.auctions_path), targets)
        shortened = short_entry_dates(close_minutes, target.hour * 60 + target.minute)
        targets = {day: symbols for day, symbols in targets.items() if day not in shortened}
        if not targets:
            parser.error("all scheduled NBBO targets fall on shortened sessions")
        start = args.start or min(targets).isoformat()
        end = args.end or max(targets).isoformat()
    # The simulator prices SPY at the same entry source for its benchmark.  Keep
    # that reference observation alongside each day's rotating strategy basket.
    for day_symbols in targets.values():
        day_symbols.add(REFERENCE_SYMBOL)
    if incremental:
        count, quotes, deferred = update_nbbo(
            output,
            end,
            targets,
            args.overlap_days,
            args.batch_size,
            args.lookback_seconds,
            args.requests_per_minute,
            target,
        )
    else:
        count, quotes, deferred = download_nbbo(
            targets,
            start,
            end,
            output,
            target,
            args.batch_size,
            args.lookback_seconds,
            args.requests_per_minute,
        )
    report.set("Quotes in dataset", quotes, "quotes")
    report.set("Symbols in dataset", count, "symbols")
    report.set("Short sessions skipped", len(shortened), "sessions")
    report.set("Sessions deferred", deferred, "sessions")
    written_manifest = json.loads(output.with_suffix(".json").read_text())
    if written_manifest["missing_quote_count"]:
        report.details.append(f"Missing quote details: {output.with_suffix('.json')} (missing_quotes)")



if __name__ == "__main__":
    main()
