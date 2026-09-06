"""Download split-adjusted Alpaca SIP NBBO snapshots at a scheduled time."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time as clock_time, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .download_auctions import (
    _download_splits,
    _ordered_splits,
    _request_headers,
    _security_symbol,
    symbols_from_file,
    symbols_from_trade_csv,
)


QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
DEFAULT_TRADING_URL = "https://paper-api.alpaca.markets/v2"
FORMAT_VERSION = 1
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


def _trading_headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_DATA_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_DATA_SECRET")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials must be set")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "accept": "application/json",
    }


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


def _calendar_days(
    session: requests.Session,
    start: str,
    end: str,
    limiter: RateLimiter,
) -> list[date]:
    payload = _request_json(
        session,
        os.environ.get("ALPACA_URL", DEFAULT_TRADING_URL).rstrip("/") + "/calendar",
        _trading_headers(),
        {"start": start, "end": end},
        limiter,
    )
    if not isinstance(payload, list):
        raise ValueError("Alpaca returned a non-list calendar response")
    return sorted({pd.Timestamp(row["date"]).date() for row in payload})


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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
) -> object:
    return _request_json(
        session,
        QUOTES_URL,
        headers,
        {
            "symbols": ",".join(symbols),
            "start": start.isoformat().replace("+00:00", "Z"),
            # Alpaca treats end as inclusive. Filtering again is the causal guard.
            "end": target.isoformat().replace("+00:00", "Z"),
            "feed": "sip",
            "sort": "desc",
            "limit": limit,
        },
        limiter,
    )


def _download_quote_rows(
    session: requests.Session,
    headers: dict[str, str],
    symbols: list[str],
    sessions: list[date],
    target: clock_time,
    batch_size: int,
    lookback_seconds: int,
    limiter: RateLimiter,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing = 0
    for day in tqdm(sessions, desc="Downloading scheduled NBBO", unit="session"):
        target_at = target_timestamp(day, target)
        for batch in _batches(symbols, batch_size):
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
                if symbol in selected:
                    rows.append(selected[symbol])
                    continue
                fallback = _quote_request(
                    session,
                    headers,
                    limiter,
                    [symbol],
                    target_at - timedelta(seconds=lookback_seconds),
                    target_at,
                    1,
                )
                candidate = select_causal_quotes(fallback, day, target_at).get(symbol)
                if candidate is None:
                    missing += 1
                else:
                    rows.append(candidate)
    if missing:
        print(f"summary: {missing:,} symbol-sessions had no causal quote")
    return rows


def _ordered_unique_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique = {(str(row["symbol"]), str(row["date"])): row for row in rows}
    return sorted(unique.values(), key=lambda row: (str(row["symbol"]), str(row["date"])))


def merge_quote_rows(
    existing_rows: Iterable[dict[str, object]],
    refreshed_rows: Iterable[dict[str, object]],
    refreshed_symbols: set[str],
    refresh_start: str,
) -> list[dict[str, object]]:
    kept = [
        row
        for row in existing_rows
        if not (
            str(row["symbol"]) in refreshed_symbols
            and str(row["date"]) >= refresh_start
        )
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
        if int(np.asarray(data["format_version"]).item()) != FORMAT_VERSION:
            raise ValueError("unsupported NBBO NPZ format version")
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
    update_metadata: dict[str, object] | None = None,
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
        "symbols": symbols,
        "symbol_count": len(symbols),
        "quote_count": len(ordered),
        "format": "numpy-npz-columnar-v1",
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
    symbols: list[str],
    start: str,
    end: str,
    target: clock_time,
    batch_size: int,
    lookback_seconds: int,
    requests_per_minute: int,
) -> tuple[list[dict[str, object]], int]:
    limiter = RateLimiter(requests_per_minute)
    headers = _request_headers()
    with requests.Session() as session:
        sessions = _calendar_days(session, start, end, limiter)
        sessions, deferred = eligible_sessions(sessions, target)
        rows = _download_quote_rows(
            session,
            headers,
            symbols,
            sessions,
            target,
            batch_size,
            lookback_seconds,
            limiter,
        )
    return rows, deferred


def download_nbbo(
    symbols: list[str],
    start: str,
    end: str,
    output: Path,
    target: clock_time,
    batch_size: int = 10,
    lookback_seconds: int = 60,
    requests_per_minute: int = 180,
) -> tuple[int, int, int]:
    rows, deferred = _fetch(
        symbols, start, end, target, batch_size, lookback_seconds, requests_per_minute
    )
    headers = _request_headers()
    split_end = max(pd.Timestamp(end).date(), datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        splits = _download_splits(session, headers, symbols, start, split_end)
    _write_dataset(output, rows, splits, symbols, start, end, target, lookback_seconds)
    return len(symbols), len(_ordered_unique_rows(rows)), deferred


def update_nbbo(
    output: Path,
    end: str,
    additional_symbols: set[str],
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
    start = str(manifest["start"])
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
    existing_symbols = {_security_symbol(value) for value in manifest["symbols"]}
    new_symbols = {_security_symbol(value) for value in additional_symbols} - existing_symbols
    all_symbols = sorted(existing_symbols | new_symbols)
    refreshed, deferred = _fetch(
        sorted(existing_symbols),
        refresh_start,
        end,
        target,
        batch_size,
        lookback_seconds,
        requests_per_minute,
    )
    if new_symbols:
        new_rows, new_deferred = _fetch(
            sorted(new_symbols),
            start,
            end,
            target,
            batch_size,
            lookback_seconds,
            requests_per_minute,
        )
        refreshed.extend(new_rows)
        deferred += new_deferred
    # Refresh the complete split ledger once for the final universe.
    headers = _request_headers()
    split_end = max(pd.Timestamp(end).date(), datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        splits = _download_splits(session, headers, all_symbols, start, split_end)
    merged = merge_quote_rows(
        _load_raw_rows(output), refreshed, existing_symbols | new_symbols, refresh_start
    )
    _write_dataset(
        output,
        merged,
        splits,
        all_symbols,
        start,
        end,
        target,
        lookback_seconds,
        {
            "previous_end": previous_end,
            "refresh_start": refresh_start,
            "overlap_days": overlap_days,
            "new_symbols": sorted(new_symbols),
        },
    )
    return len(all_symbols), len(merged), deferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=default_end())
    parser.add_argument("--target-time", type=parse_target_time, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--lookback-seconds", type=int, default=60)
    parser.add_argument("--symbols-from-trades", action="append", default=[])
    parser.add_argument("--symbols-file", action="append", default=[])
    parser.add_argument("--symbols", default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--requests-per-minute", type=int, default=180)
    args = parser.parse_args()
    if not args.update and not args.start:
        parser.error("--start is required unless --update is used")
    if min(args.overlap_days, args.lookback_seconds, args.batch_size, args.requests_per_minute) < 1:
        parser.error("overlap, lookback, batch size, and request rate must be positive")
    symbols = {_security_symbol(value) for value in args.symbols.split(",") if value.strip()}
    for value in args.symbols_from_trades:
        symbols.update(symbols_from_trade_csv(Path(value)))
    for value in args.symbols_file:
        symbols.update(symbols_from_file(Path(value)))
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.update:
        count, quotes, deferred = update_nbbo(
            output,
            args.end,
            symbols,
            args.overlap_days,
            args.batch_size,
            args.lookback_seconds,
            args.requests_per_minute,
            args.target_time,
        )
    else:
        symbols.add("SPY")
        target = args.target_time or parse_target_time("15:45")
        count, quotes, deferred = download_nbbo(
            sorted(symbols),
            str(args.start),
            args.end,
            output,
            target,
            args.batch_size,
            args.lookback_seconds,
            args.requests_per_minute,
        )
    suffix = f"; deferred {deferred} session(s) inside the SIP delay" if deferred else ""
    print(f"wrote {quotes:,} scheduled NBBO quotes for {count} symbols to {output}{suffix}")


if __name__ == "__main__":
    main()
