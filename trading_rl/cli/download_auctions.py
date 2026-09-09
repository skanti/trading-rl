"""Download split-adjusted Alpaca SIP opening/closing auctions for backtests."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
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

from ..market_data.download_mode import use_incremental_download
from ..market_data.download_output import current_report, download_output, download_progress, info

LOGGER = logging.getLogger(__name__)


AUCTIONS_URL = "https://data.alpaca.markets/v2/stocks/auctions"
CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
FIELDNAMES = (
    "symbol",
    "date",
    "session",
    "condition",
    "price",
    "size",
    "timestamp",
    "exchange",
)
SPLIT_FIELDNAMES = ("type", "symbol", "ex_date", "old_rate", "new_rate", "id")
FORMAT_VERSION = 1
EASTERN = ZoneInfo("America/New_York")
RECENT_SIP_SAFETY_DELAY = timedelta(minutes=20)


def default_end(now: datetime | None = None) -> str:
    """Return today's US trading-calendar date."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(EASTERN).date().isoformat()


def auction_query_end(
    requested_end: str,
    now: datetime | None = None,
) -> tuple[str, bool]:
    """Clamp current-day requests behind Alpaca's recent-SIP restriction."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    reference = reference.astimezone(timezone.utc)
    parsed = pd.Timestamp(requested_end)
    requested_day = parsed.date()
    today = reference.astimezone(EASTERN).date()
    if requested_day > today:
        raise ValueError(f"auction end {requested_end} is in the future")
    if requested_day < today:
        return requested_end, False

    delayed_cutoff = reference - RECENT_SIP_SAFETY_DELAY
    if delayed_cutoff.astimezone(EASTERN).date() < requested_day:
        raise ValueError(
            "today's delayed SIP window has not started yet; retry in 20 minutes"
        )

    if len(requested_end) > 10:
        explicit = parsed
        if explicit.tzinfo is None:
            explicit = explicit.tz_localize(timezone.utc)
        explicit_utc = explicit.tz_convert(timezone.utc).to_pydatetime()
        delayed_cutoff = min(delayed_cutoff, explicit_utc)
    return delayed_cutoff.isoformat().replace("+00:00", "Z"), True


def _security_symbol(sample_id: str) -> str:
    symbol = str(sample_id)
    if symbol.startswith("ST-"):
        symbol = symbol[3:]
    return symbol.replace("-", ".").upper()


def symbols_from_trade_csv(path: Path) -> list[str]:
    trades = pd.read_csv(path, usecols=["sample_id"])
    return sorted({_security_symbol(symbol) for symbol in trades.sample_id})


def symbols_from_file(path: Path) -> list[str]:
    """Read one symbol per line, ignoring blanks and comment lines."""
    return sorted(
        {
            _security_symbol(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    )


def flatten_auctions(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    auctions = payload.get("auctions", {})
    if not isinstance(auctions, dict):
        raise ValueError("Alpaca response has no auctions mapping")
    for symbol, days in auctions.items():
        if not isinstance(days, list):
            continue
        for day in days:
            if not isinstance(day, dict):
                continue
            auction_date = str(day.get("d", ""))
            for response_key, session_name in (("o", "open"), ("c", "close")):
                prints = day.get(response_key, [])
                if not isinstance(prints, list):
                    continue
                for record in prints:
                    if not isinstance(record, dict):
                        continue
                    rows.append(
                        {
                            "symbol": str(symbol),
                            "date": auction_date,
                            "session": session_name,
                            "condition": str(record.get("c", "")),
                            "price": record.get("p"),
                            "size": record.get("s"),
                            "timestamp": str(record.get("t", "")),
                            "exchange": str(record.get("x", "")),
                        }
                    )
    return rows


def _request_page(
    session: requests.Session,
    headers: dict[str, str],
    params: dict[str, object],
    url: str = AUCTIONS_URL,
    retries: int = 8,
) -> dict[str, object]:
    for attempt in range(retries):
        response = session.get(url, headers=headers, params=params, timeout=(10, 90))
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt + 1 == retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 2.0**attempt)
            time.sleep(delay)
            continue
        if not 200 <= response.status_code < 300:
            try:
                message = str(response.json().get("message") or response.text)
            except (TypeError, ValueError):
                message = response.text
            raise RuntimeError(
                f"Alpaca auction request failed with HTTP {response.status_code}: "
                f"{message.strip()}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Alpaca returned a non-object auction response")
        return payload
    raise AssertionError("auction retry loop exited unexpectedly")


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _request_headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_DATA_KEY")
    secret = os.environ.get("ALPACA_DATA_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_DATA_KEY and ALPACA_DATA_SECRET must be set")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "accept": "application/json",
    }


def _download_auction_rows(
    session: requests.Session,
    headers: dict[str, str],
    symbols: list[str],
    start: str,
    end: str,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with download_progress(total=len(symbols), description="Fetching auctions") as progress:
        for batch in _batches(symbols, batch_size):
            page_token: str | None = None
            while True:
                params: dict[str, object] = {
                    "symbols": ",".join(batch),
                    "start": start,
                    "end": end,
                    "feed": "sip",
                    "limit": 10_000,
                    "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token
                payload = _request_page(session, headers, params)
                rows.extend(flatten_auctions(payload))
                next_token = payload.get("next_page_token")
                if not next_token:
                    break
                page_token = str(next_token)
            progress.update(len(batch))
    return rows


def _download_splits(
    session: requests.Session,
    headers: dict[str, str],
    symbols: list[str],
    start: str,
    end: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    page_token: str | None = None
    while True:
        params: dict[str, object] = {
            "symbols": ",".join(symbols),
            "types": "forward_split,reverse_split",
            "start": start,
            "end": end,
            "limit": 1_000,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        payload = _request_page(session, headers, params, url=CORPORATE_ACTIONS_URL)
        actions = payload.get("corporate_actions", {})
        if not isinstance(actions, dict):
            raise ValueError("Alpaca response has no corporate_actions mapping")
        for response_key, action_type in (
            ("forward_splits", "forward_split"),
            ("reverse_splits", "reverse_split"),
        ):
            records = actions.get(response_key, [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                rows.append(
                    {
                        "type": action_type,
                        "symbol": record.get("symbol"),
                        "ex_date": record.get("ex_date"),
                        "old_rate": record.get("old_rate"),
                        "new_rate": record.get("new_rate"),
                        "id": record.get("id"),
                    }
                )
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        page_token = str(next_token)
    return rows


def _ordered_unique_auction_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique_rows = {tuple(row[field] for field in FIELDNAMES): row for row in rows}
    return sorted(
        unique_rows.values(),
        key=lambda row: (
            str(row["symbol"]),
            str(row["date"]),
            str(row["session"]),
            str(row["timestamp"]),
            str(row["exchange"]),
            str(row["condition"]),
        ),
    )


def merge_auction_rows(
    existing_rows: Iterable[dict[str, object]],
    refreshed_rows: Iterable[dict[str, object]],
    refreshed_symbols: set[str],
    refresh_start: str,
) -> list[dict[str, object]]:
    """Replace an overlapping tail for refreshed symbols, then deduplicate."""
    kept = [
        row
        for row in existing_rows
        if not (
            str(row["symbol"]) in refreshed_symbols
            and str(row["date"]) >= refresh_start
        )
    ]
    return _ordered_unique_auction_rows([*kept, *refreshed_rows])


def _ordered_splits(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique = {
        tuple(str(row.get(field, "")) for field in SPLIT_FIELDNAMES): row
        for row in rows
    }
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["symbol"]),
            str(row["ex_date"]),
            str(row["id"]),
        ),
    )


def _dataset_arrays(
    rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Build columnar arrays and apply the complete split ledger once."""
    symbols = np.asarray([str(row["symbol"]) for row in rows])
    dates = np.asarray([str(row["date"]) for row in rows], dtype="datetime64[D]")
    raw_price = np.asarray([row["price"] for row in rows], dtype=np.float64)
    raw_size = np.asarray([row["size"] for row in rows], dtype=np.float64)
    price = raw_price.copy()
    size = raw_size.copy()

    split_symbols: list[str] = []
    split_dates: list[str] = []
    split_old_rates: list[float] = []
    split_new_rates: list[float] = []
    for split in split_rows:
        symbol = str(split["symbol"])
        ex_date = str(split["ex_date"])
        old_rate = float(split["old_rate"])
        new_rate = float(split["new_rate"])
        if not np.isfinite(old_rate) or not np.isfinite(new_rate):
            raise ValueError(f"non-finite split rate for {symbol} on {ex_date}")
        if old_rate <= 0.0 or new_rate <= 0.0:
            raise ValueError(f"non-positive split rate for {symbol} on {ex_date}")
        ratio = new_rate / old_rate
        before_split = (symbols == symbol) & (dates < np.datetime64(ex_date, "D"))
        price[before_split] /= ratio
        size[before_split] *= ratio
        split_symbols.append(symbol)
        split_dates.append(ex_date)
        split_old_rates.append(old_rate)
        split_new_rates.append(new_rate)

    return {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int16),
        "split_adjusted": np.asarray(True),
        "symbol": symbols,
        "date": dates,
        "session": np.asarray(
            [0 if str(row["session"]) == "open" else 1 for row in rows],
            dtype=np.uint8,
        ),
        "condition": np.asarray([str(row["condition"]) for row in rows]),
        "price": price,
        "size": size,
        "raw_price": raw_price,
        "raw_size": raw_size,
        "timestamp": np.asarray([str(row["timestamp"]) for row in rows]),
        "exchange": np.asarray([str(row["exchange"]) for row in rows]),
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


def _load_raw_auction_rows(path: Path) -> list[dict[str, object]]:
    """Load raw fields retained in an adjusted NPZ for a subsequent update."""
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["format_version"]).item())
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported auction NPZ format version: {version}")
        count = len(data["symbol"])
        columns = {
            "symbol": data["symbol"].astype(str),
            "date": np.datetime_as_string(data["date"], unit="D"),
            "session": data["session"],
            "condition": data["condition"].astype(str),
            "price": data["raw_price"],
            "size": data["raw_size"],
            "timestamp": data["timestamp"].astype(str),
            "exchange": data["exchange"].astype(str),
        }
        if any(len(values) != count for values in columns.values()):
            raise ValueError(f"auction NPZ columns have inconsistent lengths: {path}")
        return [
            {
                "symbol": columns["symbol"][index],
                "date": columns["date"][index],
                "session": "open" if int(columns["session"][index]) == 0 else "close",
                "condition": columns["condition"][index],
                "price": float(columns["price"][index]),
                "size": float(columns["size"][index]),
                "timestamp": columns["timestamp"][index],
                "exchange": columns["exchange"][index],
            }
            for index in range(count)
        ]


def _write_dataset(
    output_path: Path,
    rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
    symbols: list[str],
    start: str,
    end: str,
    update_metadata: dict[str, object] | None = None,
    split_adjusted_as_of: str | None = None,
    symbol_end_dates: dict[str, str] | None = None,
) -> None:
    ordered = _ordered_unique_auction_rows(rows)
    ordered_splits = _ordered_splits(split_rows)
    arrays = _dataset_arrays(ordered, ordered_splits)
    _atomic_write_npz(output_path, arrays)

    manifest: dict[str, object] = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": AUCTIONS_URL,
        "feed": "sip",
        "start": start,
        "end": end,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "auction_print_count": len(ordered),
        "official_open_condition": "O",
        "format": "numpy-npz-columnar-v1",
        "prices": "split-adjusted",
        "split_adjusted_as_of": split_adjusted_as_of or end,
        "raw_prices_embedded": True,
        "split_ledger_embedded": True,
        "split_adjustment_count": len(ordered_splits),
    }
    if update_metadata:
        manifest["last_update"] = update_metadata
    if symbol_end_dates is not None:
        manifest["symbol_end_dates"] = symbol_end_dates
    manifest_path = output_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest_path.parent, delete=False
    ) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_manifest = Path(temporary.name)
    temporary_manifest.chmod(
        manifest_path.stat().st_mode & 0o777 if manifest_path.exists() else 0o664
    )
    temporary_manifest.replace(manifest_path)


def download_auctions(
    symbols: list[str],
    start: str,
    end: str,
    output_path: Path,
    batch_size: int = 50,
) -> tuple[int, int]:
    report = current_report()
    if report:
        report.set("Requested", len(symbols), "symbols")
    headers = _request_headers()
    start_bound = pd.Timestamp(start).date().isoformat()
    end_bound = pd.Timestamp(end).date().isoformat()
    query_end, _current_day = auction_query_end(end)
    split_end = max(
        pd.Timestamp(end).date(), datetime.now(timezone.utc).date()
    ).isoformat()
    with requests.Session() as session:
        rows = _download_auction_rows(
            session, headers, symbols, start, query_end, batch_size
        )
        split_rows = _download_splits(session, headers, symbols, start, split_end)
    rows = [
        row for row in rows if start_bound <= str(row["date"]) <= end_bound
    ]
    _write_dataset(
        output_path,
        rows,
        split_rows,
        symbols,
        start,
        end,
        split_adjusted_as_of=split_end,
    )
    if report:
        returned = {str(row["symbol"]) for row in rows}
        report.set("Full redownloads" if report.mode == "Rebuild" else "New downloads", len(returned), "symbols")
        report.set("Without auction prints", len(set(symbols) - returned), "symbols")
        report.set("Prints in dataset", len(_ordered_unique_auction_rows(rows)), "prints")
        if missing := sorted(set(symbols) - returned):
            LOGGER.warning("No auction prints returned for %d symbols (examples: %s)", len(missing), ", ".join(missing[:5]))
    return len(symbols), len(_ordered_unique_auction_rows(rows))


def update_auctions(
    output_path: Path,
    end: str,
    additional_symbols: set[str],
    overlap_days: int = 7,
    batch_size: int = 50,
    refresh_requested_only: bool = False,
) -> tuple[int, int]:
    manifest_path = output_path.with_suffix(".json")
    if not output_path.exists() or not manifest_path.exists():
        raise ValueError("--update requires the existing auction NPZ and JSON manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_start = str(manifest["start"])
    previous_end = str(manifest["end"])
    dataset_start_date = pd.Timestamp(dataset_start).date()
    previous_end_date = pd.Timestamp(previous_end).date()
    requested_end_date = pd.Timestamp(end).date()
    if requested_end_date < previous_end_date:
        raise ValueError(f"update end {end} precedes existing end {previous_end}")
    dataset_start_bound = dataset_start_date.isoformat()
    requested_end_bound = requested_end_date.isoformat()
    query_end, _current_day = auction_query_end(end)

    existing_symbols = {_security_symbol(symbol) for symbol in manifest["symbols"]}
    requested_symbols = {_security_symbol(symbol) for symbol in additional_symbols}
    if refresh_requested_only:
        if not requested_symbols:
            raise ValueError("refresh-requested-only requires an explicit symbol selection")
        requested_symbols.add("SPY")
    new_symbols = requested_symbols - existing_symbols
    all_symbols = sorted(existing_symbols | new_symbols)
    refresh_symbols = requested_symbols if refresh_requested_only else set(all_symbols)
    refresh_existing = existing_symbols & refresh_symbols
    report = current_report()
    if report:
        report.set("Requested", len(refresh_symbols), "symbols")
        report.set("Not refreshed", len(existing_symbols - refresh_symbols), "symbols")
    # A retained symbol may have missed several updates outside the shortlist.
    # Resume from its own coverage date, not the dataset's newest end date.
    saved_end_dates = manifest.get("symbol_end_dates", {})
    symbol_end_dates = {
        symbol: saved_end_dates.get(symbol, previous_end) for symbol in existing_symbols
    }
    oldest_end = min(
        (pd.Timestamp(symbol_end_dates[symbol]).date() for symbol in refresh_existing),
        default=previous_end_date,
    )
    refresh_start = max(
        dataset_start_date, oldest_end - timedelta(days=int(overlap_days) - 1)
    ).isoformat()
    info(
        f"refreshing auctions for {len(refresh_symbols):,} symbols "
        f"({len(refresh_existing):,} existing, {len(new_symbols):,} new); "
        f"retaining {len(all_symbols):,} symbols in the dataset"
    )
    existing_rows = _load_raw_auction_rows(output_path)

    headers = _request_headers()
    split_end = max(requested_end_date, datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        refreshed_existing_rows = []
        if refresh_existing:
            refreshed_existing_rows = _download_auction_rows(
                session,
                headers,
                sorted(refresh_existing),
                refresh_start,
                query_end,
                batch_size,
            )
        # Alpaca can occasionally include a print just outside the requested
        # calendar boundary. Excluding it prevents a retained row before the
        # replacement tail from acquiring a second, differently typed copy.
        refreshed_rows = [
            row
            for row in refreshed_existing_rows
            if refresh_start <= str(row["date"]) <= requested_end_bound
        ]
        if new_symbols:
            info(
                f"new symbols: downloading full retained history for "
                f"{', '.join(sorted(new_symbols))}"
            )
            new_symbol_rows = _download_auction_rows(
                session,
                headers,
                sorted(new_symbols),
                dataset_start,
                query_end,
                batch_size,
            )
            refreshed_rows.extend(
                row
                for row in new_symbol_rows
                if dataset_start_bound <= str(row["date"]) <= requested_end_bound
            )
        # Split actions are tiny and can be revised, so refresh their full retained
        # history instead of attempting a fragile incremental action merge.
        split_rows = _download_splits(
            session, headers, all_symbols, dataset_start, split_end
        )

    merged = merge_auction_rows(
        existing_rows,
        refreshed_rows,
        refresh_symbols,
        refresh_start,
    )
    update_metadata = {
        "previous_end": previous_end,
        "refresh_start": refresh_start,
        "overlap_days": int(overlap_days),
        "new_symbols": sorted(new_symbols),
        "refreshed_symbols": sorted(refresh_symbols),
    }
    symbol_end_dates.update({symbol: end for symbol in refresh_symbols})
    _write_dataset(
        output_path,
        merged,
        split_rows,
        all_symbols,
        dataset_start,
        end,
        update_metadata,
        split_adjusted_as_of=split_end,
        symbol_end_dates=symbol_end_dates,
    )
    if report:
        returned = {str(row["symbol"]) for row in refreshed_rows}
        report.set("Updated", len(refresh_existing & returned), "symbols")
        report.set("New downloads", len(new_symbols & returned), "symbols")
        report.set("Without auction prints", len(refresh_symbols - returned), "symbols")
        report.set("Prints in dataset", len(merged), "prints")
        if missing := sorted(refresh_symbols - returned):
            LOGGER.warning("No auction prints returned for %d symbols (examples: %s)", len(missing), ", ".join(missing[:5]))
    return len(all_symbols), len(merged)


@download_output("Auctions", LOGGER, exit_on_error=True)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=None,
        help="inclusive RFC-3339 or YYYY-MM-DD; required for an initial download",
    )
    parser.add_argument(
        "--end",
        default=default_end(),
        help="inclusive RFC-3339 or YYYY-MM-DD (default: today in New York)",
    )
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update",
        action="store_true",
        help="require an existing dataset (default: create if absent, update if present)",
    )
    mode.add_argument("--rebuild", action="store_true", help="redownload the full dataset")
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=7,
        help="calendar days replaced during updates to capture late corrections (default: 7)",
    )
    parser.add_argument(
        "--refresh-requested-only",
        action="store_true",
        help=(
            "when updating, refresh only explicitly selected symbols plus SPY; "
            "retain other stored history"
        ),
    )
    parser.add_argument(
        "--symbols-from-trades",
        action="append",
        default=[],
        help="CSV containing the simulator's sample_id column; may be repeated",
    )
    parser.add_argument(
        "--symbols-file",
        action="append",
        default=[],
        help="text file containing one symbol per line; may be repeated",
    )
    parser.add_argument("--symbols", default="", help="additional comma-separated symbols")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    report = current_report()
    report.output = args.output
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.overlap_days < 1:
        parser.error("--overlap-days must be positive")
    output_path = Path(args.output)
    try:
        incremental = use_incremental_download(
            output_path, update=args.update, rebuild=args.rebuild
        )
    except ValueError as error:
        parser.error(str(error))
    report.mode = "Rebuild" if args.rebuild else "Update" if incremental else "Initial download"
    for label in ("Requested", "Updated", "New downloads", "Full redownloads", "Without auction prints", "Not refreshed"):
        report.set(label, 0, "symbols")
    if args.start is None and output_path.exists():
        args.start = json.loads(output_path.with_suffix(".json").read_text())["start"]
    if not incremental and not args.start:
        parser.error("--start is required for an initial download")
    try:
        query_end, current_day = auction_query_end(args.end)
    except ValueError as error:
        parser.error(str(error))
    if current_day:
        info(
            f"current-day SIP query is delayed through {query_end}; "
            "today's closing auctions may be incomplete and will be filled by "
            "the next overlap refresh"
        )
    symbols = {_security_symbol(symbol) for symbol in args.symbols.split(",") if symbol.strip()}
    for path in args.symbols_from_trades:
        symbols.update(symbols_from_trade_csv(Path(path)))
    for path in args.symbols_file:
        symbols.update(symbols_from_file(Path(path)))
    if args.rebuild and not symbols and output_path.exists():
        symbols.update(json.loads(output_path.with_suffix(".json").read_text())["symbols"])
    if args.refresh_requested_only and not symbols:
        parser.error("--refresh-requested-only requires an explicit symbol selection")
    if incremental:
        count, prints = update_auctions(
            output_path,
            args.end,
            symbols,
            args.overlap_days,
            args.batch_size,
            refresh_requested_only=args.refresh_requested_only,
        )
    else:
        symbols.add("SPY")
        count, prints = download_auctions(
            sorted(symbols),
            str(args.start),
            args.end,
            output_path,
            args.batch_size,
        )



if __name__ == "__main__":
    main()
