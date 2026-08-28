"""Download split-adjusted Alpaca SIP opening/closing auctions for backtests."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests


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
        response.raise_for_status()
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
    for batch_number, batch in enumerate(_batches(symbols, batch_size), start=1):
        page_token: str | None = None
        page_number = 0
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
            page_number += 1
            page_rows = flatten_auctions(payload)
            rows.extend(page_rows)
            print(
                f"batch {batch_number}: page {page_number}, "
                f"{len(page_rows):,} auction prints"
            )
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            page_token = str(next_token)
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
    headers = _request_headers()
    start_bound = pd.Timestamp(start).date().isoformat()
    end_bound = pd.Timestamp(end).date().isoformat()
    split_end = max(
        pd.Timestamp(end).date(), datetime.now(timezone.utc).date()
    ).isoformat()
    with requests.Session() as session:
        rows = _download_auction_rows(session, headers, symbols, start, end, batch_size)
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
    return len(symbols), len(_ordered_unique_auction_rows(rows))


def update_auctions(
    output_path: Path,
    end: str,
    additional_symbols: set[str],
    overlap_days: int = 7,
    batch_size: int = 50,
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
    refresh_start_date = max(
        dataset_start_date,
        previous_end_date - timedelta(days=int(overlap_days) - 1),
    )
    refresh_start = refresh_start_date.isoformat()
    dataset_start_bound = dataset_start_date.isoformat()
    requested_end_bound = requested_end_date.isoformat()

    existing_symbols = {_security_symbol(symbol) for symbol in manifest["symbols"]}
    new_symbols = {_security_symbol(symbol) for symbol in additional_symbols} - existing_symbols
    all_symbols = sorted(existing_symbols | new_symbols)
    existing_rows = _load_raw_auction_rows(output_path)

    headers = _request_headers()
    split_end = max(requested_end_date, datetime.now(timezone.utc).date()).isoformat()
    with requests.Session() as session:
        refreshed_existing_rows = _download_auction_rows(
            session,
            headers,
            sorted(existing_symbols),
            refresh_start,
            end,
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
            print(
                f"new symbols: downloading full retained history for "
                f"{', '.join(sorted(new_symbols))}"
            )
            new_symbol_rows = _download_auction_rows(
                session,
                headers,
                sorted(new_symbols),
                dataset_start,
                end,
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
        existing_symbols | new_symbols,
        refresh_start,
    )
    update_metadata = {
        "previous_end": previous_end,
        "refresh_start": refresh_start,
        "overlap_days": int(overlap_days),
        "new_symbols": sorted(new_symbols),
    }
    _write_dataset(
        output_path,
        merged,
        split_rows,
        all_symbols,
        dataset_start,
        end,
        update_metadata,
        split_adjusted_as_of=split_end,
    )
    return len(all_symbols), len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=None,
        help="inclusive RFC-3339 or YYYY-MM-DD; required for an initial download",
    )
    parser.add_argument("--end", required=True, help="inclusive RFC-3339 or YYYY-MM-DD")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--update",
        action="store_true",
        help="refresh an overlapping tail of an existing dataset and retain its symbol set",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=7,
        help="calendar days replaced during --update to capture late corrections (default: 7)",
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
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.overlap_days < 1:
        parser.error("--overlap-days must be positive")
    if not args.update and not args.start:
        parser.error("--start is required unless --update is used")
    symbols = {_security_symbol(symbol) for symbol in args.symbols.split(",") if symbol.strip()}
    for path in args.symbols_from_trades:
        symbols.update(symbols_from_trade_csv(Path(path)))
    for path in args.symbols_file:
        symbols.update(symbols_from_file(Path(path)))
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.update:
        count, prints = update_auctions(
            output_path,
            args.end,
            symbols,
            args.overlap_days,
            args.batch_size,
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
    print(f"wrote {prints:,} auction prints for {count} symbols to {args.output}")


if __name__ == "__main__":
    main()
