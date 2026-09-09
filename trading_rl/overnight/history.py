"""Historical session, universe and daily-liquidity inputs shared by ranking and simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..market_data.calendar import auction_close_minutes, short_entry_dates
from ..market_data.schema import BAR_INDEX, validate_bar_columns, validate_bar_manifest

REFERENCE_SYMBOL = "SPY"
EXTENDED_OPEN_MINUTE = 4 * 60
REGULAR_OPEN_MINUTE = 9 * 60 + 30
REGULAR_CLOSE_MINUTE = 16 * 60
MIN_USABLE_SESSION_BARS = 120
BAR_ORIGIN = datetime(2010, 1, 1, tzinfo=timezone.utc)
EASTERN = ZoneInfo("America/New_York")
DEFAULT_DATA_DIR = "/data/ppv1/updates/bars_1min_2022-01-01"
DEFAULT_DAILY_DATA_DIR = "/data/ppv1/updates/bars_1day_2022-01-01"
DEFAULT_AUCTIONS_PATH = "/data/ppv1/updates/alpaca_auctions_2022-01-01.npz"
DEFAULT_SECURITY_MASTER_CACHE = (
    "/tmp/trading/baseline_cache/nasdaq_security_master.json"
)
NASDAQ_SYMBOL_DIRECTORY_URLS = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
)
NON_COMPANY_NAME_PATTERN = re.compile(
    r"\b(closed[ -]end fund|fund|etf|etn|exchange[ -]traded|business development company|"
    r"warrants?|rights?|units?|preferred (stock|shares?)|notes?|bonds?|debentures?)\b|"
    r"\broyalty trust\b|\bacquisition (corp(?:oration)?|company|co\.?|inc\.?|limited|ltd\.?)\b",
    re.IGNORECASE,
)
OPERATING_TRUST_PATTERN = re.compile(
    r"\b(common stock|reit|realty|properties|property|commercial|residential|mortgage)\b",
    re.IGNORECASE,
)


def _security_symbol(sample_id: str) -> str:
    return str(sample_id).replace("-", ".").upper()


def _official_opening_auctions(
    path: Path,
    dates: pd.DatetimeIndex | None = None,
    symbols: np.ndarray | None = None,
) -> pd.DataFrame:
    """Select requested official opens without expanding every print into pandas."""
    if path.suffix.lower() != ".npz":
        raise ValueError(f"auction data must use the split-adjusted NPZ format: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "symbol",
            "date",
            "session",
            "condition",
            "price",
            "size",
            "exchange",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing auction arrays: {sorted(missing)}")
        if "split_adjusted" not in data.files or not bool(
            data["split_adjusted"].item()
        ):
            raise ValueError(f"{path} does not contain pre-adjusted auction prices")
        session_codes = np.asarray(data["session"], dtype=np.uint8)
        conditions = np.asarray(data["condition"]).astype(str, copy=False)
        price_values = np.asarray(data["price"], dtype=np.float64)
        raw_price_values = (
            np.asarray(data["raw_price"], dtype=np.float64)
            if "raw_price" in data.files
            else price_values
        )
        selected = np.flatnonzero(
            (session_codes == 0)
            & (conditions == "O")
            & np.isfinite(price_values)
            & (price_values > 0.0)
        )
        if not len(selected):
            raise ValueError(f"{path} contains no condition-O opening auctions")

        date_values = np.asarray(data["date"], dtype="datetime64[D]")
        if dates is not None:
            wanted_dates = np.asarray(pd.DatetimeIndex(dates), dtype="datetime64[D]")
            selected = selected[np.isin(date_values[selected], wanted_dates)]

        symbol_values = np.asarray(data["symbol"]).astype(str, copy=False)
        selected_symbols = np.char.upper(symbol_values[selected])
        if symbols is not None:
            wanted_symbols = np.asarray(
                [_security_symbol(sample_id) for sample_id in symbols], dtype=str
            )
            keep = np.isin(selected_symbols, wanted_symbols)
            selected = selected[keep]
            selected_symbols = selected_symbols[keep]

        official = pd.DataFrame(
            {
                "symbol": selected_symbols,
                "date": pd.to_datetime(date_values[selected]),
                "price": price_values[selected],
                "raw_price": raw_price_values[selected],
                "size": np.asarray(data["size"], dtype=np.float64)[selected],
                "exchange": np.asarray(data["exchange"]).astype(str, copy=False)[
                    selected
                ],
            }
        )
    if official.empty:
        return official
    primary_venue_priority = {"N": 0, "Q": 0, "P": 0, "A": 0, "T": 1, "V": 2}
    official["venue_priority"] = (
        official.exchange.map(primary_venue_priority).fillna(3).astype(np.int8)
    )
    official.sort_values(
        ["size", "venue_priority"], ascending=[False, True], inplace=True
    )
    official.drop_duplicates(["symbol", "date"], inplace=True)
    return official


def load_primary_auction_exchange_mask(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    exchange: str,
    *,
    official: pd.DataFrame | None = None,
) -> tuple[np.ndarray, int]:
    """Use historical primary auctions to refine a current-exchange universe.

    Missing symbol-dates remain eligible because the current security master has
    already applied the requested venue filter. Known historical auctions override
    it, which handles listing transfers without excluding symbols lacking downloaded
    auction history before they are ever selected.

    A venue is matched against the whole family of codes the SIP uses for one
    listing market. Alpaca reports a single Nasdaq opening cross under both "Q"
    and "T", and on some sessions only the "T" copy survives, so matching "Q"
    alone would silently drop genuine Nasdaq listings on those dates. Where a
    name really did list elsewhere the primary print carries that venue's own
    code and still wins the size/priority dedupe, so listing transfers remain
    detected.
    """
    exchange_codes = {"nasdaq": frozenset({"Q", "T"})}
    if exchange not in exchange_codes:
        raise ValueError(f"unsupported exchange filter: {exchange}")
    wanted = exchange_codes[exchange]
    if official is None:
        official = _official_opening_auctions(path, dates, symbols)
    mask = np.ones((len(dates), len(symbols)), dtype=bool)
    rows = pd.Index(dates).get_indexer(pd.DatetimeIndex(official["date"]))
    symbol_index = pd.Index([_security_symbol(sample_id) for sample_id in symbols])
    columns = symbol_index.get_indexer(official["symbol"].astype(str).str.upper())
    valid = (rows >= 0) & (columns >= 0)
    matching_exchange = official["exchange"].astype(str).isin(wanted).to_numpy()
    mask[rows[valid], columns[valid]] = matching_exchange[valid]
    return mask, int(valid.sum())


def is_company_security(record: dict[str, object]) -> tuple[bool, str]:
    """Classify common company equity from Nasdaq's symbol-directory fields."""
    name = str(record.get("name", ""))
    if str(record.get("test_issue", "N")).upper() == "Y":
        return False, "test issue"
    if str(record.get("etf", "N")).upper() == "Y":
        return False, "ETF/ETP"
    match = NON_COMPANY_NAME_PATTERN.search(name)
    if match:
        return False, match.group(0).lower()
    if "trust" in name.lower() and not OPERATING_TRUST_PATTERN.search(name):
        return False, "non-operating trust"
    return True, "company equity"


def load_nasdaq_security_master(
    cache_path: Path,
    refresh: bool = False,
    max_age_days: int = 7,
) -> dict[str, dict[str, object]]:
    """Load a cached Nasdaq security master or refresh it from official files."""
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text())
        fetched_at = datetime.fromisoformat(str(payload["fetched_at"]))
        age = datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)
        cached_securities = dict(payload["securities"])
        has_exchange_metadata = all(
            "exchange" in record for record in cached_securities.values()
        )
        if age.days < int(max_age_days) and has_exchange_metadata:
            return cached_securities

    frames = []
    for url in NASDAQ_SYMBOL_DIRECTORY_URLS:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "trading-rl security-universe research"},
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), sep="|")
        frame = frame[
            ~frame.iloc[:, 0].astype(str).str.startswith("File Creation Time")
        ]
        if "Symbol" in frame.columns:
            frame = frame.rename(columns={"Symbol": "symbol", "Security Name": "name"})
            frame["exchange"] = "Q"
        else:
            frame = frame.rename(
                columns={
                    "ACT Symbol": "symbol",
                    "Security Name": "name",
                    "Exchange": "exchange",
                }
            )
        frame = frame.rename(columns={"ETF": "etf", "Test Issue": "test_issue"})
        frames.append(frame.loc[:, ["symbol", "name", "etf", "test_issue", "exchange"]])
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        "symbol", keep="first"
    )
    securities = {
        str(row.symbol).upper(): {
            "name": str(row.name),
            "etf": str(row.etf),
            "test_issue": str(row.test_issue),
            "exchange": str(row.exchange),
        }
        for row in combined.itertuples(index=False)
    }
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(NASDAQ_SYMBOL_DIRECTORY_URLS),
        "securities": securities,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=cache_path.parent,
        prefix=cache_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, cache_path)
    return securities


def company_universe_mask(
    symbols: np.ndarray,
    security_master: dict[str, dict[str, object]],
    reference_symbol: str = REFERENCE_SYMBOL,
    keep_unclassified: bool = True,
) -> tuple[np.ndarray, dict[str, int], int]:
    """Return a company-only mask, preserving the benchmark reference.

    Symbols absent from today's security master are historical/unclassified,
    not necessarily non-company assets. Keeping them avoids introducing
    survivorship bias into historical simulations.
    """
    mask = np.zeros(len(symbols), dtype=bool)
    reasons: dict[str, int] = {}
    unclassified = 0
    for index, sample_id in enumerate(np.asarray(symbols, dtype=str)):
        if sample_id == reference_symbol:
            mask[index] = True
            continue
        record = security_master.get(_security_symbol(sample_id))
        if record is None:
            unclassified += 1
            reason = "missing from current security master"
            keep = bool(keep_unclassified)
        else:
            keep, reason = is_company_security(record)
        mask[index] = keep
        if not keep:
            reasons[reason] = reasons.get(reason, 0) + 1
    return mask, reasons, unclassified


def exchange_universe_mask(
    symbols: np.ndarray,
    security_master: Mapping[str, Mapping[str, object]],
    exchange: str,
    reference_symbol: str = REFERENCE_SYMBOL,
) -> np.ndarray:
    """Restrict candidates to a current primary listing venue, keeping SPY as benchmark."""
    exchange_codes = {"nasdaq": "Q"}
    if exchange not in exchange_codes:
        raise ValueError(f"unsupported exchange filter: {exchange}")
    wanted = exchange_codes[exchange]
    return np.asarray(
        [
            sample_id == reference_symbol
            or str(
                security_master.get(_security_symbol(sample_id), {}).get("exchange", "")
            )
            == wanted
            for sample_id in np.asarray(symbols, dtype=str)
        ],
        dtype=bool,
    )


def _parse_clock(value: str) -> int:
    """Parse a regular-session Eastern ``HH:MM`` clock."""
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("time must use HH:MM") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise argparse.ArgumentTypeError("time must use a valid 24-hour clock")
    result = hour * 60 + minute
    if not REGULAR_OPEN_MINUTE <= result <= REGULAR_CLOSE_MINUTE:
        raise argparse.ArgumentTypeError("time must be inside 09:30..16:00 Eastern")
    return result


def _parse_day(value: str) -> pd.Timestamp:
    """Parse a calendar date without accepting ambiguous timestamp formats."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error
    return pd.Timestamp(parsed.date())


def reference_session_calendar(
    reference_path: Path,
    session_close_minutes: Mapping[date, int] | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Derive New York trading sessions directly from the reference minute bars."""
    if not reference_path.exists():
        raise FileNotFoundError(f"reference minute bars do not exist: {reference_path}")
    source = np.load(reference_path, mmap_mode="r")
    validate_bar_columns(source, "1Min", str(reference_path))
    if not len(source):
        raise ValueError(f"invalid reference minute bars: {reference_path}")
    seconds = np.asarray(source[:, 0], dtype=np.int64)
    if (seconds < 0).any() or not np.all(seconds[:-1] < seconds[1:]):
        raise ValueError(
            f"reference timestamps must be non-negative and sorted: {reference_path}"
        )

    dates: list[pd.Timestamp] = []
    context_starts: list[int] = []
    first_day = int(seconds[0] // 86_400)
    last_day = int(seconds[-1] // 86_400)
    for day_offset in range(first_day, last_day + 1):
        session_date = (BAR_ORIGIN + timedelta(days=day_offset)).date()
        regular_open = datetime.combine(session_date, time(9, 30), tzinfo=EASTERN)
        close_minute = (session_close_minutes or {}).get(
            session_date, REGULAR_CLOSE_MINUTE
        )
        regular_close = datetime.combine(
            session_date,
            time(close_minute // 60, close_minute % 60),
            tzinfo=EASTERN,
        )
        open_second = int(
            (regular_open.astimezone(timezone.utc) - BAR_ORIGIN).total_seconds()
        )
        close_second = int(
            (regular_close.astimezone(timezone.utc) - BAR_ORIGIN).total_seconds()
        )
        start = int(np.searchsorted(seconds, open_second, side="left"))
        stop = int(np.searchsorted(seconds, close_second, side="right"))
        if stop - start < MIN_USABLE_SESSION_BARS:
            continue
        observed = seconds[start:stop]
        aligned = observed[observed % 60 == 0]
        if (
            aligned.size < MIN_USABLE_SESSION_BARS
            or int(aligned[0]) > open_second + 30 * 60
            or int(aligned[-1]) < close_second - 30 * 60
        ):
            # Holidays and sessions without a usable opening are omitted. A
            # shortened session stays because it can still be the morning exit for
            # the preceding position; its afternoon entry is disabled separately.
            continue
        context_open = datetime.combine(session_date, time(4), tzinfo=EASTERN)
        context_second = int(
            (context_open.astimezone(timezone.utc) - BAR_ORIGIN).total_seconds()
        )
        dates.append(pd.Timestamp(session_date))
        context_starts.append(context_second)
    if len(dates) < 2:
        raise ValueError(f"{reference_path} contains fewer than two complete sessions")
    return pd.DatetimeIndex(dates), np.asarray(context_starts, dtype=np.int64)


def simulation_symbols(minute_data_dir: Path, daily_data_dir: Path) -> np.ndarray:
    """Use symbols supported by both stores, plus the minute-only SPY benchmark."""
    minute_symbols = {path.stem for path in minute_data_dir.glob("*.npy")}
    daily_symbols = {path.stem for path in daily_data_dir.glob("*.npy")}
    if REFERENCE_SYMBOL not in minute_symbols:
        raise FileNotFoundError(
            f"reference minute bars do not exist: {minute_data_dir / f'{REFERENCE_SYMBOL}.npy'}"
        )
    symbols = (minute_symbols & daily_symbols) | {REFERENCE_SYMBOL}
    if len(symbols) < 2:
        raise ValueError(
            "the minute and daily bar stores have no common tradable symbols"
        )
    return np.asarray(sorted(symbols), dtype=str)


def _dataset_manifest(data_dir: Path, timeframe: str) -> dict[str, object]:
    """Validate the downloader manifest for one split-adjusted bar store."""
    manifest_path = data_dir / "_download_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bar dataset manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_bar_manifest(manifest, timeframe, str(manifest_path))
    return manifest


def _manifest_fingerprint(data_dir: Path, timeframe: str) -> dict[str, object]:
    manifest = _dataset_manifest(data_dir, timeframe)
    manifest_path = data_dir / "_download_manifest.json"
    stat = manifest_path.stat()
    return {
        "path": str(data_dir.resolve()),
        "directory_mtime_ns": int(data_dir.stat().st_mtime_ns),
        "manifest_size": int(stat.st_size),
        "manifest_mtime_ns": int(stat.st_mtime_ns),
        "completed_through": manifest.get("completed_through"),
        "updated_at": manifest.get("updated_at"),
    }


def load_daily_dollar_volume(
    path: Path,
    date_positions: Mapping[pd.Timestamp, int],
    date_count: int,
) -> np.ndarray:
    """Align split-adjusted daily turnover to the shared New York session calendar."""
    result = np.full(date_count, np.nan, dtype=np.float64)
    if not path.exists():
        return result
    daily = np.load(path, mmap_mode="r")
    validate_bar_columns(daily, "1Day", str(path))
    if len(daily) and not np.all(daily[:-1, 0] < daily[1:, 0]):
        raise ValueError(f"{path} timestamps must be strictly increasing")
    # Daily timestamps are midnight in New York, including the EST/EDT transition.
    daily_dates = (
        pd.to_datetime(
            np.asarray(daily[:, 0], dtype=np.int64),
            unit="s",
            origin="2010-01-01",
            utc=True,
        )
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    if daily_dates.duplicated().any():
        raise ValueError(f"{path} contains duplicate New York session dates")
    positions = np.asarray(
        [date_positions.get(stamp, -1) for stamp in daily_dates], dtype=np.int64
    )
    volume = np.asarray(daily[:, BAR_INDEX["volume"]], dtype=np.float64)
    vwap = np.asarray(daily[:, BAR_INDEX["vwap_mills"]], dtype=np.float64)
    close = np.asarray(daily[:, BAR_INDEX["close_mills"]], dtype=np.float64)
    price = np.where(np.isfinite(vwap) & (vwap > 0.0), vwap, close)
    valid = (
        (positions >= 0)
        & np.isfinite(price)
        & (price > 0.0)
        & np.isfinite(volume)
        & (volume > 0.0)
    )
    result[positions[valid]] = price[valid] * volume[valid] / 1000.0
    return result


@dataclass(frozen=True)
class HistoricalWindow:
    requested_start: pd.Timestamp
    end_date: pd.Timestamp
    dates: pd.DatetimeIndex
    context_sod: np.ndarray
    entry_session_mask: np.ndarray
    shortened_entries: set[date]


def historical_window(
    all_dates: pd.DatetimeIndex,
    all_context_sod: np.ndarray,
    auction_path: Path,
    *,
    since: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    months: int | None,
    ema_span: int,
    min_history_days: int,
    min_trading_days: int,
    entry_time: int,
) -> HistoricalWindow:
    """Choose the replay interval, preceding warm-up, and eligible entry sessions.

    The end date is the final exit session. Warm-up never relaxes the minimum
    history requirements when the local dataset starts too late.
    """
    requested_end = end_date if end_date is not None else pd.Timestamp(all_dates[-1])
    eligible_end = all_dates[all_dates <= requested_end]
    if eligible_end.empty:
        raise ValueError("end-date precedes the local dataset")
    final_exit = pd.Timestamp(eligible_end[-1])
    requested_start = (
        since
        if since is not None
        else final_exit - pd.DateOffset(months=int(months or 12))
    )
    first_entries = np.flatnonzero(all_dates >= requested_start)
    if not first_entries.size:
        raise ValueError("--since follows the local dataset")
    first_entry_index = int(first_entries[0])
    end_index = int(np.searchsorted(all_dates.to_numpy(), np.datetime64(final_exit)))
    if first_entry_index >= end_index:
        raise ValueError(
            "the requested interval must contain an entry and a later exit session"
        )
    warmup = max(3 * ema_span, min_history_days + 1, min_trading_days + 1)
    start_index = max(0, first_entry_index - warmup)
    dates = all_dates[start_index : end_index + 1]
    context_sod = all_context_sod[start_index : end_index + 1]
    requested_dates = [
        stamp.date() for stamp in dates if requested_start <= stamp < final_exit
    ]
    requested_closes = auction_close_minutes(auction_path, requested_dates)
    shortened = short_entry_dates(requested_closes, entry_time)
    return HistoricalWindow(
        requested_start,
        final_exit,
        dates,
        context_sod,
        np.asarray([stamp.date() not in shortened for stamp in dates], dtype=bool),
        shortened,
    )
