"""Explicit price loading shared by live signals, reconciliation and simulation.

Importing this module never loads datasets or contacts a data provider.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..market_data.schema import BAR_INDEX
from .history import _official_opening_auctions, _security_symbol
from .price_archives import open_price_archive

MINUTE_PRICE_COLUMNS = {
    f"minute-{field}": BAR_INDEX[f"{field}_mills"]
    for field in ("open", "high", "low", "close", "vwap")
}
ENTRY_PRICE_SOURCES = (*MINUTE_PRICE_COLUMNS, "nbbo-ask")
EXIT_PRICE_SOURCES = (*MINUTE_PRICE_COLUMNS, "opening-auction", "nbbo-bid")

DEFAULT_TRANSACTION_COST_BPS = 1.0

DEFAULT_NBBO_PATH = "/data/ppv1/updates/alpaca_nbbo_1545_2022-01-01.npz"
DEFAULT_EXIT_NBBO_PATH = "/data/ppv1/updates/alpaca_nbbo_0935_2022-01-01.npz"


def resolve_transaction_cost_bps(
    value: float | None,
    entry_price_source: str,
    exit_price_source: str,
) -> float:
    """Default additional costs by source pair, preserving explicit overrides."""
    if value is None:
        value = (
            0.0
            if (entry_price_source, exit_price_source)
            == ("nbbo-ask", "opening-auction")
            else DEFAULT_TRANSACTION_COST_BPS
        )
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    return float(value)


def load_opening_auction_prices(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    *,
    official: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build a date-by-symbol primary-opening matrix from adjusted NPZ prices."""
    if official is None:
        official = _official_opening_auctions(path, dates, symbols)
    prices = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    rows = pd.Index(dates).get_indexer(pd.DatetimeIndex(official["date"]))
    symbol_index = pd.Index([_security_symbol(sample_id) for sample_id in symbols])
    columns = symbol_index.get_indexer(official["symbol"].astype(str).str.upper())
    valid = (rows >= 0) & (columns >= 0)
    prices[rows[valid], columns[valid]] = official["price"].to_numpy(dtype=np.float64)[
        valid
    ]
    return prices


def load_scheduled_nbbo_prices(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    side: str,
    target_minute: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one quote side at the configured clock; missing rows remain unavailable."""
    if side not in {"bid", "ask"}:
        raise ValueError("NBBO side must be bid or ask")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"NBBO data must use the split-adjusted NPZ format: {path}")
    with open_price_archive(path) as data:
        required = {
            "symbol",
            "date",
            "target_timestamp",
            "timestamp",
            f"{side}_price",
            f"raw_{side}_price",
            f"{side}_exchange",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing NBBO arrays: {sorted(missing)}")
        if "split_adjusted" not in data.files or not bool(
            data["split_adjusted"].item()
        ):
            raise ValueError(f"{path} does not contain pre-adjusted NBBO prices")
        symbol_values = data.derived(
            "upper_symbols", lambda: np.char.upper(np.asarray(data["symbol"]).astype(str, copy=False))
        )
        date_values = np.asarray(data["date"], dtype="datetime64[D]")
        # SIP quotes mix whole seconds and fractional precision up to nanoseconds.
        # Infer neither column's format from its first row: all are ISO 8601.
        targets = data.derived(
            "parsed_targets", lambda: pd.to_datetime(
                np.asarray(data["target_timestamp"]).astype(str, copy=False),
                format="ISO8601", utc=True,
            ),
        )
        stamps = data.derived(
            "parsed_timestamps", lambda: pd.to_datetime(
                np.asarray(data["timestamp"]).astype(str, copy=False),
                format="ISO8601", utc=True,
            ),
        )
        if targets.hasnans or stamps.hasnans:
            raise ValueError(f"{path} contains missing NBBO timestamps")
        quote_prices = np.asarray(data[f"{side}_price"], dtype=np.float64)
        raw_prices = np.asarray(data[f"raw_{side}_price"], dtype=np.float64)
        exchanges = np.asarray(data[f"{side}_exchange"]).astype(str, copy=False)
        wanted_dates = np.asarray(pd.DatetimeIndex(dates), dtype="datetime64[D]")
        wanted_symbols = np.asarray(
            [_security_symbol(sample_id) for sample_id in symbols], dtype=str
        )
        selected = (
            np.isin(date_values, wanted_dates)
            & np.isin(symbol_values, wanted_symbols)
            & np.isfinite(quote_prices)
            & (quote_prices > 0.0)
        )
        local_targets = targets[selected].tz_convert("America/New_York")
        if (
            (
                target_minute is not None
                and (
                    (local_targets.hour * 60 + local_targets.minute) != target_minute
                ).any()
            )
            or (local_targets.second != 0).any()
            or (local_targets.microsecond != 0).any()
            or (local_targets.nanosecond != 0).any()
            or not np.array_equal(
                local_targets.tz_localize(None)
                .normalize()
                .to_numpy(dtype="datetime64[D]"),
                date_values[selected],
            )
        ):
            raise ValueError(
                f"{path} NBBO target timestamps do not match the configured time/date"
            )
        if (stamps[selected] > targets[selected]).any():
            raise ValueError(f"{path} contains a post-target NBBO quote")
        rows = pd.DataFrame(
            {
                "symbol": symbol_values[selected],
                "date": pd.to_datetime(date_values[selected]),
                "price": quote_prices[selected],
                "raw_price": raw_prices[selected],
                "staleness_minutes": (
                    (targets[selected] - stamps[selected]).total_seconds() / 60.0
                ),
                "timestamp": stamps[selected],
                "target_timestamp": targets[selected],
                "exchange": exchanges[selected],
            }
        )
    if rows.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{path} contains duplicate symbol-date NBBO snapshots")
    prices = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    staleness = np.full_like(prices, np.inf)
    row_indices = pd.Index(dates).get_indexer(pd.DatetimeIndex(rows["date"]))
    symbol_index = pd.Index([_security_symbol(sample_id) for sample_id in symbols])
    columns = symbol_index.get_indexer(rows["symbol"])
    valid = (row_indices >= 0) & (columns >= 0)
    prices[row_indices[valid], columns[valid]] = rows["price"].to_numpy()[valid]
    staleness[row_indices[valid], columns[valid]] = rows[
        "staleness_minutes"
    ].to_numpy()[valid]
    return prices, staleness, rows


def load_scheduled_nbbo_asks(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    target_minute: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return load_scheduled_nbbo_prices(path, dates, symbols, "ask", target_minute)
