"""Strict benchmark-price selection for live reconciliation."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..market_data.schema import validate_bar_columns
from .history import (
    BAR_ORIGIN,
    EASTERN,
    _official_opening_auctions,
    _security_symbol,
)
from .backtest import (
    MINUTE_PRICE_COLUMNS,
    load_scheduled_nbbo_prices,
)


class MissingBenchmarkData(ValueError):
    """A session cannot be compared without changing its requested benchmark."""


@dataclass(frozen=True)
class BenchmarkPrice:
    price: float
    raw_price: float
    staleness_minutes: float = 0.0
    exchange: str = ""


def _split_factors(
    path: Path, targets: Mapping[str, tuple[date, int]]
) -> dict[str, float]:
    if not path.exists():
        raise MissingBenchmarkData(f"split ledger does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "split_symbol",
            "split_ex_date",
            "split_old_rate",
            "split_new_rate",
            "symbol",
        }
        if not required.issubset(data.files):
            raise ValueError(
                f"{path} is missing the split ledger needed to compare bars with raw fills"
            )
        symbols = np.asarray(data["split_symbol"], dtype=str)
        dates = np.asarray(data["split_ex_date"], dtype="datetime64[D]")
        old = np.asarray(data["split_old_rate"], dtype=float)
        new = np.asarray(data["split_new_rate"], dtype=float)
        if not (symbols.shape == dates.shape == old.shape == new.shape):
            raise ValueError(f"{path} has inconsistent split ledger arrays")
        if (
            not np.isfinite(old).all()
            or not np.isfinite(new).all()
            or (old <= 0).any()
            or (new <= 0).any()
        ):
            raise ValueError(f"{path} has invalid split rates")
        covered = set(np.asarray(data["symbol"], dtype=str))
        result = {}
        for symbol, (day, _) in targets.items():
            security = _security_symbol(symbol)
            if security not in covered:
                raise MissingBenchmarkData(
                    f"{symbol}: split ledger coverage is unavailable in {path}"
                )
            selected = (symbols == security) & (dates > np.datetime64(day, "D"))
            result[symbol] = float(np.prod(new[selected] / old[selected]))
        return result


def load_benchmark_prices(
    source: str,
    path: Path | None,
    targets: Mapping[str, tuple[date, int]],
    *,
    max_staleness_minutes: float,
    split_path: Path,
) -> dict[str, BenchmarkPrice]:
    """Require the selected source/time for every symbol; never substitute marks."""
    if path is None or not path.exists():
        raise MissingBenchmarkData(f"{source} data does not exist: {path}")
    result: dict[str, BenchmarkPrice] = {}
    missing: list[str] = []
    if source in MINUTE_PRICE_COLUMNS:
        factors = _split_factors(split_path, targets)
        for symbol, (day, minute) in targets.items():
            bar_path = path / f"{_security_symbol(symbol)}.npy"
            if not bar_path.exists():
                missing.append(f"{symbol} {day}: minute bars are unavailable")
                continue
            bars = np.load(bar_path, mmap_mode="r", allow_pickle=False)
            validate_bar_columns(bars, "1Min", str(bar_path))
            target = datetime.combine(
                day, time(minute // 60, minute % 60), tzinfo=EASTERN
            )
            seconds = int((target.astimezone(UTC) - BAR_ORIGIN).total_seconds())
            indices = np.flatnonzero(bars[:, 0] == seconds)
            if len(indices) > 1:
                raise ValueError(
                    f"{bar_path} has duplicate bars at {target.isoformat()}"
                )
            if not len(indices):
                missing.append(
                    f"{symbol} {day} {minute // 60:02d}:{minute % 60:02d} ET: exact bar is unavailable"
                )
                continue
            price = float(bars[indices[0], MINUTE_PRICE_COLUMNS[source]]) / 1000.0
            if not np.isfinite(price) or price <= 0:
                missing.append(f"{symbol} {day}: {source} price is unavailable")
                continue
            result[symbol] = BenchmarkPrice(price, price * factors[symbol])
    else:
        for day in sorted({day for day, _ in targets.values()}):
            symbols = [
                symbol
                for symbol, (target_day, _) in targets.items()
                if target_day == day
            ]
            dates = pd.DatetimeIndex([pd.Timestamp(day)])
            if source == "opening-auction":
                rows = _official_opening_auctions(path, dates, np.asarray(symbols))
            elif source in {"nbbo-ask", "nbbo-bid"}:
                _, _, rows = load_scheduled_nbbo_prices(
                    path,
                    dates,
                    np.asarray(symbols),
                    source.split("-")[1],
                    None,
                )
            else:
                raise ValueError(f"unknown benchmark price source: {source}")
            by_symbol = {str(row.symbol): row for row in rows.itertuples(index=False)}
            for symbol in symbols:
                minute = targets[symbol][1]
                row = by_symbol.get(_security_symbol(symbol))
                label = f"{symbol} {day} {minute // 60:02d}:{minute % 60:02d} ET"
                if row is None:
                    missing.append(f"{label}: {source} is unavailable")
                    continue
                age = 0.0
                if source.startswith("nbbo-"):
                    stamp = row.target_timestamp.astimezone(EASTERN)
                    if stamp.hour * 60 + stamp.minute != minute:
                        missing.append(
                            f"{label}: file contains a {stamp:%H:%M} ET snapshot"
                        )
                        continue
                    age = float(row.staleness_minutes)
                    if age > min(1.0, max_staleness_minutes):
                        missing.append(
                            f"{label}: {source} quote is {age * 60:.2f}s old"
                        )
                        continue
                if not np.isfinite(row.raw_price) or row.raw_price <= 0:
                    raise ValueError(f"{path}: invalid raw {source} price for {symbol}")
                result[symbol] = BenchmarkPrice(
                    float(row.price), float(row.raw_price), age, str(row.exchange)
                )
    if missing:
        raise MissingBenchmarkData(f"{source}: " + "; ".join(missing))
    return result
