"""Optional SPY reference, independent of strategy execution and account returns."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from .metrics import EASTERN, EquityPoint


def spy_buy_and_hold(
    series: Sequence[EquityPoint],
    bars: Sequence[Mapping[str, Any]],
    completed_through: date,
) -> dict[str, Any]:
    """Normalize adjusted daily closes to the strategy's first date and capital.

    Missing dates stay missing. In particular, never move the investment's start
    to a later available bar or fabricate a live price from yesterday's close.
    """
    result: dict[str, Any] = {
        "symbol": "SPY",
        "basis": "adjusted_daily_close",
        "status": "unavailable",
        "as_of": None,
        "points": [],
    }
    if not series or not math.isfinite(series[0].equity) or series[0].equity <= 0:
        return result
    start = series[0].day
    if start > completed_through:
        result["status"] = "pending"
        return result
    prices: dict[date, float] = {}
    for bar in bars:
        timestamp = datetime.fromisoformat(str(bar["t"]))
        if timestamp.tzinfo is None:
            raise ValueError("SPY bar timestamp must include a timezone")
        day = timestamp.astimezone(EASTERN).date()
        if day < start or day > completed_through:
            continue
        close = float(bar["c"])
        if not math.isfinite(close) or close <= 0:
            continue
        if day in prices:
            raise ValueError(f"duplicate SPY daily bar for {day}")
        prices[day] = close
    if start not in prices:
        return result
    capital = series[0].equity
    for day, close in sorted(prices.items()):
        change = close / prices[start] - 1.0
        result["points"].append({
            "day": day.isoformat(),
            "equity": capital * (1.0 + change),
            "profit_loss": capital * change,
            "profit_loss_pct": change,
        })
    result["status"] = "available"
    result["as_of"] = max(prices).isoformat()
    return result
