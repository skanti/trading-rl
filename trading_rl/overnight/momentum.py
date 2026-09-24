"""Pure momentum allocation and risk policies shared by live and simulation."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .strategies import LiquidityTrendVolConfig, LiquidityTrendVolPolicy


def momentum_weights(momentum, count):
    """Equal-weight momentum leaders; ties preserve causal liquidity order."""
    values = np.asarray(momentum, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("incomplete allocation momentum")
    weights = np.zeros(len(values))
    if len(values):
        count = min(count, len(values))
        weights[np.argsort(-values, kind="stable")[:count]] = 1 / count
    return weights


def allocation_snapshot(
    names, start_closes, end_closes, history_dates, trade_date, config
):
    """Archive only completed daily inputs needed to reproduce the allocation."""
    starts, ends = (
        np.asarray(start_closes, dtype=float),
        np.asarray(end_closes, dtype=float),
    )
    if (
        starts.shape != (len(names),)
        or ends.shape != starts.shape
        or not np.isfinite([starts, ends]).all()
        or (starts <= 0).any()
        or (ends <= 0).any()
    ):
        raise ValueError("incomplete momentum daily closes")
    values = ends / starts - 1
    weights = momentum_weights(values, config.allocation_count)
    return {
        "version": 1,
        "trade_date": str(trade_date),
        "history_dates": [str(pd.Timestamp(day).date()) for day in history_dates],
        "candidates": [
            {
                "symbol": str(name),
                "start_close": float(start),
                "end_close": float(end),
                "momentum": float(value),
            }
            for name, start, end, value in zip(names, starts, ends, values, strict=True)
        ],
        "symbols": [
            str(name) for name, weight in zip(names, weights, strict=True) if weight > 0
        ],
    }


def replay_allocation(snapshot, config, top, trade_date):
    if snapshot["version"] != 1 or snapshot["trade_date"] != str(trade_date):
        raise ValueError("momentum allocation date/version mismatch")
    days = pd.DatetimeIndex(snapshot["history_dates"])
    if (
        len(days) != config.allocation_window + 1
        or days.has_duplicates
        or not days.is_monotonic_increasing
        or days[-1] >= pd.Timestamp(trade_date)
    ):
        raise ValueError("momentum allocation needs strictly completed session history")
    candidates = snapshot["candidates"]
    names = [row["symbol"] for row in candidates]
    if len(names) != top or len(set(names)) != top:
        raise ValueError("momentum shortlist must contain top distinct symbols")
    calculated = allocation_snapshot(
        names,
        [row["start_close"] for row in candidates],
        [row["end_close"] for row in candidates],
        days,
        trade_date,
        config,
    )
    if calculated["symbols"] != snapshot["symbols"] or not np.allclose(
        [row["momentum"] for row in candidates],
        [row["momentum"] for row in calculated["candidates"]],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("momentum allocation differs from saved daily prices")
    return calculated["symbols"]


def historical_allocation(dates, symbols, daily_closes, row, shortlist, config):
    start = row - config.allocation_window - 1
    if start < 0:
        raise ValueError("insufficient completed momentum history")
    return allocation_snapshot(
        symbols[shortlist],
        daily_closes[start, shortlist],
        daily_closes[row - 1, shortlist],
        dates[start:row],
        dates[row].date(),
        config,
    )


@dataclass(frozen=True)
class LiquidityRegimeConfig(LiquidityTrendVolConfig):
    trend_window: int = 150
    volatility_window: int = 10
    weak_trend_multiplier: float = 0.0
    trend_buffer: float = 0.0
    momentum_window: int = 0
    excluded_entry_weekday: int | None = None

    def __post_init__(self):
        # Reuse promoted-policy validation; only the cash regime extends its range.
        base = {
            field.name: getattr(self, field.name)
            for field in fields(LiquidityTrendVolConfig)
        }
        base["weak_trend_multiplier"] = 1.0
        LiquidityTrendVolConfig(**base)
        for name, low, high in (
            ("weak_trend_multiplier", 0, 1),
            ("trend_buffer", -0.1, 0.1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if type(self.momentum_window) is not int or self.momentum_window < 0:
            raise ValueError("momentum_window must be a non-negative integer")
        if self.excluded_entry_weekday is not None and (
            type(self.excluded_entry_weekday) is not int
            or not 0 <= self.excluded_entry_weekday <= 4
        ):
            raise ValueError(
                "excluded_entry_weekday must be null or an integer from 0 (Monday) to 4 (Friday)"
            )


class LiquidityRegimePolicy(LiquidityTrendVolPolicy):
    def prepare(self, context):
        self.weekdays = context.dates.dayofweek.to_numpy()

    def exposure(self, row):
        if (
            self.config.excluded_entry_weekday is not None
            and self.weekdays[row] == self.config.excluded_entry_weekday
        ):
            return 0.0
        return super().exposure(row)

    def __init__(self, config, spy_marks):
        super().__init__(config, spy_marks)
        marks = pd.Series(spy_marks, dtype=float)
        previous = marks.shift(1)
        average = marks.rolling(config.trend_window).mean().shift(1)
        strong = previous >= average * (1 + config.trend_buffer)
        if config.momentum_window:
            strong &= previous >= marks.shift(config.momentum_window + 1)
        self.strong_trend = strong.to_numpy()


@dataclass(frozen=True)
class LiquidityMomentumConfig(LiquidityRegimeConfig):
    volatility_target: float = 0.5
    weak_trend_multiplier: float = 0.25
    allocation_window: int = 60
    allocation_count: int = 0  # zero means rank tilt across every selected member
    allocation_windows: tuple[int, ...] | None = None

    def __post_init__(self):
        super().__post_init__()
        if type(self.allocation_window) is not int or self.allocation_window < 1:
            raise ValueError("allocation_window must be a positive integer")
        if type(self.allocation_count) is not int or self.allocation_count < 0:
            raise ValueError("allocation_count must be a non-negative integer")
        if self.allocation_windows is not None:
            values = self.allocation_windows
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(type(value) is not int or value < 1 for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    "allocation_windows must be a nonempty list of distinct positive integers"
                )
            object.__setattr__(self, "allocation_windows", tuple(values))


class LiquidityMomentumPolicy(LiquidityRegimePolicy):
    def prepare(self, context):
        super().prepare(context)
        if context.daily_closes is None or context.daily_closes.shape != (
            len(context.dates),
            len(context.symbols),
        ):
            raise ValueError("momentum allocation requires date-by-symbol daily closes")
        prices = pd.DataFrame(context.daily_closes)
        prices = prices.where(np.isfinite(prices) & (prices > 0))
        windows = self.config.allocation_windows or (self.config.allocation_window,)
        self.momentums = [
            (prices.shift(1) / prices.shift(window + 1) - 1).to_numpy()
            for window in windows
        ]

    def weights(self, row, selected):
        if len(selected) == 0:
            return np.empty(0)
        return np.mean(
            [
                self._weights(momentum[row, selected], len(selected))
                for momentum in self.momentums
            ],
            axis=0,
        )

    def _weights(self, momentum, size):
        if not np.isfinite(momentum).all():
            raise ValueError("incomplete allocation momentum")
        if self.config.allocation_count:
            return momentum_weights(momentum, self.config.allocation_count)
        ranks = np.argsort(np.argsort(momentum, kind="stable"), kind="stable") + 1
        return ranks / ranks.sum()


@dataclass(frozen=True)
class LiquidityMomentumFocusConfig(LiquidityMomentumConfig):
    volatility_target: float = 0.35
    volatility_window: int = 20
    trend_window: int = 100
    weak_trend_multiplier: float = 0.1
    allocation_window: int = 10
    allocation_count: int = 3
    allocation_windows: tuple[int, ...] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.allocation_windows is not None or self.allocation_count < 1:
            raise ValueError(
                "momentum focus requires one allocation_window and a positive allocation_count"
            )
        if (
            self.trend_buffer != 0
            or self.momentum_window != 0
            or self.excluded_entry_weekday is not None
        ):
            raise ValueError(
                "momentum focus uses only the SPY moving-average trend filter"
            )
