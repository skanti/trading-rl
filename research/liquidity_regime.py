"""Experimental long/cash regime filter with optional momentum confirmation."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest_strategies import StrategySpec
from trading_rl.overnight.strategies import (
    LiquidityTrendVolConfig,
    LiquidityTrendVolPolicy,
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
        base = {field.name: getattr(self, field.name) for field in fields(LiquidityTrendVolConfig)}
        base["weak_trend_multiplier"] = 1.0
        LiquidityTrendVolConfig(**base)
        for name, low, high in (("weak_trend_multiplier", 0, 1), ("trend_buffer", -0.1, 0.1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if type(self.momentum_window) is not int or self.momentum_window < 0:
            raise ValueError("momentum_window must be a non-negative integer")
        if self.excluded_entry_weekday is not None and (type(self.excluded_entry_weekday) is not int or not 0 <= self.excluded_entry_weekday <= 4):
            raise ValueError("excluded_entry_weekday must be null or an integer from 0 (Monday) to 4 (Friday)")


class LiquidityRegimePolicy(LiquidityTrendVolPolicy):
    def prepare(self, context):
        self.weekdays = context.dates.dayofweek.to_numpy()

    def exposure(self, row):
        if self.config.excluded_entry_weekday is not None and self.weekdays[row] == self.config.excluded_entry_weekday:
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


STRATEGY = StrategySpec(
    "liquidity-regime-vol", "Liquidity: experimental regime + volatility target",
    LiquidityRegimeConfig, LiquidityRegimePolicy, experimental=True,
    default_top=6, default_ema_span=20,
)
