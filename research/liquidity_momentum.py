"""Experimental momentum allocation inside the causal liquidity shortlist."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.liquidity_regime import LiquidityRegimeConfig, LiquidityRegimePolicy
from trading_rl.overnight.backtest_strategies import StrategySpec


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
            if not isinstance(values, (list, tuple)) or not values or any(type(value) is not int or value < 1 for value in values) or len(set(values)) != len(values):
                raise ValueError("allocation_windows must be a nonempty list of distinct positive integers")
            object.__setattr__(self, "allocation_windows", tuple(values))


class LiquidityMomentumPolicy(LiquidityRegimePolicy):
    def prepare(self, context):
        super().prepare(context)
        if context.daily_closes is None or context.daily_closes.shape != (len(context.dates), len(context.symbols)):
            raise ValueError("momentum allocation requires date-by-symbol daily closes")
        prices = pd.DataFrame(context.daily_closes)
        prices = prices.where(np.isfinite(prices) & (prices > 0))
        windows = self.config.allocation_windows or (self.config.allocation_window,)
        self.momentums = [(prices.shift(1) / prices.shift(window + 1) - 1).to_numpy() for window in windows]

    def weights(self, row, selected):
        if len(selected) == 0:
            return np.empty(0)
        return np.mean([self._weights(momentum[row, selected], len(selected)) for momentum in self.momentums], axis=0)

    def _weights(self, momentum, size):
        if not np.isfinite(momentum).all():
            raise ValueError("incomplete allocation momentum")
        if self.config.allocation_count:
            count = min(self.config.allocation_count, size)
            weights = np.zeros(size)
            weights[np.argsort(-momentum, kind="stable")[:count]] = 1 / count
            return weights
        ranks = np.argsort(np.argsort(momentum, kind="stable"), kind="stable") + 1
        return ranks / ranks.sum()


STRATEGY = StrategySpec(
    "liquidity-momentum-vol", "Liquidity: experimental momentum allocation + volatility target",
    LiquidityMomentumConfig, LiquidityMomentumPolicy, experimental=True,
    requires_daily_closes=True,
)
