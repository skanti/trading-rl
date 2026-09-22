"""Experimental equal blend of one- and two-week stock momentum baskets.

Selected using reused 2023-2026 history. This is an in-sample research preset,
not a live-approved policy or an independently validated forward return claim.
"""

from dataclasses import dataclass

from research.liquidity_momentum import LiquidityMomentumConfig, LiquidityMomentumPolicy
from trading_rl.overnight.backtest_strategies import StrategySpec


@dataclass(frozen=True)
class LiquidityMomentumBlendConfig(LiquidityMomentumConfig):
    volatility_target: float = 0.35
    volatility_window: int = 20
    trend_window: int = 100
    weak_trend_multiplier: float = 0.0
    allocation_window: int = 10
    allocation_count: int = 4
    allocation_windows: tuple[int, ...] | None = (5, 10)


STRATEGY = StrategySpec(
    "liquidity-momentum-blend", "Liquidity: experimental momentum blend + volatility target",
    LiquidityMomentumBlendConfig, LiquidityMomentumPolicy, experimental=True,
    requires_daily_closes=True,
)
