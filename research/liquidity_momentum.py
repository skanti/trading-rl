"""Backtest registration for the experimental liquidity momentum family."""

from trading_rl.overnight.backtest_strategies import StrategySpec
from trading_rl.overnight.momentum import (
    LiquidityMomentumConfig,
    LiquidityMomentumPolicy,
)

STRATEGY = StrategySpec(
    "liquidity-momentum-vol", "Liquidity: experimental momentum allocation + volatility target",
    LiquidityMomentumConfig, LiquidityMomentumPolicy, experimental=True,
    requires_daily_closes=True,
)
