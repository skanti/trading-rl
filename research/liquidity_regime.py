"""Backtest registration for the experimental liquidity regime family."""

from trading_rl.overnight.backtest_strategies import StrategySpec
from trading_rl.overnight.momentum import LiquidityRegimeConfig, LiquidityRegimePolicy

STRATEGY = StrategySpec(
    "liquidity-regime-vol", "Liquidity: experimental regime + volatility target",
    LiquidityRegimeConfig, LiquidityRegimePolicy, experimental=True,
    default_top=6, default_ema_span=20,
)
