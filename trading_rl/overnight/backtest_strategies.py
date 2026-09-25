"""Packaged backtest strategies and optional external plugin registration.

A plugin exports STRATEGY (StrategySpec). Its policy implements exposure(row) and
observe(unit_return); the common engine owns selection, execution and accounting.
Live and reconciliation import shared policies directly, never this registry.
"""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .momentum import (
    LiquidityMomentumBlendConfig,
    LiquidityMomentumConfig,
    LiquidityMomentumFocusConfig,
    LiquidityMomentumPolicy,
    LiquidityRegimeConfig,
    LiquidityRegimePolicy,
)
from .strategies import LiquidityTrendVolConfig, LiquidityTrendVolPolicy


@dataclass(frozen=True)
class PolicyContext:
    dates: pd.DatetimeIndex
    symbols: np.ndarray
    daily_closes: np.ndarray | None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    label: str
    config_type: type | None = None
    policy_type: type | None = None
    experimental: bool = False
    default_top: int = 12
    default_ema_span: int = 10
    requires_daily_closes: bool = False

    def load_config(self, path: Path | None = None):
        if self.config_type is None:
            if path is not None:
                raise ValueError(f"{self.name} has no strategy configuration")
            return None
        values = {} if path is None else json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("strategy config must be a JSON object")  # noqa: TRY004 - invalid serialized input
        try:
            return self.config_type(**values)
        except TypeError as error:
            raise ValueError(f"invalid {self.name} config: {error}") from error


_REGISTRY = {
    "liquidity-momentum-focus": StrategySpec(
        "liquidity-momentum-focus", "Liquidity: momentum focus + volatility target",
        LiquidityMomentumFocusConfig, LiquidityMomentumPolicy, requires_daily_closes=True,
    ),
    "liquidity-fixed": StrategySpec("liquidity-fixed", "Liquidity: fixed exposure"),
    "liquidity-trend-vol": StrategySpec(
        "liquidity-trend-vol", "Liquidity: trend + volatility target",
        LiquidityTrendVolConfig, LiquidityTrendVolPolicy,
    ),
    "liquidity-regime-vol": StrategySpec(
        "liquidity-regime-vol", "Liquidity: experimental regime + volatility target",
        LiquidityRegimeConfig, LiquidityRegimePolicy, experimental=True,
        default_top=6, default_ema_span=20,
    ),
    "liquidity-momentum-vol": StrategySpec(
        "liquidity-momentum-vol", "Liquidity: experimental momentum allocation + volatility target",
        LiquidityMomentumConfig, LiquidityMomentumPolicy, experimental=True,
        requires_daily_closes=True,
    ),
    "liquidity-momentum-blend": StrategySpec(
        "liquidity-momentum-blend", "Liquidity: experimental momentum blend + volatility target",
        LiquidityMomentumBlendConfig, LiquidityMomentumPolicy, experimental=True,
        requires_daily_closes=True,
    ),
}
BUILTIN_STRATEGIES = tuple(_REGISTRY)


def register_plugin(module_name: str):
    spec = getattr(importlib.import_module(module_name), "STRATEGY", None)
    if not isinstance(spec, StrategySpec) or not spec.experimental:
        raise ValueError("a backtest plugin must export an experimental StrategySpec as STRATEGY")
    if spec.config_type is None or spec.policy_type is None:
        raise ValueError("a plugin requires config_type and policy_type")
    existing = _REGISTRY.get(spec.name)
    if spec.name in BUILTIN_STRATEGIES:
        raise ValueError(f"reserved strategy name: {spec.name}; already registered")
    if existing is not None and existing != spec:
        raise ValueError(f"strategy already registered: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get_strategy(name: str) -> StrategySpec:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backtest strategy {name!r}; available: {', '.join(BUILTIN_STRATEGIES)}; use --strategy-plugin for others")
    return _REGISTRY[name]
