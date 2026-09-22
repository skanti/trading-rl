"""Backtester-only strategy plugins; never imported by live or reconciliation.

A plugin exports STRATEGY (StrategySpec). Its policy implements exposure(row) and
observe(unit_return); the common engine owns selection, execution and accounting.
"""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    "liquidity-fixed": StrategySpec("liquidity-fixed", "Liquidity: fixed exposure"),
    "liquidity-trend-vol": StrategySpec(
        "liquidity-trend-vol", "Liquidity: trend + volatility target",
        LiquidityTrendVolConfig, LiquidityTrendVolPolicy,
    ),
}
EXPERIMENTAL_MODULES = {
    "liquidity-regime-vol": "research.liquidity_regime",
    "liquidity-momentum-vol": "research.liquidity_momentum",
    "liquidity-momentum-blend": "research.liquidity_momentum_blend",
}
BUILTIN_STRATEGIES = (*_REGISTRY, *EXPERIMENTAL_MODULES)


def register_plugin(module_name: str):
    spec = getattr(importlib.import_module(module_name), "STRATEGY", None)
    if not isinstance(spec, StrategySpec) or not spec.experimental:
        raise ValueError("a backtest plugin must export an experimental StrategySpec as STRATEGY")
    if spec.config_type is None or spec.policy_type is None:
        raise ValueError("a plugin requires config_type and policy_type")
    if spec.name in EXPERIMENTAL_MODULES and module_name != EXPERIMENTAL_MODULES[spec.name]:
        raise ValueError(f"reserved strategy name: {spec.name}")
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(f"strategy already registered: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get_strategy(name: str) -> StrategySpec:
    if name in EXPERIMENTAL_MODULES and name not in _REGISTRY:
        register_plugin(EXPERIMENTAL_MODULES[name])
    if name not in _REGISTRY:
        raise ValueError(f"unknown backtest strategy {name!r}; available: {', '.join(BUILTIN_STRATEGIES)}; use --strategy-plugin for others")
    return _REGISTRY[name]
