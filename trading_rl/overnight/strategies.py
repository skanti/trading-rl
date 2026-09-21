"""Named, causal exposure policies shared by overnight simulation and live execution."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Shared overnight cap; intraday buying power can be higher.
MAX_OVERNIGHT_EXPOSURE = 2.0

STRATEGY_LABELS = {
    "liquidity-fixed": "Liquidity: fixed exposure",
    "liquidity-trend-vol": "Liquidity: trend + volatility target",
}
STRATEGIES = tuple(STRATEGY_LABELS)


@dataclass(frozen=True)
class LiquidityTrendVolConfig:
    volatility_target: float = 0.35
    max_exposure: float = MAX_OVERNIGHT_EXPOSURE
    volatility_window: int = 20
    warmup_exposure: float = 1.0
    trend_window: int = 100
    weak_trend_multiplier: float = 0.25
    # The promoted research started observing basket returns at this inception.
    # Ranking and SPY trend still use all preceding market history.
    risk_history_start: str = "2023-01-01"

    def __post_init__(self):
        for name in ("volatility_window", "trend_window"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
        for name in (
            "volatility_target",
            "max_exposure",
            "warmup_exposure",
            "weak_trend_multiplier",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if (
            self.volatility_target <= 0
            or not 0 < self.max_exposure <= MAX_OVERNIGHT_EXPOSURE
        ):
            raise ValueError(
                "volatility_target must be positive and max_exposure must be in (0, 2]"
            )
        if not 0 < self.warmup_exposure <= self.max_exposure:
            raise ValueError("warmup_exposure must be in (0, max_exposure]")
        if not 0 < self.weak_trend_multiplier <= 1:
            raise ValueError("weak_trend_multiplier must be in (0, 1]")
        try:
            stamp = pd.Timestamp(self.risk_history_start)
            if pd.isna(stamp) or stamp.strftime("%Y-%m-%d") != self.risk_history_start:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError("risk_history_start must be YYYY-MM-DD") from None

    def as_dict(self):
        return asdict(self)


def load_trend_vol_config(path: Path | None) -> LiquidityTrendVolConfig:
    if path is None:
        return LiquidityTrendVolConfig()
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("strategy config must be a JSON object")  # noqa: TRY004 - invalid serialized content
    try:
        return LiquidityTrendVolConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid liquidity-trend-vol config: {error}") from error


class LiquidityTrendVolPolicy:
    """Decide exposure before observing the current basket's overnight return."""

    def __init__(self, config: LiquidityTrendVolConfig, spy_marks: np.ndarray):
        self.config = config
        marks = pd.Series(spy_marks, dtype=float)
        previous = marks.shift(1)
        average = marks.rolling(config.trend_window).mean().shift(1)
        # Missing trend history takes the research's conservative regime.
        self.strong_trend = (previous >= average).to_numpy()
        self.returns: list[float] = []

    @property
    def annualized_volatility(self) -> float | None:
        if len(self.returns) < self.config.volatility_window:
            return None
        return float(
            np.std(self.returns[-self.config.volatility_window :], ddof=1)
            * np.sqrt(252)
        )

    def exposure(self, row: int) -> float:
        config = self.config
        exposure = config.warmup_exposure
        volatility = self.annualized_volatility
        if volatility is not None and volatility > 0:
            exposure = config.volatility_target / volatility
        exposure = min(config.max_exposure, exposure)
        return float(
            exposure * (1.0 if self.strong_trend[row] else config.weak_trend_multiplier)
        )

    def observe(self, unscaled_net_return: float) -> None:
        if not np.isfinite(unscaled_net_return):
            raise ValueError(
                "liquidity-trend-vol risk history requires finite basket returns"
            )
        self.returns.append(float(unscaled_net_return))
