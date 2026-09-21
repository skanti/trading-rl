"""Pure preparation of causal unit-exposure history for the shared risk policy."""

import numpy as np
import pandas as pd

from .portfolio import basket_returns, select_strategy_basket
from .ranking import liquidity_features
from .strategies import (
    SPY_TREND_PRICE_SOURCES,
    LiquidityTrendVolConfig,
    LiquidityTrendVolPolicy,
)


class BasketHistory:
    """One causal ranking/selection implementation for backtests and live warmup."""

    def __init__(
        self,
        symbols,
        dollar_volume,
        *,
        top,
        ema_span,
        min_history_days,
        minimum_trading_days,
        liquidity_scheme,
        issuers=None,
        dedupe_share_classes=True,
        reference_symbol="SPY",
    ):
        self.symbols = np.asarray(symbols)
        self.scores, self.completed_days = liquidity_features(
            dollar_volume, liquidity_scheme, ema_span, min_history_days
        )
        if self.scores.shape[1] != len(self.symbols):
            raise ValueError("liquidity columns must match symbols")
        self.top = top
        self.minimum_trading_days = minimum_trading_days
        self.issuers = issuers
        self.dedupe_share_classes = dedupe_share_classes
        self.reference_symbol = reference_symbol
        self.unused_prices = np.full(len(symbols), np.nan)

    def select(
        self,
        row,
        *,
        entry_allowed=True,
        entry_prices=None,
        entry_staleness=None,
        entry_price_source="nbbo-ask",
        max_entry_staleness_minutes=1,
        exchange_mask=None,
    ):
        if not entry_allowed:
            return np.array([], dtype=int)
        return select_strategy_basket(
            self.symbols,
            self.scores[row],
            self.completed_days[row],
            self.unused_prices if entry_prices is None else entry_prices,
            self.unused_prices if entry_staleness is None else entry_staleness,
            top=self.top,
            minimum_trading_days=self.minimum_trading_days,
            max_entry_staleness_minutes=max_entry_staleness_minutes,
            entry_price_source=entry_price_source,
            issuers=self.issuers,
            dedupe_share_classes=self.dedupe_share_classes,
            reference_symbol=self.reference_symbol,
            execution_exchange_mask=exchange_mask,
        )


def historical_baskets(
    dates,
    symbols,
    dollar_volume,
    entry_allowed,
    exchange_mask,
    issuers,
    policy_config,
    *,
    top,
    ema_span,
    min_history_days,
    minimum_trading_days,
    liquidity_scheme,
):
    """Rank completed entry sessions; the last calendar row is today's exit only."""
    history = BasketHistory(
        symbols,
        dollar_volume,
        top=top,
        ema_span=ema_span,
        min_history_days=min_history_days,
        minimum_trading_days=minimum_trading_days,
        liquidity_scheme=liquidity_scheme,
        issuers=issuers,
    )
    rows = np.flatnonzero(dates[:-1] >= pd.Timestamp(policy_config.risk_history_start))
    rows = rows[-policy_config.volatility_window :]
    if len(rows) != policy_config.volatility_window:
        raise ValueError("live risk history needs a complete volatility window")
    baskets = [
        history.select(
            row, entry_allowed=entry_allowed[row], exchange_mask=exchange_mask[row]
        )
        for row in rows
    ]
    return rows, baskets


def unit_return(entry_prices, exit_prices):
    return basket_returns(
        entry_prices,
        exit_prices,
        slots=max(1, np.size(entry_prices)),
        require_complete=True,
    ).unscaled_return


def risk_signal(
    dates, observations, spy_marks, config: LiquidityTrendVolConfig,
    *, spy_trend_price_source="daily-close",
):
    """Observe all completed exits before asking for today's afternoon exposure."""
    if spy_trend_price_source not in SPY_TREND_PRICE_SOURCES:
        raise ValueError("unsupported SPY trend price source")
    dates = pd.DatetimeIndex(dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("risk calendar must be unique and increasing")
    if len(observations) != config.volatility_window:
        raise ValueError("risk history is incomplete")
    expected = dates[-config.volatility_window - 1 :]
    for index, observation in enumerate(observations):
        if (
            pd.Timestamp(observation["entry_date"]) != expected[index]
            or pd.Timestamp(observation["exit_date"]) != expected[index + 1]
        ):
            raise ValueError(
                "risk observations must cover every completed interval through today"
            )
    marks = np.asarray(spy_marks, dtype=float)
    if marks.shape != (len(dates),):
        raise ValueError("SPY marks must match the risk calendar")
    trailing = marks[-config.trend_window - 1 : -1]
    if (
        len(trailing) != config.trend_window
        or not np.isfinite(trailing).all()
        or (trailing <= 0).any()
    ):
        raise ValueError("SPY trend history is incomplete")
    policy = LiquidityTrendVolPolicy(config, marks)
    for observation in observations:
        policy.observe(observation["unscaled_return"])
    return {
        "strategy": "liquidity-trend-vol",
        "trade_date": str(dates[-1].date()),
        "completed_exit_date": str(dates[-1].date()),
        "spy_mark_date": str(dates[-2].date()),
        "spy_trend_price_source": spy_trend_price_source,
        "parameters": config.as_dict(),
        "target_exposure": policy.exposure(len(dates) - 1),
        "annualized_volatility": policy.annualized_volatility,
        "spy_previous_mark": float(trailing[-1]),
        "spy_trend_average": float(trailing.mean()),
        "strong_trend": bool(policy.strong_trend[-1]),
        "entry_price_source": "nbbo-ask",
        "exit_price_source": "opening-auction",
        "transaction_cost_bps": 0.0,
        "observations": observations,
    }
