"""Causal overnight liquidity strategies.

Rank stocks using only completed sessions, buy an equal-weight basket at the
configured afternoon entry, and liquidate it the following morning. Raw daily
liquidity and point-in-time prices are cached so changing the basket size or
EMA span does not rescan the minute files when the required warm-up range is
unchanged. Alpaca-style rankings use cumulative same-session share volume or
trade count through a pre-entry ranking time.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ..market_data.calendar import auction_close_minutes
from ..market_data.schema import validate_bar_columns

# Re-export historical helper names for compatibility with existing callers.
from .execution_prices import (  # noqa: F401
    DEFAULT_EXIT_NBBO_PATH,
    DEFAULT_NBBO_PATH,
    DEFAULT_TRANSACTION_COST_BPS,
    ENTRY_PRICE_SOURCES,
    EXIT_PRICE_SOURCES,
    MINUTE_PRICE_COLUMNS,
    load_opening_auction_prices,
    load_scheduled_nbbo_asks,
    load_scheduled_nbbo_prices,
    resolve_transaction_cost_bps,
)
from .history import (
    DEFAULT_AUCTIONS_PATH,
    DEFAULT_DAILY_DATA_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_SECURITY_MASTER_CACHE,
    EXTENDED_OPEN_MINUTE,
    REFERENCE_SYMBOL,
    _manifest_fingerprint,
    _official_opening_auctions,
    _parse_clock,
    _parse_day,
    _security_symbol,
    company_universe_mask,
    exchange_universe_mask,
    historical_window,
    load_daily_closes,
    load_daily_dollar_volume,
    load_nasdaq_security_master,
    load_primary_auction_exchange_mask,
    reference_session_calendar,
    simulation_symbols,
)
from .portfolio import (
    basket_quantities,
    basket_returns,
    equal_notional,
    select_strategy_basket,  # noqa: F401 - compatibility export
)
from .ranking import build_issuer_map
from .risk_history import BasketHistory
from .strategies import (
    MAX_OVERNIGHT_EXPOSURE as MAX_OVERNIGHT_LEVERAGE,
)
from .strategies import (
    SPY_TREND_PRICE_SOURCES,
    STRATEGIES,
    STRATEGY_LABELS,
    LiquidityTrendVolConfig,
    LiquidityTrendVolPolicy,
    load_trend_vol_config,
)

LOGGER = logging.getLogger(__name__)

# Base Reg T maintenance is 25%; brokers raise it on concentrated, volatile books, and
# Alpaca's own requirement on this basket sits near 32%. Used only for reporting.
MAINTENANCE_MARGIN = 0.30
# Alpaca accrues margin interest on a 360-day year, per calendar day.
MARGIN_INTEREST_DIVISOR = 360.0


def _parse_percentage(value: str) -> float:
    """Parse an interest rate written as a percentage into a fraction.

    Accepts "5.25%" or a bare "5.25", both meaning 5.25% -> 0.0525.

    A bare value below half a percent is rejected rather than accepted. Margin rates
    are never that low, so such a value is almost certainly the old fractional form
    (0.0525), and silently reading it as 0.0525% would understate the borrow cost a
    hundredfold without any visible sign.
    """
    text = value.strip().rstrip("%").strip()
    try:
        percent = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a percentage; write it like 5.25% or 5.25"
        ) from None
    if percent < 0.0:
        raise argparse.ArgumentTypeError("the rate must be non-negative")
    if 0.0 < percent < 0.5:
        raise argparse.ArgumentTypeError(
            f"{value!r} looks like a fraction, not a percentage. This flag takes "
            f"percent, so write {percent * 100.0:g}% if you meant that rate"
        )
    if percent > 100.0:
        raise argparse.ArgumentTypeError(f"{value!r} exceeds 100%")
    return percent / 100.0


def _cache_metadata(
    minute_data_dir: Path,
    daily_data_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    entry_minute: int,
    exit_minute: int,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "minute-open",
) -> dict[str, object]:
    return {
        "version": 8,
        "entry_price_source": entry_price_source,
        "exit_price_source": exit_price_source,
        "minute_price_timing": "bar-start-labelled execution window",
        "minute_data": _manifest_fingerprint(minute_data_dir, "1Min"),
        "daily_data": _manifest_fingerprint(daily_data_dir, "1Day"),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "entry_minute": int(entry_minute),
        "exit_minute": int(exit_minute),
    }


def _symbol_daily_arrays(
    sample_id: str,
    date_positions: dict[pd.Timestamp, int],
    context_sod: np.ndarray,
    minute_data_dir: Path,
    daily_data_dir: Path,
    entry_minute: int,
    exit_minute: int,
    date_count: int,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "minute-open",
) -> tuple[
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    storage_symbol = _security_symbol(sample_id)
    dollar_liquidity = load_daily_dollar_volume(
        daily_data_dir / f"{storage_symbol}.npy", date_positions, date_count
    )
    entry_prices = np.full(date_count, np.nan, dtype=np.float64)
    morning_prices = np.full(date_count, np.nan, dtype=np.float64)
    entry_staleness = np.full(date_count, np.inf, dtype=np.float64)
    morning_staleness = np.full(date_count, np.inf, dtype=np.float64)

    source_path = minute_data_dir / f"{storage_symbol}.npy"
    if not source_path.exists():
        return (
            sample_id,
            dollar_liquidity,
            entry_prices,
            morning_prices,
            entry_staleness,
            morning_staleness,
        )

    source = np.load(source_path, mmap_mode="r")
    validate_bar_columns(source, "1Min", str(source_path))
    source_seconds = np.asarray(source[:, 0], dtype=np.int64)
    if len(source) and not np.all(source_seconds[:-1] < source_seconds[1:]):
        raise ValueError(f"{source_path} timestamps must be strictly increasing")

    # Non-open fields describe hypothetical execution over the labelled minute.
    # Never select a later bar; missing fields use only an earlier valid value
    # of the same field, with its age passed to the existing staleness checks.
    # External NBBO/auction prices replace these baseline arrays in main().
    for minute, price_source, prices, staleness in (
        (entry_minute, entry_price_source, entry_prices, entry_staleness),
        (exit_minute, exit_price_source, morning_prices, morning_staleness),
    ):
        bar_source = (
            "minute-open"
            if price_source in {"nbbo-ask", "nbbo-bid", "opening-auction"}
            else price_source
        )
        column = MINUTE_PRICE_COLUMNS[bar_source]
        values = source[:, column]
        valid_positions = np.flatnonzero(np.isfinite(values) & (values > 0))
        valid_seconds = source_seconds[valid_positions]
        requested = context_sod + (int(minute) - EXTENDED_OPEN_MINUTE) * 60
        positions = np.searchsorted(valid_seconds, requested, side="right") - 1
        available = positions >= 0
        observed_positions = valid_positions[positions[available]]
        prices[available] = np.asarray(values[observed_positions], dtype=np.float64) / 1000.0
        staleness[available] = (requested[available] - source_seconds[observed_positions]) / 60.0
    return (
        sample_id,
        dollar_liquidity,
        entry_prices,
        morning_prices,
        entry_staleness,
        morning_staleness,
    )


def build_daily_cache(
    minute_data_dir: Path,
    daily_data_dir: Path,
    dates: pd.DatetimeIndex,
    context_sod: np.ndarray,
    entry_minute: int,
    exit_minute: int,
    workers: int,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "minute-open",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Scan each symbol once and build dense date-by-symbol arrays."""
    symbols = simulation_symbols(minute_data_dir, daily_data_dir)
    date_positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    context_sod = np.asarray(context_sod, dtype=np.int64)
    if context_sod.shape != (len(dates),):
        raise ValueError("context session timestamps must match cache dates")
    dollar_liquidity = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    entry_prices = np.full_like(dollar_liquidity, np.nan)
    morning_prices = np.full_like(dollar_liquidity, np.nan)
    entry_staleness = np.full_like(dollar_liquidity, np.inf)
    morning_staleness = np.full_like(dollar_liquidity, np.inf)

    def submit(symbol: str):
        return _symbol_daily_arrays(
            symbol,
            date_positions,
            context_sod,
            minute_data_dir,
            daily_data_dir,
            entry_minute,
            exit_minute,
            len(dates),
            entry_price_source,
            exit_price_source,
        )

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(submit, str(symbol)): index for index, symbol in enumerate(symbols)}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="building simulation cache", unit="symbol"
        ):
            column = futures[future]
            (
                _,
                symbol_dollar_liquidity,
                symbol_entry,
                symbol_morning,
                symbol_entry_staleness,
                symbol_morning_staleness,
            ) = future.result()
            dollar_liquidity[:, column] = symbol_dollar_liquidity
            entry_prices[:, column] = symbol_entry
            morning_prices[:, column] = symbol_morning
            entry_staleness[:, column] = symbol_entry_staleness
            morning_staleness[:, column] = symbol_morning_staleness
    return (
        symbols,
        dollar_liquidity,
        entry_prices,
        morning_prices,
        entry_staleness,
        morning_staleness,
    )


def load_or_build_cache(
    cache_path: Path,
    metadata: dict[str, object],
    minute_data_dir: Path,
    daily_data_dir: Path,
    dates: pd.DatetimeIndex,
    context_sod: np.ndarray,
    entry_minute: int,
    exit_minute: int,
    workers: int,
    rebuild: bool,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "minute-open",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_metadata = json.loads(str(cache["metadata"].item()))
            if cached_metadata == metadata and np.array_equal(
                cache["dates"].astype("datetime64[D]"), dates.to_numpy(dtype="datetime64[D]")
            ):
                return (
                    cache["symbols"],
                    cache["dollar_volume"],
                    cache["entry_prices"],
                    cache["morning_prices"],
                    cache["entry_staleness"],
                    cache["morning_staleness"],
                )

    arrays = build_daily_cache(
        minute_data_dir,
        daily_data_dir,
        dates,
        context_sod,
        entry_minute,
        exit_minute,
        workers,
        entry_price_source,
        exit_price_source,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            metadata=json.dumps(metadata, sort_keys=True),
            dates=dates.to_numpy(dtype="datetime64[D]"),
            symbols=arrays[0],
            dollar_volume=arrays[1],
            entry_prices=arrays[2],
            morning_prices=arrays[3],
            entry_staleness=arrays[4],
            morning_staleness=arrays[5],
        )
    os.replace(temporary_path, cache_path)
    return arrays


def _profit_factor(returns: pd.Series) -> float:
    losses = float(-returns.clip(upper=0.0).sum())
    gains = float(returns.clip(lower=0.0).sum())
    return gains / losses if losses else (float("inf") if gains else float("nan"))


def strategy_metrics(returns: pd.Series) -> dict[str, float | int]:
    values = pd.Series(returns, dtype=np.float64)
    if values.empty or not np.isfinite(values).all():
        raise ValueError("strategy returns must be non-empty and finite")
    equity = (1.0 + values).cumprod()
    equity_with_origin = pd.concat((pd.Series([1.0]), equity.reset_index(drop=True)), ignore_index=True)
    drawdown = 1.0 - equity_with_origin / equity_with_origin.cummax()
    volatility = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    annualized_return = float(equity.iloc[-1] ** (252.0 / len(values)) - 1.0)
    downside_deviation = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    maximum_drawdown = float(drawdown.max())
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    return {
        "periods": len(values),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "average_win": float(wins.mean()) if len(wins) else float("nan"),
        "average_loss": float(losses.mean()) if len(losses) else float("nan"),
        "best_return": float(values.max()),
        "worst_return": float(values.min()),
        "win_rate": float(values.gt(0.0).mean()),
        "profit_factor": _profit_factor(values),
        "annualized_volatility": volatility * np.sqrt(252.0),
        "sharpe_zero_cash_rate": float(np.sqrt(252.0) * values.mean() / volatility)
        if volatility > 0.0
        else float("nan"),
        "sortino_zero_cash_rate": float(
            np.sqrt(252.0) * values.mean() / downside_deviation
        )
        if downside_deviation > 0.0
        else float("nan"),
        "max_drawdown": maximum_drawdown,
        "calmar_ratio": annualized_return / maximum_drawdown
        if maximum_drawdown > 0.0
        else float("nan"),
    }


def _benchmark_metrics(returns: pd.Series) -> dict[str, object]:
    if np.isfinite(returns.to_numpy()).all():
        return strategy_metrics(returns)
    return {key: 0 if key == "periods" else float("nan")
            for key in strategy_metrics(np.zeros(1))}


def _metric_text(summary: dict[str, object]) -> str:
    minimum_trading_days = int(summary["minimum_completed_trading_days"])
    if summary["liquidity_scheme"] == "dollar_ema":
        return (
            f"lagged log-dollar-volume EMA({summary['ema_span_sessions']}), "
            f"minimum {minimum_trading_days} completed trading days"
        )
    if summary["liquidity_scheme"] == "turnover_stability":
        return (
            f"lagged log-dollar-volume EMA({summary['ema_span_sessions']}) less its "
            f"{summary['ema_span_sessions']}-session dispersion, "
            f"minimum {minimum_trading_days} completed trading days"
        )
    raise ValueError(f"unknown liquidity scheme: {summary['liquidity_scheme']}")


def print_summary_table(summary: dict[str, object], console: Console | None = None) -> None:
    """Render the human-facing CLI report while leaving JSON optional."""
    output = console or Console()
    strategy = summary["strategy_metrics"]
    spy_buy_hold = summary["spy_buy_and_hold_metrics"]
    if not all(isinstance(metrics, dict) for metrics in (strategy, spy_buy_hold)):
        raise TypeError("summary metric groups must be mappings")

    comparison = Table(title=summary.get("strategy_label", STRATEGY_LABELS["liquidity-fixed"]), show_header=True, header_style="bold")
    comparison.add_column("Metric")
    comparison.add_column(f"Top {summary['top']}", justify="right")
    comparison.add_column("SPY buy & hold", justify="right")
    starting_capital = float(summary["budget"]) if summary.get("budget") is not None else 1.0
    comparison.add_row("Start capital", *(f"${starting_capital:,.2f}" for _ in range(2)))
    ending_capitals = (
        float(summary.get(
            "ending_equity", starting_capital * (1.0 + float(strategy["total_return"])),
        )),
        starting_capital * (1.0 + float(spy_buy_hold["total_return"])),
    )
    comparison.add_row(
        "End capital",
        *(f"${value:,.2f}" if np.isfinite(value) else "n/a" for value in ending_capitals),
    )
    strategy_trades = int(summary["trades"])
    trade_counts = (strategy_trades, int(spy_buy_hold["periods"] > 0))
    best_trade_count = min(trade_counts)
    comparison.add_row(
        "Round trips",
        *(
            f"[bold]{value:,}[/bold]" if value == best_trade_count else f"{value:,}"
            for value in trade_counts
        ),
    )
    rows = (
        ("Total return", "total_return", 100.0, "%", 2, True),
        ("Annualized return", "annualized_return", 100.0, "%", 2, True),
        ("Mean per night", "mean_return", 10_000.0, " bps", 2, True),
        ("Median per night", "median_return", 10_000.0, " bps", 2, True),
        ("Average winning night", "average_win", 10_000.0, " bps", 2, True),
        ("Average losing night", "average_loss", 10_000.0, " bps", 2, True),
        ("Best night", "best_return", 100.0, "%", 2, True),
        ("Worst night", "worst_return", 100.0, "%", 2, True),
        ("Win rate", "win_rate", 100.0, "%", 2, True),
        ("Profit factor", "profit_factor", 1.0, "", 2, True),
        ("Annualized volatility", "annualized_volatility", 100.0, "%", 2, False),
        ("Sharpe (zero cash rate)", "sharpe_zero_cash_rate", 1.0, "", 2, True),
        ("Sortino (zero cash rate)", "sortino_zero_cash_rate", 1.0, "", 2, True),
        ("Maximum drawdown", "max_drawdown", 100.0, "%", 2, False),
        ("Calmar ratio", "calmar_ratio", 1.0, "", 2, True),
    )
    for label, key, scale, suffix, precision, higher_is_better in rows:
        values = (
            float(strategy[key]) * scale,
            float(spy_buy_hold[key]) * scale,
        )
        finite_values = [value for value in values if np.isfinite(value)]
        winning_value = (
            (max(finite_values) if higher_is_better else min(finite_values))
            if finite_values
            else float("nan")
        )
        formatted = []
        for value in values:
            text = "n/a" if np.isnan(value) else f"{value:.{precision}f}{suffix}"
            if np.isfinite(value) and np.isclose(value, winning_value):
                text = f"[bold]{text}[/bold]"
            formatted.append(text)
        comparison.add_row(
            label,
            *formatted,
        )
    output.print(comparison)

    details = Table(show_header=False, box=None, padding=(0, 1))
    details.add_column(style="bold")
    details.add_column()
    details.add_row(
        "Period",
        f"{summary['first_entry_date']} {summary['entry_time_eastern']} -> "
        f"{summary['last_exit_date']} {str(summary['exit_time_eastern']).split()[0]} Eastern",
    )
    details.add_row("Sessions / trades", f"{strategy['periods']} / {summary['trades']:,}")
    skipped_short = int(summary.get("skipped_short_entry_sessions", 0))
    if skipped_short:
        details.add_row("Short sessions", f"{skipped_short} afternoon entries skipped")
    details.add_row(
        "Liquidity ranking",
        _metric_text(summary),
    )
    if summary.get("strategy") == "liquidity-trend-vol":
        details.add_row("Exposure policy", f"{summary['strategy_config']}; mean exposure {summary['average_exposure']:.2f}x")
        details.add_row("Financing", f"{summary['margin_interest_rate']:.2%} annual; calendar days / 360; borrow drag {summary['annual_borrow_drag']:.2%}/yr")
    details.add_row("Cost", f"{summary['transaction_cost_bps_per_side']:.2f} bps per side")
    source_descriptions = {
        **{
            source: f"Alpaca SIP minute-bar {source.removeprefix('minute-')}"
            for source in MINUTE_PRICE_COLUMNS
        },
        "nbbo-ask": "latest causal SIP ask",
        "nbbo-bid": "latest causal SIP bid",
        "opening-auction": "split-adjusted primary opening auction (Alpaca SIP condition O)",
    }
    for side in ("entry", "exit"):
        source = summary[f"{side}_price_source"]
        clock = str(summary[f"{side}_time_eastern"]).split()[0]
        timing = f"at {clock} Eastern"
        if source in MINUTE_PRICE_COLUMNS and source != "minute-open":
            hour, minute = map(int, clock.split(":"))
            end_minute = (hour * 60 + minute + 1) % (24 * 60)
            timing = (
                f"during {clock}–{end_minute // 60:02d}:{end_minute % 60:02d} Eastern "
                "(hypothetical fill)"
            )
        details.add_row(
            f"{side.title()} price source",
            f"{source}: {source_descriptions[source]} {timing}",
        )
    if summary.get("skipped_missing_prices", 0):
        details.add_row(
            "WARNING: missing prices",
            f"{summary['skipped_missing_prices']} symbol/date positions skipped across "
            f"{summary['missing_price_sessions']} sessions; original allocations held as cash. "
            "Missing exits are retrospective exclusions.",
        )
    if summary.get("missing_benchmark_sessions", 0):
        details.add_row("WARNING: benchmark gaps",
                        f"{summary['missing_benchmark_sessions']} sessions unavailable; "
                        "benchmark aggregate metrics are not reported")
    if summary.get("budget") is None:
        details.add_row("Capital basis", "$1.00 normalized start; set --budget for dollar sizing")
    if summary.get("budget") is not None:
        details.add_row(
            "Position sizing",
            f"{summary['share_mode']} shares; "
            f"mean deployed ${float(summary['average_capital_deployed']):,.2f} "
            f"({float(summary['average_capital_utilization']):.2%}), "
            f"mean basket {float(summary['average_executed_basket_size']):.2f}/"
            f"{int(summary['basket_size'])}",
        )
        if summary["share_mode"] == "whole":
            details.add_row(
                "Whole-share effects",
                f"minimum utilization {float(summary['minimum_capital_utilization']):.2%}; "
                f"minimum basket {int(summary['minimum_executed_basket_size'])}/"
                f"{int(summary['basket_size'])}; skipped selections "
                f"{int(summary['skipped_selections']):,}; mean weight spread "
                f"{float(summary['average_position_weight_spread']):.2%}",
            )
    details.add_row(
        "KPI sampling",
        f"daily at {str(summary['exit_time_eastern']).split()[0]}; "
        "buy-and-hold SPY remains continuously invested",
    )
    if float(summary.get("leverage", 1.0)) > 1.0:
        unlevered = summary["unlevered_metrics"]
        details.add_row(
            "Leverage",
            f"{summary['leverage']:.2f}x at {summary['margin_interest_rate']:.2%} annual "
            f"(rate/360 per calendar day, mean hold "
            f"{summary['mean_holding_calendar_days']:.2f}d); "
            f"borrow drag {summary['annual_borrow_drag']:.2%}/yr",
        )
        details.add_row(
            "Unlevered comparison",
            f"return {unlevered['annualized_return']:.2%}, "
            f"Sharpe {unlevered['sharpe_zero_cash_rate']:.2f}, "
            f"max drawdown {unlevered['max_drawdown']:.2%}",
        )
        details.add_row(
            "Margin headroom",
            f"worst session leaves {summary['worst_session_margin_ratio']:.0%} equity "
            f"against a {summary['maintenance_margin']:.0%} floor; "
            f"that session breaches at {summary['margin_breach_leverage']:.2f}x",
        )
    # Tolerate summaries built before these keys existed rather than raising in display.
    duplicate_days = int(summary.get("sessions_with_two_classes_of_one_issuer", 0))
    unnamed = int(summary.get("symbols_without_an_issuer_name", 0))
    if "deduped_share_classes" not in summary:
        detail = None
    elif summary["deduped_share_classes"]:
        detail = "one share class per company"
        if duplicate_days:
            detail = f"FAILED: {duplicate_days} session(s) still hold two classes of one issuer"
        if unnamed:
            detail += f"; {unnamed} traded symbol(s) had no name in the security master"
    else:
        detail = "disabled"
        if duplicate_days:
            detail += f"; {duplicate_days} session(s) hold two classes of one issuer"
    if detail is not None:
        details.add_row("Share classes", detail)
    details.add_row("Unique symbols", f"{summary['unique_symbols_traded']:,}")
    details.add_row(
        "Daily membership changes",
        f"mean {summary['average_daily_membership_replacements']:.2f}, "
        f"maximum {summary['maximum_daily_membership_replacements']}",
    )
    details.add_row(
        "Membership stability",
        f"retention {summary['average_daily_membership_retention']:.2%}, "
        f"Jaccard {summary['average_daily_membership_jaccard']:.3f}",
    )
    details.add_row(
        "Stale exit marks",
        f"{summary['stale_exit_marks_over_10_minutes']} over 10 minutes; "
        f"maximum {summary['maximum_exit_staleness_minutes']:.1f} minutes",
    )
    details.add_row("Cache", str(summary.get("cache_path", "not written")))
    output.print(details)


def print_symbol_trade_counts(
    trades: pd.DataFrame, console: Console | None = None
) -> None:
    """Print each symbol's selection frequency and average net return."""
    output = console or Console()
    counts = trades.groupby("sample_id").size().astype(int)
    average_returns = trades.groupby("sample_id")["net_return"].mean()
    session_count = int(trades["entry_date"].nunique())
    symbols = sorted(
        counts.index,
        key=lambda symbol: (-int(counts[symbol]), _security_symbol(str(symbol))),
    )
    table = Table(title="Per-symbol trade frequency", header_style="bold")
    table.add_column("Symbol")
    table.add_column("Trades / nights", justify="right")
    table.add_column("Average net/trade", justify="right")
    for symbol in symbols:
        count = int(counts[symbol])
        table.add_row(
            _security_symbol(str(symbol)),
            f"{count:,} / {count / session_count:.1%}",
            f"{float(average_returns[symbol]):+.3%}",
        )
    output.print(table)


def run_backtest(
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    dollar_volume: np.ndarray,
    entry_prices: np.ndarray,
    morning_prices: np.ndarray,
    entry_staleness: np.ndarray,
    morning_staleness: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top: int,
    ema_span: int,
    min_history_days: int,
    minimum_trading_days: int,
    transaction_cost_bps: float,
    max_entry_staleness_minutes: int,
    max_exit_staleness_minutes: int,
    liquidity_scheme: str = "turnover_stability",
    entry_minute: int = 15 * 60 + 45,
    exit_minute: int = 9 * 60 + 30,
    issuers: Mapping[str, str] | None = None,
    dedupe_share_classes: bool = True,
    leverage: float = 1.0,
    margin_interest_rate: float | None = None,
    maintenance_margin: float = MAINTENANCE_MARGIN,
    reference_symbol: str = REFERENCE_SYMBOL,
    share_mode: str = "fractional",
    budget: float | None = None,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "opening-auction",
    execution_exchange_mask: np.ndarray | None = None,
    entry_session_mask: np.ndarray | None = None,
    strategy: str = "liquidity-fixed",
    strategy_config: LiquidityTrendVolConfig | None = None,
    spy_trend_marks: np.ndarray | None = None,
    spy_trend_price_source: str = "daily-close",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if spy_trend_price_source not in SPY_TREND_PRICE_SOURCES:
        raise ValueError("unsupported SPY trend price source")
    trend_vol = strategy == "liquidity-trend-vol"
    if not np.isfinite(leverage) or not 0 < leverage <= MAX_OVERNIGHT_LEVERAGE:
        raise ValueError("leverage must be in (0, 2]")
    if strategy_config is not None and not trend_vol:
        raise ValueError("strategy_config applies only to liquidity-trend-vol")
    config = strategy_config or LiquidityTrendVolConfig()
    policy = None
    if margin_interest_rate is None:
        margin_interest_rate = 0.0675 if trend_vol else 0.0
    if trend_vol:
        if exit_minute >= entry_minute:
            raise ValueError("liquidity-trend-vol requires exit-time before entry-time so risk observations are completed")
        if leverage != 1.0:
            raise ValueError("liquidity-trend-vol controls exposure; --leverage applies only to liquidity-fixed")
        if spy_trend_marks is None or np.asarray(spy_trend_marks).shape != (len(dates),):
            raise ValueError("liquidity-trend-vol requires one SPY trend price per date")
        if start_date < pd.Timestamp(config.risk_history_start):
            raise ValueError("liquidity-trend-vol start_date precedes risk_history_start")
        policy = LiquidityTrendVolPolicy(config, spy_trend_marks)
    if not np.isfinite(margin_interest_rate) or not 0 <= margin_interest_rate <= 1:
        raise ValueError("margin_interest_rate must be a fraction between 0 and 1")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if int(minimum_trading_days) < 1:
        raise ValueError("minimum_trading_days must be positive")
    if share_mode not in {"fractional", "whole"}:
        raise ValueError("share_mode must be 'fractional' or 'whole'")
    if entry_price_source not in ENTRY_PRICE_SOURCES:
        raise ValueError(f"entry_price_source must be one of {ENTRY_PRICE_SOURCES}")
    if exit_price_source not in EXIT_PRICE_SOURCES:
        raise ValueError(f"exit_price_source must be one of {EXIT_PRICE_SOURCES}")
    if budget is not None and (not np.isfinite(budget) or float(budget) <= 0.0):
        raise ValueError("budget must be finite and positive")
    if share_mode == "whole" and budget is None:
        raise ValueError("whole-share sizing requires a budget")
    simulation_budget = float(budget) if budget is not None else 1.0
    basket_history = BasketHistory(
        symbols, dollar_volume, top=top, ema_span=ema_span, min_history_days=min_history_days,
        minimum_trading_days=int(minimum_trading_days), liquidity_scheme=liquidity_scheme,
        issuers=issuers, dedupe_share_classes=dedupe_share_classes, reference_symbol=reference_symbol,
    )
    scores = basket_history.scores
    if execution_exchange_mask is not None:
        execution_exchange_mask = np.asarray(execution_exchange_mask, dtype=bool)
        if execution_exchange_mask.shape != scores.shape:
            raise ValueError("execution_exchange_mask must match the liquidity arrays")
    interval_entries = (dates >= start_date) & (dates < end_date)
    interval_entries[-1] = False
    interval_entries[:-1] &= dates[1:] <= end_date
    if entry_session_mask is None:
        entry_session_mask = np.ones(len(dates), dtype=bool)
    else:
        entry_session_mask = np.asarray(entry_session_mask, dtype=bool)
        if entry_session_mask.shape != (len(dates),):
            raise ValueError("entry_session_mask must match dates")
    skipped_short_entries = int((interval_entries & ~entry_session_mask).sum())
    entries = np.flatnonzero(interval_entries)
    if not entries.size:
        raise ValueError("the requested interval contains no entry sessions")
    if entries[-1] + 1 >= len(dates):
        raise ValueError("the final entry session has no following exit session")
    reference_matches = np.flatnonzero(symbols == str(reference_symbol))
    if reference_matches.size != 1:
        raise ValueError(f"cache must contain exactly one {reference_symbol}")
    reference_index = int(reference_matches[0])
    cost = 2.0 * float(transaction_cost_bps) / 10_000.0

    pieces: list[pd.DataFrame] = []
    session_records: list[dict[str, object]] = []
    skipped_prices: list[dict[str, object]] = []
    missing_benchmark_dates: list[str] = []
    entry_age_limit = min(max_entry_staleness_minutes, 1) if entry_price_source == "nbbo-ask" else max_entry_staleness_minutes
    exit_age_limit = min(max_exit_staleness_minutes, 1) if exit_price_source == "nbbo-bid" else max_exit_staleness_minutes
    spy_initial_entry = float(entry_prices[entries[0], reference_index])
    spy_exit_prices: list[float] = []
    current_equity = simulation_budget
    replay_entries = entries
    if trend_vol:
        replay_entries = np.flatnonzero(
            (dates >= pd.Timestamp(config.risk_history_start)) & (dates < end_date)
        )
        replay_entries = replay_entries[replay_entries < len(dates) - 1]
        replay_entries = replay_entries[dates[replay_entries + 1] <= end_date]
    for date_index in replay_entries:
        cash_session = not entry_session_mask[date_index]
        exposure = policy.exposure(date_index) if policy is not None else float(leverage)
        selected = basket_history.select(
            date_index, entry_allowed=not cash_session,
            entry_prices=entry_prices[date_index], entry_staleness=entry_staleness[date_index],
            max_entry_staleness_minutes=int(max_entry_staleness_minutes),
            entry_price_source=entry_price_source,
            # The legacy fixed-exposure model used exit-day venue membership.
            # Trend/vol and live warmup both use entry-day membership.
            exchange_mask=(execution_exchange_mask[date_index if trend_vol else date_index + 1]
                           if execution_exchange_mask is not None else None),
        )
        selected_ranks = np.arange(1, len(selected) + 1)
        selected_entries = entry_prices[date_index, selected]
        selected_entry_staleness = entry_staleness[date_index, selected]
        exits = morning_prices[date_index + 1, selected]
        exit_staleness = morning_staleness[date_index + 1, selected]
        observation = basket_returns(
            selected_entries, exits, slots=top, cost_bps=transaction_cost_bps,
            entry_staleness=selected_entry_staleness, exit_staleness=exit_staleness,
            max_entry_age=entry_age_limit, max_exit_age=exit_age_limit,
        )
        missing_entry, missing_exit = observation.missing_entry, observation.missing_exit
        missing = observation.missing
        if trend_vol and missing.any():
            names = ", ".join(symbols[selected[missing]])
            raise ValueError(f"liquidity-trend-vol requires complete fresh prices on {dates[date_index].date()}: {names}")
        unscaled_return = observation.unscaled_return
        if policy is not None:
            policy.observe(unscaled_return)
        if dates[date_index] < start_date:
            continue
        for index in np.flatnonzero(missing):
            reason = ", ".join(side for side, mask in (("entry", missing_entry), ("exit", missing_exit)) if mask[index])
            record = {
                "symbol": str(symbols[selected[index]]),
                "entry_date": str(dates[date_index].date()),
                "exit_date": str(dates[date_index + 1].date()),
                "reason": f"missing or stale {reason} price",
            }
            skipped_prices.append(record)
            LOGGER.warning("Skipping %s (%s -> %s): %s; allocation remains cash",
                           record["symbol"], record["entry_date"], record["exit_date"], record["reason"])
        session_budget = current_equity
        quantities = np.zeros(len(selected), dtype=np.float64)
        valid = ~missing
        if valid.any():
            # Preserve the original per-stock allocation, including skipped slots.
            quantities[valid] = basket_quantities(
                selected_entries[valid], session_budget * exposure, share_mode, slots=top
            )
        executed = quantities > 0.0
        skipped = int((~executed).sum())
        if not executed.any() and not missing.any() and not cash_session:
            raise ValueError(
                f"equity ${session_budget:,.2f} cannot buy one share from the selected "
                f"basket on {dates[date_index].date()}"
            )
        selected = selected[executed]
        selected_ranks = selected_ranks[executed]
        selected_entries = selected_entries[executed]
        quantities = quantities[executed]
        exits = exits[executed]
        exit_staleness = exit_staleness[executed]
        gross = exits / selected_entries - 1.0
        entry_notional = quantities * selected_entries
        exit_notional = quantities * exits
        net_return = gross - cost
        gross_pnl = exit_notional - entry_notional
        transaction_cost_dollars = entry_notional * cost
        net_pnl = entry_notional * net_return
        holding_calendar_days = max(1, (dates[date_index + 1] - dates[date_index]).days)
        borrow_dollars = (
            max(0.0, float(entry_notional.sum()) - session_budget)
            * margin_interest_rate * holding_calendar_days / MARGIN_INTEREST_DIVISOR
        )
        session_net_pnl = float(net_pnl.sum()) - borrow_dollars
        session_return = session_net_pnl / session_budget
        current_equity = session_budget + session_net_pnl
        if current_equity <= 0:
            raise ValueError(f"portfolio equity exhausted on {dates[date_index + 1].date()}")
        session_records.append({
            "entry_date": str(dates[date_index].date()),
            "exit_date": str(dates[date_index + 1].date()),
            "portfolio_start_equity": session_budget,
            "portfolio_end_equity": current_equity,
            "portfolio_return": session_return,
            "capital_deployed": float(entry_notional.sum()),
            "exit_position_value": float(exit_notional.sum()),
            "exposure": exposure,
            "unscaled_return": unscaled_return,
            "borrow_cost": borrow_dollars,
            "borrow_return": borrow_dollars / session_budget,
            "traded": bool(executed.any()),
            "skipped_selections": skipped,
            "skipped_missing_prices": int(missing.sum()),
        })
        pieces.append(
            pd.DataFrame(
                {
                    "entry_date": str(dates[date_index].date()),
                    "exit_date": str(dates[date_index + 1].date()),
                    "liquidity_scheme": liquidity_scheme,
                    "rank": selected_ranks,
                    "sample_id": symbols[selected],
                    "liquidity_score": scores[date_index, selected],
                    "share_mode": share_mode,
                    "budget": session_budget,
                    "target_notional": equal_notional(session_budget * exposure, top),
                    "exposure": exposure,
                    "portfolio_start_equity": session_budget,
                    "portfolio_end_equity": current_equity,
                    "portfolio_return": session_return,
                    "quantity": quantities,
                    "entry_price": selected_entries,
                    "entry_price_source": entry_price_source,
                    "exit_price": exits,
                    "exit_price_source": exit_price_source,
                    "entry_notional": entry_notional,
                    "exit_notional": exit_notional,
                    "entry_staleness_minutes": entry_staleness[date_index, selected],
                    "exit_staleness_minutes": exit_staleness,
                    "gross_return": gross,
                    "gross_pnl": gross_pnl,
                    "transaction_cost": cost,
                    "transaction_cost_dollars": transaction_cost_dollars,
                    "net_return": net_return,
                    "net_pnl": net_pnl,
                    "skipped_selections": skipped,
                }
            )
        )
        spy_exit = morning_prices[date_index + 1, reference_index]
        # Buy once at the reporting window's start, then retain SPY continuously.
        # Subsequent afternoon entry quotes are irrelevant, including cash sessions.
        missing_spy_entry = date_index == entries[0] and (
            not np.isfinite(spy_initial_entry) or spy_initial_entry <= 0.0
            or not np.isfinite(entry_staleness[date_index, reference_index])
            or entry_staleness[date_index, reference_index] > entry_age_limit
        )
        if missing_spy_entry:
            spy_initial_entry = float("nan")
        if (
            missing_spy_entry
            or not np.isfinite(spy_exit) or spy_exit <= 0.0
            or not np.isfinite(morning_staleness[date_index + 1, reference_index])
            or morning_staleness[date_index + 1, reference_index] > exit_age_limit
        ):
            day = str(dates[date_index].date())
            missing_benchmark_dates.append(day)
            LOGGER.warning("%s buy-and-hold benchmark unavailable for %s -> %s: missing or stale price",
                           reference_symbol, day, dates[date_index + 1].date())
            spy_exit_prices.append(float("nan"))
        else:
            spy_exit_prices.append(float(spy_exit))

    trades = pd.concat(pieces, ignore_index=True)
    sessions = pd.DataFrame(session_records).set_index("entry_date")
    deployed_by_entry = sessions.capital_deployed
    utilization_by_entry = deployed_by_entry / sessions.portfolio_start_equity
    unlevered_daily = sessions.unscaled_return if trend_vol else (sessions.portfolio_return + sessions.borrow_return) / leverage

    holding_days = pd.Series(
        (pd.to_datetime(sessions.exit_date.values) - pd.to_datetime(sessions.index)).days,
        index=sessions.index, dtype=np.float64,
    ).clip(lower=1.0)
    borrow_per_session = sessions.borrow_return
    daily = sessions.portfolio_return
    spy_buy_hold_equity = np.asarray(spy_exit_prices, dtype=np.float64) / spy_initial_entry
    spy_buy_hold_returns = np.empty_like(spy_buy_hold_equity)
    spy_buy_hold_returns[0] = spy_buy_hold_equity[0] - 1.0
    spy_buy_hold_returns[1:] = spy_buy_hold_equity[1:] / spy_buy_hold_equity[:-1] - 1.0
    side_cost = float(transaction_cost_bps) / 10_000.0
    spy_buy_hold_returns[0] -= side_cost
    spy_buy_hold_returns[-1] -= side_cost
    spy_buy_hold_daily = pd.Series(spy_buy_hold_returns, index=daily.index, dtype=np.float64)
    difference_buy_hold = daily - spy_buy_hold_daily
    sessions["strategy_return"] = daily
    sessions["spy_buy_and_hold_return"] = spy_buy_hold_daily
    by_date = {day: set(group.sample_id) for day, group in trades.groupby("entry_date", sort=True)}
    memberships = [by_date.get(day, set()) for day in sessions.index]
    replacements = [
        len(current - previous) for previous, current in itertools.pairwise(memberships)
    ]
    retentions = [
        len(current & previous) / len(current) if current else 1.0
        for previous, current in itertools.pairwise(memberships)
    ]
    jaccards = [
        len(current & previous) / len(current | previous) if current | previous else 1.0
        for previous, current in itertools.pairwise(memberships)
    ]
    exposed = sessions.capital_deployed / sessions.portfolio_start_equity
    ratios = sessions.portfolio_end_equity / sessions.exit_position_value.replace(0, np.nan)
    margin_ratio = float(ratios[exposed > 1].min()) if (exposed > 1).any() else 1.0
    worst_session = float(unlevered_daily.min())
    breach_leverage = (
        1.0 / (float(maintenance_margin) * (1.0 + worst_session) - worst_session)
        if worst_session < 0 else float("inf")
    )

    # The dedupe rule reads company names out of the security master, so it can fail
    # quietly if a name format changes or a symbol is missing. Report both, rather than
    # trusting that it worked: a silent no-op is the failure mode that matters.
    traded_symbols = sorted(trades.sample_id.unique())
    resolved = {symbol: (issuers or {}).get(symbol, symbol) for symbol in traded_symbols}
    unresolved = [symbol for symbol, key in resolved.items() if key == symbol]
    same_issuer_days = 0
    for _, group in trades.groupby("entry_date", sort=False):
        keys = [resolved[symbol] for symbol in group.sample_id]
        same_issuer_days += len(keys) != len(set(keys))

    executed_basket_sizes = trades.groupby("entry_date", sort=True).size().reindex(sessions.index, fill_value=0)
    position_weights = trades.entry_notional / trades.groupby("entry_date").entry_notional.transform("sum")
    weight_spreads = position_weights.groupby(trades.entry_date).agg(lambda values: values.max() - values.min())
    skipped_by_entry = sessions.skipped_selections

    liquidity_descriptions = {
        "dollar_ema": "completed regular-session dollar volume",
        "turnover_stability": "completed dollar volume less its own dispersion",
    }
    summary: dict[str, object] = {
        "strategy": strategy,
        "strategy_config": config.as_dict() if trend_vol else {"leverage": float(leverage)},
        "spy_trend_price_source": spy_trend_price_source if trend_vol else None,
        "exchange_membership_session": "entry" if trend_vol else "exit",
        "strategy_label": STRATEGY_LABELS[strategy],
        "leverage_label": f"dynamic exposure, {config.max_exposure:g}× cap" if trend_vol else f"{leverage:g}× leverage",
        "average_exposure": float(sessions.exposure.mean()),
        "maximum_exposure": float(sessions.exposure.max()),
        "liquidity_scheme": liquidity_scheme,
        "entry_time_eastern": f"{entry_minute // 60:02d}:{entry_minute % 60:02d}",
        "entry_price_source": entry_price_source,
        "exit_time_eastern": f"{exit_minute // 60:02d}:{exit_minute % 60:02d} next trading session",
        "exit_price_source": exit_price_source,
        "first_entry_date": str(pd.Timestamp(daily.index[0]).date()),
        "last_entry_date": str(pd.Timestamp(daily.index[-1]).date()),
        "last_exit_date": str(sessions.exit_date.iloc[-1]),
        "top": int(top),
        "basket_size": int(top),
        "share_mode": share_mode,
        "budget": float(budget) if budget is not None else None,
        "ending_equity": float(simulation_budget * np.prod(1.0 + daily.to_numpy())),
        "net_profit": float(simulation_budget * (np.prod(1.0 + daily.to_numpy()) - 1.0)),
        "average_capital_deployed": float(deployed_by_entry.mean()),
        "average_capital_utilization": float(utilization_by_entry.mean()),
        "minimum_capital_utilization": float(utilization_by_entry.min()),
        "average_executed_basket_size": float(executed_basket_sizes.mean()),
        "minimum_executed_basket_size": int(executed_basket_sizes.min()),
        "skipped_selections": int(skipped_by_entry.sum()),
        "skipped_short_entry_sessions": skipped_short_entries,
        "skipped_missing_prices": len(skipped_prices),
        "skipped_price_details": skipped_prices,
        "missing_price_sessions": int((sessions.skipped_missing_prices > 0).sum()),
        "all_cash_sessions": int((executed_basket_sizes == 0).sum()),
        "missing_benchmark_sessions": len(missing_benchmark_dates),
        "missing_benchmark_dates": missing_benchmark_dates,
        "daily_portfolio": sessions.reset_index().to_dict("records"),
        "average_position_weight_spread": float(weight_spreads.mean()),
        "liquidity_metric": liquidity_descriptions[liquidity_scheme],
        "ema_span_sessions": int(ema_span),
        "minimum_liquidity_history_sessions": int(min_history_days),
        "minimum_completed_trading_days": int(minimum_trading_days),
        "transaction_cost_bps_per_side": float(transaction_cost_bps),
        "deduped_share_classes": bool(dedupe_share_classes and issuers is not None),
        "sessions_with_two_classes_of_one_issuer": int(same_issuer_days),
        "symbols_without_an_issuer_name": len(unresolved),
        "leverage": float(leverage),
        "margin_interest_rate": float(margin_interest_rate),
        "annual_borrow_drag": float(borrow_per_session.sum()) / (len(unlevered_daily) / 252.0),
        "mean_holding_calendar_days": float(holding_days.mean()),
        "maintenance_margin": float(maintenance_margin),
        "worst_session_margin_ratio": float(margin_ratio),
        "margin_breach_leverage": float(breach_leverage),
        "unlevered_metrics": strategy_metrics(unlevered_daily),
        "max_entry_staleness_minutes": int(max_entry_staleness_minutes),
        "max_exit_staleness_minutes": int(max_exit_staleness_minutes),
        "stale_exit_marks_over_10_minutes": int(trades.exit_staleness_minutes.gt(10.0).sum()),
        "maximum_exit_staleness_minutes": float(trades.exit_staleness_minutes.max()),
        "trades": len(trades),
        "unique_symbols_traded": int(trades.sample_id.nunique()),
        "average_daily_membership_replacements": float(np.mean(replacements))
        if replacements
        else 0.0,
        "maximum_daily_membership_replacements": int(max(replacements, default=0)),
        "average_daily_membership_retention": float(np.mean(retentions))
        if retentions
        else 1.0,
        "average_daily_membership_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "strategy_metrics": strategy_metrics(daily),
        "spy_buy_and_hold_metrics": _benchmark_metrics(spy_buy_hold_daily),
        "versus_spy_buy_and_hold": {
            "mean_excess_return": float(difference_buy_hold.mean()),
            "median_excess_return": float(difference_buy_hold.median()),
            "outperformance_days": int(difference_buy_hold.gt(0.0).sum()),
            "underperformance_days": int(difference_buy_hold.lt(0.0).sum()),
            "daily_return_correlation": float(daily.corr(spy_buy_hold_daily)),
        },
    }
    years = (pd.Timestamp(summary["last_exit_date"]) - pd.Timestamp(summary["first_entry_date"])).days / 365.25
    summary["strategy_metrics"]["calendar_cagr"] = float((1 + summary["strategy_metrics"]["total_return"]) ** (1 / years) - 1)
    return trades, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the causal overnight basket used by live trading."
        )
    )
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default="liquidity-trend-vol",
        help="strategy family (default: liquidity-trend-vol)",
    )
    parser.add_argument("--strategy-config", type=Path, default=None,
                        help="JSON overrides for liquidity-trend-vol exposure parameters")
    parser.add_argument(
        "--spy-trend-price-source", choices=SPY_TREND_PRICE_SOURCES, default="daily-close",
        help="SPY trend input; minute-open-1559 reproduces the original research",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="save trades, portfolio, summary and chart together; liquidity-trend-vol defaults under /tmp/trading-backtests/candidate/liquidity-trend-vol")
    parser.add_argument("--top", type=int, default=12, help="daily basket size")
    period = parser.add_mutually_exclusive_group()
    period.add_argument(
        "--months",
        type=int,
        default=None,
        help="trailing calendar months (default: 12 when --since is omitted)",
    )
    period.add_argument(
        "--since",
        type=_parse_day,
        default=None,
        metavar="YYYY-MM-DD",
        help="anchor the first eligible entry session on or after this date",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_day,
        default=None,
        help="final exit date, default: latest data date",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=10,
        help="liquidity EMA span; also the turnover-stability dispersion span",
    )
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=100,
        help="minimum completed observed sessions before a stock can be selected",
    )
    parser.add_argument(
        "--liquidity-scheme",
        choices=("dollar_ema", "turnover_stability"),
        default="turnover_stability",
        help="completed-session liquidity formula shared with live trading",
    )
    parser.add_argument("--entry-time", type=_parse_clock, default=_parse_clock("15:45"))
    parser.add_argument(
        "--entry-price-source",
        choices=ENTRY_PRICE_SOURCES,
        default="nbbo-ask",
        help="minute-* selects a field of the bar starting at --entry-time; non-open "
        "fields model hypothetical fills over that minute, known only at its end. "
        "nbbo-ask uses the latest causal SIP ask at 15:45 (default: nbbo-ask)",
    )
    parser.add_argument("--exit-time", type=_parse_clock, default=_parse_clock("09:30"))
    parser.add_argument(
        "--exit-price-source",
        choices=EXIT_PRICE_SOURCES,
        default="opening-auction",
        help="opening-auction uses the primary opening cross; minute-* selects a field "
        "of the bar starting at --exit-time. Non-open fields model hypothetical fills "
        "over that minute, known only at its end; nbbo-bid uses the latest causal SIP bid "
        "at --exit-time (default: opening-auction)",
    )
    parser.add_argument(
        "--auctions-path",
        default=DEFAULT_AUCTIONS_PATH,
        help="split-adjusted NPZ written by scripts/download_auctions.py",
    )
    parser.add_argument(
        "--nbbo-path",
        default=DEFAULT_NBBO_PATH,
        help="split-adjusted NPZ written by scripts/download_nbbo.py",
    )
    parser.add_argument(
        "--exit-nbbo-path",
        default=DEFAULT_EXIT_NBBO_PATH,
        help="split-adjusted NBBO NPZ for --exit-price-source nbbo-bid",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=None,
        help="additional bps per side (default: 0 for ask/auction, 1 otherwise)",
    )
    parser.add_argument(
        "--share-mode",
        choices=("fractional", "whole"),
        default="fractional",
        help="fractional preserves exact equal notionals; whole rounds each allocation down",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="initial portfolio equity, compounded between baskets; required by --share-mode whole",
    )
    parser.add_argument("--max-entry-staleness-minutes", type=int, default=10)
    parser.add_argument(
        "--max-exit-staleness-minutes",
        type=int,
        default=24 * 60,
        help="maximum age of the causal exit mark; avoids dropping a selected name with lookahead",
    )
    parser.add_argument(
        "--minute-bars-dir",
        default=DEFAULT_DATA_DIR,
        help="split-adjusted 1-minute bars used for execution prices",
    )
    parser.add_argument(
        "--daily-bars-dir",
        default=DEFAULT_DAILY_DATA_DIR,
        help="split-adjusted 1-day bars used for completed-session dollar-volume ranking",
    )
    parser.add_argument("--cache-dir", default="/tmp/trading/baseline_cache")
    parser.add_argument(
        "--asset-filter",
        choices=("companies", "all"),
        default="companies",
        help="companies excludes funds, ETFs/ETNs, units, preferreds, rights/warrants, and SPAC shells",
    )
    parser.add_argument(
        "--exchange-filter",
        choices=("all", "nasdaq"),
        default="nasdaq",
        help="restrict the candidate universe before ranking; SPY remains as benchmark. "
        "Defaults to nasdaq to match live execution; pass all for the unrestricted universe",
    )
    parser.add_argument(
        "--unclassified-asset-policy",
        choices=("keep", "exclude"),
        default="keep",
        help=(
            "policy for historical symbols absent from today's security master; "
            "keep avoids survivorship bias"
        ),
    )
    parser.add_argument("--security-master-cache", default=DEFAULT_SECURITY_MASTER_CACHE)
    parser.add_argument("--security-master-max-age-days", type=int, default=7)
    parser.add_argument("--refresh-security-master", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--no-dedupe-share-classes",
        dest="dedupe_share_classes",
        action="store_false",
        help="allow two share classes of the same company in one basket (e.g. GOOG and GOOGL)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="gross exposure as a multiple of equity; overnight holds are capped at 2.0 by Reg T",
    )
    parser.add_argument(
        "--margin-interest-rate",
        type=_parse_percentage,
        default=None,
        metavar="PERCENT",
        help="annual interest charged on the borrowed portion, as a percentage: Alpaca "
        "charges 6.75%% non-elite / 5.25%% elite, accrued as rate/360 per calendar day. "
        "Required above 1x",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--show-symbol-trade-frequency",
        action="store_true",
        help="print the per-symbol trade-frequency and average-return table",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_trend_vol_config(args.strategy_config) if args.strategy == "liquidity-trend-vol" else None
        if args.strategy_config is not None and args.strategy != "liquidity-trend-vol":
            raise ValueError("--strategy-config requires --strategy liquidity-trend-vol")
        if args.strategy == "liquidity-trend-vol":
            if args.leverage != 1.0:
                raise ValueError("liquidity-trend-vol controls exposure; --leverage applies only to liquidity-fixed")
            if args.margin_interest_rate is None:
                args.margin_interest_rate = 0.0675
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        args.transaction_cost_bps = resolve_transaction_cost_bps(
            args.transaction_cost_bps, args.entry_price_source, args.exit_price_source,
        )
    except ValueError as error:
        parser.error(str(error))

    if not np.isfinite(args.leverage) or args.leverage < 1.0:
        parser.error("--leverage must be at least 1.0")
    if args.leverage > MAX_OVERNIGHT_LEVERAGE:
        # Alpaca's 4x multiplier is day-trading buying power. This strategy holds
        # overnight by construction, so Reg T's 2x requirement governs and there is no
        # legitimate run above it -- an override would only produce an untradeable curve.
        parser.error(
            f"--leverage cannot exceed {MAX_OVERNIGHT_LEVERAGE:.1f} for an overnight hold; "
            "Alpaca's 4x multiplier is day-trading buying power and does not survive the close"
        )
    if args.leverage > 1.0 and args.margin_interest_rate is None:
        # Without a borrow rate the result is a pure scalar multiple and the Sharpe
        # ratio is unchanged, which would be misleading rather than merely incomplete.
        parser.error("--margin-interest-rate is required when --leverage exceeds 1.0")
    if args.budget is not None and args.budget <= 0.0:
        parser.error("--budget must be positive")
    if args.share_mode == "whole" and args.budget is None:
        parser.error("--budget is required when --share-mode whole")

    if (
        (args.months is not None and args.months < 1)
        or args.ema_span < 1
        or args.min_history_days < 1
        or args.min_trading_days < 1
    ):
        parser.error(
            "months, ema-span, min-history-days, and min-trading-days must be positive"
        )
    if args.liquidity_scheme == "turnover_stability" and args.ema_span < 2:
        parser.error("--ema-span must be at least 2 for turnover stability")
    if args.security_master_max_age_days < 1:
        parser.error("security-master-max-age-days must be positive")
    if args.top < 1:
        parser.error("top must be positive")
    if args.exit_price_source == "opening-auction" and args.exit_time != 9 * 60 + 30:
        parser.error("--exit-price-source opening-auction requires --exit-time 09:30")
    if args.entry_price_source == "nbbo-ask" and args.entry_time != 15 * 60 + 45:
        parser.error("--entry-price-source nbbo-ask requires --entry-time 15:45")
    if (
        args.workers < 1
        or args.max_entry_staleness_minutes < 0
        or args.max_exit_staleness_minutes < 0
    ):
        parser.error("workers must be positive and staleness must be non-negative")

    auction_path = Path(args.auctions_path)
    if not auction_path.exists():
        parser.error(
            f"auction data required for official session-close filtering does not exist: "
            f"{auction_path}"
        )
    data_dir = Path(args.minute_bars_dir)
    daily_data_dir = Path(args.daily_bars_dir)
    known_session_closes = auction_close_minutes(auction_path, None)
    all_dates, all_context_sod = reference_session_calendar(
        data_dir / f"{REFERENCE_SYMBOL}.npy", known_session_closes
    )
    try:
        window = historical_window(
            all_dates, all_context_sod, auction_path,
            since=args.since, end_date=args.end_date, months=args.months,
            ema_span=args.ema_span, min_history_days=args.min_history_days,
            min_trading_days=args.min_trading_days, entry_time=args.entry_time,
        )
    except ValueError as error:
        parser.error(str(error))
    requested_start, end_date = window.requested_start, window.end_date
    cache_dates, cache_context_sod = window.dates, window.context_sod
    entry_session_mask, shortened_entries = window.entry_session_mask, window.shortened_entries
    # Shared history and session calendar make strategy comparisons independent
    # of a report's start date; feature caches can be reused across variants.
    history = all_dates <= end_date
    cache_dates, cache_context_sod = all_dates[history], all_context_sod[history]
    entry_session_mask = np.array([
        known_session_closes.get(day.date(), 960) > args.entry_time for day in cache_dates
    ])
    spy_trend_marks = None
    if args.strategy == "liquidity-trend-vol":
        if requested_start < pd.Timestamp(config.risk_history_start):
            parser.error("--since precedes liquidity-trend-vol risk_history_start")
        if args.spy_trend_price_source == "daily-close":
            spy_trend_marks = load_daily_closes(daily_data_dir / f"{REFERENCE_SYMBOL}.npy", cache_dates)
        else:
            spy_trend_marks = _symbol_daily_arrays(
                REFERENCE_SYMBOL, {day: i for i, day in enumerate(cache_dates)},
                cache_context_sod, data_dir, daily_data_dir, 959, 570, len(cache_dates),
            )[2]
    if shortened_entries:
        print(
            f"short sessions: skipping {len(shortened_entries):,} afternoon entr"
            f"{'y' if len(shortened_entries) == 1 else 'ies'} at {args.entry_time // 60:02d}:"
            f"{args.entry_time % 60:02d}"
        )

    metadata = _cache_metadata(
        data_dir,
        daily_data_dir,
        pd.Timestamp(cache_dates[0]),
        pd.Timestamp(cache_dates[-1]),
        args.entry_time,
        args.exit_time,
        args.entry_price_source,
        args.exit_price_source,
    )
    cache_name = (
        f"liquidity_{cache_dates[0]:%Y%m%d}_{cache_dates[-1]:%Y%m%d}_"
        f"e{args.entry_time:04d}_{args.entry_price_source}_"
        f"x{args.exit_time:04d}_{args.exit_price_source}_v8.npz"
    )
    cache_path = Path(args.cache_dir) / cache_name
    (
        symbols,
        dollar_volume,
        entry_prices,
        morning_prices,
        entry_staleness,
        morning_staleness,
    ) = load_or_build_cache(
        cache_path,
        metadata,
        data_dir,
        daily_data_dir,
        cache_dates,
        cache_context_sod,
        args.entry_time,
        args.exit_time,
        args.workers,
        args.rebuild_cache,
        args.entry_price_source,
        args.exit_price_source,
    )
    unfiltered_candidates = int(len(symbols) - int((symbols == REFERENCE_SYMBOL).sum()))
    excluded_asset_reasons: dict[str, int] = {}
    unclassified_asset_symbols = 0
    security_master: dict[str, dict[str, object]] = {}
    if args.asset_filter == "companies" or args.exchange_filter != "all":
        security_master = load_nasdaq_security_master(
            Path(args.security_master_cache),
            refresh=args.refresh_security_master,
            max_age_days=args.security_master_max_age_days,
        )
    if args.asset_filter == "companies":
        company_mask, excluded_asset_reasons, unclassified_asset_symbols = company_universe_mask(
            symbols,
            security_master,
            keep_unclassified=args.unclassified_asset_policy == "keep",
        )
        symbols = symbols[company_mask]
        dollar_volume = dollar_volume[:, company_mask]
        entry_prices = entry_prices[:, company_mask]
        morning_prices = morning_prices[:, company_mask]
        entry_staleness = entry_staleness[:, company_mask]
        morning_staleness = morning_staleness[:, company_mask]
        retained_candidates = int(len(symbols) - int((symbols == REFERENCE_SYMBOL).sum()))
        print(
            f"company universe: retained {retained_candidates:,}/{unfiltered_candidates:,} "
            f"candidate symbols; excluded {unfiltered_candidates - retained_candidates:,}"
        )
        if excluded_asset_reasons:
            reason_text = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(excluded_asset_reasons.items())
            )
            print(f"exclusions: {reason_text}")
        if unclassified_asset_symbols:
            print(
                f"historical/unclassified symbols: {unclassified_asset_symbols:,} "
                f"(policy={args.unclassified_asset_policy})"
            )
    if args.exchange_filter != "all":
        exchange_mask = exchange_universe_mask(
            symbols, security_master, args.exchange_filter
        )
        before_exchange_filter = int(
            len(symbols) - int((symbols == REFERENCE_SYMBOL).sum())
        )
        symbols = symbols[exchange_mask]
        dollar_volume = dollar_volume[:, exchange_mask]
        entry_prices = entry_prices[:, exchange_mask]
        morning_prices = morning_prices[:, exchange_mask]
        entry_staleness = entry_staleness[:, exchange_mask]
        morning_staleness = morning_staleness[:, exchange_mask]
        retained_exchange_candidates = int(
            len(symbols) - int((symbols == REFERENCE_SYMBOL).sum())
        )
        print(
            f"{args.exchange_filter} universe: retained {retained_exchange_candidates:,}/"
            f"{before_exchange_filter:,} candidate symbols before ranking"
        )
    if args.entry_price_source == "nbbo-ask":
        nbbo_path = Path(args.nbbo_path)
        if not nbbo_path.exists():
            parser.error(f"NBBO data does not exist: {nbbo_path}")
        entry_prices, entry_staleness, nbbo_rows = load_scheduled_nbbo_asks(
            nbbo_path, cache_dates, symbols, args.entry_time
        )
        print(
            f"NBBO entries: loaded {len(nbbo_rows):,} causal 15:45 SIP asks from "
            f"{nbbo_path}"
        )
    official_auctions = None
    if auction_path.exists() and (
        args.exchange_filter != "all" or args.exit_price_source == "opening-auction"
    ):
        official_auctions = _official_opening_auctions(
            auction_path, cache_dates, symbols
        )

    execution_exchange_mask = None
    if args.exchange_filter != "all" and official_auctions is not None:
        execution_exchange_mask, known_exchange_sessions = (
            load_primary_auction_exchange_mask(
                auction_path,
                cache_dates,
                symbols,
                args.exchange_filter,
                official=official_auctions,
            )
        )
        print(
            f"historical exchange check: matched {known_exchange_sessions:,} "
            "symbol-sessions from primary auctions"
        )
    if args.exit_price_source == "opening-auction":
        morning_prices = load_opening_auction_prices(
            auction_path, cache_dates, symbols, official=official_auctions
        )
        morning_staleness = np.where(np.isfinite(morning_prices), 0.0, np.inf)
        official_opens = int(np.isfinite(morning_prices).sum())
        print(
            f"auction exits: loaded {official_opens:,} primary opening prices from "
            f"{auction_path}"
        )
    if args.exit_price_source == "nbbo-bid":
        exit_nbbo_path = Path(args.exit_nbbo_path)
        if not exit_nbbo_path.exists():
            parser.error(f"exit NBBO data does not exist: {exit_nbbo_path}")
        morning_prices, morning_staleness, nbbo_exit_rows = load_scheduled_nbbo_prices(
            exit_nbbo_path, cache_dates, symbols, "bid", args.exit_time
        )
        print(f"NBBO exits: loaded {len(nbbo_exit_rows):,} causal SIP bids from {exit_nbbo_path}")
    trades, summary = run_backtest(
        dates=cache_dates,
        symbols=symbols,
        dollar_volume=dollar_volume,
        entry_prices=entry_prices,
        morning_prices=morning_prices,
        entry_staleness=entry_staleness,
        morning_staleness=morning_staleness,
        start_date=requested_start,
        end_date=end_date,
        top=args.top,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        minimum_trading_days=args.min_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        issuers=build_issuer_map(symbols, security_master),
        dedupe_share_classes=args.dedupe_share_classes,
        leverage=args.leverage,
        margin_interest_rate=args.margin_interest_rate or 0.0,
        max_entry_staleness_minutes=args.max_entry_staleness_minutes,
        max_exit_staleness_minutes=args.max_exit_staleness_minutes,
        liquidity_scheme=args.liquidity_scheme,
        entry_minute=args.entry_time,
        exit_minute=args.exit_time,
        share_mode=args.share_mode,
        budget=args.budget,
        entry_price_source=args.entry_price_source,
        exit_price_source=args.exit_price_source,
        execution_exchange_mask=execution_exchange_mask,
        entry_session_mask=entry_session_mask,
        strategy=args.strategy,
        strategy_config=config,
        spy_trend_marks=spy_trend_marks,
        spy_trend_price_source=args.spy_trend_price_source,
    )
    summary["cache_path"] = str(cache_path)
    summary["asset_filter"] = args.asset_filter
    summary["exchange_filter"] = args.exchange_filter
    summary["unclassified_asset_policy"] = args.unclassified_asset_policy
    summary["unclassified_asset_symbols"] = unclassified_asset_symbols
    summary["unfiltered_candidate_symbols"] = unfiltered_candidates
    summary["excluded_asset_reasons"] = excluded_asset_reasons
    summary["candidate_symbols"] = int(
        len(symbols) - int((symbols == REFERENCE_SYMBOL).sum())
    )
    summary["symbol_trade_counts"] = {
        _security_symbol(str(symbol)): int(count)
        for symbol, count in trades.groupby("sample_id").size().items()
    }
    summary["symbol_average_net_return"] = {
        _security_symbol(str(symbol)): float(mean_return)
        for symbol, mean_return in trades.groupby("sample_id")["net_return"].mean().items()
    }
    output_dir = args.output_dir
    if output_dir is None and args.strategy == "liquidity-trend-vol":
        root = Path("/tmp/trading-backtests/candidate") / args.strategy
        root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=f"{requested_start:%Y%m%d}_{end_date:%Y%m%d}_", dir=root))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_csv = args.output_csv or str(output_dir / "trades.csv")
        args.summary_json = args.summary_json or str(output_dir / "summary.json")
        pd.DataFrame(summary["daily_portfolio"]).to_csv(output_dir / "portfolio.csv", index=False)
    summary["run_config"] = {key: str(value) if isinstance(value, (Path, pd.Timestamp)) else value for key, value in vars(args).items()}
    summary["data_metadata"] = metadata
    def fingerprint(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    summary["input_sha256"] = {
        str(path): fingerprint(path)
        for path in dict.fromkeys([
            auction_path, Path(args.security_master_cache),
            daily_data_dir / f"{REFERENCE_SYMBOL}.npy",
            data_dir / f"{REFERENCE_SYMBOL}.npy",
            *([Path(args.nbbo_path)] if args.entry_price_source == "nbbo-ask" else []),
            *([Path(args.exit_nbbo_path)] if args.exit_price_source == "nbbo-bid" else []),
        ]) if path.is_file()
    }
    summary["source_sha256"] = {
        str(path.relative_to(Path(__file__).parents[1])): fingerprint(path)
        for path in Path(__file__).parents[1].rglob("*.py")
    }
    summary["risk_history_start"] = config.risk_history_start if config else None
    if args.strategy == "liquidity-trend-vol":
        from .backtest_audit import minute_mark_audit

        marks, audit = minute_mark_audit(trades, summary, data_dir)
        summary["minute_mark_audit"] = audit
        marks.to_csv(output_dir / "minute_marks.csv", index=False)
        print(f"Minute-open mark drawdown: {audit['minute_open_mark_drawdown']:.2%}")
    print_summary_table(summary)
    if args.show_symbol_trade_frequency:
        print_symbol_trade_counts(trades)

    try:
        from .backtest_plot import write_equity_plot

        summary["plot_path"] = str(write_equity_plot(summary, output_dir=output_dir))
        print(f"wrote equity plot to {summary['plot_path']}")
    except Exception as error:  # noqa: BLE001 - plot failure must not discard numerical results
        LOGGER.warning("could not write equity plot: %s", error)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_csv, index=False)
        print(f"wrote {len(trades)} trades to {output_csv}")
    if args.summary_json:
        summary_json = Path(args.summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
        print(f"wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
