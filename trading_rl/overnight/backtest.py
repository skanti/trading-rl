"""Causal daily-liquidity overnight baseline.

Rank stocks using only completed sessions, buy an equal-weight basket at the
configured afternoon entry, and liquidate it the following morning. Raw daily
liquidity and point-in-time prices are cached so changing the basket size or
EMA span does not rescan the minute files when the required warm-up range is
unchanged. Alpaca-style rankings use cumulative same-session share volume or
trade count through a pre-entry ranking time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ..market_data.calendar import auction_close_minutes
from ..market_data.schema import BAR_INDEX, validate_bar_columns
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
    load_nasdaq_security_master,
    load_primary_auction_exchange_mask,
    reference_session_calendar,
    simulation_symbols,
    historical_window, load_daily_dollar_volume,
)
from .ranking import (
    build_issuer_map, causal_completed_trading_days, liquidity_scores, top_ranked_indices,
)


LOGGER = logging.getLogger(__name__)

MINUTE_PRICE_COLUMNS = {
    f"minute-{field}": BAR_INDEX[f"{field}_mills"]
    for field in ("open", "high", "low", "close", "vwap")
}
ENTRY_PRICE_SOURCES = (*MINUTE_PRICE_COLUMNS, "nbbo-ask")
EXIT_PRICE_SOURCES = (*MINUTE_PRICE_COLUMNS, "opening-auction", "nbbo-bid")

DEFAULT_TRANSACTION_COST_BPS = 1.0

# Reg T governs anything held past the close, so an overnight strategy cannot reach the
# 4x day-trading buying power Alpaca reports as `multiplier`.
MAX_OVERNIGHT_LEVERAGE = 2.0
# Base Reg T maintenance is 25%; brokers raise it on concentrated, volatile books, and
# Alpaca's own requirement on this basket sits near 32%. Used only for reporting.
MAINTENANCE_MARGIN = 0.30
# Alpaca accrues margin interest on a 360-day year, per calendar day.
MARGIN_INTEREST_DIVISOR = 360.0
DEFAULT_NBBO_PATH = "/data/ppv1/updates/alpaca_nbbo_1545_2022-01-01.npz"
DEFAULT_EXIT_NBBO_PATH = "/data/ppv1/updates/alpaca_nbbo_0935_2022-01-01.npz"


def resolve_transaction_cost_bps(
    value: float | None, entry_price_source: str, exit_price_source: str,
) -> float:
    """Default additional costs by source pair, preserving explicit overrides."""
    if value is None:
        value = (
            0.0
            if (entry_price_source, exit_price_source) == ("nbbo-ask", "opening-auction")
            else DEFAULT_TRANSACTION_COST_BPS
        )
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    return float(value)


def load_opening_auction_prices(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    *,
    official: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build a date-by-symbol primary-opening matrix from adjusted NPZ prices."""
    if official is None:
        official = _official_opening_auctions(path, dates, symbols)
    prices = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    rows = pd.Index(dates).get_indexer(pd.DatetimeIndex(official["date"]))
    symbol_index = pd.Index([_security_symbol(sample_id) for sample_id in symbols])
    columns = symbol_index.get_indexer(official["symbol"].astype(str).str.upper())
    valid = (rows >= 0) & (columns >= 0)
    prices[rows[valid], columns[valid]] = official["price"].to_numpy(dtype=np.float64)[
        valid
    ]
    return prices


def load_scheduled_nbbo_prices(
    path: Path,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    side: str,
    target_minute: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one quote side at the configured clock; missing rows remain unavailable."""
    if side not in {"bid", "ask"}:
        raise ValueError("NBBO side must be bid or ask")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"NBBO data must use the split-adjusted NPZ format: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "symbol",
            "date",
            "target_timestamp",
            "timestamp",
            f"{side}_price",
            f"raw_{side}_price",
            f"{side}_exchange",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing NBBO arrays: {sorted(missing)}")
        if "split_adjusted" not in data.files or not bool(
            data["split_adjusted"].item()
        ):
            raise ValueError(f"{path} does not contain pre-adjusted NBBO prices")
        symbol_values = np.char.upper(
            np.asarray(data["symbol"]).astype(str, copy=False)
        )
        date_values = np.asarray(data["date"], dtype="datetime64[D]")
        # SIP quotes mix whole seconds and fractional precision up to nanoseconds.
        # Infer neither column's format from its first row: all are ISO 8601.
        targets = pd.to_datetime(
            np.asarray(data["target_timestamp"]).astype(str, copy=False),
            format="ISO8601", utc=True,
        )
        stamps = pd.to_datetime(
            np.asarray(data["timestamp"]).astype(str, copy=False),
            format="ISO8601", utc=True,
        )
        if targets.hasnans or stamps.hasnans:
            raise ValueError(f"{path} contains missing NBBO timestamps")
        quote_prices = np.asarray(data[f"{side}_price"], dtype=np.float64)
        raw_prices = np.asarray(data[f"raw_{side}_price"], dtype=np.float64)
        exchanges = np.asarray(data[f"{side}_exchange"]).astype(str, copy=False)
        wanted_dates = np.asarray(pd.DatetimeIndex(dates), dtype="datetime64[D]")
        wanted_symbols = np.asarray(
            [_security_symbol(sample_id) for sample_id in symbols], dtype=str
        )
        selected = (
            np.isin(date_values, wanted_dates)
            & np.isin(symbol_values, wanted_symbols)
            & np.isfinite(quote_prices)
            & (quote_prices > 0.0)
        )
        local_targets = targets[selected].tz_convert("America/New_York")
        if (
            (target_minute is not None and
             ((local_targets.hour * 60 + local_targets.minute) != target_minute).any())
            or (local_targets.second != 0).any()
            or (local_targets.microsecond != 0).any()
            or (local_targets.nanosecond != 0).any()
            or not np.array_equal(
                local_targets.tz_localize(None).normalize().to_numpy(dtype="datetime64[D]"),
                date_values[selected],
            )
        ):
            raise ValueError(f"{path} NBBO target timestamps do not match the configured time/date")
        if (stamps[selected] > targets[selected]).any():
            raise ValueError(f"{path} contains a post-target NBBO quote")
        rows = pd.DataFrame(
            {
                "symbol": symbol_values[selected],
                "date": pd.to_datetime(date_values[selected]),
                "price": quote_prices[selected],
                "raw_price": raw_prices[selected],
                "staleness_minutes": (
                    (targets[selected] - stamps[selected]).total_seconds() / 60.0
                ),
                "timestamp": stamps[selected],
                "target_timestamp": targets[selected],
                "exchange": exchanges[selected],
            }
        )
    if rows.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{path} contains duplicate symbol-date NBBO snapshots")
    prices = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    staleness = np.full_like(prices, np.inf)
    row_indices = pd.Index(dates).get_indexer(pd.DatetimeIndex(rows["date"]))
    symbol_index = pd.Index(
        [_security_symbol(sample_id) for sample_id in symbols]
    )
    columns = symbol_index.get_indexer(rows["symbol"])
    valid = (row_indices >= 0) & (columns >= 0)
    prices[row_indices[valid], columns[valid]] = rows["price"].to_numpy()[valid]
    staleness[row_indices[valid], columns[valid]] = rows[
        "staleness_minutes"
    ].to_numpy()[valid]
    return prices, staleness, rows


def load_scheduled_nbbo_asks(
    path: Path, dates: pd.DatetimeIndex, symbols: np.ndarray,
    target_minute: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return load_scheduled_nbbo_prices(path, dates, symbols, "ask", target_minute)


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
        "periods": int(len(values)),
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


def select_strategy_basket(
    symbols: np.ndarray,
    scores: np.ndarray,
    completed_days: np.ndarray,
    entry_prices: np.ndarray,
    entry_staleness: np.ndarray,
    *,
    top: int,
    minimum_trading_days: int,
    max_entry_staleness_minutes: int,
    entry_price_source: str,
    issuers: Mapping[str, str] | None = None,
    dedupe_share_classes: bool = True,
    execution_exchange_mask: np.ndarray | None = None,
    reference_symbol: str = REFERENCE_SYMBOL,
) -> np.ndarray:
    """Select intended members before execution availability or sizing drops trades."""
    eligible = symbols != str(reference_symbol)
    if execution_exchange_mask is not None:
        eligible &= execution_exchange_mask
    # Sparse NBBO coverage must never replace an intended member with a lower rank.
    if entry_price_source != "nbbo-ask":
        eligible &= (
            np.isfinite(entry_prices) & (entry_prices > 0.0)
            & (entry_staleness <= max_entry_staleness_minutes)
        )
    return top_ranked_indices(
        scores, symbols, top, completed_days=completed_days,
        minimum_trading_days=minimum_trading_days, eligible_mask=eligible,
        issuers=issuers if dedupe_share_classes else None,
    )


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
    spy_overnight = summary["spy_overnight_metrics"]
    spy_buy_hold = summary["spy_buy_and_hold_metrics"]
    if not all(isinstance(metrics, dict) for metrics in (strategy, spy_overnight, spy_buy_hold)):
        raise TypeError("summary metric groups must be mappings")

    comparison = Table(title="Overnight liquidity baseline", show_header=True, header_style="bold")
    comparison.add_column("Metric")
    comparison.add_column(f"Top {summary['top']}", justify="right")
    comparison.add_column("SPY overnight", justify="right")
    comparison.add_column("SPY buy & hold", justify="right")
    strategy_trades = int(summary["trades"])
    spy_overnight_trades = int(spy_overnight["periods"])
    trade_counts = (strategy_trades, spy_overnight_trades, int(spy_buy_hold["periods"] > 0))
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
            float(spy_overnight[key]) * scale,
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
    if summary.get("budget") is not None:
        details.add_row(
            "Position sizing",
            f"{summary['share_mode']} shares from ${float(summary['budget']):,.0f} initial equity; "
            f"ending ${float(summary['ending_equity']):,.2f}; "
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


def basket_quantities(
    entry_prices: np.ndarray,
    budget: float,
    share_mode: str = "fractional",
) -> np.ndarray:
    """Size one equal-notional basket in fractional or whole shares.

    Whole-share sizing deliberately rounds each name down independently. This never
    exceeds the budget and never redistributes a costly name's unused allocation into
    cheaper names, which would change the strategy's intended equal weighting. A stock
    priced above its per-name target therefore receives zero shares and its allocation
    remains cash.
    """
    prices = np.asarray(entry_prices, dtype=np.float64)
    if prices.ndim != 1 or not prices.size:
        raise ValueError("entry_prices must be a non-empty one-dimensional array")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("entry_prices must be finite and positive")
    if not np.isfinite(budget) or float(budget) <= 0.0:
        raise ValueError("budget must be finite and positive")
    if share_mode not in {"fractional", "whole"}:
        raise ValueError("share_mode must be 'fractional' or 'whole'")

    target_notional = float(budget) / prices.size
    quantities = target_notional / prices
    return np.floor(quantities) if share_mode == "whole" else quantities


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
    margin_interest_rate: float = 0.0,
    maintenance_margin: float = MAINTENANCE_MARGIN,
    reference_symbol: str = REFERENCE_SYMBOL,
    share_mode: str = "fractional",
    budget: float | None = None,
    entry_price_source: str = "minute-open",
    exit_price_source: str = "opening-auction",
    execution_exchange_mask: np.ndarray | None = None,
    entry_session_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if transaction_cost_bps < 0.0:
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
    scores = liquidity_scores(dollar_volume, liquidity_scheme, ema_span, min_history_days)
    completed_trading_days = causal_completed_trading_days(dollar_volume)
    if execution_exchange_mask is not None:
        execution_exchange_mask = np.asarray(execution_exchange_mask, dtype=bool)
        if execution_exchange_mask.shape != scores.shape:
            raise ValueError("execution_exchange_mask must match the liquidity arrays")
    interval_entries = (dates >= start_date) & (dates < end_date)
    if entry_session_mask is None:
        entry_session_mask = np.ones(len(dates), dtype=bool)
    else:
        entry_session_mask = np.asarray(entry_session_mask, dtype=bool)
        if entry_session_mask.shape != (len(dates),):
            raise ValueError("entry_session_mask must match dates")
    skipped_short_entries = int((interval_entries & ~entry_session_mask).sum())
    entries = np.flatnonzero(interval_entries & entry_session_mask)
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
    spy_returns: list[float] = []
    spy_exit_prices: list[float] = []
    current_equity = simulation_budget
    for date_index in entries:
        selected = select_strategy_basket(
            symbols, scores[date_index], completed_trading_days[date_index],
            entry_prices[date_index], entry_staleness[date_index],
            top=top, minimum_trading_days=int(minimum_trading_days),
            max_entry_staleness_minutes=int(max_entry_staleness_minutes),
            entry_price_source=entry_price_source, issuers=issuers,
            dedupe_share_classes=dedupe_share_classes, reference_symbol=reference_symbol,
            execution_exchange_mask=(execution_exchange_mask[date_index + 1] if execution_exchange_mask is not None else None),
        )
        selected_ranks = np.arange(1, int(top) + 1)
        selected_entries = entry_prices[date_index, selected]
        selected_entry_staleness = entry_staleness[date_index, selected]
        exits = morning_prices[date_index + 1, selected]
        exit_staleness = morning_staleness[date_index + 1, selected]
        missing_entry = (
            ~np.isfinite(selected_entries) | (selected_entries <= 0.0)
            | ~np.isfinite(selected_entry_staleness)
            | (selected_entry_staleness > entry_age_limit)
        )
        missing_exit = (
            ~np.isfinite(exits) | (exits <= 0.0)
            | ~np.isfinite(exit_staleness) | (exit_staleness > exit_age_limit)
        )
        missing = missing_entry | missing_exit
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
                selected_entries[valid], session_budget * (valid.sum() / int(top)), share_mode
            )
        executed = quantities > 0.0
        skipped = int((~executed).sum())
        if not executed.any() and not missing.any():
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
        session_net_pnl = float(net_pnl.sum())
        session_return = session_net_pnl / session_budget
        current_equity = session_budget + session_net_pnl
        session_records.append({
            "entry_date": str(dates[date_index].date()),
            "exit_date": str(dates[date_index + 1].date()),
            "portfolio_start_equity": session_budget,
            "portfolio_end_equity": current_equity,
            "portfolio_return": session_return,
            "capital_deployed": float(entry_notional.sum()),
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
                    "target_notional": session_budget / int(top),
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
        spy_entry = entry_prices[date_index, reference_index]
        spy_exit = morning_prices[date_index + 1, reference_index]
        if (
            not np.isfinite(spy_entry) or spy_entry <= 0.0
            or not np.isfinite(spy_exit) or spy_exit <= 0.0
            or not np.isfinite(entry_staleness[date_index, reference_index])
            or not np.isfinite(morning_staleness[date_index + 1, reference_index])
            or entry_staleness[date_index, reference_index] > entry_age_limit
            or morning_staleness[date_index + 1, reference_index] > exit_age_limit
        ):
            day = str(dates[date_index].date())
            missing_benchmark_dates.append(day)
            LOGGER.warning("%s benchmark unavailable for %s -> %s: missing or stale price",
                           reference_symbol, day, dates[date_index + 1].date())
            spy_returns.append(float("nan"))
            spy_exit_prices.append(float("nan"))
        else:
            spy_returns.append(float(spy_exit / spy_entry - 1.0 - cost))
            spy_exit_prices.append(float(spy_exit))

    trades = pd.concat(pieces, ignore_index=True)
    sessions = pd.DataFrame(session_records).set_index("entry_date")
    deployed_by_entry = sessions.capital_deployed
    utilization_by_entry = deployed_by_entry / sessions.portfolio_start_equity
    unlevered_daily = sessions.portfolio_return

    # Leverage is not a free scalar. Scaling returns alone leaves the Sharpe ratio
    # unchanged, so it would say nothing. What makes it a real trade-off is the borrow
    # cost and the compounding of a deeper drawdown. Per-trade returns in `trades` stay
    # unlevered; only the portfolio series is levered.
    #
    # Alpaca accrues margin interest on the overnight debit balance as
    # rate / 360 per CALENDAR day, so a Friday entry held to Monday is charged three
    # days, not one. Deriving the accrual from each trade's actual entry-to-exit span
    # captures weekends and holidays instead of assuming a flat trading-day divisor.
    exit_by_entry = sessions.exit_date
    holding_days = pd.Series(
        (pd.to_datetime(exit_by_entry.values) - pd.to_datetime(exit_by_entry.index)).days,
        index=exit_by_entry.index,
        dtype=np.float64,
    ).clip(lower=1.0)
    borrow_per_session = (
        max(0.0, leverage - 1.0)
        * float(margin_interest_rate)
        * holding_days
        * utilization_by_entry
        / MARGIN_INTEREST_DIVISOR
    )
    daily = float(leverage) * unlevered_daily - borrow_per_session
    spy_daily = pd.Series(spy_returns, index=daily.index, dtype=np.float64)
    difference = daily - spy_daily
    spy_buy_hold_equity = np.asarray(spy_exit_prices, dtype=np.float64) / float(
        entry_prices[entries[0], reference_index]
    )
    spy_buy_hold_returns = np.empty_like(spy_buy_hold_equity)
    spy_buy_hold_returns[0] = spy_buy_hold_equity[0] - 1.0
    spy_buy_hold_returns[1:] = spy_buy_hold_equity[1:] / spy_buy_hold_equity[:-1] - 1.0
    side_cost = float(transaction_cost_bps) / 10_000.0
    spy_buy_hold_returns[0] -= side_cost
    spy_buy_hold_returns[-1] -= side_cost
    spy_buy_hold_daily = pd.Series(spy_buy_hold_returns, index=daily.index, dtype=np.float64)
    difference_buy_hold = daily - spy_buy_hold_daily
    by_date = {day: set(group.sample_id) for day, group in trades.groupby("entry_date", sort=True)}
    memberships = [by_date.get(day, set()) for day in sessions.index]
    replacements = [
        len(current - previous) for previous, current in zip(memberships, memberships[1:])
    ]
    retentions = [
        len(current & previous) / len(current) if current else 1.0
        for previous, current in zip(memberships, memberships[1:])
    ]
    jaccards = [
        len(current & previous) / len(current | previous) if current | previous else 1.0
        for previous, current in zip(memberships, memberships[1:])
    ]
    # A levered book is force-liquidated when equity / position value falls through the
    # maintenance requirement. Report the worst session against that floor rather than
    # silently producing an equity curve the broker would never have let you hold.
    worst_session = float(unlevered_daily.min())
    if leverage > 1.0 and worst_session < 0.0:
        drop = abs(worst_session)
        margin_ratio = (1.0 - leverage * drop) / (leverage * (1.0 - drop))
        breach_leverage = 1.0 / (float(maintenance_margin) * (1.0 - drop) + drop)
    else:
        margin_ratio = 1.0
        breach_leverage = float("inf")

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
        "strategy": f"causal_{liquidity_scheme}_overnight_long",
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
        "trades": int(len(trades)),
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
        "spy_overnight_metrics": _benchmark_metrics(spy_daily),
        "spy_buy_and_hold_metrics": _benchmark_metrics(spy_buy_hold_daily),
        "versus_spy": {
            "mean_excess_return": float(difference.mean()),
            "median_excess_return": float(difference.median()),
            "outperformance_days": int(difference.gt(0.0).sum()),
            "underperformance_days": int(difference.lt(0.0).sum()),
            "daily_return_correlation": float(daily.corr(spy_daily)),
        },
        "versus_spy_buy_and_hold": {
            "mean_excess_return": float(difference_buy_hold.mean()),
            "median_excess_return": float(difference_buy_hold.median()),
            "outperformance_days": int(difference_buy_hold.gt(0.0).sum()),
            "underperformance_days": int(difference_buy_hold.lt(0.0).sum()),
            "daily_return_correlation": float(daily.corr(spy_buy_hold_daily)),
        },
    }
    return trades, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the causal overnight basket used by live trading."
        )
    )
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
        default="minute-open",
        help="minute-* selects a field of the bar starting at --entry-time; non-open "
        "fields model hypothetical fills over that minute, known only at its end. "
        "nbbo-ask uses the latest causal SIP ask at 15:45 (default: minute-open)",
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
        help="additional cost in bps per side (default: 0.0 for nbbo-ask → opening-auction; 1.0 otherwise)",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.transaction_cost_bps = resolve_transaction_cost_bps(
            args.transaction_cost_bps, args.entry_price_source, args.exit_price_source,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.leverage < 1.0:
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
    print_summary_table(summary)
    if args.show_symbol_trade_frequency:
        print_symbol_trade_counts(trades)

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
