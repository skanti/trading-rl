"""Causal daily-liquidity overnight baseline.

Rank stocks using only completed sessions, buy an equal-weight basket at the
configured afternoon entry, and liquidate it the following morning. Raw daily
liquidity and point-in-time prices are cached so changing the basket size or
EMA span does not rescan the minute files when the required warm-up range is
unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from classify_evaluate import parse_anchor_time
from week_dataset import forward_fill_positions


REFERENCE_SYMBOL = "ST-SPY"
EXTENDED_OPEN_MINUTE = 4 * 60
DEFAULT_DAYS_PATH = "/data/ppv1/updates/full_2026-08-22_10d.csv"
DEFAULT_DATA_DIR = "/data/ppv1/updates/full_2026-08-22"


def causal_ema_log_liquidity(
    dollar_volume: np.ndarray,
    span: int,
    min_history_days: int,
) -> np.ndarray:
    """Return lagged EMA scores; row ``t`` can only use rows before ``t``.

    The EMA is applied to ``log1p(dollar_volume)``. This preserves liquidity
    ordering while preventing one exceptional volume day from dominating the
    ranking for weeks. Missing sessions decay an already-started EMA toward
    zero and do not count toward the minimum observed history.
    """
    values = np.asarray(dollar_volume, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("dollar_volume must have shape (dates, symbols)")
    if int(span) < 1:
        raise ValueError("span must be positive")
    if int(min_history_days) < 1:
        raise ValueError("min_history_days must be positive")
    if np.isfinite(values).any() and np.nanmin(values) < 0.0:
        raise ValueError("dollar volume must be non-negative")

    alpha = 2.0 / (int(span) + 1.0)
    dates, symbols = values.shape
    scores = np.full((dates, symbols), np.nan, dtype=np.float64)
    ema = np.zeros(symbols, dtype=np.float64)
    observations = np.zeros(symbols, dtype=np.int32)
    started = np.zeros(symbols, dtype=bool)
    for date_index in range(dates):
        eligible = observations >= int(min_history_days)
        scores[date_index, eligible] = ema[eligible]

        observed = np.isfinite(values[date_index]) & (values[date_index] > 0.0)
        transformed = np.zeros(symbols, dtype=np.float64)
        transformed[observed] = np.log1p(values[date_index, observed])
        continuing = started | observed
        first = observed & ~started
        ema[continuing] = (1.0 - alpha) * ema[continuing] + alpha * transformed[continuing]
        ema[first] = transformed[first]
        observations[observed] += 1
        started |= observed
    return scores


def top_liquid_indices(
    scores: np.ndarray,
    entry_prices: np.ndarray,
    top: int,
    symbols: np.ndarray,
) -> np.ndarray:
    """Select the highest causal scores with a price available at entry."""
    score = np.asarray(scores, dtype=np.float64)
    prices = np.asarray(entry_prices, dtype=np.float64)
    names = np.asarray(symbols)
    if score.ndim != 1 or prices.shape != score.shape or names.shape != score.shape:
        raise ValueError("scores, entry_prices, and symbols must be matching vectors")
    if int(top) < 1:
        raise ValueError("top must be positive")
    eligible = np.flatnonzero(np.isfinite(score) & np.isfinite(prices) & (prices > 0.0))
    if eligible.size < int(top):
        raise ValueError(f"only {eligible.size} causally eligible symbols are available for top={top}")
    if eligible.size > int(top):
        local = np.argpartition(score[eligible], -int(top))[-int(top) :]
        eligible = eligible[local]
    # A lexical secondary key makes exact score ties reproducible.
    order = np.lexsort((names[eligible], -score[eligible]))
    return eligible[order]


def _parse_clock(value: str) -> int:
    return parse_anchor_time(value)


def _calendar(days: pd.DataFrame) -> pd.DataFrame:
    calendar = days.loc[:, ["date", "context_sod_sec"]].drop_duplicates()
    if calendar.date.duplicated().any():
        raise ValueError("symbols disagree about session start timestamps")
    return calendar.sort_values("date").reset_index(drop=True)


def _cache_metadata(
    days_path: Path,
    data_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    entry_minute: int,
    exit_minute: int,
) -> dict[str, object]:
    stat = days_path.stat()
    return {
        "version": 2,
        "days_path": str(days_path.resolve()),
        "days_size": int(stat.st_size),
        "days_mtime_ns": int(stat.st_mtime_ns),
        "data_dir": str(data_dir.resolve()),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "entry_minute": int(entry_minute),
        "exit_minute": int(exit_minute),
    }


def _symbol_daily_arrays(
    sample_id: str,
    rows: pd.DataFrame,
    date_positions: dict[pd.Timestamp, int],
    context_sod: np.ndarray,
    data_dir: Path,
    entry_minute: int,
    exit_minute: int,
    date_count: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    liquidity = np.full(date_count, np.nan, dtype=np.float64)
    entry_prices = np.full(date_count, np.nan, dtype=np.float64)
    morning_prices = np.full(date_count, np.nan, dtype=np.float64)
    entry_staleness = np.full(date_count, np.inf, dtype=np.float64)
    morning_staleness = np.full(date_count, np.inf, dtype=np.float64)
    source_path = data_dir / f"{sample_id}.npy"
    if not source_path.exists():
        return (
            sample_id,
            liquidity,
            entry_prices,
            morning_prices,
            entry_staleness,
            morning_staleness,
        )

    ordered = rows.sort_values("date")
    positions = np.array([date_positions[pd.Timestamp(date)] for date in ordered.date], dtype=np.int64)
    starts = ordered.sod_idx.to_numpy(dtype=np.int64)
    ends = ordered.eod_idx.to_numpy(dtype=np.int64)
    valid_ranges = (starts >= 0) & (ends >= starts)
    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"{sample_id} must contain seconds, price_mills, and volume")
    valid_ranges &= ends < len(source)
    if valid_ranges.any():
        low = int(starts[valid_ranges].min())
        high = int(ends[valid_ranges].max())
        block = np.asarray(source[low : high + 1, 1:3], dtype=np.float64)
        dollar_volume = np.where(
            np.isfinite(block[:, 0]) & (block[:, 0] > 0.0) & np.isfinite(block[:, 1]) & (block[:, 1] > 0.0),
            block[:, 0] * block[:, 1] / 1000.0,
            0.0,
        )
        prefix = np.concatenate(([0.0], np.cumsum(dollar_volume, dtype=np.float64)))
        local_start = starts[valid_ranges] - low
        local_end = ends[valid_ranges] - low + 1
        liquidity[positions[valid_ranges]] = prefix[local_end] - prefix[local_start]

    # Prices are requested for every exchange session, including a selected
    # stock's first missing session. Keeping the last observable mark is causal;
    # the caller applies separate entry and exit staleness limits.
    session_starts = context_sod
    entry_secs = session_starts + (int(entry_minute) - EXTENDED_OPEN_MINUTE) * 60
    morning_secs = session_starts + (int(exit_minute) - EXTENDED_OPEN_MINUTE) * 60
    requested = np.concatenate((entry_secs, morning_secs))
    source_seconds = np.asarray(source[:, 0])
    available = np.searchsorted(source_seconds, requested, side="right") > 0
    prices = np.full(len(requested), np.nan, dtype=np.float64)
    staleness = np.full(len(requested), np.inf, dtype=np.float64)
    if available.any():
        price_positions = forward_fill_positions(source, requested[available], sample_id)
        observed_seconds = np.asarray(source[price_positions, 0], dtype=np.int64)
        observed_prices = np.asarray(source[price_positions, 1], dtype=np.float64) / 1000.0
        valid_price = np.isfinite(observed_prices) & (observed_prices > 0.0)
        target_positions = np.flatnonzero(available)
        prices[target_positions[valid_price]] = observed_prices[valid_price]
        staleness[target_positions[valid_price]] = (
            requested[target_positions[valid_price]] - observed_seconds[valid_price]
        ) / 60.0
    split = date_count
    entry = prices[:split]
    morning = prices[split:]
    entry_prices[:] = entry
    morning_prices[:] = morning
    entry_staleness[:] = staleness[:split]
    morning_staleness[:] = staleness[split:]
    return (
        sample_id,
        liquidity,
        entry_prices,
        morning_prices,
        entry_staleness,
        morning_staleness,
    )


def build_daily_cache(
    days: pd.DataFrame,
    data_dir: Path,
    dates: pd.DatetimeIndex,
    entry_minute: int,
    exit_minute: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan each symbol once and build dense date-by-symbol arrays."""
    selected_days = days[days.date.isin(dates)].copy()
    symbols = np.array(sorted(selected_days.sample_id.unique()), dtype=str)
    date_positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    context = _calendar(selected_days).set_index("date").reindex(dates).context_sod_sec
    if context.isna().any():
        raise ValueError("one or more cache dates are missing session timestamps")
    context_sod = context.to_numpy(dtype=np.int64)
    grouped = {str(symbol): group for symbol, group in selected_days.groupby("sample_id", sort=False)}
    liquidity = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    entry_prices = np.full_like(liquidity, np.nan)
    morning_prices = np.full_like(liquidity, np.nan)
    entry_staleness = np.full_like(liquidity, np.inf)
    morning_staleness = np.full_like(liquidity, np.inf)

    def submit(symbol: str):
        return _symbol_daily_arrays(
            symbol,
            grouped[symbol],
            date_positions,
            context_sod,
            data_dir,
            entry_minute,
            exit_minute,
            len(dates),
        )

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(submit, str(symbol)): index for index, symbol in enumerate(symbols)}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="building daily liquidity cache", unit="symbol"
        ):
            column = futures[future]
            (
                _,
                symbol_liquidity,
                symbol_entry,
                symbol_morning,
                symbol_entry_staleness,
                symbol_morning_staleness,
            ) = future.result()
            liquidity[:, column] = symbol_liquidity
            entry_prices[:, column] = symbol_entry
            morning_prices[:, column] = symbol_morning
            entry_staleness[:, column] = symbol_entry_staleness
            morning_staleness[:, column] = symbol_morning_staleness
    return symbols, liquidity, entry_prices, morning_prices, entry_staleness, morning_staleness


def load_or_build_cache(
    cache_path: Path,
    metadata: dict[str, object],
    days: pd.DataFrame,
    data_dir: Path,
    dates: pd.DatetimeIndex,
    entry_minute: int,
    exit_minute: int,
    workers: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        days,
        data_dir,
        dates,
        entry_minute,
        exit_minute,
        workers,
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
    return {
        "periods": int(len(values)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "win_rate": float(values.gt(0.0).mean()),
        "profit_factor": _profit_factor(values),
        "annualized_volatility": volatility * np.sqrt(252.0),
        "sharpe_zero_cash_rate": float(np.sqrt(252.0) * values.mean() / volatility)
        if volatility > 0.0
        else float("nan"),
        "max_drawdown": float(drawdown.max()),
    }


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
    transaction_cost_bps: float,
    max_entry_staleness_minutes: int,
    max_exit_staleness_minutes: int,
    entry_minute: int = 15 * 60 + 55,
    exit_minute: int = 9 * 60 + 45,
    reference_symbol: str = REFERENCE_SYMBOL,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative")
    scores = causal_ema_log_liquidity(dollar_volume, ema_span, min_history_days)
    entries = np.flatnonzero((dates >= start_date) & (dates < end_date))
    if not entries.size:
        raise ValueError("the requested interval contains no entry sessions")
    if entries[-1] + 1 >= len(dates):
        raise ValueError("the final entry session has no following exit session")
    reference_matches = np.flatnonzero(symbols == str(reference_symbol))
    if reference_matches.size != 1:
        raise ValueError(f"cache must contain exactly one {reference_symbol}")
    reference_index = int(reference_matches[0])
    stock_mask = symbols != str(reference_symbol)
    stock_indices = np.flatnonzero(stock_mask)
    stock_symbols = symbols[stock_mask]
    cost = 2.0 * float(transaction_cost_bps) / 10_000.0

    pieces: list[pd.DataFrame] = []
    spy_returns: list[float] = []
    for date_index in entries:
        executable_entries = np.where(
            entry_staleness[date_index, stock_mask] <= int(max_entry_staleness_minutes),
            entry_prices[date_index, stock_mask],
            np.nan,
        )
        selected_local = top_liquid_indices(
            scores[date_index, stock_mask],
            executable_entries,
            top,
            stock_symbols,
        )
        selected = stock_indices[selected_local]
        exits = morning_prices[date_index + 1, selected]
        exit_staleness = morning_staleness[date_index + 1, selected]
        missing_exit = (
            ~np.isfinite(exits)
            | (exits <= 0.0)
            | (exit_staleness > int(max_exit_staleness_minutes))
        )
        if missing_exit.any():
            missing = stock_symbols[selected_local[missing_exit]]
            raise ValueError(
                f"selected symbols lack a fresh next-session exit on {dates[date_index].date()}: "
                + ", ".join(missing.tolist())
            )
        gross = exits / entry_prices[date_index, selected] - 1.0
        pieces.append(
            pd.DataFrame(
                {
                    "entry_date": str(dates[date_index].date()),
                    "exit_date": str(dates[date_index + 1].date()),
                    "rank": np.arange(1, int(top) + 1),
                    "sample_id": symbols[selected],
                    "liquidity_score": scores[date_index, selected],
                    "entry_price": entry_prices[date_index, selected],
                    "exit_price": exits,
                    "entry_staleness_minutes": entry_staleness[date_index, selected],
                    "exit_staleness_minutes": exit_staleness,
                    "gross_return": gross,
                    "transaction_cost": cost,
                    "net_return": gross - cost,
                }
            )
        )
        spy_entry = entry_prices[date_index, reference_index]
        spy_exit = morning_prices[date_index + 1, reference_index]
        if (
            not np.isfinite(spy_entry)
            or not np.isfinite(spy_exit)
            or entry_staleness[date_index, reference_index] > int(max_entry_staleness_minutes)
            or morning_staleness[date_index + 1, reference_index] > int(max_exit_staleness_minutes)
        ):
            raise ValueError(f"{reference_symbol} lacks a fresh entry or exit on {dates[date_index].date()}")
        spy_returns.append(float(spy_exit / spy_entry - 1.0 - cost))

    trades = pd.concat(pieces, ignore_index=True)
    daily = trades.groupby("entry_date", sort=True).net_return.mean()
    spy_daily = pd.Series(spy_returns, index=daily.index, dtype=np.float64)
    difference = daily - spy_daily
    summary: dict[str, object] = {
        "strategy": "causal_liquidity_ema_overnight_long",
        "entry_time_eastern": f"{entry_minute // 60:02d}:{entry_minute % 60:02d}",
        "exit_time_eastern": f"{exit_minute // 60:02d}:{exit_minute % 60:02d} next trading session",
        "first_entry_date": str(pd.Timestamp(daily.index[0]).date()),
        "last_entry_date": str(pd.Timestamp(daily.index[-1]).date()),
        "last_exit_date": str(trades.exit_date.iloc[-1]),
        "top": int(top),
        "liquidity_metric": "lagged EMA of log1p(completed regular-session dollar volume)",
        "ema_span_sessions": int(ema_span),
        "minimum_liquidity_history_sessions": int(min_history_days),
        "transaction_cost_bps_per_side": float(transaction_cost_bps),
        "max_entry_staleness_minutes": int(max_entry_staleness_minutes),
        "max_exit_staleness_minutes": int(max_exit_staleness_minutes),
        "stale_exit_marks_over_10_minutes": int(trades.exit_staleness_minutes.gt(10.0).sum()),
        "maximum_exit_staleness_minutes": float(trades.exit_staleness_minutes.max()),
        "trades": int(len(trades)),
        "strategy_metrics": strategy_metrics(daily),
        "spy_overnight_metrics": strategy_metrics(spy_daily),
        "versus_spy": {
            "mean_excess_return": float(difference.mean()),
            "median_excess_return": float(difference.median()),
            "outperformance_days": int(difference.gt(0.0).sum()),
            "underperformance_days": int(difference.lt(0.0).sum()),
            "daily_return_correlation": float(daily.corr(spy_daily)),
        },
    }
    return trades, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trade the causally most-liquid stocks overnight, ranked by prior-session EMA dollar volume."
        )
    )
    parser.add_argument("--top", type=int, default=100, help="daily basket size, e.g. 50 or 100")
    parser.add_argument("--months", type=int, default=12, help="trailing calendar months")
    parser.add_argument("--start-date", default=None, help="optional YYYY-MM-DD override for --months")
    parser.add_argument("--end-date", default=None, help="final exit date, default: latest data date")
    parser.add_argument("--ema-span", type=int, default=20, help="liquidity EMA span; 1 uses only the prior day")
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument("--entry-time", type=_parse_clock, default=_parse_clock("15:55"))
    parser.add_argument("--exit-time", type=_parse_clock, default=_parse_clock("09:45"))
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0, help="cost per side")
    parser.add_argument("--max-entry-staleness-minutes", type=int, default=10)
    parser.add_argument(
        "--max-exit-staleness-minutes",
        type=int,
        default=24 * 60,
        help="maximum age of the causal exit mark; avoids dropping a selected name with lookahead",
    )
    parser.add_argument("--days-path", default=DEFAULT_DAYS_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", default="/tmp/trading/baseline_cache")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    if args.months < 1 or args.ema_span < 1 or args.min_history_days < 1:
        parser.error("months, ema-span, and min-history-days must be positive")
    if (
        args.workers < 1
        or args.max_entry_staleness_minutes < 0
        or args.max_exit_staleness_minutes < 0
    ):
        parser.error("workers must be positive and staleness must be non-negative")

    days_path = Path(args.days_path)
    data_dir = Path(args.data_dir)
    days = pd.read_csv(
        days_path,
        usecols=["sample_id", "date", "sod_idx", "eod_idx", "context_sod_sec"],
    )
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    calendar = _calendar(days)
    all_dates = pd.DatetimeIndex(calendar.date)
    requested_end = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp(all_dates[-1])
    eligible_end = all_dates[all_dates <= requested_end]
    if eligible_end.empty:
        parser.error("end-date precedes the local dataset")
    end_date = pd.Timestamp(eligible_end[-1])
    requested_start = (
        pd.Timestamp(args.start_date)
        if args.start_date
        else end_date - pd.DateOffset(months=int(args.months))
    )
    first_entry_candidates = np.flatnonzero(all_dates >= requested_start)
    if not first_entry_candidates.size:
        parser.error("start-date follows the local dataset")
    first_entry_index = int(first_entry_candidates[0])
    end_index = int(np.searchsorted(all_dates.to_numpy(), np.datetime64(end_date)))
    if first_entry_index >= end_index:
        parser.error("the requested interval must contain an entry and a later exit session")
    warmup = max(3 * int(args.ema_span), int(args.min_history_days) + 1)
    scan_start_index = max(0, first_entry_index - warmup)
    cache_dates = all_dates[scan_start_index : end_index + 1]
    cache_days = days[days.date.between(cache_dates[0], cache_dates[-1])]

    metadata = _cache_metadata(
        days_path,
        data_dir,
        pd.Timestamp(cache_dates[0]),
        pd.Timestamp(cache_dates[-1]),
        args.entry_time,
        args.exit_time,
    )
    cache_name = (
        f"liquidity_{cache_dates[0]:%Y%m%d}_{cache_dates[-1]:%Y%m%d}_"
        f"e{args.entry_time:04d}_x{args.exit_time:04d}_v2.npz"
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
        cache_days,
        data_dir,
        cache_dates,
        args.entry_time,
        args.exit_time,
        args.workers,
        args.rebuild_cache,
    )
    trades, summary = run_backtest(
        cache_dates,
        symbols,
        dollar_volume,
        entry_prices,
        morning_prices,
        entry_staleness,
        morning_staleness,
        requested_start,
        end_date,
        args.top,
        args.ema_span,
        args.min_history_days,
        args.transaction_cost_bps,
        args.max_entry_staleness_minutes,
        args.max_exit_staleness_minutes,
        args.entry_time,
        args.exit_time,
    )
    summary["cache_path"] = str(cache_path)
    summary["candidate_symbols"] = int(len(symbols) - int((symbols == REFERENCE_SYMBOL).sum()))
    print(json.dumps(summary, indent=2, allow_nan=True))

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
