"""Minute-open equity marks for portfolios sized by the backtest engine."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..market_data.schema import BAR_INDEX, validate_bar_columns


def minute_mark_audit(
    trades: pd.DataFrame,
    summary: dict,
    minute_data_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Carry causal marks, charge financing, and reconcile each liquidation.

    Minute opens are marks rather than executable bids. Missing prints carry
    forward, and intraminute extremes are not observed. The final mark uses the
    modeled exit price, including an opening auction where configured.
    """
    sources = {}
    origin = pd.Timestamp("2010-01-01", tz="UTC")
    capital = float(summary.get("budget") or 1.0)
    peak, maximum_drawdown, minimum_margin = capital, 0.0, 1.0
    worst, sampled, endpoint_error = None, 0, 0.0
    marks = []
    entry_clock = summary["entry_time_eastern"].split()[0]
    exit_clock = summary["exit_time_eastern"].split()[0]
    groups = {day: group for day, group in trades.groupby("entry_date")}
    for session in summary["daily_portfolio"]:
        basket = groups.get(session["entry_date"])
        if basket is None or basket.empty:
            continue
        endpoints = [
            pd.Timestamp(f"{session[key]} {clock}", tz="America/New_York").tz_convert(
                "UTC"
            )
            for key, clock in (("entry_date", entry_clock), ("exit_date", exit_clock))
        ]
        start, end = [int((stamp - origin).total_seconds()) for stamp in endpoints]
        stamps = np.arange(start, end + 1, 60, dtype=np.int64)
        position_value = np.zeros(len(stamps))
        for trade in basket.itertuples():
            symbol = trade.sample_id
            if symbol not in sources:
                path = minute_data_dir / f"{symbol}.npy"
                bars = np.load(path, mmap_mode="r")
                validate_bar_columns(bars, "1Min", str(path))
                column = BAR_INDEX["open_mills"]
                valid = bars[:, column] > 0
                sources[symbol] = (
                    np.asarray(bars[valid, 0], dtype=np.int64),
                    np.asarray(bars[valid, column], dtype=float) / 1000,
                )
            seconds, prices = sources[symbol]
            indices = np.searchsorted(seconds, stamps, side="right") - 1
            if (indices < 0).any():
                raise ValueError(
                    f"no causal minute mark for {symbol} on {session['entry_date']}"
                )
            values = prices[indices].copy()
            values[-1] = trade.exit_price
            position_value += trade.quantity * values
        deployed = float(basket.entry_notional.sum())
        half_cost = float(basket.transaction_cost_dollars.sum()) / 2
        curve = (
            session["portfolio_start_equity"]
            - deployed
            + position_value
            - half_cost
            - session["borrow_cost"] * (stamps - start) / (end - start)
        )
        curve[-1] -= half_cost
        error = (
            abs(curve[-1] - session["portfolio_end_equity"])
            / session["portfolio_start_equity"]
        )
        endpoint_error = max(endpoint_error, error)
        if error > 1e-12:
            raise ValueError(
                f"minute audit does not reconcile on {session['entry_date']}: {error}"
            )
        peaks = np.maximum.accumulate(np.maximum(curve, peak))
        drawdown = 1 - curve / peaks
        index = int(np.argmax(drawdown))
        if drawdown[index] > maximum_drawdown:
            maximum_drawdown = float(drawdown[index])
            worst = (origin + pd.Timedelta(seconds=int(stamps[index]))).isoformat()
        minimum_margin = min(minimum_margin, float(np.min(curve / position_value)))
        peak = max(peak, float(curve.max()))
        sampled += len(stamps)
        marks.append(
            {
                "entry_date": session["entry_date"],
                "intrahold_max_drawdown": float(drawdown.max()),
                "min_equity": float(curve.min()),
                "max_equity": float(curve.max()),
            }
        )
    return pd.DataFrame(marks), {
        "minute_open_mark_drawdown": maximum_drawdown,
        "worst_drawdown_timestamp": worst,
        "minimum_equity_to_position_value": minimum_margin,
        "sampled_minute_marks": sampled,
        "max_endpoint_error": endpoint_error,
    }
