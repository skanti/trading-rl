"""Small, headless equity plots built from the backtest's reported return series."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def equity_curve(summary: dict[str, object]) -> pd.DataFrame:
    """Include initial capital and preserve unknown returns as gaps, never zeroes."""
    sessions = pd.DataFrame(summary["daily_portfolio"])
    capital = float(summary["budget"]) if summary.get("budget") is not None else 1.0
    dates = pd.to_datetime([sessions.entry_date.iloc[0], *sessions.exit_date])
    curves = {}
    for name, column in (
        ("strategy", "strategy_return"),
        ("spy_buy_and_hold", "spy_buy_and_hold_return"),
    ):
        returns = sessions[column].to_numpy(dtype=float)
        curves[name] = capital * np.r_[1.0, np.cumprod(1.0 + returns)]
    return pd.DataFrame(curves, index=pd.DatetimeIndex(dates, name="date"))


def write_equity_plot(
    summary: dict[str, object],
    output_dir: Path | None = None,
    *,
    comparisons: list[dict] | None = None,
) -> Path:
    """Write a unique WebP under the temp directory without opening a GUI."""
    # Keep plotting imports out of parser startup and numerical-only API calls.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure
    from matplotlib.ticker import StrMethodFormatter

    curves = equity_curve(summary)
    directory = (
        output_dir
        if output_dir is not None
        else Path(tempfile.gettempdir()) / "trading-backtests"
    )
    directory.mkdir(parents=True, exist_ok=True)
    first, last = curves.index[0], curves.index[-1]
    with tempfile.NamedTemporaryFile(
        prefix=f"equity_{first:%Y%m%d}_{last:%Y%m%d}_",
        suffix=".webp",
        dir=directory,
        delete=False,
    ) as handle:
        path = Path(handle.name)

    figure = Figure(figsize=(11, 6), dpi=120)
    FigureCanvasAgg(figure)
    try:
        axis = figure.subplots()
        series = []
        for item in comparisons or [summary]:
            label = item["strategy"] if comparisons else item.get("strategy_label", f"Top {item['top']} strategy")
            series.append((equity_curve(item)["strategy"], label, item["strategy_metrics"]))
        series.append((curves["spy_buy_and_hold"], "SPY buy & hold", summary["spy_buy_and_hold_metrics"]))
        for values, label, metrics in series:
            ending = float(values.iloc[-1])
            annualized = float(metrics["annualized_return"])
            if np.isfinite(ending) and np.isfinite(annualized):
                label += f" · ${ending:,.2f} · {annualized:.2%} annualized"
            else:
                label += " · incomplete data"
            axis.plot(values.index, values, label=label, linewidth=1.7)

        axis.set_title(
            "Strategy comparison vs SPY buy & hold" if comparisons else "Overnight strategy vs SPY buy & hold",
            loc="left",
            fontweight="bold",
            pad=14,
        )
        axis.set_ylabel(
            "Portfolio value ($)"
            if summary.get("budget") is not None
            else "Portfolio value ($1 normalized start)"
        )
        axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.2f}"))
        locator = AutoDateLocator(minticks=3, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.margins(x=0.01)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        entry_time = str(summary["entry_time_eastern"]).split()[0]
        exit_time = str(summary["exit_time_eastern"]).split()[0]
        leverage_label = summary.get(
            "leverage_label", f"{float(summary.get('leverage', 1.0)):g}× leverage"
        )
        if comparisons:
            leverage_label = "each strategy uses its own exposure policy"
        caption = (
            f"{first:%Y-%m-%d} – {last:%Y-%m-%d} · Start ${curves.strategy.iloc[0]:,.2f}"
            f" · {float(summary['transaction_cost_bps_per_side']):g} bp per side"
            f" · {leverage_label}\n"
            f"{summary['entry_price_source']} {entry_time} → {summary['exit_price_source']} {exit_time} ET"
        )
        missing = int(summary.get("missing_benchmark_sessions", 0))
        if missing:
            caption += f"\nSPY: {missing} missing observations; curve stops when compounded return becomes unknown."
        if summary.get("skipped_missing_prices", 0):
            caption += f"\nStrategy: {summary['skipped_missing_prices']} missing-price allocations held as cash."
        figure.text(0.12, 0.035, caption, fontsize=9, va="bottom")
        figure.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.91,
            bottom=0.23
            if missing or summary.get("skipped_missing_prices", 0)
            else 0.18,
        )
        figure.savefig(path, format="webp", pil_kwargs={"quality": 85, "method": 2})
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        figure.clear()
    return path
