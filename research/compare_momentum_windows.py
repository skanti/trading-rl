"""Compare momentum lookbacks with fixed sizing through the production backtester."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from research.basket_screen import load_inputs
from research.compare_momentum_counts import fingerprint
from trading_rl.overnight.backtest import run_backtest
from trading_rl.overnight.backtest_audit import minute_mark_audit
from trading_rl.overnight.momentum import LiquidityMomentumFocusConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True,
                        help="Exact 10-session portfolio reference")
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--minute-data-dir", type=Path,
                        default=Path("/data/ppv1/updates/bars_1min_2022-01-01"))
    args = parser.parse_args()
    if any(window < 1 for window in args.windows):
        parser.error("windows must be positive")
    manifest = json.loads((args.input_dir / "inputs-manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if fingerprint(args.input_dir / name) != expected:
            raise ValueError(f"frozen input fingerprint changed: {name}")
    inputs = load_inputs(args.input_dir)
    policy = LiquidityMomentumFocusConfig(
        allocation_count=3, allocation_window=10, trend_window=100,
        volatility_window=20, volatility_target=.35, max_exposure=2,
        weak_trend_multiplier=.1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "historical in-sample sensitivity; no live configuration changes",
        "windows": args.windows,
        "fixed_parameters": policy.as_dict(),
        "varying_parameter": "allocation_window",
        "inputs": manifest,
        "minute_data_dir": str(args.minute_data_dir),
        "reference_summary": str(args.reference_summary),
        "source_sha256": {
            str(path): fingerprint(path)
            for path in [Path(__file__), *Path("trading_rl/overnight").glob("*.py")]
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records, annual = [], []
    figure = Figure(figsize=(11, 8), dpi=150)
    FigureCanvasAgg(figure)
    equity_axis, drawdown_axis = figure.subplots(2, 1, sharex=True)
    for window in args.windows:
        trades, summary = run_backtest(**{
            **inputs, "strategy": "liquidity-momentum-focus",
            "strategy_config": replace(policy, allocation_window=window),
        })
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        if window == 10:
            reference = json.loads(args.reference_summary.read_text())
            pd.testing.assert_frame_equal(
                portfolio, pd.DataFrame(reference["daily_portfolio"]), check_exact=True,
            )
        marks, audit = minute_mark_audit(trades, summary, args.minute_data_dir)
        summary["minute_mark_audit"] = audit
        summary["comparison_protocol"] = str(args.output_dir / "protocol.json")
        folder = args.output_dir / f"momentum-{window}d"
        folder.mkdir(exist_ok=True)
        trades.to_csv(folder / "trades.csv", index=False)
        portfolio.to_csv(folder / "portfolio.csv", index=False)
        marks.to_csv(folder / "minute_marks.csv", index=False)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        metrics = summary["strategy_metrics"]
        records.append({
            "momentum_sessions": window,
            "calendar_cagr": metrics["calendar_cagr"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe": metrics["sharpe_zero_cash_rate"],
            "minute_drawdown": audit["minute_open_mark_drawdown"],
            "exit_drawdown": metrics["max_drawdown"],
            "average_exposure": summary["average_exposure"],
            "ending_equity": summary["ending_equity"],
            "first_entry": summary["first_entry_date"],
            "last_exit": summary["last_exit_date"],
        })
        for year, rows in portfolio.groupby(pd.to_datetime(portfolio.exit_date).dt.year):
            equity = np.r_[1., np.cumprod(1 + rows.strategy_return)]
            annual.append({
                "momentum_sessions": window, "exit_year": int(year),
                "return": float(equity[-1] - 1),
                "annualized_volatility": float(rows.strategy_return.std(ddof=1) * np.sqrt(252)),
                "exit_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))),
            })
        dates = pd.to_datetime([summary["first_entry_date"], *portfolio.exit_date])
        equity = np.r_[summary["budget"], portfolio.portfolio_end_equity]
        equity_axis.plot(dates, equity, label=f"{window}-session momentum")
        drawdown_axis.plot(dates, equity / np.maximum.accumulate(equity) - 1,
                           label=f"{window}-session momentum")
        print(json.dumps(records[-1]), flush=True)
    pd.DataFrame(records).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(annual).to_csv(args.output_dir / "annual_returns.csv", index=False)
    equity_axis.set_ylabel("Equity ($, log scale)")
    equity_axis.set_yscale("log")
    drawdown_axis.set_ylabel("Drawdown at daily exit")
    drawdown_axis.yaxis.set_major_formatter(PercentFormatter(1))
    for axis in (equity_axis, drawdown_axis):
        axis.grid(alpha=.2)
        axis.legend()
    window_label = " vs ".join(str(window) for window in args.windows)
    figure.suptitle(f"Momentum-focus: {window_label} sessions\n"
                   "N=3 · SPY100 · vol20 · 35% target · 2× cap · weak multiplier 0.1 · in-sample")
    figure.tight_layout()
    figure.savefig(args.output_dir / "momentum_window_comparison.png")
    print(f"Results and chart: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
