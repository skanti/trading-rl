"""Compare weak-trend sizing through the production backtester and minute audit."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from research.basket_screen import load_inputs
from trading_rl.overnight.backtest import run_backtest
from trading_rl.overnight.backtest_audit import minute_mark_audit
from trading_rl.overnight.momentum import LiquidityMomentumFocusConfig


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0, .1, .25, .5, .75, 1])
    parser.add_argument(
        "--minute-data-dir", type=Path,
        default=Path("/data/ppv1/updates/bars_1min_2022-01-01"),
    )
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "inputs-manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if fingerprint(args.input_dir / name) != expected:
            raise ValueError(f"frozen input fingerprint changed: {name}")
    inputs = load_inputs(args.input_dir)
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in args.multipliers):
        parser.error("multipliers must be finite and between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = LiquidityMomentumFocusConfig(allocation_count=3, allocation_window=10, trend_window=100, volatility_window=20, volatility_target=.35, weak_trend_multiplier=0)
    protocol = {
        "status": "historical in-sample sensitivity; no live configuration changes",
        "multipliers": args.multipliers,
        "fixed_parameters": policy.as_dict(),
        "varying_parameter": "weak_trend_multiplier",
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
    for multiplier in args.multipliers:
        config = replace(policy, weak_trend_multiplier=multiplier)
        trades, summary = run_backtest(
            **{**inputs, "strategy": "liquidity-momentum-focus", "strategy_config": config}
        )
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        if multiplier == 0:
            reference = json.loads(args.reference_summary.read_text())
            pd.testing.assert_frame_equal(
                portfolio, pd.DataFrame(reference["daily_portfolio"]), check_exact=True,
            )
        marks, audit = minute_mark_audit(trades, summary, args.minute_data_dir)
        summary["minute_mark_audit"] = audit
        summary["comparison_protocol"] = str(args.output_dir / "protocol.json")
        folder = args.output_dir / f"weak-{multiplier:g}"
        folder.mkdir(exist_ok=True)
        trades.to_csv(folder / "trades.csv", index=False)
        portfolio.to_csv(folder / "portfolio.csv", index=False)
        marks.to_csv(folder / "minute_marks.csv", index=False)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        metrics = summary["strategy_metrics"]
        records.append({
            "weak_trend_multiplier": multiplier,
            "calendar_cagr": metrics["calendar_cagr"],
            "total_return": summary["ending_equity"] / summary["budget"] - 1,
            "ending_equity": summary["ending_equity"],
            "sharpe": metrics["sharpe_zero_cash_rate"],
            "minute_drawdown": audit["minute_open_mark_drawdown"],
            "exit_drawdown": metrics["max_drawdown"],
            "average_exposure": summary["average_exposure"],
            "trades": summary["trades"],
            "traded_sessions": int(portfolio.traded.sum()),
            "total_sessions": len(portfolio),
            "short_sessions_skipped": summary["skipped_short_entry_sessions"],
            "first_entry": summary["first_entry_date"],
            "last_exit": summary["last_exit_date"],
        })
        for year, rows in portfolio.groupby(pd.to_datetime(portfolio.exit_date).dt.year):
            annual.append({
                "weak_trend_multiplier": multiplier, "exit_year": int(year),
                "return": float(np.prod(1 + rows.strategy_return) - 1),
            })
        pd.DataFrame(records).to_csv(args.output_dir / "comparison.csv", index=False)
        pd.DataFrame(annual).to_csv(args.output_dir / "annual_returns.csv", index=False)
        print(
            f"Weak multiplier={multiplier:g}: CAGR {metrics['calendar_cagr']:.2%}; "
            f"Sharpe {metrics['sharpe_zero_cash_rate']:.3f}; "
            f"minute drawdown {audit['minute_open_mark_drawdown']:.2%}", flush=True,
        )
    frame = pd.DataFrame(records).sort_values("weak_trend_multiplier")
    figure = Figure(figsize=(11, 8), dpi=150)
    FigureCanvasAgg(figure)
    for axis, (column, label) in zip(
        figure.subplots(3, 1, sharex=True),
        [("calendar_cagr", "Annual return (CAGR)"),
         ("sharpe", "Sharpe"), ("minute_drawdown", "Minute-mark drawdown")], strict=True,
    ):
        axis.plot(frame.weak_trend_multiplier, frame[column], marker="o")
        axis.axvline(0, color="darkorange", linestyle="--", label="Current multiplier=0")
        if column == "minute_drawdown":
            axis.axhline(.20, color="gray", linestyle=":", label="20% drawdown")
        if column != "sharpe":
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylabel(label)
        axis.set_xticks(frame.weak_trend_multiplier)
        axis.grid(alpha=.2)
        axis.legend(loc="best")
    axis.set_xlabel("Weak-trend exposure multiplier")
    figure.suptitle(
        "Liquidity momentum focus: weak-trend sizing\n"
        f"{records[0]['first_entry']}–{records[0]['last_exit']} · "
        "N=3 · 10/100/20 · 35% target · 2× cap · in-sample",
    )
    figure.tight_layout()
    chart = args.output_dir / "weak_trend_comparison.png"
    figure.savefig(chart)
    print(f"Comparison: {(args.output_dir / 'comparison.csv').resolve()}")
    print(f"Annual returns: {(args.output_dir / 'annual_returns.csv').resolve()}")
    print(f"Chart: {chart.resolve()}")


if __name__ == "__main__":
    main()
