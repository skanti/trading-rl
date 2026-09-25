"""Freeze a development-selected rule, then audit it and open the 2026 holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from trading_rl.overnight.momentum import LiquidityRegimeConfig
from research.regime_screen import metrics
from trading_rl.overnight.backtest import run_backtest
from trading_rl.overnight.backtest_audit import minute_mark_audit
from trading_rl.overnight.history import DEFAULT_DATA_DIR
from trading_rl.overnight.strategies import LiquidityTrendVolConfig


def evaluate(inputs, output, minute_dir):
    trades, summary = run_backtest(**inputs)
    output.mkdir(parents=True, exist_ok=True)
    portfolio = pd.DataFrame(summary["daily_portfolio"])
    marks, audit = minute_mark_audit(trades, summary, minute_dir)
    summary["minute_mark_audit"] = audit
    summary["folds"] = {}
    for name, mask in {
        "train": portfolio.entry_date < "2025-01-01",
        "validation": (portfolio.entry_date >= "2025-01-01") & (portfolio.entry_date < "2026-01-01"),
        "holdout": portfolio.entry_date >= "2026-01-01",
    }.items():
        fold = portfolio.loc[mask]
        summary["folds"][name] = metrics(fold.strategy_return, fold.entry_date.to_numpy(), fold.exit_date.to_numpy())
    trades.to_csv(output / "trades.csv", index=False)
    portfolio.to_csv(output / "portfolio.csv", index=False)
    marks.to_csv(output / "minute_marks.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minute-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen = pd.read_csv(args.screen)
    best = screen.loc[screen.eligible].sort_values("selection_score", ascending=False).iloc[0]
    keys = ("trend_window", "volatility_window", "volatility_target", "weak_trend_multiplier", "trend_buffer", "momentum_window")
    params = {key: int(best[key]) if key.endswith("window") else float(best[key]) for key in keys}
    selection = {"strategy": "liquidity-regime-vol", "top": int(best.top), "ema_span": int(best.ema_span),
                 "parameters": params, "selection_score": float(best.selection_score),
                 "screen_sha256": hashlib.sha256(args.screen.read_bytes()).hexdigest(),
                 "selection_period": "2023-2025 only", "holdout": "2026"}
    # Write the locked choice before any holdout metric is evaluated.
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.output_dir / "strategy-config.json").write_text(json.dumps(params, indent=2) + "\n")
    inputs = load_inputs(args.input_dir)
    inputs["strategy_config"] = LiquidityTrendVolConfig()
    results = {"liquidity-trend-vol": evaluate(inputs, args.output_dir / "liquidity-trend-vol", args.minute_dir)}
    inputs.update(strategy="liquidity-regime-vol", strategy_config=LiquidityRegimeConfig(**params),
                  top=selection["top"], ema_span=selection["ema_span"])
    results["liquidity-regime-vol"] = evaluate(inputs, args.output_dir / "liquidity-regime-vol", args.minute_dir)
    # Extra execution friction on top of observed asks/auctions; the strategy is frozen.
    for cost in (1, 5):
        results[f"regime-{cost}bp"] = evaluate(
            dict(inputs, transaction_cost_bps=float(cost)), args.output_dir / f"regime-{cost}bp", args.minute_dir,
        )
    concise = {name: {"cagr": result["strategy_metrics"]["calendar_cagr"],
                      "minute_drawdown": result["minute_mark_audit"]["minute_open_mark_drawdown"],
                      "daily_drawdown": result["strategy_metrics"]["max_drawdown"],
                      "folds": result["folds"]} for name, result in results.items()}
    (args.output_dir / "comparison.json").write_text(json.dumps(concise, indent=2) + "\n")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in ("liquidity-trend-vol", "liquidity-regime-vol"):
        frame = pd.DataFrame(results[name]["daily_portfolio"])
        dates = pd.to_datetime(frame.exit_date)
        equity = frame.portfolio_end_equity.to_numpy()
        peaks = np.maximum.accumulate(np.r_[10000.0, equity])[1:]
        axes[0].plot(dates, equity, label=name)
        axes[1].plot(dates, 100 * (equity / peaks - 1), label=name)
    for ax in axes:
        ax.axvline(pd.Timestamp("2026-01-01"), color="grey", linestyle="--", label="2026 holdout" if ax is axes[0] else None)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Equity ($)")
    axes[1].set_ylabel("Exit-mark drawdown (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=160)
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
