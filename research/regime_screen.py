"""Screen causal exposure rules using archived unit basket returns.

This is a cheap hypothesis screen, not an execution simulator. Negative exposures
are optimistic short diagnostics: they omit bid/ask differences, borrow access,
dividend liabilities and whole-share effects. Never promote them from this table.
The selected long/cash rule must be rerun through trading-backtest.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(returns, dates, exits):
    returns = np.asarray(returns, dtype=float)
    equity = np.cumprod(1 + returns)
    years = (pd.Timestamp(exits[-1]) - pd.Timestamp(dates[0])).days / 365.25
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "cagr": float(equity[-1] ** (1 / years) - 1),
        "drawdown": float(np.max(1 - equity / peaks)),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--spy-prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-drawdown", type=float, default=0.20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    portfolio = pd.read_csv(args.portfolio)
    spy = pd.read_csv(args.spy_prices, index_col="date").daily_close
    dates = portfolio.entry_date.to_numpy()
    exits = portfolio.exit_date.to_numpy()
    unit = portfolio.unscaled_return.to_numpy()
    hold = (pd.to_datetime(exits) - pd.to_datetime(dates)).days.to_numpy()
    # Do not evaluate 2026 until a configuration has been frozen separately.
    folds = {"train": dates < "2025-01-01",
             "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}
    rows = []
    grid = itertools.product(
        (10, 20, 40, 60, 100, 150, 200),
        (10, 20, 40), (0.25, 0.35, 0.50, 0.70),
        (0.0, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0),
    )
    for trend, window, target, weak in grid:
        strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
        vol = pd.Series(unit).rolling(window).std(ddof=1).shift(1).to_numpy() * np.sqrt(252)
        base = np.minimum(2.0, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
        exposure = base * np.where(strong, 1.0, weak)
        # Early closes contribute zero observations and no deployed financing.
        traded = portfolio.traded.to_numpy(dtype=bool)
        result = exposure * unit - np.maximum(exposure - 1, 0) * 0.0675 * hold / 360 * traded
        row = {"trend_window": trend, "volatility_window": window,
               "volatility_target": target, "weak_trend_multiplier": weak,
               "short_diagnostic_only": weak < 0}
        for fold, mask in folds.items():
            row.update({f"{fold}_{key}": value for key, value in metrics(result[mask], dates[mask], exits[mask]).items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Maximize the weaker development-period CAGR, subject to both drawdown caps.
    frame["selection_score"] = frame[["train_cagr", "validation_cagr"]].min(axis=1)
    frame["eligible"] = ((frame.train_drawdown <= args.max_drawdown)
                         & (frame.validation_drawdown <= args.max_drawdown)
                         & ~frame.short_diagnostic_only)
    frame = frame.sort_values("selection_score", ascending=False)
    frame.to_csv(args.output_dir / "screen.csv", index=False)
    eligible = frame.loc[frame.eligible]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    manifest = {
        "grid_trials": len(frame), "selection": selected,
        "selection_rule": "maximize min(2023-2024 CAGR, 2025 CAGR), long/cash only, both drawdown limits",
        "max_development_drawdown": args.max_drawdown,
        "holdout": "2026 excluded from evaluation and selection",
        "limitations": __doc__,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.portfolio, args.spy_prices)},
    }
    (args.output_dir / "screen.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(frame.loc[frame.eligible].head(12).to_string(index=False))
    print("Optimistic short diagnostics:")
    print(frame.loc[frame.short_diagnostic_only].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
