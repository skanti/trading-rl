"""Screen single-horizon equal-weight baskets, explicitly using in-sample history.

The common engine supplies unit basket returns. The cheap exposure screen is
only a shortlist: finalists must match the common engine and its minute audit.
All trials are retained; exit-only drawdown cannot certify the requested limit.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from trading_rl.overnight.momentum import LiquidityMomentumConfig
from research.regime_screen import metrics
from trading_rl.overnight.backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 10, 15, 20])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 4, 5, 6])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    grid = {
        "allocation_window": args.windows,
        "allocation_count": args.counts,
        "trend_window": [60, 80, 100, 120, 150, 200],
        "volatility_window": [10, 20, 40],
        "volatility_target": [.30, .35, .40],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "grid": grid, "status": "exploratory/in-sample; all 2023-2026 history reused",
        "fixed": "top12 liquidity EMA10; one equal-weight basket; cash below SPY SMA; 2x cap",
        "targets": "calendar CAGR >1, zero-cash Sharpe >=2.5, exact minute drawdown <.20",
    }, indent=2) + "\n")
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    results = []
    for horizon, count in itertools.product(grid["allocation_window"], grid["allocation_count"]):
        config = LiquidityMomentumConfig(
            allocation_window=horizon, allocation_count=count, allocation_windows=None,
            volatility_target=100., max_exposure=1., weak_trend_multiplier=1.,
        )
        _, summary = run_backtest(**dict(inputs, strategy="liquidity-momentum-vol", strategy_config=config))
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        portfolio.to_csv(args.output_dir / f"unit-{horizon}d-top{count}.csv", index=False)
        unit = portfolio.unscaled_return.to_numpy()
        dates, exits = pd.DatetimeIndex(portfolio.entry_date), pd.DatetimeIndex(portfolio.exit_date)
        hold, traded = (exits - dates).days.to_numpy(), portfolio.traded.to_numpy()
        for trend, window, target in itertools.product(grid["trend_window"], grid["volatility_window"], grid["volatility_target"]):
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
            exposure = np.minimum(2., np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0))) * strong
            returns = exposure * unit - np.maximum(exposure - 1, 0) * inputs["margin_interest_rate"] * hold / 360 * traded
            record = dict(allocation_window=horizon, allocation_count=count, trend_window=trend,
                          volatility_window=window, volatility_target=target, **metrics(returns, dates, exits))
            results.append(record)
        pd.DataFrame(results).to_csv(args.output_dir / "screen.csv", index=False)
        print(f"Screened {horizon}d top{count}; {len(results)} trials", flush=True)
    frame = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    frame["exit_screen_pass"] = (frame.cagr > 1) & (frame.sharpe >= 2.5) & (frame.drawdown < .2)
    frame.to_csv(args.output_dir / "screen.csv", index=False)
    print(frame.loc[frame.exit_screen_pass].head(20).to_string(index=False))
    print("Highest Sharpe:")
    print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
