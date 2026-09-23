"""Test simpler fixed sizing and longer volatility histories on frozen baskets."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.regime_screen import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    grid = {"trend_window": [60, 80, 100, 120, 150, 200],
            "volatility_window": [0, 60, 80, 100],
            "volatility_target": [.3, .35, .4], "fixed_exposure": [1., 1.5, 2.]}
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "grid": grid, "status": "adaptive exploratory/in-sample; all periods reused",
        "units": str(args.unit_dir), "zero_window": "fixed exposure; no volatility targeting",
    }, indent=2) + "\n")
    results = []
    for path in sorted(args.unit_dir.glob("unit-*.csv")):
        horizon = int(path.stem.split("-")[1][:-1])
        count = int(path.stem.split("top")[1])
        portfolio = pd.read_csv(path)
        unit = portfolio.unscaled_return.to_numpy()
        dates, exits = pd.DatetimeIndex(portfolio.entry_date), pd.DatetimeIndex(portfolio.exit_date)
        hold, traded = (exits - dates).days.to_numpy(), portfolio.traded.to_numpy()
        for trend, window in itertools.product(grid["trend_window"], grid["volatility_window"]):
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252) if window else None
            for target in grid["volatility_target"] if window else grid["fixed_exposure"]:
                base = np.minimum(2., np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0))) if window else target
                exposure = base * strong
                returns = exposure * unit - np.maximum(exposure - 1, 0) * inputs["margin_interest_rate"] * hold / 360 * traded
                results.append(dict(allocation_window=horizon, allocation_count=count, trend_window=trend,
                                    volatility_window=window, volatility_target=target if window else None,
                                    fixed_exposure=target if not window else None, **metrics(returns, dates, exits)))
    frame = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    frame["exit_screen_pass"] = (frame.cagr > 1) & (frame.sharpe >= 2.5) & (frame.drawdown < .2)
    frame.to_csv(args.output_dir / "screen.csv", index=False)
    print(frame.loc[frame.exit_screen_pass].head(20).to_string(index=False))
    print("Highest Sharpe:")
    print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
