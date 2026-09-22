"""Audit the frozen momentum grid over all history, explicitly in-sample.

This includes 2026 and is not a new holdout. Any choice based on this table is
exploratory/in-sample and requires future independent validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.frontier_screen import drawdown_bound, mark_bounds
from research.regime_screen import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = load_inputs(args.input_dir)
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    frame = pd.read_csv(args.frontier_dir / "screen.csv")
    references = {}
    for index, row in frame.iterrows():
        windows = json.loads(row.allocation_windows)
        name = f"momentum-{'-'.join(map(str, windows))}-top{int(row.allocation_count)}"
        if name not in references:
            references[name] = mark_bounds(args.frontier_dir / name)
        data, low, high, _ = references[name]
        unit = data.unscaled_return.to_numpy()
        dates, exits = pd.DatetimeIndex(data.entry_date), pd.DatetimeIndex(data.exit_date)
        hold, traded = (exits - dates).days.to_numpy(), data.traded.to_numpy()
        vol = pd.Series(unit).rolling(int(row.volatility_window)).std().shift(1).to_numpy() * np.sqrt(252)
        strong = (spy.shift(1) >= spy.rolling(int(row.trend_window)).mean().shift(1)).reindex(dates).to_numpy()
        exposure = np.minimum(2, np.divide(row.volatility_target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0))) * strong
        borrow = np.maximum(exposure - 1, 0) * .0675 * hold / 360 * traded
        returns = exposure * unit - borrow
        for key, value in metrics(returns, dates, exits).items():
            frame.loc[index, f"full_{key}"] = value
        frame.loc[index, "full_minute_drawdown_bound"] = drawdown_bound(returns, exposure, borrow, low, high)
    frame["full_target_met"] = (frame.full_cagr > 1) & (frame.full_minute_drawdown_bound <= .20)
    frame = frame.sort_values("full_cagr", ascending=False)
    frame.to_csv(args.frontier_dir / "full-period-exploratory.csv", index=False)
    print("Meets full-period numerical target (IN-SAMPLE, NOT HOLDOUT):")
    print(frame.loc[frame.full_target_met].head(20).to_string(index=False))
    print("Best return with <=20% minute drawdown bound:")
    print(frame.loc[frame.full_minute_drawdown_bound <= .20].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
