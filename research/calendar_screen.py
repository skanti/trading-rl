"""Screen calendar exposure effects on fixed baskets using development data only."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    rows = []
    for family, archive in {
        "equal": "allocation/equal-12.npz",
        "momentum20-top8": "allocation/momentum20-8.npz",
        "momentum60-rank": "allocation/momentum60-12.npz",
    }.items():
        with np.load(args.input_dir / archive, allow_pickle=False) as data:
            unit, dates, exits = data["unit"], pd.DatetimeIndex(data["dates"]), pd.DatetimeIndex(data["exits"])
        hold = (exits - dates).days.to_numpy()
        traded = pd.Series(inputs["entry_session_mask"], index=inputs["dates"]).reindex(dates).to_numpy()
        folds = {"train": dates < "2025-01-01", "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}
        rules = [("none", -1), ("weekend-sqrt", -1), ("weekend-half", -1)] + [("skip-weekday", day) for day in range(5)]
        for trend, window, target, weak, (rule, weekday) in itertools.product((100, 150), (10, 20), (0.35, 0.5), (0, 0.25), rules):
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
            base = np.minimum(2.0, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
            exposure = base * np.where(strong, 1, weak)
            if rule == "weekend-sqrt":
                exposure /= np.sqrt(hold)
            elif rule == "weekend-half":
                exposure *= np.where(hold > 1, 0.5, 1)
            elif rule == "skip-weekday":
                exposure *= dates.dayofweek != weekday
            returns = exposure * unit - np.maximum(exposure - 1, 0) * 0.0675 * hold / 360 * traded
            row = {"family": family, "calendar_rule": rule, "weekday": weekday, "trend_window": trend,
                   "volatility_window": window, "volatility_target": target, "weak_trend_multiplier": weak}
            for name, mask in folds.items():
                row.update({f"{name}_{key}": value for key, value in metrics(returns[mask], dates[mask], exits[mask]).items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["eligible"] = ((frame.train_cagr > 0.6717894951153125) & (frame.validation_cagr > 1.1084904227949472)
                         & (frame.train_drawdown <= 0.20) & (frame.validation_drawdown <= 0.20))
    frame["selection_score"] = ((1 + frame.train_cagr) ** 2 * (1 + frame.validation_cagr)) ** (1 / 3) - 1
    frame = frame.sort_values("selection_score", ascending=False)
    frame.to_csv(args.output_dir / "screen.csv", index=False)
    selected = frame.loc[frame.eligible]
    (args.output_dir / "selection.json").write_text(json.dumps(None if selected.empty else selected.iloc[0].to_dict(), indent=2) + "\n")
    print(selected.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
