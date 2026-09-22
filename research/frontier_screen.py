"""Refine long/cash rules with conservative minute-mark drawdown bounds.

For the same fractional basket, archived min/max equity marks bound its unit
price returns. Removing none/all of the reference financing gives conservative
lower/upper bounds. The sweep charges its own financing and compares each low
against every preceding high (including that interval's high). This can overstate
drawdown; it cannot certify away a loss hidden by exit-only marks. Finalists still
need the exact common-engine audit. 2026 is excluded from selection.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.regime_screen import metrics


def mark_bounds(reference):
    summary = json.loads((reference / "summary.json").read_text())
    if summary["share_mode"] != "fractional" or summary["transaction_cost_bps_per_side"] != 0:
        raise ValueError("bounds require fractional zero-additional-cost reference trades")
    frame = pd.read_csv(reference / "portfolio.csv").set_index("entry_date")
    marks = pd.read_csv(reference / "minute_marks.csv").set_index("entry_date").reindex(frame.index)
    if (frame.loc[frame.traded, "exposure"] <= 0).any():
        raise ValueError("reference must deploy the intended basket on every trading interval")
    low = ((marks.min_equity / frame.portfolio_start_equity - 1) / frame.exposure).fillna(0).to_numpy()
    high = ((marks.max_equity / frame.portfolio_start_equity - 1 + frame.borrow_return) / frame.exposure).fillna(0).to_numpy()
    if marks.loc[frame.traded].isna().any().any():
        raise ValueError("missing reference minute marks")
    return frame.reset_index(), low, high, summary


def drawdown_bound(returns, exposure, borrow, low, high):
    equity = np.r_[1., np.cumprod(1 + returns)[:-1]]
    interval_high = equity * (1 + exposure * high)
    interval_low = equity * (1 + exposure * low - borrow)
    peaks = np.maximum.accumulate(np.r_[1., interval_high])[1:]
    return float(max(0., np.max(1 - interval_low / peaks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    frame, low, high, summary = mark_bounds(args.reference)
    if summary["top"] != 12 or summary["ema_span_sessions"] != 10:
        raise ValueError("this screen fixes the original twelve-name, ten-session liquidity basket")
    dates, exits = pd.DatetimeIndex(frame.entry_date), pd.DatetimeIndex(frame.exit_date)
    observed_bound = drawdown_bound(frame.strategy_return.to_numpy(), frame.exposure.to_numpy(), frame.borrow_return.to_numpy(), low, high)
    exact = summary["minute_mark_audit"]["minute_open_mark_drawdown"]
    if observed_bound + 1e-12 < exact:
        raise ValueError("reference drawdown bound does not cover the exact audit")
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    unit = frame.unscaled_return.to_numpy()
    hold, traded = (exits - dates).days.to_numpy(), frame.traded.to_numpy()
    folds = {"train": dates < "2025-01-01", "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}
    records = []
    for trend, window, target, weak, excluded in itertools.product(
        (100, 125, 150, 175, 200), (10, 15, 20, 40), (0.30, 0.35, 0.40, 0.45, 0.50), (0, 0.25), (-1, 0, 3),
    ):
        vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
        strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
        base = np.minimum(2, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
        exposure = base * np.where(strong, 1, weak) * (dates.dayofweek != excluded)
        borrow = np.maximum(exposure - 1, 0) * .0675 * hold / 360 * traded
        returns = exposure * unit - borrow
        record = {"trend_window": trend, "volatility_window": window, "volatility_target": target,
                  "weak_trend_multiplier": weak, "excluded_entry_weekday": excluded}
        for fold, mask in folds.items():
            record.update({f"{fold}_{key}": value for key, value in metrics(returns[mask], dates[mask], exits[mask]).items()})
            record[f"{fold}_minute_drawdown_bound"] = drawdown_bound(returns[mask], exposure[mask], borrow[mask], low[mask], high[mask])
        records.append(record)
    results = pd.DataFrame(records)
    results["score"] = ((1 + results.train_cagr) ** 2 * (1 + results.validation_cagr)) ** (1 / 3) - 1
    results["eligible"] = ((results.train_minute_drawdown_bound <= .20) & (results.validation_minute_drawdown_bound <= .20)
                           & (results.train_cagr > .6717894951153125) & (results.validation_cagr > 1.1084904227949472))
    # Report neighborhood stability; do not hide the number of tried configurations.
    results["neighbor_median_score"] = np.nan
    for index, row in results.iterrows():
        neighbors = results.loc[(results.volatility_window == row.volatility_window)
                                & (results.weak_trend_multiplier == row.weak_trend_multiplier)
                                & (results.excluded_entry_weekday == row.excluded_entry_weekday)
                                & ((results.trend_window - row.trend_window).abs() <= 25)
                                & ((results.volatility_target - row.volatility_target).abs() <= .05000001)]
        results.loc[index, "neighbor_median_score"] = neighbors.score.median()
    results = results.sort_values(["neighbor_median_score", "score"], ascending=False)
    results.to_csv(args.output_dir / "screen.csv", index=False)
    selected = results.loc[results.eligible]
    (args.output_dir / "selection.json").write_text(json.dumps({
        "selected": None if selected.empty else selected.iloc[0].to_dict(),
        "trials": len(results), "reference_exact_drawdown": exact, "reference_bound": observed_bound,
        "selection": "highest neighbor-median development CAGR among qualifying minute-bound candidates",
        "holdout": "2026 omitted from selection; already reused in previous research",
    }, indent=2) + "\n")
    print(selected.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
