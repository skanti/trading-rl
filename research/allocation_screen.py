"""Test causal stock allocation inside the unchanged twelve-name liquidity basket.

2026 is omitted from selection. It has already been inspected in earlier work,
so any later evaluation there is a reused holdout, not new independent evidence.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.regime_screen import metrics
from trading_rl.overnight.history import DEFAULT_DAILY_DATA_DIR, load_daily_closes
from trading_rl.overnight.portfolio import basket_returns
from trading_rl.overnight.risk_history import BasketHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, default=Path(DEFAULT_DAILY_DATA_DIR))
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--short-horizon", action="store_true",
                        help="screen lagged 1/2/5/10-session continuation and reversal")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    dates = inputs["dates"]
    closes = np.column_stack([load_daily_closes(args.daily_dir / f"{symbol}.npy", dates) for symbol in inputs["symbols"]])
    np.savez_compressed(args.output_dir / "daily-closes.npz", dates=dates.to_numpy(dtype="datetime64[D]"), symbols=inputs["symbols"], closes=closes)
    history = BasketHistory(inputs["symbols"], inputs["dollar_volume"], top=12, ema_span=10,
                            min_history_days=20, minimum_trading_days=100,
                            liquidity_scheme="turnover_stability", issuers=inputs["issuers"])
    rows = np.flatnonzero((dates >= "2023-01-01") & (dates < inputs["end_date"]))
    selected = [history.select(row, entry_allowed=inputs["entry_session_mask"][row],
                               exchange_mask=inputs["execution_exchange_mask"][row]) for row in rows]
    price_returns = []
    for row, basket in zip(rows, selected, strict=True):
        entries, exits = inputs["entry_prices"][row, basket], inputs["morning_prices"][row + 1, basket]
        basket_returns(entries, exits, slots=12, require_complete=True,
                       entry_staleness=inputs["entry_staleness"][row, basket], max_entry_age=1,
                       exit_staleness=inputs["morning_staleness"][row + 1, basket], max_exit_age=1440)
        price_returns.append(exits / entries - 1)
    frame = pd.DataFrame(closes)
    features = {}
    for window in ((1, 2, 5, 10) if args.short_horizon else (20, 60, 120)):
        features[f"momentum{window}"] = (frame.shift(1) / frame.shift(window + 1) - 1).to_numpy()
        if args.short_horizon:
            features[f"reversal{window}"] = -features[f"momentum{window}"]
    features["inversevol20"] = (1 / frame.pct_change(fill_method=None).rolling(20).std().shift(1)).to_numpy()
    features["equal"] = np.ones_like(closes)
    entry_dates, exit_dates = dates[rows].strftime("%Y-%m-%d").to_numpy(), dates[rows + 1].strftime("%Y-%m-%d").to_numpy()
    hold = (dates[rows + 1] - dates[rows]).days.to_numpy()
    traded = inputs["entry_session_mask"][rows]
    spy = pd.Series(inputs["spy_trend_marks"])
    folds = {"train": entry_dates < "2025-01-01",
             "validation": (entry_dates >= "2025-01-01") & (entry_dates < "2026-01-01")}
    records, failures = [], []
    for feature, count in itertools.product(features, (4, 6, 8, 12)):
        if feature in {"inversevol20", "equal"} and count != 12:
            continue
        unit = []
        for row, basket, returns in zip(rows, selected, price_returns, strict=True):
            if not len(basket):
                unit.append(0.0)
                continue
            values = features[feature][row, basket]
            if not np.isfinite(values).all():
                failures.append({"feature": feature, "count": count, "first_missing_date": str(dates[row].date())})
                break
            if feature.startswith(("momentum", "reversal")):
                if count < 12:
                    weights = np.zeros(len(basket))
                    weights[np.argsort(-values, kind="stable")[:count]] = 1 / count
                else:
                    # Bounded tilt: momentum ranks receive weights 1..12.
                    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable") + 1
                    weights = ranks / ranks.sum()
            else:
                weights = values / values.sum()
            unit.append(float(weights @ returns))
        if len(unit) != len(rows):
            continue
        unit = np.array(unit)
        np.savez_compressed(args.output_dir / f"{feature}-{count}.npz", unit=unit, dates=entry_dates.astype("U10"), exits=exit_dates.astype("U10"))
        grid = itertools.product((100, 150, 200), (10, 20), (0.35, 0.5), (0, 0.25), (10, 20, 60), (0, 0.01)) if args.confirmation else itertools.product((100, 150), (10, 20), (0.35, 0.5), (0, 0.25), (0,), (0,))
        for trend, window, target, weak, confirmation, buffer in grid:
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1) * (1 + buffer)).to_numpy()[rows]
            if confirmation:
                strong &= (spy.shift(1) >= spy.shift(confirmation + 1)).to_numpy()[rows]
            base = np.minimum(2.0, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
            exposure = base * np.where(strong, 1, weak)
            returns = exposure * unit - np.maximum(exposure - 1, 0) * 0.0675 * hold / 360 * traded
            record = {"allocation_feature": feature, "allocation_count": count, "trend_window": trend,
                      "volatility_window": window, "volatility_target": target, "weak_trend_multiplier": weak,
                      "momentum_window": confirmation, "trend_buffer": buffer}
            for name, mask in folds.items():
                record.update({f"{name}_{key}": value for key, value in metrics(returns[mask], entry_dates[mask], exit_dates[mask]).items()})
            records.append(record)
    results = pd.DataFrame(records)
    # Require development improvements in both periods over the shipped policy.
    results["eligible"] = ((results.train_cagr > 0.6717894951153125) & (results.validation_cagr > 1.1084904227949472)
                           & (results.train_drawdown <= 0.20) & (results.validation_drawdown <= 0.20))
    results["selection_score"] = ((1 + results.train_cagr) ** 2 * (1 + results.validation_cagr)) ** (1 / 3) - 1
    results = results.sort_values("selection_score", ascending=False)
    results.to_csv(args.output_dir / "screen.csv", index=False)
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    selection = results.loc[results.eligible]
    (args.output_dir / "selection.json").write_text(json.dumps(None if selection.empty else selection.iloc[0].to_dict(), indent=2) + "\n")
    print(results.head(12).to_string(index=False))
    print("Eligible:")
    print(selection.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
