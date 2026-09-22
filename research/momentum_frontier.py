"""Use common-engine unit portfolios and minute bounds to select momentum rules."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.frontier_screen import drawdown_bound, mark_bounds
from research.liquidity_momentum import LiquidityMomentumConfig
from research.regime_screen import metrics
from trading_rl.overnight.backtest import run_backtest
from trading_rl.overnight.backtest_audit import minute_mark_audit
from trading_rl.overnight.history import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    with np.load(args.input_dir / "allocation/daily-closes.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["symbols"], inputs["symbols"])
        np.testing.assert_array_equal(archive["dates"], inputs["dates"].to_numpy(dtype="datetime64[D]"))
        inputs["daily_closes"] = archive["closes"]
    spy = pd.Series(inputs["spy_trend_marks"], index=inputs["dates"])
    records = []
    for windows, count in itertools.product(((5,), (10,), (20,), (5, 10)), (4, 5, 6)):
        name = f"momentum-{'-'.join(map(str, windows))}-top{count}"
        output = args.output_dir / name
        output.mkdir(parents=True, exist_ok=True)
        config = LiquidityMomentumConfig(allocation_windows=windows, allocation_count=count,
                                         max_exposure=1.0, volatility_target=100.0, weak_trend_multiplier=1.0)
        trades, summary = run_backtest(**dict(inputs, strategy="liquidity-momentum-vol", strategy_config=config))
        marks, audit = minute_mark_audit(trades, summary, Path(DEFAULT_DATA_DIR))
        summary["minute_mark_audit"] = audit
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        portfolio.to_csv(output / "portfolio.csv", index=False)
        marks.to_csv(output / "minute_marks.csv", index=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        frame, low, high, _ = mark_bounds(output)
        unit = frame.unscaled_return.to_numpy()
        dates, exits = pd.DatetimeIndex(frame.entry_date), pd.DatetimeIndex(frame.exit_date)
        hold, traded = (exits - dates).days.to_numpy(), frame.traded.to_numpy()
        folds = {"train": dates < "2025-01-01", "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}
        # No calendar exclusions or market-confirmation tuning in this experiment.
        for trend, window, target in itertools.product((100, 150), (10, 20), (.30, .325, .35, .375, .40)):
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).reindex(dates).to_numpy()
            exposure = np.minimum(2, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0))) * strong
            borrow = np.maximum(exposure - 1, 0) * .0675 * hold / 360 * traded
            returns = exposure * unit - borrow
            record = {"allocation_windows": json.dumps(windows), "allocation_count": count, "trend_window": trend,
                      "volatility_window": window, "volatility_target": target, "weak_trend_multiplier": 0.0}
            for fold, mask in folds.items():
                record.update({f"{fold}_{key}": value for key, value in metrics(returns[mask], dates[mask], exits[mask]).items()})
                record[f"{fold}_minute_drawdown_bound"] = drawdown_bound(returns[mask], exposure[mask], borrow[mask], low[mask], high[mask])
            records.append(record)
        print(f"Prepared and screened {name}", flush=True)
    results = pd.DataFrame(records)
    results["score"] = ((1 + results.train_cagr) ** 2 * (1 + results.validation_cagr)) ** (1 / 3) - 1
    results["eligible"] = ((results.train_minute_drawdown_bound <= .20) & (results.validation_minute_drawdown_bound <= .20)
                           & (results.train_cagr > .6717894951153125) & (results.validation_cagr > 1.1084904227949472))
    results = results.sort_values("score", ascending=False)
    results.to_csv(args.output_dir / "screen.csv", index=False)
    selected = results.loc[results.eligible]
    (args.output_dir / "selection.json").write_text(json.dumps({
        "selected": None if selected.empty else selected.iloc[0].to_dict(), "trials": len(results),
        "selection": "highest development CAGR among candidates with both minute drawdown bounds <=20%",
        "2026": "never included in selection; already reused historical validation",
    }, indent=2) + "\n")
    print(selected.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
