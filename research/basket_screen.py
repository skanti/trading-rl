"""Prepare shared-code basket observations, then screen development periods only."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.regime_screen import metrics
from trading_rl.overnight.portfolio import basket_returns
from trading_rl.overnight.risk_history import BasketHistory


def load_inputs(directory):
    with np.load(directory / "inputs.npz", allow_pickle=False) as data:
        args = {key: data[key] for key in data.files}
    args.update(json.loads((directory / "inputs.json").read_text()))
    args["dates"] = pd.DatetimeIndex(args["dates"])
    for key in ("start_date", "end_date"):
        args[key] = pd.Timestamp(args[key])
    return args


def observations(args, top, ema_span):
    history = BasketHistory(
        args["symbols"], args["dollar_volume"], top=top, ema_span=ema_span,
        min_history_days=args["min_history_days"],
        minimum_trading_days=args["minimum_trading_days"],
        liquidity_scheme=args["liquidity_scheme"], issuers=args["issuers"],
    )
    rows = np.flatnonzero((args["dates"] >= "2023-01-01") & (args["dates"] < args["end_date"]))
    result = []
    for row in rows:
        selected = history.select(row, entry_allowed=args["entry_session_mask"][row],
                                  exchange_mask=args["execution_exchange_mask"][row])
        observation = basket_returns(
            args["entry_prices"][row, selected], args["morning_prices"][row + 1, selected],
            slots=top, entry_staleness=args["entry_staleness"][row, selected],
            exit_staleness=args["morning_staleness"][row + 1, selected],
            max_entry_age=1, max_exit_age=args["max_exit_staleness_minutes"],
            require_complete=True,
        )
        result.append(observation.unscaled_return)
    return rows, np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-drawdown", type=float, default=0.20)
    parser.add_argument("--confirmation", action="store_true",
                        help="test momentum confirmation and a moving-average buffer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    table, failures = [], []
    baskets = itertools.product((6, 10, 12), (10, 20)) if args.confirmation else itertools.product((4, 6, 8, 10, 12, 16), (5, 10, 20))
    for top, ema in baskets:
        try:
            indices, unit = observations(inputs, top, ema)
        except ValueError as error:
            failures.append({"top": top, "ema_span": ema, "error": str(error)})
            continue
        dates = inputs["dates"][indices].strftime("%Y-%m-%d").to_numpy()
        exits = inputs["dates"][indices + 1].strftime("%Y-%m-%d").to_numpy()
        hold = (pd.to_datetime(exits) - pd.to_datetime(dates)).days.to_numpy()
        traded = inputs["entry_session_mask"][indices]
        spy = pd.Series(inputs["spy_trend_marks"])
        folds = {"train": dates < "2025-01-01",
                 "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}
        np.savez_compressed(args.output_dir / f"basket_top{top}_ema{ema}.npz", unit=unit, dates=dates.astype("U10"), exits=exits.astype("U10"))
        grid = itertools.product(
            (60, 100, 150, 200), (10, 20), (0.35, 0.5), (0, 0.25), (0, 0.01, 0.02), (0, 10, 20, 60),
        ) if args.confirmation else itertools.product(
            (20, 60, 100, 150, 200), (10, 20, 40), (0.25, 0.35, 0.5), (0, 0.25, 0.5), (0,), (0,),
        )
        for trend, window, target, weak, buffer, momentum in grid:
            strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1) * (1 + buffer)).to_numpy()[indices]
            if momentum:
                strong &= (spy.shift(1) >= spy.shift(momentum + 1)).to_numpy()[indices]
            vol = pd.Series(unit).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
            base = np.minimum(2.0, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
            exposure = base * np.where(strong, 1.0, weak)
            result = exposure * unit - np.maximum(exposure - 1, 0) * 0.0675 * hold / 360 * traded
            record = {"top": top, "ema_span": ema, "trend_window": trend, "volatility_window": window,
                      "volatility_target": target, "weak_trend_multiplier": weak,
                      "trend_buffer": buffer, "momentum_window": momentum}
            for fold, mask in folds.items():
                record.update({f"{fold}_{key}": value for key, value in metrics(result[mask], dates[mask], exits[mask]).items()})
            table.append(record)
    frame = pd.DataFrame(table)
    frame["selection_score"] = frame[["train_cagr", "validation_cagr"]].min(axis=1)
    frame["eligible"] = (frame.train_drawdown <= args.max_drawdown) & (frame.validation_drawdown <= args.max_drawdown)
    frame = frame.sort_values("selection_score", ascending=False)
    frame.to_csv(args.output_dir / "screen.csv", index=False)
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(frame.loc[frame.eligible].head(20).to_string(index=False))
    print("Failures:", failures)


if __name__ == "__main__":
    main()
