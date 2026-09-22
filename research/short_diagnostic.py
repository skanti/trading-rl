"""Bid-priced, whole-share short scenarios; historical borrow access is unknown.

This is diagnostic research only. It assumes borrow availability and no dividend
liability, so it cannot establish an executable short strategy. Default 3% annual
stock borrow is a scenario assumption, not an observed historical fee series.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.basket_screen import load_inputs
from research.regime_screen import metrics
from trading_rl.overnight.execution_prices import (
    DEFAULT_NBBO_PATH,
    load_scheduled_nbbo_prices,
)
from trading_rl.overnight.portfolio import basket_quantities, basket_returns
from trading_rl.overnight.risk_history import BasketHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nbbo-path", type=Path, default=Path(DEFAULT_NBBO_PATH))
    parser.add_argument("--short-borrow-rate", type=float, default=0.03)
    args = parser.parse_args()
    if not 0 <= args.short_borrow_rate <= 1:
        parser.error("short-borrow-rate must be a fraction in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.input_dir)
    bids, age, _ = load_scheduled_nbbo_prices(args.nbbo_path, inputs["dates"], inputs["symbols"], "bid", 945)
    history = BasketHistory(inputs["symbols"], inputs["dollar_volume"], top=12, ema_span=10,
                            min_history_days=20, minimum_trading_days=100,
                            liquidity_scheme="turnover_stability", issuers=inputs["issuers"])
    indices = np.flatnonzero((inputs["dates"] >= "2023-01-01") & (inputs["dates"] < inputs["end_date"]))
    baskets = [history.select(row, entry_allowed=inputs["entry_session_mask"][row],
                              exchange_mask=inputs["execution_exchange_mask"][row]) for row in indices]
    observations = []
    for row, selected in zip(indices, baskets, strict=True):
        entry, exits = inputs["entry_prices"][row, selected], inputs["morning_prices"][row + 1, selected]
        observations.append(basket_returns(entry, exits, slots=12, require_complete=True,
                                           entry_staleness=inputs["entry_staleness"][row, selected], max_entry_age=1,
                                           exit_staleness=inputs["morning_staleness"][row + 1, selected], max_exit_age=1440).unscaled_return)
        basket_returns(bids[row, selected], exits, slots=12, require_complete=True,
                       entry_staleness=age[row, selected], max_entry_age=1)
    dates, exits = inputs["dates"][indices], inputs["dates"][indices + 1]
    holding = (exits - dates).days.to_numpy()
    spy = pd.Series(inputs["spy_trend_marks"])
    records = []
    for trend, window, target, weak in itertools.product((100, 150, 200), (10, 20), (0.35, 0.5), (0, 0.25, -0.25, -0.5, -1)):
        vol = pd.Series(observations).rolling(window).std().shift(1).to_numpy() * np.sqrt(252)
        strong = (spy.shift(1) >= spy.rolling(trend).mean().shift(1)).to_numpy()[indices]
        base = np.minimum(2, np.divide(target, vol, out=np.ones_like(vol), where=np.isfinite(vol) & (vol > 0)))
        exposures = base * np.where(strong, 1, weak)
        # Whole shares and compounding in both directions; reset budget at each fold.
        record = {"trend_window": trend, "volatility_window": window, "volatility_target": target,
                  "weak_trend_multiplier": weak, "assumed_short_borrow_rate": args.short_borrow_rate}
        for fold, mask in {"train": dates < "2025-01-01", "validation": (dates >= "2025-01-01") & (dates < "2026-01-01")}.items():
            equity, returns, short_days = 10000.0, [], 0
            for k in np.flatnonzero(mask):
                selected, row, exposure = baskets[k], indices[k], exposures[k]
                if len(selected) == 0 or exposure == 0:
                    returns.append(0.0)
                    continue
                direction = 1 if exposure > 0 else -1
                entry = inputs["entry_prices"][row, selected] if direction > 0 else bids[row, selected]
                exit_prices = inputs["morning_prices"][row + 1, selected]
                quantities = basket_quantities(entry, equity * abs(exposure), "whole", slots=12)
                notional = float(quantities @ entry)
                financing = (max(0, notional - equity) * 0.0675 if direction > 0 else notional * args.short_borrow_rate) * holding[k] / 360
                pnl = direction * float(quantities @ (exit_prices - entry)) - financing
                returns.append(pnl / equity)
                equity += pnl
                short_days += direction < 0
                if equity <= 0:
                    raise ValueError("short diagnostic exhausted equity")
            record.update({f"{fold}_{key}": value for key, value in metrics(returns, dates[mask], exits[mask]).items()})
            record[f"{fold}_short_sessions"] = short_days
        records.append(record)
    frame = pd.DataFrame(records)
    frame["selection_score"] = ((1 + frame.train_cagr) ** 2 * (1 + frame.validation_cagr)) ** (1 / 3) - 1
    frame["within_daily_drawdown_limit"] = (frame.train_drawdown <= .20) & (frame.validation_drawdown <= .20)
    frame = frame.sort_values("selection_score", ascending=False)
    frame.to_csv(args.output_dir / "scenarios.csv", index=False)
    (args.output_dir / "assumptions.json").write_text(json.dumps({
        "executable_strategy": False, "scope": "2023-2025 only", "limitations": __doc__,
        "entry": "observed SIP ask for long, bid for short; <=1 minute old",
        "exit": "primary opening auction", "short_borrow_rate": args.short_borrow_rate,
        "short_dividends": "not available, omitted", "share_mode": "whole, split-adjusted prices",
        "broker_shortable_history": "not available, assumed all selected stocks borrowable",
    }, indent=2) + "\n")
    print(frame.loc[frame.within_daily_drawdown_limit].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
