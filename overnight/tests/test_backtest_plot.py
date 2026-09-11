import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from trading_rl.overnight import backtest
from trading_rl.overnight.backtest_plot import equity_curve, write_equity_plot


def inputs(returns):
    dates = pd.bdate_range("2026-08-24", periods=len(returns) + 5)
    symbols = np.asarray(["SPY", "AAA"])
    prices = np.full((len(dates), 2), 100.0)
    exits = prices.copy()
    for index, value in enumerate(returns, start=5):
        exits[index, 1] = 100.0 * (1 + value)
    return {
        "dates": dates,
        "symbols": symbols,
        "dollar_volume": np.tile([100.0, 1000.0], (len(dates), 1)),
        "entry_prices": prices,
        "morning_prices": exits,
        "entry_staleness": np.zeros_like(prices),
        "morning_staleness": np.zeros_like(prices),
        "start_date": dates[4],
        "end_date": dates[-1],
        "top": 1,
        "ema_span": 2,
        "min_history_days": 2,
        "minimum_trading_days": 2,
        "transaction_cost_bps": 0.0,
        "max_entry_staleness_minutes": 1,
        "max_exit_staleness_minutes": 1,
        "budget": 10_000.0,
        "entry_price_source": "nbbo-ask",
    }


class BacktestPlotTest(unittest.TestCase):
    def test_curve_matches_reported_levered_returns_and_benchmark_costs(self):
        args = inputs([0.012, -0.005, 0.004])
        args.update(
            leverage=2.0, margin_interest_rate=0.036, transaction_cost_bps=1.0
        )
        _, summary = backtest.run_backtest(**args)
        curve = equity_curve(summary)
        self.assertEqual(curve.strategy.iloc[0], 10000)
        self.assertAlmostEqual(
            curve.strategy.iloc[-1], summary["ending_equity"]
        )
        self.assertAlmostEqual(
            curve.spy_buy_and_hold.iloc[-1],
            10000 * (1 + summary["spy_buy_and_hold_metrics"]["total_return"]),
        )
        self.assertEqual(
            str(curve.index[0].date()), summary["first_entry_date"]
        )
        self.assertEqual(str(curve.index[-1].date()), summary["last_exit_date"])
        self.assertEqual(len(curve), 4)
        # Friday-to-Monday borrowing applies before the next entry.
        self.assertAlmostEqual(
            curve.strategy.iloc[1], 10000 * (1 + 2 * 0.0118 - 0.036 * 3 / 360)
        )

    def test_unknown_benchmark_return_is_not_treated_as_cash(self):
        _, summary = backtest.run_backtest(**inputs([0.01, 0, -0.01]))
        summary["daily_portfolio"][1]["spy_buy_and_hold_return"] = float("nan")
        curve = equity_curve(summary)
        self.assertTrue(np.isfinite(curve.spy_buy_and_hold.iloc[1]))
        self.assertTrue(curve.spy_buy_and_hold.iloc[2:].isna().all())
        self.assertTrue(np.isfinite(curve.strategy).all())

    def test_writes_unique_valid_webp_files_for_normalized_and_budgeted_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for budget in (None, 10000):
                args = inputs([0.02, -0.01, 0.003])
                args["budget"] = budget
                _, summary = backtest.run_backtest(**args)
                path = write_equity_plot(summary, Path(directory))
                paths.append(path)
                self.assertEqual(path.suffix, ".webp")
                with Image.open(path) as image:
                    self.assertEqual(image.format, "WEBP")
                    self.assertEqual(image.size, (1320, 720))
                    image.load()
            self.assertNotEqual(paths[0], paths[1])


if __name__ == "__main__":
    unittest.main()
