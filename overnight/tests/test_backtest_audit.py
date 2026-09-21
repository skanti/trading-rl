import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tests.bar_fixtures import ohlcv_fixture
from trading_rl.overnight.backtest_audit import minute_mark_audit


class MinuteAuditTest(unittest.TestCase):
    def test_intrahold_loss_and_weekend_interest_reconcile_to_liquidation(self):
        stamps = pd.to_datetime(
            ["2026-01-02T20:45Z", "2026-01-02T21:00Z", "2026-01-05T14:30Z"]
        )
        seconds = (
            (stamps - pd.Timestamp("2010-01-01", tz="UTC")).total_seconds()
        ).astype(np.int64)
        # Minute exit says 95; the modeled auction liquidation is 90.
        bars = ohlcv_fixture(np.column_stack((seconds, [100000, 80000, 95000])))
        borrow = 1000 * 0.0675 * 3 / 360
        end_equity = 1000 - 200 - 0.4 - borrow
        trades = pd.DataFrame(
            [
                {
                    "entry_date": "2026-01-02",
                    "sample_id": "A",
                    "quantity": 20.0,
                    "exit_price": 90.0,
                    "entry_notional": 2000.0,
                    "transaction_cost_dollars": 0.4,
                }
            ]
        )
        summary = {
            "budget": 1000.0,
            "entry_time_eastern": "15:45",
            "exit_time_eastern": "09:30",
            "daily_portfolio": [
                {
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-01-05",
                    "portfolio_start_equity": 1000.0,
                    "portfolio_end_equity": end_equity,
                    "borrow_cost": borrow,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "A.npy", bars)
            marks, audit = minute_mark_audit(trades, summary, Path(directory))
            self.assertGreater(audit["minute_open_mark_drawdown"], 0.4)
            self.assertLess(audit["max_endpoint_error"], 1e-12)
            self.assertEqual(len(marks), 1)
            self.assertGreater(audit["sampled_minute_marks"], 1000)
            summary["daily_portfolio"][0]["portfolio_end_equity"] += 1
            with self.assertRaisesRegex(ValueError, "does not reconcile"):
                minute_mark_audit(trades, summary, Path(directory))


if __name__ == "__main__":
    unittest.main()
