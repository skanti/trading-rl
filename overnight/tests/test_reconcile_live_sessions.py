import json
import tempfile
import unittest
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np

from backtest import BAR_ORIGIN, EASTERN
from reconcile_live_sessions import (
    attach_broker_fees,
    execution_context,
    format_usd,
    infer_entry_minute,
    pnl_comparison_context,
    reconcile_execution,
    replay_ranking,
    summarize_broker_fees,
)


def seconds(day: date, clock: time) -> int:
    stamp = datetime.combine(day, clock, tzinfo=EASTERN).astimezone(UTC)
    return int((stamp - BAR_ORIGIN).total_seconds())


def order(quantity: float, price: float, filled_at: str) -> dict[str, object]:
    return {
        "status": "filled",
        "filled_qty": str(quantity),
        "filled_avg_price": str(price),
        "filled_at": filled_at,
    }


class ReconcileLiveSessionsTest(unittest.TestCase):
    def test_formats_currency_sign_before_symbol(self):
        self.assertEqual(format_usd(-10.26, signed=True), "-$10.26")
        self.assertEqual(format_usd(1.13, signed=True), "+$1.13")
        self.assertEqual(format_usd(1.96), "$1.96")
        self.assertEqual(
            pnl_comparison_context(1.13, label="gross P&L", bps=1.15),
            "Simulator gross P&L lower by $1.13 (1.15 bps)",
        )
        self.assertEqual(
            execution_context(-1.15, side="entry"),
            "Simulator entry price 1.15 bps higher than actual",
        )
        self.assertEqual(
            execution_context(0.0, side="exit"),
            "Simulator exit price matches actual (0.00 bps)",
        )

    def test_reconciles_actual_fills_with_minute_open_and_auction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minute_dir = root / "minute"
            minute_dir.mkdir()
            entry_day = date(2026, 8, 28)
            exit_day = date(2026, 8, 31)
            np.save(
                minute_dir / "AAPL.npy",
                np.asarray(
                    [
                        [seconds(entry_day, time(15, 58)), 99_000, 100, 10],
                        [seconds(entry_day, time(15, 59)), 100_000, 100, 10],
                    ],
                    dtype=np.int32,
                ),
            )
            auctions = root / "auctions.npz"
            np.savez_compressed(
                auctions,
                format_version=np.asarray(1, dtype=np.int16),
                split_adjusted=np.asarray(True),
                symbol=np.asarray(["AAPL"]),
                date=np.asarray([exit_day.isoformat()], dtype="datetime64[D]"),
                session=np.asarray([0], dtype=np.uint8),
                condition=np.asarray(["O"]),
                price=np.asarray([110.0]),
                raw_price=np.asarray([110.0]),
                size=np.asarray([1_000.0]),
                exchange=np.asarray(["Q"]),
            )
            summary = {
                "configuration": {"entry_time": "15:59"},
                "position": {
                    "status": "closed",
                    "entry_date": entry_day.isoformat(),
                    "exit_date": exit_day.isoformat(),
                    "symbols": ["AAPL"],
                    "entry_orders": {
                        "AAPL": order(10, 100.5, "2026-08-28T19:59:00.2Z")
                    },
                    "exit_orders": {"AAPL": order(10, 110, "2026-08-31T13:30:00.2Z")},
                    "entry_account_snapshot": {"equity": "1000"},
                    "exit_account_snapshot": {"equity": "1095"},
                },
            }

            result = reconcile_execution(
                summary,
                minute_dir,
                auctions,
                transaction_cost_bps=1.0,
            )

            row = result["rows"][0]
            totals = result["totals"]
            self.assertEqual(result["entry_time"], "15:59")
            self.assertAlmostEqual(row["simulator_entry_price"], 100.0)
            self.assertAlmostEqual(row["simulator_exit_price"], 110.0)
            self.assertAlmostEqual(row["entry_slippage_bps"], 50.0)
            self.assertAlmostEqual(totals["actual_gross_pnl"], 95.0)
            self.assertAlmostEqual(totals["simulator_gross_pnl"], 100.5)
            self.assertAlmostEqual(totals["actual_minus_simulator_gross_pnl"], -5.5)
            self.assertAlmostEqual(totals["entry_execution_slippage_bps"], 50.0)
            self.assertAlmostEqual(totals["exit_execution_slippage_bps"], 0.0)
            self.assertAlmostEqual(totals["entry_execution_pnl_impact"], -5.5)
            self.assertAlmostEqual(totals["exit_execution_pnl_impact"], 0.0)
            self.assertAlmostEqual(totals["quantity_pnl_impact"], 0.0)
            self.assertAlmostEqual(
                totals["entry_execution_pnl_impact"]
                + totals["exit_execution_pnl_impact"]
                + totals["quantity_pnl_impact"],
                totals["actual_minus_simulator_gross_pnl"],
            )
            self.assertAlmostEqual(totals["broker_minus_actual_fill_pnl"], 0.0)

    def test_uses_exit_date_fee_activities_for_actual_net_pnl(self):
        fees = summarize_broker_fees(
            [
                {
                    "activity_type": "FEE",
                    "activity_sub_type": "REG",
                    "date": "2026-08-31",
                    "net_amount": "-0.21",
                    "description": "REG fee on 2026-08-31 by account-number",
                },
                {
                    "activity_type": "FEE",
                    "activity_sub_type": "TAF",
                    "date": "2026-08-31",
                    "net_amount": "-0.01",
                },
                {
                    "activity_type": "FEE",
                    "activity_sub_type": "REG",
                    "date": "2026-08-28",
                    "net_amount": "-0.14",
                },
            ],
            date(2026, 8, 31),
        )
        result = {
            "totals": {
                "actual_entry_notional": 1005.0,
                "actual_gross_pnl": -10.26,
                "simulator_net_pnl": -13.35,
                "broker_equity_pnl": -10.50,
            }
        }

        attach_broker_fees(result, fees)

        totals = result["totals"]
        self.assertEqual(fees["count"], 2)
        self.assertAlmostEqual(fees["cost"], 0.22)
        self.assertAlmostEqual(fees["breakdown"]["REG"]["cost"], 0.21)
        self.assertNotIn("account-number", fees["activities"][0]["description"])
        self.assertAlmostEqual(totals["actual_net_pnl_after_broker_fees"], -10.48)
        self.assertAlmostEqual(totals["actual_minus_simulator_net_pnl"], 2.87)
        self.assertAlmostEqual(
            totals["actual_minus_simulator_net_bps"], 2.87 / 1005 * 10_000
        )
        self.assertAlmostEqual(totals["broker_minus_fee_adjusted_fill_pnl"], -0.02)

    def test_entry_time_prefers_archived_configuration(self):
        summary = {
            "configuration": {"entry_time": "15:45"},
            "position": {"entry_dispatch_target_at": "2026-08-28T19:59:00Z"},
        }

        self.assertEqual(infer_entry_minute(summary), 15 * 60 + 45)
        self.assertEqual(infer_entry_minute(summary, 15 * 60 + 40), 15 * 60 + 40)

    def test_replays_ranking_from_archived_daily_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            ticks = Path(directory) / "ticks.jsonl"
            rows = []
            for day in ("2026-08-25", "2026-08-26", "2026-08-27"):
                rows.extend(
                    [
                        {
                            "symbol": "A",
                            "t": f"{day}T04:00:00Z",
                            "v": 1_000,
                            "vw": 100,
                            "c": 100,
                        },
                        {
                            "symbol": "B",
                            "t": f"{day}T04:00:00Z",
                            "v": 100,
                            "vw": 100,
                            "c": 100,
                        },
                    ]
                )
            ticks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {
                "configuration": {
                    "top": 1,
                    "ema_span": 2,
                    "min_history_days": 2,
                    "minimum_trading_days": 2,
                    "liquidity_scheme": "dollar_ema",
                },
                "ranking": {},
                "position": {
                    "entry_date": "2026-08-28",
                    "symbols": ["A"],
                },
            }

            replay = replay_ranking(summary, ticks)

            self.assertEqual(replay["replayed_symbols"], ["A"])
            self.assertEqual(replay["overlap_count"], 1)
            self.assertEqual(replay["jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
