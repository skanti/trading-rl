import json
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from scripts.tests.bar_fixtures import ohlcv_fixture
from rich.console import Console

from trading_rl.overnight.history import (
    BAR_ORIGIN,
    EASTERN,
)
from trading_rl.overnight.broker_fees import summarize_broker_fees
from trading_rl.overnight.reconciliation_prices import MissingBenchmarkData
from trading_rl.overnight.reconcile_live_sessions import (
    attach_broker_fees,
    broker_fees_for_session,
    build_parser,
    execution_timing,
    execution_context,
    format_usd,
    infer_entry_minute,
    pnl_comparison_context,
    print_overview,
    reconcile_execution,
    replay_ranking,
    select_reporting_benchmark,
    summary_execution_timing,
    summarize_results,
)


def seconds(day: date, clock: time) -> int:
    stamp = datetime.combine(day, clock, tzinfo=EASTERN).astimezone(UTC)
    return int((stamp - BAR_ORIGIN).total_seconds())


def order(
    quantity: float,
    price: float,
    filled_at: str,
    submitted_at: str | None = None,
) -> dict[str, object]:
    return {
        "status": "filled",
        "filled_qty": str(quantity),
        "filled_avg_price": str(price),
        "filled_at": filled_at,
        "submitted_at": submitted_at or filled_at,
    }


class ReconcileLiveSessionsTest(unittest.TestCase):
    def test_reconciliation_mode_defaults_to_strict_schedule(self):
        parser = build_parser()

        self.assertEqual(
            parser.parse_args([]).reconciliation_mode,
            "strict-schedule",
        )
        with patch("sys.stderr", new_callable=StringIO), self.assertRaises(SystemExit) as error:
            parser.parse_args(["--reconciliation-mode", "actual-time"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(
            parser.parse_args(
                ["--reconciliation-mode", "actual-time-minute-bar"]
            ).reconciliation_mode,
            "actual-time-minute-bar",
        )

    def test_formats_currency_sign_before_symbol(self):
        self.assertEqual(format_usd(-10.26, signed=True), "-$10.26")
        self.assertEqual(format_usd(1.13, signed=True), "+$1.13")
        self.assertEqual(format_usd(1.96), "$1.96")
        self.assertEqual(
            pnl_comparison_context(1.13, label="gross P&L", bps=1.15),
            "Sim gross P&L < actual by $1.13 (1.15 bps)",
        )
        self.assertEqual(
            execution_context(-1.15, side="entry"),
            "Sim entry price > actual by 1.15 bps",
        )
        self.assertEqual(
            execution_context(0.0, side="exit"),
            "Sim exit price = actual (0.00 bps)",
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
                ohlcv_fixture(np.asarray(
                    [
                        [seconds(entry_day, time(15, 58)), 99_000, 100, 10],
                        [seconds(entry_day, time(15, 59)), 100_000, 100, 10],
                    ],
                    dtype=np.int32,
                )),
            )
            auctions = root / "auctions.npz"
            np.savez_compressed(
                auctions,
                format_version=np.asarray(1, dtype=np.int16),
                split_adjusted=np.asarray(True),
                split_symbol=np.asarray([], dtype=str),
                split_ex_date=np.asarray([], dtype="datetime64[D]"),
                split_old_rate=np.asarray([], dtype=float),
                split_new_rate=np.asarray([], dtype=float),
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
                    "exit_orders": {
                        "AAPL": order(
                            10,
                            110,
                            "2026-08-31T13:30:00.2Z",
                            "2026-08-31T12:00:00Z",
                        )
                    },
                    "entry_account_snapshot": {"equity": "1000"},
                    "exit_account_snapshot": {"equity": "1095"},
                },
            }

            result = reconcile_execution(
                summary,
                minute_dir,
                auctions,
                entry_price_source="minute-open",
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
            self.assertTrue(result["timing"]["schedule_comparable"])
            self.assertTrue(result["timing"]["exit"]["opening_auction_comparable"])
            self.assertEqual(result["warnings"], [])

            nbbo = root / "nbbo.npz"
            np.savez_compressed(
                nbbo,
                format_version=np.asarray(1, dtype=np.int16),
                split_adjusted=np.asarray(True),
                split_symbol=np.asarray([], dtype=str),
                split_ex_date=np.asarray([], dtype="datetime64[D]"),
                split_old_rate=np.asarray([], dtype=float),
                split_new_rate=np.asarray([], dtype=float),
                symbol=np.asarray(["AAPL"]),
                date=np.asarray([entry_day.isoformat()], dtype="datetime64[D]"),
                target_timestamp=np.asarray(["2026-08-28T19:59:00Z"]),
                timestamp=np.asarray(["2026-08-28T19:58:59Z"]),
                ask_price=np.asarray([101.0]),
                raw_ask_price=np.asarray([101.0]),
                ask_exchange=np.asarray(["Q"]),
            )
            nbbo_result = reconcile_execution(
                summary,
                minute_dir,
                auctions,
                nbbo_path=nbbo,
                transaction_cost_bps=1.0,
            )

            self.assertEqual(nbbo_result["entry_price_source"], "nbbo-ask")
            self.assertEqual(
                nbbo_result["rows"][0]["simulator_entry_source"], "nbbo-ask"
            )
            self.assertAlmostEqual(
                nbbo_result["rows"][0]["simulator_entry_price"], 101.0
            )
            self.assertEqual(nbbo_result["warnings"], [])

            # Existing quotes at a different scheduled time must not be used or
            # reported as missing at that file's time rather than the entry time.
            with np.load(nbbo, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays["target_timestamp"] = np.asarray(["2026-08-28T19:45:00Z"])
            arrays["timestamp"] = np.asarray(["2026-08-28T19:44:59Z"])
            np.savez_compressed(nbbo, **arrays)
            with self.assertRaisesRegex(MissingBenchmarkData, "15:59 ET: file contains a 15:45 ET snapshot"):
                reconcile_execution(summary, minute_dir, auctions, nbbo_path=nbbo)

    def test_off_schedule_exit_adds_per_symbol_actual_time_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minute_dir = root / "minute"
            minute_dir.mkdir()
            entry_day = date(2026, 8, 28)
            exit_day = date(2026, 8, 31)
            np.save(
                minute_dir / "AAPL.npy",
                ohlcv_fixture(np.asarray(
                    [
                        [seconds(entry_day, time(15, 59)), 100_000, 100, 10],
                        [seconds(exit_day, time(12, 45)), 109_000, 100, 10],
                    ],
                    dtype=np.int32,
                )),
            )
            auctions = root / "auctions.npz"
            np.savez_compressed(
                auctions,
                format_version=np.asarray(1, dtype=np.int16),
                split_adjusted=np.asarray(True),
                split_symbol=np.asarray([], dtype=str),
                split_ex_date=np.asarray([], dtype="datetime64[D]"),
                split_old_rate=np.asarray([], dtype=float),
                split_new_rate=np.asarray([], dtype=float),
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
                        "AAPL": order(10, 100.0, "2026-08-28T19:59:00.2Z")
                    },
                    "exit_orders": {
                        "AAPL": order(
                            10,
                            109.1,
                            "2026-08-31T16:45:03Z",
                            "2026-08-31T16:45:00Z",
                        )
                    },
                },
            }

            result = reconcile_execution(
                summary, minute_dir, auctions, entry_price_source="minute-open",
                exit_price_source="minute-open", actual_time_benchmark=True,
            )

        timing = result["timing"]
        self.assertFalse(timing["schedule_comparable"])
        self.assertFalse(timing["exit"]["submitted_before_auction_cutoff"])
        self.assertFalse(timing["exit"]["opening_auction_comparable"])
        self.assertAlmostEqual(timing["exit"]["minimum_offset_minutes"], 195.05)
        self.assertEqual(
            timing["exit"]["actual_time_benchmark"],
            {
                "alignment": "per_symbol_fill_minute",
                "price_source": "minute-open",
            },
        )
        warning = result["warnings"][0]
        self.assertIn("exit is off schedule for minute-open", warning)
        self.assertIn("1 exceed the 1-min tolerance", warning)
        self.assertNotIn("AAPL", warning)
        row = result["rows"][0]
        self.assertEqual(row["actual_time_entry_bar_at"], "2026-08-28T15:59:00-04:00")
        self.assertEqual(row["actual_time_entry_price"], 100.0)
        self.assertEqual(row["actual_time_exit_bar_at"], "2026-08-31T12:45:00-04:00")
        self.assertEqual(row["actual_time_exit_price"], 109.0)
        self.assertAlmostEqual(
            result["totals"]["actual_time_exit_slippage_bps"],
            (109.1 / 109.0 - 1.0) * 10_000.0,
        )
        self.assertAlmostEqual(
            result["totals"]["actual_time_simulator_gross_pnl"], 90.0
        )
        self.assertAlmostEqual(
            result["totals"]["actual_minus_actual_time_simulator_gross_pnl"],
            1.0,
        )
        self.assertEqual(
            select_reporting_benchmark(result, prefer_actual_time=True),
            "actual_time_1_min",
        )

    def test_off_schedule_entry_is_not_schedule_comparable(self):
        entry_day = date(2026, 8, 28)
        exit_day = date(2026, 8, 31)
        timing, warnings = execution_timing(
            {"AAPL": order(10, 100, "2026-08-28T20:02:00Z")},
            {
                "AAPL": order(
                    10,
                    110,
                    "2026-08-31T13:30:00.2Z",
                    "2026-08-31T12:00:00Z",
                )
            },
            ["AAPL"],
            entry_day,
            exit_day,
            15 * 60 + 59,
            1.0,
        )

        self.assertFalse(timing["schedule_comparable"])
        self.assertFalse(timing["entry"]["comparable"])
        self.assertTrue(timing["exit"]["opening_auction_comparable"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("1/1 symbols exceed the 1-min tolerance", warnings[0])
        self.assertIn("3.0–3.0 minutes after the scheduled 15:59 ET entry", warnings[0])
        self.assertNotIn("AAPL", warnings[0])

    def test_actual_time_mode_uses_the_entry_fill_minute_per_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minute_dir = root / "minute"
            minute_dir.mkdir()
            entry_day = date(2026, 8, 28)
            exit_day = date(2026, 8, 31)
            np.save(
                minute_dir / "AAPL.npy",
                ohlcv_fixture(np.asarray(
                    [
                        [seconds(entry_day, time(15, 59)), 100_000, 100, 10],
                        [seconds(entry_day, time(16, 2)), 101_000, 100, 10],
                        [seconds(exit_day, time(9, 30)), 110_000, 100, 10],
                    ],
                    dtype=np.int32,
                )),
            )
            auctions = root / "auctions.npz"
            np.savez_compressed(
                auctions,
                format_version=np.asarray(1, dtype=np.int16),
                split_adjusted=np.asarray(True),
                split_symbol=np.asarray([], dtype=str),
                split_ex_date=np.asarray([], dtype="datetime64[D]"),
                split_old_rate=np.asarray([], dtype=float),
                split_new_rate=np.asarray([], dtype=float),
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
                    "entry_orders": {"AAPL": order(10, 101, "2026-08-28T20:02:03Z")},
                    "exit_orders": {
                        "AAPL": order(
                            10,
                            110,
                            "2026-08-31T13:30:02Z",
                            "2026-08-31T12:00:00Z",
                        )
                    },
                },
            }

            result = reconcile_execution(
                summary, minute_dir, auctions, entry_price_source="minute-open",
                exit_price_source="minute-open", actual_time_benchmark=True,
            )
            select_reporting_benchmark(result, prefer_actual_time=True)

        row = result["rows"][0]
        self.assertEqual(row["simulator_entry_price"], 101.0)
        self.assertEqual(row["actual_time_entry_price"], 101.0)
        self.assertEqual(row["actual_time_exit_price"], 110.0)
        self.assertEqual(row["actual_time_entry_bar_at"], "2026-08-28T16:02:00-04:00")
        self.assertEqual(result["reporting_benchmark"], "actual_time_1_min")
        self.assertAlmostEqual(
            result["totals"]["actual_time_simulator_gross_pnl"], 90.0
        )
        self.assertAlmostEqual(
            result["totals"]["actual_minus_actual_time_simulator_gross_pnl"],
            0.0,
        )
        self.assertAlmostEqual(
            row["actual_time_entry_execution_pnl_impact"]
            + row["actual_time_exit_execution_pnl_impact"]
            + row["quantity_pnl_impact"],
            row["actual_minus_actual_time_simulator_gross_pnl"],
        )

    def test_summary_timing_can_skip_off_schedule_session_without_market_data(self):
        summary = {
            "configuration": {"entry_time": "15:59"},
            "position": {
                "status": "closed",
                "entry_date": "2026-09-02",
                "exit_date": "2026-09-03",
                "symbols": ["AAPL"],
                "entry_orders": {"AAPL": order(10, 100, "2026-09-02T19:59:00Z")},
                "exit_orders": {
                    "AAPL": order(
                        10,
                        101,
                        "2026-09-03T16:45:00Z",
                        "2026-09-03T16:44:00Z",
                    )
                },
            },
        }

        timing, warnings = summary_execution_timing(summary)

        self.assertFalse(timing["schedule_comparable"])
        self.assertTrue(timing["entry"]["comparable"])
        self.assertFalse(timing["exit"]["opening_auction_comparable"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("exit is not opening-auction comparable", warnings[0])

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

    def test_empty_fee_response_is_cached_as_pending_during_grace_period(self):
        class Client:
            def account_activities(self, *_args, **_kwargs):
                return []

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "fee_activities.json"
            summary, warning = broker_fees_for_session(
                Client(),
                cache,
                date(2026, 9, 1),
                as_of=datetime(2026, 9, 1, 15, 0, tzinfo=UTC),
            )
            persisted = json.loads(cache.read_text())

        self.assertEqual(summary["status"], "pending")
        self.assertEqual(persisted["status"], "pending")
        self.assertEqual(summary["count"], 0)
        self.assertIn("remains provisional", str(warning))

    def test_empty_fee_response_confirms_after_grace_period(self):
        summary = summarize_broker_fees(
            [],
            date(2026, 9, 1),
            fetched_at=datetime(2026, 9, 3, 15, 0, tzinfo=UTC),
        )

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["cost"], 0.0)

    def test_overview_uses_dollar_totals_and_notional_weighted_bps(self):
        results = [
            {
                "entry_date": "2026-08-28",
                "exit_date": "2026-08-31",
                "symbols": ["A", "B"],
                "transaction_cost_bps_per_side": 1.0,
                "broker_fees": {"status": "complete"},
                "ranking_replay": {"status": "complete", "overlap_count": 1},
                "totals": {
                    "actual_entry_notional": 1_000.0,
                    "actual_gross_pnl": 10.0,
                    "simulator_gross_pnl": 8.0,
                    "entry_execution_slippage_bps": 2.0,
                    "exit_execution_slippage_bps": -1.0,
                    "actual_broker_fee_cost": 1.0,
                    "simulator_transaction_cost": 2.0,
                    "actual_net_pnl_after_broker_fees": 9.0,
                    "simulator_net_pnl": 6.0,
                    "broker_equity_pnl": 9.5,
                    "broker_minus_fee_adjusted_fill_pnl": 0.5,
                },
            },
            {
                "entry_date": "2026-08-31",
                "exit_date": "2026-09-01",
                "symbols": ["C"],
                "transaction_cost_bps_per_side": 1.0,
                "broker_fees": {"status": "pending"},
                "totals": {
                    "actual_entry_notional": 3_000.0,
                    "actual_gross_pnl": 30.0,
                    "simulator_gross_pnl": 33.0,
                    "entry_execution_slippage_bps": 6.0,
                    "exit_execution_slippage_bps": 3.0,
                    "simulator_transaction_cost": 6.0,
                    "simulator_net_pnl": 27.0,
                    "broker_equity_pnl": None,
                },
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["trades"], 3)
        self.assertAlmostEqual(summary["actual_gross_pnl_total"], 40.0)
        self.assertAlmostEqual(summary["simulator_gross_pnl_total"], 41.0)
        self.assertAlmostEqual(summary["actual_minus_simulator_gross_pnl_total"], -1.0)
        self.assertAlmostEqual(summary["actual_minus_simulator_gross_bps"], -2.5)
        self.assertAlmostEqual(summary["entry_execution_slippage_bps"], 5.0)
        self.assertAlmostEqual(summary["exit_execution_slippage_bps"], 2.0)
        self.assertEqual(summary["fee_confirmed_sessions"], 1)
        self.assertAlmostEqual(summary["actual_broker_fee_cost_total"], 1.0)
        self.assertAlmostEqual(summary["simulator_transaction_cost_total"], 2.0)
        self.assertAlmostEqual(summary["actual_net_pnl_total"], 9.0)
        self.assertAlmostEqual(summary["simulator_net_pnl_total"], 6.0)
        self.assertAlmostEqual(summary["actual_minus_simulator_net_pnl_total"], 3.0)
        self.assertAlmostEqual(summary["unexplained_residual_total"], 0.5)
        self.assertAlmostEqual(summary["ranking_overlap_fraction"], 0.5)

    def test_overview_uses_selected_actual_time_entry_and_exit_benchmarks(self):
        result = {
            "entry_date": "2026-09-02",
            "exit_date": "2026-09-03",
            "symbols": ["A"],
            "reporting_benchmark": "actual_time_1_min",
            "transaction_cost_bps_per_side": 1.0,
            "broker_fees": {"status": "pending"},
            "totals": {
                "actual_entry_notional": 1_000.0,
                "actual_gross_pnl": 91.0,
                "simulator_gross_pnl": 50.0,
                "actual_time_simulator_gross_pnl": 90.0,
                "entry_execution_slippage_bps": 0.0,
                "actual_time_entry_slippage_bps": 2.0,
                "exit_execution_slippage_bps": 100.0,
                "actual_time_exit_slippage_bps": 10.0,
                "simulator_transaction_cost": 2.0,
                "actual_time_simulator_transaction_cost": 2.0,
                "simulator_net_pnl": 48.0,
                "actual_time_simulator_net_pnl": 88.0,
                "broker_equity_pnl": None,
            },
        }

        summary = summarize_results([result])

        self.assertEqual(summary["actual_time_benchmark_sessions"], 1)
        self.assertAlmostEqual(summary["simulator_gross_pnl_total"], 90.0)
        self.assertAlmostEqual(summary["actual_minus_simulator_gross_pnl_total"], 1.0)
        self.assertAlmostEqual(summary["entry_execution_slippage_bps"], 2.0)
        self.assertAlmostEqual(summary["exit_execution_slippage_bps"], 10.0)
        self.assertAlmostEqual(summary["simulator_net_pnl_total"], 88.0)

        output = StringIO()
        console = Console(file=output, force_terminal=False, width=160)
        with patch("trading_rl.overnight.reconcile_live_sessions.CONSOLE", console):
            print_overview(
                [result],
                skipped_sessions=[
                    (
                        date(2026, 9, 2),
                        ["exit is not opening-auction comparable: late fills"],
                    )
                ],
            )
        rendered = output.getvalue()
        self.assertIn("Skipped sessions", rendered)
        self.assertIn("2026-09-02: exit schedule mismatch", rendered)

    def test_actual_time_mode_does_not_fall_back_to_scheduled_benchmark(self):
        result = {
            "totals": {"actual_time_simulator_gross_pnl": None},
            "warnings": ["actual-time entry 1-min benchmark is unavailable"],
        }

        with self.assertRaisesRegex(
            ValueError,
            "requires complete per-symbol entry and exit 1-min benchmarks",
        ):
            select_reporting_benchmark(result, prefer_actual_time=True)

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

    def test_infers_missing_scheme_from_archived_ranking_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            ticks = Path(directory) / "ticks.jsonl"
            rows = []
            for day_index in range(15):
                day = (date(2026, 8, 1) + timedelta(days=day_index)).isoformat()
                for symbol in ("A", "B", "C"):
                    volume = {
                        "A": 10_000 if day_index % 2 else 100,
                        "B": 3_000,
                        "C": 5_000,
                    }[symbol]
                    rows.append(
                        {
                            "symbol": symbol,
                            "t": f"{day}T04:00:00Z",
                            "v": volume,
                            "vw": 100,
                            "c": 100,
                        }
                    )
            ticks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {
                "configuration": {
                    "top": 2,
                    "ema_span": 2,
                    "min_history_days": 2,
                    "minimum_trading_days": 2,
                },
                "ranking": {},
                "position": {
                    "entry_date": "2026-08-20",
                    "symbols": ["C", "B"],
                },
            }
            explicit = replay_ranking(
                summary, ticks, liquidity_scheme_override="dollar_ema"
            )
            summary["ranking"]["candidates"] = explicit["top_ranking"]

            inferred = replay_ranking(summary, ticks)

            self.assertEqual(inferred["liquidity_scheme"], "dollar_ema")
            self.assertEqual(
                inferred["liquidity_scheme_source"], "archived_ranking_scores"
            )
            self.assertEqual(inferred["warnings"], [])


if __name__ == "__main__":
    unittest.main()
