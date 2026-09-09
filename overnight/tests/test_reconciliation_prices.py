from datetime import date, time
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rich.console import Console

from trading_rl.overnight import reconcile_live_sessions as reconcile
from trading_rl.overnight.backtest import ENTRY_PRICE_SOURCES, EXIT_PRICE_SOURCES
from trading_rl.overnight.reconciliation_prices import MissingBenchmarkData
from overnight.tests.test_reconcile_live_sessions import order, seconds


class ReconciliationPricesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bars = self.root / "bars"
        self.bars.mkdir()
        self.entry_day = date(2026, 8, 28)
        self.exit_day = date(2026, 8, 31)
        self.values = np.asarray(
            [
                [
                    seconds(self.entry_day, time(15, 59)),
                    10000,
                    12000,
                    9000,
                    11000,
                    100,
                    10,
                    10500,
                ],
                [
                    seconds(self.exit_day, time(9, 30)),
                    20000,
                    22000,
                    19000,
                    21000,
                    100,
                    10,
                    20500,
                ],
                [
                    seconds(self.exit_day, time(9, 35)),
                    20000,
                    22000,
                    19000,
                    21000,
                    100,
                    10,
                    20500,
                ],
            ],
            dtype=np.int32,
        )
        np.save(self.bars / "AAPL.npy", self.values)
        self.auctions = self.root / "auctions.npz"
        np.savez(
            self.auctions,
            split_adjusted=True,
            symbol=["AAPL"],
            date=np.asarray(["2026-08-31"], dtype="datetime64[D]"),
            session=[0],
            condition=["O"],
            price=[20.0],
            raw_price=[40.0],
            size=[100],
            exchange=["Q"],
            split_symbol=["AAPL"],
            split_ex_date=np.asarray(["2026-09-01"], dtype="datetime64[D]"),
            split_old_rate=[1.0],
            split_new_rate=[2.0],
        )
        self.entry_nbbo = self.root / "entry.npz"
        self.exit_nbbo = self.root / "exit.npz"
        for path, day, clock, quote, price in [
            (self.entry_nbbo, "2026-08-28", "19:59:00", "19:58:59", 10.0),
            (self.exit_nbbo, "2026-08-31", "13:35:00", "13:34:59", 20.0),
        ]:
            np.savez(
                path,
                split_adjusted=True,
                symbol=["AAPL"],
                date=np.asarray([day], dtype="datetime64[D]"),
                target_timestamp=[f"{day}T{clock}Z"],
                timestamp=[f"{day}T{quote}Z"],
                ask_price=[price],
                raw_ask_price=[price * 2],
                ask_exchange=["Q"],
                bid_price=[price],
                raw_bid_price=[price * 2],
                bid_exchange=["Q"],
            )
        self.summary = {
            "configuration": {"entry_time": "15:59"},
            "position": {
                "status": "closed",
                "entry_date": "2026-08-28",
                "exit_date": "2026-08-31",
                "symbols": ["AAPL"],
                "entry_orders": {"AAPL": order(10, 20, "2026-08-28T19:59:00Z")},
                "exit_orders": {
                    "AAPL": order(
                        10, 40, "2026-08-31T13:30:00Z", "2026-08-31T12:00:00Z"
                    )
                },
            },
        }

    def run_reconcile(self, entry="nbbo-ask", exit="opening-auction", **kwargs):
        return reconcile.reconcile_execution(
            self.summary,
            self.bars,
            self.auctions,
            nbbo_path=self.entry_nbbo,
            exit_nbbo_path=self.exit_nbbo,
            entry_price_source=entry,
            exit_price_source=exit,
            exit_minute=570 if exit == "opening-auction" else 575,
            **kwargs,
        )

    def rewrite_npz(self, path, **changes):
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        np.savez(path, **(arrays | changes))

    def test_all_source_combinations_use_selected_fields_and_raw_split_factors(self):
        entry_prices = dict(zip(ENTRY_PRICE_SOURCES, [10, 12, 9, 11, 10.5, 10]))
        exit_prices = dict(zip(EXIT_PRICE_SOURCES, [20, 22, 19, 21, 20.5, 20, 20]))
        for entry, expected_entry in entry_prices.items():
            for exit, expected_exit in exit_prices.items():
                with self.subTest(entry=entry, exit=exit):
                    result = self.run_reconcile(entry, exit)
                    row = result["rows"][0]
                    self.assertEqual(result["entry_price_source"], entry)
                    self.assertEqual(result["exit_price_source"], exit)
                    expected_cost = 0.0 if (entry, exit) == ("nbbo-ask", "opening-auction") else 1.0
                    self.assertEqual(result["transaction_cost_bps_per_side"], expected_cost)
                    self.assertEqual(row["simulator_entry_price"], expected_entry)
                    self.assertEqual(row["simulator_exit_price"], expected_exit)
                    self.assertEqual(
                        row["simulator_entry_price_comparable"], expected_entry * 2
                    )
                    self.assertEqual(
                        row["simulator_exit_price_comparable"], expected_exit * 2
                    )
                    self.assertAlmostEqual(
                        result["totals"]["simulator_gross_pnl"],
                        200 * (expected_exit / expected_entry - 1),
                    )
                    impact = sum(
                        row[k]
                        for k in [
                            "entry_execution_pnl_impact",
                            "exit_execution_pnl_impact",
                            "quantity_pnl_impact",
                        ]
                    )
                    self.assertAlmostEqual(
                        impact, row["actual_minus_simulator_gross_pnl"]
                    )

    def test_missing_nbbo_never_uses_available_minute_bars(self):
        self.entry_nbbo.unlink()
        with self.assertRaisesRegex(
            MissingBenchmarkData, "nbbo-ask data does not exist"
        ):
            self.run_reconcile()

    def test_historical_nbbo_with_mixed_timestamp_precision_reconciles(self):
        with np.load(self.entry_nbbo, allow_pickle=False) as stored:
            arrays = {
                key: stored[key] if stored[key].ndim == 0 else np.repeat(stored[key], 2)
                for key in stored.files
            }
        arrays.update(
            date=np.asarray(["2022-01-03", "2026-08-28"], dtype="datetime64[D]"),
            target_timestamp=["2022-01-03T20:45:00.000Z", "2026-08-28T19:59:00Z"],
            timestamp=["2022-01-03T20:44:59.886123456Z", "2026-08-28T19:58:59Z"],
        )
        np.savez(self.entry_nbbo, **arrays)
        result = self.run_reconcile()
        row = result["rows"][0]
        self.assertEqual(row["simulator_entry_price"], 10.)
        self.assertEqual(row["simulator_entry_price_comparable"], 20.)
        self.assertEqual(row["entry_staleness_minutes"], 1 / 60)

    def test_explicit_transaction_cost_overrides_source_defaults(self):
        for entry, exit in [("nbbo-ask", "opening-auction"), ("minute-vwap", "minute-vwap")]:
            for cost in [0.0, 1.0, 2.5]:
                with self.subTest(entry=entry, exit=exit, cost=cost):
                    result = self.run_reconcile(entry, exit, transaction_cost_bps=cost)
                    self.assertEqual(result["transaction_cost_bps_per_side"], cost)
                    totals = result["totals"]
                    self.assertAlmostEqual(
                        totals["simulator_gross_pnl"] - totals["simulator_net_pnl"],
                        totals["simulator_transaction_cost"],
                    )
                    if cost == 0:
                        self.assertEqual(totals["simulator_transaction_cost"], 0.0)

    def test_one_missing_symbol_rejects_the_whole_basket(self):
        position = self.summary["position"]
        position["symbols"].append("MSFT")
        position["entry_orders"]["MSFT"] = dict(position["entry_orders"]["AAPL"])
        position["exit_orders"]["MSFT"] = dict(position["exit_orders"]["AAPL"])
        with self.assertRaisesRegex(MissingBenchmarkData, "MSFT"):
            self.run_reconcile()

    def test_minute_requires_exact_bar_and_selected_field(self):
        np.save(self.bars / "AAPL.npy", self.values[:2])
        with self.assertRaisesRegex(
            MissingBenchmarkData, "09:35 ET: exact bar is unavailable"
        ):
            self.run_reconcile("minute-open", "minute-vwap")
        self.values[2, 7] = 0
        np.save(self.bars / "AAPL.npy", self.values)
        with self.assertRaisesRegex(
            MissingBenchmarkData, "minute-vwap price is unavailable"
        ):
            self.run_reconcile("minute-open", "minute-vwap")

    def test_nbbo_requires_target_time_and_caps_quote_age_at_60_seconds(self):
        self.rewrite_npz(
            self.entry_nbbo,
            target_timestamp=["2026-08-28T19:45:00Z"],
            timestamp=["2026-08-28T19:44:59Z"],
        )
        with self.assertRaisesRegex(
            MissingBenchmarkData, "file contains a 15:45 ET snapshot"
        ):
            self.run_reconcile()
        self.rewrite_npz(
            self.entry_nbbo,
            target_timestamp=["2026-08-28T19:59:00Z"],
            timestamp=["2026-08-28T19:57:59Z"],
        )
        with self.assertRaisesRegex(MissingBenchmarkData, "61.00s old"):
            self.run_reconcile(max_entry_staleness_minutes=10)
        self.rewrite_npz(self.entry_nbbo, timestamp=["2026-08-28T19:58:00Z"])
        self.assertEqual(self.run_reconcile()["rows"][0]["entry_staleness_minutes"], 1)

    def test_corrupt_data_is_an_error_not_a_missing_price_skip(self):
        self.rewrite_npz(self.entry_nbbo, timestamp=["2026-08-28T19:59:01Z"])
        with self.assertRaises(ValueError) as error:
            self.run_reconcile()
        self.assertNotIsInstance(error.exception, MissingBenchmarkData)

    def test_split_effective_on_exit_day_uses_separate_entry_and_exit_factors(self):
        self.rewrite_npz(
            self.auctions,
            split_ex_date=np.asarray(["2026-08-31"], dtype="datetime64[D]"),
        )
        result = self.run_reconcile("minute-open", "minute-open")
        self.assertEqual(result["rows"][0]["simulator_entry_price_comparable"], 20)
        self.assertEqual(result["rows"][0]["simulator_exit_price_comparable"], 20)

    def test_minute_prices_use_split_ledger_without_requiring_exit_auction_print(self):
        self.rewrite_npz(
            self.auctions, date=np.asarray(["2026-08-27"], dtype="datetime64[D]")
        )
        result = self.run_reconcile("minute-vwap", "minute-close")
        self.assertEqual(result["rows"][0]["simulator_exit_price"], 21)

    def test_continuous_exit_schedule_does_not_require_auction_submission(self):
        self.summary["position"]["exit_orders"]["AAPL"] = order(
            10, 40, "2026-08-31T13:35:00Z"
        )
        timing, warnings = reconcile.summary_execution_timing(
            self.summary, exit_minute=575, exit_price_source="nbbo-bid"
        )
        self.assertTrue(timing["schedule_comparable"])
        self.assertEqual(warnings, [])
        timing, _ = reconcile.summary_execution_timing(self.summary)
        self.assertFalse(timing["schedule_comparable"])

    def test_forensic_mode_uses_explicit_selected_fields_at_fill_minutes(self):
        result = self.run_reconcile(
            "minute-vwap", "minute-close", actual_time_benchmark=True
        )
        self.assertEqual(result["rows"][0]["actual_time_entry_price"], 10.5)
        self.assertEqual(result["rows"][0]["actual_time_exit_price"], 21)
        with self.assertRaisesRegex(ValueError, "requires explicit minute-"):
            self.run_reconcile(actual_time_benchmark=True)

    def test_cli_skips_whole_incomplete_session_and_preserves_matched_totals(self):
        work = self.root / "work"
        output_dir = self.root / "out"
        output_dir.mkdir()
        for day, exit_day in [
            ("2026-08-28", "2026-08-31"),
            ("2026-08-31", "2026-09-01"),
        ]:
            summary = json.loads(json.dumps(self.summary))
            position = summary["position"]
            position.update(entry_date=day, exit_date=exit_day)
            position["entry_orders"]["AAPL"]["filled_at"] = f"{day}T19:59:00Z"
            position["exit_orders"]["AAPL"]["filled_at"] = f"{exit_day}T13:30:00Z"
            position["exit_orders"]["AAPL"]["submitted_at"] = f"{exit_day}T12:00:00Z"
            folder = work / day
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(summary))
        stale_csv = output_dir / "2026-08-31.csv"
        stale_csv.write_text("old fallback output")
        argv = [
            "trading-reconcile",
            "--since",
            "2026-08-28",
            "--work-dir",
            str(work),
            "--nbbo-path",
            str(self.entry_nbbo),
            "--auctions-path",
            str(self.auctions),
            "--skip-broker-fees",
            "--skip-ranking-replay",
            "--output-dir",
            str(output_dir),
        ]
        stream = StringIO()
        with (
            patch("sys.argv", argv),
            patch.object(reconcile, "CONSOLE", Console(file=stream, width=180)),
        ):
            reconcile.main()
        text = stream.getvalue()
        self.assertIn("Missing-price sessions", text)
        self.assertIn("Warning (2026-08-31): skipped entire session", text)
        self.assertGreater(text.index("Warning ("), text.rfind("└"))
        self.assertFalse(stale_csv.exists())
        completed = json.loads((output_dir / "2026-08-28.json").read_text())
        self.assertEqual(completed["totals"]["actual_gross_pnl"], 200)
        self.assertEqual(completed["totals"]["simulator_gross_pnl"], 200)
        self.assertEqual(completed["transaction_cost_bps_per_side"], 0.0)
        self.assertEqual(completed["totals"]["simulator_net_pnl"], 200)
        skipped = json.loads((output_dir / "2026-08-31.json").read_text())
        self.assertEqual(skipped["status"], "skipped")

        # A run with no comparable prices completes with an explicit empty result,
        # without retaining a prior successful table or failing as a data error.
        stream = StringIO()
        only_missing = [argv[0], *argv[3:], "--entry-date", "2026-08-31"]
        with patch("sys.argv", only_missing), patch.object(
            reconcile, "CONSOLE", Console(file=stream, width=180)
        ):
            reconcile.main()
        self.assertIn("No sessions with complete benchmark prices", stream.getvalue())
        self.assertIn("skipped entire session", stream.getvalue())

    def test_cli_sources_match_backtester_and_reject_aliases(self):
        parser = reconcile.build_parser()
        args = parser.parse_args([])
        self.assertEqual(
            (args.entry_price_source, args.exit_price_source),
            ("nbbo-ask", "opening-auction"),
        )
        for flag in ["--entry-price-source", "--exit-price-source"]:
            with (
                patch("sys.stderr", new_callable=StringIO),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([flag, "minute-bar"])
