import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from trading_rl.overnight.backtest import (
    build_parser,
    load_scheduled_nbbo_prices,
    print_summary_table,
    run_backtest,
)


class MissingNbboTest(unittest.TestCase):
    def inputs(self):
        dates = pd.date_range("2026-08-24", periods=5, freq="B")
        prices = np.full((5, 4), 100.0)
        return dict(
            dates=dates,
            symbols=np.array(["SPY", "A", "B", "C"]),
            dollar_volume=np.array([[1000.0, 4000.0, 3000.0, 2000.0]] * 5),
            entry_prices=prices.copy(),
            morning_prices=prices * 1.1,
            entry_staleness=np.zeros_like(prices),
            morning_staleness=np.zeros_like(prices),
            start_date=dates[2],
            end_date=dates[4],
            top=2,
            ema_span=1,
            min_history_days=1,
            minimum_trading_days=1,
            transaction_cost_bps=1.0,
            max_entry_staleness_minutes=10,
            max_exit_staleness_minutes=1440,
            liquidity_scheme="dollar_ema",
            budget=1000.0,
            entry_price_source="nbbo-ask",
            exit_price_source="nbbo-bid",
            exit_minute=575,
        )

    def test_missing_exit_leaves_original_allocation_in_cash_and_recovers_next_date(
        self,
    ):
        args = self.inputs()
        args["morning_prices"][3, 1] = np.nan
        with self.assertLogs(
            "trading_rl.overnight.backtest", level="WARNING"
        ) as warning:
            trades, summary = run_backtest(**args)
        first = trades[trades.entry_date == "2026-08-26"]
        self.assertEqual(first.sample_id.tolist(), ["B"])
        self.assertEqual(first.entry_notional.tolist(), [500.0])
        self.assertEqual(first.quantity.tolist(), [5.0])
        self.assertEqual(first["rank"].tolist(), [2])
        self.assertAlmostEqual(first.portfolio_return.iloc[0], 0.0499)
        self.assertEqual(
            set(trades[trades.entry_date == "2026-08-27"].sample_id), {"A", "B"}
        )
        self.assertAlmostEqual(summary["ending_equity"], 1000 * 1.0499 * 1.0998)
        self.assertEqual(summary["skipped_missing_prices"], 1)
        self.assertEqual(summary["missing_price_sessions"], 1)
        self.assertEqual(summary["minimum_capital_utilization"], 0.5)
        self.assertIn("2026-08-27", warning.output[0])
        self.assertNotIn("C", first.sample_id.tolist())

    def test_nbbo_staleness_over_sixty_seconds_skips_despite_looser_mark_limit(self):
        args = self.inputs()
        args["morning_staleness"][3, 1] = 1.1
        with self.assertLogs("trading_rl.overnight.backtest", level="WARNING"):
            trades, summary = run_backtest(**args)
        self.assertEqual(summary["skipped_missing_prices"], 1)
        self.assertEqual(
            trades[trades.entry_date == "2026-08-26"].sample_id.tolist(), ["B"]
        )

    def test_all_missing_session_remains_in_daily_accounting(self):
        args = self.inputs()
        args["morning_prices"][3, 1:3] = np.nan
        with self.assertLogs("trading_rl.overnight.backtest", level="WARNING"):
            trades, summary = run_backtest(**args)
        self.assertEqual(summary["strategy_metrics"]["periods"], 2)
        self.assertEqual(summary["all_cash_sessions"], 1)
        self.assertEqual(summary["daily_portfolio"][0]["portfolio_return"], 0.0)
        self.assertEqual(summary["daily_portfolio"][0]["capital_deployed"], 0.0)
        self.assertAlmostEqual(summary["ending_equity"], 1099.8)
        self.assertEqual(summary["minimum_executed_basket_size"], 0)
        self.assertEqual(len(trades), 2)

    def test_entire_run_missing_including_benchmark_renders_warnings(self):
        args = self.inputs()
        args["morning_prices"][:] = np.nan
        with self.assertLogs("trading_rl.overnight.backtest", level="WARNING"):
            trades, summary = run_backtest(**args)
        self.assertTrue(trades.empty)
        self.assertEqual(summary["ending_equity"], 1000.0)
        self.assertEqual(summary["skipped_missing_prices"], 4)
        self.assertEqual(summary["missing_benchmark_sessions"], 2)
        self.assertTrue(np.isnan(summary["spy_overnight_metrics"]["total_return"]))
        console = Console(file=io.StringIO(), record=True, width=160)
        print_summary_table(summary, console)
        text = console.export_text()
        self.assertIn("WARNING: missing prices", text)
        self.assertIn("WARNING: benchmark gaps", text)

    def test_whole_share_skip_preserves_slot_budget(self):
        args = self.inputs()
        args["share_mode"] = "whole"
        args["entry_prices"][:, 2] = 120.0
        args["entry_prices"][2, 1] = np.nan
        with self.assertLogs("trading_rl.overnight.backtest", level="WARNING"):
            trades, _ = run_backtest(**args)
        first = trades[trades.entry_date == "2026-08-26"]
        self.assertEqual(first.quantity.tolist(), [4.0])
        self.assertEqual(first.entry_notional.tolist(), [480.0])

    def test_bid_loader_checks_target_clock_and_retains_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bids.npz"
            np.savez_compressed(
                path,
                split_adjusted=np.asarray(True),
                symbol=np.array(["A"]),
                date=np.array(["2026-08-27"], dtype="datetime64[D]"),
                target_timestamp=np.array(["2026-08-27T13:35:00Z"]),
                timestamp=np.array(["2026-08-27T13:34:59Z"]),
                bid_price=np.array([100.0]),
                raw_bid_price=np.array([200.0]),
                bid_exchange=np.array(["Q"]),
                ask_price=np.array([105.0]),
            )
            prices, age, _ = load_scheduled_nbbo_prices(
                path,
                pd.DatetimeIndex(["2026-08-27"]),
                np.array(["A", "B"]),
                "bid",
                575,
            )
            self.assertEqual(prices[0, 0], 100.0)
            self.assertAlmostEqual(age[0, 0], 1 / 60)
            self.assertTrue(np.isnan(prices[0, 1]))
            self.assertTrue(np.isinf(age[0, 1]))
            with self.assertRaisesRegex(ValueError, "target timestamps"):
                load_scheduled_nbbo_prices(
                    path,
                    pd.DatetimeIndex(["2026-08-27"]),
                    np.array(["A"]),
                    "bid",
                    945,
                )

    def test_exit_nbbo_cli(self):
        args = build_parser().parse_args(
            [
                "--exit-price-source",
                "nbbo-bid",
                "--exit-time",
                "09:35",
                "--exit-nbbo-path",
                "/tmp/bids.npz",
            ]
        )
        self.assertEqual(args.exit_price_source, "nbbo-bid")
        self.assertEqual(args.exit_nbbo_path, "/tmp/bids.npz")


class ScheduledNbboTimestampTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nbbo.npz"
        self.symbols = np.array(["A", "B", "C", "D"])
        self.dates = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2026-09-01", "2026-09-03"])
        self.arrays = dict(
            split_adjusted=np.asarray(True),
            symbol=self.symbols,
            date=self.dates.to_numpy(dtype="datetime64[D]"),
            target_timestamp=[
                "2022-01-03T20:45:00Z", "2022-01-04T20:45:00.000Z",
                "2026-09-01T19:45:00.000000+00:00", "2026-09-03T15:45:00.000000000-04:00",
            ],
            timestamp=[
                "2022-01-03T20:44:59.886Z", "2022-01-04T20:45:00Z",
                "2026-09-01T19:44:59.999999Z", "2026-09-03T15:44:59.999999999-04:00",
            ],
            ask_price=[10., 20., 30., 40.], raw_ask_price=[20., 40., 60., 80.],
            bid_price=[9., 19., 29., 39.], raw_bid_price=[18., 38., 58., 78.],
            ask_exchange=["Q"] * 4, bid_exchange=["Q"] * 4,
        )

    def load(self, side="ask"):
        np.savez_compressed(self.path, **self.arrays)
        return load_scheduled_nbbo_prices(self.path, self.dates, self.symbols, side, 945)

    def test_mixed_iso_precision_and_offsets_preserve_prices_and_nanosecond_age(self):
        for side in ("bid", "ask"):
            with self.subTest(side=side):
                prices, staleness, rows = self.load(side)
                np.testing.assert_array_equal(prices.diagonal(), self.arrays[f"{side}_price"])
                np.testing.assert_allclose(
                    staleness.diagonal(), np.asarray([0.114, 0., 0.000001, 0.000000001]) / 60.,
                    rtol=0., atol=1e-16,
                )
                np.testing.assert_array_equal(rows.raw_price, self.arrays[f"raw_{side}_price"])
                self.assertEqual(rows.iloc[-1].timestamp.value, pd.Timestamp("2026-09-03T19:44:59.999999999Z").value)

    def test_post_target_quote_is_rejected_even_one_nanosecond_late(self):
        self.arrays["timestamp"][-1] = "2026-09-03T19:45:00.000000001Z"
        with self.assertRaisesRegex(ValueError, "post-target NBBO quote"):
            self.load()

    def test_target_must_still_match_the_scheduled_minute_exactly(self):
        self.arrays["target_timestamp"][-1] = "2026-09-03T19:45:00.000000001Z"
        with self.assertRaisesRegex(ValueError, "target timestamps"):
            self.load()

    def test_missing_and_malformed_timestamps_fail_instead_of_producing_nan_age(self):
        for column in ("target_timestamp", "timestamp"):
            for value in ("", "NaT", "not-a-timestamp"):
                with self.subTest(column=column, value=value):
                    original = self.arrays[column][0]
                    self.arrays[column][0] = value
                    with self.assertRaises(ValueError):
                        self.load()
                    self.arrays[column][0] = original
