import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from rich.console import Console

from overnight.backtest import (
    DEFAULT_TRANSACTION_COST_BPS,
    _symbol_daily_arrays,
    activity_union_candidate_mask,
    basket_quantities,
    causal_ema_log_liquidity,
    causal_completed_trading_days,
    company_universe_mask,
    exchange_universe_mask,
    is_company_security,
    liquidity_scores,
    load_opening_auction_prices,
    load_primary_auction_exchange_mask,
    print_scheme_comparison,
    print_symbol_trade_counts,
    print_summary_table,
    reference_session_calendar,
    run_backtest,
    strategy_metrics,
    top_liquid_indices,
)


def write_auction_npz(path: Path, rows: list[dict[str, object]]) -> None:
    np.savez_compressed(
        path,
        split_adjusted=np.asarray(True),
        symbol=np.asarray([row["symbol"] for row in rows]),
        date=np.asarray([row["date"] for row in rows], dtype="datetime64[D]"),
        session=np.asarray(
            [0 if row["session"] == "open" else 1 for row in rows], dtype=np.uint8
        ),
        condition=np.asarray([row["condition"] for row in rows]),
        price=np.asarray([row["price"] for row in rows], dtype=np.float64),
        size=np.asarray([row["size"] for row in rows], dtype=np.float64),
        exchange=np.asarray([row["exchange"] for row in rows]),
    )


class OvernightLiquidityBaselineTest(unittest.TestCase):
    def test_default_transaction_cost_is_one_basis_point_per_side(self):
        self.assertEqual(DEFAULT_TRANSACTION_COST_BPS, 1.0)

    def test_opening_auction_loader_uses_only_official_condition_o(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auctions.npz"
            write_auction_npz(
                path,
                [
                    {
                        "symbol": "AAPL",
                        "date": "2026-08-25",
                        "session": "open",
                        "condition": "O",
                        "price": 100.0,
                        "size": 10,
                        "exchange": "V",
                    },
                    {
                        "symbol": "AAPL",
                        "date": "2026-08-25",
                        "session": "open",
                        "condition": "O",
                        "price": 50.5,
                        "size": 2_000,
                        "exchange": "Q",
                    },
                    {
                        "symbol": "AAPL",
                        "date": "2026-08-25",
                        "session": "close",
                        "condition": "6",
                        "price": 102.0,
                        "size": 2_000,
                        "exchange": "Q",
                    },
                ],
            )

            prices = load_opening_auction_prices(
                path,
                pd.DatetimeIndex(["2026-08-25"]),
                np.array(["AAPL", "MSFT"]),
            )

        self.assertEqual(prices[0, 0], 50.5)
        self.assertTrue(np.isnan(prices[0, 1]))

    def test_historical_exchange_mask_overrides_known_listing_transfer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auctions.npz"
            write_auction_npz(
                path,
                [
                    {
                        "symbol": "PLTR",
                        "date": "2024-11-21",
                        "session": "open",
                        "condition": "O",
                        "price": 62.61,
                        "size": 735_725,
                        "exchange": "N",
                    },
                    {
                        "symbol": "PLTR",
                        "date": "2024-11-27",
                        "session": "open",
                        "condition": "O",
                        "price": 66.00,
                        "size": 800_000,
                        "exchange": "Q",
                    },
                ],
            )

            mask, known = load_primary_auction_exchange_mask(
                path,
                pd.DatetimeIndex(["2024-11-21", "2024-11-27"]),
                np.array(["PLTR", "UNKNOWN"]),
                "nasdaq",
            )

        self.assertEqual(known, 2)
        self.assertEqual(mask.tolist(), [[False, True], [True, True]])

    def test_nasdaq_mask_accepts_the_t_coded_copy_of_one_opening_cross(self):
        """Alpaca publishes the Nasdaq cross as both Q and T; some sessions carry
        only T (e.g. 2023-01-30), and matching Q alone emptied the universe."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auctions.npz"
            write_auction_npz(
                path,
                [
                    # 2023-01-27: both tape copies of the same Nasdaq cross.
                    {
                        "symbol": "AAPL",
                        "date": "2023-01-27",
                        "session": "open",
                        "condition": "O",
                        "price": 143.10,
                        "size": 406_648,
                        "exchange": "Q",
                    },
                    {
                        "symbol": "AAPL",
                        "date": "2023-01-27",
                        "session": "open",
                        "condition": "O",
                        "price": 143.10,
                        "size": 406_648,
                        "exchange": "T",
                    },
                    # 2023-01-30: only the T copy survived.
                    {
                        "symbol": "AAPL",
                        "date": "2023-01-30",
                        "session": "open",
                        "condition": "O",
                        "price": 144.90,
                        "size": 593_788,
                        "exchange": "T",
                    },
                    # A NYSE listing still loses to its own primary print, so a
                    # smaller Nasdaq-book cross cannot make it look Nasdaq-listed.
                    {
                        "symbol": "UNH",
                        "date": "2023-01-30",
                        "session": "open",
                        "condition": "O",
                        "price": 494.00,
                        "size": 180_000,
                        "exchange": "N",
                    },
                    {
                        "symbol": "UNH",
                        "date": "2023-01-30",
                        "session": "open",
                        "condition": "O",
                        "price": 494.00,
                        "size": 1_200,
                        "exchange": "T",
                    },
                ],
            )

            mask, known = load_primary_auction_exchange_mask(
                path,
                pd.DatetimeIndex(["2023-01-27", "2023-01-30"]),
                np.array(["AAPL", "UNH"]),
                "nasdaq",
            )

        self.assertEqual(known, 3)
        # AAPL stays eligible on both sessions; UNH is excluded only where its
        # NYSE print is known, and stays eligible where no auction was recorded.
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

    def test_whole_share_sizing_rounds_down_without_exceeding_budget(self):
        prices = np.array([120.0, 300.0, 700.0])

        fractional = basket_quantities(prices, 1_200.0, "fractional")
        whole = basket_quantities(prices, 1_200.0, "whole")

        np.testing.assert_allclose(fractional, [10 / 3, 4 / 3, 4 / 7])
        np.testing.assert_array_equal(whole, [3.0, 1.0, 0.0])
        self.assertLessEqual(float(np.dot(whole, prices)), 1_200.0)
        self.assertTrue(np.equal(whole, np.floor(whole)).all())

    def test_whole_share_backtest_weights_returns_by_integer_notional_and_idle_cash(self):
        dates = pd.date_range("2026-08-24", periods=4, freq="B")
        symbols = np.array(["SPY", "A", "B"])
        dollar_volume = np.array(
            [
                [1_000.0, 3_000.0, 2_000.0],
                [1_000.0, 3_100.0, 2_100.0],
                [1_000.0, 3_200.0, 2_200.0],
                [1_000.0, 3_300.0, 2_300.0],
            ]
        )
        activity = np.ones_like(dollar_volume)
        entry_prices = np.array([[100.0, 100.0, 300.0]] * 4)
        morning_prices = entry_prices.copy()
        morning_prices[3] = [100.0, 110.0, 270.0]
        staleness = np.zeros_like(entry_prices)

        common = dict(
            dates=dates,
            symbols=symbols,
            dollar_volume=dollar_volume,
            alpaca_share_volume=activity,
            alpaca_trade_count=activity,
            entry_prices=entry_prices,
            morning_prices=morning_prices,
            entry_staleness=staleness,
            morning_staleness=staleness,
            start_date=dates[2],
            end_date=dates[3],
            top=2,
            exclude_top=0,
            ema_span=1,
            min_history_days=1,
            minimum_trading_days=1,
            transaction_cost_bps=0.0,
            max_entry_staleness_minutes=0,
            max_exit_staleness_minutes=0,
            budget=1_000.0,
        )

        fractional_trades, fractional_summary = run_backtest(
            **common, share_mode="fractional"
        )
        whole_trades, whole_summary = run_backtest(**common, share_mode="whole")

        self.assertAlmostEqual(fractional_summary["strategy_metrics"]["mean_return"], 0.0)
        self.assertAlmostEqual(whole_summary["strategy_metrics"]["mean_return"], 0.02)
        np.testing.assert_array_equal(whole_trades.quantity, [5.0, 1.0])
        self.assertAlmostEqual(float(whole_trades.entry_notional.sum()), 800.0)
        self.assertAlmostEqual(whole_summary["average_capital_utilization"], 0.8)
        self.assertAlmostEqual(fractional_summary["average_capital_utilization"], 1.0)

    def test_completed_trading_day_count_is_strictly_lagged(self):
        volume = np.array(
            [
                [100.0, np.nan],
                [110.0, 200.0],
                [np.nan, 210.0],
                [120.0, 220.0],
            ]
        )

        counts = causal_completed_trading_days(volume)

        self.assertEqual(counts.tolist(), [[0, 0], [1, 0], [2, 1], [2, 2]])

    def test_liquidity_score_is_strictly_lagged(self):
        volume = np.array([[100.0], [110.0], [120.0], [130.0]])
        changed = volume.copy()
        changed[2:] *= 1_000_000.0
        score = causal_ema_log_liquidity(volume, span=3, min_history_days=1)
        changed_score = causal_ema_log_liquidity(changed, span=3, min_history_days=1)

        self.assertTrue(np.isnan(score[0, 0]))
        self.assertAlmostEqual(score[2, 0], changed_score[2, 0])
        self.assertNotAlmostEqual(score[3, 0], changed_score[3, 0])

    def test_log_ema_prevents_one_spike_from_immediately_reordering_stable_liquidity(self):
        volume = np.array(
            [
                [100.0, 10.0],
                [100.0, 10.0],
                [100.0, 1_000.0],
                [100.0, 10.0],
            ]
        )
        smoothed = causal_ema_log_liquidity(volume, span=20, min_history_days=1)
        unsmoothed = causal_ema_log_liquidity(volume, span=1, min_history_days=1)

        self.assertGreater(smoothed[3, 0], smoothed[3, 1])
        self.assertLess(unsmoothed[3, 0], unsmoothed[3, 1])

    def test_top_selection_excludes_symbols_without_a_current_entry_price(self):
        selected = top_liquid_indices(
            scores=np.array([5.0, 4.0, 3.0]),
            entry_prices=np.array([np.nan, 100.0, 100.0]),
            top=2,
            symbols=np.array(["A", "B", "C"]),
        )
        self.assertEqual(selected.tolist(), [1, 2])

    def test_metrics_compound_period_returns(self):
        metrics = strategy_metrics(np.array([0.10, -0.05]))
        self.assertAlmostEqual(metrics["total_return"], 0.045)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.05)

    def test_cli_summary_is_rendered_as_a_comparison_table(self):
        metrics = strategy_metrics(np.array([0.01, -0.005]))
        summary = {
            "top": 50,
            "exclude_top": 0,
            "basket_size": 50,
            "liquidity_scheme": "dollar_ema",
            "liquidity_metric": "completed regular-session dollar volume",
            "ranking_time_eastern": "15:15",
            "entry_time_eastern": "15:55",
            "exit_time_eastern": "09:45 next trading session",
            "first_entry_date": "2026-08-19",
            "last_exit_date": "2026-08-21",
            "trades": 100,
            "ema_span_sessions": 20,
            "minimum_liquidity_history_sessions": 20,
            "transaction_cost_bps_per_side": 1.0,
            "unique_symbols_traded": 51,
            "average_daily_membership_replacements": 1.0,
            "maximum_daily_membership_replacements": 2,
            "average_daily_membership_retention": 0.98,
            "average_daily_membership_jaccard": 0.96,
            "stale_exit_marks_over_10_minutes": 0,
            "maximum_exit_staleness_minutes": 1.0,
            "strategy_metrics": metrics,
            "spy_overnight_metrics": metrics,
            "spy_buy_and_hold_metrics": metrics,
        }
        console = Console(record=True, width=100, color_system=None)
        print_summary_table(summary, console)
        rendered = console.export_text()

        self.assertIn("Overnight liquidity baseline", rendered)
        self.assertIn("Top 50", rendered)
        self.assertIn("SPY overnight", rendered)
        self.assertIn("SPY buy & hold", rendered)
        self.assertNotIn('"strategy_metrics"', rendered)

    def test_alpaca_activity_schemes_use_same_day_raw_rankings(self):
        dollar = np.array([[1_000.0, 10.0], [1_000.0, 10.0]])
        shares = np.array([[10.0, 100.0], [20.0, 200.0]])
        trades = np.array([[50.0, 5.0], [60.0, 6.0]])

        volume_scores = liquidity_scores(dollar, shares, trades, "alpaca_volume", 20, 1)
        trade_scores = liquidity_scores(dollar, shares, trades, "alpaca_trades", 20, 1)

        self.assertEqual(volume_scores.tolist(), shares.tolist())
        self.assertEqual(trade_scores.tolist(), trades.tolist())
        self.assertGreater(volume_scores[0, 1], volume_scores[0, 0])
        self.assertGreater(trade_scores[0, 0], trade_scores[0, 1])

    def test_activity_union_is_rebuilt_each_day_without_carry_forward(self):
        shares = np.array(
            [
                [100.0, 90.0, 1.0, 1.0],
                [1.0, 1.0, 90.0, 100.0],
            ]
        )
        trades = np.array(
            [
                [1.0, 1.0, 100.0, 90.0],
                [1.0, 100.0, 1.0, 90.0],
            ]
        )
        mask = activity_union_candidate_mask(
            shares, trades, np.array(["A", "B", "C", "D"]), candidates_per_metric=1
        )

        self.assertEqual(mask[0].tolist(), [True, False, True, False])
        self.assertEqual(mask[1].tolist(), [False, True, False, True])

    def test_activity_union_ema_uses_lagged_dollar_history(self):
        dollar = np.array([[100.0, 1_000.0], [110.0, 900.0], [120.0, 800.0]])
        activity = np.ones_like(dollar)

        union_scores = liquidity_scores(
            dollar, activity, activity, "activity_union_ema", ema_span=2, min_history_days=1
        )
        full_scores = liquidity_scores(
            dollar, activity, activity, "dollar_ema", ema_span=2, min_history_days=1
        )

        np.testing.assert_allclose(union_scores, full_scores, equal_nan=True)
        self.assertTrue(np.isnan(union_scores[0]).all())

    def test_activity_cutoff_excludes_the_ranking_minute_bar(self):
        context_start = int(
            (
                pd.Timestamp("2026-08-24 04:00", tz="America/New_York").tz_convert("UTC")
                - pd.Timestamp("2010-01-01", tz="UTC")
            ).total_seconds()
        )
        regular_start = context_start + (9 * 60 + 30 - 4 * 60) * 60
        ranking_second = context_start + (15 * 60 + 15 - 4 * 60) * 60
        seconds = np.arange(regular_start, regular_start + 391 * 60, 60, dtype=np.int64)
        source = np.column_stack(
            (
                seconds,
                np.full(391, 100_000, dtype=np.int64),
                np.ones(391, dtype=np.int64),
                np.ones(391, dtype=np.int64),
            )
        )
        ranking_row = int(np.flatnonzero(seconds == ranking_second)[0])
        source[ranking_row, 2:] = (1_000, 100)
        source[ranking_row + 1, 2:] = (2_000, 200)
        with tempfile.TemporaryDirectory() as directory:
            minute_dir = Path(directory) / "minute"
            daily_dir = Path(directory) / "daily"
            minute_dir.mkdir()
            daily_dir.mkdir()
            np.save(minute_dir / "X.npy", source)
            daily_second = int(
                (
                    pd.Timestamp("2026-08-24", tz="America/New_York").tz_convert("UTC")
                    - pd.Timestamp("2010-01-01", tz="UTC")
                ).total_seconds()
            )
            np.save(
                daily_dir / "X.npy",
                np.array(
                    [[daily_second, 90_000, 110_000, 80_000, 95_000, 50, 10, 100_000]],
                    dtype=np.int64,
                ),
            )
            result = _symbol_daily_arrays(
                "X",
                {pd.Timestamp("2026-08-24"): 0},
                np.array([context_start]),
                minute_dir,
                daily_dir,
                15 * 60 + 55,
                9 * 60 + 45,
                15 * 60 + 15,
                1,
            )

        self.assertEqual(result[2][0], 345.0)
        self.assertEqual(result[3][0], 345.0)
        self.assertEqual(result[1][0], 5_000.0)

    def test_daily_dollar_volume_falls_back_to_close_when_vwap_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            minute_dir = Path(directory) / "minute"
            daily_dir = Path(directory) / "daily"
            minute_dir.mkdir()
            daily_dir.mkdir()
            daily_second = int(
                (
                    pd.Timestamp("2026-01-05", tz="America/New_York").tz_convert("UTC")
                    - pd.Timestamp("2010-01-01", tz="UTC")
                ).total_seconds()
            )
            np.save(
                daily_dir / "X.npy",
                np.array(
                    [[daily_second, 90_000, 110_000, 80_000, 95_000, 50, 10, 0]],
                    dtype=np.int64,
                ),
            )
            result = _symbol_daily_arrays(
                "X",
                {pd.Timestamp("2026-01-05"): 0},
                np.array([0]),
                minute_dir,
                daily_dir,
                15 * 60 + 55,
                9 * 60 + 45,
                15 * 60 + 15,
                1,
            )

        self.assertEqual(result[1][0], 4_750.0)

    def test_reference_calendar_is_derived_from_complete_minute_sessions(self):
        def session_seconds(day: str, close: str) -> np.ndarray:
            start = pd.Timestamp(f"{day} 09:30", tz="America/New_York").tz_convert("UTC")
            end = pd.Timestamp(f"{day} {close}", tz="America/New_York").tz_convert("UTC")
            timestamps = pd.date_range(start, end, freq="1min")
            return (
                (timestamps - pd.Timestamp("2010-01-01", tz="UTC"))
                .total_seconds()
                .to_numpy(dtype=np.int64)
            )

        seconds = np.concatenate(
            (
                session_seconds("2026-01-05", "16:00"),
                session_seconds("2026-01-06", "13:00"),
                session_seconds("2026-01-07", "16:00"),
            )
        )
        bars = np.column_stack((seconds, np.full(len(seconds), 100_000, dtype=np.int64)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SPY.npy"
            np.save(path, bars)
            dates, context_sod = reference_session_calendar(path)

        self.assertEqual(dates.strftime("%Y-%m-%d").tolist(), ["2026-01-05", "2026-01-07"])
        context_times = pd.to_datetime(
            context_sod, unit="s", origin="2010-01-01", utc=True
        ).tz_convert("America/New_York")
        self.assertEqual(context_times.strftime("%H:%M").tolist(), ["04:00", "04:00"])

    def test_scheme_comparison_reports_stability_kpis(self):
        metrics = strategy_metrics(np.array([0.01, -0.005]))
        summaries = {}
        for index, scheme in enumerate(("dollar_ema", "alpaca_volume", "alpaca_trades")):
            summaries[scheme] = {
                "first_entry_date": "2026-08-19",
                "last_exit_date": "2026-08-21",
                "basket_size": 10,
                "liquidity_scheme": scheme,
                "liquidity_metric": {
                    "dollar_ema": "completed regular-session dollar volume",
                    "alpaca_volume": "share volume",
                    "alpaca_trades": "trade count",
                }[scheme],
                "ranking_time_eastern": "15:15",
                "ema_span_sessions": 20,
                "minimum_liquidity_history_sessions": 20,
                "strategy_metrics": metrics,
                "average_daily_membership_replacements": float(index),
                "average_daily_membership_retention": 1.0 - index * 0.1,
                "average_daily_membership_jaccard": 1.0 - index * 0.2,
                "unique_symbols_traded": 10 + index,
            }
        console = Console(record=True, width=120, color_system=None)

        print_scheme_comparison(summaries, console)
        rendered = console.export_text()

        self.assertIn("Lagged $ EMA", rendered)
        self.assertIn("Alpaca volume", rendered)
        self.assertIn("Alpaca trades", rendered)
        self.assertIn("Membership retention", rendered)

    def test_company_filter_rejects_funds_and_non_common_instruments(self):
        cases = (
            ({"name": "Example Technology Inc. - Common Stock", "etf": "N"}, True),
            ({"name": "Foreign Company plc - American Depositary Shares", "etf": "N"}, True),
            ({"name": "Example Realty Trust Common Stock", "etf": "N"}, True),
            ({"name": "Example Equity ETF", "etf": "Y"}, False),
            ({"name": "Example Income Fund - Closed End Fund", "etf": "N"}, False),
            ({"name": "Example Partners LP Common Units", "etf": "N"}, False),
            ({"name": "Example Acquisition Corp. Common Stock", "etf": "N"}, False),
            ({"name": "Example Royalty Trust Common Stock", "etf": "N"}, False),
            ({"name": "BlackRock Income Trust", "etf": "N"}, False),
        )
        for record, expected in cases:
            with self.subTest(name=record["name"]):
                self.assertEqual(is_company_security(record)[0], expected)

    def test_symbol_trade_frequency_shows_counts_and_session_percentages(self):
        trades = {
            "dollar_ema": pd.DataFrame(
                {
                    "entry_date": ["2026-08-20", "2026-08-21", "2026-08-21"],
                    "sample_id": ["AAPL", "AAPL", "MSFT"],
                    "net_return": [0.01, 0.02, -0.01],
                }
            ),
            "alpaca_volume": pd.DataFrame(
                {
                    "entry_date": ["2026-08-20", "2026-08-21"],
                    "sample_id": ["MSFT", "MSFT"],
                    "net_return": [0.03, 0.01],
                }
            ),
        }
        console = Console(record=True, width=120, color_system=None)

        print_symbol_trade_counts(trades, console)
        rendered = console.export_text()

        self.assertIn("Per-symbol trade frequency", rendered)
        self.assertIn("AAPL", rendered)
        self.assertIn("MSFT", rendered)
        self.assertIn("2 / 100.0%", rendered)
        self.assertIn("+1.500%", rendered)
        self.assertIn("-1.000%", rendered)

    def test_company_mask_keeps_spy_only_as_reference_and_excludes_unknowns(self):
        symbols = np.array(["SPY", "AAPL", "ETHA", "UNKNOWN"])
        security_master = {
            "SPY": {"name": "SPDR S&P 500 ETF Trust", "etf": "Y"},
            "AAPL": {"name": "Apple Inc. - Common Stock", "etf": "N"},
            "ETHA": {"name": "iShares Ethereum Trust ETF", "etf": "Y"},
        }

        mask, reasons, unclassified = company_universe_mask(
            symbols, security_master, keep_unclassified=False
        )

        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertEqual(reasons["ETF/ETP"], 1)
        self.assertEqual(reasons["missing from current security master"], 1)
        self.assertEqual(unclassified, 1)

    def test_company_mask_keeps_historical_unclassified_symbols_by_default(self):
        mask, reasons, unclassified = company_universe_mask(
            np.array(["DELISTED"]), security_master={}
        )

        self.assertEqual(mask.tolist(), [True])
        self.assertEqual(reasons, {})
        self.assertEqual(unclassified, 1)

    def test_exchange_mask_reranks_nasdaq_candidates_and_keeps_spy_benchmark(self):
        symbols = np.array(["SPY", "AAPL", "JPM", "UNKNOWN"])
        security_master = {
            "AAPL": {"exchange": "Q"},
            "JPM": {"exchange": "N"},
        }

        mask = exchange_universe_mask(symbols, security_master, "nasdaq")

        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_rank_segment_can_exclude_the_most_liquid_names(self):
        selected = top_liquid_indices(
            scores=np.array([5.0, 4.0, 3.0, 2.0]),
            entry_prices=np.full(4, 100.0),
            top=4,
            symbols=np.array(["A", "B", "C", "D"]),
            exclude_top=2,
        )
        self.assertEqual(selected.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
