import unittest
from datetime import date
from pathlib import Path
import tempfile

import numpy as np
from scripts.tests.bar_fixtures import ohlcv_fixture
import pandas as pd
from rich.console import Console

from trading_rl.market_data.calendar import auction_close_minutes, short_entry_dates
from trading_rl.overnight.backtest import (
    DEFAULT_TRANSACTION_COST_BPS,
    _symbol_daily_arrays,
    basket_quantities,
    causal_ema_log_liquidity,
    causal_turnover_stability,
    causal_completed_trading_days,
    company_universe_mask,
    exchange_universe_mask,
    is_company_security,
    liquidity_scores,
    load_opening_auction_prices,
    load_primary_auction_exchange_mask,
    load_scheduled_nbbo_asks,
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
        timestamp=np.asarray([row.get("timestamp", "") for row in rows]),
    )


class OvernightLiquidityBaselineTest(unittest.TestCase):
    def test_default_transaction_cost_is_one_basis_point_per_side(self):
        self.assertEqual(DEFAULT_TRANSACTION_COST_BPS, 1.0)

    def test_scheduled_nbbo_loader_returns_adjusted_asks_and_quote_age(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nbbo.npz"
            np.savez_compressed(
                path,
                split_adjusted=np.asarray(True),
                symbol=np.asarray(["AAPL"]),
                date=np.asarray(["2026-09-02"], dtype="datetime64[D]"),
                target_timestamp=np.asarray(["2026-09-02T19:45:00Z"]),
                timestamp=np.asarray(["2026-09-02T19:44:57Z"]),
                ask_price=np.asarray([100.0]),
                raw_ask_price=np.asarray([200.0]),
                ask_exchange=np.asarray(["Q"]),
            )

            prices, staleness, rows = load_scheduled_nbbo_asks(
                path,
                pd.DatetimeIndex(["2026-09-02"]),
                np.asarray(["AAPL", "MSFT"]),
            )

        self.assertEqual(prices[0, 0], 100.0)
        self.assertTrue(np.isnan(prices[0, 1]))
        self.assertAlmostEqual(staleness[0, 0], 0.05)
        self.assertTrue(np.isinf(staleness[0, 1]))
        self.assertEqual(float(rows.iloc[0].raw_price), 200.0)

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

    def test_turnover_stability_demotes_a_briefly_enormous_name(self):
        """STEADY and SPIKY average the same turnover; only SPIKY is erratic."""
        sessions = 40
        steady = np.full(sessions, 1_000_000.0)
        # SPIKY trades more on every measure, but a quarter of that arrives in
        # bursts. It has to out-rank STEADY on the raw level for the test to say
        # anything -- log1p is concave, so the EMA already discounts burstiness
        # on its own and an equal-mean burst name loses without any penalty.
        spiky = np.full(sessions, 1_500_000.0)
        spiky[::4] = 20_000_000.0
        volume = np.column_stack((steady, spiky))

        level = causal_ema_log_liquidity(volume, 10, 5)
        stability = causal_turnover_stability(volume, 10, 5)

        # The burst name leads on the raw level; the dispersion penalty reverses it.
        self.assertGreater(level[-1, 1], level[-1, 0])
        self.assertGreater(stability[-1, 0], stability[-1, 1])

    def test_turnover_stability_reads_only_prior_sessions(self):
        volume = np.full((30, 1), 1_000_000.0)
        baseline = causal_turnover_stability(volume, 10, 5)

        # Rewriting the final session must not move any score at or before it.
        volume[-1, 0] = 9_000_000_000.0
        revised = causal_turnover_stability(volume, 10, 5)

        np.testing.assert_allclose(baseline, revised, equal_nan=True)

    def test_turnover_stability_rejects_a_degenerate_span(self):
        with self.assertRaises(ValueError):
            causal_turnover_stability(np.full((10, 1), 1.0), 1, 5)

    def test_turnover_stability_uses_ema_span_for_dispersion(self):
        volume = np.array(
            [[100.0], [200.0], [400.0], [800.0], [1_600.0], [3_200.0]]
        )
        span = 4
        level = causal_ema_log_liquidity(volume, span, min_history_days=1)
        expected_spread = (
            pd.DataFrame(np.log1p(volume))
            .rolling(span, min_periods=span // 2)
            .std()
            .shift(1)
            .to_numpy()
        )

        actual = causal_turnover_stability(volume, span, min_history_days=1)

        np.testing.assert_allclose(
            actual,
            level - np.nan_to_num(expected_spread, nan=0.0),
            equal_nan=True,
        )

    def test_turnover_stability_can_replay_the_legacy_dispersion_span(self):
        sessions = np.arange(30, dtype=np.float64)[:, None]
        volume = 1_000.0 * np.exp(np.sin(sessions / 3.0))

        legacy = causal_turnover_stability(
            volume,
            ema_span=10,
            min_history_days=5,
            dispersion_span=20,
        )
        current = causal_turnover_stability(
            volume,
            ema_span=10,
            min_history_days=5,
        )

        self.assertFalse(np.allclose(legacy, current, equal_nan=True))

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
        entry_prices = np.array([[100.0, 100.0, 300.0]] * 4)
        morning_prices = entry_prices.copy()
        morning_prices[3] = [100.0, 110.0, 270.0]
        staleness = np.zeros_like(entry_prices)

        common = dict(
            dates=dates,
            symbols=symbols,
            dollar_volume=dollar_volume,
            entry_prices=entry_prices,
            morning_prices=morning_prices,
            entry_staleness=staleness,
            morning_staleness=staleness,
            start_date=dates[2],
            end_date=dates[3],
            top=2,
            ema_span=1,
            min_history_days=1,
            minimum_trading_days=1,
            transaction_cost_bps=0.0,
            max_entry_staleness_minutes=0,
            max_exit_staleness_minutes=0,
            liquidity_scheme="dollar_ema",
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

    def test_short_session_is_skipped_for_entry_but_remains_the_prior_exit(self):
        dates = pd.DatetimeIndex(
            ["2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01"]
        )
        symbols = np.array(["SPY", "A"])
        dollar_volume = np.array(
            [[1_000.0, 2_000.0 + index] for index in range(len(dates))]
        )
        entry_prices = np.full((len(dates), 2), 100.0)
        morning_prices = np.full((len(dates), 2), 101.0)
        staleness = np.zeros_like(entry_prices)

        trades, summary = run_backtest(
            dates=dates,
            symbols=symbols,
            dollar_volume=dollar_volume,
            entry_prices=entry_prices,
            morning_prices=morning_prices,
            entry_staleness=staleness,
            morning_staleness=staleness,
            start_date=dates[1],
            end_date=dates[4],
            top=1,
            ema_span=1,
            min_history_days=1,
            minimum_trading_days=1,
            transaction_cost_bps=0.0,
            max_entry_staleness_minutes=1,
            max_exit_staleness_minutes=1,
            liquidity_scheme="dollar_ema",
            entry_session_mask=np.array([True, True, False, True, True]),
        )

        self.assertEqual(trades.entry_date.tolist(), ["2026-11-25", "2026-11-30"])
        self.assertEqual(trades.exit_date.iloc[0], "2026-11-27")
        self.assertEqual(summary["skipped_short_entry_sessions"], 1)

    def test_auction_close_times_identify_short_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auctions.npz"
            write_auction_npz(
                path,
                [
                    {
                        "symbol": "SPY",
                        "date": "2026-11-25",
                        "session": "close",
                        "condition": "6",
                        "price": 600.0,
                        "size": 1_000,
                        "timestamp": "2026-11-25T21:00:00Z",
                        "exchange": "P",
                    },
                    {
                        "symbol": "SPY",
                        "date": "2026-11-27",
                        "session": "close",
                        "condition": "6",
                        "price": 601.0,
                        "size": 1_000,
                        "timestamp": "2026-11-27T18:00:00Z",
                        "exchange": "P",
                    },
                ],
            )
            closes = auction_close_minutes(
                path, [date(2026, 11, 25), date(2026, 11, 27)]
            )

        self.assertEqual(closes[date(2026, 11, 25)], 16 * 60)
        self.assertEqual(closes[date(2026, 11, 27)], 13 * 60)
        self.assertEqual(
            short_entry_dates(closes, 15 * 60 + 45), {date(2026, 11, 27)}
        )

    def test_nbbo_backtest_does_not_replace_a_selected_symbol_with_a_lower_rank(self):
        dates = pd.date_range("2026-08-24", periods=4, freq="B")
        symbols = np.array(["SPY", "HIGH", "LOW"])
        dollar_volume = np.array(
            [
                [1_000.0, 3_000.0, 2_000.0],
                [1_000.0, 3_100.0, 2_100.0],
                [1_000.0, 3_200.0, 2_200.0],
                [1_000.0, 3_300.0, 2_300.0],
            ]
        )
        entry_prices = np.array([[100.0, 100.0, 100.0]] * 4)
        entry_prices[2, 1] = np.nan
        morning_prices = np.array([[100.0, 101.0, 101.0]] * 4)
        staleness = np.zeros_like(entry_prices)

        with self.assertRaisesRegex(
            ValueError,
            r"fresh scheduled NBBO entry on 2026-08-26: HIGH",
        ):
            run_backtest(
                dates=dates,
                symbols=symbols,
                dollar_volume=dollar_volume,
                entry_prices=entry_prices,
                morning_prices=morning_prices,
                entry_staleness=staleness,
                morning_staleness=staleness,
                start_date=dates[2],
                end_date=dates[3],
                top=1,
                ema_span=1,
                min_history_days=1,
                minimum_trading_days=1,
                transaction_cost_bps=0.0,
                max_entry_staleness_minutes=1,
                max_exit_staleness_minutes=1,
                liquidity_scheme="dollar_ema",
                entry_price_source="nbbo-ask",
            )

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
            "basket_size": 50,
            "liquidity_scheme": "dollar_ema",
            "liquidity_metric": "completed regular-session dollar volume",
            "entry_time_eastern": "15:55",
            "exit_time_eastern": "09:45 next trading session",
            "first_entry_date": "2026-08-19",
            "last_exit_date": "2026-08-21",
            "trades": 100,
            "ema_span_sessions": 20,
            "minimum_liquidity_history_sessions": 20,
            "minimum_completed_trading_days": 100,
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

    def test_liquidity_scores_support_only_live_ranking_schemes(self):
        dollar = np.array([[100.0, 1_000.0], [110.0, 900.0], [120.0, 800.0]])
        ema = liquidity_scores(dollar, "dollar_ema", ema_span=2, min_history_days=1)
        stable = liquidity_scores(
            dollar, "turnover_stability", ema_span=2, min_history_days=1
        )

        self.assertEqual(ema.shape, dollar.shape)
        self.assertEqual(stable.shape, dollar.shape)
        with self.assertRaisesRegex(ValueError, "unknown liquidity scheme"):
            liquidity_scores(dollar, "alpaca_volume", ema_span=2, min_history_days=1)

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
                ohlcv_fixture(np.array(
                    [[daily_second, 90_000, 110_000, 80_000, 95_000, 50, 10, 0]],
                    dtype=np.int64,
                )),
            )
            result = _symbol_daily_arrays(
                "X",
                {pd.Timestamp("2026-01-05"): 0},
                np.array([0]),
                minute_dir,
                daily_dir,
                15 * 60 + 55,
                9 * 60 + 45,
                1,
            )

        self.assertEqual(result[1][0], 4_750.0)

    def test_symbol_execution_prices_use_open_from_full_ohlcv_columns(self):
        session = pd.Timestamp("2026-01-05 04:00", tz="America/New_York")
        origin = pd.Timestamp("2010-01-01", tz="UTC")
        context_start = int((session.tz_convert("UTC") - origin).total_seconds())
        morning = context_start + (9 * 60 + 30 - 4 * 60) * 60
        entry = context_start + (15 * 60 + 45 - 4 * 60) * 60
        with tempfile.TemporaryDirectory() as directory:
            minute_dir = Path(directory) / "minute"
            daily_dir = Path(directory) / "daily"
            minute_dir.mkdir()
            daily_dir.mkdir()
            np.save(
                minute_dir / "X.npy",
                ohlcv_fixture(np.array([[morning, 99_000], [entry, 100_000]], dtype=np.int64)),
            )

            result = _symbol_daily_arrays(
                "X",
                {pd.Timestamp("2026-01-05"): 0},
                np.array([context_start]),
                minute_dir,
                daily_dir,
                15 * 60 + 45,
                9 * 60 + 30,
                1,
            )

        self.assertEqual(result[2][0], 100.0)
        self.assertEqual(result[3][0], 99.0)
        self.assertEqual(result[4][0], 0.0)
        self.assertEqual(result[5][0], 0.0)

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
            np.save(path, ohlcv_fixture(bars))
            dates, context_sod = reference_session_calendar(
                path, {date(2026, 1, 6): 13 * 60}
            )

        self.assertEqual(
            dates.strftime("%Y-%m-%d").tolist(),
            ["2026-01-05", "2026-01-06", "2026-01-07"],
        )
        context_times = pd.to_datetime(
            context_sod, unit="s", origin="2010-01-01", utc=True
        ).tz_convert("America/New_York")
        self.assertEqual(
            context_times.strftime("%H:%M").tolist(), ["04:00", "04:00", "04:00"]
        )

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
        trades = pd.DataFrame(
            {
                "entry_date": ["2026-08-20", "2026-08-21", "2026-08-21"],
                "sample_id": ["AAPL", "AAPL", "MSFT"],
                "net_return": [0.01, 0.02, -0.01],
            }
        )
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

if __name__ == "__main__":
    unittest.main()
