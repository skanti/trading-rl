"""Causality, accounting and report-window invariance of promoted strategies."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest import build_parser, run_backtest
from trading_rl.overnight.strategies import (
    LiquidityTrendVolConfig,
    LiquidityTrendVolPolicy,
    load_trend_vol_config,
)


def fixture():
    dates = pd.bdate_range("2022-07-01", periods=170)
    prices = np.full((len(dates), 3), 100.0)
    morning = prices.copy()
    overnight = 0.012 * np.sin(np.arange(len(dates)) * 1.7)
    morning[1:, 1] *= 1 + overnight[:-1]
    morning[1:, 2] *= 1 - overnight[:-1] / 3
    return {
        "dates": dates,
        "symbols": np.array(["SPY", "A", "B"]),
        "dollar_volume": np.tile([100.0, 300.0, 200.0], (len(dates), 1)),
        "entry_prices": prices,
        "morning_prices": morning,
        "entry_staleness": np.zeros_like(prices),
        "morning_staleness": np.zeros_like(prices),
        "start_date": pd.Timestamp("2023-01-02"),
        "end_date": dates[-1],
        "top": 2,
        "ema_span": 10,
        "min_history_days": 20,
        "minimum_trading_days": 100,
        "transaction_cost_bps": 1.0,
        "max_entry_staleness_minutes": 10,
        "max_exit_staleness_minutes": 1440,
        "entry_price_source": "nbbo-ask",
        "strategy": "liquidity-trend-vol",
        "budget": 10000.0,
        "margin_interest_rate": 0.0675,
        "spy_trend_marks": np.arange(len(dates), dtype=float) + 100,
    }


class StrategyTest(unittest.TestCase):
    def test_policy_uses_twenty_completed_returns_and_lagged_trend(self):
        config = LiquidityTrendVolConfig()
        marks = np.arange(150, dtype=float) + 100
        policy = LiquidityTrendVolPolicy(config, marks)
        returns = [0.01, -0.02] * 10
        for value in returns:
            self.assertEqual(policy.exposure(120), 1.0)
            policy.observe(value)
        expected = min(2.0, 0.35 / (np.std(returns, ddof=1) * np.sqrt(252)))
        self.assertAlmostEqual(policy.exposure(120), expected)
        changed = marks.copy()
        changed[120:] = 1.0
        other = LiquidityTrendVolPolicy(config, changed)
        for value in returns:
            other.observe(value)
        self.assertEqual(other.exposure(120), policy.exposure(120))
        self.assertAlmostEqual(other.exposure(121), expected * 0.25)

    def test_zero_volatility_and_missing_trend_keep_research_fallbacks(self):
        policy = LiquidityTrendVolPolicy(LiquidityTrendVolConfig(), np.arange(130, dtype=float))
        for _ in range(20):
            policy.observe(0.0)
        self.assertEqual(policy.exposure(120), 1.0)
        self.assertEqual(policy.exposure(10), 0.25)

    def test_reporting_start_preserves_exposures_and_resets_capital(self):
        args = fixture()
        _, full = run_backtest(**args)
        args["start_date"] = args["dates"][-12]
        _, short = run_backtest(**args)
        expected = pd.DataFrame(full["daily_portfolio"]).iloc[-11:]
        actual = pd.DataFrame(short["daily_portfolio"])
        np.testing.assert_allclose(actual.exposure, expected.exposure)
        np.testing.assert_allclose(
            actual.strategy_return, expected.strategy_return, atol=1e-14
        )
        self.assertEqual(actual.portfolio_start_equity.iloc[0], 10000.0)

    def test_future_returns_cannot_change_current_exposure_or_membership(self):
        args = fixture()
        cut = len(args["dates"]) - 10
        original, first = run_backtest(**args)
        args["morning_prices"][cut + 1 :, 1:] *= 1.2
        args["spy_trend_marks"][cut:] = 1.0
        args["dollar_volume"][cut:, 2] *= 100
        changed, second = run_backtest(**args)
        day = str(args["dates"][cut].date())
        a, b = (
            pd.DataFrame(first["daily_portfolio"]),
            pd.DataFrame(second["daily_portfolio"]),
        )
        np.testing.assert_array_equal(
            a.loc[a.entry_date <= day, "exposure"],
            b.loc[b.entry_date <= day, "exposure"],
        )
        self.assertEqual(
            original.loc[original.entry_date <= day, "sample_id"].tolist(),
            changed.loc[changed.entry_date <= day, "sample_id"].tolist(),
        )

    def test_cash_sessions_and_calendar_day_financing(self):
        args = fixture()
        args["entry_session_mask"] = np.ones(len(args["dates"]), dtype=bool)
        args["entry_session_mask"][-6] = False
        args["entry_prices"][-6] = np.nan
        args["strategy_config"] = replace(
            LiquidityTrendVolConfig(), warmup_exposure=2.0, volatility_target=1.0
        )
        trades, summary = run_backtest(**args)
        frame = pd.DataFrame(summary["daily_portfolio"])
        cash = frame.loc[frame.entry_date == str(args["dates"][-6].date())].iloc[0]
        self.assertEqual(cash.strategy_return, 0.0)
        self.assertEqual(cash.borrow_cost, 0.0)
        self.assertEqual(summary["missing_benchmark_sessions"], 0)
        self.assertTrue(
            np.isfinite(summary["spy_buy_and_hold_metrics"]["total_return"])
        )
        hold = (
            pd.to_datetime(frame.exit_date) - pd.to_datetime(frame.entry_date)
        ).dt.days
        expected = (
            np.maximum(frame.capital_deployed - frame.portfolio_start_equity, 0)
            * 0.0675
            * hold
            / 360
        )
        np.testing.assert_allclose(frame.borrow_cost, expected)
        self.assertTrue((hold == 3).any())
        pnl = (
            trades.groupby("entry_date")
            .net_pnl.sum()
            .reindex(frame.entry_date, fill_value=0)
            .to_numpy()
        )
        np.testing.assert_allclose(
            frame.portfolio_end_equity, frame.portfolio_start_equity + pnl - expected
        )

    def test_fixed_and_trend_vol_share_accounting(self):
        args = fixture()
        args["strategy_config"] = replace(
            LiquidityTrendVolConfig(),
            max_exposure=1.0,
            warmup_exposure=1.0,
            volatility_target=10.0,
            weak_trend_multiplier=1.0,
        )
        trend_vol_trades, trend_vol = run_backtest(**args)
        args.update(strategy="liquidity-fixed", strategy_config=None)
        fixed_trades, fixed = run_backtest(**args)
        np.testing.assert_allclose(trend_vol_trades.quantity, fixed_trades.quantity)
        np.testing.assert_allclose(
            pd.DataFrame(trend_vol["daily_portfolio"]).strategy_return,
            pd.DataFrame(fixed["daily_portfolio"]).strategy_return,
        )

    def test_fixed_exposure_sizes_trades_and_compounds_financing(self):
        args = fixture()
        args.update(strategy="liquidity-fixed", leverage=2.0)
        trades, summary = run_backtest(**args)
        frame = pd.DataFrame(summary["daily_portfolio"])
        np.testing.assert_allclose(
            frame.capital_deployed, 2 * frame.portfolio_start_equity
        )
        np.testing.assert_allclose(
            frame.portfolio_start_equity.iloc[1:], frame.portfolio_end_equity.iloc[:-1]
        )
        self.assertAlmostEqual(
            summary["ending_equity"], frame.portfolio_end_equity.iloc[-1]
        )
        self.assertAlmostEqual(trades.entry_notional.iloc[0], 10000.0)

    def test_whole_share_financing_uses_actual_borrowing(self):
        args = fixture()
        args.update(share_mode="whole", budget=1000.0)
        trades, summary = run_backtest(**args)
        frame = pd.DataFrame(summary["daily_portfolio"])
        np.testing.assert_array_equal(trades.quantity, np.floor(trades.quantity))
        self.assertTrue(
            (
                frame.capital_deployed
                <= frame.portfolio_start_equity * frame.exposure + 1e-10
            ).all()
        )
        hold = (
            pd.to_datetime(frame.exit_date) - pd.to_datetime(frame.entry_date)
        ).dt.days
        np.testing.assert_allclose(
            frame.borrow_cost,
            np.maximum(frame.capital_deployed - frame.portfolio_start_equity, 0)
            * 0.0675
            * hold
            / 360,
        )

    def test_missing_selected_entry_or_exit_fails(self):
        for column in ("entry_prices", "morning_prices"):
            args = fixture()
            args[column][-4, 1] = np.nan
            with self.assertRaisesRegex(ValueError, "complete fresh prices"):
                run_backtest(**args)

    def test_exit_after_requested_end_is_excluded(self):
        args = fixture()
        args["end_date"] = pd.Timestamp("2023-02-04")  # Saturday
        _, summary = run_backtest(**args)
        self.assertLessEqual(pd.Timestamp(summary["last_exit_date"]), args["end_date"])

    def test_cli_and_config_validation(self):
        self.assertEqual(build_parser().parse_args([]).spy_trend_price_source, "daily-close")
        self.assertEqual(
            build_parser().parse_args(["--spy-trend-price-source", "minute-open-1559"]).spy_trend_price_source,
            "minute-open-1559",
        )
        self.assertEqual(build_parser().parse_args([]).strategy, "liquidity-momentum-focus")
        self.assertEqual(
            build_parser().parse_args(["--strategy", "liquidity-fixed"]).strategy, "liquidity-fixed"
        )
        self.assertEqual(
            build_parser().parse_args(["--strategy", "liquidity-trend-vol"]).strategy, "liquidity-trend-vol"
        )
        for values in (
            {"volatility_target": float("nan")},
            {"max_exposure": 3},
            {"trend_window": 2.5},
            {"warmup_exposure": 0},
            {"weak_trend_multiplier": -1},
            {"risk_history_start": "bad"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LiquidityTrendVolConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"volatility_target": 0.25}))
            self.assertEqual(load_trend_vol_config(path).volatility_target, 0.25)
            path.write_text('{"volatilty_target": 0.25}')
            with self.assertRaisesRegex(ValueError, "invalid liquidity-trend-vol config"):
                load_trend_vol_config(path)


if __name__ == "__main__":
    unittest.main()
