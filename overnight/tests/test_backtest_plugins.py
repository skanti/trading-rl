"""Experimental policies share execution without entering live strategy discovery."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_strategies import fixture

from research.liquidity_momentum import LiquidityMomentumConfig, LiquidityMomentumPolicy
from trading_rl.overnight.momentum import LiquidityMomentumFocusConfig
from research.liquidity_regime import LiquidityRegimeConfig, LiquidityRegimePolicy
from trading_rl.overnight.backtest import build_parser, run_backtest
from trading_rl.overnight.backtest_strategies import (
    PolicyContext,
    StrategySpec,
    get_strategy,
    register_plugin,
)
from trading_rl.overnight.portfolio import basket_quantities, basket_returns
from trading_rl.overnight.strategies import STRATEGIES, LiquidityTrendVolConfig


class BacktestPluginTest(unittest.TestCase):
    def test_focus_uses_one_equal_weight_basket_and_lagged_prices(self):
        config = get_strategy("liquidity-momentum-focus").load_config()
        self.assertEqual((config.allocation_window, config.allocation_count), (10, 3))
        self.assertEqual((config.trend_window, config.volatility_window, config.volatility_target), (100, 20, .35))
        self.assertIsNone(config.allocation_windows)
        self.assertIn("liquidity-momentum-focus", STRATEGIES)
        dates = pd.bdate_range("2023-01-01", periods=180)
        symbols = np.array([f"S{i}" for i in range(6)])
        daily = 100. + np.arange(180)[:, None] * np.arange(1, 7)[None, :]
        spy = np.arange(180) + 100.
        policy = LiquidityMomentumPolicy(config, spy)
        policy.prepare(PolicyContext(dates, symbols, daily))
        np.testing.assert_array_equal(policy.weights(160, np.arange(6)), [0, 0, 0, 1/3, 1/3, 1/3])
        changed = daily.copy()
        changed[160:, 0] *= 100
        other = LiquidityMomentumPolicy(config, spy)
        other.prepare(PolicyContext(dates, symbols, changed))
        np.testing.assert_array_equal(policy.weights(160, np.arange(6)), other.weights(160, np.arange(6)))
        self.assertGreater(other.weights(161, np.arange(6))[0], 0)
        for values in ({"allocation_windows": [5, 10]}, {"allocation_count": 0}):
            with self.assertRaises(ValueError):
                LiquidityMomentumFocusConfig(**values)

    def test_focus_matches_existing_engine_and_preserves_warmup_on_short_reports(self):
        args = fixture()
        x = np.arange(len(args["dates"]))
        daily = np.column_stack([100 + x, 100 + 2 * x, 100 + x / 2])
        config = LiquidityMomentumFocusConfig()
        args.update(strategy="liquidity-momentum-focus", strategy_config=config, daily_closes=daily)
        trades, summary = run_backtest(**args)
        comparison, old = run_backtest(**dict(args, strategy="liquidity-momentum-vol",
                                             strategy_config=LiquidityMomentumConfig(**config.as_dict())))
        pd.testing.assert_frame_equal(trades, comparison)
        self.assertEqual(summary["daily_portfolio"], old["daily_portfolio"])
        _, short = run_backtest(**dict(args, start_date=args["dates"][-12]))
        np.testing.assert_allclose(pd.DataFrame(short["daily_portfolio"]).exposure,
                                   pd.DataFrame(summary["daily_portfolio"]).exposure.iloc[-11:])

    def test_weighted_sizing_return_and_reserved_cash(self):
        prices = np.array([10., 20.])
        weights = np.array([0.25, 0.5])
        np.testing.assert_array_equal(basket_quantities(prices, 1000, weights=weights), [25., 25.])
        result = basket_returns(prices, [11, 18], slots=2, weights=weights, cost_bps=1)
        self.assertAlmostEqual(result.unscaled_return, 0.25 * 0.1 - 0.5 * 0.1 - 0.75 * 0.0002)
        missing = basket_returns(prices, [11, np.nan], slots=2, weights=weights)
        self.assertAlmostEqual(missing.unscaled_return, 0.025)
        for invalid in ([0.5, 0.6], [-0.1, 0.5], [float("nan"), 0], [1]):
            with self.assertRaises(ValueError):
                basket_quantities(prices, 1000, weights=invalid)

    def test_momentum_allocations_are_causal_and_engine_accounts_for_them(self):
        args = fixture()
        rows = np.arange(len(args["dates"]))
        daily = np.column_stack([100 + rows, 100 + 2 * rows, 100 + rows / 2])
        config = LiquidityMomentumConfig(allocation_window=20, trend_window=20)
        args.update(strategy="liquidity-momentum-vol", strategy_config=config, daily_closes=daily)
        trades, summary = run_backtest(**args)
        first = trades.iloc[:2]
        self.assertAlmostEqual(first.entry_notional.iloc[0] / first.entry_notional.iloc[1], 2.0)
        expected_unit = (2 * (args["morning_prices"][132, 1] / 100 - 1)
                         + (args["morning_prices"][132, 2] / 100 - 1)) / 3 - 0.0002
        day = str(args["dates"][131].date())
        actual = next(row for row in summary["daily_portfolio"] if row["entry_date"] == day)
        self.assertAlmostEqual(actual["unscaled_return"], expected_unit)
        cut = len(args["dates"]) - 10
        changed_daily = daily.copy()
        changed_daily[cut:, 2] *= 100
        changed, _ = run_backtest(**dict(args, daily_closes=changed_daily))
        day = str(args["dates"][cut].date())
        pd.testing.assert_frame_equal(trades.loc[trades.entry_date <= day], changed.loc[changed.entry_date <= day])
        context = PolicyContext(args["dates"], args["symbols"], daily)
        policy = LiquidityMomentumPolicy(config, args["spy_trend_marks"])
        policy.prepare(context)
        np.testing.assert_allclose(policy.weights(cut, np.array([1, 2])), [2 / 3, 1 / 3])
        self.assertNotIn("liquidity-momentum-vol", STRATEGIES)

    def test_momentum_blend_averages_allocations_without_extra_exposure(self):
        args = fixture()
        n = len(args["dates"])
        x = np.arange(n, dtype=float)
        daily = np.column_stack([100 + x, 100 + x + 15 * np.sin(x / 4), 100 + .8 * x])
        context = PolicyContext(args["dates"], args["symbols"], daily)
        policies = []
        for windows in ((5,), (10,), (5, 10)):
            config = LiquidityMomentumConfig(allocation_windows=windows, allocation_count=1, trend_window=20)
            policy = LiquidityMomentumPolicy(config, args["spy_trend_marks"])
            policy.prepare(context)
            policies.append(policy)
        for row in range(120, n):
            weights = [policy.weights(row, np.array([1, 2])) for policy in policies]
            np.testing.assert_allclose(weights[2], (weights[0] + weights[1]) / 2)
            self.assertAlmostEqual(weights[2].sum(), 1)
        args.update(strategy="liquidity-momentum-vol", strategy_config=config, daily_closes=daily)
        _, summary = run_backtest(**args)
        self.assertEqual(summary["skipped_selections"], 0)
        for bad in ([], [5, 5], [True, 10], [0, 10]):
            with self.assertRaises(ValueError):
                LiquidityMomentumConfig(allocation_windows=bad)

    def test_cli_defaults_and_live_separation(self):
        current = build_parser().parse_args([])
        experimental = build_parser().parse_args(["--strategy", "liquidity-regime-vol"])
        self.assertEqual((current.strategy, current.top, current.ema_span), ("liquidity-momentum-focus", 12, 10))
        self.assertEqual((experimental.top, experimental.ema_span), (6, 20))
        self.assertNotIn("liquidity-regime-vol", STRATEGIES)
        explicit = build_parser().parse_args(["--strategy", "liquidity-regime-vol", "--top", "8", "--ema-span", "5"])
        self.assertEqual((explicit.top, explicit.ema_span), (8, 5))
        blend = build_parser().parse_args(["--strategy", "liquidity-momentum-blend"])
        self.assertEqual((blend.top, blend.ema_span), (12, 10))
        config = get_strategy(blend.strategy).load_config()
        self.assertEqual(config.allocation_windows, (5, 10))
        self.assertEqual((config.allocation_count, config.volatility_window, config.trend_window), (4, 20, 100))
        self.assertEqual(config.weak_trend_multiplier, 0)
        self.assertNotIn(blend.strategy, STRATEGIES)

    def test_calendar_filter_uses_entry_weekday_and_preserves_observations(self):
        args = fixture()
        args.update(strategy="liquidity-regime-vol", strategy_config=LiquidityRegimeConfig(trend_window=20, excluded_entry_weekday=0))
        trades, summary = run_backtest(**args)
        self.assertFalse((pd.to_datetime(trades.entry_date).dt.dayofweek == 0).any())
        sessions = pd.DataFrame(summary["daily_portfolio"])
        monday = sessions.loc[pd.to_datetime(sessions.entry_date).dt.dayofweek == 0]
        self.assertTrue((monday.exposure == 0).all())
        self.assertTrue((monday.unscaled_return != 0).any())
        with self.assertRaises(ValueError):
            LiquidityRegimeConfig(excluded_entry_weekday=5)

    def test_matching_parameters_reproduce_promoted_accounting(self):
        args = fixture()
        original_trades, original = run_backtest(**args)
        args.update(strategy="liquidity-regime-vol", strategy_config=LiquidityRegimeConfig(**LiquidityTrendVolConfig().as_dict()))
        trades, experimental = run_backtest(**args)
        pd.testing.assert_frame_equal(trades, original_trades)
        self.assertEqual(experimental["daily_portfolio"], original["daily_portfolio"])
        self.assertTrue(experimental["experimental"])

    def test_cash_regime_preserves_risk_observations_and_reenters(self):
        args = fixture()
        args.update(strategy="liquidity-regime-vol", strategy_config=LiquidityRegimeConfig(trend_window=10))
        cut = len(args["dates"]) - 15
        args["spy_trend_marks"][cut:] = 1.0
        trades, summary = run_backtest(**args)
        frame = pd.DataFrame(summary["daily_portfolio"])
        cash = frame.loc[frame.exposure == 0]
        self.assertGreater(len(cash), 0)
        self.assertTrue((cash.capital_deployed == 0).all())
        self.assertTrue((cash.borrow_cost == 0).all())
        self.assertTrue((cash.portfolio_start_equity == cash.portfolio_end_equity).all())
        self.assertFalse(trades.entry_date.isin(cash.entry_date).any())
        self.assertGreater(frame.exposure.iloc[-1], 0)
        # Even cash sessions observe the modeled basket for future volatility.
        self.assertTrue((cash.unscaled_return != 0).any())
        args["start_date"] = args["dates"][-12]
        _, short = run_backtest(**args)
        np.testing.assert_allclose(pd.DataFrame(short["daily_portfolio"]).exposure, frame.exposure.iloc[-11:])

    def test_entire_report_can_be_cash(self):
        args = fixture()
        args.update(strategy="liquidity-regime-vol", strategy_config=LiquidityRegimeConfig())
        args["spy_trend_marks"] = np.arange(len(args["dates"]), 0, -1, dtype=float)
        trades, summary = run_backtest(**args)
        self.assertTrue(trades.empty)
        self.assertEqual(summary["ending_equity"], 10000.0)
        self.assertEqual(summary["strategy_metrics"]["total_return"], 0.0)

    def test_confirmation_and_buffer_use_only_completed_daily_closes(self):
        marks = np.arange(180, dtype=float) + 100
        config = LiquidityRegimeConfig(trend_window=20, momentum_window=10, trend_buffer=0.01)
        policy = LiquidityRegimePolicy(config, marks)
        changed = marks.copy()
        changed[160:] = 1.0
        other = LiquidityRegimePolicy(config, changed)
        self.assertEqual(policy.exposure(160), other.exposure(160))
        self.assertEqual(other.exposure(161), 0.0)

    def test_config_is_strict_and_does_not_loosen_live_validation(self):
        for values in ({"weak_trend_multiplier": -0.1}, {"trend_buffer": float("nan")},
                       {"momentum_window": 2.5}, {"max_exposure": 3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LiquidityRegimeConfig(**values)
        with self.assertRaises(ValueError):
            LiquidityTrendVolConfig(weak_trend_multiplier=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"momentum_window": 20}))
            self.assertEqual(get_strategy("liquidity-regime-vol").load_config(path).momentum_window, 20)
            path.write_text('{"unknown_setting": 1}')
            with self.assertRaisesRegex(ValueError, "invalid liquidity-regime-vol config"):
                get_strategy("liquidity-regime-vol").load_config(path)

    def test_external_plugin_and_invalid_exposure(self):
        module = types.ModuleType("test_risk_plugin")

        class InvalidPolicy(LiquidityRegimePolicy):
            def exposure(self, row):
                return -1.0

        module.STRATEGY = StrategySpec("test-risk-plugin", "Test only", LiquidityRegimeConfig, InvalidPolicy, True)
        with patch.dict(sys.modules, {module.__name__: module}):
            parsed = build_parser().parse_args(["--strategy-plugin", module.__name__, "--strategy", "test-risk-plugin"])
            args = fixture()
            args.update(strategy=parsed.strategy, strategy_config=LiquidityRegimeConfig())
            with self.assertRaisesRegex(ValueError, "invalid exposure"):
                run_backtest(**args)
            module.STRATEGY = StrategySpec("liquidity-fixed", "Conflict", LiquidityRegimeConfig, InvalidPolicy, True)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_plugin(module.__name__)
            module.STRATEGY = StrategySpec("liquidity-momentum-blend", "Conflict", LiquidityRegimeConfig, InvalidPolicy, True)
            with self.assertRaisesRegex(ValueError, "reserved"):
                register_plugin(module.__name__)


if __name__ == "__main__":
    unittest.main()
