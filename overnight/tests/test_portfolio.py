"""Shared allocation and modeled-return edge cases used by live and simulation."""

import unittest

import numpy as np

from trading_rl.overnight.portfolio import (
    basket_quantities,
    basket_returns,
    equal_notional,
)
from trading_rl.overnight.risk_history import unit_return
from trading_rl.overnight.strategies import (
    LiquidityTrendVolConfig,
    LiquidityTrendVolPolicy,
)


class PortfolioTest(unittest.TestCase):
    def test_unavailable_slots_and_unaffordable_whole_shares_stay_in_cash(self):
        prices = np.array([100.0, 250.0, 600.0])
        fractional = basket_quantities(prices, 1000, slots=4)
        whole = basket_quantities(prices, 1000, "whole", slots=4)
        np.testing.assert_allclose(fractional * prices, [250, 250, 250])
        np.testing.assert_array_equal(whole, [2, 1, 0])
        self.assertEqual(np.sum(whole * prices), 450)
        self.assertEqual(equal_notional(1000, 3, round_to_cents=True), 333.33)
        self.assertEqual(equal_notional(1000, 3), 1000 / 3)
        with self.assertRaises(ValueError):
            basket_quantities(prices, 1000, slots=2)

    def test_price_validation_cash_weights_and_live_complete_history(self):
        observed = basket_returns(
            [100, 200],
            [105, 190],
            slots=4,
            cost_bps=10,
            exit_staleness=[0, 2],
            max_exit_age=1,
        )
        np.testing.assert_array_equal(observed.missing, [False, True])
        self.assertAlmostEqual(observed.unscaled_return, 0.012)
        with self.assertRaisesRegex(ValueError, "complete positive"):
            basket_returns(
                [100, 200],
                [105, 190],
                slots=4,
                require_complete=True,
                exit_staleness=[0, 2],
                max_exit_age=1,
            )
        self.assertTrue(
            basket_returns([100], [105], slots=1, entry_staleness=[-1]).missing[0]
        )
        self.assertEqual(basket_returns([], [], slots=12).unscaled_return, 0)
        complete = basket_returns(
            [100, 200], [105, 190], slots=2, require_complete=True
        )
        self.assertEqual(unit_return([100, 200], [105, 190]), complete.unscaled_return)
        with self.assertRaises(ValueError):
            unit_return(100, 101)

    def test_risk_reporting_and_policy_share_the_completed_trailing_window(self):
        policy = LiquidityTrendVolPolicy(
            LiquidityTrendVolConfig(), np.arange(130) + 100.0
        )
        self.assertIsNone(policy.annualized_volatility)
        policy.observe(
            50.0
        )  # An old shock falls out of both risk reporting and sizing.
        recent = [0.02, -0.01] * 10
        for value in recent:
            policy.observe(value)
        volatility = float(np.std(recent, ddof=1) * np.sqrt(252))
        self.assertEqual(policy.annualized_volatility, volatility)
        self.assertEqual(policy.exposure(129), min(2.0, 0.35 / volatility))
