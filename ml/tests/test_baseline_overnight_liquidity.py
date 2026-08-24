import unittest

import numpy as np

from baseline.overnight_liquidity import (
    causal_ema_log_liquidity,
    strategy_metrics,
    top_liquid_indices,
)


class OvernightLiquidityBaselineTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

