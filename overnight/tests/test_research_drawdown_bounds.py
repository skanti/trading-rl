"""A conservative screen must never conceal a worse exact minute drawdown."""

import unittest

import numpy as np

from research.frontier_screen import drawdown_bound


class DrawdownBoundTest(unittest.TestCase):
    def test_bound_covers_paths_with_financing_and_peak_after_trough(self):
        rng = np.random.default_rng(112)
        paths = rng.normal(0, .007, (40, 100)).cumsum(axis=1)
        paths[:, 0] = 0
        # Reference construction matches the min/max transformation in mark_bounds.
        reference_exposure = rng.uniform(.25, 2, 40)
        reference_borrow = np.maximum(reference_exposure - 1, 0) * .0675 / 360
        fraction = np.linspace(0, 1, 100)
        reference_curves = 1 + reference_exposure[:, None] * paths - reference_borrow[:, None] * fraction
        low = (reference_curves.min(axis=1) - 1) / reference_exposure
        high = (reference_curves.max(axis=1) - 1 + reference_borrow) / reference_exposure
        for _ in range(20):
            exposure = rng.uniform(0, 2, 40)
            borrow = np.maximum(exposure - 1, 0) * .0675 / 360
            returns = exposure * paths[:, -1] - borrow
            starting_equity = np.r_[1, np.cumprod(1 + returns)[:-1]]
            curve = (starting_equity[:, None] * (1 + exposure[:, None] * paths - borrow[:, None] * fraction)).ravel()
            peak = np.maximum.accumulate(np.r_[1, curve])[1:]
            exact = np.max(1 - curve / peak)
            bound = drawdown_bound(returns, exposure, borrow, low, high)
            self.assertGreaterEqual(bound + 1e-12, exact)


if __name__ == "__main__":
    unittest.main()
