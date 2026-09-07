"""Construct full OHLCV test data from compact timestamp/open/volume fixtures.

This only synthesizes test candles; real legacy datasets must be re-downloaded.
"""

import numpy as np


def ohlcv_fixture(compact):
    compact = np.asarray(compact)
    if compact.shape[1] == 8:
        return compact
    seconds, price = compact[:, 0], compact[:, 1]
    volume = compact[:, 2] if compact.shape[1] >= 3 else np.ones(len(compact))
    trades = compact[:, 3] if compact.shape[1] >= 4 else np.ones(len(compact))
    return np.column_stack(
        (
            seconds,
            price,
            price + 2000,
            np.maximum(price - 1000, 0),
            price + 500,
            volume,
            trades,
            price + 200,
        )
    ).astype(compact.dtype)
