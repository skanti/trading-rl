"""Lightweight shared helpers for resolving trade-derived prices."""

from __future__ import annotations

import numpy as np


def forward_fill_positions(
    source: np.ndarray, secs: np.ndarray, sample_id: str
) -> np.ndarray:
    """Locate the latest valid trade at or before every requested timestamp."""
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"{sample_id} must contain timestamp and open-price columns")
    # Keep the mmap-backed seconds column zero-copy; evaluation calls this for
    # many large windows and the stored int32 range is sufficient here.
    source_secs = np.asarray(source[:, 0])
    position = np.searchsorted(source_secs, secs, side="right") - 1
    if (position < 0).any():
        raise ValueError(f"{sample_id} has no print at or before the requested time")

    selected_prices = np.asarray(source[position, 1], dtype=np.float64)
    invalid = ~np.isfinite(selected_prices) | (selected_prices <= 0)
    if invalid.any():
        position = position.copy()
        # Invalid prints are missing observations, not prices. Walk backward
        # to the preceding valid print rather than leaking the next valid one.
        for invalid_position in np.unique(position[invalid]):
            valid_position = int(invalid_position) - 1
            while valid_position >= 0:
                price = float(source[valid_position, 1])
                if np.isfinite(price) and price > 0:
                    break
                valid_position -= 1
            if valid_position < 0:
                raise ValueError(
                    f"{sample_id} has no valid price at or before the requested time"
                )
            position[position == invalid_position] = valid_position
    return position
