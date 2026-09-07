"""Versioned column definitions for downloaded bar arrays."""

import numpy as np


BAR_SCHEMA_VERSION = 2
OHLCV_COLUMNS = (
    "seconds",
    "open_mills",
    "high_mills",
    "low_mills",
    "close_mills",
    "volume",
    "trades",
    "vwap_mills",
)
BAR_COLUMNS = {"1Min": OHLCV_COLUMNS, "1Day": OHLCV_COLUMNS}
BAR_INDEX = {name: index for index, name in enumerate(OHLCV_COLUMNS)}


def validate_bar_columns(array: np.ndarray, timeframe: str, label: str) -> None:
    if array.ndim != 2 or array.shape[1] != len(BAR_COLUMNS[timeframe]):
        raise ValueError(
            f"{label}: expected {timeframe} OHLCV schema v{BAR_SCHEMA_VERSION} "
            f"with columns {BAR_COLUMNS[timeframe]}, got shape {array.shape}. "
            "Re-download legacy bars into a new directory; missing fields cannot be inferred."
        )


def validate_bar_manifest(manifest: dict, timeframe: str, label: str) -> None:
    version = manifest.get("schema_version")
    # Daily layout did not change; existing daily datasets remain readable.
    allowed_versions = (
        (BAR_SCHEMA_VERSION,) if timeframe == "1Min" else (None, 1, BAR_SCHEMA_VERSION)
    )
    if (
        version not in allowed_versions
        or manifest.get("timeframe") != timeframe
        or manifest.get("adjustment") != "split"
        or manifest.get("columns") != list(BAR_COLUMNS[timeframe])
    ):
        raise ValueError(
            f"{label}: incompatible {timeframe} bar schema; expected split-adjusted "
            f"OHLCV schema v{BAR_SCHEMA_VERSION}. Re-download legacy minute bars "
            "into a new directory; do not mix layouts."
        )
