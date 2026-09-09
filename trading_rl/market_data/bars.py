"""Shared Alpaca bar retrieval and compact NumPy storage logic."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .schema import BAR_COLUMNS

BAR_EPOCH = datetime(2010, 1, 1, tzinfo=UTC)


def normalize_symbol(value: object) -> str:
    """Return Alpaca's dot-separated symbol from either stored symbol notation."""
    symbol = str(value).upper()
    if symbol.startswith("ST-"):
        symbol = symbol[3:]
    return symbol.replace("-", ".")


def download_alpaca_bars(
    symbols: Sequence[str],
    start: date | datetime,
    end: date | datetime,
    timeframe: str,
    feed: str,
    adjustment: str,
    request_page: Callable[[dict[str, object]], Mapping[str, object]],
) -> dict[str, list[dict[str, Any]]]:
    """Download every page for a multi-symbol Alpaca historical-bars request.

    Authentication, retry, and rate-limit policy stay with the caller. This shared
    routine owns request construction, pagination, symbol normalization, and response
    validation so live and bulk downloads cannot interpret the endpoint differently.
    """
    if timeframe not in BAR_COLUMNS:
        raise ValueError(f"unsupported bar timeframe: {timeframe}")
    cleaned = list(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols))
    if not cleaned:
        raise ValueError("at least one symbol is required")
    params: dict[str, object] = {
        "symbols": ",".join(cleaned),
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": feed,
        "adjustment": adjustment,
        "limit": 10_000,
        "sort": "asc",
    }
    output: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in cleaned}
    page_token: str | None = None
    while True:
        payload = request_page(dict(params))
        if not isinstance(payload, Mapping):
            raise ValueError("Alpaca bars response must be an object")
        page_bars = payload.get("bars", {})
        if not isinstance(page_bars, Mapping):
            raise ValueError("Alpaca response bars must be an object")
        for raw_symbol, raw_bars in page_bars.items():
            symbol = normalize_symbol(raw_symbol)
            if symbol not in output:
                continue
            if not isinstance(raw_bars, list):
                raise ValueError(f"Alpaca bars for {symbol} must be a list")
            output[symbol].extend(dict(bar) for bar in raw_bars)
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        next_token = str(next_token)
        if next_token == page_token:
            raise RuntimeError("Alpaca bar pagination token repeated")
        page_token = next_token
        params["page_token"] = next_token
    return output


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def encode_alpaca_bars(
    bars: Sequence[Mapping[str, object]],
    symbol: str,
    timeframe: str,
) -> np.ndarray:
    """Encode Alpaca bars using the shared mill-price NumPy schema."""
    if timeframe not in BAR_COLUMNS:
        raise ValueError(f"unsupported bar timeframe: {timeframe}")
    dtype = np.int64 if timeframe == "1Day" else np.int32
    encoded: list[list[float]] = []
    for bar in bars:
        missing = {"t", "o", "h", "l", "c", "v", "vw"}.difference(bar)
        if missing:
            raise ValueError(f"missing OHLCV fields for {symbol}: {sorted(missing)}")
        timestamp = bar.get("t")
        if not timestamp:
            raise ValueError(f"missing bar timestamp for {symbol}")
        parsed = pd.Timestamp(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(UTC)
        else:
            parsed = parsed.tz_convert(UTC)
        seconds = (parsed.to_pydatetime() - BAR_EPOCH).total_seconds()
        if seconds < 0:
            continue
        if seconds != round(seconds):
            raise ValueError(f"sub-second bar timestamp for {symbol}: {timestamp}")
        values = [
            seconds,
            np.rint(_number(bar.get("o"), f"{symbol}.o") * 1000.0),
        ]
        values.extend(
            np.rint(_number(bar.get(field), f"{symbol}.{field}") * 1000.0)
            for field in ("h", "l", "c")
        )
        values.extend(
            (
                np.rint(_number(bar.get("v"), f"{symbol}.v")),
                np.rint(_number(bar.get("n") or 0, f"{symbol}.n")),
            )
        )
        vwap = bar.get("vw")
        values.append(
            np.rint(
                (0.0 if vwap is None else _number(vwap, f"{symbol}.vw"))
                * 1000.0
            )
        )
        encoded.append(values)
    if not encoded:
        return np.empty((0, len(BAR_COLUMNS[timeframe])), dtype=dtype)
    values = np.asarray(encoded, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite encoded bar value for {symbol}")
    if (values < 0).any():
        raise ValueError(f"negative encoded bar value for {symbol}")
    if values.max() > np.iinfo(dtype).max:
        raise ValueError(f"bar value exceeds {dtype.__name__} for {symbol}")
    array = values.astype(dtype)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    if not (array[:-1, 0] < array[1:, 0]).all():
        raise ValueError(f"duplicate or unsorted bar timestamps for {symbol}")
    return array


def validate_bar_replacement_range(base: np.ndarray, update: np.ndarray) -> None:
    """Require the stored range's boundaries while accepting revised interior bars."""
    if len(update) == 0 or int(update[-1, 0]) < int(base[-1, 0]):
        raise ValueError("replacement ends before the stored endpoint")
    if int(update[0, 0]) > int(base[0, 0]):
        raise ValueError("replacement starts after the stored start")


def merge_bar_arrays(
    base: np.ndarray,
    update: np.ndarray,
    *,
    expected_columns: int | None = None,
    anchor_seconds: int | None = None,
) -> np.ndarray | None:
    """Validate overlap and merge; changed values return ``None`` for full refresh.

    With an explicit anchor, only that bar's values must match. The refreshed
    tail is authoritative, including timestamps removed or restored by the provider.
    Missing anchors or truncated range boundaries still raise.
    """
    for label, array in (("base", base), ("update", update)):
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"invalid {label} bar shape: {array.shape}")
        if expected_columns is not None and array.shape[1] != expected_columns:
            raise ValueError(f"invalid {label} bar shape: {array.shape}")
        if not (array[:-1, 0] < array[1:, 0]).all():
            raise ValueError(f"{label} bar timestamps are not strictly increasing")
    if base.shape[1] != update.shape[1]:
        raise ValueError(
            f"base/update bar column mismatch: {base.shape[1]} != {update.shape[1]}"
        )
    if anchor_seconds is not None:
        base_start = int(np.searchsorted(base[:, 0], anchor_seconds))
        update_start = int(np.searchsorted(update[:, 0], anchor_seconds))
        if base_start == len(base) or int(base[base_start, 0]) != anchor_seconds:
            raise ValueError("expected anchor is absent from stored bars")
        if update_start == len(update) or int(update[update_start, 0]) != anchor_seconds:
            raise ValueError("expected anchor is missing from update; overlap cannot be verified")
        tail = update[update_start:]
        validate_bar_replacement_range(base[base_start:], tail)
        if not np.array_equal(base[base_start], tail[0]):
            return None
        return np.vstack((base[:base_start], tail))
    first = int(update[0, 0])
    last = int(base[-1, 0])
    if first > last:
        return None
    overlap_end = min(last, int(update[-1, 0]))
    base_start = int(np.searchsorted(base[:, 0], first, side="left"))
    base_end = int(np.searchsorted(base[:, 0], overlap_end, side="right"))
    update_end = int(np.searchsorted(update[:, 0], overlap_end, side="right"))
    if not np.array_equal(base[base_start:base_end], update[:update_end]):
        return None
    if int(update[-1, 0]) <= last:
        return base.copy()
    merged = np.vstack((base[:base_start], update))
    if not (merged[:-1, 0] < merged[1:, 0]).all():
        raise ValueError("merged bar timestamps are not strictly increasing")
    return merged


def atomic_save_bar_array(
    path: Path,
    array: np.ndarray,
    *,
    expected_columns: int | None = None,
) -> None:
    """Atomically replace a validated compact bar array."""
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"invalid bar array shape: {array.shape}")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise ValueError(f"invalid bar array shape: {array.shape}")
    if not (array[:-1, 0] < array[1:, 0]).all():
        raise ValueError("bar timestamps must be strictly increasing")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o664
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        np.save(temporary, array)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(mode)
    os.replace(temporary_path, path)
