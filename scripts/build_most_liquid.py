"""Build a historical liquid-stock shortlist from downloaded daily bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


ANNO = np.datetime64("2010-01-01T00:00:00")
COLUMN_INDEX = {"volume": 5, "trades": 6}
VWAP_INDEX = 7
PRICE_INDICES = (1, 2, 3, 4, 7)
MINUTE_INT32_MAX = np.iinfo(np.int32).max
DEFAULT_BARS_DIR = Path("/data/ppv1/updates/bars_1day_2016-01-01")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "most_liquid.txt"


def _validate_dataset(bars_dir: Path) -> None:
    manifest_path = bars_dir / "_download_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"daily-bar manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("timeframe") != "1Day":
        raise ValueError(
            f"expected a 1Day dataset, found {manifest.get('timeframe')!r}"
        )
    if manifest.get("adjustment") != "split":
        raise ValueError(
            f"expected split-adjusted bars, found {manifest.get('adjustment')!r}"
        )
    columns = tuple(manifest.get("columns", ()))
    expected = ("volume", "trades", "vwap_mills")
    if not all(column in columns for column in expected):
        raise ValueError(f"daily bars do not contain {expected}: {columns}")


def _eligible_rows(array: np.ndarray, since_seconds: int) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] < 7:
        raise ValueError(f"invalid daily-bar shape: {array.shape}")
    return array[array[:, 0] >= since_seconds]


def historical_top_symbols(
    bars_dir: Path,
    since: str,
    top: int = 50,
    metrics: tuple[str, ...] = ("dollar_volume",),
) -> tuple[list[str], int]:
    """Return symbols appearing in a daily top-N for any requested metric."""
    if top < 1:
        raise ValueError("top must be positive")
    unknown_metrics = set(metrics).difference((*COLUMN_INDEX, "dollar_volume"))
    if unknown_metrics:
        raise ValueError(f"unknown metrics: {sorted(unknown_metrics)}")

    _validate_dataset(bars_dir)
    paths = []
    excluded_incompatible = []
    for path in sorted(bars_dir.glob("*.npy")):
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] < 8:
            raise ValueError(f"invalid daily-bar shape in {path}: {array.shape}")
        if any(
            np.max(array[:, index], initial=0) > MINUTE_INT32_MAX
            for index in PRICE_INDICES
        ):
            excluded_incompatible.append(path.stem)
            continue
        paths.append(path)
    if not paths:
        raise ValueError(f"no .npy bars found in {bars_dir}")
    if excluded_incompatible:
        print(
            "excluded symbols incompatible with int32 minute prices: "
            + ", ".join(excluded_incompatible)
        )

    since_day = np.datetime64(since, "D")
    since_seconds = int((since_day - ANNO) / np.timedelta64(1, "s"))
    timestamps: set[int] = set()
    for path in paths:
        array = np.load(path, mmap_mode="r")
        rows = _eligible_rows(array, since_seconds)
        timestamps.update(int(value) for value in rows[:, 0])
    if not timestamps:
        raise ValueError(f"dataset has no rows on or after {since}")

    ordered_timestamps = np.asarray(sorted(timestamps), dtype=np.int64)
    timestamp_to_row = {
        int(timestamp): row for row, timestamp in enumerate(ordered_timestamps)
    }
    metric_values = {
        metric: np.zeros((len(ordered_timestamps), len(paths)), dtype=np.float64)
        for metric in metrics
    }

    for symbol_index, path in enumerate(paths):
        array = np.load(path, mmap_mode="r")
        rows = _eligible_rows(array, since_seconds)
        row_indices = np.fromiter(
            (timestamp_to_row[int(value)] for value in rows[:, 0]),
            dtype=np.int64,
            count=len(rows),
        )
        for metric, values in metric_values.items():
            if metric == "dollar_volume":
                price_mills = np.where(
                    rows[:, VWAP_INDEX] > 0,
                    rows[:, VWAP_INDEX],
                    rows[:, 4],
                )
                values[row_indices, symbol_index] = (
                    rows[:, COLUMN_INDEX["volume"]].astype(np.float64)
                    * price_mills.astype(np.float64)
                    / 1000.0
                )
            else:
                values[row_indices, symbol_index] = rows[:, COLUMN_INDEX[metric]]

    selected_indices: set[int] = set()
    daily_count = min(top, len(paths))
    partition_index = len(paths) - daily_count
    for values in metric_values.values():
        daily_top = np.argpartition(values, partition_index, axis=1)[
            :, partition_index:
        ]
        for day_index, symbol_indices in enumerate(daily_top):
            selected_indices.update(
                int(symbol_index)
                for symbol_index in symbol_indices
                if values[day_index, symbol_index] > 0
            )

    symbols = sorted(paths[index].stem for index in selected_indices)
    return symbols, len(ordered_timestamps)


def _write_symbols(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write("\n".join(symbols) + "\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-dir", type=Path, default=DEFAULT_BARS_DIR)
    parser.add_argument("--since", default="2022-01-01")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--metric",
        choices=("dollar-volume", "volume", "trades"),
        default="dollar-volume",
        help="daily ranking metric (default: volume multiplied by VWAP)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = (args.metric.replace("-", "_"),)
    symbols, trading_days = historical_top_symbols(
        args.bars_dir, args.since, args.top, metrics
    )
    _write_symbols(args.output, symbols)
    print(
        f"wrote {len(symbols):,} symbols to {args.output} "
        f"from {trading_days:,} trading days using {', '.join(metrics)}"
    )


if __name__ == "__main__":
    main()
