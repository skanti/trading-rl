"""Build day/index metadata from per-symbol market ``.npy`` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from trading_rl.market_data.schema import validate_bar_columns
import pandas as pd
from tqdm import tqdm


RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60
FULL_SESSION_INTERVALS = RTH_CLOSE_MINUTE - RTH_OPEN_MINUTE


def rows_for_file(
    npy_path: Path,
    anno: str,
    window_size: int,
    rollout_size: int,
) -> list[tuple[str, str, int, int, int]]:
    data = np.load(npy_path, mmap_mode="r")
    validate_bar_columns(data, "1Min", str(npy_path))
    secs = np.asarray(data[:, 0])
    if secs.size < window_size + rollout_size or not np.all(secs[:-1] < secs[1:]):
        return []
    dt = pd.to_datetime(secs, unit="s", origin=anno, utc=True).tz_convert("US/Eastern")
    dates = np.asarray(dt.date)
    minutes = np.asarray(dt.hour * 60 + dt.minute)
    seconds = np.asarray(dt.second)
    rows: list[tuple[str, str, int, int, int]] = []
    for day in np.unique(dates):
        valid = np.flatnonzero((dates == day) & (minutes >= RTH_OPEN_MINUTE) & (minutes <= RTH_CLOSE_MINUTE))
        if valid.size < rollout_size + 1:
            continue
        if rollout_size == FULL_SESSION_INTERVALS:
            session_secs = secs[valid]
            is_complete = (
                valid.size == FULL_SESSION_INTERVALS + 1
                and minutes[valid[0]] == RTH_OPEN_MINUTE
                and minutes[valid[-1]] == RTH_CLOSE_MINUTE
                and np.all(seconds[valid] == 0)
                and np.all(np.diff(session_secs) == 60)
            )
            if not is_complete:
                continue
        sod_idx, eod_idx = int(valid[0]), int(valid[-1])
        if sod_idx < window_size - 1 or eod_idx - sod_idx < rollout_size:
            continue
        rows.append((npy_path.stem, day.isoformat(), sod_idx, eod_idx, 0))
    return rows


def build_days(
    data_dir: str,
    output_path: str,
    anno: str,
    window_size: int,
    rollout_size: int,
) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("*.npy"))
    if not paths:
        raise ValueError(f"no .npy files found in {data_dir}")
    rows: list[tuple[str, str, int, int, int]] = []
    for path in tqdm(paths, desc="index market days"):
        rows.extend(rows_for_file(path, anno, window_size, rollout_size))
    days = pd.DataFrame(rows, columns=("sample_id", "date", "sod_idx", "eod_idx", "ctx_idx"))
    if days.empty:
        raise ValueError("no regular-session samples met the requested window and rollout sizes")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    days.to_csv(output, index=False)
    return days


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--anno", default="2010-01-01")
    parser.add_argument("--window_size", type=int, default=4096)
    parser.add_argument("--rollout_size", type=int, default=64)
    args = parser.parse_args()
    result = build_days(args.data_dir, args.output_path, args.anno, args.window_size, args.rollout_size)
    print(f"wrote {len(result)} symbol-days to {args.output_path}")
