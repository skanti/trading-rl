from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "build_most_liquid.py"
SPEC = importlib.util.spec_from_file_location("build_most_liquid", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_most_liquid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_most_liquid)


def daily_rows(volumes: list[int]) -> np.ndarray:
    rows = []
    for index, volume in enumerate(volumes):
        rows.append(
            [
                index * 86_400,
                10_000,
                10_000,
                10_000,
                10_000,
                volume,
                100,
                10_000,
            ]
        )
    return np.asarray(rows, dtype=np.int64)


class MostLiquidTests(unittest.TestCase):
    def make_bars_dir(self, directory: str) -> Path:
        bars_dir = Path(directory)
        (bars_dir / "_download_manifest.json").write_text(
            json.dumps(
                {
                    "timeframe": "1Day",
                    "adjustment": "split",
                    "columns": [
                        "seconds",
                        "open_mills",
                        "high_mills",
                        "low_mills",
                        "close_mills",
                        "volume",
                        "trades",
                        "vwap_mills",
                    ],
                }
            ),
            encoding="utf-8",
        )
        np.save(bars_dir / "FADED.npy", daily_rows([9_000, 1, 1]))
        np.save(bars_dir / "CURRENT.npy", daily_rows([10, 5_000, 6_000]))
        return bars_dir

    def test_trailing_window_drops_a_stale_past_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = self.make_bars_dir(directory)

            everything, all_sessions = build_most_liquid.historical_top_symbols(
                bars_dir, "2010-01-01", top=1, lookback_sessions=None
            )
            trailing, trailing_sessions = build_most_liquid.historical_top_symbols(
                bars_dir, "2010-01-01", top=1, lookback_sessions=2
            )

        self.assertEqual(everything, ["CURRENT", "FADED"])
        self.assertEqual(all_sessions, 3)
        self.assertEqual(trailing, ["CURRENT"])
        self.assertEqual(trailing_sessions, 2)

    def test_lookback_beyond_available_history_keeps_all_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = self.make_bars_dir(directory)

            symbols, sessions = build_most_liquid.historical_top_symbols(
                bars_dir, "2010-01-01", top=1, lookback_sessions=500
            )

        self.assertEqual(symbols, ["CURRENT", "FADED"])
        self.assertEqual(sessions, 3)

    def test_symbols_are_prioritized_by_consistent_top_n_appearances(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            (bars_dir / "_download_manifest.json").write_text(
                json.dumps(
                    {
                        "timeframe": "1Day",
                        "adjustment": "split",
                        "columns": [
                            "seconds",
                            "open_mills",
                            "high_mills",
                            "low_mills",
                            "close_mills",
                            "volume",
                            "trades",
                            "vwap_mills",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.save(bars_dir / "AAA_SPIKE.npy", daily_rows([9_000, 1, 1]))
            np.save(bars_dir / "ZZZ_STABLE.npy", daily_rows([10, 5_000, 6_000]))

            symbols, _ = build_most_liquid.historical_top_symbols(
                bars_dir, "2010-01-01", top=1, lookback_sessions=None
            )

        self.assertEqual(symbols, ["ZZZ_STABLE", "AAA_SPIKE"])

    def test_non_positive_explicit_lookback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = self.make_bars_dir(directory)
            with self.assertRaisesRegex(ValueError, "lookback sessions must be positive"):
                build_most_liquid.historical_top_symbols(
                    bars_dir, "2010-01-01", top=1, lookback_sessions=0
                )


if __name__ == "__main__":
    unittest.main()
