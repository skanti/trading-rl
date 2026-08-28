from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "download_bars.py"
SPEC = importlib.util.spec_from_file_location("download_bars", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download_bars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_bars)


def bars(*rows: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int32)


class BarUpdateTests(unittest.TestCase):
    def test_storage_names_drop_legacy_asset_prefix(self):
        self.assertEqual(download_bars.storage_ticker("ST-BRK-B"), "BRK.B")
        self.assertEqual(download_bars.storage_ticker("BRK.B"), "BRK.B")
        self.assertEqual(download_bars.clean_ticker("ST-BRK-B"), "BRK.B")

    def test_minute_schema_skips_split_adjusted_price_overflow(self):
        frame = download_bars.pd.DataFrame(
            [
                {
                    "t": "2026-08-27T19:59:00Z",
                    "o": 2_500_000.0,
                    "v": 1_000,
                    "n": 100,
                }
            ]
        )

        array = download_bars.dataframe_to_array(frame, "ABTC.npy", "1Min")

        self.assertIsNone(array)

    def test_daily_schema_retains_ohlcv_trades_and_vwap(self):
        frame = download_bars.pd.DataFrame(
            [
                {
                    "t": "2026-08-27T04:00:00Z",
                    "o": 100.125,
                    "h": 105.5,
                    "l": 98.25,
                    "c": 104.75,
                    "v": 1_000_000,
                    "n": 42_000,
                    "vw": 102.375,
                }
            ]
        )

        array = download_bars.dataframe_to_array(frame, "AAPL.npy", "1Day")

        self.assertIsNotNone(array)
        self.assertEqual(array.shape, (1, 8))
        self.assertEqual(array.dtype, np.int64)
        self.assertEqual(
            array[0, 1:].tolist(),
            [100_125, 105_500, 98_250, 104_750, 1_000_000, 42_000, 102_375],
        )

    def test_exact_overlap_is_replaced_and_new_tail_is_appended(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3))
        update = bars((2, 101, 11, 2), (3, 102, 12, 3), (4, 103, 13, 4))

        merged = download_bars.merge_bar_arrays(base, update)

        np.testing.assert_array_equal(
            merged,
            bars(
                (1, 100, 10, 1),
                (2, 101, 11, 2),
                (3, 102, 12, 3),
                (4, 103, 13, 4),
            ),
        )

    def test_adjusted_overlap_change_requires_full_refresh(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3))
        split_adjusted_update = bars(
            (2, 50, 22, 2),
            (3, 51, 24, 3),
            (4, 52, 26, 4),
        )

        self.assertIsNone(download_bars.merge_bar_arrays(base, split_adjusted_update))

    def test_missing_or_nonoverlapping_rows_require_full_refresh(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3))

        self.assertIsNone(
            download_bars.merge_bar_arrays(
                base,
                bars((2, 101, 11, 2), (4, 103, 13, 4)),
            )
        )
        self.assertIsNone(download_bars.merge_bar_arrays(base, bars((4, 103, 13, 4))))

    def test_ticker_update_escalates_to_full_download_on_difference(self):
        day = 24 * 60 * 60
        base = bars(
            (0, 100, 10, 1),
            (20 * day, 101, 11, 2),
            (40 * day, 102, 12, 3),
        )
        changed_tail = bars(
            (20 * day, 50, 22, 2),
            (40 * day, 51, 24, 3),
            (41 * day, 52, 26, 4),
        )
        full_refresh = bars(
            (0, 49, 20, 1),
            (20 * day, 50, 22, 2),
            (40 * day, 51, 24, 3),
            (41 * day, 52, 26, 4),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AAPL.npy"
            np.save(path, base)
            placeholder = download_bars.pd.DataFrame({"download": [1]})
            with (
                mock.patch.object(
                    download_bars,
                    "download_ticker",
                    side_effect=[placeholder, placeholder],
                ) as downloader,
                mock.patch.object(
                    download_bars,
                    "dataframe_to_array",
                    side_effect=[changed_tail, full_refresh],
                ),
            ):
                succeeded = download_bars.process_ticker(
                    "AAPL",
                    source="alpaca",
                    out_dir=directory,
                    since=download_bars.ANNO,
                    update_existing=True,
                    overlap_days=30,
                )

            self.assertTrue(succeeded)
            self.assertEqual(downloader.call_count, 2)
            incremental_since = downloader.call_args_list[0].args[2]
            full_since = downloader.call_args_list[1].args[2]
            self.assertGreater(incremental_since, full_since)
            np.testing.assert_array_equal(np.load(path), full_refresh)


if __name__ == "__main__":
    unittest.main()
