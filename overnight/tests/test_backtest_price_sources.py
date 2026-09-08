import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest import (
    _cache_metadata,
    _symbol_daily_arrays,
    build_parser,
    load_or_build_cache,
)


class MinutePriceSourcesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.minute_dir = self.root / "minute"
        self.daily_dir = self.root / "daily"
        self.minute_dir.mkdir()
        self.daily_dir.mkdir()
        self.dates = pd.DatetimeIndex(["2026-01-05"])
        origin = pd.Timestamp("2010-01-01", tz="UTC")
        session = pd.Timestamp("2026-01-05 04:00", tz="America/New_York")
        self.context = np.array(
            [int((session.tz_convert("UTC") - origin).total_seconds())]
        )
        self.morning = self.context[0] + (9 * 60 + 30 - 4 * 60) * 60
        self.entry = self.context[0] + (15 * 60 + 45 - 4 * 60) * 60
        self.bars = np.array(
            [
                [self.morning, 90_000, 94_000, 88_000, 93_000, 700, 12, 91_000],
                [self.entry, 100_000, 105_000, 98_000, 104_000, 900, 15, 102_000],
            ],
            dtype=np.int32,
        )
        self.save_bars(self.bars)
        daily_second = self.context[0] - 4 * 60 * 60
        np.save(
            self.daily_dir / "SPY.npy",
            np.array(
                [
                    [daily_second, 90_000, 105_000, 88_000, 104_000, 2000, 30, 99_000],
                ],
                dtype=np.int64,
            ),
        )
        # Cache construction needs a tradable stock as well as the benchmark.
        np.save(self.minute_dir / "X.npy", self.bars)
        np.save(self.daily_dir / "X.npy", np.load(self.daily_dir / "SPY.npy"))

    def save_bars(self, bars):
        np.save(self.minute_dir / "SPY.npy", bars)

    def read_prices(self, entry_source, exit_source):
        return _symbol_daily_arrays(
            "SPY",
            {self.dates[0]: 0},
            self.context,
            self.minute_dir,
            self.daily_dir,
            15 * 60 + 45,
            9 * 60 + 30,
            1,
            entry_source,
            exit_source,
        )

    def test_each_entry_and_exit_field_reads_its_own_column(self):
        expected = {
            "minute-open": (100.0, 90.0),
            "minute-high": (105.0, 94.0),
            "minute-low": (98.0, 88.0),
            "minute-close": (104.0, 93.0),
            "minute-vwap": (102.0, 91.0),
        }
        for entry_source, entry_values in expected.items():
            for exit_source, exit_values in expected.items():
                with self.subTest(entry=entry_source, exit=exit_source):
                    args = build_parser().parse_args(
                        [
                            "--entry-price-source",
                            entry_source,
                            "--exit-price-source",
                            exit_source,
                        ]
                    )
                    result = self.read_prices(
                        args.entry_price_source, args.exit_price_source
                    )
                    self.assertEqual(result[1][0], 198_000.0)
                    self.assertEqual(result[2][0], entry_values[0])
                    self.assertEqual(result[3][0], exit_values[1])
                    self.assertEqual(result[4][0], 0.0)
                    self.assertEqual(result[5][0], 0.0)

    def test_missing_vwap_uses_prior_vwap_and_reports_its_age(self):
        bars = self.bars.copy()
        bars[1, 7] = 0
        prior = self.bars[1].copy()
        prior[0] -= 120
        prior[7] = 101_000
        future = self.bars[1].copy()
        future[0] += 60
        future[7] = 999_000
        self.save_bars(np.vstack([bars[0], prior, bars[1], future]))
        result = self.read_prices("minute-vwap", "minute-open")
        self.assertEqual(result[2][0], 101.0)
        self.assertEqual(result[4][0], 2.0)
        self.assertEqual(result[3][0], 90.0)

    def test_no_valid_prior_field_stays_missing_even_with_future_value(self):
        bars = self.bars.astype(float)
        bars[:, 7] = [np.nan, 0]
        future = self.bars[1].copy()
        future[0] += 60
        self.save_bars(np.vstack([bars, future]))
        result = self.read_prices("minute-vwap", "minute-vwap")
        self.assertTrue(np.isnan(result[2][0]))
        self.assertTrue(np.isnan(result[3][0]))
        self.assertTrue(np.isinf(result[4][0]))
        self.assertTrue(np.isinf(result[5][0]))

    def test_empty_minute_file_stays_unavailable(self):
        self.save_bars(np.empty((0, 8), dtype=np.int32))
        result = self.read_prices("minute-close", "minute-vwap")
        self.assertTrue(np.isnan(result[2][0]))
        self.assertTrue(np.isinf(result[5][0]))

    def test_cache_rebuilds_for_either_source_change_and_reuses_matching_sources(self):
        cache_path = self.root / "cache.npz"
        common = (
            self.minute_dir,
            self.daily_dir,
            self.dates,
            self.context,
            15 * 60 + 45,
            9 * 60 + 30,
            1,
            False,
        )
        previous_metadata = None
        for entry, exit_source, entry_price, exit_price in (
            ("minute-open", "minute-open", 100.0, 90.0),
            ("minute-vwap", "minute-open", 102.0, 90.0),
            ("minute-vwap", "minute-close", 102.0, 93.0),
        ):
            with self.subTest(entry=entry, exit=exit_source):
                with patch(
                    "trading_rl.overnight.backtest._manifest_fingerprint",
                    return_value={},
                ):
                    metadata = _cache_metadata(
                        self.minute_dir,
                        self.daily_dir,
                        self.dates[0],
                        self.dates[-1],
                        15 * 60 + 45,
                        9 * 60 + 30,
                        entry,
                        exit_source,
                    )
                self.assertNotEqual(metadata, previous_metadata)
                previous_metadata = metadata
                result = load_or_build_cache(
                    cache_path, metadata, *common, entry, exit_source
                )
                self.assertEqual(result[2][0, 0], entry_price)
                self.assertEqual(result[3][0, 0], exit_price)
                with patch("trading_rl.overnight.backtest.build_daily_cache") as build:
                    reused = load_or_build_cache(
                        cache_path, metadata, *common, entry, exit_source
                    )
                    build.assert_not_called()
                for actual, cached in zip(result, reused):
                    np.testing.assert_array_equal(actual, cached)
