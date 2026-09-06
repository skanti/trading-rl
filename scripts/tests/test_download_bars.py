from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


from trading_rl.cli import download_bars


def bars(*rows: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int32)


class BarUpdateTests(unittest.TestCase):
    def test_repetitive_logs_are_summarized_with_example_symbols(self):
        summary = download_bars.DeferredDownloadSummary("DOWNLOAD_BARS")
        records = [
            logging.LogRecord(
                "DOWNLOAD_BARS",
                logging.WARNING,
                __file__,
                1,
                "Historical overlap changed; downloading full retained history, ticker=%s",
                (ticker,),
                None,
            )
            for ticker in ("MSFT", "AAPL")
        ]
        routine_info = logging.LogRecord(
            "DOWNLOAD_BARS",
            logging.INFO,
            __file__,
            1,
            "Incremental update complete, ticker=%s, old_rows=%d, new_rows=%d",
            ("NVDA", 10, 11),
            None,
        )
        startup_info = logging.LogRecord(
            "DOWNLOAD_BARS",
            logging.INFO,
            __file__,
            1,
            "Tickers provided, tickers_num=%d",
            (3,),
            None,
        )

        self.assertTrue(all(not summary.filter(record) for record in records))
        self.assertFalse(summary.filter(routine_info))
        self.assertTrue(summary.filter(startup_info))

        destination = mock.Mock()
        summary.write(destination)
        destination.info.assert_any_call(
            "Download event summary: updates=%d, warnings=%d, errors=%d", 1, 2, 0
        )
        destination.info.assert_any_call(
            "  %s %s: %d%s",
            "WARNING",
            "Historical overlap changed; downloading full retained history",
            2,
            " (e.g. AAPL, MSFT)",
        )

    def test_extensionless_ticker_list_is_loaded_as_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minute-symbols"
            path.write_text("SPY\nAAPL\n\n", encoding="utf-8")

            tickers = download_bars.load_tickers(str(path))

        self.assertEqual(tickers, ["SPY", "AAPL"])

    def test_missing_ticker_list_path_fails_before_an_api_request(self):
        with self.assertRaisesRegex(FileNotFoundError, "ticker list does not exist"):
            download_bars.load_tickers("/tmp/does-not-exist/minute-symbols")

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

    def test_alpaca_batch_collects_all_symbols_across_pages(self):
        pages = [
            {
                "bars": {
                    "AAPL": [
                        {
                            "t": "2026-08-27T04:00:00Z",
                            "o": 100.0,
                            "h": 101.0,
                            "l": 99.0,
                            "c": 100.5,
                            "v": 1_000,
                            "n": 100,
                            "vw": 100.25,
                        }
                    ]
                },
                "next_page_token": "next",
            },
            {
                "bars": {
                    "MSFT": [
                        {
                            "t": "2026-08-27T04:00:00Z",
                            "o": 500.0,
                            "h": 501.0,
                            "l": 499.0,
                            "c": 500.5,
                            "v": 2_000,
                            "n": 200,
                            "vw": 500.25,
                        }
                    ]
                },
                "next_page_token": None,
            },
        ]
        with (
            mock.patch.dict(
                download_bars.os.environ,
                {"ALPACA_DATA_KEY": "key", "ALPACA_DATA_SECRET": "secret"},
            ),
            mock.patch.object(
                download_bars, "request_json", side_effect=pages
            ) as requester,
        ):
            frames = download_bars.download_bars_alpaca_batch(
                ["AAPL", "MSFT"], download_bars.ANNO, "1Day"
            )

        self.assertEqual(set(frames), {"AAPL", "MSFT"})
        self.assertEqual(len(frames["AAPL"]), 1)
        self.assertEqual(len(frames["MSFT"]), 1)
        self.assertEqual(requester.call_count, 2)
        self.assertEqual(requester.call_args_list[0].args[2]["symbols"], "AAPL,MSFT")
        self.assertEqual(requester.call_args_list[1].args[2]["page_token"], "next")

    def test_alpaca_batch_persists_each_symbol_through_normal_processor(self):
        frames = {
            "AAPL": download_bars.pd.DataFrame({"t": ["2026-08-27T04:00:00Z"]}),
            "MSFT": download_bars.pd.DataFrame({"t": ["2026-08-27T04:00:00Z"]}),
        }
        tasks = [
            (0, "AAPL", download_bars.ANNO),
            (1, "MSFT", download_bars.ANNO),
        ]
        with (
            mock.patch.object(
                download_bars, "download_bars_alpaca_batch", return_value=frames
            ) as downloader,
            mock.patch.object(
                download_bars, "process_ticker", side_effect=[True, False]
            ) as processor,
        ):
            outcomes = download_bars.process_alpaca_batch(
                tasks,
                out_dir="/unused",
                since=download_bars.ANNO,
                skip_existing=False,
                update_existing=True,
                overlap_days=30,
                timeframe="1Day",
            )

        self.assertEqual(outcomes, [(0, True), (1, False)])
        downloader.assert_called_once_with(["AAPL", "MSFT"], download_bars.ANNO, "1Day")
        self.assertIs(processor.call_args_list[0].kwargs["initial_df"], frames["AAPL"])
        self.assertIs(processor.call_args_list[1].kwargs["initial_df"], frames["MSFT"])

    def test_equal_range_batches_preserve_input_priority(self):
        scheduled: list[list[str]] = []

        def process(tasks, **_kwargs):
            scheduled.append([ticker for _index, ticker, _since in tasks])
            return [(index, True) for index, _ticker, _since in tasks]

        with tempfile.TemporaryDirectory() as directory:
            tickers_path = Path(directory) / "prioritized.txt"
            tickers_path.write_text("NVDA\nTSLA\nAAPL\nMSFT\n", encoding="utf-8")
            output = Path(directory) / "bars"
            with mock.patch.object(
                download_bars, "process_alpaca_batch", side_effect=process
            ):
                download_bars.main(
                    source="alpaca",
                    tickers_path=str(tickers_path),
                    out_dir=str(output),
                    since=download_bars.ANNO,
                    workers_num=0,
                    update_existing=True,
                    timeframe="1Min",
                    batch_size=2,
                )

        self.assertEqual(scheduled, [["NVDA", "TSLA"], ["AAPL", "MSFT"]])

    def test_failed_batch_is_split_to_isolate_a_bad_symbol(self):
        tasks = [
            (0, "AAPL", download_bars.ANNO),
            (1, "BAD", download_bars.ANNO),
        ]

        def download(symbols, since, timeframe):
            if len(symbols) > 1:
                raise RuntimeError("bad batch")
            return {symbols[0]: download_bars.pd.DataFrame({"t": ["value"]})}

        with (
            mock.patch.object(
                download_bars, "download_bars_alpaca_batch", side_effect=download
            ) as downloader,
            mock.patch.object(download_bars, "process_ticker", return_value=True),
        ):
            outcomes = download_bars.process_alpaca_batch(
                tasks,
                out_dir="/unused",
                since=download_bars.ANNO,
                skip_existing=False,
                update_existing=True,
                overlap_days=30,
                timeframe="1Day",
            )

        self.assertEqual(outcomes, [(0, True), (1, True)])
        self.assertEqual(downloader.call_count, 3)


if __name__ == "__main__":
    unittest.main()
