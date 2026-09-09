from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np


from trading_rl.cli import download_bars
from scripts.tests.bar_fixtures import ohlcv_fixture


def bars(*rows: tuple[int, int, int, int]) -> np.ndarray:
    return ohlcv_fixture(np.asarray(rows, dtype=np.int32))


def minute_frame(array: np.ndarray):
    return download_bars.pd.DataFrame(
        {
            "t": [download_bars.ANNO + download_bars.timedelta(seconds=int(t)) for t in array[:, 0]],
            "o": array[:, 1] / 1000.0,
            "h": array[:, 2] / 1000.0,
            "l": array[:, 3] / 1000.0,
            "c": array[:, 4] / 1000.0,
            "v": array[:, 5],
            "n": array[:, 6],
            "vw": array[:, 7] / 1000.0,
        }
    )


class BarUpdateTests(unittest.TestCase):
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
                    "h": 2_500_001.0,
                    "l": 2_499_999.0,
                    "c": 2_500_000.0,
                    "vw": 2_500_000.0,
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

    def test_matching_anchor_replaces_corrected_tail_even_without_new_bars(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3))
        for extend in (False, True):
            update = bars((2, 101, 11, 2), (3, 150, 25, 5))
            if extend:
                update = np.vstack((update, bars((4, 160, 26, 6))))
            merged = download_bars.merge_bar_arrays(base, update, anchor_seconds=2)
            np.testing.assert_array_equal(merged, np.vstack((base[:1], update)))

    def test_any_changed_anchor_field_requires_full_refresh(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3))
        for column in range(1, 8):
            with self.subTest(column=column):
                update = base[1:].copy()
                update[0, column] += 1
                self.assertIsNone(
                    download_bars.merge_bar_arrays(base, update, anchor_seconds=2)
                )

    def test_expected_anchor_and_complete_stored_overlap_are_required(self):
        base = bars((1, 100, 10, 1), (2, 101, 11, 2), (3, 102, 12, 3), (4, 103, 13, 4))
        for update, message in (
            (base[2:], "expected anchor"),
            (bars((5, 104, 14, 5)), "expected anchor"),
            (base[1:3], "stored endpoint"),
            (base[[1, 3]], "missing previously stored"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                download_bars.merge_bar_arrays(base, update, anchor_seconds=2)

    def test_minute_batches_validate_each_symbols_planned_anchor(self):
        day = 86_400
        aapl = bars((0, 100, 10, 1), (20 * day, 101, 11, 2), (40 * day, 102, 12, 3))
        msft = bars((0, 200, 10, 1), (30 * day, 201, 11, 2), (40 * day, 202, 12, 3))
        updates = {
            "AAPL": bars((20 * day, 101, 11, 2), (40 * day, 110, 20, 4), (41 * day, 111, 21, 5)),
            "MSFT": bars((20 * day, 199, 10, 1), (30 * day, 201, 11, 2), (40 * day, 210, 20, 4), (41 * day, 211, 21, 5)),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.save(output / "AAPL.npy", aapl)
            np.save(output / "MSFT.npy", msft)
            tickers_path = output / "tickers.txt"
            tickers_path.write_text("AAPL\nMSFT\n")
            with mock.patch.object(
                download_bars, "download_bars_alpaca_batch",
                return_value={symbol: minute_frame(array) for symbol, array in updates.items()},
            ) as downloader:
                download_bars.main(
                    "alpaca", str(tickers_path), directory, download_bars.ANNO,
                    workers_num=0, update_existing=True, batch_size=2,
                )
            downloader.assert_called_once_with(
                ["AAPL", "MSFT"], download_bars.ANNO + download_bars.timedelta(days=20), "1Min"
            )
            np.testing.assert_array_equal(np.load(output / "AAPL.npy"), np.vstack((aapl[:1], updates["AAPL"])))
            np.testing.assert_array_equal(np.load(output / "MSFT.npy"), np.vstack((msft[:1], updates["MSFT"][1:])))

    def test_invalid_minute_replacements_fail_the_run_and_preserve_the_file(self):
        day = 86_400
        base = bars((0, 100, 10, 1), (20 * day, 101, 11, 2), (30 * day, 102, 12, 3), (40 * day, 103, 13, 4))
        changed = base[1:].copy()
        changed[:, 1] += 10
        responses = {
            "missing anchor": [base[2:]],
            "missing interior timestamp": [base[[1, 3]]],
            "truncated tail": [base[1:3]],
            "full refresh drops prefix": [changed, changed],
            "full refresh drops interior": [changed, base[[0, 1, 3]]],
        }
        for case, arrays in responses.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                stored = output / "AAPL.npy"
                np.save(stored, base)
                original = stored.read_bytes()
                tickers_path = output / "tickers.txt"
                tickers_path.write_text("AAPL\n")
                with mock.patch.object(
                    download_bars, "download_ticker",
                    side_effect=[minute_frame(array) for array in arrays],
                ) as downloader, self.assertRaisesRegex(RuntimeError, "AAPL"):
                    download_bars.main(
                        "alpaca", str(tickers_path), directory, download_bars.ANNO,
                        workers_num=1, update_existing=True, batch_size=1,
                    )
                self.assertEqual(downloader.call_count, len(arrays))
                self.assertEqual(stored.read_bytes(), original)
                self.assertEqual((output / "_failed_tickers.txt").read_text(), "AAPL\n")

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

    def test_minute_request_preserves_the_planned_anchor_timestamp(self):
        anchor = download_bars.ANNO + download_bars.timedelta(hours=14, minutes=31)
        with (
            mock.patch.dict(
                download_bars.os.environ,
                {"ALPACA_DATA_KEY": "key", "ALPACA_DATA_SECRET": "secret"},
            ),
            mock.patch.object(download_bars, "request_json", return_value={"bars": {}}) as requester,
        ):
            download_bars.download_bars_alpaca_batch(["AAPL"], anchor, "1Min")
        self.assertEqual(requester.call_args.args[2]["start"], anchor.isoformat())

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

    def test_progress_reports_finished_work_before_a_slow_first_download(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as directory:
                progress_seen = threading.Event()
                slow_finished = threading.Event()
                updates = []

                def process(ticker, **_kwargs):
                    if ticker == "SLOW":
                        progress_seen.wait(timeout=5)
                        slow_finished.set()
                        return False
                    return True

                def process_batch(tasks, **kwargs):
                    return [(index, process(ticker, **kwargs)) for index, ticker, _since in tasks]

                def advance(count):
                    updates.append((count, slow_finished.is_set()))
                    progress_seen.set()

                tickers = ["SLOW", "FAST"] if batch_size == 1 else ["SLOW", "NEXT", "FAST", "LAST"]
                tickers_path = Path(directory) / "tickers.txt"
                tickers_path.write_text("\n".join(tickers) + "\n")
                output = Path(directory) / "bars"
                with (
                    mock.patch.object(download_bars, "process_ticker", side_effect=process),
                    mock.patch.object(download_bars, "process_alpaca_batch", side_effect=process_batch),
                    mock.patch.object(download_bars, "download_progress") as progress,
                    self.assertRaisesRegex(RuntimeError, "SLOW"),
                ):
                    progress.return_value.__enter__.return_value.update.side_effect = advance
                    download_bars.main(
                        source="alpaca",
                        tickers_path=str(tickers_path),
                        out_dir=str(output),
                        since=download_bars.ANNO,
                        workers_num=2,
                        batch_size=batch_size,
                    )

                self.assertEqual(updates[0], (batch_size, False))
                self.assertEqual(sum(count for count, _ in updates), len(tickers))
                self.assertEqual((output / "_failed_tickers.txt").read_text(), "SLOW\n")

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
