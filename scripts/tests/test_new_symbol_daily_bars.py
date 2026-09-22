"""Only verified new symbols may defer an empty first daily-history download."""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_rl.cli import download_bars


def daily_frame():
    return pd.DataFrame([{
        "t": "2026-09-22T04:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10,
        "v": 1000, "n": 100, "vw": 10,
    }])


ACTION = {"old_symbol": "NHIC", "new_symbol": "NWCL", "process_date": "2026-09-22"}
CUTOFF = datetime(2026, 9, 22, 3, 59, 59, 999999, tzinfo=UTC)


class NewSymbolDailyBarsTest(unittest.TestCase):
    def test_corporate_action_requires_exact_new_symbol_and_excluded_date(self):
        for change, expected in (({}, True), ({"new_symbol": "OTHER"}, False),
                                 ({"old_symbol": "NWCL"}, False),
                                 ({"old_symbol": ""}, False),
                                 ({"process_date": "2026-09-21"}, False),
                                 ({"process_date": "2026-09-23"}, False)):
            payload = {"corporate_actions": {"name_changes": [dict(ACTION, **change)]}}
            with self.subTest(change=change), patch.dict(download_bars.os.environ, {
                "ALPACA_DATA_KEY": "test", "ALPACA_DATA_SECRET": "test",
            }), patch.object(download_bars, "completed_daily_bar_end", return_value=CUTOFF), patch.object(
                download_bars, "request_json", return_value=payload,
            ) as request:
                result = download_bars.confirm_new_daily_symbol("NWCL")
            self.assertEqual(result is not None, expected)
            self.assertEqual(request.call_args.args[2]["start"], "2026-09-22")
            self.assertEqual(request.call_args.args[2]["end"], "2026-09-22")

    def test_confirmation_handles_pagination_and_rejects_repeated_tokens(self):
        empty = {"corporate_actions": {}, "next_page_token": "page2"}
        with patch.dict(download_bars.os.environ, {
            "ALPACA_DATA_KEY": "test", "ALPACA_DATA_SECRET": "test",
        }), patch.object(download_bars, "completed_daily_bar_end", return_value=CUTOFF):
            with patch.object(download_bars, "request_json", side_effect=[
                empty, {"corporate_actions": {"name_changes": [ACTION]}},
            ]) as request:
                self.assertIsNotNone(download_bars.confirm_new_daily_symbol("NWCL"))
                self.assertEqual(request.call_count, 2)
            with patch.object(download_bars, "request_json", return_value=empty), self.assertRaisesRegex(ValueError, "repeated"):
                download_bars.confirm_new_daily_symbol("NWCL")

    def test_daily_skip_completes_single_and_batch_runs_without_creating_a_bar_file(self):
        frames = {"AAPL": daily_frame(), "NWCL": pd.DataFrame()}
        for batch_size in (1, 100):
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tickers = root / "symbols.txt"
                tickers.write_text("NWCL\nAAPL\n")
                output = root / "bars"
                output.mkdir()
                (output / "_failed_tickers.txt").write_text("NWCL\n")
                with patch.object(download_bars, "download_bars_alpaca_batch", return_value=frames), patch.object(
                    download_bars, "confirm_new_daily_symbol", return_value=ACTION,
                ):
                    download_bars.main("alpaca", str(tickers), str(output), download_bars.ANNO,
                                       timeframe="1Day", update_existing=True, workers_num=2, batch_size=batch_size)
                manifest = json.loads((output / download_bars.DATASET_MANIFEST).read_text())
                self.assertEqual((manifest["success_count"], manifest["deferred_count"], manifest["failed_count"]), (1, 1, 0))
                self.assertEqual(manifest["deferred_listings"], {"NWCL": ACTION})
                self.assertEqual((output / "_failed_tickers.txt").read_text(), "")
                self.assertFalse((output / "NWCL.npy").exists())
                self.assertTrue((output / "AAPL.npy").exists())
                # A subsequent completed daily bar is saved normally, with no exception lookup.
                with patch.object(download_bars, "confirm_new_daily_symbol") as confirm:
                    self.assertTrue(download_bars.process_ticker(
                        "NWCL", "alpaca", str(output), download_bars.ANNO,
                        timeframe="1Day", update_existing=True, initial_df=daily_frame(),
                    ))
                    confirm.assert_not_called()
                self.assertTrue((output / "NWCL.npy").exists())

    def test_exception_expires_when_the_symbol_has_a_completed_date(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(download_bars.os.environ, {
            "ALPACA_DATA_KEY": "test", "ALPACA_DATA_SECRET": "test",
        }), patch.object(download_bars, "completed_daily_bar_end", return_value=CUTOFF + download_bars.timedelta(days=1)), patch.object(
            download_bars, "request_json", return_value={"corporate_actions": {"name_changes": [ACTION]}},
        ):
            self.assertFalse(download_bars.process_ticker(
                "NWCL", "alpaca", directory, download_bars.ANNO, timeframe="1Day", initial_df=pd.DataFrame(),
            ))

    def test_unconfirmed_empty_history_and_failed_lookups_still_fail(self):
        for result, error in ((None, None), (None, RuntimeError("HTTP 403"))):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                download_bars, "confirm_new_daily_symbol", return_value=result, side_effect=error,
            ):
                self.assertFalse(download_bars.process_ticker(
                    "UNKNOWN", "alpaca", directory, download_bars.ANNO, timeframe="1Day", initial_df=pd.DataFrame(),
                ))

    def test_existing_files_minutes_and_other_sources_never_use_the_exception(self):
        for source, timeframe, existing, update in (
            ("alpaca", "1Day", True, True), ("alpaca", "1Day", True, False),
            ("alpaca", "1Min", False, False), ("polygon", "1Day", False, False),
        ):
            with self.subTest(source=source, timeframe=timeframe, existing=existing), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "NWCL.npy"
                if existing:
                    np.save(path, download_bars.dataframe_to_array(daily_frame(), "NWCL", "1Day"))
                    original = path.read_bytes()
                with patch.object(download_bars, "confirm_new_daily_symbol") as confirm:
                    self.assertFalse(download_bars.process_ticker(
                        "NWCL", source, directory, download_bars.ANNO, timeframe=timeframe,
                        update_existing=update, initial_df=pd.DataFrame(),
                    ))
                    confirm.assert_not_called()
                if existing:
                    self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
