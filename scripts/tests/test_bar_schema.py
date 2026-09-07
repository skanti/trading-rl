import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from trading_rl.cli import download_bars
from trading_rl.market_data.bars import encode_alpaca_bars
from trading_rl.market_data.schema import BAR_COLUMNS, BAR_SCHEMA_VERSION
from trading_rl.overnight.backtest import _dataset_manifest, _manifest_fingerprint


def sample_bar():
    return {
        "t": "2026-08-27T19:59:00Z",
        "o": 100.125,
        "h": 105.5,
        "l": 98.25,
        "c": 104.75,
        "v": 1234,
        "n": 42,
        "vw": 102.375,
    }


class BarSchemaTests(unittest.TestCase):
    def test_polygon_adapter_preserves_ohlc_and_vwap(self):
        bar = sample_bar()
        bar["t"] = int(pd.Timestamp(bar["t"]).timestamp() * 1000)
        response = mock.Mock()
        response.json.return_value = {"resultsCount": 1, "results": [bar]}
        with (
            mock.patch.dict(download_bars.os.environ, {"POLYGON_KEY": "test"}),
            mock.patch.object(download_bars.requests, "get", return_value=response),
        ):
            frame = download_bars.download_bars_polygon("AAPL", download_bars.ANNO)
        encoded = download_bars.dataframe_to_array(frame, "AAPL", "1Min")
        np.testing.assert_array_equal(
            encoded, encode_alpaca_bars([sample_bar()], "AAPL", "1Min")
        )

    def test_minute_and_daily_encoding_share_columns_and_preserve_all_fields(self):
        for timeframe, dtype in (("1Min", np.int32), ("1Day", np.int64)):
            with self.subTest(timeframe=timeframe):
                encoded = encode_alpaca_bars([sample_bar()], "AAPL", timeframe)
                self.assertEqual(encoded.shape, (1, 8))
                self.assertEqual(encoded.dtype, dtype)
                self.assertEqual(
                    encoded[0, 1:].tolist(),
                    [100125, 105500, 98250, 104750, 1234, 42, 102375],
                )
        self.assertEqual(BAR_COLUMNS["1Min"], BAR_COLUMNS["1Day"])

    def test_minute_fields_are_not_silently_invented_or_overflowed(self):
        for field in ("h", "l", "c", "vw"):
            with self.subTest(field=field):
                missing = sample_bar()
                del missing[field]
                with self.assertRaisesRegex(ValueError, "missing OHLCV"):
                    encode_alpaca_bars([missing], "AAPL", "1Min")
                overflow = {**sample_bar(), field: 2_500_000.0}
                with self.assertRaisesRegex(ValueError, "int32"):
                    encode_alpaca_bars([overflow], "AAPL", "1Min")
                # Daily bars still support the larger split-adjusted range.
                self.assertEqual(
                    encode_alpaca_bars([overflow], "AAPL", "1Day").dtype, np.int64
                )
        encoded = encode_alpaca_bars([{**sample_bar(), "vw": None}], "AAPL", "1Min")
        self.assertEqual(encoded[0, 7], 0)

    def test_download_writes_versioned_data_and_backtest_rejects_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tickers = root / "tickers.txt"
            tickers.write_text("AAPL\n")
            output = root / "bars"
            with mock.patch.object(
                download_bars,
                "download_ticker",
                return_value=pd.DataFrame([sample_bar()]),
            ):
                download_bars.main(
                    "alpaca",
                    str(tickers),
                    str(output),
                    download_bars.ANNO,
                    workers_num=0,
                    batch_size=1,
                )
            stored = np.load(output / "AAPL.npy", allow_pickle=False)
            self.assertEqual(stored.shape, (1, 8))
            self.assertEqual(stored.dtype, np.int32)
            np.testing.assert_array_equal(
                stored, encode_alpaca_bars([sample_bar()], "AAPL", "1Min")
            )
            manifest = _dataset_manifest(output, "1Min")
            self.assertEqual(manifest["schema_version"], BAR_SCHEMA_VERSION)
            before = _manifest_fingerprint(output, "1Min")
            manifest_path = output / "_download_manifest.json"
            updated = {**manifest, "updated_at": "different-generation"}
            manifest_path.write_text(json.dumps(updated))
            self.assertNotEqual(before, _manifest_fingerprint(output, "1Min"))

            swapped = list(manifest["columns"])
            swapped[2], swapped[5] = swapped[5], swapped[2]
            for changes in (
                {"schema_version": 1},
                {"schema_version": None},
                {"columns": swapped},
                {"timeframe": "1Day"},
                {"adjustment": "raw"},
            ):
                with self.subTest(changes=changes):
                    manifest_path.write_text(json.dumps({**manifest, **changes}))
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        _dataset_manifest(output, "1Min")

    def test_mixed_store_fails_before_requests_or_overwriting_any_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "AAPL.npy"
            legacy = root / "OLD.npy"
            np.save(good, encode_alpaca_bars([sample_bar()], "AAPL", "1Min"))
            np.save(legacy, np.array([[100, 100000, 1000, 10]], dtype=np.int32))
            original = {path: path.read_bytes() for path in (good, legacy)}
            tickers = root / "tickers.txt"
            tickers.write_text("AAPL\n")
            with mock.patch.object(download_bars, "download_ticker") as request:
                with self.assertRaisesRegex(ValueError, "Re-download legacy"):
                    download_bars.main(
                        "alpaca",
                        str(tickers),
                        directory,
                        download_bars.ANNO,
                        batch_size=1,
                    )
            request.assert_not_called()
            for path, contents in original.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertFalse((root / "_download_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
