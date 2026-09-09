"""CLI mode selection must never silently overwrite existing history."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from trading_rl.cli import download_auctions, download_nbbo
from trading_rl.market_data.download_mode import use_incremental_download


class DownloadModeTest(unittest.TestCase):
    def test_modes_and_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data.npz"
            manifest = output.with_suffix(".json")
            self.assertFalse(use_incremental_download(output))
            with self.assertRaisesRegex(ValueError, "requires"):
                use_incremental_download(output, update=True)
            for partial in (output, manifest):
                partial.touch()
                for rebuild in (False, True):
                    with self.assertRaisesRegex(ValueError, "incomplete dataset"):
                        use_incremental_download(output, rebuild=rebuild)
                partial.unlink()
            output.touch()
            manifest.touch()
            self.assertTrue(use_incremental_download(output))
            self.assertTrue(use_incremental_download(output, update=True))
            self.assertFalse(use_incremental_download(output, rebuild=True))

    def test_both_clis_create_update_and_explicitly_rebuild(self):
        for module in (download_auctions, download_nbbo):
            for existing, rebuild in ((False, False), (True, False), (True, True)):
                with self.subTest(module=module.__name__, existing=existing, rebuild=rebuild), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "data.npz"
                    symbols = Path(directory) / "symbols.txt"
                    symbols.write_text("AAPL\n")
                    manifest = {"start": "2026-01-02", "end": "2026-09-02", "target_time": "15:45", "missing_quote_count": 0}
                    if existing:
                        output.touch()
                        output.with_suffix(".json").write_text(json.dumps(manifest))
                    argv = ["download", "--output", str(output), "--symbols-file", str(symbols), "--end", "2026-09-02"]
                    if not existing:
                        argv += ["--start", "2026-01-02"]
                    if rebuild:
                        argv += ["--rebuild"]
                    is_nbbo = module is download_nbbo
                    initial_name = "download_nbbo" if is_nbbo else "download_auctions"
                    update_name = "update_nbbo" if is_nbbo else "update_auctions"
                    def result(*args, **kwargs):
                        output.with_suffix(".json").write_text(json.dumps(manifest))
                        return (1, 1, 0) if is_nbbo else (1, 1)
                    with mock.patch("sys.argv", argv), mock.patch.object(module, initial_name, side_effect=result) as initial, mock.patch.object(
                        module, update_name, side_effect=result
                    ) as update, mock.patch.object(download_nbbo, "targets_from_symbols", return_value={date(2026, 1, 2): {"AAPL", "SPY"}}):
                        module.main()
                    self.assertEqual(update.call_count, int(existing and not rebuild))
                    self.assertEqual(initial.call_count, int(not existing or rebuild))


if __name__ == "__main__":
    unittest.main()
