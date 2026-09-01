from pathlib import Path
import tempfile
import unittest

from universe import merge_existing_symbols


class ExportAlpacaCompaniesTest(unittest.TestCase):
    def test_current_universe_drops_symbols_not_returned_by_alpaca(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "master.txt"
            output.write_text("DELISTED\nAAPL\n", encoding="utf-8")

            symbols, added = merge_existing_symbols(
                ["AAPL", "MSFT"], output, enabled=False
            )

        self.assertEqual(symbols, ["AAPL", "MSFT"])
        self.assertEqual(added, 2)

    def test_merge_preserves_historical_symbols_and_adds_current_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "master.txt"
            output.write_text("DELISTED\nAAPL\n", encoding="utf-8")

            symbols, added = merge_existing_symbols(
                ["AAPL", "MSFT"], output, enabled=True
            )

        self.assertEqual(symbols, ["AAPL", "DELISTED", "MSFT"])
        self.assertEqual(added, 1)


if __name__ == "__main__":
    unittest.main()
