from contextlib import contextmanager
from datetime import date, time
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from rich.console import Console

from trading_rl.cli import download_auctions, download_bars, download_nbbo
from trading_rl.market_data import download_output
from scripts.tests.test_download_bars import bars, minute_frame
from scripts.tests.test_download_nbbo import quote_row


@contextmanager
def captured_report():
    stream = io.StringIO()
    reports = []
    original = download_output.DownloadReport.render

    def render(report):
        original(report)
        reports.append(report)

    with mock.patch.object(download_output, "CONSOLE", Console(file=stream, width=130, color_system=None)), mock.patch.object(
        download_output.DownloadReport, "render", render
    ):
        yield stream, reports


class DownloadOutputTest(unittest.TestCase):
    def test_warning_groups_follow_table_and_logger_configuration_is_restored(self):
        logger = logging.getLogger("download-output-test")
        before = (logger.handlers[:], logger.level, logger.propagate)
        root_before = logging.getLogger().handlers[:]

        @download_output.download_output("Test", logger)
        def run():
            report = download_output.current_report()
            report.set("Updated", 2, "symbols")
            for symbol in ("AAPL", "MSFT", "NVDA", "AMZN"):
                logger.warning("Missing quote for %s [literal]", symbol)

        with captured_report() as (stream, reports):
            run()
        text = stream.getvalue()
        self.assertLess(text.index("Updated"), text.index("Warning (4 events)"))
        self.assertIn("[literal]", text)
        self.assertIn("1 more events", text)
        self.assertEqual(sum(reports[0].message_counts.values()), 4)
        self.assertEqual((logger.handlers, logger.level, logger.propagate), before)
        self.assertEqual(logging.getLogger().handlers, root_before)
        self.assertIsNone(download_output.current_report())

    def test_fatal_cli_error_prints_failed_summary_without_a_traceback(self):
        @download_output.download_output("NBBO", logging.getLogger("download-output-failure"), exit_on_error=True)
        def run():
            download_output.current_report().set("New quotes", 0, "quotes")
            raise RuntimeError("API unavailable")

        with captured_report() as (stream, reports):
            with self.assertRaises(SystemExit) as failure:
                run()
        self.assertEqual(failure.exception.code, 1)
        self.assertIn("FAILED", stream.getvalue())
        self.assertIn("API unavailable", stream.getvalue())
        self.assertNotIn("Traceback", stream.getvalue())
        self.assertEqual(reports[0].metrics["New quotes"], (0, "quotes"))

    def test_bars_classify_saved_results_and_failures_with_concurrent_workers(self):
        original = bars((60, 10_000, 100, 2), (120, 11_000, 110, 3))
        extended = bars((60, 10_000, 100, 2), (120, 11_000, 110, 3), (180, 12_000, 120, 4))
        revised = original.copy()
        revised[:, 1:5] //= 2
        revised[:, 7] //= 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for symbol in ("UPDATED", "UNCHANGED", "REDOWNLOADED", "FAILED"):
                np.save(root / f"{symbol}.npy", original)
            symbols = root / "symbols.txt"
            symbols.write_text("UPDATED\nUNCHANGED\nREDOWNLOADED\nNEW\nFAILED\n")

            def fetch(symbol, *_args):
                if symbol == "FAILED":
                    raise RuntimeError("API unavailable")
                return minute_frame({"UPDATED": extended, "UNCHANGED": original, "REDOWNLOADED": revised, "NEW": original}[symbol])

            with captured_report() as (stream, reports), mock.patch.object(download_bars, "download_ticker", side_effect=fetch):
                with self.assertRaisesRegex(RuntimeError, "FAILED"):
                    download_bars.main("alpaca", str(symbols), str(root), download_bars.ANNO,
                                       workers_num=3, batch_size=1, update_existing=True)
            report = reports[0]
            for label in ("Updated", "Unchanged", "New downloads", "Full redownloads", "Failed"):
                self.assertEqual(report.metrics[label], (1, "symbols"))
            self.assertEqual(report.metrics["Skipped"], (0, "symbols"))
            self.assertEqual(report.metrics["Requested"], (5, "symbols"))
            np.testing.assert_array_equal(np.load(root / "FAILED.npy"), original)
            self.assertEqual((root / "_failed_tickers.txt").read_text(), "FAILED\n")
            self.assertIn("FAILED", stream.getvalue())

    def test_failed_bar_save_is_never_counted_as_an_update(self):
        original = bars((60, 10_000, 100, 2))
        report = download_output.DownloadReport("Bars")
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "AAPL.npy", original)
            with mock.patch.object(download_bars, "save_array", side_effect=OSError("disk full")):
                self.assertFalse(download_bars.process_ticker(
                    "AAPL", "alpaca", directory, download_bars.ANNO, update_existing=True,
                    initial_df=minute_frame(original), report=report,
                ))
        self.assertEqual(report.outcomes, {"AAPL": "Failed"})

    def test_nbbo_summary_separates_updates_backfills_missing_and_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.npz"
            download_nbbo._write_dataset(output, [quote_row("AAPL", "2026-09-02", 100.)], [], ["AAPL"],
                                        "2026-01-02", "2026-09-02", time(15, 45), 60, 1, 1)
            targets = {date(2026, 1, 2): {"AAPL"}, date(2026, 9, 2): {"AAPL", "MSFT", "MISSING"}, date(2026, 9, 3): {"AAPL"}}
            refreshed = [quote_row("AAPL", "2026-01-02", 90.), quote_row("AAPL", "2026-09-02", 101.), quote_row("MSFT", "2026-09-02", 200.)]

            @download_output.download_output("NBBO", download_nbbo.LOGGER)
            def run():
                return download_nbbo.update_nbbo(output, "2026-09-03", targets, 7, 100, 60, 180)

            with captured_report() as (_, reports), mock.patch.object(download_nbbo, "_fetch", return_value=(
                refreshed, 1, [{"symbol": "MISSING", "date": "2026-09-02"}], {"2026-01-02", "2026-09-02"}
            )), mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(download_nbbo, "_download_splits", return_value=[]):
                run()
            report = reports[0]
            for label in ("Updated quotes", "New quotes", "Historical backfills"):
                self.assertEqual(report.metrics[label], (1, "quotes"))
            self.assertEqual(report.metrics["Missing this run"], (1, "symbol/date pairs"))
            self.assertEqual(report.metrics["Targets deferred"], (1, "symbol/date pairs"))

    def test_nbbo_rebuild_counts_quotes_and_not_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.npz"

            @download_output.download_output("NBBO", download_nbbo.LOGGER)
            def run():
                download_output.current_report().mode = "Rebuild"
                download_nbbo.download_nbbo({date(2026, 9, 2): {"AAPL", "MSFT"}}, "2026-09-02", "2026-09-02", output, time(15, 45))

            with captured_report() as (_, reports), mock.patch.object(download_nbbo, "_fetch", return_value=(
                [quote_row("AAPL", "2026-09-02", 100.), quote_row("MSFT", "2026-09-02", 200.)], 0, [], {"2026-09-02"}
            )), mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(download_nbbo, "_download_splits", return_value=[]):
                run()
            self.assertEqual(reports[0].metrics["Full redownloads"], (2, "quotes"))

    def test_auction_summary_distinguishes_missing_symbols_from_new_downloads(self):
        row = {"symbol": "AAPL", "date": "2026-09-02", "session": 0, "condition": "O", "price": 100., "size": 100., "timestamp": "2026-09-02T13:30:00Z", "exchange": "Q"}
        with tempfile.TemporaryDirectory() as directory:
            @download_output.download_output("Auctions", download_auctions.LOGGER)
            def run():
                download_auctions.download_auctions(["AAPL", "MISSING"], "2026-09-02", "2026-09-02", Path(directory) / "auctions.npz")

            with captured_report() as (stream, reports), mock.patch.object(download_auctions, "_download_auction_rows", return_value=[row]), mock.patch.object(
                download_auctions, "_request_headers", return_value={}
            ), mock.patch.object(download_auctions, "_download_splits", return_value=[]):
                run()
            self.assertEqual(reports[0].metrics["New downloads"], (1, "symbols"))
            self.assertEqual(reports[0].metrics["Without auction prints"], (1, "symbols"))
            self.assertLess(stream.getvalue().index("Prints in dataset"), stream.getvalue().index("Warning (1 event)"))


if __name__ == "__main__":
    unittest.main()
