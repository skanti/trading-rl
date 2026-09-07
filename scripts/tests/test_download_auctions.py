from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from trading_rl.cli import download_auctions


def auction_row(symbol: str, date: str, price: object) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": date,
        "session": "open",
        "condition": "O",
        "price": price,
        "size": 100,
        "timestamp": f"{date}T13:30:00Z",
        "exchange": "Q",
    }


class AuctionUpdateTests(unittest.TestCase):
    def test_default_end_uses_the_new_york_calendar_date(self):
        self.assertEqual(
            download_auctions.default_end(
                datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
            ),
            "2026-09-01",
        )

    def test_current_day_query_is_clamped_behind_delayed_sip(self):
        query_end, current_day = download_auctions.auction_query_end(
            "2026-09-01",
            now=datetime(2026, 9, 1, 14, 12, tzinfo=timezone.utc),
        )

        self.assertTrue(current_day)
        self.assertEqual(query_end, "2026-09-01T13:52:00Z")

    def test_historical_query_end_is_unchanged(self):
        query_end, current_day = download_auctions.auction_query_end(
            "2026-08-31",
            now=datetime(2026, 9, 1, 14, 12, tzinfo=timezone.utc),
        )

        self.assertFalse(current_day)
        self.assertEqual(query_end, "2026-08-31")

    def test_symbols_file_normalizes_legacy_ids_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "symbols.txt"
            path.write_text("AAPL\n\n# benchmark\nST-BRK-B\nmsft\n", encoding="utf-8")

            symbols = download_auctions.symbols_from_file(path)

        self.assertEqual(symbols, ["AAPL", "BRK.B", "MSFT"])

    def test_overlapping_tail_is_replaced_instead_of_blindly_appended(self):
        existing = [
            auction_row("AAPL", "2026-08-24", "100.0"),
            auction_row("AAPL", "2026-08-25", "101.0"),
            auction_row("AAPL", "2026-08-26", "102.0"),
            auction_row("MSFT", "2026-08-26", "500.0"),
        ]
        refreshed = [
            auction_row("AAPL", "2026-08-25", 101.5),
            auction_row("AAPL", "2026-08-27", 103.0),
        ]

        merged = download_auctions.merge_auction_rows(
            existing,
            refreshed,
            refreshed_symbols={"AAPL"},
            refresh_start="2026-08-25",
        )

        self.assertEqual(
            [(row["symbol"], row["date"], row["price"]) for row in merged],
            [
                ("AAPL", "2026-08-24", "100.0"),
                ("AAPL", "2026-08-25", 101.5),
                ("AAPL", "2026-08-27", 103.0),
                ("MSFT", "2026-08-26", "500.0"),
            ],
        )

    def test_update_fetches_only_overlap_but_backfills_new_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "auctions.npz"
            download_auctions._write_dataset(
                output,
                [auction_row("AAPL", "2026-08-27", "100.0")],
                [],
                ["AAPL"],
                "2022-08-27",
                "2026-08-27",
            )

            calls: list[tuple[list[str], str, str]] = []

            def fake_download(_session, _headers, symbols, start, end, _batch_size):
                calls.append((symbols, start, end))
                if symbols == ["AAPL"]:
                    return [
                        auction_row("AAPL", "2026-08-20", 99.0),
                        auction_row("AAPL", "2026-08-28", 101.0),
                    ]
                return [auction_row("MSFT", "2022-08-29", 250.0)]

            with (
                mock.patch.object(
                    download_auctions, "_request_headers", return_value={}
                ),
                mock.patch.object(
                    download_auctions,
                    "_download_auction_rows",
                    side_effect=fake_download,
                ),
                mock.patch.object(
                    download_auctions, "_download_splits", return_value=[]
                ),
            ):
                symbol_count, print_count = download_auctions.update_auctions(
                    output,
                    end="2026-08-28",
                    additional_symbols={"MSFT"},
                    overlap_days=7,
                )

            self.assertEqual(
                calls,
                [
                    (["AAPL"], "2026-08-21", "2026-08-28"),
                    (["MSFT"], "2022-08-27", "2026-08-28"),
                ],
            )
            self.assertEqual((symbol_count, print_count), (2, 2))
            rows = download_auctions._load_raw_auction_rows(output)
            self.assertEqual(
                [(row["symbol"], row["date"]) for row in rows],
                [("AAPL", "2026-08-28"), ("MSFT", "2022-08-29")],
            )
            updated_manifest = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated_manifest["symbols"], ["AAPL", "MSFT"])
            self.assertEqual(
                updated_manifest["last_update"]["refresh_start"], "2026-08-21"
            )

    def test_requested_refresh_retains_other_history_and_catches_up_returning_symbols(self):
        for resume_requested_only in (True, False):
            with self.subTest(resume_requested_only=resume_requested_only), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "auctions.npz"
                download_auctions._write_dataset(
                    output,
                    [auction_row(symbol, "2025-01-10", 100.0) for symbol in ("AAPL", "MSFT", "SPY")],
                    [],
                    ["AAPL", "MSFT", "SPY"],
                    "2022-01-01",
                    "2025-01-10",
                )

                def fake_download(_session, _headers, symbols, start, end, _batch_size):
                    return [auction_row(symbol, end, 110.0) for symbol in symbols]

                with (
                    mock.patch.object(download_auctions, "_request_headers", return_value={}),
                    mock.patch.object(download_auctions, "_download_auction_rows", side_effect=fake_download) as fetch,
                    mock.patch.object(download_auctions, "_download_splits", return_value=[]) as splits,
                ):
                    download_auctions.update_auctions(
                        output, "2025-02-10", {"MSFT", "NVDA"}, refresh_requested_only=True
                    )
                    self.assertEqual(
                        [call.args[2:5] for call in fetch.call_args_list],
                        [
                            (["MSFT", "SPY"], "2025-01-04", "2025-02-10"),
                            (["NVDA"], "2022-01-01", "2025-02-10"),
                        ],
                    )
                    # Retained history still participates in split adjustments.
                    self.assertEqual(splits.call_args.args[2], ["AAPL", "MSFT", "NVDA", "SPY"])
                    rows = download_auctions._load_raw_auction_rows(output)
                    self.assertEqual(
                        [(row["date"], row["price"]) for row in rows if row["symbol"] == "AAPL"],
                        [("2025-01-10", 100.0)],
                    )
                    manifest = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(manifest["symbol_end_dates"]["AAPL"], "2025-01-10")
                    self.assertEqual(manifest["symbol_end_dates"]["MSFT"], "2025-02-10")
                    self.assertEqual(manifest["last_update"]["refreshed_symbols"], ["MSFT", "NVDA", "SPY"])

                    fetch.reset_mock()
                    download_auctions.update_auctions(
                        output, "2025-03-10", {"AAPL"},
                        refresh_requested_only=resume_requested_only,
                    )
                    expected = ["AAPL", "SPY"] if resume_requested_only else ["AAPL", "MSFT", "NVDA", "SPY"]
                    self.assertEqual(fetch.call_args.args[2:5], (expected, "2025-01-04", "2025-03-10"))
                    manifest = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(manifest["symbol_end_dates"]["AAPL"], "2025-03-10")
                    self.assertEqual(
                        manifest["symbol_end_dates"]["MSFT"],
                        "2025-02-10" if resume_requested_only else "2025-03-10",
                    )

    def test_requested_refresh_rejects_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "auctions.npz"
            download_auctions._write_dataset(
                output, [], [], ["SPY"], "2022-01-01", "2025-01-10"
            )
            with self.assertRaisesRegex(ValueError, "explicit symbol selection"):
                download_auctions.update_auctions(
                    output, "2025-02-10", set(), refresh_requested_only=True
                )

    def test_npz_prices_and_sizes_are_pre_adjusted_from_embedded_split_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "auctions.npz"
            rows = [
                auction_row("AAPL", "2026-08-25", 100.0),
                auction_row("AAPL", "2026-08-27", 110.0),
            ]
            splits = [
                {
                    "type": "forward_split",
                    "symbol": "AAPL",
                    "ex_date": "2026-08-26",
                    "old_rate": 1,
                    "new_rate": 2,
                    "id": "split-1",
                }
            ]

            download_auctions._write_dataset(
                output,
                rows,
                splits,
                ["AAPL"],
                "2022-01-01",
                "2026-08-27",
            )

            with download_auctions.np.load(output, allow_pickle=False) as data:
                self.assertTrue(bool(data["split_adjusted"].item()))
                self.assertEqual(data["raw_price"].tolist(), [100.0, 110.0])
                self.assertEqual(data["price"].tolist(), [50.0, 110.0])
                self.assertEqual(data["raw_size"].tolist(), [100.0, 100.0])
                self.assertEqual(data["size"].tolist(), [200.0, 100.0])
                self.assertEqual(data["split_symbol"].tolist(), ["AAPL"])


if __name__ == "__main__":
    unittest.main()
