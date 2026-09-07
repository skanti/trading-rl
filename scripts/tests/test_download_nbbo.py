from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from trading_rl.cli import download_nbbo


def quote_row(symbol: str, day: str, ask: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": day,
        "target_timestamp": f"{day}T19:45:00Z",
        "timestamp": f"{day}T19:44:59Z",
        "bid_price": ask - 0.02,
        "ask_price": ask,
        "bid_size": 100.0,
        "ask_size": 200.0,
        "bid_exchange": "Q",
        "ask_exchange": "Q",
    }


class DownloadNbboTest(unittest.TestCase):
    def test_trade_csv_preserves_rotating_daily_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            path.write_text(
                "entry_date,sample_id\n"
                "2026-09-01,AAPL\n"
                "2026-09-01,MSFT\n"
                "2026-09-02,MSFT\n"
                "2026-09-02,NVDA\n"
            )

            targets = download_nbbo.targets_from_trade_csv(path)

        self.assertEqual(targets[date(2026, 9, 1)], {"AAPL", "MSFT"})
        self.assertEqual(targets[date(2026, 9, 2)], {"MSFT", "NVDA"})
        self.assertEqual(set().union(*targets.values()), {"AAPL", "MSFT", "NVDA"})

    def test_downloader_queries_only_each_days_members(self):
        targets = {
            date(2026, 9, 1): {"AAPL", "MSFT"},
            date(2026, 9, 2): {"MSFT", "NVDA"},
        }

        def response(_session, _headers, _limiter, symbols, _start, target, _limit):
            stamp = target.isoformat().replace("+00:00", "Z")
            return {
                "quotes": {
                    symbol: [
                        {"t": stamp, "bp": 99.9, "ap": 100.0, "bs": 1, "as": 2}
                    ]
                    for symbol in symbols
                }
            }

        with mock.patch.object(
            download_nbbo, "_quote_request", side_effect=response
        ) as request:
            rows = download_nbbo._download_quote_rows(
                mock.Mock(),
                {},
                targets,
                time(15, 45),
                100,
                60,
                download_nbbo.RateLimiter(10_000),
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[3], ["AAPL", "MSFT"])
        self.assertEqual(request.call_args_list[1].args[3], ["MSFT", "NVDA"])

    def test_selects_latest_quote_at_or_before_target(self):
        day = date(2026, 9, 2)
        target = datetime(2026, 9, 2, 19, 45, tzinfo=timezone.utc)
        payload = {
            "quotes": {
                "AAPL": [
                    {"t": "2026-09-02T19:44:59Z", "bp": 99.9, "ap": 100.0, "bs": 1, "as": 2},
                    {"t": "2026-09-02T19:45:00Z", "bp": 100.0, "ap": 100.1, "bs": 3, "as": 4},
                    {"t": "2026-09-02T19:45:00.001Z", "bp": 100.1, "ap": 100.2, "bs": 5, "as": 6},
                ]
            }
        }

        selected = download_nbbo.select_causal_quotes(payload, day, target)

        self.assertEqual(selected["AAPL"]["ask_price"], 100.1)
        self.assertEqual(selected["AAPL"]["timestamp"], "2026-09-02T19:45:00Z")

    def test_split_adjusts_prices_and_sizes(self):
        arrays = download_nbbo._dataset_arrays(
            [quote_row("NVDA", "2024-06-07", 1_200.0)],
            [
                {
                    "type": "forward_split",
                    "symbol": "NVDA",
                    "ex_date": "2024-06-10",
                    "old_rate": "1",
                    "new_rate": "10",
                    "id": "split-1",
                }
            ],
        )

        self.assertEqual(float(arrays["ask_price"][0]), 120.0)
        self.assertEqual(float(arrays["ask_size"][0]), 2_000.0)
        self.assertEqual(float(arrays["raw_ask_price"][0]), 1_200.0)

    def test_overlap_replaces_the_complete_refreshed_date_membership(self):
        existing = [
            quote_row("AAPL", "2026-08-24", 100.0),
            quote_row("AAPL", "2026-08-25", 101.0),
            quote_row("MSFT", "2026-08-25", 500.0),
        ]
        refreshed = [quote_row("AAPL", "2026-08-25", 101.5)]

        merged = download_nbbo.merge_quote_rows(existing, refreshed, {"2026-08-25"})

        self.assertEqual(
            [(row["symbol"], row["date"], row["ask_price"]) for row in merged],
            [
                ("AAPL", "2026-08-24", 100.0),
                ("AAPL", "2026-08-25", 101.5),
            ],
        )

    def test_missing_or_overly_stale_target_fails_the_download(self):
        target = datetime(2026, 9, 2, 19, 45, tzinfo=timezone.utc)
        stale = {
            "quotes": {
                "AAPL": [
                    {
                        "t": "2026-09-02T19:43:00Z",
                        "bp": 99.9,
                        "ap": 100.0,
                        "bs": 1,
                        "as": 2,
                    }
                ]
            }
        }
        with mock.patch.object(download_nbbo, "_quote_request", return_value=stale):
            with self.assertRaisesRegex(
                ValueError, r"within 60s.*2026-09-02:AAPL"
            ):
                download_nbbo._download_quote_rows(
                    mock.Mock(),
                    {},
                    {date(2026, 9, 2): {"AAPL"}},
                    target.astimezone(download_nbbo.EASTERN).time().replace(tzinfo=None),
                    100,
                    60,
                    download_nbbo.RateLimiter(10_000),
                )

    def test_current_snapshot_is_deferred_until_sip_delay_passes(self):
        sessions, deferred = download_nbbo.eligible_sessions(
            [date(2026, 9, 4)],
            time(15, 45),
            now=datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc),
        )

        self.assertEqual(sessions, [])
        self.assertEqual(deferred, 1)


if __name__ == "__main__":
    unittest.main()
