from __future__ import annotations

from datetime import date, datetime, time, timezone
import unittest

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

    def test_overlap_replaces_only_refreshed_symbol_tail(self):
        existing = [
            quote_row("AAPL", "2026-08-24", 100.0),
            quote_row("AAPL", "2026-08-25", 101.0),
            quote_row("MSFT", "2026-08-25", 500.0),
        ]
        refreshed = [quote_row("AAPL", "2026-08-25", 101.5)]

        merged = download_nbbo.merge_quote_rows(
            existing, refreshed, {"AAPL"}, "2026-08-25"
        )

        self.assertEqual(
            [(row["symbol"], row["date"], row["ask_price"]) for row in merged],
            [
                ("AAPL", "2026-08-24", 100.0),
                ("AAPL", "2026-08-25", 101.5),
                ("MSFT", "2026-08-25", 500.0),
            ],
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
