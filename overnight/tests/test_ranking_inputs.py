import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_rl.overnight import ranking_inputs


class RankingInputsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_bars(self, days, symbol="AAPL"):
        timestamps = (
            pd.DatetimeIndex(days).tz_localize("America/New_York").tz_convert("UTC")
        )
        seconds = (timestamps - pd.Timestamp("2010-01-01", tz="UTC")).total_seconds()
        np.save(
            self.root / f"{symbol}.npy",
            np.asarray(
                [[second, 1000, 1000, 1000, 1000, 100, 10, 1000] for second in seconds],
                dtype=np.int64,
            ),
        )

    @staticmethod
    def sessions(days):
        return [{"date": day, "open": "09:30", "close": "16:00"} for day in days]

    def test_calendar_retains_missing_bar_sessions_and_skips_holidays(self):
        self.write_bars(["2026-07-02", "2026-07-06", "2026-07-08"])
        # July 3 and the weekend are not sessions; July 7 has missing bars.
        calendar = self.sessions(
            ["2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08"]
        )
        with patch.object(
            ranking_inputs, "fetch_ranking_calendar", return_value=calendar
        ):
            dates, _ = ranking_inputs.daily_ranking_calendar(
                self.root,
                ["AAPL"],
                now=datetime.fromisoformat("2026-07-09T12:00:00+00:00"),
            )
        self.assertEqual(
            dates.strftime("%Y-%m-%d").tolist(), [row["date"] for row in calendar]
        )

    def test_unfinished_and_future_bars_are_excluded_using_new_york_date(self):
        self.write_bars(["2026-03-05", "2026-03-06", "2026-03-09", "2026-03-10"])
        calendar = self.sessions(["2026-03-05", "2026-03-06"])
        # After the DST switch: midnight UTC is still March 9 in New York.
        with patch.object(
            ranking_inputs, "fetch_ranking_calendar", return_value=calendar
        ) as fetch:
            dates, _ = ranking_inputs.daily_ranking_calendar(
                self.root,
                ["AAPL"],
                now=datetime.fromisoformat("2026-03-10T00:30:00+00:00"),
            )
        fetch.assert_called_once_with(date(2026, 3, 5), date(2026, 3, 6))
        self.assertEqual(dates[-1], pd.Timestamp("2026-03-06"))

    def test_official_early_close_is_used(self):
        closes = ranking_inputs.session_closes(
            [{"date": "2026-11-27", "open": "09:30", "close": "13:00"}],
            date(2026, 11, 27),
            date(2026, 11, 27),
        )
        self.assertEqual(closes, {date(2026, 11, 27): 13 * 60})

    def test_invalid_calendars_fail_instead_of_guessing_session_times(self):
        session = self.sessions(["2026-07-02"])[0]
        for sessions in (
            [],
            [session, session],
            [{"date": "2026-07-02", "open": "09:30"}],
            [{**session, "close": "09:00"}],
            [{**session, "close": "25:00"}],
            [None],
        ):
            with self.subTest(sessions=sessions), self.assertRaises(ValueError):
                ranking_inputs.session_closes(
                    sessions, date(2026, 7, 2), date(2026, 7, 2)
                )

    def test_malformed_offline_calendar_has_actionable_error(self):
        self.write_bars(["2026-07-02", "2026-07-06"])
        calendar = self.root / "calendar.json"
        calendar.write_text(json.dumps(self.sessions(["2026-07-02", "2026-07-06"])))
        with self.assertRaisesRegex(
            ValueError, "calendar JSON requires start, end, and sessions"
        ):
            ranking_inputs.daily_ranking_calendar(
                self.root, ["AAPL"], calendar_path=calendar
            )

    def test_duplicate_daily_dates_cannot_inflate_history(self):
        self.write_bars(["2026-07-02", "2026-07-02", "2026-07-06"])
        with self.assertRaisesRegex(
            ValueError, "timestamps must be strictly increasing"
        ):
            ranking_inputs.daily_ranking_calendar(self.root, ["AAPL"])

    def test_shortlist_cannot_silently_drop_missing_daily_files(self):
        self.write_bars(["2026-07-02", "2026-07-06"])
        shortlist = self.root / "symbols.txt"
        shortlist.write_text("AAPL\nMSFT\n")
        with self.assertRaisesRegex(
            ValueError, "shortlist symbols have no daily bars: MSFT"
        ):
            ranking_inputs.daily_ranking_symbols(self.root, shortlist)
        self.assertEqual(
            ranking_inputs.daily_ranking_symbols(self.root).tolist(), ["AAPL"]
        )

    def test_fetch_reuses_authenticated_client_and_only_requests_calendar(self):
        with (
            patch(
                "trading_rl.overnight.live.load_credentials",
                return_value=("test-key", "test-secret"),
            ),
            patch("trading_rl.overnight.live.AlpacaClient") as client,
            patch.dict(
                os.environ, {"ALPACA_URL": "https://paper-api.alpaca.markets/v2"}
            ),
        ):
            client.return_value.calendar.return_value = self.sessions(["2026-07-02"])
            result = ranking_inputs.fetch_ranking_calendar(
                date(2026, 7, 1), date(2026, 7, 6)
            )
        client.assert_called_once_with(
            "test-key", "test-secret", trading_url="https://paper-api.alpaca.markets/v2"
        )
        self.assertEqual(len(client.return_value.method_calls), 1)
        client.return_value.calendar.assert_called_once_with(
            date(2026, 7, 1), date(2026, 7, 6)
        )
        self.assertEqual(result, self.sessions(["2026-07-02"]))


if __name__ == "__main__":
    unittest.main()
