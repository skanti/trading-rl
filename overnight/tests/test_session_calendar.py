"""Calendar coverage, causal exit cutoffs, and final morning-session replay."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_rl.market_data import session_calendar
from trading_rl.overnight.backtest import build_parser
from trading_rl.overnight.backtest_calendar import ORIGIN, backtest_sessions
from trading_rl.overnight.history import historical_window


def records(*days):
    return [{"date": day, "open": "09:30", "close": "16:00"} for day in days]


class SessionCalendarTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "calendar.json"

    def save(self, sessions, start="2026-01-01", end="2026-12-31"):
        self.path.write_text(json.dumps({"start": start, "end": end, "sessions": sessions}))

    def test_offline_coverage_and_hash_are_independent_of_unrelated_future_sessions(self):
        self.save(records("2026-09-21", "2026-09-22", "2026-09-23"))
        with patch.object(session_calendar, "fetch_calendar") as fetch:
            first = session_calendar.load_calendar(date(2026, 9, 21), date(2026, 9, 22), path=self.path, offline=True)
            self.save(records("2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"))
            second = session_calendar.load_calendar(date(2026, 9, 21), date(2026, 9, 22), path=self.path, offline=True)
            fetch.assert_not_called()
        self.assertEqual(first.metadata["sessions_sha256"], second.metadata["sessions_sha256"])
        self.assertEqual(first.sessions, records("2026-09-21", "2026-09-22"))
        with self.assertRaisesRegex(ValueError, "coverage"):
            session_calendar.load_calendar(date(2025, 1, 1), date(2026, 9, 22), path=self.path, offline=True)

    def test_invalid_snapshot_does_not_fall_back_to_guessed_sessions(self):
        for sessions in ([], records("2026-09-21", "2026-09-21"), [{"date": "2026-09-21"}],
                         [{"date": "2026-09-21", "open": "09:30", "close": "09:00"}]):
            self.save(sessions)
            with self.subTest(sessions=sessions), self.assertRaisesRegex(ValueError, "invalid calendar snapshot"):
                session_calendar.load_calendar(date(2026, 9, 21), date(2026, 9, 22), path=self.path)

    def test_refresh_replaces_removed_sessions_and_failure_preserves_previous_snapshot(self):
        self.save(records("2026-09-21", "2026-09-22", "2026-09-23"))
        with patch.object(session_calendar, "fetch_calendar", side_effect=RuntimeError("unavailable")), self.assertRaises(RuntimeError):
            session_calendar.load_calendar(date(2026, 9, 21), date(2026, 9, 23), path=self.path, refresh=True)
        self.assertEqual(len(json.loads(self.path.read_text())["sessions"]), 3)
        with patch.object(session_calendar, "fetch_calendar", return_value=records("2026-09-21", "2026-09-23")):
            updated = session_calendar.load_calendar(date(2026, 9, 21), date(2026, 9, 23), path=self.path, refresh=True)
        self.assertEqual([row["date"] for row in updated.sessions], ["2026-09-21", "2026-09-23"])

    def inputs(self, days, last_exit):
        # No SPY file or afternoon minute data exists in this fixture.
        stamps = pd.DatetimeIndex([days[0]]).tz_localize("America/New_York").tz_convert("UTC")
        second = int((stamps[0] - ORIGIN).total_seconds())
        np.save(self.root / "AAPL.npy", np.array([[second, 1000, 1000, 1000, 1000, 100, 10, 1000]], dtype=np.int64))
        auctions = self.root / "auctions.npz"
        np.savez(auctions, date=np.array([last_exit], dtype="datetime64[D]"),
                 session=np.array([0]), condition=np.array(["O"]), price=np.array([100.0]))
        self.save(records(*days))
        return build_parser().parse_args([
            "--daily-bars-dir", str(self.root), "--minute-bars-dir", str(self.root / "absent"),
            "--calendar-path", str(self.path), "--auctions-path", str(auctions),
        ])

    def test_final_morning_exit_does_not_need_spy_or_completed_daily_bars(self):
        args = self.inputs(["2026-09-21", "2026-09-22", "2026-09-23"], "2026-09-23")
        dates, context, _, _ = backtest_sessions(args, now=pd.Timestamp("2026-09-23T14:00:00Z"))
        self.assertEqual(dates[-1], pd.Timestamp("2026-09-23"))
        self.assertEqual(len(dates), 3)
        hours = pd.to_datetime(context, unit="s", origin="2010-01-01", utc=True).tz_convert("America/New_York")
        self.assertEqual(list(hours.hour), [4, 4, 4])

    def test_future_exits_excluded_even_if_archive_contains_prices(self):
        args = self.inputs(["2026-09-21", "2026-09-22", "2026-09-23"], "2026-09-23")
        dates, _, _, _ = backtest_sessions(args, now=pd.Timestamp("2026-09-23T13:00:00Z"))
        self.assertEqual(dates[-1], pd.Timestamp("2026-09-22"))
        args.end_date = pd.Timestamp("2026-09-23")
        with self.assertRaisesRegex(ValueError, "has not elapsed"):
            backtest_sessions(args, now=pd.Timestamp("2026-09-23T13:00:00Z"))

    def test_holidays_early_closes_and_missing_observed_sessions_follow_calendar(self):
        # Thanksgiving is absent; the following day closes early.
        args = self.inputs(["2026-11-25", "2026-11-27", "2026-11-30"], "2026-11-30")
        sessions = records("2026-11-25", "2026-11-27", "2026-11-30")
        sessions[1]["close"] = "13:00"
        self.save(sessions)
        dates, context, closes, _ = backtest_sessions(args, now=pd.Timestamp("2026-11-30T15:00:00Z"))
        self.assertEqual(dates.strftime("%Y-%m-%d").tolist(), [row["date"] for row in sessions])
        window = historical_window(
            dates, context, self.root / "no-close-prints.npz", since=dates[0], end_date=dates[-1],
            months=None, ema_span=2, min_history_days=2, min_trading_days=2,
            entry_time=945, session_closes=closes,
        )
        self.assertEqual(window.shortened_entries, {date(2026, 11, 27)})
        self.assertEqual(window.entry_session_mask.tolist(), [True, False, True])

    def test_local_four_am_context_tracks_dst(self):
        args = self.inputs(["2026-03-06", "2026-03-09"], "2026-03-09")
        _, context, _, _ = backtest_sessions(args, now=pd.Timestamp("2026-03-09T15:00:00Z"))
        utc = pd.to_datetime(context, unit="s", origin="2010-01-01", utc=True)
        self.assertEqual(list(utc.hour), [9, 8])


if __name__ == "__main__":
    unittest.main()
