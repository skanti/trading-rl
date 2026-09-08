from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from trading_rl.overnight.broker_fees import (
    broker_fees_for_session,
    cache_fee_activities,
    load_fee_cache,
)


class FeeCacheTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fee_activities.json"
        self.day = date(2026, 9, 1)
        self.now = datetime(2026, 9, 2, 16, tzinfo=UTC)
        self.rows = [
            {"activity_type": "FEE", "date": "2026-09-01", "net_amount": "-0.21"}
        ]
        self.client = Mock()
        self.client.account_activities.return_value = self.rows

    def fetch(self, when=None, **kwargs):
        return broker_fees_for_session(
            self.client,
            self.path,
            self.day,
            as_of=when or self.now,
            **kwargs,
        )

    def test_recent_cache_reused_until_hourly_boundary(self):
        first, _ = self.fetch()
        again, _ = self.fetch(self.now + timedelta(minutes=59))
        self.assertEqual(first, again)
        self.client.account_activities.assert_called_once_with(
            "FEE",
            after=date(2026, 8, 31),
            until=date(2026, 9, 5),
        )
        self.fetch(self.now + timedelta(hours=1))
        self.assertEqual(self.client.account_activities.call_count, 2)

    def test_unchanged_response_advances_successful_check(self):
        self.fetch()
        checked = self.now + timedelta(hours=1)
        self.fetch(checked)
        persisted = json.loads(self.path.read_text())
        self.assertEqual(persisted["last_checked_at"], checked.isoformat())
        self.fetch(checked + timedelta(minutes=2))
        self.assertEqual(self.client.account_activities.call_count, 2)

    def test_historical_confirmed_cache_refreshes_weekly_and_accepts_corrections(self):
        old_check = self.now + timedelta(days=10)
        self.fetch(old_check)
        self.fetch(old_check + timedelta(days=6, hours=23))
        self.client.account_activities.assert_called_once()
        self.client.account_activities.return_value = [
            dict(self.rows[0], net_amount="-0.19")
        ]
        corrected, _ = self.fetch(old_check + timedelta(days=7))
        self.assertAlmostEqual(corrected["cost"], 0.19)
        self.assertEqual(self.client.account_activities.call_count, 2)

    def test_first_posted_fee_does_not_stop_recent_refreshes(self):
        self.fetch()
        self.client.account_activities.return_value = self.rows + [
            dict(self.rows[0], net_amount="-0.01")
        ]
        updated, _ = self.fetch(self.now + timedelta(hours=1))
        self.assertEqual(updated["count"], 2)
        self.assertAlmostEqual(updated["cost"], 0.22)

    def test_pending_zero_needs_successful_check_after_grace_period(self):
        self.client.account_activities.return_value = []
        pending, _ = self.fetch()
        self.assertEqual(pending["status"], "pending")
        later = self.now + timedelta(days=10)
        offline, _ = broker_fees_for_session(None, self.path, self.day, as_of=later)
        self.assertEqual(offline["status"], "pending")
        before = self.path.read_bytes()
        self.client.account_activities.side_effect = RuntimeError("temporary failure")
        failed, warning = self.fetch(later)
        self.assertEqual(failed["status"], "pending")
        self.assertIn("refresh failed", warning)
        self.assertEqual(self.path.read_bytes(), before)
        self.client.account_activities.side_effect = None
        confirmed, _ = self.fetch(later + timedelta(minutes=2))
        self.assertEqual(confirmed["status"], "complete")
        self.assertEqual(confirmed["cost"], 0)

    def test_unexpected_empty_response_preserves_fees_and_retries_hourly(self):
        old_check = self.now + timedelta(days=10)
        self.fetch(old_check)
        checked = old_check + timedelta(days=7)
        self.client.account_activities.return_value = []
        retained, warning = self.fetch(checked)
        self.assertEqual(retained["cost"], 0.21)
        self.assertIn("retaining", warning)
        self.assertEqual(retained["fetched_at"], old_check.isoformat())
        self.assertEqual(retained["last_checked_at"], checked.isoformat())
        self.fetch(checked + timedelta(minutes=2))
        self.assertEqual(self.client.account_activities.call_count, 2)
        self.client.account_activities.return_value = self.rows
        restored, warning = self.fetch(checked + timedelta(hours=1))
        self.assertIsNone(warning)
        self.assertNotIn("refresh_warning", restored)
        self.assertEqual(self.client.account_activities.call_count, 3)

    def test_force_refresh_bypasses_fresh_cache(self):
        self.fetch()
        self.fetch(force_refresh=True)
        self.assertEqual(self.client.account_activities.call_count, 2)

    def test_invalid_cache_and_wrong_date_are_refetched(self):
        for payload in [
            "{",
            "[]",
            json.dumps({"activity_date": "2026-08-31", "count": 1, "cost": 2}),
            json.dumps({"activity_date": "2026-09-01", "count": 1, "cost": "NaN"}),
        ]:
            with self.subTest(payload=payload):
                self.path.write_text(payload)
                self.client.reset_mock()
                result, _ = self.fetch()
                self.assertEqual(result["cost"], 0.21)
                self.client.account_activities.assert_called_once()

    def test_missing_or_future_check_timestamp_cannot_suppress_refresh(self):
        for stamp in [None, (self.now + timedelta(days=1)).isoformat()]:
            with self.subTest(stamp=stamp):
                self.path.write_text(
                    json.dumps(
                        {
                            "activity_date": self.day.isoformat(),
                            "count": 1,
                            "cost": 0.21,
                            "fetched_at": stamp,
                        }
                    )
                )
                self.client.reset_mock()
                self.fetch()
                self.client.account_activities.assert_called_once()

    def test_failed_request_without_cache_remains_unavailable(self):
        self.client.account_activities.side_effect = RuntimeError("offline")
        result, warning = self.fetch()
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("offline", warning)
        self.assertFalse(self.path.exists())

    def test_interrupted_atomic_write_preserves_previous_cache(self):
        self.fetch()
        before = self.path.read_bytes()
        with patch(
            "trading_rl.overnight.broker_fees.os.replace",
            side_effect=OSError("write failed"),
        ):
            with self.assertRaises(OSError):
                cache_fee_activities(
                    self.path, self.day, self.rows, as_of=self.now + timedelta(hours=1)
                )
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertIsNotNone(load_fee_cache(self.path, self.day))
