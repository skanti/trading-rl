"""Ranking alerts preserve trading retries and never send real test email."""

import fcntl
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import Mock, patch

from test_live import config

from trading_rl.notifications import send_text_email
from trading_rl.overnight.callbacks import RankingFailure, RankingFailureEmail
from trading_rl.overnight.live import (
    EASTERN,
    RANKING_PIPELINE_VERSION,
    DailyArtifacts,
    StateStore,
    _validate_args,
    parse_live_arguments,
    rank_for_day,
    run_daemon,
)

DAY = date(2026, 9, 21)


class RankingNotificationTest(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.store = StateStore(self.root / "state.json")
        self.settings = replace(config(), strategy_name="liquidity-trend-vol")
        self.failure = RankingFailure(
            DAY,
            self.settings.strategy_name,
            "2026-09-21T18:00:00+00:00",
            "ValueError",
            "missing opening auction",
            "ValueError: missing opening auction",
            self.root / str(DAY) / "live.log",
        )

    def notifier(self):
        return RankingFailureEmail(
            "armen.avetisyan.to@gmail.com",
            self.root / "delivered.json",
            config_path=self.root / "smtp.yaml",
            timeout_seconds=7,
        )

    def test_rank_error_callback_runs_after_unlock_and_preserves_error(self):
        broker = Mock()
        original = ValueError("calendar unavailable")
        broker.calendar.side_effect = original
        received = []

        def callback(failure):
            with self.store.lock_path.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                received.append(failure)

        with self.assertRaises(ValueError) as caught:
            rank_for_day(
                broker, self.store, self.settings, DAY, on_ranking_failure=callback
            )
        self.assertIs(caught.exception, original)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].trade_date, DAY)
        self.assertIn("calendar unavailable", received[0].traceback)
        self.assertFalse(self.store.path.exists())

    def test_callback_failure_cannot_replace_ranking_failure(self):
        broker = Mock()
        original = ValueError("daily data unavailable")
        broker.calendar.side_effect = original
        callback = Mock(side_effect=OSError("SMTP unavailable"))
        with (
            self.assertLogs("overnight-liquidity-live", level="ERROR"),
            self.assertRaises(ValueError) as caught,
        ):
            rank_for_day(
                broker, self.store, self.settings, DAY, on_ranking_failure=callback
            )
        self.assertIs(caught.exception, original)
        callback.assert_called_once()

    def test_risk_preparation_failure_notifies_without_saving_ranking(self):
        bars = self.root / "bars"
        bars.mkdir()
        (bars / "_download_manifest.json").write_text(
            json.dumps({"timeframe": "1Day", "adjustment": "split"})
        )
        (bars / "A.npy").touch()
        settings = replace(self.settings, daily_bars_dir=bars)
        broker = Mock()
        broker.calendar.return_value = [
            {"date": str(day)} for day in (date(2026, 9, 17), date(2026, 9, 18), DAY)
        ]
        callback = Mock()
        mocks = {
            "eligible_assets": ["A", "B"],
            "load_nasdaq_security_master": {},
            "seed_missing_daily_cache": {},
            "refresh_daily_cache": {},
            "dollar_volume_shortlist": (["A", "B"], 2, []),
            "load_cached_daily_bars": {},
            "completed_liquidity_ranking": [("A", 2.0, 2), ("B", 1.0, 2)],
        }
        with ExitStack() as stack:
            mocked = {}
            for name, value in mocks.items():
                mocked[name] = stack.enter_context(
                    patch("trading_rl.overnight.live." + name, return_value=value)
                )
            stack.enter_context(
                patch(
                    "trading_rl.overnight.live.prepare_live_risk",
                    side_effect=ValueError("missing opening auction"),
                )
            )
            with self.assertRaisesRegex(ValueError, "missing opening auction"):
                rank_for_day(
                    broker, self.store, settings, DAY, on_ranking_failure=callback
                )
        callback.assert_called_once()
        self.assertEqual(
            callback.call_args.args[0].error_message, "missing opening auction"
        )
        self.assertIn("SPY", mocked["seed_missing_daily_cache"].call_args.args[2])
        self.assertFalse(self.store.path.exists())

    def test_successful_cached_ranking_does_not_notify(self):
        settings = config()
        ranking = {
            "trade_date": str(DAY),
            "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
            "liquidity_scheme": settings.liquidity_scheme,
        }
        self.store.save({"version": 1, "ranking": ranking})
        callback = Mock()
        self.assertEqual(
            rank_for_day(
                Mock(), self.store, settings, DAY, on_ranking_failure=callback
            ),
            ranking,
        )
        callback.assert_not_called()

    @patch("trading_rl.overnight.callbacks.send_text_email")
    def test_email_has_details_and_deduplicates_across_restarts(self, send):
        callback = self.notifier()
        callback(self.failure)
        callback(replace(self.failure, error_message="another failure on retry"))
        self.notifier()(self.failure)
        send.assert_called_once()
        recipient, subject, body = send.call_args.args
        self.assertEqual(recipient, "armen.avetisyan.to@gmail.com")
        self.assertIn("[LIVE]", subject)
        for value in (
            str(DAY),
            self.settings.strategy_name,
            self.failure.error_message,
            str(self.failure.log_path),
            self.failure.traceback,
        ):
            self.assertIn(value, body)
        self.assertEqual(send.call_args.kwargs["timeout_seconds"], 7)
        callback(replace(self.failure, trade_date=date(2026, 9, 22)))
        self.assertEqual(send.call_count, 2)

    @patch("trading_rl.overnight.callbacks.send_text_email")
    def test_failed_delivery_is_retried_and_never_marked_delivered(self, send):
        send.side_effect = [OSError("SMTP unavailable"), None]
        callback = self.notifier()
        with self.assertRaises(OSError):
            callback(self.failure)
        self.assertFalse(callback.state_path.exists())
        self.notifier()(self.failure)
        self.assertEqual(send.call_count, 2)
        self.assertTrue(callback.state_path.exists())

    @patch("trading_rl.overnight.callbacks.send_text_email")
    def test_daemon_retries_ranking_and_sends_one_alert(self, send):
        broker = Mock()
        broker.calendar.side_effect = ValueError("daily data unavailable")
        with (
            patch(
                "trading_rl.overnight.live._market_session_status",
                return_value=(True, date(2026, 9, 22), True),
            ),
            patch("trading_rl.overnight.live.datetime", wraps=datetime) as clock,
            patch("trading_rl.overnight.live.time_module.sleep") as sleep,
            self.assertLogs("overnight-liquidity-live", level="ERROR"),
            self.assertRaises(StopIteration),
        ):
            clock.now.return_value = datetime(2026, 9, 21, 14, tzinfo=EASTERN)

            def advance(_seconds):
                if sleep.call_count == 2:
                    raise StopIteration
                clock.now.return_value = datetime(2026, 9, 21, 14, 1, tzinfo=EASTERN)

            sleep.side_effect = advance
            run_daemon(
                broker,
                self.store,
                self.settings,
                time(14),
                time(15, 45),
                time(6),
                20,
                75,
                DailyArtifacts(self.root),
                on_ranking_failure=self.notifier(),
            )
        self.assertEqual(broker.calendar.call_count, 2)
        send.assert_called_once()
        broker.submit_order.assert_not_called()

    @patch("trading_rl.notifications.smtplib.SMTP")
    def test_smtp_uses_tls_auth_timeout_and_only_admin_recipient(self, smtp):
        path = self.root / "smtp.yaml"
        path.write_text(
            "smtp:\n  host: smtp.example.com\n  port: 587\n  user: bot@example.com\n  password: test-secret\nnotifications:\n  recipients: [someone-else@example.com]\n"
        )
        connection = smtp.return_value.__enter__.return_value
        connection.send_message.return_value = {}
        send_text_email(
            "admin@example.com",
            "Ranking failed",
            "Example error",
            config_path=path,
            timeout_seconds=7,
        )
        smtp.assert_called_once_with("smtp.example.com", 587, timeout=7)
        connection.starttls.assert_called_once()
        connection.login.assert_called_once_with("bot@example.com", "test-secret")
        message = connection.send_message.call_args.args[0]
        self.assertEqual(message["To"], "admin@example.com")
        self.assertEqual(
            connection.send_message.call_args.kwargs["to_addrs"], ["admin@example.com"]
        )
        self.assertIn("Example error", message.get_content())
        self.assertNotIn("test-secret", message.as_string())

    def test_configuration_defaults_and_cli_override(self):
        parser, args, _ = parse_live_arguments(
            ["run", "--submit", "--allow-live-endpoint"]
        )
        _validate_args(parser, args)
        self.assertEqual(args.ranking_failure_email, "armen.avetisyan.to@gmail.com")
        _, disabled, _ = parse_live_arguments(["rank", "--ranking-failure-email", ""])
        self.assertEqual(disabled.ranking_failure_email, "")


if __name__ == "__main__":
    unittest.main()
