import json
import tempfile
import threading
import unittest
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from baseline import performance, reporting
from baseline.dashboard_config import (
    DashboardConfigError,
    auth_email,
    load_dashboard_config,
    service_account,
    try_load_dashboard_config,
)
from baseline.dashboard_publisher import (
    build_snapshot,
    load_state,
    session_records,
)
from baseline.performance import EASTERN

UTC = timezone.utc


def stamp(day: date) -> int:
    """Alpaca stamps a 1D point at UTC midnight of the day AFTER the session."""
    return int(datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=UTC).timestamp())


def history(pairs, base_value=100000.0):
    days = [day for day, _ in pairs]
    equities = [value for _, value in pairs]
    profits, percents = [], []
    previous = base_value
    for value in equities:
        profits.append(value - previous)
        percents.append((value - previous) / previous if previous else 0.0)
        previous = value
    return {
        "timestamp": [stamp(day) for day in days],
        "equity": equities,
        "profit_loss": profits,
        "profit_loss_pct": percents,
        "base_value": base_value,
        "timeframe": "1D",
    }


def order(qty, price):
    return {"filled_qty": str(qty), "filled_avg_price": str(price)}


def dashboard_config():
    return OmegaConf.create(
        {
            "dashboard": {"url": "https://dash.example", "title": "Overnight Liquidity"},
            "auth": {"username": "gambler8", "password": "leverage1000x", "email_domain": "d.local"},
            "smtp": {"host": "smtp.example", "port": 587, "user": "bot@example", "password": "pw"},
            "notifications": {"recipients": ["a@example", "b@example"]},
        }
    )


class SessionDateTest(unittest.TestCase):
    def test_timestamp_resolves_to_the_eastern_session_date(self):
        # 2026-08-25T00:00Z is 2026-08-24 20:00 in New York: the Monday session.
        self.assertEqual(performance.session_date(1787616000), date(2026, 8, 24))

    def test_a_utc_read_would_be_off_by_one(self):
        moment = datetime.fromtimestamp(1787616000, tz=UTC)
        self.assertEqual(moment.date(), date(2026, 8, 25))
        self.assertEqual(performance.session_date(1787616000), date(2026, 8, 24))


class EquitySeriesTest(unittest.TestCase):
    def test_points_are_sorted_and_deduplicated(self):
        payload = history([(date(2026, 3, 3), 101.0), (date(2026, 3, 2), 100.0), (date(2026, 3, 3), 102.0)])
        series = performance.equity_series(payload)
        self.assertEqual([point.day for point in series], [date(2026, 3, 2), date(2026, 3, 3)])
        self.assertEqual(series[-1].equity, 102.0)

    def test_since_trims_alpaca_backfill_before_account_inception(self):
        payload = history([(date(2025, 1, 2), 100000.0), (date(2026, 3, 2), 101000.0)])
        series = performance.equity_series(payload, since=date(2026, 1, 1))
        self.assertEqual([point.day for point in series], [date(2026, 3, 2)])

    def test_missing_equity_entries_are_skipped(self):
        payload = history([(date(2026, 3, 2), 100.0), (date(2026, 3, 3), 101.0)])
        payload["equity"][1] = None
        self.assertEqual(len(performance.equity_series(payload)), 1)

    def test_empty_history_is_not_an_error(self):
        self.assertEqual(performance.equity_series({}), [])


class HistoryPeriodTest(unittest.TestCase):
    def test_period_reaches_past_inception(self):
        now = datetime(2026, 8, 25, tzinfo=EASTERN)
        self.assertEqual(performance.history_period("2025-07-17T22:22:32Z", now), "2A")

    def test_a_young_account_still_asks_for_a_whole_year(self):
        now = datetime(2026, 8, 25, tzinfo=EASTERN)
        self.assertEqual(performance.history_period("2026-08-01T00:00:00Z", now), "1A")

    def test_unparseable_creation_falls_back(self):
        self.assertEqual(performance.history_period("not-a-date"), "1A")


class PerformanceTableTest(unittest.TestCase):
    def setUp(self):
        # Fri 21 Aug closes at 108,000; Mon 24 Aug at 109,000. "Now" is Tue 25 Aug.
        self.payload = history(
            [
                (date(2026, 7, 31), 105000.0),
                (date(2026, 8, 20), 107000.0),
                (date(2026, 8, 21), 108000.0),
                (date(2026, 8, 24), 109000.0),
            ]
        )
        self.series = performance.equity_series(self.payload)
        self.now = datetime(2026, 8, 25, 10, 0, tzinfo=EASTERN)

    def table(self, equity, last_equity):
        account = {"equity": str(equity), "last_equity": str(last_equity)}
        return performance.performance_table(self.series, account, self.payload, now=self.now)

    def test_today_anchors_on_alpacas_previous_close(self):
        bucket = self.table(109500.0, 109000.0)["today"]
        self.assertEqual(bucket.start_equity, 109000.0)
        self.assertAlmostEqual(bucket.pnl, 500.0)

    def test_week_anchors_on_the_close_before_monday(self):
        bucket = self.table(109500.0, 109000.0)["week"]
        self.assertEqual(bucket.start_day, date(2026, 8, 21))
        self.assertAlmostEqual(bucket.pnl, 1500.0)

    def test_month_anchors_on_the_final_close_of_the_previous_month(self):
        bucket = self.table(109500.0, 109000.0)["month"]
        self.assertEqual(bucket.start_day, date(2026, 7, 31))
        self.assertAlmostEqual(bucket.pnl, 4500.0)

    def test_year_and_inception_fall_back_to_base_value_when_no_earlier_close(self):
        table = self.table(109500.0, 109000.0)
        self.assertAlmostEqual(table["year"].pnl, 9500.0)
        self.assertAlmostEqual(table["inception"].pnl, 9500.0)
        self.assertAlmostEqual(table["inception"].pnl_pct, 0.095)

    def test_a_flat_untraded_account_reports_zeroes_rather_than_raising(self):
        payload = history([(date(2026, 8, 24), 100000.0)])
        series = performance.equity_series(payload)
        table = performance.performance_table(
            series, {"equity": "100000", "last_equity": "100000"}, payload, now=self.now
        )
        for key in performance.BUCKET_ORDER:
            self.assertEqual(table[key].pnl, 0.0)
            self.assertEqual(table[key].pnl_pct, 0.0)

    def test_empty_series_does_not_divide_by_zero(self):
        table = performance.performance_table([], {"equity": "0", "last_equity": "0"}, {}, now=self.now)
        self.assertEqual(table["inception"].pnl_pct, 0.0)


class StatisticsTest(unittest.TestCase):
    def test_drawdown_extremes_and_hit_rate(self):
        series = performance.equity_series(
            history(
                [
                    (date(2026, 3, 2), 100.0),
                    (date(2026, 3, 3), 120.0),
                    (date(2026, 3, 4), 90.0),
                    (date(2026, 3, 5), 110.0),
                ]
            )
        )
        stats = performance.statistics(series)
        self.assertAlmostEqual(stats.max_drawdown, 30.0)
        self.assertAlmostEqual(stats.max_drawdown_pct, 0.25)
        self.assertEqual(stats.best_day.day, date(2026, 3, 3))
        self.assertEqual(stats.worst_day.day, date(2026, 3, 4))
        self.assertEqual((stats.winning_sessions, stats.losing_sessions), (2, 1))

    def test_flat_series_has_no_winners_and_no_drawdown(self):
        series = performance.equity_series(history([(date(2026, 3, 2), 100.0), (date(2026, 3, 3), 100.0)]))
        stats = performance.statistics(series)
        self.assertEqual(stats.max_drawdown, 0.0)
        self.assertEqual(stats.win_rate, 0.0)
        self.assertIsNone(stats.best_day)

    def test_empty_series(self):
        self.assertEqual(performance.statistics([]).sessions, 0)


class ClosedBasketTest(unittest.TestCase):
    def test_per_symbol_profit_and_totals(self):
        position = {
            "entry_orders": {"NVDA": order(10, 100.0), "TSLA": order(5, 200.0)},
            "exit_orders": {"NVDA": order(10, 110.0), "TSLA": order(5, 190.0)},
        }
        trades = performance.closed_basket(position)
        self.assertEqual([trade.symbol for trade in trades], ["NVDA", "TSLA"])
        self.assertAlmostEqual(trades[0].pnl, 100.0)
        self.assertAlmostEqual(trades[0].pnl_pct, 0.1)
        self.assertAlmostEqual(trades[1].pnl, -50.0)
        totals = performance.basket_totals(trades)
        self.assertAlmostEqual(totals["pnl"], 50.0)
        self.assertAlmostEqual(totals["entry_notional"], 2000.0)

    def test_a_partial_exit_is_measured_on_the_quantity_actually_sold(self):
        position = {
            "entry_orders": {"NVDA": order(10, 100.0)},
            "exit_orders": {"NVDA": order(4, 110.0)},
        }
        trade = performance.closed_basket(position)[0]
        self.assertAlmostEqual(trade.qty, 4.0)
        self.assertAlmostEqual(trade.exit_notional, 440.0)
        self.assertAlmostEqual(trade.pnl, -560.0)

    def test_symbols_without_an_exit_or_without_a_fill_are_omitted(self):
        position = {
            "entry_orders": {"NVDA": order(10, 100.0), "AMD": order(0, 0), "MSFT": order(3, 400.0)},
            "exit_orders": {"NVDA": order(10, 110.0), "AMD": order(0, 0)},
        }
        self.assertEqual([trade.symbol for trade in performance.closed_basket(position)], ["NVDA"])

    def test_no_position_yields_an_empty_basket(self):
        self.assertEqual(performance.closed_basket({}), [])
        self.assertEqual(performance.basket_totals([])["pnl"], 0.0)


class FilledNotionalTest(unittest.TestCase):
    def test_matches_the_arithmetic_the_daily_summary_records(self):
        orders = {"A": order(2, 3.5), "B": order(1.5, 10.0)}
        self.assertAlmostEqual(performance.filled_notional(orders), 22.0)

    def test_unfilled_and_malformed_orders_contribute_nothing(self):
        orders = {"A": {"filled_qty": None, "filled_avg_price": None}, "B": {"filled_qty": "x"}}
        self.assertEqual(performance.filled_notional(orders), 0.0)


class DigestTest(unittest.TestCase):
    def setUp(self):
        self.payload = history([(date(2026, 8, 21), 108000.0), (date(2026, 8, 24), 109000.0)])
        self.account = {
            "equity": "109500",
            "last_equity": "109000",
            "cash": "5000",
            "created_at": "2026-01-02T00:00:00Z",
        }
        self.position = {
            "entry_date": "2026-08-24",
            "exit_date": "2026-08-25",
            "status": "closed",
            "entry_orders": {"NVDA": order(10, 100.0)},
            "exit_orders": {"NVDA": order(10, 110.0)},
        }
        self.config = dashboard_config()
        self.now = datetime(2026, 8, 25, 9, 40, tzinfo=EASTERN)

    def digest(self, position=None):
        return reporting.build_digest(self.account, self.payload, position, self.config, now=self.now)

    def test_subject_carries_the_days_result(self):
        self.assertIn("+$500.00", self.digest(self.position).subject)
        self.assertIn("2026-08-25", self.digest(self.position).subject)

    def test_text_contains_every_period_row_and_the_trade(self):
        text = reporting.render_text(self.digest(self.position))
        for label in ("Today", "Week to date", "Month to date", "Year to date", "Since inception"):
            self.assertIn(label, text)
        self.assertIn("NVDA", text)
        self.assertIn("+$100.00", text)
        self.assertIn("https://dash.example", text)

    def test_html_contains_the_dashboard_link_and_the_total_row(self):
        html = reporting.render_html(self.digest(self.position))
        self.assertIn('href="https://dash.example"', html)
        self.assertIn("Total", html)
        self.assertIn("NVDA", html)

    def test_an_empty_basket_renders_the_fallback_line(self):
        text = reporting.render_text(self.digest(None))
        html = reporting.render_html(self.digest(None))
        self.assertIn("No positions were held overnight.", text)
        self.assertIn("No positions were held overnight.", html)

    def test_message_is_multipart_and_addressed_to_every_recipient(self):
        message = reporting.build_message(self.digest(self.position), self.config)
        self.assertTrue(message.is_multipart())
        self.assertIn("a@example", message["To"])
        self.assertIn("b@example", message["To"])
        self.assertEqual(
            {part.get_content_type() for part in message.walk() if part.get_content_maintype() == "text"},
            {"text/plain", "text/html"},
        )

    def test_send_digest_uses_starttls_and_logs_in(self):
        with mock.patch("baseline.reporting.smtplib.SMTP") as smtp:
            session = smtp.return_value.__enter__.return_value
            recipients = reporting.send_digest(self.digest(self.position), self.config)
        smtp.assert_called_once_with("smtp.example", 587, timeout=30)
        session.starttls.assert_called_once()
        session.login.assert_called_once_with("bot@example", "pw")
        session.send_message.assert_called_once()
        self.assertEqual(recipients, ["a@example", "b@example"])

    def test_build_digest_live_asks_for_a_period_covering_inception(self):
        client = mock.Mock()
        client.account.return_value = self.account
        client.portfolio_history.return_value = self.payload
        reporting.build_digest_live(client, self.position, self.config, now=self.now)
        client.portfolio_history.assert_called_once_with(period="1A", timeframe="1D")


class FakePublisherClient:
    def __init__(self, positions=(), account=None, history_payload=None):
        self._positions = list(positions)
        self._account = account or {
            "equity": "109500",
            "last_equity": "109000",
            "cash": "5000",
            "created_at": "2026-01-02T00:00:00Z",
            "account_number": "PA1",
            "status": "ACTIVE",
        }
        self._history = history_payload or history([(date(2026, 8, 24), 109000.0)])

    def account(self):
        return dict(self._account)

    def positions(self):
        return [dict(item) for item in self._positions]

    def portfolio_history(self, period="1A", timeframe="1D"):
        return dict(self._history)

    def clock(self):
        return {"is_open": False, "next_open": "2026-08-25T09:30:00-04:00"}


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.config = dashboard_config()
        self.now = datetime(2026, 8, 25, 9, 40, tzinfo=EASTERN)

    def test_snapshot_shape_and_numeric_coercion(self):
        client = FakePublisherClient(
            positions=[{"symbol": "NVDA", "qty": "10", "market_value": "1100", "unrealized_pl": "100"}]
        )
        snapshot = build_snapshot(client, {}, self.config, now=self.now)
        self.assertEqual(
            set(snapshot),
            {
                "version", "updated_at", "trading_day", "account", "performance", "statistics",
                "equity_curve", "positions", "strategy", "closed_basket", "basket_totals",
                "market", "meta",
            },
        )
        # Alpaca's numeric strings become numbers so the browser never parses strings.
        self.assertIsInstance(snapshot["account"]["equity"], float)
        self.assertIsInstance(snapshot["positions"][0]["qty"], float)
        self.assertEqual(snapshot["account"]["status"], "ACTIVE")
        self.assertEqual(set(snapshot["performance"]), set(performance.BUCKET_ORDER))

    def test_an_untraded_account_publishes_empty_collections(self):
        snapshot = build_snapshot(FakePublisherClient(), {}, self.config, now=self.now)
        self.assertEqual(snapshot["positions"], [])
        self.assertEqual(snapshot["closed_basket"], [])
        self.assertEqual(snapshot["basket_totals"], {})
        self.assertIsNone(snapshot["strategy"]["status"])

    def test_strategy_state_and_closed_basket_are_carried_through(self):
        state = {
            "position": {
                "status": "closed",
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "symbols": ["NVDA"],
                "entry_orders": {"NVDA": order(10, 100.0)},
                "exit_orders": {"NVDA": order(10, 110.0)},
            },
            "ranking": {"trade_date": "2026-08-24"},
        }
        snapshot = build_snapshot(FakePublisherClient(), state, self.config, now=self.now)
        self.assertEqual(snapshot["strategy"]["status"], "closed")
        self.assertEqual(snapshot["strategy"]["ranking_trade_date"], "2026-08-24")
        self.assertAlmostEqual(snapshot["closed_basket"][0]["pnl"], 100.0)
        self.assertAlmostEqual(snapshot["basket_totals"]["pnl"], 100.0)

    def test_a_failing_clock_does_not_fail_the_snapshot(self):
        client = FakePublisherClient()
        client.clock = mock.Mock(side_effect=RuntimeError("boom"))
        self.assertEqual(build_snapshot(client, {}, self.config, now=self.now)["market"], {})


class WorkDirectoryTest(unittest.TestCase):
    def test_missing_state_file_is_treated_as_not_yet_traded(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_state(Path(directory) / "state.json"), {})

    def test_corrupt_state_file_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("{not json")
            self.assertEqual(load_state(path), {})

    def test_session_records_are_newest_first_and_include_realized_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day, pnl in (("2026-08-21", 10.0), ("2026-08-24", -5.0)):
                folder = root / day
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "trading_day": day,
                            "last_action": "exit",
                            "position": {
                                "status": "closed",
                                "entry_orders": {"NVDA": order(10, 100.0)},
                                "exit_orders": {"NVDA": order(10, 100.0 + pnl / 10)},
                            },
                            "execution": {"realized_pnl_before_fees": pnl},
                        }
                    )
                )
            (root / "not-a-day").mkdir()
            records = session_records(root)
        self.assertEqual([record["trading_day"] for record in records], ["2026-08-24", "2026-08-21"])
        self.assertAlmostEqual(records[0]["realized_pnl"], -5.0)
        self.assertAlmostEqual(records[0]["trades"][0]["pnl"], -5.0)

    def test_an_error_only_summary_is_still_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2026-08-24").mkdir()
            (root / "2026-08-24" / "summary.json").write_text(
                json.dumps({"trading_day": "2026-08-24", "last_action": "error", "error": "403"})
            )
            records = session_records(root)
        self.assertEqual(records[0]["error"], "403")
        self.assertEqual(records[0]["trades"], [])

    def test_a_missing_service_account_fails_fast_rather_than_hanging(self):
        # Application default credentials block on the GCP metadata server off-cloud,
        # which would stall every publish from the trading daemon.
        from baseline.dashboard_publisher import _firestore_client

        config = OmegaConf.create(
            {"firebase": {"project_id": "p", "service_account": "/nonexistent/key.json"}}
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(DashboardConfigError) as caught:
                _firestore_client(config)
        self.assertIn("service-account", str(caught.exception))

    def test_a_missing_work_directory_yields_no_sessions(self):
        self.assertEqual(session_records(Path("/nonexistent/ppv1")), [])


class GuardedSideEffectTest(unittest.TestCase):
    """`_run_guarded` is what keeps SMTP and Firestore from taking down the daemon."""

    def test_an_exception_is_contained(self):
        from baseline.live_overnight_liquidity import _run_guarded

        def explode():
            raise RuntimeError("smtp refused the connection")

        _run_guarded("test", explode, 5.0)  # must return normally

    def test_a_hang_is_abandoned_rather_than_blocking_the_loop(self):
        from baseline.live_overnight_liquidity import _run_guarded

        started = threading.Event()
        release = threading.Event()

        def stall():
            started.set()
            release.wait(30)  # simulates an unreachable Firestore with no deadline

        began = monotonic()
        try:
            _run_guarded("test", stall, 0.5)
            elapsed = monotonic() - began
        finally:
            release.set()

        self.assertTrue(started.wait(5))
        # The daemon moved on long before the stalled call would have finished.
        self.assertLess(elapsed, 5.0)

    def test_even_a_baseexception_does_not_escape(self):
        from baseline.live_overnight_liquidity import _run_guarded

        def explode():
            raise KeyboardInterrupt

        _run_guarded("test", explode, 5.0)

    def test_the_worker_never_blocks_process_exit(self):
        from baseline.live_overnight_liquidity import _run_guarded

        names = set()

        def note():
            names.add(threading.current_thread().daemon)

        _run_guarded("test", note, 5.0)
        self.assertEqual(names, {True})


class DaemonIntegrationTest(unittest.TestCase):
    """The daemon's two outbound side effects must never be able to break trading."""

    def test_a_failing_publish_is_swallowed(self):
        from baseline.live_overnight_liquidity import StateStore, _publish_dashboard

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            with mock.patch(
                "baseline.dashboard_publisher.publish", side_effect=RuntimeError("firestore down")
            ):
                # Must return, not raise: an exit is in flight when this runs.
                _publish_dashboard(mock.Mock(), store, Path(directory), None, True)

    def test_publishing_can_be_disabled(self):
        from baseline.live_overnight_liquidity import _publish_dashboard

        with mock.patch("baseline.dashboard_publisher.build_snapshot") as build:
            _publish_dashboard(mock.Mock(), mock.Mock(), Path("/tmp"), None, False)
        build.assert_not_called()

    def test_an_unimportable_dependency_is_contained(self):
        from baseline.live_overnight_liquidity import _publish_dashboard

        with mock.patch.dict("sys.modules", {"baseline.dashboard_publisher": None}):
            _publish_dashboard(mock.Mock(), mock.Mock(), Path("/tmp"), None, True)

    def test_a_missing_config_file_is_contained(self):
        from baseline.live_overnight_liquidity import _send_exit_digest

        _send_exit_digest(mock.Mock(), {"status": "closed"}, "/nonexistent/config.yaml", True)

    def test_a_failing_digest_is_swallowed(self):
        from baseline.live_overnight_liquidity import _send_exit_digest

        with mock.patch("baseline.reporting.send_digest", side_effect=RuntimeError("smtp down")):
            _send_exit_digest(mock.Mock(), {"status": "closed"}, None, True)

    def test_the_digest_only_fires_once_the_position_is_actually_closed(self):
        from baseline.live_overnight_liquidity import _send_exit_digest

        with mock.patch("baseline.reporting.build_digest_live") as build:
            _send_exit_digest(mock.Mock(), {"status": "exiting"}, None, True)
            _send_exit_digest(mock.Mock(), {"status": "exit_plan"}, None, True)
        build.assert_not_called()

    def test_run_emails_the_digest_but_does_not_publish_by_default(self):
        from baseline.live_overnight_liquidity import build_parser

        args = build_parser().parse_args(["run", "--submit"])
        self.assertFalse(args.no_email)
        self.assertFalse(args.dashboard)

    def test_run_daemon_does_not_publish_unless_asked(self):
        import inspect

        from baseline.live_overnight_liquidity import run_daemon

        signature = inspect.signature(run_daemon)
        self.assertIs(signature.parameters["publish_dashboard"].default, False)
        self.assertIs(signature.parameters["email_digest"].default, True)

    def test_publishing_is_opt_in(self):
        from baseline.live_overnight_liquidity import build_parser

        args = build_parser().parse_args(["run", "--submit", "--dashboard"])
        self.assertTrue(args.dashboard)
        self.assertEqual(args.publish_interval_seconds, 300.0)

    def test_publishing_can_be_pinned_to_strategy_actions_only(self):
        from baseline.live_overnight_liquidity import build_parser

        args = build_parser().parse_args(
            ["run", "--submit", "--dashboard", "--publish-interval-seconds", "0"]
        )
        self.assertEqual(args.publish_interval_seconds, 0.0)


class ConfigTest(unittest.TestCase):
    def write(self, directory, body):
        path = Path(directory) / "config.yaml"
        path.write_text(body)
        return path

    def test_the_committed_config_loads(self):
        config = load_dashboard_config()
        self.assertEqual(auth_email(config), "gambler8@trading-dashboard-ccdd5.local")
        self.assertEqual(str(config.firebase.project_id), "trading-dashboard-ccdd5")

    def test_the_committed_config_holds_no_alpaca_credentials(self):
        # Alpaca keys stay in the environment (ml/.env), never in the committed config.
        self.assertIsNone(load_dashboard_config().get("alpaca"))

    def test_a_missing_required_key_is_named_in_the_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "dashboard:\n  url: x\n")
            with self.assertRaises(DashboardConfigError) as caught:
                load_dashboard_config(path)
        self.assertIn("firebase.project_id", str(caught.exception))

    # A self-contained fixture: deriving one by editing the committed config.yaml
    # silently stops testing anything as soon as that file changes.
    COMPLETE = """
dashboard: {url: "https://x", title: "T"}
auth: {username: u, password: pppppp, email_domain: d.local}
firebase: {project_id: p, collection: accounts, document: paper}
smtp: {host: h, port: 587, user: u@x, password: pw}
notifications: {recipients: %s}
"""

    def test_an_empty_recipient_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, self.COMPLETE % "[]")
            with self.assertRaises(DashboardConfigError) as caught:
                load_dashboard_config(path)
        self.assertIn("recipients", str(caught.exception))

    def test_several_recipients_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, self.COMPLETE % '["a@x", "b@y"]')
            config = load_dashboard_config(path)
        self.assertEqual(list(config.notifications.recipients), ["a@x", "b@y"])

    def test_the_committed_config_addresses_both_recipients(self):
        recipients = list(load_dashboard_config().notifications.recipients)
        self.assertIn("armen.avetisyan.to@gmail.com", recipients)
        self.assertIn("mobilephone.felix@gmail.com", recipients)

    def test_try_load_returns_none_instead_of_raising(self):
        self.assertIsNone(try_load_dashboard_config("/nonexistent/config.yaml"))

    def test_service_account_accepts_an_inline_key_mapping(self):
        config = OmegaConf.create({"firebase": {"service_account": {"type": "service_account", "project_id": "p"}}})
        self.assertEqual(service_account(config)["project_id"], "p")

    def test_service_account_accepts_a_path(self):
        config = OmegaConf.create({"firebase": {"service_account": "~/key.json"}})
        self.assertTrue(str(service_account(config)).endswith("key.json"))


if __name__ == "__main__":
    unittest.main()
