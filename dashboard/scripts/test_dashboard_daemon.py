from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf

import dashboard_metrics as metrics
import dashboard_daemon
import dashboard_digest
from dashboard_config import DashboardConfigError
from dashboard_daemon import (
    AlpacaClient,
    _firestore_client,
    _is_paper_url,
    _trading_mode,
    build_parser,
    build_snapshot,
    first_trade_date,
    load_state,
    session_records,
)


def history(rows: list[tuple[date, float]]) -> dict[str, object]:
    return {
        "timestamp": [
            datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp() + 86400
            for day, _ in rows
        ],
        "equity": [equity for _, equity in rows],
        "profit_loss": [0.0 for _ in rows],
        "profit_loss_pct": [0.0 for _ in rows],
        "base_value": rows[0][1] if rows else 0.0,
    }


def order(qty: float, price: float) -> dict[str, str]:
    return {"filled_qty": str(qty), "filled_avg_price": str(price)}


class FakeAlpacaClient:
    trading_url = "https://paper-api.alpaca.markets/v2"

    def __init__(self) -> None:
        self._history = history([(date(2026, 8, 24), 109000.0)])

    def account(self):
        return {
            "account_number": "PA1",
            "status": "ACTIVE",
            "equity": "109500",
            "last_equity": "109000",
            "cash": "5000",
            "created_at": "2026-01-02T00:00:00Z",
        }

    def positions(self):
        return [{"symbol": "NVDA", "qty": "10", "market_value": "1100"}]

    def portfolio_history(self, period="1A", timeframe="1D"):
        return dict(self._history)

    def clock(self):
        return {"is_open": False, "next_open": "2026-08-25T09:30:00-04:00"}


class MetricsTest(unittest.TestCase):
    def test_equity_series_is_sorted_and_deduplicated(self):
        payload = history(
            [
                (date(2026, 3, 3), 101.0),
                (date(2026, 3, 2), 100.0),
                (date(2026, 3, 3), 102.0),
            ]
        )
        series = metrics.equity_series(payload)
        self.assertEqual([point.day for point in series], [date(2026, 3, 2), date(2026, 3, 3)])
        self.assertEqual(series[-1].equity, 102.0)

    def test_performance_and_drawdown(self):
        payload = history(
            [
                (date(2026, 7, 31), 100.0),
                (date(2026, 8, 21), 120.0),
                (date(2026, 8, 24), 90.0),
            ]
        )
        series = metrics.equity_series(payload)
        table = metrics.performance_table(
            series,
            {"equity": "95", "last_equity": "90"},
            payload,
            now=datetime(2026, 8, 25, 10, 0, tzinfo=metrics.EASTERN),
        )
        self.assertEqual(set(table), set(metrics.BUCKET_ORDER))
        self.assertEqual(table["today"].pnl, 5.0)
        self.assertEqual(metrics.statistics(series).max_drawdown, 30.0)

    def test_realized_equity_uses_exit_date_and_excludes_open_session(self):
        sessions = [
            {
                "trading_day": "2026-08-26",
                "entry_date": "2026-08-26",
                "exit_date": "2026-08-27",
                "status": "open",
                "realized_pnl": None,
            },
            {
                "trading_day": "2026-08-25",
                "entry_date": "2026-08-25",
                "exit_date": "2026-08-26",
                "status": "closed",
                "realized_pnl": -45.47051112189365,
            },
        ]

        result = metrics.realized_equity_series(
            sessions,
            base_value=100000.0,
            inception=date(2026, 8, 25),
        )

        self.assertEqual([point.day for point in result], [date(2026, 8, 25), date(2026, 8, 26)])
        self.assertEqual(result[0].equity, 100000.0)
        self.assertAlmostEqual(result[-1].equity, 99954.5294888781)
        self.assertAlmostEqual(result[-1].profit_loss, -45.47051112189365)

    def test_realized_equity_aggregates_baskets_closed_on_the_same_day(self):
        sessions = [
            {
                "trading_day": "2026-08-25",
                "entry_date": "2026-08-25",
                "exit_date": "2026-08-26",
                "status": "closed",
                "realized_pnl": -45.0,
            },
            {
                "trading_day": "2026-08-26",
                "entry_date": "2026-08-26",
                "exit_date": "2026-08-26",
                "status": "closed",
                "realized_pnl": 20.0,
            },
        ]

        result = metrics.realized_equity_series(
            sessions,
            base_value=100000.0,
            inception=date(2026, 8, 25),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1].day, date(2026, 8, 26))
        self.assertEqual(result[-1].profit_loss, -25.0)

    def test_realized_equity_starts_flat_before_the_first_exit(self):
        result = metrics.realized_equity_series(
            [],
            base_value=100000.0,
            inception=date(2026, 8, 25),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].equity, 100000.0)

    def test_closed_basket_totals(self):
        trades = metrics.closed_basket(
            {
                "entry_orders": {"NVDA": order(10, 100.0)},
                "exit_orders": {"NVDA": order(10, 110.0)},
            }
        )
        self.assertEqual(trades[0].pnl, 100.0)
        self.assertEqual(metrics.basket_totals(trades)["pnl"], 100.0)

    def test_closed_basket_aggregates_partial_exit_attempts(self):
        position = {
            "entry_date": "2026-08-25",
            "entry_orders": {"NVDA": order(10, 100.0)},
            "exit_orders": {"NVDA": order(6, 111.0)},
        }
        orders = [
            {
                "id": "first",
                "client_order_id": "olq-20260825-x-NVDA",
                "symbol": "NVDA",
                "side": "sell",
                "status": "canceled",
                "filled_qty": "4",
                "filled_avg_price": "109",
            },
            {
                "id": "second",
                "client_order_id": "olq-20260825-x-NVDA-2",
                "symbol": "NVDA",
                "side": "sell",
                "status": "filled",
                "filled_qty": "6",
                "filled_avg_price": "111",
            },
        ]

        trades = metrics.closed_basket_from_order_history(position, orders)

        self.assertEqual(trades[0].qty, 10.0)
        self.assertEqual(trades[0].exit_notional, 1102.0)
        self.assertEqual(trades[0].exit_price, 110.2)
        self.assertEqual(trades[0].pnl, 102.0)


class SnapshotTest(unittest.TestCase):
    def test_snapshot_preserves_the_frontend_contract(self):
        state = {
            "position": {
                "status": "closed",
                "share_mode": "whole",
                "symbols": ["NVDA"],
                "budget": 1000.0,
                "per_symbol_notional": 1000.0,
                "estimated_deployed_notional": 900.0,
                "target_quantities": {"NVDA": 9},
                "skipped_symbols": [],
                "entry_orders": {"NVDA": order(10, 100.0)},
                "exit_orders": {"NVDA": order(10, 110.0)},
            },
            "ranking": {"trade_date": "2026-08-24"},
        }
        snapshot = build_snapshot(
            FakeAlpacaClient(),
            state,
            inception=date(2026, 8, 24),
            now=datetime(2026, 8, 25, 9, 40, tzinfo=metrics.EASTERN),
            session_history=[
                {
                    "trading_day": "2026-08-24",
                    "entry_date": "2026-08-24",
                    "exit_date": "2026-08-25",
                    "status": "closed",
                    "realized_pnl": 100.0,
                }
            ],
        )
        self.assertEqual(
            set(snapshot),
            {
                "version",
                "updated_at",
                "trading_day",
                "account",
                "performance",
                "statistics",
                "equity_curve",
                "positions",
                "strategy",
                "closed_basket",
                "basket_totals",
                "market",
                "meta",
            },
        )
        self.assertIsInstance(snapshot["account"]["equity"], float)
        self.assertIsInstance(snapshot["positions"][0]["qty"], float)
        self.assertEqual(snapshot["basket_totals"]["pnl"], 100.0)
        self.assertEqual(snapshot["strategy"]["share_mode"], "whole")
        self.assertEqual(snapshot["strategy"]["target_quantities"], {"NVDA": 9})
        self.assertEqual(snapshot["strategy"]["estimated_deployed_notional"], 900.0)
        self.assertEqual(snapshot["equity_curve"][-1]["day"], "2026-08-25")
        self.assertEqual(snapshot["equity_curve"][-1]["profit_loss"], 100.0)


class ArtifactTest(unittest.TestCase):
    def test_missing_or_corrupt_state_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.assertEqual(load_state(path), {})
            path.write_text("{bad json")
            self.assertEqual(load_state(path), {})

    def test_sessions_are_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day, pnl in (("2026-08-21", 10.0), ("2026-08-24", -5.0)):
                folder = root / day
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "trading_day": day,
                            "position": {
                                "status": "closed",
                                "entry_orders": {"NVDA": order(10, 100.0)},
                                "exit_orders": {"NVDA": order(10, 100.0 + pnl / 10)},
                            },
                            "execution": {"realized_pnl_before_fees": pnl},
                        }
                    )
                )
            records = session_records(root)
        self.assertEqual([row["trading_day"] for row in records], ["2026-08-24", "2026-08-21"])
        self.assertEqual(records[0]["trades"][0]["pnl"], -5.0)

    def test_inception_is_the_earliest_entry_with_a_real_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day, qty in (("2026-08-24", 0), ("2026-08-25", 12), ("2026-08-26", 12)):
                folder = root / day
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "trading_day": day,
                            "position": {
                                "entry_date": day,
                                "entry_orders": {"NVDA": order(qty, 100.0)},
                            },
                        }
                    )
                )
            self.assertEqual(first_trade_date(root), date(2026, 8, 25))

    def test_session_history_reconciles_partial_fills_and_deduplicates_artifacts(self):
        position = {
            "status": "closed",
            "entry_date": "2026-08-25",
            "exit_date": "2026-08-26",
            "entry_orders": {"NVDA": order(10, 100.0)},
            "exit_orders": {"NVDA": order(6, 111.0)},
        }
        order_history = [
            {
                "id": "first",
                "client_order_id": "olq-20260825-x-NVDA",
                "symbol": "NVDA",
                "side": "sell",
                "filled_qty": "4",
                "filled_avg_price": "109",
            },
            {
                "id": "second",
                "client_order_id": "olq-20260825-x-NVDA-2",
                "symbol": "NVDA",
                "side": "sell",
                "filled_qty": "6",
                "filled_avg_price": "111",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day in ("2026-08-25", "2026-08-26"):
                folder = root / day
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "trading_day": day,
                            "position": position,
                            "execution": {
                                "entry_filled_notional": 1000.0,
                                "exit_filled_notional": 666.0,
                                "realized_return_before_fees": -0.334,
                            },
                        }
                    )
                )

            records = session_records(root, order_history=order_history)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trading_day"], "2026-08-25")
        self.assertEqual(records[0]["exit_notional"], 1102.0)
        self.assertEqual(records[0]["realized_return"], 0.102)


class SafetyTest(unittest.TestCase):
    def test_paper_endpoint_check_uses_the_hostname(self):
        self.assertTrue(_is_paper_url("https://paper-api.alpaca.markets/v2"))
        self.assertFalse(_is_paper_url("https://paper-api.alpaca.markets.evil.example/v2"))

    def test_trading_mode_is_detected_from_known_https_endpoint(self):
        self.assertEqual(_trading_mode("https://paper-api.alpaca.markets/v2"), "paper")
        self.assertEqual(_trading_mode("https://api.alpaca.markets/v2"), "live")
        with self.assertRaisesRegex(ValueError, "unsupported Alpaca trading endpoint"):
            _trading_mode("https://paper-api.alpaca.markets.evil.example/v2")
        with self.assertRaisesRegex(ValueError, "must use https"):
            _trading_mode("http://api.alpaca.markets/v2")

    def test_live_endpoint_needs_no_extra_parser_flag(self):
        args = build_parser().parse_args(
            ["--trading-url", "https://api.alpaca.markets/v2"]
        )
        self.assertEqual(_trading_mode(args.trading_url), "live")
        self.assertFalse(hasattr(args, "allow_live_endpoint"))

    def test_publish_rejects_unknown_firestore_mode_before_connecting(self):
        config = OmegaConf.create(
            {"firebase": {"collection": "accounts", "document": "current"}}
        )
        with self.assertRaisesRegex(ValueError, "unsupported trading mode"):
            dashboard_daemon.publish({}, [], config, trading_mode="unknown")

    def test_publish_uses_stable_firestore_document_for_live_mode(self):
        config = OmegaConf.create(
            {"firebase": {"collection": "accounts", "document": "current"}}
        )
        client = mock.Mock()
        reference = client.collection.return_value.document.return_value
        reference.get.return_value.to_dict.return_value = {}
        with mock.patch.object(dashboard_daemon, "_firestore_client", return_value=client):
            dashboard_daemon.publish({}, [], config, trading_mode="live")

        client.collection.assert_called_once_with("accounts")
        client.collection.return_value.document.assert_called_once_with("current")

    def test_publish_clears_sessions_when_account_mode_changes(self):
        config = OmegaConf.create(
            {"firebase": {"collection": "accounts", "document": "current"}}
        )
        client = mock.Mock()
        reference = client.collection.return_value.document.return_value
        reference.get.return_value.to_dict.return_value = {"meta": {"mode": "paper"}}
        stale_session = mock.Mock()
        reference.collection.return_value.list_documents.return_value = [stale_session]

        with mock.patch.object(dashboard_daemon, "_firestore_client", return_value=client):
            dashboard_daemon.publish({}, [], config, trading_mode="live")

        client.batch.return_value.delete.assert_called_once_with(stale_session)

    def test_daemon_is_the_default_and_once_is_explicit(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).once)
        self.assertTrue(parser.parse_args(["--once"]).once)

    def test_transient_alpaca_connection_errors_are_retried(self):
        client = AlpacaClient("key", "secret", max_retries=1)
        response = mock.Mock(status_code=200, content=b"{}")
        response.json.return_value = {"status": "ACTIVE"}
        with mock.patch.object(
            client.session,
            "get",
            side_effect=[dashboard_daemon.requests.ConnectionError("offline"), response],
        ):
            with mock.patch.object(dashboard_daemon.time_module, "sleep"):
                self.assertEqual(client.account(), {"status": "ACTIVE"})

    def test_missing_service_account_fails_fast(self):
        config = OmegaConf.create(
            {"firebase": {"project_id": "p", "service_account": "/nonexistent/key.json"}}
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DashboardConfigError):
                _firestore_client(config)


class DigestTest(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "dashboard": {"title": "Overnight Liquidity", "url": "https://dash.example"},
                "smtp": {
                    "host": "smtp.example",
                    "port": 587,
                    "user": "bot@example",
                    "password": "app-password",
                },
                "notifications": {"recipients": ["a@example", "b@example"]},
            }
        )
        self.state = {
            "position": {
                "status": "closed",
                "entry_date": "2026-08-25",
                "exit_date": "2026-08-26",
                "exit_completed_at": "2026-08-26T09:35:00-04:00",
            }
        }
        self.snapshot = {
            "trading_day": "2026-08-26",
            "meta": {"mode": "live"},
            "account": {"equity": 100500.0, "cash": 100500.0},
            "performance": {
                "today": {
                    "label": "Today",
                    "pnl": 500.0,
                    "pnl_pct": 0.005,
                }
            },
            "closed_basket": [
                {
                    "symbol": "NVDA",
                    "qty": 10.0,
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "pnl": 100.0,
                    "pnl_pct": 0.1,
                }
            ],
            "basket_totals": {
                "entry_notional": 1000.0,
                "exit_notional": 1100.0,
                "pnl": 100.0,
                "pnl_pct": 0.1,
            },
        }

    def test_digest_keys_a_closed_basket_even_after_its_exit_day(self):
        self.assertIsNotNone(dashboard_digest.digest_key(self.state))
        open_state = {"position": {**self.state["position"], "status": "open"}}
        self.assertIsNone(dashboard_digest.digest_key(open_state))

    def test_message_is_multipart_and_addressed_to_all_recipients(self):
        message = dashboard_digest.build_message(self.snapshot, self.state, self.config)
        self.assertTrue(message.is_multipart())
        self.assertIn("a@example", message["To"])
        self.assertIn("[LIVE]", message["Subject"])
        self.assertIn("+$100.00", message["Subject"])
        self.assertNotIn("+$500.00", message["Subject"])
        text = dashboard_digest.render_text(self.snapshot, self.state, self.config)
        self.assertIn("NVDA", text)
        self.assertIn("TOTAL", text)

    def test_html_labels_mode_and_includes_fill_based_total(self):
        html = dashboard_digest.render_html(self.snapshot, self.state, self.config)
        self.assertIn("[LIVE]", html)
        self.assertIn("TOTAL", html)
        self.assertIn("+$100.00", html)

    def test_html_colors_profits_green_and_losses_red(self):
        html = dashboard_digest.render_html(self.snapshot, self.state, self.config)
        self.assertIn("color:#047857", html)
        losing = dict(self.snapshot)
        losing["performance"] = {
            "today": {"label": "Today", "pnl": -5.0, "pnl_pct": -0.001}
        }
        losing["closed_basket"] = [
            {**self.snapshot["closed_basket"][0], "pnl": -5.0, "pnl_pct": -0.005}
        ]
        self.assertIn("color:#be123c", dashboard_digest.render_html(losing, self.state, self.config))

    def test_send_uses_starttls_and_login(self):
        with mock.patch("dashboard_digest.smtplib.SMTP") as smtp:
            session = smtp.return_value.__enter__.return_value
            recipients = dashboard_digest.send_digest(self.snapshot, self.state, self.config)
        smtp.assert_called_once_with("smtp.example", 587, timeout=30)
        session.starttls.assert_called_once()
        session.login.assert_called_once_with("bot@example", "app-password")
        session.send_message.assert_called_once()
        self.assertEqual(recipients, ["a@example", "b@example"])

    def test_delivery_marker_is_durable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest-state.json"
            dashboard_digest.mark_delivered(path, "first")
            dashboard_digest.mark_delivered(path, "first")
            dashboard_digest.mark_delivered(path, "second")
            self.assertEqual(dashboard_digest.delivered_keys(path), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
