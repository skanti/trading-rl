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
from dashboard_config import DashboardConfigError
from dashboard_daemon import (
    AlpacaClient,
    _firestore_client,
    _is_paper_url,
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

    def test_closed_basket_totals(self):
        trades = metrics.closed_basket(
            {
                "entry_orders": {"NVDA": order(10, 100.0)},
                "exit_orders": {"NVDA": order(10, 110.0)},
            }
        )
        self.assertEqual(trades[0].pnl, 100.0)
        self.assertEqual(metrics.basket_totals(trades)["pnl"], 100.0)


class SnapshotTest(unittest.TestCase):
    def test_snapshot_preserves_the_frontend_contract(self):
        state = {
            "position": {
                "status": "closed",
                "symbols": ["NVDA"],
                "entry_orders": {"NVDA": order(10, 100.0)},
                "exit_orders": {"NVDA": order(10, 110.0)},
            },
            "ranking": {"trade_date": "2026-08-24"},
        }
        snapshot = build_snapshot(
            FakeAlpacaClient(),
            state,
            now=datetime(2026, 8, 25, 9, 40, tzinfo=metrics.EASTERN),
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


class SafetyTest(unittest.TestCase):
    def test_paper_endpoint_check_uses_the_hostname(self):
        self.assertTrue(_is_paper_url("https://paper-api.alpaca.markets/v2"))
        self.assertFalse(_is_paper_url("https://paper-api.alpaca.markets.evil.example/v2"))

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


if __name__ == "__main__":
    unittest.main()
