import unittest
from datetime import date, datetime
from unittest import mock

from dashboard.scripts.test_dashboard_daemon import FakeAlpacaClient
from trading_rl.dashboard.benchmark import spy_buy_and_hold
from trading_rl.dashboard.daemon import AlpacaClient, build_snapshot
from trading_rl.dashboard.metrics import EASTERN, EquityPoint


class BenchmarkTest(unittest.TestCase):
    def test_daily_closes_use_the_same_capital_and_preserve_missing_dates(self):
        series = [EquityPoint(date(2026, 8, 24), 10000, 0, 0)]
        bars = [
            {"t": "2026-08-24T04:00:00Z", "c": 100},
            {"t": "2026-08-26T04:00:00Z", "c": 110},
            {"t": "2026-08-27T04:00:00Z", "c": 999},
        ]
        result = spy_buy_and_hold(series, bars, date(2026, 8, 26))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["as_of"], "2026-08-26")
        self.assertEqual([p["day"] for p in result["points"]], ["2026-08-24", "2026-08-26"])
        self.assertAlmostEqual(result["points"][-1]["equity"], 11000)
        self.assertAlmostEqual(result["points"][-1]["profit_loss_pct"], 0.1)

    def test_missing_initial_close_does_not_rebase_to_a_later_day(self):
        series = [EquityPoint(date(2026, 8, 24), 10000, 0, 0)]
        result = spy_buy_and_hold(series, [{"t": "2026-08-25T04:00:00Z", "c": 100}], date(2026, 8, 25))
        self.assertEqual(result["points"], [])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(spy_buy_and_hold(series, [], date(2026, 8, 23))["status"], "pending")
        self.assertEqual(spy_buy_and_hold([], [], date(2026, 8, 25))["points"], [])

    def test_snapshot_publishes_even_when_optional_benchmark_fails(self):
        client = FakeAlpacaClient()
        client.spy_daily_bars = mock.Mock(side_effect=RuntimeError("market data unavailable"))
        snapshot = build_snapshot(client, {}, inception=date(2026, 8, 24), now=datetime(2026, 8, 25, 10, tzinfo=EASTERN))
        self.assertEqual(snapshot["benchmark"]["status"], "unavailable")
        self.assertEqual(snapshot["account"]["equity"], 109500)
        start, end = client.spy_daily_bars.call_args.args
        self.assertEqual(start, date(2026, 8, 24))
        self.assertEqual(end.isoformat(), "2026-08-24T23:59:59.999999-04:00")

    def test_client_requests_all_adjustments_and_follows_pagination(self):
        client = AlpacaClient("trading-key", "trading-secret", data_key="data-key", data_secret="data-secret")
        pages = [
            {"bars": {"SPY": [{"t": "2026-03-06T05:00:00Z", "c": 100}]}, "next_page_token": "next"},
            {"bars": {"SPY": [{"t": "2026-03-09T04:00:00Z", "c": 101}]}, "next_page_token": None},
        ]
        with mock.patch.object(client, "_get", side_effect=pages) as get:
            bars = client.spy_daily_bars(date(2026, 3, 6), datetime(2026, 3, 10, tzinfo=EASTERN))
        self.assertEqual(len(bars), 2)
        self.assertEqual(get.call_args_list[0].kwargs["adjustment"], "all")
        self.assertEqual(get.call_args_list[0].kwargs["timeframe"], "1Day")
        self.assertEqual(get.call_args_list[0].kwargs["feed"], "sip")
        self.assertTrue(get.call_args_list[0].kwargs["data"])
        self.assertEqual(get.call_args_list[1].kwargs["page_token"], "next")
        self.assertEqual(client.data_session.headers["APCA-API-KEY-ID"], "data-key")
        result = spy_buy_and_hold([EquityPoint(date(2026, 3, 6), 100, 0, 0)], bars, date(2026, 3, 9))
        self.assertEqual([p["day"] for p in result["points"]], ["2026-03-06", "2026-03-09"])

    def test_optional_data_retries_have_a_bounded_delay(self):
        client = AlpacaClient("key", "secret")
        limited = mock.Mock(status_code=429, headers={"Retry-After": "3600"})
        success = mock.Mock(status_code=200, content=b"{}")
        success.json.return_value = {"bars": {"SPY": []}}
        with (
            mock.patch.object(client.data_session, "get", side_effect=[limited, success]) as get,
            mock.patch("trading_rl.dashboard.daemon.time_module.sleep") as sleep,
        ):
            self.assertEqual(client.spy_daily_bars(date(2026, 3, 6), datetime(2026, 3, 10, tzinfo=EASTERN)), [])
        sleep.assert_called_once_with(8.0)
        self.assertEqual(get.call_args.kwargs["timeout"], 10.0)
