import json
import logging
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np

from baseline.live_overnight_liquidity import (
    DEFAULT_EXCHANGES,
    EASTERN,
    RANKING_PIPELINE_VERSION,
    AlpacaClient,
    DailyArtifacts,
    DailyLogHandler,
    StateStore,
    StrategyConfig,
    _completed_session_end,
    _select_unconflicted_candidates,
    _validate_args,
    _validate_exit_clock,
    available_budget,
    build_parser,
    completed_liquidity_ranking,
    eligible_assets,
    enter_for_day,
    exit_position,
    fast_activity_candidates,
)


def config(top=2):
    return StrategyConfig(
        top=top,
        ema_span=2,
        min_history_days=2,
        minimum_trading_days=2,
        lookback_calendar_days=90,
        activity_candidates=100,
        feed="iex",
        exchanges=DEFAULT_EXCHANGES,
        data_batch_size=100,
        data_workers=1,
        capital=None,
        capital_fraction=0.95,
        cash_buffer_fraction=0.02,
        fill_timeout_seconds=1.0,
        poll_seconds=0.001,
    )


def bar(day, volume, vwap=100.0):
    return {"t": f"{day}T04:00:00Z", "v": volume, "vw": vwap, "c": vwap}


class FakeBroker:
    def __init__(self):
        self.current_positions = {
            "HELD": {"symbol": "HELD", "qty": "1.5"},
        }
        self.orders = {}
        self.submissions = []

    def positions(self):
        return list(self.current_positions.values())

    def position(self, symbol):
        return self.current_positions.get(symbol)

    def list_orders(self, status="open"):
        return []

    def account(self):
        return {
            "cash": "1000",
            "buying_power": "4000",
            "non_marginable_buying_power": "1000",
            "equity": "1000",
        }

    def calendar(self, start, end):
        return [{"date": "2026-08-25"}]

    def order_by_client_id(self, client_order_id):
        return self.orders.get(client_order_id)

    def submit_order(self, payload):
        self.submissions.append(dict(payload))
        symbol = payload["symbol"]
        if payload["side"] == "buy":
            quantity = "0.95"
            self.current_positions[symbol] = {"symbol": symbol, "qty": quantity}
        else:
            quantity = payload["qty"]
            self.current_positions.pop(symbol, None)
        order = {
            "id": f"order-{len(self.submissions)}",
            "client_order_id": payload["client_order_id"],
            "symbol": symbol,
            "side": payload["side"],
            "status": "filled",
            "qty": quantity,
            "notional": payload.get("notional"),
            "filled_qty": quantity,
            "filled_avg_price": "100",
            "submitted_at": "2026-08-24T19:55:00Z",
            "filled_at": "2026-08-24T19:55:00Z",
        }
        self.orders[payload["client_order_id"]] = order
        return order

    def order(self, order_id):
        return next(order for order in self.orders.values() if order["id"] == order_id)

    def cancel_order(self, order_id):
        raise AssertionError("filled fake orders must not be canceled")


class FakeActivityClient:
    def most_active_stocks(self, by, top=100):
        symbols = ["AAPL", "QQQ"] if by == "volume" else ["MSFT", "QQQ"]
        return {"most_actives": [{"symbol": symbol} for symbol in symbols]}


class FakeQueuedBroker(FakeBroker):
    def submit_order(self, payload):
        if payload["side"] != "sell":
            return super().submit_order(payload)
        self.submissions.append(dict(payload))
        order = {
            "id": f"order-{len(self.submissions)}",
            "client_order_id": payload["client_order_id"],
            "symbol": payload["symbol"],
            "side": "sell",
            "status": "accepted",
            "qty": payload["qty"],
            "notional": None,
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": "2026-08-25T13:00:00Z",
            "filled_at": None,
        }
        self.orders[payload["client_order_id"]] = order
        return order


class FakeClockBroker:
    def __init__(self, timestamp, is_open):
        self.timestamp = timestamp
        self.is_open = is_open

    def clock(self):
        return {"timestamp": self.timestamp, "is_open": self.is_open}


class LiveOvernightLiquidityTest(unittest.TestCase):
    def test_exit_time_accepts_0900(self):
        parser = build_parser()
        args = parser.parse_args(["exit", "--exit-time", "09:00"])

        parsed = _validate_args(parser, args)

        self.assertIsInstance(parsed, StrategyConfig)
        self.assertEqual(args.ranking_time.strftime("%H:%M"), "15:00")
        self.assertEqual(parsed.poll_seconds, 1.0)

        preview_args = parser.parse_args(["preview"])
        preview_config = _validate_args(parser, preview_args)
        self.assertIsInstance(preview_config, StrategyConfig)

    def test_preopen_exit_clock_allows_queued_orders(self):
        preopen = FakeClockBroker("2026-08-25T09:00:00-04:00", False)
        regular = FakeClockBroker("2026-08-25T09:30:00-04:00", True)

        self.assertFalse(_validate_exit_clock(preopen, date(2026, 8, 25)))
        self.assertTrue(_validate_exit_clock(regular, date(2026, 8, 25)))

    def test_daily_artifacts_write_market_data_logs_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = date(2026, 8, 24)
            store = StateStore(root / "state.json")
            state = store.load()
            state["ranking"] = {"trade_date": day.isoformat(), "candidates": []}
            store.save(state)
            artifacts = DailyArtifacts(root)

            metadata = artifacts.write_ticks(
                day,
                {"AAPL": [bar("2026-08-21", 123)]},
                feed="sip",
                start=date(2026, 8, 1),
                end=day,
            )
            artifacts.write_summary(day, "rank", store, config())

            handler = DailyLogHandler(root, fixed_day=day)
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            record = logging.LogRecord("test", logging.INFO, __file__, 1, "rank complete", (), None)
            handler.emit(record)

            daily = root / day.isoformat()
            tick = json.loads((daily / "ticks.jsonl").read_text())
            summary = json.loads((daily / "summary.json").read_text())
            self.assertEqual(tick["symbol"], "AAPL")
            self.assertEqual(tick["timeframe"], "1Day")
            self.assertEqual(metadata["rows"], 1)
            self.assertEqual(summary["last_action"], "rank")
            self.assertEqual(summary["market_data"]["feed"], "sip")
            self.assertEqual((daily / "live.log").read_text(), "INFO rank complete\n")

    def test_ranking_excludes_trade_date_bar(self):
        sessions = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]
        original = {
            "A": [bar("2026-08-20", 100), bar("2026-08-21", 100)],
            "B": [bar("2026-08-20", 10), bar("2026-08-21", 10)],
        }
        leaked = {symbol: list(bars) for symbol, bars in original.items()}
        leaked["B"].append(bar("2026-08-24", 10_000_000))

        expected = completed_liquidity_ranking(original, sessions, sessions[-1], 2, 2)
        actual = completed_liquidity_ranking(leaked, sessions, sessions[-1], 2, 2)

        self.assertEqual(expected, actual)
        self.assertEqual(actual[0][0], "A")

    def test_ranking_requires_minimum_completed_trading_days(self):
        sessions = [
            date(2026, 8, 19),
            date(2026, 8, 20),
            date(2026, 8, 21),
            date(2026, 8, 24),
        ]
        bars = {
            "ESTABLISHED": [
                bar("2026-08-19", 100),
                bar("2026-08-20", 100),
                bar("2026-08-21", 100),
            ],
            "RECENT": [bar("2026-08-20", 1_000), bar("2026-08-21", 1_000)],
        }

        ranking = completed_liquidity_ranking(
            bars,
            sessions,
            sessions[-1],
            ema_span=2,
            min_history_days=1,
            minimum_trading_days=3,
        )

        self.assertEqual([symbol for symbol, _, _ in ranking], ["ESTABLISHED"])

    def test_only_fractionable_exchange_listed_assets_are_eligible(self):
        common = {"status": "active", "tradable": True, "fractionable": True, "class": "us_equity"}
        assets = [
            {**common, "symbol": "GOOD", "exchange": "NASDAQ"},
            {**common, "symbol": "OTC", "exchange": "OTC"},
            {**common, "symbol": "WHOLE", "exchange": "NYSE", "fractionable": False},
            {**common, "symbol": "HALT", "exchange": "NYSE", "tradable": False},
        ]

        self.assertEqual(eligible_assets(assets, DEFAULT_EXCHANGES), ["GOOD"])

    def test_company_filter_excludes_etfs_and_unknown_assets(self):
        common = {
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "class": "us_equity",
            "exchange": "NASDAQ",
        }
        assets = [
            {**common, "symbol": "AAPL"},
            {**common, "symbol": "QQQ"},
            {**common, "symbol": "UNKNOWN"},
        ]
        security_master = {
            "AAPL": {"name": "Apple Inc. - Common Stock", "etf": "N"},
            "QQQ": {"name": "Invesco QQQ Trust ETF", "etf": "Y"},
        }

        self.assertEqual(
            eligible_assets(assets, DEFAULT_EXCHANGES, security_master), ["AAPL"]
        )

    def test_fast_activity_candidates_union_both_screens_and_previous_day(self):
        common = {
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "class": "us_equity",
            "exchange": "NASDAQ",
        }
        assets = [
            {**common, "symbol": symbol}
            for symbol in ("AAPL", "MSFT", "NVDA", "QQQ")
        ]
        security_master = {
            symbol: {"name": f"{symbol} Company - Common Stock", "etf": "N"}
            for symbol in ("AAPL", "MSFT", "NVDA")
        }
        security_master["QQQ"] = {"name": "Invesco QQQ Trust ETF", "etf": "Y"}

        symbols, diagnostics = fast_activity_candidates(
            FakeActivityClient(),
            assets,
            DEFAULT_EXCHANGES,
            security_master,
            {"screened_candidate_symbols": ["NVDA"]},
        )

        self.assertEqual(symbols, ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(diagnostics["raw_candidate_symbols"], 4)
        self.assertEqual(diagnostics["excluded_or_unavailable_candidates"], 1)

    def test_most_actives_uses_v1beta1_screener_endpoint(self):
        client = AlpacaClient("key", "secret")
        calls = []

        def request(method, base, path, **kwargs):
            calls.append((method, base, path, kwargs))
            return {"most_actives": [{"symbol": "AAPL"}]}

        client._request = request
        response = client.most_active_stocks("trades", 100)

        self.assertEqual(response["most_actives"][0]["symbol"], "AAPL")
        self.assertEqual(calls[0][1], "https://data.alpaca.markets/v1beta1")
        self.assertEqual(calls[0][2], "screener/stocks/most-actives")
        self.assertEqual(calls[0][3]["params"], {"by": "trades", "top": 100})
        self.assertTrue(calls[0][3]["data_credentials"])

    def test_data_requests_use_separate_credentials_and_completed_timestamp(self):
        client = AlpacaClient(
            "trading-key",
            "trading-secret",
            data_key="data-key",
            data_secret="data-secret",
        )
        trading_session = client._session()
        data_session = client._session(data_credentials=True)

        self.assertIsNot(trading_session, data_session)
        self.assertEqual(trading_session.headers["APCA-API-KEY-ID"], "trading-key")
        self.assertEqual(data_session.headers["APCA-API-KEY-ID"], "data-key")

        calls = []

        def request(method, base, path, **kwargs):
            calls.append((method, base, path, kwargs))
            return {"bars": {}, "next_page_token": None}

        client._request = request
        completed_end = _completed_session_end(date(2026, 8, 24))
        client.historical_daily_bars(
            ["AAPL"], date(2026, 2, 25), completed_end, "sip"
        )

        self.assertEqual(completed_end.isoformat(), "2026-08-24T23:59:59-04:00")
        self.assertEqual(calls[0][3]["params"]["end"], completed_end.isoformat())
        self.assertTrue(calls[0][3]["data_credentials"])

    def test_budget_uses_cash_instead_of_margin_buying_power(self):
        budget = available_budget(
            {"cash": "1000", "buying_power": "4000", "equity": "1000"}, config(top=10)
        )
        self.assertEqual(budget, 950.0)

    def test_budget_respects_reserved_non_marginable_buying_power(self):
        budget = available_budget(
            {"cash": "1000", "buying_power": "4000", "non_marginable_buying_power": "600"},
            config(top=10),
        )
        self.assertEqual(budget, 588.0)

    def test_entry_skips_conflicting_position_and_is_idempotent(self):
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["ranking"] = {
                "trade_date": "2026-08-24",
                "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
                "candidates": [
                    {"rank": 1, "symbol": "HELD"},
                    {"rank": 2, "symbol": "A"},
                    {"rank": 3, "symbol": "B"},
                ],
            }
            store.save(state)

            first = enter_for_day(broker, store, config(), date(2026, 8, 24), submit=True)
            second = enter_for_day(broker, store, config(), date(2026, 8, 24), submit=True)

        self.assertEqual(first["symbols"], ["A", "B"])
        self.assertEqual(second["status"], "open")
        self.assertEqual(len(broker.submissions), 2)
        self.assertTrue(all(order["side"] == "buy" for order in broker.submissions))

    def test_exit_sells_only_strategy_owned_symbols(self):
        broker = FakeBroker()
        broker.current_positions.update(
            {"A": {"symbol": "A", "qty": "0.75"}, "B": {"symbol": "B", "qty": "1.25"}}
        )
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "open",
                "symbols": ["A", "B"],
                "filled_symbols": ["A", "B"],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            result = exit_position(broker, store, config(), submit=True)

        self.assertEqual(result["status"], "closed")
        self.assertEqual(set(broker.current_positions), {"HELD"})
        self.assertEqual({order["symbol"] for order in broker.submissions}, {"A", "B"})
        self.assertTrue(all(order["side"] == "sell" for order in broker.submissions))

    def test_preopen_exit_stays_queued_and_is_reconciled_after_open(self):
        broker = FakeQueuedBroker()
        broker.current_positions["A"] = {"symbol": "A", "qty": "0.75"}
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "open",
                "symbols": ["A"],
                "filled_symbols": ["A"],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            queued = exit_position(
                broker,
                store,
                config(top=1),
                submit=True,
                now=datetime(2026, 8, 25, 9, 0, tzinfo=EASTERN),
                wait_for_fill=False,
            )
            queued_order = broker.orders["olq-20260824-x-A"]
            self.assertEqual(queued["status"], "exit_queued")
            self.assertEqual(queued_order["status"], "accepted")
            self.assertIn("A", broker.current_positions)
            self.assertEqual(broker.submissions[0]["time_in_force"], "day")

            queued_order["status"] = "filled"
            queued_order["filled_qty"] = "0.75"
            broker.current_positions.pop("A")
            closed = exit_position(
                broker,
                store,
                config(top=1),
                submit=True,
                now=datetime(2026, 8, 25, 9, 30, tzinfo=EASTERN),
                wait_for_fill=True,
            )

        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["exit_orders"]["A"]["status"], "filled")
        self.assertEqual(len(broker.submissions), 1)

    def test_exit_retries_a_canceled_order_with_a_new_id(self):
        broker = FakeBroker()
        broker.current_positions["A"] = {"symbol": "A", "qty": "0.75"}
        broker.orders["olq-20260824-x-A"] = {
            "id": "canceled-order",
            "client_order_id": "olq-20260824-x-A",
            "symbol": "A",
            "side": "sell",
            "status": "canceled",
            "qty": "0.75",
            "filled_qty": "0",
        }
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "exit_incomplete",
                "symbols": ["A"],
                "filled_symbols": ["A"],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            result = exit_position(broker, store, config(top=1), submit=True)

        self.assertEqual(result["status"], "closed")
        self.assertEqual(broker.submissions[0]["client_order_id"], "olq-20260824-x-A-2")

    def test_exit_recovers_an_entry_accepted_before_state_was_saved(self):
        broker = FakeBroker()
        broker.current_positions["A"] = {"symbol": "A", "qty": "0.75"}
        broker.orders["olq-20260824-e-A"] = {
            "id": "recovered-entry",
            "client_order_id": "olq-20260824-e-A",
            "symbol": "A",
            "side": "buy",
            "status": "filled",
            "qty": "0.75",
            "filled_qty": "0.75",
        }
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "entering",
                "symbols": ["A"],
                "filled_symbols": [],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            result = exit_position(broker, store, config(top=1), submit=True)

        self.assertEqual(result["status"], "closed")
        self.assertEqual(broker.submissions[0]["side"], "sell")
        self.assertNotIn("A", broker.current_positions)


if __name__ == "__main__":
    unittest.main()


def _ranking(*pairs):
    """Build a ranking payload from (symbol, issuer) pairs in rank order."""
    return {
        "candidates": [
            {"rank": index + 1, "symbol": symbol, "issuer": issuer, "score": 100.0 - index}
            for index, (symbol, issuer) in enumerate(pairs)
        ]
    }


class ShareClassDedupeTest(unittest.TestCase):
    """One company must not occupy two slots in the basket."""

    def test_the_better_ranked_share_class_wins_and_the_basket_backfills(self):
        ranking = _ranking(
            ("GOOGL", "alphabet inc."),
            ("NVDA", "nvidia corporation"),
            ("GOOG", "alphabet inc."),
            ("AAPL", "apple inc."),
        )
        selected = _select_unconflicted_candidates(ranking, set(), set(), 3)
        self.assertEqual(selected, ["GOOGL", "NVDA", "AAPL"])

    def test_the_surviving_class_is_whichever_ranks_higher_not_a_fixed_ticker(self):
        # If GOOG ever out-ranks GOOGL, GOOG is the one that should be held.
        ranking = _ranking(
            ("GOOG", "alphabet inc."),
            ("GOOGL", "alphabet inc."),
            ("NVDA", "nvidia corporation"),
        )
        self.assertEqual(_select_unconflicted_candidates(ranking, set(), set(), 2), ["GOOG", "NVDA"])

    def test_deduping_can_be_switched_off(self):
        ranking = _ranking(("GOOGL", "alphabet inc."), ("GOOG", "alphabet inc."))
        selected = _select_unconflicted_candidates(
            ranking, set(), set(), 2, dedupe_share_classes=False
        )
        self.assertEqual(selected, ["GOOGL", "GOOG"])

    def test_a_ranking_without_issuers_still_selects(self):
        # State written before issuers were recorded must not break entry.
        legacy = {"candidates": [{"rank": 1, "symbol": "GOOGL"}, {"rank": 2, "symbol": "GOOG"}]}
        self.assertEqual(_select_unconflicted_candidates(legacy, set(), set(), 2), ["GOOGL", "GOOG"])

    def test_conflicts_and_duplicates_are_skipped_together(self):
        ranking = _ranking(
            ("GOOGL", "alphabet inc."),
            ("GOOG", "alphabet inc."),
            ("NVDA", "nvidia corporation"),
            ("AAPL", "apple inc."),
        )
        selected = _select_unconflicted_candidates(ranking, {"GOOGL"}, set(), 2)
        # GOOGL is held, so GOOG becomes the best available Alphabet class.
        self.assertEqual(selected, ["GOOG", "NVDA"])

    def test_too_few_distinct_issuers_raises(self):
        ranking = _ranking(("GOOGL", "alphabet inc."), ("GOOG", "alphabet inc."))
        with self.assertRaises(RuntimeError):
            _select_unconflicted_candidates(ranking, set(), set(), 2)
