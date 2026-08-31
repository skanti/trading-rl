import json
import logging
import tempfile
import threading
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from rich.console import Console

from live import (
    DEFAULT_EXCHANGES,
    EASTERN,
    RANKING_PIPELINE_VERSION,
    AlpacaClient,
    DailyArtifacts,
    DailyLogHandler,
    StateStore,
    StrategyConfig,
    _daily_bars_to_array,
    _completed_session_end,
    _exit_order_time_in_force,
    _merge_daily_arrays,
    _market_session_status,
    _print_entry_plan,
    _select_unconflicted_candidates,
    _validate_args,
    _validate_exit_clock,
    available_budget,
    build_parser,
    completed_liquidity_ranking,
    dollar_volume_shortlist,
    eligible_assets,
    enter_for_day,
    exit_position,
    refresh_daily_cache,
    seed_missing_daily_cache,
    whole_share_order_plan,
)


def config(top=2, fill_timeout_seconds=1.0, share_mode="fractional"):
    return StrategyConfig(
        top=top,
        ema_span=2,
        min_history_days=2,
        minimum_trading_days=2,
        lookback_calendar_days=90,
        daily_bars_dir=Path("/unused/daily-bars"),
        liquidity_shortlist=Path("/unused/most-liquid.txt"),
        shortlist_since=date(2022, 1, 1),
        shortlist_daily_top=50,
        shortlist_lookback_sessions=250,
        daily_overlap_days=30,
        feed="iex",
        quote_feed="iex",
        exchanges=DEFAULT_EXCHANGES,
        data_batch_size=100,
        data_workers=1,
        order_submit_workers=8,
        capital=None,
        capital_fraction=0.95,
        cash_buffer_fraction=0.02,
        fill_timeout_seconds=fill_timeout_seconds,
        poll_seconds=0.001,
        entry_preflight_seconds=10.0,
        share_mode=share_mode,
        quote_max_age_seconds=120.0,
    )


def bar(day, volume, vwap=100.0):
    return {"t": f"{day}T04:00:00Z", "v": volume, "vw": vwap, "c": vwap}


def full_bar(day, volume, vwap=100.0):
    return {
        **bar(day, volume, vwap),
        "o": vwap,
        "h": vwap,
        "l": vwap,
        "n": 100,
    }


class FakeBroker:
    def __init__(self):
        self.current_positions = {
            "HELD": {"symbol": "HELD", "qty": "1.5"},
        }
        self.orders = {}
        self.submissions = []
        self.latest_quote_rows = {}
        self.latest_quote_calls = []

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

    def latest_quotes(self, symbols, feed):
        self.latest_quote_calls.append((list(symbols), feed))
        return {symbol: self.latest_quote_rows[symbol] for symbol in symbols}

    def order_by_client_id(self, client_order_id):
        return self.orders.get(client_order_id)

    def submit_order(self, payload):
        self.submissions.append(dict(payload))
        symbol = payload["symbol"]
        if payload["side"] == "buy":
            quantity = str(payload.get("qty") or "0.95")
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
            "time_in_force": payload["time_in_force"],
            "submitted_at": "2026-08-24T19:55:00Z",
            "filled_at": "2026-08-24T19:55:00Z",
        }
        self.orders[payload["client_order_id"]] = order
        return order

    def order(self, order_id):
        return next(order for order in self.orders.values() if order["id"] == order_id)

    def cancel_order(self, order_id):
        raise AssertionError("filled fake orders must not be canceled")


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
            "time_in_force": payload["time_in_force"],
            "submitted_at": "2026-08-25T13:00:00Z",
            "filled_at": None,
        }
        self.orders[payload["client_order_id"]] = order
        return order


class FakeWorkingExitBroker(FakeQueuedBroker):
    def __init__(self):
        super().__init__()
        self.cancellations = []

    def cancel_order(self, order_id):
        self.cancellations.append(order_id)
        raise AssertionError(
            "a working exit order must not be canceled on the fill timeout"
        )


class FakeConcurrentBroker(FakeBroker):
    def __init__(self):
        super().__init__()
        self.active_submissions = 0
        self.max_active_submissions = 0
        self._activity_lock = threading.Lock()
        self._submission_lock = threading.Lock()

    def submit_order(self, payload):
        with self._activity_lock:
            self.active_submissions += 1
            self.max_active_submissions = max(
                self.max_active_submissions, self.active_submissions
            )
        try:
            time.sleep(0.01)
            with self._submission_lock:
                return super().submit_order(payload)
        finally:
            with self._activity_lock:
                self.active_submissions -= 1


class FakeClockBroker:
    def __init__(self, timestamp, is_open):
        self.timestamp = timestamp
        self.is_open = is_open

    def clock(self):
        return {"timestamp": self.timestamp, "is_open": self.is_open}


class FakeDailyBarsClient:
    def __init__(self, bars):
        self.bars = bars
        self.adjustments = []

    def historical_daily_bars(self, symbols, start, end, feed, adjustment="raw"):
        self.adjustments.append(adjustment)
        return {symbol: list(self.bars.get(symbol, [])) for symbol in symbols}


class FakeBatchOmissionDailyBarsClient(FakeDailyBarsClient):
    def __init__(self, bars, omitted_symbol):
        super().__init__(bars)
        self.omitted_symbol = omitted_symbol
        self.requests = []

    def historical_daily_bars(self, symbols, start, end, feed, adjustment="raw"):
        self.adjustments.append(adjustment)
        self.requests.append((tuple(symbols), start))
        return {
            symbol: (
                []
                if len(symbols) > 1
                and start == date(2016, 1, 1)
                and symbol == self.omitted_symbol
                else list(self.bars.get(symbol, []))
            )
            for symbol in symbols
        }


class LiveOvernightLiquidityTest(unittest.TestCase):
    def test_market_session_status_uses_alpaca_calendar(self):
        class CalendarBroker:
            def __init__(self):
                self.requests = []

            def calendar(self, start, end):
                self.requests.append((start, end))
                return [
                    {"date": "2026-08-31"},
                    {"date": "2026-09-01"},
                ]

        broker = CalendarBroker()

        sunday = _market_session_status(broker, date(2026, 8, 30))
        monday = _market_session_status(broker, date(2026, 8, 31))

        self.assertEqual(sunday, (False, date(2026, 8, 31)))
        self.assertEqual(monday, (True, date(2026, 9, 1)))
        self.assertEqual(
            broker.requests,
            [
                (date(2026, 8, 30), date(2026, 9, 9)),
                (date(2026, 8, 31), date(2026, 9, 10)),
            ],
        )

    def test_whole_share_preview_prints_per_symbol_and_total_sizing(self):
        console = Console(record=True, width=140, color_system=None)
        _print_entry_plan(
            {
                "share_mode": "whole",
                "symbols": ["A", "B", "C"],
                "budget": 1_200.0,
                "per_symbol_notional": 400.0,
                "sizing_prices": {"A": 120.0, "B": 300.0, "C": 700.0},
                "target_quantities": {"A": 3, "B": 1, "C": 0},
                "estimated_deployed_notional": 660.0,
                "entry_orders": {},
                "exit_date": "2026-08-25",
                "status": "planned",
            },
            submit=False,
            console=console,
        )

        output = console.export_text()
        self.assertIn("Est. value", output)
        self.assertIn("$360.00", output)
        self.assertIn("$660.00", output)
        self.assertIn("2 orders / 3 selected; 1 skipped", output)
        self.assertIn("55.0% of budget", output)

    def test_fractional_preview_prints_total_sizing(self):
        console = Console(record=True, width=120, color_system=None)
        _print_entry_plan(
            {
                "share_mode": "fractional",
                "symbols": ["A", "B"],
                "budget": 1_000.0,
                "per_symbol_notional": 500.0,
                "entry_orders": {},
                "exit_date": "2026-08-25",
                "status": "planned",
            },
            submit=False,
            console=console,
        )

        output = console.export_text()
        self.assertIn("$1,000.00", output)
        self.assertIn("2 orders / 2 selected", output)
        self.assertIn("100.0% of budget", output)

    def test_exit_time_accepts_0900(self):
        parser = build_parser()
        args = parser.parse_args(["exit", "--exit-time", "09:00"])

        parsed = _validate_args(parser, args)

        self.assertIsInstance(parsed, StrategyConfig)
        self.assertEqual(args.ranking_time.strftime("%H:%M"), "14:00")
        self.assertEqual(args.share_mode, "whole")
        self.assertEqual(args.feed, "sip")
        self.assertEqual(args.quote_feed, "iex")
        self.assertEqual(args.capital_fraction, 1.0)
        self.assertEqual(args.quote_max_age_seconds, 120.0)
        self.assertEqual(args.entry_preflight_seconds, 10.0)
        self.assertEqual(args.order_submit_workers, 8)
        self.assertEqual(args.shortlist_since, date(2022, 1, 1))
        self.assertEqual(args.shortlist_daily_top, 50)
        self.assertEqual(args.daily_overlap_days, 30)
        self.assertEqual(args.exchanges, "NASDAQ")
        self.assertEqual(parsed.capital_fraction, 1.0)
        self.assertEqual(parsed.feed, "sip")
        self.assertEqual(parsed.quote_feed, "iex")
        self.assertEqual(parsed.quote_max_age_seconds, 120.0)
        self.assertEqual(parsed.entry_preflight_seconds, 10.0)
        self.assertEqual(parsed.order_submit_workers, 8)
        self.assertEqual(parsed.shortlist_since, date(2022, 1, 1))
        self.assertEqual(parsed.shortlist_daily_top, 50)
        self.assertEqual(parsed.daily_overlap_days, 30)
        self.assertEqual(parsed.exchanges, frozenset({"NASDAQ"}))
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
            record = logging.LogRecord(
                "test", logging.INFO, __file__, 1, "rank complete", (), None
            )
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

    def test_ranking_falls_back_to_close_when_vwap_is_zero(self):
        sessions = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]
        bars = {
            "CLOSE": [
                {**bar("2026-08-20", 100, 0.0), "c": 100.0},
                {**bar("2026-08-21", 100, 0.0), "c": 100.0},
            ],
            "VWAP": [bar("2026-08-20", 100, 10.0), bar("2026-08-21", 100, 10.0)],
        }

        ranking = completed_liquidity_ranking(
            bars, sessions, sessions[-1], ema_span=2, min_history_days=2
        )

        self.assertEqual([symbol for symbol, _, _ in ranking], ["CLOSE", "VWAP"])

    def test_daily_cache_refresh_uses_split_adjustment_and_appends_exact_overlap(self):
        base_bars = [full_bar("2026-08-20", 100), full_bar("2026-08-21", 110)]
        update_bars = [full_bar("2026-08-21", 110), full_bar("2026-08-24", 120)]
        client = FakeDailyBarsClient({"AAPL": update_bars})
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(bars_dir / "AAPL.npy", _daily_bars_to_array(base_bars, "AAPL"))

            diagnostics = refresh_daily_cache(
                client,
                bars_dir,
                ["AAPL"],
                date(2026, 8, 24),
                _completed_session_end(date(2026, 8, 24)),
                "sip",
                batch_size=100,
                workers=1,
                overlap_days=30,
            )
            saved = np.load(bars_dir / "AAPL.npy")

        self.assertEqual(len(saved), 3)
        self.assertEqual(client.adjustments, ["split"])
        self.assertEqual(diagnostics["updated_files"], 1)
        self.assertEqual(diagnostics["full_refreshes"], 0)

    def test_newly_eligible_company_is_seeded_with_split_adjusted_history(self):
        client = FakeDailyBarsClient(
            {"NEW": [full_bar("2026-08-21", 100), full_bar("2026-08-24", 110)]}
        )
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            diagnostics = seed_missing_daily_cache(
                client,
                bars_dir,
                ["NEW"],
                date(2022, 1, 1),
                _completed_session_end(date(2026, 8, 24)),
                "sip",
                batch_size=100,
                workers=1,
            )
            saved = np.load(bars_dir / "NEW.npy")

        self.assertEqual(saved.shape, (2, 8))
        self.assertEqual(client.adjustments, ["split"])
        self.assertEqual(diagnostics["new_symbols_requested"], 1)
        self.assertEqual(diagnostics["new_symbols_added"], 1)

    def test_daily_cache_refresh_fails_closed_when_previous_session_is_missing(self):
        client = FakeDailyBarsClient({"AAPL": [full_bar("2026-08-21", 110)]})
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(
                bars_dir / "AAPL.npy",
                _daily_bars_to_array([full_bar("2026-08-21", 110)], "AAPL"),
            )
            with self.assertRaisesRegex(
                RuntimeError, "expected completed session 2026-08-24"
            ):
                refresh_daily_cache(
                    client,
                    bars_dir,
                    ["AAPL"],
                    date(2026, 8, 24),
                    _completed_session_end(date(2026, 8, 24)),
                    "sip",
                    batch_size=100,
                    workers=1,
                    overlap_days=30,
                )

    def test_full_refresh_retries_a_symbol_omitted_from_a_batch(self):
        bars = {
            symbol: [
                full_bar("2026-08-21", 999),
                full_bar("2026-08-24", 120),
            ]
            for symbol in ("AAPL", "APXT")
        }
        client = FakeBatchOmissionDailyBarsClient(bars, "APXT")
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            (bars_dir / "_download_manifest.json").write_text(
                json.dumps({"since": "2016-01-01T00:00:00+00:00"})
            )
            for symbol in bars:
                np.save(
                    bars_dir / f"{symbol}.npy",
                    _daily_bars_to_array([full_bar("2026-08-21", 100)], symbol),
                )

            diagnostics = refresh_daily_cache(
                client,
                bars_dir,
                list(bars),
                date(2026, 8, 24),
                _completed_session_end(date(2026, 8, 24)),
                "sip",
                batch_size=100,
                workers=1,
                overlap_days=30,
            )

        self.assertEqual(diagnostics["full_refreshes"], 2)
        self.assertIn((("APXT",), date(2016, 1, 1)), client.requests)

    def test_dollar_volume_shortlist_uses_daily_union_and_excludes_int32_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(
                bars_dir / "A.npy",
                _daily_bars_to_array(
                    [full_bar("2026-08-20", 1_000), full_bar("2026-08-21", 10)],
                    "A",
                ),
            )
            np.save(
                bars_dir / "B.npy",
                _daily_bars_to_array(
                    [full_bar("2026-08-20", 10), full_bar("2026-08-21", 2_000)],
                    "B",
                ),
            )
            np.save(
                bars_dir / "OVERFLOW.npy",
                _daily_bars_to_array(
                    [full_bar("2026-08-20", 1_000_000, 3_000_000.0)],
                    "OVERFLOW",
                ),
            )

            symbols, sessions, excluded = dollar_volume_shortlist(
                bars_dir, date(2022, 1, 1), top=1
            )

        self.assertEqual(symbols, ["A", "B"])
        self.assertEqual(sessions, 2)
        self.assertEqual(excluded, ["OVERFLOW"])

    def test_dollar_volume_shortlist_falls_back_to_close_for_zero_vwap(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            fallback = _daily_bars_to_array(
                [full_bar("2026-08-20", 1_000, 100.0)], "CLOSE"
            )
            fallback[:, 7] = 0
            np.save(bars_dir / "CLOSE.npy", fallback)
            np.save(
                bars_dir / "VWAP.npy",
                _daily_bars_to_array([full_bar("2026-08-20", 1_000, 10.0)], "VWAP"),
            )

            symbols, _, _ = dollar_volume_shortlist(bars_dir, date(2022, 1, 1), top=1)

        self.assertEqual(symbols, ["CLOSE"])

    def test_daily_encoder_stores_the_missing_vwap_sentinel_not_the_close(self):
        """A zero-volume session reports vw=0. Storing the close instead would make
        this writer disagree with scripts/download_bars.py byte-for-byte, and the
        overlap check would then re-download those symbols in full on every rank."""
        no_trades = {
            "t": "2026-08-20T04:00:00Z",
            "o": 19.0,
            "h": 19.0,
            "l": 19.0,
            "c": 19.0,
            "v": 0,
            "n": 0,
            "vw": 0,
        }
        absent_vwap = {**no_trades, "t": "2026-08-21T04:00:00Z", "vw": None}

        array = _daily_bars_to_array([no_trades, absent_vwap], "QUIET")

        self.assertEqual(array[0, 4], 19_000)
        self.assertEqual(array[0, 7], 0)
        self.assertEqual(array[1, 7], 0)

    def test_daily_encoder_matches_a_traded_sessions_reported_vwap(self):
        array = _daily_bars_to_array([full_bar("2026-08-20", 1_000, 12.345)], "TRADED")

        self.assertEqual(array[0, 7], 12_345)

    def test_daily_encoder_rejects_a_negative_vwap(self):
        with self.assertRaises(ValueError):
            _daily_bars_to_array(
                [{**full_bar("2026-08-20", 1_000, 10.0), "vw": -1.0}], "BAD"
            )

    def test_refresh_encoding_round_trips_without_a_spurious_full_refresh(self):
        """The sentinel must survive encode -> merge, or the symbol is flagged as
        revised and re-downloaded from the manifest epoch on every rank."""
        bars = [
            full_bar("2026-08-20", 1_000, 10.0),
            {**full_bar("2026-08-21", 0, 10.0), "v": 0, "n": 0, "vw": 0},
        ]
        stored = _daily_bars_to_array(bars, "QUIET")

        merged = _merge_daily_arrays(stored, _daily_bars_to_array(bars, "QUIET"))

        self.assertIsNotNone(merged)
        np.testing.assert_array_equal(merged, stored)

    def test_dollar_volume_shortlist_lookback_drops_a_stale_past_leader(self):
        """FADED led on the oldest session only. Unioning every session keeps it
        forever; a trailing window retires it once it stops leading."""
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(
                bars_dir / "FADED.npy",
                _daily_bars_to_array(
                    [
                        full_bar("2026-08-19", 9_000),
                        full_bar("2026-08-20", 1),
                        full_bar("2026-08-21", 1),
                    ],
                    "FADED",
                ),
            )
            np.save(
                bars_dir / "CURRENT.npy",
                _daily_bars_to_array(
                    [
                        full_bar("2026-08-19", 10),
                        full_bar("2026-08-20", 5_000),
                        full_bar("2026-08-21", 6_000),
                    ],
                    "CURRENT",
                ),
            )

            everything, all_sessions, _ = dollar_volume_shortlist(
                bars_dir, date(2022, 1, 1), top=1
            )
            windowed, windowed_sessions, _ = dollar_volume_shortlist(
                bars_dir, date(2022, 1, 1), top=1, lookback_sessions=2
            )

        self.assertEqual(everything, ["CURRENT", "FADED"])
        self.assertEqual(all_sessions, 3)
        self.assertEqual(windowed, ["CURRENT"])
        self.assertEqual(windowed_sessions, 2)

    def test_dollar_volume_shortlist_lookback_beyond_history_keeps_every_session(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(
                bars_dir / "A.npy",
                _daily_bars_to_array(
                    [full_bar("2026-08-20", 1_000), full_bar("2026-08-21", 10)], "A"
                ),
            )
            np.save(
                bars_dir / "B.npy",
                _daily_bars_to_array(
                    [full_bar("2026-08-20", 10), full_bar("2026-08-21", 2_000)], "B"
                ),
            )

            symbols, sessions, _ = dollar_volume_shortlist(
                bars_dir, date(2022, 1, 1), top=1, lookback_sessions=500
            )

        self.assertEqual(symbols, ["A", "B"])
        self.assertEqual(sessions, 2)

    def test_dollar_volume_shortlist_rejects_a_non_positive_lookback(self):
        with tempfile.TemporaryDirectory() as directory:
            bars_dir = Path(directory)
            np.save(
                bars_dir / "A.npy",
                _daily_bars_to_array([full_bar("2026-08-20", 1_000)], "A"),
            )
            with self.assertRaises(ValueError):
                dollar_volume_shortlist(
                    bars_dir, date(2022, 1, 1), top=1, lookback_sessions=0
                )

    def test_daily_array_merge_detects_historical_revision(self):
        base = _daily_bars_to_array(
            [full_bar("2026-08-20", 100), full_bar("2026-08-21", 110)], "A"
        )
        revised = _daily_bars_to_array(
            [full_bar("2026-08-21", 999), full_bar("2026-08-24", 120)], "A"
        )

        self.assertIsNone(_merge_daily_arrays(base, revised))

    def test_only_fractionable_exchange_listed_assets_are_eligible(self):
        common = {
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "class": "us_equity",
        }
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

    def test_whole_share_universe_does_not_require_fractionable_assets(self):
        assets = [
            {
                "symbol": "WHOLE",
                "status": "active",
                "tradable": True,
                "fractionable": False,
                "class": "us_equity",
                "exchange": "NASDAQ",
            }
        ]

        self.assertEqual(
            eligible_assets(assets, DEFAULT_EXCHANGES, require_fractionable=False),
            ["WHOLE"],
        )

    def test_latest_quotes_uses_market_data_credentials_and_requested_feed(self):
        client = AlpacaClient("key", "secret")
        calls = []

        def request(method, base, path, **kwargs):
            calls.append((method, base, path, kwargs))
            return {"quotes": {"AAPL": {"ap": 230.0, "t": "2026-08-24T19:59:00Z"}}}

        client._request = request
        quotes = client.latest_quotes(["AAPL"], "sip")

        self.assertEqual(quotes["AAPL"]["ap"], 230.0)
        self.assertEqual(calls[0][1], "https://data.alpaca.markets/v2")
        self.assertEqual(calls[0][2], "stocks/quotes/latest")
        self.assertEqual(calls[0][3]["params"], {"symbols": "AAPL", "feed": "sip"})
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
        client.historical_daily_bars(["AAPL"], date(2026, 2, 25), completed_end, "sip")

        self.assertEqual(completed_end.isoformat(), "2026-08-24T23:59:59-04:00")
        self.assertEqual(calls[0][3]["params"]["end"], completed_end.isoformat())
        self.assertTrue(calls[0][3]["data_credentials"])

    def test_budget_uses_cash_instead_of_margin_buying_power(self):
        budget = available_budget(
            {"cash": "1000", "buying_power": "4000", "equity": "1000"}, config(top=10)
        )
        self.assertEqual(budget, 950.0)

    def test_budget_does_not_treat_non_marginable_power_as_stock_cash(self):
        budget = available_budget(
            {
                "cash": "1000",
                "buying_power": "4000",
                "non_marginable_buying_power": "600",
            },
            config(top=10),
        )
        self.assertEqual(budget, 950.0)

    def test_budget_respects_regular_stock_buying_power(self):
        budget = available_budget(
            {
                "cash": "1000",
                "buying_power": "600",
                "non_marginable_buying_power": "1000",
            },
            config(top=10),
        )
        self.assertEqual(budget, 600.0)

    def test_whole_share_plan_matches_simulator_rounding_and_leaves_idle_cash(self):
        quoted_at = "2026-08-24T19:59:00Z"
        plan = whole_share_order_plan(
            ["A", "B", "C"],
            {
                "A": {"ap": 120.0, "t": quoted_at},
                "B": {"ap": 300.0, "t": quoted_at},
                "C": {"ap": 700.0, "t": quoted_at},
            },
            1_200.0,
            datetime(2026, 8, 24, 15, 59, tzinfo=EASTERN),
            60.0,
        )

        self.assertEqual(plan["target_quantities"], {"A": 3, "B": 1, "C": 0})
        self.assertEqual(plan["skipped_symbols"], ["C"])
        self.assertEqual(plan["estimated_deployed_notional"], 660.0)

    def test_whole_share_plan_rejects_stale_quotes(self):
        with self.assertRaisesRegex(RuntimeError, "maximum is 60.0s"):
            whole_share_order_plan(
                ["A"],
                {"A": {"ap": 100.0, "t": "2026-08-24T19:57:00Z"}},
                1_000.0,
                datetime(2026, 8, 24, 15, 59, tzinfo=EASTERN),
                60.0,
            )

    def test_whole_share_entry_submits_integer_qty_and_is_restart_idempotent(self):
        broker = FakeBroker()
        broker.latest_quote_rows = {
            "A": {"ap": 120.0, "t": "2026-08-24T19:59:00Z"},
            "B": {"ap": 600.0, "t": "2026-08-24T19:59:00Z"},
        }
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["ranking"] = {
                "trade_date": "2026-08-24",
                "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
                "candidates": [
                    {"rank": 1, "symbol": "A"},
                    {"rank": 2, "symbol": "B"},
                ],
            }
            store.save(state)

            first = enter_for_day(
                broker,
                store,
                config(share_mode="whole"),
                date(2026, 8, 24),
                submit=True,
                now=datetime(2026, 8, 24, 15, 59, tzinfo=EASTERN),
            )
            second = enter_for_day(
                broker,
                store,
                config(share_mode="fractional"),
                date(2026, 8, 24),
                submit=True,
                now=datetime(2026, 8, 24, 15, 59, 1, tzinfo=EASTERN),
            )

        self.assertEqual(first["share_mode"], "whole")
        self.assertEqual(first["quote_feed"], "iex")
        self.assertEqual(first["target_quantities"], {"A": 3, "B": 0})
        self.assertEqual(first["skipped_symbols"], ["B"])
        self.assertEqual(first["estimated_deployed_notional"], 360.0)
        self.assertEqual(first["filled_symbols"], ["A"])
        self.assertEqual(broker.latest_quote_calls, [(["A", "B"], "iex")])
        self.assertEqual(second["status"], "open")
        self.assertEqual(len(broker.submissions), 1)
        self.assertEqual(broker.submissions[0]["qty"], "3")
        self.assertNotIn("notional", broker.submissions[0])
        self.assertEqual(
            second["entry_orders"]["A"]["dispatch_started_at"],
            first["entry_orders"]["A"]["dispatch_started_at"],
        )
        self.assertTrue(second["entry_orders"]["A"]["recovered_by_client_order_id"])

    def test_preflight_persists_sizing_then_dispatches_orders_concurrently(self):
        broker = FakeConcurrentBroker()
        symbols = [f"S{index}" for index in range(8)]
        broker.latest_quote_rows = {
            symbol: {"ap": 10.0, "t": "2026-08-24T19:58:50Z"} for symbol in symbols
        }
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["ranking"] = {
                "trade_date": "2026-08-24",
                "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
                "candidates": [
                    {"rank": index + 1, "symbol": symbol}
                    for index, symbol in enumerate(symbols)
                ],
            }
            store.save(state)
            target = datetime(2026, 8, 24, 15, 59, tzinfo=EASTERN)

            prepared = enter_for_day(
                broker,
                store,
                config(top=8, share_mode="whole"),
                date(2026, 8, 24),
                submit=True,
                now=target - timedelta(seconds=10),
                preflight_only=True,
                dispatch_target=target,
            )

            self.assertEqual(prepared["status"], "planned")
            self.assertEqual(prepared["order_submit_workers"], 8)
            self.assertEqual(broker.submissions, [])
            self.assertEqual(len(broker.latest_quote_calls), 1)

            entered = enter_for_day(
                broker,
                store,
                config(top=8, share_mode="whole"),
                date(2026, 8, 24),
                submit=True,
                now=target,
                dispatch_target=target,
            )

        self.assertEqual(entered["status"], "open")
        self.assertEqual(len(broker.submissions), 8)
        self.assertEqual(len(broker.latest_quote_calls), 1)
        self.assertGreater(broker.max_active_submissions, 1)
        self.assertTrue(
            all(
                "dispatch_duration_ms" in order
                for order in entered["entry_orders"].values()
            )
        )

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

            first = enter_for_day(
                broker, store, config(), date(2026, 8, 24), submit=True
            )
            second = enter_for_day(
                broker, store, config(), date(2026, 8, 24), submit=True
            )

        self.assertEqual(first["symbols"], ["A", "B"])
        self.assertEqual(second["status"], "open")
        self.assertEqual(len(broker.submissions), 2)
        self.assertTrue(all(order["side"] == "buy" for order in broker.submissions))
        self.assertEqual(first["entry_account_snapshot"]["cash"], "1000")
        self.assertEqual(first["entry_account_snapshot"]["buying_power"], "4000")

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

    def test_whole_share_opg_exit_stays_queued_and_is_reconciled_after_open(self):
        broker = FakeQueuedBroker()
        broker.current_positions.clear()
        broker.current_positions["A"] = {"symbol": "A", "qty": "1"}
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "open",
                "share_mode": "whole",
                "symbols": ["A"],
                "filled_symbols": ["A"],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            queued = exit_position(
                broker,
                store,
                config(top=1, share_mode="whole"),
                submit=True,
                now=datetime(2026, 8, 25, 9, 0, tzinfo=EASTERN),
                wait_for_fill=False,
            )
            queued_order = broker.orders["olq-20260824-x-A"]
            self.assertEqual(queued["status"], "exit_queued")
            self.assertEqual(queued_order["status"], "accepted")
            self.assertEqual(queued["exit_orders"]["A"]["time_in_force"], "opg")
            self.assertIn("A", broker.current_positions)
            self.assertEqual(broker.submissions[0]["time_in_force"], "opg")

            queued_order["status"] = "filled"
            queued_order["filled_qty"] = "1"
            broker.current_positions.pop("A")
            closed = exit_position(
                broker,
                store,
                config(top=1, share_mode="whole"),
                submit=True,
                now=datetime(2026, 8, 25, 9, 30, tzinfo=EASTERN),
                wait_for_fill=True,
            )

        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["exit_orders"]["A"]["status"], "filled")
        self.assertEqual(closed["exit_account_snapshot"]["equity"], "1000")
        self.assertIn("exit_account_snapshot_at", closed)
        self.assertEqual(len(broker.submissions), 1)

    def test_fractional_exit_remains_a_day_order(self):
        position = {"share_mode": "fractional"}
        preopen = datetime(2026, 8, 25, 9, 0, tzinfo=EASTERN)

        self.assertEqual(_exit_order_time_in_force(position, preopen), "day")

    def test_whole_share_exit_uses_day_only_as_post_open_recovery(self):
        position = {"share_mode": "whole"}
        preopen = datetime(2026, 8, 25, 9, 0, tzinfo=EASTERN)
        after_open = datetime(2026, 8, 25, 9, 30, tzinfo=EASTERN)

        self.assertEqual(_exit_order_time_in_force(position, preopen), "opg")
        self.assertEqual(_exit_order_time_in_force(position, after_open), "day")
        with self.assertRaisesRegex(RuntimeError, "09:28 ET OPG cutoff"):
            _exit_order_time_in_force(
                position, datetime(2026, 8, 25, 9, 29, tzinfo=EASTERN)
            )

    def test_cancelled_opg_uses_a_new_day_order_after_open(self):
        broker = FakeBroker()
        broker.current_positions["A"] = {"symbol": "A", "qty": "1"}
        broker.orders["olq-20260824-x-A"] = {
            "id": "canceled-opg",
            "client_order_id": "olq-20260824-x-A",
            "symbol": "A",
            "side": "sell",
            "status": "canceled",
            "qty": "1",
            "filled_qty": "0",
            "time_in_force": "opg",
        }
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = store.load()
            state["position"] = {
                "entry_date": "2026-08-24",
                "exit_date": "2026-08-25",
                "status": "exit_incomplete",
                "share_mode": "whole",
                "symbols": ["A"],
                "filled_symbols": ["A"],
                "entry_orders": {},
                "exit_orders": {},
            }
            store.save(state)

            result = exit_position(
                broker,
                store,
                config(top=1, share_mode="whole"),
                submit=True,
                now=datetime(2026, 8, 25, 9, 30, tzinfo=EASTERN),
            )

        self.assertEqual(result["status"], "closed")
        self.assertEqual(broker.submissions[0]["client_order_id"], "olq-20260824-x-A-2")
        self.assertEqual(broker.submissions[0]["time_in_force"], "day")

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

    def test_working_exit_survives_the_fill_timeout_without_replacement(self):
        broker = FakeWorkingExitBroker()
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

            first = exit_position(
                broker,
                store,
                config(top=1, fill_timeout_seconds=0.005),
                submit=True,
            )
            second = exit_position(
                broker,
                store,
                config(top=1, fill_timeout_seconds=0.005),
                submit=True,
            )

        self.assertEqual(first["status"], "exiting")
        self.assertEqual(second["status"], "exiting")
        self.assertEqual(len(broker.submissions), 1)
        self.assertEqual(broker.cancellations, [])

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
            {
                "rank": index + 1,
                "symbol": symbol,
                "issuer": issuer,
                "score": 100.0 - index,
            }
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
        self.assertEqual(
            _select_unconflicted_candidates(ranking, set(), set(), 2), ["GOOG", "NVDA"]
        )

    def test_deduping_can_be_switched_off(self):
        ranking = _ranking(("GOOGL", "alphabet inc."), ("GOOG", "alphabet inc."))
        selected = _select_unconflicted_candidates(
            ranking, set(), set(), 2, dedupe_share_classes=False
        )
        self.assertEqual(selected, ["GOOGL", "GOOG"])

    def test_a_ranking_without_issuers_still_selects(self):
        # State written before issuers were recorded must not break entry.
        legacy = {
            "candidates": [
                {"rank": 1, "symbol": "GOOGL"},
                {"rank": 2, "symbol": "GOOG"},
            ]
        }
        self.assertEqual(
            _select_unconflicted_candidates(legacy, set(), set(), 2), ["GOOGL", "GOOG"]
        )

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
