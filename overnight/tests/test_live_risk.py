"""Live promotion: model parity, fail-closed entries and restart-safe exits."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from test_live import FakeBroker, config

from trading_rl.overnight.live import (
    EASTERN,
    RANKING_PIPELINE_VERSION,
    DailyArtifacts,
    StateStore,
    _exit_order_summary,
    _validate_args,
    enter_for_day,
    exit_position,
    parse_live_arguments,
    trend_vol_budget,
)
from trading_rl.overnight.live_config import load_live_settings
from trading_rl.overnight.live_risk import (
    RiskPriceProvider,
    configuration_fingerprint,
    prepare_live_risk,
    signal_is_current,
)
from trading_rl.overnight.risk_history import (
    historical_baskets,
    risk_signal,
    unit_return,
)
from trading_rl.overnight.strategies import LiquidityTrendVolConfig


def trend_config(**overrides):
    return replace(
        config(),
        **{
            "strategy_name": "liquidity-trend-vol",
            "capital_fraction": 1.0,
            **overrides,
        },
    )


def account(**overrides):
    return {
        "equity": "10000",
        "cash": "10000",
        "buying_power": "40000",
        "regt_buying_power": "20000",
        "multiplier": "4",
        "status": "ACTIVE",
        **overrides,
    }


def snapshot(settings, day=date(2026, 8, 24)):
    dates = pd.bdate_range(end=day, periods=130)
    observations = [
        {
            "entry_date": str(dates[i].date()),
            "exit_date": str(dates[i + 1].date()),
            "unscaled_return": 0.01 if i % 2 else -0.01,
            "symbols": ["A", "B"],
            "entry_prices": [100.0, 100.0],
            "exit_prices": [101.0, 101.0] if i % 2 else [99.0, 99.0],
        }
        for i in range(len(dates) - 21, len(dates) - 1)
    ]
    signal = risk_signal(
        dates, observations, np.arange(len(dates)) + 100, settings.risk_config
    )
    signal["spy_history"] = [
        {"date": str(dates[i].date()), "minute_open": float(i + 100)}
        for i in range(len(dates) - 101, len(dates) - 1)
    ]
    signal.update(
        configuration_sha256=configuration_fingerprint(settings), input_sha256="fixture"
    )
    return signal


class RiskHistoryTest(unittest.TestCase):
    def test_live_import_boundary_in_fresh_interpreter(self):
        script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(part in fullname for part in ('backtest', 'experiments', 'matplotlib')):
            raise AssertionError('unsafe live import: ' + fullname)
sys.meta_path.insert(0, Guard())
import trading_rl.overnight.live
import trading_rl.overnight.reconciliation_prices
import trading_rl.overnight.reconcile_live_sessions
"""
        subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True
        )

    def test_live_history_matches_simulator_exposure_and_short_session_cash(self):
        from test_strategies import fixture

        from trading_rl.overnight.backtest import run_backtest

        args = fixture()
        args["transaction_cost_bps"] = 0.0
        args["entry_session_mask"] = np.ones(len(args["dates"]), dtype=bool)
        args["entry_session_mask"][-10] = False
        _, summary = run_backtest(**args)
        today = len(args["dates"]) - 2
        dates = args["dates"][: today + 1]
        policy_config = LiquidityTrendVolConfig()
        rows, baskets = historical_baskets(
            dates,
            args["symbols"],
            args["dollar_volume"][: today + 1],
            args["entry_session_mask"][: today + 1],
            np.ones((today + 1, 3), dtype=bool),
            {},
            policy_config,
            top=2,
            ema_span=10,
            min_history_days=20,
            minimum_trading_days=100,
            liquidity_scheme="turnover_stability",
        )
        observations = [
            {
                "entry_date": str(dates[row].date()),
                "exit_date": str(dates[row + 1].date()),
                "unscaled_return": unit_return(
                    args["entry_prices"][row, basket],
                    args["morning_prices"][row + 1, basket],
                ),
            }
            for row, basket in zip(rows, baskets, strict=True)
        ]
        signal = risk_signal(
            dates, observations, args["spy_trend_marks"][: today + 1], policy_config
        )
        expected = summary["daily_portfolio"][-1]
        self.assertAlmostEqual(
            signal["target_exposure"], expected["exposure"], places=13
        )
        self.assertEqual(observations[-8]["unscaled_return"], 0.0)
        marks = args["spy_trend_marks"][: today + 1].copy()
        marks[-1] = 1e12
        unchanged = risk_signal(dates, observations, marks, policy_config)
        self.assertEqual(unchanged["target_exposure"], signal["target_exposure"])
        observations[-1]["unscaled_return"] += 0.10
        changed = risk_signal(dates, observations, marks, policy_config)
        self.assertLess(changed["target_exposure"], signal["target_exposure"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            risk_signal(dates, observations[:-1], marks, policy_config)
        marks[-2] = np.nan
        with self.assertRaisesRegex(ValueError, "SPY trend history"):
            risk_signal(dates, observations, marks, policy_config)

    def test_missing_prices_fail_and_splits_preserve_return_basis(self):
        with self.assertRaises(ValueError):
            unit_return([100, np.nan], [101, 99])
        with tempfile.TemporaryDirectory() as directory:
            settings = trend_config(
                risk_nbbo_path=Path(directory) / "absent.npz",
                risk_auctions_path=Path(directory) / "absent.npz",
            )
            provider = RiskPriceProvider(
                Mock(), settings, [], [], datetime.now(EASTERN)
            )
            provider.asks[(date(2026, 8, 21), "A")] = 200
            provider.splits = [
                {"symbol": "A", "ex_date": "2026-08-24", "old_rate": 1, "new_rate": 2}
            ]
            entry = provider.adjusted_entry("A", date(2026, 8, 21), date(2026, 8, 24))
            self.assertAlmostEqual(unit_return([entry], [101]), 0.01)

    def test_stale_spy_mark_fails_instead_of_using_any_bar_that_day(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = trend_config(
                risk_nbbo_path=Path(directory) / "absent.npz",
                risk_auctions_path=Path(directory) / "absent.npz",
            )
            broker = Mock(data_url="https://data.alpaca.markets/v2")
            broker._request.return_value = {
                "bars": {
                    "SPY": [
                        {"t": "2026-08-20T13:30:00Z", "o": 100},
                        {"t": "2026-08-21T19:59:00Z", "o": 101},
                    ]
                }
            }
            provider = RiskPriceProvider(
                broker, settings, [], [], datetime(2026, 8, 24, 14, tzinfo=EASTERN)
            )
            with self.assertRaisesRegex(ValueError, "stale SPY"):
                provider.spy_marks(
                    pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24"]), 2
                )

    def test_preopen_bootstrap_fails_before_network_or_any_order(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "completed opening auction"):
            prepare_live_risk(
                client,
                trend_config(),
                date(2026, 8, 24),
                {},
                Path("/unused"),
                datetime(2026, 8, 24, 6, tzinfo=EASTERN),
            )
        self.assertEqual(client.mock_calls, [])

    def test_snapshot_rejects_date_config_or_incomplete_changes(self):
        settings = trend_config()
        signal = snapshot(settings)
        self.assertTrue(signal_is_current(signal, settings, date(2026, 8, 24)))
        self.assertFalse(signal_is_current(signal, settings, date(2026, 8, 25)))
        self.assertFalse(
            signal_is_current(signal, replace(settings, top=3), date(2026, 8, 24))
        )
        for invalid in (None, "2", float("nan"), True):
            self.assertFalse(
                signal_is_current(
                    {**signal, "target_exposure": invalid}, settings, date(2026, 8, 24)
                )
            )
        signal["observations"].pop()
        self.assertFalse(signal_is_current(signal, settings, date(2026, 8, 24)))


class RiskSizingTest(unittest.TestCase):
    def size(
        self, broker_account=None, assets=None, positions=(), orders=(), settings=None
    ):
        return trend_vol_budget(
            broker_account or account(),
            settings or trend_config(),
            2,
            positions,
            orders,
            assets or [{"marginable": True}] * 2,
        )

    def test_equity_basis_buffer_and_overnight_cap_despite_four_times_buying_power(
        self,
    ):
        result = self.size(account(cash="2500"))
        self.assertEqual(result["budget"], 19600)
        self.assertEqual(result["effective_exposure"], 1.96)
        self.assertEqual(self.size(account(regt_buying_power="12000"))["budget"], 11760)
        self.assertEqual(self.size(settings=trend_config(capital=3000))["budget"], 5880)
        self.assertEqual(
            self.size(settings=trend_config(capital_fraction=0.5))["budget"], 9800
        )

    def test_holdings_orders_cash_and_nonmarginable_limits(self):
        self.assertEqual(
            self.size(
                positions=[{"market_value": "6000"}],
                orders=[{"side": "buy", "notional": "1000"}],
            )["budget"],
            12740,
        )
        self.assertEqual(
            self.size(account(multiplier="1", cash="4000"))["budget"], 3920
        )
        self.assertEqual(
            self.size(account(cash="5000"), assets=[{"marginable": False}])["budget"],
            4900,
        )
        self.assertEqual(
            self.size(
                assets=[{"marginable": True, "maintenance_margin_requirement": 100}]
            )["budget"],
            9800,
        )
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            self.size(account(trading_blocked=True))
        with self.assertRaisesRegex(RuntimeError, "safely value"):
            self.size(orders=[{"side": "buy", "qty": "10"}])
        with self.assertRaises(RuntimeError):
            self.size(account(regt_buying_power="0"))


class RiskLifecycleTest(unittest.TestCase):
    def test_defaults_and_legacy_yaml_compatibility(self):
        parser, args, _ = parse_live_arguments(["preview"])
        settings = _validate_args(parser, args)
        self.assertEqual(settings.strategy_name, "liquidity-trend-vol")
        self.assertEqual(settings.share_mode, "fractional")
        self.assertEqual(settings.risk_config, LiquidityTrendVolConfig())
        self.assertEqual(args.exit_time.isoformat(), "06:00:00")
        self.assertEqual(args.entry_time.isoformat(), "15:45:00")
        legacy = OmegaConf.to_container(load_live_settings(), resolve=True)
        legacy["strategy"].pop("name")
        legacy.pop("risk")
        for name in ("risk_minute_bars_dir", "risk_nbbo_path", "risk_auctions_path"):
            legacy["data"].pop(name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            OmegaConf.save(legacy, path)
            loaded = load_live_settings(path)
        self.assertEqual(loaded.strategy.name, "liquidity-fixed")

    def test_missing_risk_blocks_new_entries_but_saved_plan_resumes_and_exits(self):
        broker = FakeBroker()
        broker.current_positions = {}
        broker.account = Mock(return_value=account())
        broker.list_assets = Mock(
            return_value=[{"symbol": s, "marginable": True} for s in ("A", "B")]
        )
        settings = trend_config()
        day = date(2026, 8, 24)
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            ranking = {
                "trade_date": str(day),
                "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
                "candidates": [{"rank": 1, "symbol": "A"}, {"rank": 2, "symbol": "B"}],
            }
            store.save({"version": 1, "ranking": ranking})
            with self.assertRaisesRegex(RuntimeError, "complete risk signal"):
                enter_for_day(broker, store, settings, day, submit=True)
            self.assertEqual(broker.submissions, [])
            ranking["risk_signal"] = snapshot(settings)
            store.save({"version": 1, "ranking": ranking})
            planned = enter_for_day(
                broker, store, settings, day, submit=True, preflight_only=True
            )
            persisted = copy.deepcopy(planned)
            state = store.load()
            state["ranking"].pop("risk_signal")
            store.save(state)
            broker.account.side_effect = AssertionError(
                "must reuse the persisted budget"
            )
            entered = enter_for_day(broker, store, settings, day, submit=True)
            resumed = enter_for_day(broker, store, settings, day, submit=True)
            self.assertEqual(entered["budget"], persisted["budget"])
            self.assertEqual(resumed["risk_signal"], persisted["risk_signal"])
            self.assertEqual(len(broker.submissions), 2)
            broker.account.side_effect = None
            exited = exit_position(
                broker,
                store,
                settings,
                submit=True,
                now=datetime(2026, 8, 25, 6, tzinfo=EASTERN),
                wait_for_fill=False,
            )
            self.assertEqual(exited["status"], "closed")
            self.assertTrue(
                all(order["time_in_force"] == "day" for order in broker.submissions)
            )

    def test_old_positions_exit_and_old_summary_keeps_fixed_strategy(self):
        broker = FakeBroker()
        broker.current_positions = {"A": {"symbol": "A", "qty": ".75"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.json")
            store.save(
                {
                    "version": 1,
                    "position": {
                        "entry_date": "2026-08-24",
                        "exit_date": "2026-08-25",
                        "status": "open",
                        "symbols": ["A"],
                        "entry_orders": {"A": {}},
                        "share_mode": "fractional",
                    },
                }
            )
            artifacts = DailyArtifacts(root)
            path = artifacts.directory(date(2026, 8, 24)) / "summary.json"
            path.write_text(
                json.dumps(
                    {"configuration": {"top": 2}, "position": store.load()["position"]}
                )
            )
            exit_position(
                broker,
                store,
                trend_config(),
                submit=True,
                now=datetime(2026, 8, 25, 6, tzinfo=EASTERN),
                wait_for_fill=False,
            )
            artifacts.write_summary(date(2026, 8, 24), "exit", store, trend_config())
            self.assertEqual(
                json.loads(path.read_text())["strategy"], "liquidity-fixed"
            )
            artifacts.write_summary(date(2026, 8, 25), "exit", store, trend_config())
            current = json.loads((root / "2026-08-25" / "summary.json").read_text())
            self.assertEqual(current["strategy"], "liquidity-trend-vol")
            self.assertEqual(broker.submissions[0]["time_in_force"], "day")

    def test_broker_receipt_cutoff_fractional_day_and_missing_timestamp(self):
        position = {"exit_date": "2026-08-25", "share_mode": "fractional"}
        for timestamp, expected in (
            ("2026-08-25T10:00:00Z", True),
            ("2026-08-25T13:27:59Z", True),
            ("2026-08-25T13:28:00Z", False),
            ("2026-08-25T13:30:00Z", False),
            ("2026-08-24T10:00:00Z", False),
            (None, None),
        ):
            summary = _exit_order_summary(
                {"time_in_force": "day", "submitted_at": timestamp}, position
            )
            self.assertIs(summary["received_before_nasdaq_open_cutoff"], expected)
