"""Promoted focus allocation, cash lifecycle, and immutable decision replay."""

import copy
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from test_live import FakeBroker, config
from test_live_risk import account

from trading_rl.overnight.decision_replay import replay_entry_decision
from trading_rl.overnight.live import (
    EASTERN,
    RANKING_PIPELINE_VERSION,
    StateStore,
    _validate_args,
    enter_for_day,
    exit_position,
    parse_live_arguments,
)
from trading_rl.overnight.live_risk import configuration_fingerprint, signal_is_current
from trading_rl.overnight.momentum import (
    LiquidityMomentumFocusConfig,
    allocation_snapshot,
)
from trading_rl.overnight.reconcile_live_sessions import (
    print_overview,
    reconcile_execution,
    select_reporting_benchmark,
)
from trading_rl.overnight.risk_history import (
    historical_baskets,
    risk_signal,
    unit_return,
)


class MomentumLiveTest(unittest.TestCase):
    def setup_session(self, cash=False, share_mode="fractional", count=4):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = StateStore(Path(directory.name) / "state.json")
        settings = replace(
            config(top=6, share_mode=share_mode),
            strategy_name="liquidity-momentum-focus",
            risk_config=LiquidityMomentumFocusConfig(allocation_count=count),
            capital_fraction=1.0,
        )
        day = date(2026, 8, 24)
        dates = pd.bdate_range(end=day, periods=150)
        names = list("ABCDEF")

        def allocation(row):
            return allocation_snapshot(
                names,
                [100.0] * 6,
                list(range(101, 107)),
                dates[row - 11 : row],
                dates[row].date(),
                settings.risk_config,
            )

        observations = []
        for row in range(129, 149):
            item = allocation(row)
            entries, exits = [100.0] * count, [101.0 if row % 2 else 99.0] * count
            observations.append(
                {
                    "entry_date": str(dates[row].date()),
                    "exit_date": str(dates[row + 1].date()),
                    "symbols": item["symbols"],
                    "entry_prices": entries,
                    "exit_prices": exits,
                    "unscaled_return": unit_return(entries, exits),
                    "allocation": item,
                    "entry_allowed": True,
                }
            )
        spy = np.arange(len(dates), dtype=float) + 100
        if cash:
            spy[-2] = 1.0
        risk = risk_signal(dates, observations, spy, settings.risk_config)
        risk.update(
            configuration_sha256=configuration_fingerprint(settings),
            input_sha256="fixture",
            spy_history=[
                {"date": str(dates[i].date()), "price": spy[i]} for i in range(49, 149)
            ],
        )
        ranking = {
            "trade_date": str(day),
            "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
            "risk_signal": risk,
            "allocation": allocation(149),
            "candidates": [
                {"symbol": name, "rank": i + 1} for i, name in enumerate(names)
            ],
        }
        store.save({"version": 1, "ranking": ranking})
        broker = FakeBroker()
        broker.current_positions = {}
        broker.account = Mock(return_value=account())
        broker.list_assets = Mock(
            return_value=[{"symbol": name, "marginable": True} for name in names]
        )
        broker.latest_quote_rows = {
            name: {"ap": 100.0, "bp": 99.0, "t": "2026-08-24T19:44:59Z"}
            for name in names
        }
        return store, settings, day, broker

    def test_four_stock_sizing_replay_resume_and_exit(self):
        for mode in ("fractional", "whole"):
            with self.subTest(mode=mode):
                store, settings, day, broker = self.setup_session(share_mode=mode)
                plan = enter_for_day(
                    broker,
                    store,
                    settings,
                    day,
                    True,
                    now=datetime(2026, 8, 24, 15, 45, tzinfo=EASTERN),
                    preflight_only=True,
                )
                self.assertEqual(plan["symbols"], list("CDEF"))
                self.assertEqual(plan["per_symbol_notional"], plan["budget"] / 4)
                replay = replay_entry_decision({"position": plan})
                self.assertEqual(replay["status"], "complete", replay)
                self.assertEqual(len(replay["orders"]), 4)
                state = store.load()
                state[
                    "ranking"
                ] = {}  # prepared plans survive restarts and ranking replacement
                store.save(state)
                # Changing the default after preflight must preserve the saved N=4 plan.
                settings = replace(settings, risk_config=LiquidityMomentumFocusConfig())
                entered = enter_for_day(broker, store, settings, day, True)
                self.assertEqual(len(broker.submissions), 4)
                self.assertEqual(
                    replay_entry_decision({"position": entered})["status"], "complete"
                )
                closed = exit_position(
                    broker,
                    store,
                    settings,
                    True,
                    now=datetime(2026, 8, 25, 6, tzinfo=EASTERN),
                    wait_for_fill=False,
                )
                self.assertEqual(closed["status"], "closed")

    def test_three_stock_default_and_old_ranking_invalidation(self):
        parser, args, _ = parse_live_arguments(["preview"])
        default = _validate_args(parser, args)
        self.assertEqual(default.top, 12)
        self.assertEqual(default.risk_config.allocation_count, 3)
        store, settings, day, broker = self.setup_session(count=3)
        plan = enter_for_day(broker, store, settings, day, False)
        self.assertEqual(plan["symbols"], list("DEF"))
        replay = replay_entry_decision({"position": plan})
        self.assertEqual(replay["status"], "complete", replay)
        self.assertEqual(len(replay["orders"]), 3)
        self.assertLessEqual(plan["per_symbol_notional"] * 3, plan["budget"])
        self.assertLess(plan["budget"] - plan["per_symbol_notional"] * 3, .03)
        self.assertFalse(signal_is_current(
            plan["risk_signal"],
            replace(settings, risk_config=LiquidityMomentumFocusConfig(allocation_count=4)),
            day,
        ))

    def test_cash_is_valid_replayable_idempotent_and_never_submits(self):
        store, settings, day, broker = self.setup_session(cash=True)
        self.assertTrue(
            signal_is_current(store.load()["ranking"]["risk_signal"], settings, day)
        )
        plan = enter_for_day(broker, store, settings, day, True, preflight_only=True)
        self.assertEqual(plan["symbols"], [])
        self.assertEqual(plan["budget"], 0)
        closed = enter_for_day(broker, store, settings, day, True)
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(enter_for_day(broker, store, settings, day, True), closed)
        self.assertEqual(broker.submissions, [])
        replay = replay_entry_decision({"position": closed})
        self.assertEqual(replay["status"], "complete", replay)
        report = reconcile_execution(
            {"position": closed, "configuration": {"entry_time": "15:45"}},
            Path("/absent"),
            Path("/absent"),
        )
        self.assertEqual(report["totals"]["simulator_net_pnl"], 0)
        select_reporting_benchmark(report, prefer_actual_time=True)
        print_overview([report])

    def test_conflicts_and_corrupt_inputs_cannot_change_or_submit_basket(self):
        store, settings, day, broker = self.setup_session()
        broker.current_positions = {"C": {"symbol": "C", "qty": 1, "market_value": 100}}
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            enter_for_day(broker, store, settings, day, True)
        self.assertFalse(broker.submissions)
        broker.current_positions = {}
        plan = enter_for_day(broker, store, settings, day, False)
        for mutation in ("momentum", "future", "risk"):
            damaged = copy.deepcopy(plan)
            if mutation == "momentum":
                damaged["ranking_snapshot"]["allocation"]["candidates"][0][
                    "end_close"
                ] = 10000
            elif mutation == "future":
                damaged["ranking_snapshot"]["allocation"]["history_dates"][-1] = str(
                    day
                )
            else:
                damaged["risk_signal"]["observations"][0]["allocation"]["symbols"] = (
                    list("ABCD")
                )
            self.assertNotEqual(
                replay_entry_decision({"position": damaged})["status"], "complete"
            )

    def test_shared_history_matches_backtest_through_cash_and_short_sessions(self):
        from test_strategies import fixture

        from trading_rl.overnight.backtest import run_backtest

        args = fixture()
        n = len(args["dates"])
        policy = LiquidityMomentumFocusConfig(allocation_count=1)
        prices = np.column_stack(
            [100 + np.arange(n), 100 + 2 * np.arange(n), 100 + np.arange(n) / 2]
        )
        allowed = np.ones(n, dtype=bool)
        allowed[-10] = False
        args.update(
            strategy="liquidity-momentum-focus",
            strategy_config=policy,
            daily_closes=prices,
            transaction_cost_bps=0,
            entry_session_mask=allowed,
        )
        args["spy_trend_marks"][-5] = 1.0
        _, summary = run_backtest(**args)
        expected = {row["entry_date"]: row for row in summary["daily_portfolio"]}
        for today in range(n - 12, n - 1):
            dates = args["dates"][: today + 1]
            rows, baskets = historical_baskets(
                dates,
                args["symbols"],
                args["dollar_volume"][: today + 1],
                allowed[: today + 1],
                np.ones((today + 1, 3), bool),
                {},
                policy,
                top=2,
                ema_span=10,
                min_history_days=20,
                minimum_trading_days=100,
                liquidity_scheme="turnover_stability",
                daily_closes=prices[: today + 1],
            )
            observations = [
                {
                    "entry_date": str(dates[r].date()),
                    "exit_date": str(dates[r + 1].date()),
                    "unscaled_return": unit_return(
                        args["entry_prices"][r, b], args["morning_prices"][r + 1, b]
                    ),
                }
                for r, b in zip(rows, baskets, strict=True)
            ]
            signal = risk_signal(
                dates, observations, args["spy_trend_marks"][: today + 1], policy
            )
            self.assertAlmostEqual(
                signal["target_exposure"],
                expected[str(dates[-1].date())]["exposure"],
                places=12,
            )

    def test_cli_switch_to_previous_strategy_preserves_its_risk_default(self):
        parser, args, _ = parse_live_arguments(
            ["preview", "--strategy", "liquidity-trend-vol"]
        )
        settings = _validate_args(parser, args)
        self.assertEqual(settings.risk_config.weak_trend_multiplier, 0.25)
