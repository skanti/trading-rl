"""End-to-end decision capture, offline replay and historical migration."""

import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from test_live import FakeBroker
from test_live_risk import account, trend_config

from trading_rl.overnight.decision_replay import replay_entry_decision
from trading_rl.overnight.live import (
    RANKING_PIPELINE_VERSION,
    StateStore,
    enter_for_day,
)
from trading_rl.overnight.live_risk import configuration_fingerprint
from trading_rl.overnight.migrate_replay import migrate, upgrade_summary
from trading_rl.overnight.reconcile_live_sessions import (
    reconcile_execution,
    replay_planned_performance,
)
from trading_rl.overnight.risk_history import risk_signal, unit_return


def signal(settings):
    dates = pd.bdate_range(end="2026-08-24", periods=130)
    marks = np.arange(len(dates), dtype=float) + 100
    observations = []
    for i in range(len(dates) - 21, len(dates) - 1):
        entries = [100.0, 100.0]
        exits = [101.0 + 0.1 * (i % 3), 99.5]
        observations.append(
            {
                "entry_date": str(dates[i].date()),
                "exit_date": str(dates[i + 1].date()),
                "symbols": ["A", "B"],
                "entry_prices": entries,
                "exit_prices": exits,
                "unscaled_return": unit_return(entries, exits),
            }
        )
    result = risk_signal(dates, observations, marks, settings.risk_config)
    result.update(
        configuration_sha256=configuration_fingerprint(settings),
        input_sha256="fixture",
        spy_history=[
            {"date": str(day.date()), "minute_open": float(mark)}
            for day, mark in zip(dates[-101:-1], marks[-101:-1], strict=True)
        ],
    )
    return result


def captured_plan(root):
    settings = trend_config()
    broker = FakeBroker()
    broker.current_positions = {"HELD": {"symbol": "HELD", "market_value": "3000"}}
    broker.account = Mock(return_value=account())
    broker.list_assets = Mock(
        return_value=[
            {"symbol": name, "marginable": True, "maintenance_margin_requirement": 30}
            for name in ("A", "B")
        ]
    )
    store = StateStore(root / "state.json")
    ranking = {
        "trade_date": "2026-08-24",
        "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
        "risk_signal": signal(settings),
        "candidates": [
            {"rank": i + 1, "symbol": name, "issuer": name}
            for i, name in enumerate(("HELD", "A", "B"))
        ],
    }
    store.save({"version": 1, "ranking": ranking})
    plan = enter_for_day(
        broker, store, settings, date(2026, 8, 24), submit=True, preflight_only=True
    )
    return {"configuration": {}, "position": plan, "ranking": ranking}, broker


class DecisionReplayTest(unittest.TestCase):
    def test_live_preflight_is_exactly_replayable_without_broker_or_current_config(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            summary, broker = captured_plan(Path(directory))
            summary = json.loads(json.dumps(summary))
            # Deliberately conflicting mutable top-level defaults and later ranking.
            summary["configuration"] = {
                "top": 999,
                "capital": 1,
                "strategy_name": "liquidity-fixed",
            }
            summary["ranking"] = {"trade_date": "2030-01-01", "candidates": []}
            broker.account.side_effect = AssertionError(
                "offline replay must not call broker"
            )
            replay = replay_entry_decision(summary)
        self.assertEqual(replay["status"], "complete", replay)
        self.assertTrue(all(replay["checks"].values()))
        self.assertEqual(replay["replayed_symbols"], ["A", "B"])
        self.assertEqual(replay["budget"], 16660)
        self.assertEqual(replay["orders"]["A"], {"notional": "8330.00"})
        self.assertEqual(broker.submissions, [])

    def test_replayed_order_payloads_equal_live_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, broker = captured_plan(root)
            replay = replay_entry_decision(summary)
            enter_for_day(
                broker,
                StateStore(root / "state.json"),
                trend_config(),
                date(2026, 8, 24),
                submit=True,
            )
            actual = {row["symbol"]: row for row in broker.submissions}
        self.assertEqual(replay["order_payloads"], actual)

    def test_replay_detects_changed_inputs_policy_prices_and_saved_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            original, _ = captured_plan(Path(directory))
        for mutate in (
            lambda p: p.update(budget=p["budget"] + 1),
            lambda p: p["decision_inputs"]["positions"][0].update(market_value="8000"),
            lambda p: p["decision_inputs"]["selected_assets"][0].update(
                marginable=False
            ),
            lambda p: p["risk_signal"]["observations"][-1]["exit_prices"].__setitem__(
                0, 999
            ),
            lambda p: p["risk_signal"]["spy_history"][-1].update(minute_open=1),
            lambda p: p.update(symbols=["B", "A"]),
        ):
            modified = copy.deepcopy(original)
            mutate(modified["position"])
            self.assertNotEqual(replay_entry_decision(modified)["status"], "complete")

    def test_plan_pnl_uses_planned_sizes_not_actual_fill_notional(self):
        decision = {
            "status": "complete",
            "orders": {"A": {"notional": "1000.00"}, "B": {"qty": "2"}},
        }
        entry = {
            "A": Mock(price=100.0, raw_price=100.0),
            "B": Mock(price=50.0, raw_price=100.0),
        }
        exits = {"A": Mock(price=110.0), "B": Mock(price=55.0)}
        replay = replay_planned_performance(
            decision, {"entry": entry, "exit": exits}, 1000
        )
        self.assertAlmostEqual(replay["gross_pnl"], 120)
        self.assertEqual(replay["entry_notional"], 1200)
        self.assertAlmostEqual(replay["gross_return_on_account_equity"], 0.12)

    def test_reconciliation_replays_full_plan_separately_from_partial_fills(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = captured_plan(Path(directory))
        position = summary["position"]
        position["status"] = "closed"
        position["entry_orders"] = {
            "A": {
                "filled_qty": "50",
                "filled_avg_price": "100",
                "notional": "8330",
                "filled_at": "2026-08-24T19:45:00Z",
                "submitted_at": "2026-08-24T19:45:00Z",
            }
        }
        position["exit_orders"] = {
            "A": {
                "filled_qty": "50",
                "filled_avg_price": "101",
                "filled_at": "2026-08-25T13:30:00Z",
                "submitted_at": "2026-08-25T10:00:00Z",
            }
        }

        def prices(source, path, targets, **kwargs):
            value = 100.0 if source == "nbbo-ask" else 101.0
            return {
                symbol: Mock(
                    price=value, raw_price=value, staleness_minutes=0.0, exchange="Q"
                )
                for symbol in targets
            }

        with patch(
            "trading_rl.overnight.reconcile_live_sessions.load_benchmark_prices",
            side_effect=prices,
        ):
            result = reconcile_execution(
                summary, Path("/unused"), Path("/unused"), nbbo_path=Path("/unused")
            )
        self.assertEqual(result["decision_replay"]["status"], "complete")
        self.assertEqual(result["symbols"], ["A"])
        self.assertEqual(result["totals"]["actual_gross_pnl"], 50.0)
        self.assertAlmostEqual(result["planned_strategy"]["gross_pnl"], 166.6)
        self.assertEqual(result["planned_strategy"]["entry_notional"], 16660.0)

    def test_legacy_migration_backup_and_second_run_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root / "2026-08-24"
            day.mkdir()
            bars = []
            for d in pd.bdate_range("2026-08-03", "2026-08-21"):
                for symbol, volume in [("A", 2000), ("B", 1000)]:
                    bars.append(
                        {
                            "symbol": symbol,
                            "t": f"{d.date()}T04:00:00Z",
                            "v": volume,
                            "vw": 100.0,
                            "c": 100.0,
                        }
                    )
            (day / "ticks.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in bars)
            )
            ranking = {
                "trade_date": "2026-08-24",
                "ranking_pipeline_version": 7,
                "liquidity_scheme": "dollar_ema",
                "candidates": [{"rank": 1, "symbol": "A"}, {"rank": 2, "symbol": "B"}],
            }
            summary = {
                "trading_day": "2026-08-24",
                "configuration": {
                    "top": 2,
                    "capital": None,
                    "capital_fraction": 1.0,
                    "cash_buffer_fraction": 0.02,
                    "min_history_days": 2,
                    "minimum_trading_days": 2,
                    "ema_span": 2,
                    "liquidity_scheme": "dollar_ema",
                },
                "ranking": ranking,
                "position": {
                    "entry_date": "2026-08-24",
                    "status": "closed",
                    "share_mode": "fractional",
                    "symbols": ["A", "B"],
                    "budget": 9800.0,
                    "per_symbol_notional": 4900.0,
                    "entry_account_snapshot": account(),
                    "entry_orders": {},
                },
            }
            path = day / "summary.json"
            original = json.dumps(summary).encode()
            path.write_bytes(original)
            store = root / "state.json"
            store.write_text(
                json.dumps({"version": 1, "position": summary["position"]})
            )
            backup = root / "backups"
            report = migrate(root, backup, apply=True)
            self.assertFalse(report["errors"])
            self.assertEqual(
                (backup / "2026-08-24/summary.json").read_bytes(), original
            )
            changed = json.loads(path.read_text())
            self.assertEqual(changed["position"]["strategy_name"], "liquidity-fixed")
            self.assertFalse(
                changed["position"]["decision_inputs"]["selection_exclusions_known"]
            )
            self.assertEqual(replay_entry_decision(changed)["status"], "complete")
            second = migrate(root, root / "second-backup", apply=False)
            self.assertEqual(second["files"], [])
            self.assertEqual(
                changed["position"]["entry_account_snapshot"],
                summary["position"]["entry_account_snapshot"],
            )
            bad = copy.deepcopy(summary)
            bad["position"]["strategy_name"] = "liquidity-trend-vol"
            with self.assertRaisesRegex(ValueError, "cannot reconstruct"):
                upgrade_summary(bad, day / "ticks.jsonl")
