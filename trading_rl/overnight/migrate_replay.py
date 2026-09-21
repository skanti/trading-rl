"""Backfill historical decision inputs from archived evidence, with backups.

This finite maintenance command never contacts the broker or manages processes.
Unknown historical account conflicts remain explicitly unknown, not invented.
"""

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .decision_replay import ACCOUNT_FIELDS, replay_entry_decision
from .live import StateStore, _atomic_write_json
from .reconcile_live_sessions import (
    infer_entry_minute,
    replay_archived_plan_performance,
    replay_ranking,
)


def upgrade_summary(summary, ticks_path):
    result = copy.deepcopy(summary)
    position = result.get("position") or {}
    if not position.get("entry_date"):
        return result
    if position.get("decision_inputs"):
        return result
    name = position.get("strategy_name") or "liquidity-fixed"
    if name != "liquidity-fixed" or position.get("risk_signal"):
        raise ValueError(
            "cannot reconstruct missing trend/vol preflight broker inputs from later data"
        )
    configuration = result.get("configuration") or {}
    required = ("top", "capital", "capital_fraction", "cash_buffer_fraction")
    if any(key not in configuration for key in required):
        raise ValueError("historical sizing configuration is incomplete")
    ranking = position.get("ranking_snapshot") or result.get("ranking") or {}
    limitations = [
        "Historical holdings/orders were not archived; selection is checked against recorded ranking and actual basket."
    ]
    replay = replay_ranking(result, ticks_path)
    if ranking.get("trade_date") != position["entry_date"]:
        alternatives = [
            replay_ranking(result, ticks_path, liquidity_scheme_override=scheme)
            for scheme in ("dollar_ema", "turnover_stability")
        ]
        matching = [
            row
            for row in alternatives
            if row["actual_symbols"] == row["replayed_symbols"]
        ]
        if len(matching) != 1:
            raise ValueError(
                "missing entry-day ranking cannot be uniquely reconstructed"
            )
        replay = matching[0]
        replay["liquidity_scheme_source"] = "archived_basket_and_entry_day_ticks"
        issuers = {
            row["symbol"]: row.get("issuer", row["symbol"])
            for row in ranking.get("candidates", [])
        }
        ranking = {
            "trade_date": position["entry_date"],
            "ranking_pipeline_version": ranking.get("ranking_pipeline_version"),
            "liquidity_scheme": replay["liquidity_scheme"],
            "candidates": [
                {**row, "issuer": issuers.get(row["symbol"], row["symbol"])}
                for row in replay["top_ranking"]
            ],
            "reconstructed_from": str(ticks_path),
        }
        limitations.append(
            "Entry ranking had been overwritten by a later session; reconstructed using the unique formula matching the archived ordered basket."
        )
    elif replay["liquidity_scheme_source"] == "legacy_default":
        raise ValueError("historical ranking scheme could not be established")
    position["ranking_snapshot"] = copy.deepcopy(ranking)
    settings = {key: configuration[key] for key in required}
    settings.update(
        share_mode=position["share_mode"],
        ema_span=replay["ema_span"],
        min_history_days=replay["minimum_history_days"],
        minimum_trading_days=replay["minimum_trading_days"],
        liquidity_scheme=replay["liquidity_scheme"],
    )
    position["strategy_name"] = name
    position["strategy_parameters"] = {}
    position["decision_inputs"] = {
        "version": 1,
        "strategy_name": name,
        "provenance": "backfilled_from_entry_summary_and_archived_ranking",
        "configuration": settings,
        "account": {
            key: value
            for key, value in position["entry_account_snapshot"].items()
            if key in ACCOUNT_FIELDS
        },
        "positions": [],
        "open_orders": [],
        "selected_assets": [],
        "selection_exclusions_known": False,
        "limitations": limitations,
    }
    result["position"] = position
    check = replay_entry_decision(result)
    if check["status"] != "complete":
        raise ValueError(
            f"historical decision cannot be reconstructed exactly: {check}"
        )
    # Only annotate a summary's own entry-day configuration. Some daily summaries
    # still contain a prior position, whose settings belong to that entry day.
    if result.get("trading_day", position["entry_date"]) == position["entry_date"]:
        result["strategy"] = name
        result.setdefault("configuration", {})["strategy_name"] = name
        result["configuration"]["risk_parameters"] = {}
    position["decision_inputs"]["backfill_validation"] = {
        "decision": check["status"],
        "ranking_exact_match": replay["actual_symbols"] == replay["replayed_symbols"],
        "ranking_scheme_source": replay["liquidity_scheme_source"],
    }
    return result


def migrate(work_dir, backup_dir, *, apply=False):
    work_dir, backup_dir = Path(work_dir), Path(backup_dir)
    store = StateStore(work_dir / "state.json")
    report = {"version": 1, "applied": apply, "files": [], "errors": []}
    with store.locked():
        source = {
            path: path.read_bytes() for path in sorted(work_dir.glob("*/summary.json"))
        }
        canonical = {}
        for path, content in source.items():
            summary = json.loads(content)
            position = summary.get("position") or {}
            if path.parent.name == position.get("entry_date"):
                canonical[position["entry_date"]] = (path, summary)
        upgraded = {}
        for day, (path, summary) in canonical.items():
            try:
                upgraded[day] = upgrade_summary(summary, work_dir / day / "ticks.jsonl")
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
                report["errors"].append({"entry_date": day, "reason": str(error)})
        for day, summary in upgraded.items():
            path = work_dir / "benchmarks" / "nbbo_1559_legacy.npz"
            if path.exists() and infer_entry_minute(summary) == 959:
                summary["position"]["benchmark_paths"] = {"entry_nbbo": str(path)}
        patches = {}
        for path, content in source.items():
            summary = json.loads(content)
            day = (summary.get("position") or {}).get("entry_date")
            if day not in upgraded:
                continue
            updated = copy.deepcopy(summary)
            evidence = upgraded[day]["position"]
            for field in (
                "strategy_name",
                "strategy_parameters",
                "decision_inputs",
                "ranking_snapshot",
                "benchmark_paths",
            ):
                if field in evidence:
                    updated["position"][field] = copy.deepcopy(evidence[field])
            if path.parent.name == day:
                updated["strategy"] = evidence["strategy_name"]
                updated["ranking"] = copy.deepcopy(evidence["ranking_snapshot"])
                updated["configuration"].update(
                    evidence["decision_inputs"]["configuration"]
                )
                updated["configuration"].update(
                    strategy_name=evidence["strategy_name"],
                    risk_parameters=evidence.get("strategy_parameters", {}),
                )
            if updated != summary:
                patches[path] = updated
        if store.path.exists():
            source[store.path] = store.path.read_bytes()
            state = json.loads(source[store.path])
            day = (state.get("position") or {}).get("entry_date")
            if day in upgraded:
                updated = copy.deepcopy(state)
                for field in (
                    "strategy_name",
                    "strategy_parameters",
                    "decision_inputs",
                    "ranking_snapshot",
                    "benchmark_paths",
                ):
                    if field in upgraded[day]["position"]:
                        updated["position"][field] = copy.deepcopy(
                            upgraded[day]["position"][field]
                        )
                if updated != state:
                    patches[store.path] = updated
        for path in sorted((work_dir / "reconciliations").glob("*.json")):
            if path.stem not in upgraded:
                continue
            source[path] = path.read_bytes()
            prior = json.loads(source[path])
            updated = copy.deepcopy(prior)
            decision = replay_entry_decision(upgraded[path.stem])
            updated.update(
                strategy_name=decision["strategy_name"],
                decision_replay=decision,
                planned_strategy=replay_archived_plan_performance(decision, prior),
            )
            if updated != prior:
                patches[path] = updated
        if apply and report["errors"]:
            raise ValueError("backfill aborted: " + json.dumps(report["errors"]))
        # Validate every patch and save exact originals before any mutation.
        for path, payload in patches.items():
            if path.read_bytes() != source[path]:
                raise RuntimeError(f"concurrent modification detected: {path}")
            json.dumps(payload, allow_nan=False)
            relative = path.relative_to(work_dir)
            report["files"].append(
                {
                    "path": str(path),
                    "backup": str(backup_dir / relative),
                    "before_sha256": hashlib.sha256(source[path]).hexdigest(),
                }
            )
        if apply:
            for path in patches:
                backup = backup_dir / path.relative_to(work_dir)
                backup.parent.mkdir(parents=True, exist_ok=True)
                with backup.open("xb") as handle:
                    handle.write(source[path])
            for path, payload in patches.items():
                if path.read_bytes() != source[path]:
                    raise RuntimeError(f"concurrent modification detected: {path}")
                _atomic_write_json(path, payload)
        report["sessions"] = len(upgraded)
        report["generated_at"] = datetime.now(UTC).isoformat()
        if apply:
            _atomic_write_json(backup_dir / "migration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/data/ppv1/live"))
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate(args.work_dir, args.backup_dir, apply=args.apply)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
