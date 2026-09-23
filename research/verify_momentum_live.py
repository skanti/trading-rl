"""Read-only verification of live risk preparation against a frozen backtest."""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trading_rl.overnight.decision_replay import replay_entry_decision, replay_risk
from trading_rl.overnight.history import (
    DEFAULT_SECURITY_MASTER_CACHE,
    load_nasdaq_security_master,
)
from trading_rl.overnight.live import (
    EASTERN,
    RANKING_PIPELINE_VERSION,
    AlpacaClient,
    StateStore,
    _validate_args,
    enter_for_day,
    load_credentials,
    load_data_credentials,
    parse_live_arguments,
)
from trading_rl.overnight.live_risk import prepare_live_allocation, prepare_live_risk
from trading_rl.overnight.reconcile_live_sessions import replay_ranking


class ReadOnlyClient(AlpacaClient):
    def _request(self, method, base, path, **kwargs):
        if method != "GET" or path not in {
            "calendar",
            "corporate-actions",
            "stocks/quotes",
            "stocks/auctions",
        }:
            raise AssertionError(f"unexpected verification request: {method} {path}")
        return super()._request(method, base, path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-09-22")
    args = parser.parse_args()
    day = pd.Timestamp(args.date).date()
    live_parser, values, _ = parse_live_arguments(["preview"])
    config = _validate_args(live_parser, values)
    key, secret = load_credentials()
    data_key, data_secret = load_data_credentials(key, secret)
    client = ReadOnlyClient(
        key,
        secret,
        trading_url=values.trading_url,
        data_url=values.data_url,
        data_key=data_key,
        data_secret=data_secret,
    )
    master = load_nasdaq_security_master(Path(DEFAULT_SECURITY_MASTER_CACHE))
    signal = prepare_live_risk(
        client,
        config,
        day,
        master,
        args.output_dir,
        now=datetime.combine(day, datetime.min.time().replace(hour=14), tzinfo=EASTERN),
    )
    reference = json.loads(args.summary.read_text())
    rows = {row["entry_date"]: row for row in reference["daily_portfolio"]}
    np.testing.assert_allclose(
        [row["unscaled_return"] for row in signal["observations"]],
        [rows[row["entry_date"]]["unscaled_return"] for row in signal["observations"]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        signal["target_exposure"], rows[args.date]["exposure"], rtol=0, atol=1e-12
    )
    replay_risk(
        signal,
        config.risk_config.as_dict(),
        args.date,
        config.top,
        config.strategy_name,
    )
    archived = Path("/data/ppv1/live") / args.date / "summary.json"
    allocation = None
    decision = ranking_replay = None
    if archived.exists():
        saved = json.loads(archived.read_text())
        ranking = saved.get("position", {}).get("ranking_snapshot") or saved["ranking"]
        completed = [
            pd.Timestamp(row["date"]).date()
            for row in client.calendar(config.shortlist_since, day)
            if row["date"] < args.date
        ]
        allocation = prepare_live_allocation(config, ranking, completed, day)
        trades = pd.read_csv(args.summary.parent / "trades.csv")
        traded = trades.loc[trades.entry_date == args.date, "sample_id"].tolist()
        if signal["target_exposure"] > 0 and allocation["symbols"] != traded:
            raise AssertionError(
                f"archived live ranking allocation differs: {allocation['symbols']} != {traded}"
            )
        ranking = copy.deepcopy(ranking)
        ranking.update(
            ranking_pipeline_version=RANKING_PIPELINE_VERSION,
            allocation=allocation,
            risk_signal=signal,
        )
        inputs = saved["position"]["decision_inputs"]
        next_day = saved["position"]["exit_date"]
        # This broker is a frozen archive. It has no submit_order method or network.
        archived_broker = SimpleNamespace(
            positions=lambda: inputs["positions"],
            list_orders=lambda status: inputs["open_orders"],
            account=lambda: inputs["account"],
            list_assets=lambda: inputs["selected_assets"],
            calendar=lambda start, end: [{"date": next_day}],
        )
        store = StateStore(args.output_dir / f"preview-state-{args.date}.json")
        store.save({"version": 1, "ranking": ranking})
        plan = enter_for_day(archived_broker, store, config, day, submit=False)
        preview = {"position": plan}
        decision = replay_entry_decision(preview)
        ranking_replay = replay_ranking(preview, archived.parent / "ticks.jsonl")
        if decision["status"] != "complete" or not ranking_replay["exact_match"]:
            raise AssertionError({"decision": decision, "ranking": ranking_replay})
        (args.output_dir / f"preview-{args.date}.json").write_text(
            json.dumps(preview, indent=2) + "\n"
        )
    report = {
        "date": args.date,
        "strategy": config.strategy_name,
        "parameters": config.risk_config.as_dict(),
        "target_exposure": signal["target_exposure"],
        "observations_verified": len(signal["observations"]),
        "risk_replay": "exact within 1e-12",
        "allocation": allocation,
        "decision_replay": decision,
        "ranking_replay": ranking_replay,
    }
    (args.output_dir / f"verification-{args.date}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
