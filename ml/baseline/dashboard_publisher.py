"""Publish an Alpaca account snapshot to Firestore for the dashboard to read.

This runs as its own process, deliberately. The trading daemon in
``live_overnight_liquidity`` is the one thing on this box that must not fall over, so
it does not carry a Firestore dependency: a network partition or an expired service
account can only ever break the dashboard, never an entry or an exit. Running
separately also means the dashboard keeps refreshing while the daemon is stopped
outside trading hours.

    python -m baseline.dashboard_publisher --dry-run
    python -m baseline.dashboard_publisher
    python -m baseline.dashboard_publisher --watch --interval-seconds 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time as time_module
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from baseline import performance
from baseline.dashboard_config import (
    DashboardConfigError,
    load_dashboard_config,
    service_account
)
from baseline.performance import EASTERN

LOGGER = logging.getLogger("overnight-liquidity-publisher")

SNAPSHOT_VERSION = 1
SESSIONS_SUBCOLLECTION = "sessions"
_DAY_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Only these account fields reach the browser. Everything else Alpaca returns is either
# noise or something there is no reason to publish.
_ACCOUNT_FIELDS = (
    "account_number",
    "status",
    "currency",
    "equity",
    "last_equity",
    "cash",
    "buying_power",
    "non_marginable_buying_power",
    "long_market_value",
    "short_market_value",
    "multiplier",
    "pattern_day_trader",
    "trading_blocked",
    "account_blocked",
    "created_at"
)

_POSITION_FIELDS = (
    "symbol",
    "qty",
    "side",
    "avg_entry_price",
    "current_price",
    "market_value",
    "cost_basis",
    "unrealized_pl",
    "unrealized_plpc",
    "unrealized_intraday_pl",
    "unrealized_intraday_plpc",
    "change_today"
)


def _is_paper_url(url: str) -> bool:
    """Whether a trading endpoint points at the paper environment."""
    return "paper-api.alpaca.markets" in str(url)


def _numeric(mapping: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Project selected fields, converting Alpaca's numeric strings to floats.

    Firestore stores these as numbers so the dashboard never has to parse strings, and
    so range queries stay possible later.
    """
    result: dict[str, Any] = {}
    for field in fields:
        if field not in mapping:
            continue
        value = mapping[field]
        if isinstance(value, str):
            try:
                result[field] = float(value)
                continue
            except ValueError:
                pass
        result[field] = value
    return result


def _strategy_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """The strategy's own state, trimmed to what the dashboard displays."""
    position = state.get("position") or {}
    ranking = state.get("ranking") or {}
    return {
        "status": position.get("status"),
        "entry_date": position.get("entry_date"),
        "exit_date": position.get("exit_date"),
        "symbols": list(position.get("symbols") or []),
        "filled_symbols": list(position.get("filled_symbols") or []),
        "remaining_symbols": list(position.get("remaining_symbols") or []),
        "budget": performance._float(position.get("budget")),
        "per_symbol_notional": performance._float(position.get("per_symbol_notional")),
        "entry_completed_at": position.get("entry_completed_at"),
        "exit_completed_at": position.get("exit_completed_at"),
        "ranking_trade_date": ranking.get("trade_date"),
        "ranking_completed_at": ranking.get("completed_at"),
        "updated_at": state.get("updated_at")
    }


def load_state(state_path: Path) -> dict[str, Any]:
    """Read the daemon's state file, tolerating its absence.

    No state file simply means the strategy has not traded yet, which is a normal
    state to render rather than an error to raise.
    """
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("could not parse %s; publishing without strategy state", state_path)
        return {}


def session_records(work_dir: Path, limit: int = 120) -> list[dict[str, Any]]:
    """Turn per-day ``summary.json`` artifacts into dashboard session rows."""
    if not work_dir.exists():
        return []

    days = sorted(
        (child for child in work_dir.iterdir() if child.is_dir() and _DAY_DIRECTORY.match(child.name)),
        key=lambda child: child.name,
        reverse=True
    )[:limit]

    records: list[dict[str, Any]] = []
    for directory in days:
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("skipping unreadable summary at %s", summary_path)
            continue

        position = summary.get("position") or {}
        execution = summary.get("execution") or {}
        trades = performance.closed_basket(position)
        records.append(
            {
                "trading_day": summary.get("trading_day") or directory.name,
                "last_action": summary.get("last_action"),
                "updated_at": summary.get("updated_at"),
                "status": position.get("status"),
                "entry_date": position.get("entry_date"),
                "exit_date": position.get("exit_date"),
                "symbols": list(position.get("symbols") or []),
                "entry_notional": performance._float(execution.get("entry_filled_notional")),
                "exit_notional": performance._float(execution.get("exit_filled_notional")),
                "realized_pnl": execution.get("realized_pnl_before_fees"),
                "realized_return": execution.get("realized_return_before_fees"),
                "trades": [trade.as_dict() for trade in trades],
                "error": summary.get("error")
            }
        )
    return records


def build_snapshot(
    client: Any,
    state: Mapping[str, Any],
    config: DictConfig,
    now: datetime | None = None
) -> dict[str, Any]:
    """Fetch everything the dashboard shows and shape it into one Firestore document."""
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)

    account = client.account()
    positions = client.positions()
    period = performance.history_period(account.get("created_at"), reference)
    history = client.portfolio_history(period=period, timeframe="1D")

    inception: date | None = None
    created = account.get("created_at")
    if created:
        try:
            inception = datetime.fromisoformat(str(created).replace("Z", "+00:00")).astimezone(EASTERN).date()
        except ValueError:
            inception = None

    series = performance.equity_series(history, since=inception)
    buckets = performance.performance_table(series, account, history, now=reference)
    stats = performance.statistics(series)

    position = state.get("position") or {}
    closed = performance.closed_basket(position) if position else []

    try:
        clock = client.clock()
        market = {
            "is_open": bool(clock.get("is_open")),
            "next_open": clock.get("next_open"),
            "next_close": clock.get("next_close"),
            "timestamp": clock.get("timestamp")
        }
    except Exception:  # noqa: BLE001 - the clock is decoration, not data
        LOGGER.warning("could not read the market clock; publishing without it")
        market = {}

    return {
        "version": SNAPSHOT_VERSION,
        "updated_at": reference.isoformat(),
        "trading_day": reference.date().isoformat(),
        "account": _numeric(account, _ACCOUNT_FIELDS),
        "performance": {key: value.as_dict() for key, value in buckets.items()},
        "statistics": stats.as_dict(),
        "equity_curve": [
            {
                "day": point.day.isoformat(),
                "equity": point.equity,
                "profit_loss": point.profit_loss,
                "profit_loss_pct": point.profit_loss_pct
            }
            for point in series
        ],
        "positions": [_numeric(item, _POSITION_FIELDS) for item in positions],
        "strategy": _strategy_view(state),
        "closed_basket": [trade.as_dict() for trade in closed],
        "basket_totals": performance.basket_totals(closed) if closed else {},
        "market": market,
        "meta": {
            "trading_url": getattr(client, "trading_url", None),
            "paper": _is_paper_url(getattr(client, "trading_url", "")),
            "history_period": period
        }
    }


def _firestore_client(config: DictConfig):
    """Initialise the Admin SDK once per process."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    project_id = str(OmegaConf.select(config, "firebase.project_id"))
    if not firebase_admin._apps:
        credential = service_account(config)
        certificate = None
        if isinstance(credential, dict):
            # The key JSON pasted straight into config.yaml.
            certificate = credentials.Certificate(credential)
        elif credential is not None and credential.exists():
            certificate = credentials.Certificate(str(credential))

        if certificate is not None:
            firebase_admin.initialize_app(certificate, {"projectId": project_id})
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            firebase_admin.initialize_app(options={"projectId": project_id})
        else:
            # Do not fall through to application default credentials here. Off GCP that
            # path blocks on the metadata server until it times out, which would stall
            # every publish attempt from the trading daemon for no possible benefit.
            raise DashboardConfigError(
                f"no Firebase service-account key at {credential}. Download one from "
                "Firebase console -> Project settings -> Service accounts, or set "
                "GOOGLE_APPLICATION_CREDENTIALS."
            )
    return firestore.client()


def publish(
    snapshot: Mapping[str, Any],
    sessions: Sequence[Mapping[str, Any]],
    config: DictConfig
) -> None:
    """Write the snapshot document and the session subcollection."""
    collection = str(OmegaConf.select(config, "firebase.collection"))
    document = str(OmegaConf.select(config, "firebase.document"))

    client = _firestore_client(config)
    reference = client.collection(collection).document(document)
    reference.set(dict(snapshot))

    sessions_reference = reference.collection(SESSIONS_SUBCOLLECTION)
    batch = client.batch()
    written = 0
    for record in sessions:
        day = str(record.get("trading_day") or "")
        if not day:
            continue
        batch.set(sessions_reference.document(day), dict(record))
        written += 1
        if written % 400 == 0:  # Firestore caps a batch at 500 writes.
            batch.commit()
            batch = client.batch()
    if written % 400 != 0:
        batch.commit()

    LOGGER.info(
        "published %s/%s with %d session record(s)", collection, document, written
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an Alpaca account snapshot to Firestore for the dashboard."
    )
    parser.add_argument("--config", default=None, help="path to dashboard/config.yaml")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="daemon work directory holding per-day artifacts (default: from the trading script)"
    )
    parser.add_argument("--state-path", default=None, help="strategy state.json (default: <work-dir>/state.json)")
    parser.add_argument("--dry-run", action="store_true", help="print the snapshot and write nothing")
    parser.add_argument("--watch", action="store_true", help="publish repeatedly instead of once")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--sessions-limit", type=int, default=120)
    parser.add_argument(
        "--trading-url",
        default=None,
        help="Alpaca endpoint (default: $ALPACA_URL, else the paper endpoint)"
    )
    parser.add_argument(
        "--allow-live-endpoint",
        action="store_true",
        help="permit a trading_url other than paper-api.alpaca.markets"
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s"
    )

    config = load_dashboard_config(args.config)

    from baseline.live_overnight_liquidity import (
        DEFAULT_WORK_DIR,
        PAPER_TRADING_URL,
        AlpacaClient,
        _is_paper_endpoint,
        load_credentials
    )

    trading_url = args.trading_url or os.environ.get("ALPACA_URL") or PAPER_TRADING_URL
    if not _is_paper_endpoint(trading_url) and not args.allow_live_endpoint:
        # ml/.env carries live keys on the line above the paper ones; refuse to read a
        # live account by accident, matching the trading script's own rail.
        parser.error(
            f"refusing to publish from non-paper endpoint {trading_url} "
            "without --allow-live-endpoint"
        )

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else DEFAULT_WORK_DIR
    state_path = Path(args.state_path).expanduser() if args.state_path else work_dir / "state.json"

    key, secret = load_credentials()
    client = AlpacaClient(key, secret, trading_url)

    def publish_once() -> None:
        state = load_state(state_path)
        snapshot = build_snapshot(client, state, config)
        sessions = session_records(work_dir, limit=args.sessions_limit)
        if args.dry_run:
            print(json.dumps({"snapshot": snapshot, "sessions": sessions}, indent=2, default=str))
            return
        publish(snapshot, sessions, config)

    if not args.watch:
        publish_once()
        return 0

    LOGGER.info("watching; publishing every %.0fs", args.interval_seconds)
    while True:
        try:
            publish_once()
        except Exception as error:  # noqa: BLE001 - a publisher must outlive transient faults
            LOGGER.exception("publish failed; retrying after the interval: %s", error)
        time_module.sleep(max(30.0, args.interval_seconds))


if __name__ == "__main__":
    sys.exit(main())
