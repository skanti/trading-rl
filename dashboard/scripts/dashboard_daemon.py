"""Standalone Alpaca-to-Firestore dashboard daemon.

It only reads Alpaca and the trading daemon's JSON artifacts. Firebase failures cannot
affect trading because this process shares no execution path with the trading daemon.

    python scripts/dashboard_daemon.py --dry-run
    python scripts/dashboard_daemon.py --once
    python scripts/dashboard_daemon.py
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
from urllib.parse import urlparse

from omegaconf import DictConfig, OmegaConf
import requests

import dashboard_digest
import dashboard_metrics as performance
from dashboard_config import (
    DashboardConfigError,
    load_dashboard_config,
    service_account
)
from dashboard_metrics import EASTERN

LOGGER = logging.getLogger("dashboard-daemon")

SNAPSHOT_VERSION = 2
SESSIONS_SUBCOLLECTION = "sessions"
PAPER_TRADING_URL = "https://paper-api.alpaca.markets/v2"
DEFAULT_WORK_DIR = Path("/data/ppv1/live")
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
    return urlparse(str(url)).hostname == "paper-api.alpaca.markets"


def _normalize_api_base(url: str) -> str:
    base = str(url).rstrip("/")
    return base if base.rsplit("/", 1)[-1].startswith("v") else f"{base}/v2"


def load_credentials() -> tuple[str, str]:
    """Load Alpaca credentials without depending on the trading package."""
    key = os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "set ALPACA_KEY and ALPACA_SECRET "
            "(or APCA_API_KEY_ID and APCA_API_SECRET_KEY)"
        )
    return key, secret


class AlpacaClient:
    """Minimal read-only Alpaca client used by this daemon."""

    def __init__(
        self,
        key: str,
        secret: str,
        trading_url: str = PAPER_TRADING_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.trading_url = _normalize_api_base(trading_url)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
                "User-Agent": "trading-dashboard/1",
            }
        )

    def _get(self, path: str, **params: object) -> Any:
        url = f"{self.trading_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                if 200 <= response.status_code < 300:
                    return response.json() if response.content else None
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time_module.sleep(min(2**attempt, 8))
                continue
            if attempt >= self.max_retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 8)
            time_module.sleep(delay)
        raise AssertionError("unreachable retry loop")

    def account(self) -> dict[str, Any]:
        return dict(self._get("account"))

    def positions(self) -> list[dict[str, Any]]:
        return list(self._get("positions"))

    def portfolio_history(self, period: str = "1A", timeframe: str = "1D") -> dict[str, Any]:
        return dict(
            self._get(
                "account/portfolio/history",
                period=period,
                timeframe=timeframe,
            )
        )

    def orders(self, *, after: str) -> list[dict[str, Any]]:
        return list(
            self._get(
                "orders",
                status="all",
                after=after,
                direction="asc",
                limit=500,
            )
        )

    def clock(self) -> dict[str, Any]:
        return dict(self._get("clock"))


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


def first_trade_date(work_dir: Path) -> date | None:
    """Return the earliest strategy entry date with at least one actual fill."""
    if not work_dir.exists():
        return None

    day_directories = sorted(
        child
        for child in work_dir.iterdir()
        if child.is_dir() and _DAY_DIRECTORY.match(child.name)
    )
    for directory in day_directories:
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        position = summary.get("position") or {}
        if performance.filled_notional(position.get("entry_orders") or {}) <= 0.0:
            continue
        raw_date = position.get("entry_date") or summary.get("trading_day") or directory.name
        try:
            return date.fromisoformat(str(raw_date))
        except ValueError:
            LOGGER.warning("ignoring invalid filled entry date in %s", summary_path)
    return None


def session_records(
    work_dir: Path,
    limit: int | None = 120,
    order_history: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Turn per-day ``summary.json`` artifacts into dashboard session rows."""
    if not work_dir.exists():
        return []

    days = sorted(
        (child for child in work_dir.iterdir() if child.is_dir() and _DAY_DIRECTORY.match(child.name)),
        key=lambda child: child.name,
        reverse=True
    )

    records: list[dict[str, Any]] = []
    seen_baskets: set[tuple[str, str]] = set()
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
        entry_orders = position.get("entry_orders") or {}
        if performance.filled_notional(entry_orders) <= 0.0:
            continue
        entry_date = str(position.get("entry_date") or summary.get("trading_day") or directory.name)
        exit_date = str(position.get("exit_date") or "")
        basket_key = (entry_date, exit_date)
        if basket_key in seen_baskets:
            continue
        seen_baskets.add(basket_key)

        execution = summary.get("execution") or {}
        trades = (
            performance.closed_basket_from_order_history(position, order_history)
            if order_history is not None and position.get("status") == "closed"
            else performance.closed_basket(position)
        )
        totals = performance.basket_totals(trades) if trades else {}
        entry_notional = (
            totals["entry_notional"]
            if totals
            else performance._float(execution.get("entry_filled_notional"))
        )
        exit_notional = (
            totals["exit_notional"]
            if totals
            else performance._float(execution.get("exit_filled_notional"))
        )
        realized_pnl = totals.get("pnl") if totals else execution.get("realized_pnl_before_fees")
        realized_return = (
            totals.get("pnl_pct") if totals else execution.get("realized_return_before_fees")
        )
        records.append(
            {
                "trading_day": entry_date,
                "last_action": summary.get("last_action"),
                "updated_at": summary.get("updated_at"),
                "status": position.get("status"),
                "entry_date": position.get("entry_date"),
                "exit_date": position.get("exit_date"),
                "symbols": list(position.get("symbols") or []),
                "entry_notional": entry_notional,
                "exit_notional": exit_notional,
                "realized_pnl": realized_pnl,
                "realized_return": realized_return,
                "trades": [trade.as_dict() for trade in trades],
                "error": summary.get("error")
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def build_snapshot(
    client: Any,
    state: Mapping[str, Any],
    inception: date | None = None,
    now: datetime | None = None,
    order_history: Sequence[Mapping[str, Any]] | None = None,
    session_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch everything the dashboard shows and shape it into one Firestore document."""
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)

    account = client.account()
    positions = client.positions()
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

    period = performance.history_period(inception or account.get("created_at"), reference)
    history = client.portfolio_history(period=period, timeframe="1D")

    series_start = inception
    created = account.get("created_at")
    if series_start is None and created:
        try:
            series_start = datetime.fromisoformat(
                str(created).replace("Z", "+00:00")
            ).astimezone(EASTERN).date()
        except ValueError:
            series_start = None

    account_series = performance.equity_series(history, since=series_start)
    series = performance.realized_equity_series(
        session_history or [],
        base_value=performance.inception_equity(history, account_series),
        inception=inception,
    )
    buckets = performance.realized_performance_table(series, now=reference)
    stats = performance.statistics(series, baseline_is_first=True)

    position = state.get("position") or {}
    closed = performance.closed_basket(position) if position else []
    if position.get("status") == "closed" and position.get("entry_date"):
        try:
            history = (
                order_history
                if order_history is not None
                else client.orders(after=f"{position['entry_date']}T00:00:00Z")
            )
            closed = performance.closed_basket_from_order_history(position, history)
        except (AttributeError, NotImplementedError):
            # Small fake clients and alternate read-only clients may not expose orders.
            pass
        except Exception:  # noqa: BLE001 - fill detail is optional dashboard decoration
            LOGGER.warning("could not reconcile partial exit fills from order history")

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
            "history_period": period,
            "inception_date": series_start.isoformat() if series_start else None
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
            # path can block on the metadata server before failing.
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
    document_name = str(OmegaConf.select(config, "firebase.document"))

    client = _firestore_client(config)
    reference = client.collection(collection).document(document_name)
    reference.set(dict(snapshot))

    sessions_reference = reference.collection(SESSIONS_SUBCOLLECTION)
    batch = client.batch()
    written = 0
    published_days: set[str] = set()
    for record in sessions:
        day = str(record.get("trading_day") or "")
        if not day:
            continue
        batch.set(sessions_reference.document(day), dict(record))
        published_days.add(day)
        written += 1
        if written % 400 == 0:  # Firestore caps a batch at 500 writes.
            batch.commit()
            batch = client.batch()
    if written % 400 != 0:
        batch.commit()

    # A basket summary is written to both its entry-day and exit-day artifact
    # directories for auditing. Older dashboard versions published both as separate
    # sessions. Remove only stale documents inside the refreshed window, preserving
    # history older than ``--sessions-limit``.
    removed = 0
    if published_days:
        oldest_published = min(published_days)
        stale = []
        for session_reference in sessions_reference.list_documents():
            if session_reference.id in published_days:
                continue
            inside_refresh_window = session_reference.id >= oldest_published
            payload = session_reference.get().to_dict() or {}
            pretrade_artifact = not payload.get("entry_date")
            if inside_refresh_window or pretrade_artifact:
                stale.append(session_reference)
        for offset in range(0, len(stale), 400):
            cleanup = client.batch()
            for session_reference in stale[offset : offset + 400]:
                cleanup.delete(session_reference)
                removed += 1
            cleanup.commit()

    LOGGER.info(
        "published %s/%s with %d session record(s); removed %d stale record(s)",
        collection,
        document_name,
        written,
        removed,
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
    parser.add_argument("--dry-run", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--once", action="store_true", help="publish one snapshot and exit")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--sessions-limit", type=int, default=120)
    parser.add_argument(
        "--digest-state-path",
        default=None,
        help="delivered-email marker file (default: <work-dir>/.dashboard-digest-state.json)",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="do not send the once-per-closed-basket digest email",
    )
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

    trading_url = args.trading_url or os.environ.get("ALPACA_URL") or PAPER_TRADING_URL
    if not _is_paper_url(trading_url) and not args.allow_live_endpoint:
        # ml/.env carries live keys on the line above the paper ones; refuse to read a
        # live account by accident, matching the trading script's own rail.
        parser.error(
            f"refusing to publish from non-paper endpoint {trading_url} "
            "without --allow-live-endpoint"
        )

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else DEFAULT_WORK_DIR
    state_path = Path(args.state_path).expanduser() if args.state_path else work_dir / "state.json"
    digest_state_path = (
        Path(args.digest_state_path).expanduser()
        if args.digest_state_path
        else work_dir / ".dashboard-digest-state.json"
    )

    key, secret = load_credentials()
    client = AlpacaClient(key, secret, trading_url)

    def publish_once() -> None:
        state = load_state(state_path)
        inception = first_trade_date(work_dir)
        order_history = None
        if inception is not None:
            try:
                order_history = client.orders(after=f"{inception.isoformat()}T00:00:00Z")
            except Exception:  # noqa: BLE001 - fall back to durable artifact summaries
                LOGGER.warning("could not load order history; using artifact session totals")
        all_sessions = session_records(
            work_dir,
            limit=None,
            order_history=order_history,
        )
        sessions = all_sessions[:args.sessions_limit]
        snapshot = build_snapshot(
            client,
            state,
            inception=inception,
            order_history=order_history,
            session_history=all_sessions,
        )
        if args.dry_run:
            print(json.dumps({"snapshot": snapshot, "sessions": sessions}, indent=2, default=str))
            return
        publish(snapshot, sessions, config)
        if not args.no_email:
            key = dashboard_digest.digest_key(state, str(snapshot["trading_day"]))
            if key and key not in dashboard_digest.delivered_keys(digest_state_path):
                try:
                    recipients = dashboard_digest.send_digest(snapshot, state, config)
                except Exception as error:  # noqa: BLE001 - SMTP cannot affect publishing
                    LOGGER.exception("digest email failed; it will be retried: %s", error)
                else:
                    dashboard_digest.mark_delivered(digest_state_path, key)
                    LOGGER.info("digest emailed to %s", ", ".join(recipients))

    if args.once or args.dry_run:
        publish_once()
        return 0

    interval = max(30.0, args.interval_seconds)
    LOGGER.info("started; publishing every %.0fs", interval)
    try:
        while True:
            try:
                publish_once()
            except Exception as error:  # noqa: BLE001 - retry transient failures
                LOGGER.exception("publish failed; retrying after the interval: %s", error)
            time_module.sleep(interval)
    except KeyboardInterrupt:
        LOGGER.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
