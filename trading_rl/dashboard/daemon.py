"""Standalone Alpaca-to-Firestore dashboard daemon.

It only reads Alpaca and the trading daemon's JSON artifacts. Firebase failures cannot
affect trading because this process shares no execution path with the trading daemon.

    trading-dashboard --dry-run
    trading-dashboard --once
    trading-dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time as time_module
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

from omegaconf import DictConfig, OmegaConf
import requests

from . import digest as dashboard_digest
from . import metrics as performance
from .config import (
    DashboardConfigError,
    load_dashboard_config,
    service_account
)
from .metrics import EASTERN

from trading_rl.overnight.broker_fees import (
    FeeActivityClient,
    broker_fees_for_session,
    cache_fee_activities,
)
from trading_rl.overnight.live_config import DEFAULT_LIVE_CONFIG_PATH

LOGGER = logging.getLogger("dashboard-daemon")

SNAPSHOT_VERSION = 6
SESSIONS_SUBCOLLECTION = "sessions"
PAPER_TRADING_URL = "https://paper-api.alpaca.markets/v2"
DEFAULT_WORK_DIRS = {
    "paper": Path("/data/ppv1/paper"),
    "live": Path("/data/ppv1/live"),
}
DEFAULT_TRADING_CONFIG_PATH = DEFAULT_LIVE_CONFIG_PATH
EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
_DAY_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_PUBLIC_TRADING_CONFIG_FIELDS = {
    "schedule": (
        "time_zone",
        "ranking_time",
        "entry_time",
        "exit_time",
        "minimum_ranking_lead_minutes",
        "entry_grace_seconds",
    ),
    "strategy": (
        "top",
        "liquidity_scheme",
        "ema_span",
        "min_history_days",
        "minimum_trading_days",
        "liquidity_lookback_days",
        "exchanges",
    ),
    "data": (
        "shortlist_since",
        "shortlist_daily_top",
        "shortlist_lookback_sessions",
        "daily_overlap_days",
        "feed",
        "quote_feed",
        "data_batch_size",
        "data_workers",
    ),
    "execution": (
        "order_submit_workers",
        "capital",
        "capital_fraction",
        "cash_buffer_fraction",
        "share_mode",
        "quote_max_age_seconds",
        "fill_timeout_seconds",
        "poll_seconds",
        "entry_preflight_seconds",
    ),
}

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


def _trading_mode(url: str) -> str:
    """Detect a supported Alpaca environment without a separate mode flag."""
    parsed = urlparse(str(url))
    if parsed.scheme != "https":
        raise ValueError("Alpaca trading URL must use https")
    if parsed.hostname == "paper-api.alpaca.markets":
        return "paper"
    if parsed.hostname == "api.alpaca.markets":
        return "live"
    raise ValueError(f"unsupported Alpaca trading endpoint: {url}")


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

    def account_activities(
        self,
        activity_type: str,
        *,
        after: date | datetime | None = None,
        until: date | datetime | None = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Return every account activity of one type in chronological order."""
        if page_size < 1 or page_size > 100:
            raise ValueError("account activity page_size must be between 1 and 100")
        output: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, object] = {
                "direction": "asc",
                "page_size": int(page_size),
            }
            if after is not None:
                params["after"] = after.isoformat()
            if until is not None:
                params["until"] = until.isoformat()
            if page_token:
                params["page_token"] = page_token
            page = list(
                self._get(
                    f"account/activities/{quote(activity_type.upper(), safe='')}",
                    **params,
                )
            )
            output.extend(dict(row) for row in page)
            if len(page) < page_size:
                break
            next_token = page[-1].get("id") if page else None
            if not next_token or str(next_token) == page_token:
                raise ValueError("Alpaca account activity pagination did not advance")
            page_token = str(next_token)
        return output

    def clock(self) -> dict[str, Any]:
        return dict(self._get("clock"))

    def calendar(self, start: date, end: date) -> list[dict[str, Any]]:
        return list(
            self._get(
                "calendar",
                start=start.isoformat(),
                end=end.isoformat(),
            )
        )


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
        "share_mode": position.get("share_mode"),
        "budget": performance._float(position.get("budget")),
        "per_symbol_notional": performance._float(position.get("per_symbol_notional")),
        "estimated_deployed_notional": performance._float(
            position.get("estimated_deployed_notional")
        ),
        "target_quantities": {
            str(symbol): int(quantity)
            for symbol, quantity in dict(position.get("target_quantities") or {}).items()
        },
        "skipped_symbols": list(position.get("skipped_symbols") or []),
        "entry_completed_at": position.get("entry_completed_at"),
        "exit_completed_at": position.get("exit_completed_at"),
        "ranking_trade_date": ranking.get("trade_date"),
        "ranking_completed_at": ranking.get("created_at") or ranking.get("completed_at"),
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


def load_trading_configuration(
    path: Path,
    effective_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load active merged values, falling back to YAML when no daemon is running."""
    config_path = path.expanduser().resolve()
    if effective_path is not None and effective_path.is_file():
        try:
            effective = json.loads(effective_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid effective trading config: {effective_path}") from error
        if effective.get("version") != 1 or not isinstance(
            effective.get("configuration"), dict
        ):
            raise ValueError(f"unsupported effective trading config: {effective_path}")
        config = OmegaConf.create(effective["configuration"])
    else:
        if not config_path.is_file():
            raise FileNotFoundError(f"trading config does not exist: {config_path}")
        config = OmegaConf.load(config_path)
    public: dict[str, dict[str, Any]] = {}
    for section, fields in _PUBLIC_TRADING_CONFIG_FIELDS.items():
        values: dict[str, Any] = {}
        for field in fields:
            key = f"{section}.{field}"
            value = OmegaConf.select(config, key)
            if value is None and key != "execution.capital":
                raise ValueError(f"trading config is missing {key}")
            values[field] = value
        public[section] = values
    return public


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


def _session_fee_summary(
    work_dir: Path,
    entry_day: str,
    exit_day: date,
    fee_activities: Sequence[Mapping[str, object]] | None,
    as_of: datetime,
    fee_client: FeeActivityClient | None = None,
    force_refresh: bool = False,
) -> tuple[str, dict[str, object] | None]:
    cache_path = work_dir / entry_day / "fee_activities.json"
    if fee_activities is not None:
        summary = cache_fee_activities(cache_path, exit_day, fee_activities, as_of=as_of)
    else:
        summary, warning = broker_fees_for_session(
            fee_client, cache_path, exit_day, as_of=as_of, force_refresh=force_refresh,
        )
        if fee_client is not None and warning and (
            summary.get("status") != "pending" or "refresh failed" in warning
        ):
            LOGGER.warning("broker fees for %s: %s", entry_day, warning)
    status = summary.get("status")
    return "confirmed" if status == "complete" else str(status), summary


def session_records(
    work_dir: Path,
    limit: int | None = 120,
    order_history: Sequence[Mapping[str, Any]] | None = None,
    fee_activities: Sequence[Mapping[str, object]] | None = None,
    fees_as_of: datetime | None = None,
    fee_client: FeeActivityClient | None = None,
    force_fee_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Turn artifacts into rows whose net P&L uses confirmed Alpaca fees."""
    if not work_dir.exists():
        return []

    days = sorted(
        (child for child in work_dir.iterdir() if child.is_dir() and _DAY_DIRECTORY.match(child.name)),
        key=lambda child: child.name,
        reverse=True
    )

    candidates: dict[tuple[str, str], tuple[tuple[bool, bool, str], dict[str, Any]]] = {}
    resolved_fees: dict[tuple[str, str], tuple[str, dict[str, object] | None]] = {}
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

        execution = summary.get("execution") or {}
        entry_snapshot = position.get("entry_account_snapshot") or {}
        exit_snapshot = position.get("exit_account_snapshot") or {}
        entry_equity = performance._float(entry_snapshot.get("equity"))
        exit_equity = performance._float(exit_snapshot.get("equity"))
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
        gross_realized_pnl = (
            totals.get("pnl") if totals else execution.get("realized_pnl_before_fees")
        )
        gross_realized_return = (
            totals.get("pnl_pct") if totals else execution.get("realized_return_before_fees")
        )
        account_equity_change = (
            exit_equity - entry_equity
            if entry_equity > 0.0 and exit_equity > 0.0
            else None
        )
        fee_status = "not_applicable"
        fee_summary: dict[str, object] | None = None
        if position.get("status") == "closed" and exit_date:
            try:
                if basket_key not in resolved_fees:
                    resolved_fees[basket_key] = _session_fee_summary(
                        work_dir,
                        entry_date,
                        date.fromisoformat(exit_date),
                        fee_activities,
                        fees_as_of or datetime.now(tz=UTC),
                        fee_client=fee_client,
                        force_refresh=force_fee_refresh,
                    )
                fee_status, fee_summary = resolved_fees[basket_key]
            except (OSError, TypeError, ValueError) as error:
                LOGGER.warning("could not resolve broker fees for %s: %s", entry_date, error)
                fee_status = "unavailable"

        fee_cost = (
            performance._float(fee_summary.get("cost"))
            if fee_status == "confirmed" and fee_summary is not None
            else None
        )
        assumed_fee_cost = (
            fee_cost
            if fee_status == "confirmed"
            else 0.0 if fee_status in {"pending", "unavailable"} else None
        )
        if gross_realized_pnl is not None and assumed_fee_cost is not None:
            realized_pnl = performance._float(gross_realized_pnl) - assumed_fee_cost
            denominator = entry_equity if entry_equity > 0.0 else entry_notional
            realized_return = realized_pnl / denominator if denominator > 0.0 else None
        else:
            realized_pnl = None
            realized_return = None
        unexplained_residual = (
            account_equity_change - realized_pnl
            if account_equity_change is not None and realized_pnl is not None
            else None
        )
        record = {
            "trading_day": entry_date,
            "last_action": summary.get("last_action"),
            "updated_at": summary.get("updated_at"),
            "status": position.get("status"),
            "entry_date": position.get("entry_date"),
            "exit_date": position.get("exit_date"),
            "symbols": list(position.get("symbols") or []),
            "entry_equity": entry_equity if entry_equity > 0.0 else None,
            "exit_equity": exit_equity if exit_equity > 0.0 else None,
            "entry_notional": entry_notional,
            "exit_notional": exit_notional,
            "gross_realized_pnl": gross_realized_pnl,
            "gross_realized_return": gross_realized_return,
            "fee_status": fee_status,
            "fee_cost": fee_cost,
            "fee_activity_count": (
                int(fee_summary.get("count") or 0) if fee_summary is not None else None
            ),
            "fee_breakdown": (
                dict(fee_summary.get("breakdown") or {})
                if fee_summary is not None
                else {}
            ),
            "realized_pnl": realized_pnl,
            "realized_return": realized_return,
            "account_equity_change": account_equity_change,
            "unexplained_residual": unexplained_residual,
            "trades": [trade.as_dict() for trade in trades],
            "error": summary.get("error")
        }

        # The trading daemon can copy an open basket into weekend/non-session
        # artifacts before its canonical entry-day summary is updated at the exit.
        # Pick the most authoritative copy instead of whichever directory happens
        # to sort first: a closed basket wins, then its entry-day artifact, then the
        # latest update. Otherwise a Sunday error can hide Monday's realized result.
        priority = (
            position.get("status") == "closed",
            directory.name == entry_date,
            str(summary.get("updated_at") or ""),
        )
        existing = candidates.get(basket_key)
        if existing is None or priority > existing[0]:
            candidates[basket_key] = (priority, record)

    records = [candidate[1] for candidate in candidates.values()]
    records.sort(key=lambda record: str(record.get("trading_day") or ""), reverse=True)
    return records if limit is None else records[:limit]


def build_snapshot(
    client: Any,
    state: Mapping[str, Any],
    inception: date | None = None,
    now: datetime | None = None,
    order_history: Sequence[Mapping[str, Any]] | None = None,
    session_history: Sequence[Mapping[str, Any]] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch everything the dashboard shows and shape it into one Firestore document."""
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)

    account = client.account()
    positions = client.positions()
    market: dict[str, Any] = {}
    try:
        clock = client.clock()
        market.update({
            "is_open": bool(clock.get("is_open")),
            "next_open": clock.get("next_open"),
            "next_close": clock.get("next_close"),
            "timestamp": clock.get("timestamp")
        })
    except Exception:  # noqa: BLE001 - the clock is decoration, not data
        LOGGER.warning("could not read the market clock; publishing without it")
    try:
        calendar = client.calendar(
            reference.date() - timedelta(days=7),
            reference.date() + timedelta(days=14),
        )
        market["sessions"] = [
            {
                "date": str(session["date"]),
                "open": str(session["open"]),
                "close": str(session["close"]),
            }
            for session in calendar
            if session.get("date") and session.get("open") and session.get("close")
        ]
    except Exception:  # noqa: BLE001 - the calendar is dashboard decoration
        LOGGER.warning("could not read the market calendar; publishing without it")

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
    confirmed_series = performance.realized_equity_series(
        session_history or [],
        base_value=performance.strategy_inception_equity(
            session_history or [], history, account_series
        ),
        inception=inception,
    )
    series = performance.realized_equity_series(
        session_history or [],
        base_value=performance.strategy_inception_equity(
            session_history or [], history, account_series
        ),
        inception=inception,
        include_provisional=True,
    )
    provisional_days = {
        str(session.get("exit_date") or session.get("trading_day"))
        for session in session_history or []
        if session.get("status") == "closed"
        and session.get("realized_pnl") is not None
        and session.get("fee_status") in {"pending", "unavailable"}
    }
    trades_by_day: dict[str, int] = {}
    for session in session_history or []:
        if session.get("status") != "closed" or session.get("realized_pnl") is None:
            continue
        exit_day = str(session.get("exit_date") or session.get("trading_day") or "")
        if exit_day:
            trades_by_day[exit_day] = trades_by_day.get(exit_day, 0) + len(
                session.get("trades") or []
            )
    buckets = performance.realized_performance_table(series, now=reference)
    today = reference.date()
    bucket_boundaries = {
        "today": today,
        "week": today - timedelta(days=today.weekday()),
        "month": today.replace(day=1),
        "year": today.replace(month=1, day=1),
        "inception": None,
    }
    performance_payload: dict[str, dict[str, Any]] = {}
    for key, bucket_value in buckets.items():
        boundary = bucket_boundaries[key]
        contains_pending = any(
            (boundary is None or pending_day >= boundary.isoformat())
            and pending_day <= today.isoformat()
            for pending_day in provisional_days
        )
        performance_payload[key] = {
            **bucket_value.as_dict(),
            "status": "provisional" if contains_pending else "confirmed",
        }
    stats = performance.statistics(confirmed_series, baseline_is_first=True)

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
        "configuration": dict(configuration or {}),
        "account": _numeric(account, _ACCOUNT_FIELDS),
        "performance": performance_payload,
        "statistics": stats.as_dict(),
        "equity_curve": [
            {
                "day": point.day.isoformat(),
                "equity": point.equity,
                "profit_loss": point.profit_loss,
                "profit_loss_pct": point.profit_loss_pct,
                "provisional": point.day.isoformat() in provisional_days,
                "trades": trades_by_day.get(point.day.isoformat(), 0),
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
            "mode": _trading_mode(getattr(client, "trading_url", "")),
            "history_period": period,
            "inception_date": series_start.isoformat() if series_start else None
        }
    }


def _firestore_client(config: DictConfig):
    """Initialise the Admin SDK once per process."""
    project_id = str(OmegaConf.select(config, "firebase.project_id"))
    credential = service_account(config)
    if (
        not isinstance(credential, dict)
        and (credential is None or not credential.exists())
        and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    ):
        # Validate configuration before importing the optional dashboard dependency.
        raise DashboardConfigError(
            f"no Firebase service-account key at {credential}. Download one from "
            "Firebase console -> Project settings -> Service accounts, or set "
            "GOOGLE_APPLICATION_CREDENTIALS."
        )

    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
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
    config: DictConfig,
    *,
    trading_mode: str | None = None,
) -> None:
    """Write the snapshot document and the session subcollection."""
    collection = str(OmegaConf.select(config, "firebase.collection"))
    document_name = str(OmegaConf.select(config, "firebase.document"))
    if trading_mode is not None and trading_mode not in {"paper", "live"}:
        raise ValueError(f"unsupported trading mode: {trading_mode}")

    client = _firestore_client(config)
    reference = client.collection(collection).document(document_name)
    sessions_reference = reference.collection(SESSIONS_SUBCOLLECTION)

    # The browser reads one stable document regardless of account mode. If the
    # publisher switches between paper and live, remove the prior mode's derived
    # sessions before writing the new snapshot so histories can never mix.
    previous = reference.get().to_dict() or {}
    previous_mode = str((previous.get("meta") or {}).get("mode") or "")
    mode_changed = bool(
        trading_mode
        and previous_mode in {"paper", "live"}
        and previous_mode != trading_mode
    )
    removed = 0
    if mode_changed:
        existing_sessions = list(sessions_reference.list_documents())
        for offset in range(0, len(existing_sessions), 400):
            cleanup = client.batch()
            for session_reference in existing_sessions[offset : offset + 400]:
                cleanup.delete(session_reference)
                removed += 1
            cleanup.commit()

    reference.set(dict(snapshot))

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
        "published %s snapshot to %s/%s with %d session record(s); "
        "removed %d stale record(s)",
        trading_mode or "account",
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
        help="daemon work directory holding per-day artifacts (default: /data/ppv1/<mode>)"
    )
    parser.add_argument("--state-path", default=None, help="strategy state.json (default: <work-dir>/state.json)")
    parser.add_argument(
        "--trading-config",
        default=str(DEFAULT_TRADING_CONFIG_PATH),
        help="shared overnight live config published with each snapshot",
    )
    parser.add_argument("--dry-run", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--once", action="store_true", help="publish one snapshot and exit")
    parser.add_argument(
        "--refresh-broker-fees",
        action="store_true",
        help="bypass fee cache refresh intervals (use with --once for a one-time refresh)",
    )
    parser.add_argument("--interval-seconds", type=float, default=120.0)
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
    try:
        trading_mode = _trading_mode(trading_url)
    except ValueError as error:
        parser.error(str(error))
    LOGGER.info("detected Alpaca %s mode from %s", trading_mode, trading_url)

    work_dir = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else DEFAULT_WORK_DIRS[trading_mode]
    )
    LOGGER.info("using %s strategy artifacts from %s", trading_mode, work_dir)
    state_path = Path(args.state_path).expanduser() if args.state_path else work_dir / "state.json"
    trading_config_path = Path(args.trading_config).expanduser()
    digest_state_path = (
        Path(args.digest_state_path).expanduser()
        if args.digest_state_path
        else work_dir / ".dashboard-digest-state.json"
    )

    key, secret = load_credentials()
    client = AlpacaClient(key, secret, trading_url)

    def publish_once() -> None:
        reference = datetime.now(tz=UTC)
        state = load_state(state_path)
        configuration = load_trading_configuration(
            trading_config_path,
            work_dir / EFFECTIVE_CONFIG_FILENAME,
        )
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
            fee_client=client,
            force_fee_refresh=args.refresh_broker_fees,
            fees_as_of=reference,
        )
        sessions = all_sessions[:args.sessions_limit]
        snapshot = build_snapshot(
            client,
            state,
            inception=inception,
            now=reference,
            order_history=order_history,
            session_history=all_sessions,
            configuration=configuration,
        )
        if args.dry_run:
            print(json.dumps({"snapshot": snapshot, "sessions": sessions}, indent=2, default=str))
            return
        publish(snapshot, sessions, config, trading_mode=trading_mode)
        if not args.no_email:
            key = dashboard_digest.digest_key(state)
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
