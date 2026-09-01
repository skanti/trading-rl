"""Reconcile completed live baskets with the simulator's observable prices.

The live side uses durable order fills and account snapshots. The modeled side uses
only the split-adjusted minute-bar open at the scheduled entry minute and the primary
condition-O opening auction on the next session, matching ``backtest.py``. Results are
written per session as JSON and CSV so execution drift stays auditable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, date, datetime, time, timedelta
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtest import (
    BAR_ORIGIN,
    DEFAULT_AUCTIONS_PATH,
    DEFAULT_DATA_DIR,
    EASTERN,
    _dataset_manifest,
    _official_opening_auctions,
    _security_symbol,
)
from live import AlpacaClient, completed_liquidity_ranking, load_credentials
from price_utils import forward_fill_positions


DEFAULT_WORK_DIR = Path("/data/ppv1/live")
DEFAULT_TRANSACTION_COST_BPS = 1.0
DATE_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONSOLE = Console()


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def parse_clock(value: str) -> int:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("time must use HH:MM") from error
    if parsed.second or parsed.microsecond:
        raise argparse.ArgumentTypeError("time must use minute precision (HH:MM)")
    return parsed.hour * 60 + parsed.minute


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _timestamp(value: object, label: str) -> datetime:
    if not value:
        raise ValueError(f"missing {label}")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone: {value!r}")
    return parsed


def _clock_text(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def infer_entry_minute(
    summary: Mapping[str, object], override: int | None = None
) -> int:
    """Resolve the scheduled entry minute without confusing it with fill latency."""
    if override is not None:
        return int(override)
    configuration = summary.get("configuration") or {}
    if isinstance(configuration, Mapping) and configuration.get("entry_time"):
        return parse_clock(str(configuration["entry_time"]))
    position = summary.get("position") or {}
    if not isinstance(position, Mapping):
        raise ValueError("summary position must be a mapping")
    target = position.get("entry_dispatch_target_at")
    if target:
        scheduled = _timestamp(target, "entry dispatch target").astimezone(EASTERN)
        return scheduled.hour * 60 + scheduled.minute
    entry_orders = position.get("entry_orders") or {}
    if isinstance(entry_orders, Mapping):
        fills = [
            _timestamp(order.get("filled_at"), f"{symbol} entry fill")
            for symbol, order in entry_orders.items()
            if isinstance(order, Mapping) and order.get("filled_at")
        ]
        if fills:
            first = min(fills).astimezone(EASTERN)
            return first.hour * 60 + first.minute
    raise ValueError("entry time is absent; pass --entry-time")


def discover_closed_summaries(
    work_dir: Path,
) -> dict[date, tuple[Path, dict[str, Any]]]:
    """Return one closed artifact per entry date, preferring its own day directory."""
    found: dict[date, tuple[Path, dict[str, Any]]] = {}
    for directory in sorted(work_dir.iterdir() if work_dir.exists() else []):
        if not directory.is_dir() or not DATE_DIRECTORY.fullmatch(directory.name):
            continue
        path = directory / "summary.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            position = payload.get("position") or {}
            entry_day = parse_day(str(position.get("entry_date")))
        except (AttributeError, OSError, TypeError, ValueError):
            continue
        if position.get("status") != "closed":
            continue
        prior = found.get(entry_day)
        if prior is None or directory.name == entry_day.isoformat():
            found[entry_day] = (path, payload)
    return found


def effective_broker_runtime(work_dir: Path) -> tuple[str, float]:
    path = work_dir / "effective_config.json"
    try:
        payload = json.loads(path.read_text())
        runtime = payload["configuration"]["runtime"]
        trading_url = str(runtime["trading_url"])
        timeout = _number(runtime["request_timeout_seconds"], "request timeout")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read broker endpoint from {path}: {error}") from error
    return trading_url, timeout


def _safe_fee_activity(activity: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for field in (
        "activity_type",
        "activity_sub_type",
        "activity_subtype",
        "date",
        "net_amount",
        "symbol",
        "qty",
        "per_share_amount",
        "order_id",
    ):
        if activity.get(field) is not None:
            output[field] = activity[field]
    description = activity.get("description")
    if description:
        output["description"] = re.sub(
            r"\s+by\s+\S+\s*$", "", str(description), flags=re.IGNORECASE
        )
    return output


def summarize_broker_fees(
    activities: Sequence[Mapping[str, object]],
    exit_day: date,
    *,
    fetched_at: datetime | None = None,
) -> dict[str, object]:
    """Summarize account-level fee activities booked for the exit trade date."""
    selected: list[dict[str, object]] = []
    for activity in activities:
        activity_type = str(activity.get("activity_type") or "").upper()
        activity_day = str(activity.get("date") or "")[:10]
        if activity_type != "FEE" or activity_day != exit_day.isoformat():
            continue
        safe = _safe_fee_activity(activity)
        _number(safe.get("net_amount"), "broker fee net amount")
        selected.append(safe)

    net_amount = sum(
        _number(activity["net_amount"], "broker fee net amount")
        for activity in selected
    )
    breakdown: dict[str, dict[str, float | int]] = {}
    for activity in selected:
        subtype = str(
            activity.get("activity_sub_type")
            or activity.get("activity_subtype")
            or "UNSPECIFIED"
        ).upper()
        item = breakdown.setdefault(subtype, {"count": 0, "net_amount": 0.0})
        item["count"] = int(item["count"]) + 1
        item["net_amount"] = float(item["net_amount"]) + _number(
            activity["net_amount"], "broker fee net amount"
        )
    for item in breakdown.values():
        item["cost"] = -float(item["net_amount"])

    return {
        "status": "complete",
        "source": "alpaca_account_activities",
        "scope": "all account FEE activities whose activity date equals the exit date",
        "activity_date": exit_day.isoformat(),
        "fetched_at": (fetched_at or datetime.now(tz=UTC)).isoformat(),
        "count": len(selected),
        "net_amount": net_amount,
        "cost": -net_amount,
        "breakdown": breakdown,
        "activities": selected,
    }


def attach_broker_fees(
    result: dict[str, object], fee_summary: Mapping[str, object]
) -> None:
    result["broker_fees"] = dict(fee_summary)
    if fee_summary.get("status") != "complete":
        return
    totals = result["totals"]
    if not isinstance(totals, dict):
        raise TypeError("reconciliation totals must be mutable")
    fee_cost = _number(fee_summary.get("cost"), "broker fee cost")
    actual_net = float(totals["actual_gross_pnl"]) - fee_cost
    totals["actual_broker_fee_cost"] = fee_cost
    totals["actual_net_pnl_after_broker_fees"] = actual_net
    totals["actual_minus_simulator_net_pnl"] = actual_net - float(
        totals["simulator_net_pnl"]
    )
    totals["actual_minus_simulator_net_bps"] = (
        float(totals["actual_minus_simulator_net_pnl"])
        / float(totals["actual_entry_notional"])
        * 10_000.0
    )
    broker_pnl = totals.get("broker_equity_pnl")
    totals["broker_minus_fee_adjusted_fill_pnl"] = (
        float(broker_pnl) - actual_net if broker_pnl is not None else None
    )


def minute_open_price(
    data_dir: Path,
    symbol: str,
    session_day: date,
    minute: int,
) -> tuple[float, float]:
    """Return the same causal minute open and staleness used by the simulator."""
    path = data_dir / f"{_security_symbol(symbol)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"minute bars do not exist for {symbol}: {path}")
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.shape[1] < 3 or not len(source):
        raise ValueError(f"invalid minute bars for {symbol}: {path}")
    scheduled = datetime.combine(
        session_day,
        time(minute // 60, minute % 60),
        tzinfo=EASTERN,
    )
    target = int((scheduled.astimezone(UTC) - BAR_ORIGIN).total_seconds())
    position = int(
        forward_fill_positions(source, np.asarray([target], dtype=np.int64), symbol)[0]
    )
    observed = int(source[position, 0])
    price = _number(source[position, 1], f"{symbol} minute open") / 1000.0
    return price, (target - observed) / 60.0


def opening_auction_rows(
    auctions_path: Path,
    exit_day: date,
    symbols: Sequence[str],
) -> dict[str, dict[str, float | str]]:
    official = _official_opening_auctions(
        auctions_path,
        pd.DatetimeIndex([pd.Timestamp(exit_day)]),
        np.asarray(symbols, dtype=str),
    )
    rows: dict[str, dict[str, float | str]] = {}
    for row in official.to_dict("records"):
        symbol = str(row["symbol"]).upper()
        rows[symbol] = {
            "adjusted_price": _number(row["price"], f"{symbol} adjusted auction"),
            "raw_price": _number(row["raw_price"], f"{symbol} raw auction"),
            "exchange": str(row["exchange"]),
        }
    return rows


def _filled_order(
    orders: Mapping[str, object], symbol: str, side: str
) -> tuple[float, float]:
    order = orders.get(symbol)
    if not isinstance(order, Mapping):
        raise ValueError(f"missing {side} order for {symbol}")
    quantity = _number(order.get("filled_qty"), f"{symbol} {side} quantity")
    price = _number(order.get("filled_avg_price"), f"{symbol} {side} price")
    if quantity <= 0.0 or price <= 0.0:
        raise ValueError(f"{symbol} {side} order is not filled")
    return quantity, price


def _equity(position: Mapping[str, object], side: str) -> float | None:
    snapshot = position.get(f"{side}_account_snapshot") or {}
    if not isinstance(snapshot, Mapping) or snapshot.get("equity") is None:
        return None
    try:
        return _number(snapshot["equity"], f"{side} account equity")
    except ValueError:
        return None


def reconcile_execution(
    summary: Mapping[str, object],
    data_dir: Path,
    auctions_path: Path,
    *,
    entry_minute_override: int | None = None,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    max_entry_staleness_minutes: float = 10.0,
) -> dict[str, object]:
    position = summary.get("position") or {}
    if not isinstance(position, Mapping) or position.get("status") != "closed":
        raise ValueError("session does not contain a closed position")
    entry_day = parse_day(str(position.get("entry_date")))
    exit_day = parse_day(str(position.get("exit_date")))
    symbols = [str(value).upper() for value in position.get("symbols") or []]
    if not symbols:
        raise ValueError("closed position has no symbols")
    entry_orders = position.get("entry_orders") or {}
    exit_orders = position.get("exit_orders") or {}
    if not isinstance(entry_orders, Mapping) or not isinstance(exit_orders, Mapping):
        raise ValueError("position orders must be mappings")
    entry_minute = infer_entry_minute(summary, entry_minute_override)
    auctions = opening_auction_rows(auctions_path, exit_day, symbols)

    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for symbol in symbols:
        entry_qty, actual_entry = _filled_order(entry_orders, symbol, "entry")
        exit_qty, actual_exit = _filled_order(exit_orders, symbol, "exit")
        simulated_entry, staleness = minute_open_price(
            data_dir, symbol, entry_day, entry_minute
        )
        if staleness > max_entry_staleness_minutes:
            raise ValueError(
                f"{symbol} entry mark is {staleness:.1f} minutes stale, exceeding "
                f"{max_entry_staleness_minutes:.1f}"
            )
        auction = auctions.get(_security_symbol(symbol))
        if auction is None:
            raise ValueError(
                f"opening auction is unavailable for {symbol} on {exit_day}"
            )
        simulated_exit = float(auction["adjusted_price"])
        raw_auction = float(auction["raw_price"])
        adjustment = raw_auction / simulated_exit
        comparable_entry = simulated_entry * adjustment

        actual_entry_notional = entry_qty * actual_entry
        actual_exit_notional = exit_qty * actual_exit
        actual_pnl = actual_exit_notional - actual_entry_notional
        actual_return = actual_pnl / actual_entry_notional
        simulated_return = simulated_exit / simulated_entry - 1.0
        simulated_pnl = actual_entry_notional * simulated_return
        simulated_exit_notional = actual_entry_notional + simulated_pnl
        entry_execution_pnl_impact = actual_entry_notional * (
            raw_auction / actual_entry - 1.0 - simulated_return
        )
        exit_execution_pnl_impact = entry_qty * (actual_exit - raw_auction)
        quantity_pnl_impact = (exit_qty - entry_qty) * actual_exit
        simulated_cost = (
            float(transaction_cost_bps)
            / 10_000.0
            * (actual_entry_notional + simulated_exit_notional)
        )
        if not math.isclose(entry_qty, exit_qty, rel_tol=1e-8, abs_tol=1e-8):
            warnings.append(
                f"{symbol} quantity changed from {entry_qty:g} to {exit_qty:g}; "
                "a corporate action or partial fill may require order-history reconstruction"
            )
        rows.append(
            {
                "symbol": symbol,
                "entry_quantity": entry_qty,
                "exit_quantity": exit_qty,
                "actual_entry_price": actual_entry,
                "simulator_entry_price": simulated_entry,
                "simulator_entry_price_comparable": comparable_entry,
                "entry_slippage_bps": (actual_entry / comparable_entry - 1.0)
                * 10_000.0,
                "entry_staleness_minutes": staleness,
                "actual_exit_price": actual_exit,
                "simulator_exit_price": simulated_exit,
                "simulator_exit_price_comparable": raw_auction,
                "exit_slippage_bps": (actual_exit / raw_auction - 1.0) * 10_000.0,
                "auction_exchange": auction["exchange"],
                "actual_entry_notional": actual_entry_notional,
                "actual_exit_notional": actual_exit_notional,
                "actual_gross_return": actual_return,
                "actual_gross_pnl": actual_pnl,
                "simulator_gross_return": simulated_return,
                "simulator_gross_pnl": simulated_pnl,
                "simulator_transaction_cost": simulated_cost,
                "simulator_net_pnl": simulated_pnl - simulated_cost,
                "entry_execution_pnl_impact": entry_execution_pnl_impact,
                "exit_execution_pnl_impact": exit_execution_pnl_impact,
                "quantity_pnl_impact": quantity_pnl_impact,
                "actual_minus_simulator_gross_pnl": actual_pnl - simulated_pnl,
                "actual_minus_simulator_bps": (actual_pnl - simulated_pnl)
                / actual_entry_notional
                * 10_000.0,
            }
        )

    def total(field: str) -> float:
        return float(sum(float(row[field]) for row in rows))

    actual_entry_notional = total("actual_entry_notional")
    actual_gross_pnl = total("actual_gross_pnl")
    simulator_gross_pnl = total("simulator_gross_pnl")
    simulator_cost = total("simulator_transaction_cost")
    comparable_entry_notional = sum(
        float(row["entry_quantity"]) * float(row["simulator_entry_price_comparable"])
        for row in rows
    )
    actual_exit_at_entry_quantities = sum(
        float(row["entry_quantity"]) * float(row["actual_exit_price"]) for row in rows
    )
    auction_exit_at_entry_quantities = sum(
        float(row["entry_quantity"]) * float(row["simulator_exit_price_comparable"])
        for row in rows
    )
    entry_execution_pnl_impact = total("entry_execution_pnl_impact")
    exit_execution_pnl_impact = total("exit_execution_pnl_impact")
    quantity_pnl_impact = total("quantity_pnl_impact")
    entry_equity = _equity(position, "entry")
    exit_equity = _equity(position, "exit")
    broker_pnl = (
        exit_equity - entry_equity
        if entry_equity is not None and exit_equity is not None
        else None
    )
    return {
        "entry_date": entry_day.isoformat(),
        "exit_date": exit_day.isoformat(),
        "entry_time": _clock_text(entry_minute),
        "exit_price_source": "primary_opening_auction",
        "transaction_cost_bps_per_side": float(transaction_cost_bps),
        "symbols": symbols,
        "warnings": warnings,
        "totals": {
            "actual_entry_notional": actual_entry_notional,
            "actual_exit_notional": total("actual_exit_notional"),
            "actual_gross_pnl": actual_gross_pnl,
            "actual_gross_return_on_deployed": actual_gross_pnl / actual_entry_notional,
            "simulator_gross_pnl": simulator_gross_pnl,
            "simulator_gross_return_on_deployed": simulator_gross_pnl
            / actual_entry_notional,
            "actual_minus_simulator_gross_pnl": actual_gross_pnl - simulator_gross_pnl,
            "actual_minus_simulator_bps": (actual_gross_pnl - simulator_gross_pnl)
            / actual_entry_notional
            * 10_000.0,
            "entry_execution_slippage_bps": (
                actual_entry_notional / comparable_entry_notional - 1.0
            )
            * 10_000.0,
            "exit_execution_slippage_bps": (
                actual_exit_at_entry_quantities / auction_exit_at_entry_quantities - 1.0
            )
            * 10_000.0,
            "entry_execution_pnl_impact": entry_execution_pnl_impact,
            "exit_execution_pnl_impact": exit_execution_pnl_impact,
            "quantity_pnl_impact": quantity_pnl_impact,
            "simulator_transaction_cost": simulator_cost,
            "simulator_net_pnl": simulator_gross_pnl - simulator_cost,
            "broker_equity_pnl": broker_pnl,
            "broker_minus_actual_fill_pnl": broker_pnl - actual_gross_pnl
            if broker_pnl is not None
            else None,
        },
        "rows": rows,
    }


def _bar_day(value: object) -> date:
    return _timestamp(value, "daily bar timestamp").astimezone(EASTERN).date()


def replay_ranking(
    summary: Mapping[str, object],
    ticks_path: Path,
    *,
    liquidity_scheme_override: str | None = None,
) -> dict[str, object]:
    """Re-run the shared live/simulator ranker from the archived daily-bar dump."""
    position = summary.get("position") or {}
    configuration = summary.get("configuration") or {}
    ranking = summary.get("ranking") or {}
    if not all(
        isinstance(value, Mapping) for value in (position, configuration, ranking)
    ):
        raise ValueError(
            "summary position, configuration, and ranking must be mappings"
        )
    entry_day = parse_day(str(position.get("entry_date")))
    actual = [str(value).upper() for value in position.get("symbols") or []]
    top = int(configuration.get("top") or len(actual))
    scheme_value = (
        liquidity_scheme_override
        or configuration.get("liquidity_scheme")
        or ranking.get("liquidity_scheme")
    )
    warnings: list[str] = []
    if not scheme_value:
        scheme_value = "dollar_ema"
        warnings.append(
            "liquidity scheme was absent and was inferred as legacy dollar_ema"
        )
    scheme = str(scheme_value)
    ema_span = int(
        configuration.get("ema_span") or ranking.get("ema_span_sessions") or 10
    )
    min_history = int(
        configuration.get("min_history_days")
        or ranking.get("minimum_history_sessions")
        or 20
    )
    minimum_trading_days = int(
        configuration.get("minimum_trading_days")
        or ranking.get("minimum_completed_trading_days")
        or 100
    )

    bars: dict[str, list[dict[str, object]]] = {}
    session_days: set[date] = set()
    with ticks_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            symbol = str(row.pop("symbol")).upper()
            row.pop("timeframe", None)
            row.pop("feed", None)
            bars.setdefault(symbol, []).append(row)
            session_days.add(_bar_day(row.get("t")))
    if not bars:
        raise ValueError(f"archived ranking bars are empty: {ticks_path}")
    replayed = completed_liquidity_ranking(
        bars,
        sorted(session_days),
        entry_day,
        ema_span,
        min_history,
        minimum_trading_days,
        scheme,
    )

    issuer_by_symbol = {
        str(candidate.get("symbol", "")).upper(): str(
            candidate.get("issuer") or candidate.get("symbol") or ""
        )
        for candidate in ranking.get("candidates") or []
        if isinstance(candidate, Mapping)
    }
    selected: list[str] = []
    seen_issuers: set[str] = set()
    for symbol, _, _ in replayed:
        issuer = issuer_by_symbol.get(symbol, symbol)
        if issuer in seen_issuers:
            continue
        seen_issuers.add(issuer)
        selected.append(symbol)
        if len(selected) == top:
            break
    actual_set, selected_set = set(actual), set(selected)
    overlap = actual_set & selected_set
    return {
        "status": "complete",
        "ticks_path": str(ticks_path),
        "liquidity_scheme": scheme,
        "ema_span": ema_span,
        "minimum_history_days": min_history,
        "minimum_trading_days": minimum_trading_days,
        "candidate_symbols": len(bars),
        "ranked_symbols": len(replayed),
        "actual_symbols": actual,
        "replayed_symbols": selected,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / len(actual_set) if actual_set else 0.0,
        "jaccard": len(overlap) / len(actual_set | selected_set)
        if actual_set or selected_set
        else 1.0,
        "missing_from_replay": sorted(actual_set - selected_set),
        "extra_in_replay": sorted(selected_set - actual_set),
        "warnings": warnings,
        "top_ranking": [
            {"rank": index + 1, "symbol": symbol, "score": score, "observations": count}
            for index, (symbol, score, count) in enumerate(replayed[: max(top, 20)])
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def broker_fees_for_session(
    client: AlpacaClient | None,
    cache_path: Path,
    exit_day: date,
    *,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, object], str | None]:
    cached: dict[str, object] | None = None
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if (
                isinstance(payload, dict)
                and payload.get("activity_date") == exit_day.isoformat()
                and payload.get("status") == "complete"
            ):
                cached = payload
        except (OSError, TypeError, ValueError):
            cached = None

    if client is None:
        if cached is not None:
            return cached, f"using cached broker fees because {unavailable_reason}"
        return {
            "status": "unavailable",
            "activity_date": exit_day.isoformat(),
            "reason": unavailable_reason or "broker fee retrieval is unavailable",
        }, unavailable_reason

    query_after = exit_day - timedelta(days=1)
    query_until = exit_day + timedelta(days=4)
    try:
        activities = client.account_activities(
            "FEE", after=query_after, until=query_until
        )
        summary = summarize_broker_fees(activities, exit_day)
        summary["query"] = {
            "after": query_after.isoformat(),
            "until": query_until.isoformat(),
        }
        _atomic_json(cache_path, summary)
        return summary, None
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if cached is not None:
            return cached, f"broker fee refresh failed; using cache: {error}"
        return {
            "status": "unavailable",
            "activity_date": exit_day.isoformat(),
            "reason": str(error),
        }, f"broker fees are unavailable: {error}"


def format_usd(value: object, *, signed: bool = False) -> str:
    amount = float(value)
    if signed:
        sign = "+" if amount >= 0 else "-"
        return f"{sign}${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def format_bps(value: object) -> str:
    return f"{float(value):+.2f} bps"


def pnl_comparison_context(
    pnl: object, *, label: str, bps: object | None = None
) -> str:
    difference = float(pnl)
    if abs(difference) < 0.005:
        explanation = f"Simulator {label} matches actual"
    else:
        direction = "lower" if difference > 0.0 else "higher"
        explanation = f"Simulator {label} {direction} by {format_usd(abs(difference))}"
    if bps is not None:
        explanation += f" ({abs(float(bps)):.2f} bps)"
    return explanation


def execution_context(bps: object, *, side: str) -> str:
    price_difference = float(bps)
    if abs(price_difference) < 0.005:
        return f"Simulator {side} price matches actual (0.00 bps)"
    price_direction = "lower" if price_difference > 0.0 else "higher"
    return (
        f"Simulator {side} price {abs(price_difference):.2f} bps "
        f"{price_direction} than actual"
    )


def print_result(
    result: Mapping[str, object], *, show_symbol_breakdown: bool = False
) -> None:
    totals = result["totals"]
    ranking = result.get("ranking_replay") or {}
    ranking_text = "Unavailable"
    if isinstance(ranking, Mapping) and ranking.get("status") == "complete":
        ranking_text = (
            f"{ranking['overlap_count']}/{len(result['symbols'])} · "
            f"{ranking['liquidity_scheme']}"
        )

    comparison = Table(
        title="Live reconciliation",
        caption=f"{len(result['symbols'])} symbols · Ranking replay: {ranking_text}",
        caption_justify="left",
    )
    comparison.add_column("Metric")
    comparison.add_column("Actual", justify="right")
    comparison.add_column("Simulator", justify="right")
    comparison.add_column("Explanation", justify="right")
    comparison.add_row(
        "Gross P&L",
        format_usd(totals["actual_gross_pnl"], signed=True),
        format_usd(totals["simulator_gross_pnl"], signed=True),
        pnl_comparison_context(
            totals["actual_minus_simulator_gross_pnl"],
            label="gross P&L",
            bps=totals["actual_minus_simulator_bps"],
        ),
    )
    modeled_cost = float(totals["simulator_transaction_cost"])
    broker_fees = result.get("broker_fees") or {}
    fee_complete = (
        isinstance(broker_fees, Mapping)
        and broker_fees.get("status") == "complete"
        and totals.get("actual_broker_fee_cost") is not None
    )
    actual_fee_cost = float(totals["actual_broker_fee_cost"]) if fee_complete else None
    comparison.add_row(
        "Costs",
        (
            f"-{format_usd(actual_fee_cost)}"
            if actual_fee_cost
            else "$0.00"
            if actual_fee_cost == 0.0
            else "Unavailable"
        ),
        f"-{format_usd(modeled_cost)}" if modeled_cost else "$0.00",
        ("Actual: Fees*, " if fee_complete else "Actual: unavailable, ")
        + f"Simulator: {float(result['transaction_cost_bps_per_side']):g}bps per side",
    )
    actual_account_result = totals.get("broker_equity_pnl")
    comparison.add_row(
        "Net P&L",
        format_usd(totals["actual_net_pnl_after_broker_fees"], signed=True)
        if fee_complete
        else "Unavailable",
        format_usd(totals["simulator_net_pnl"], signed=True),
        pnl_comparison_context(
            totals["actual_minus_simulator_net_pnl"],
            label="net P&L",
            bps=totals["actual_minus_simulator_net_bps"],
        )
        if fee_complete
        else "Actual fees unavailable",
    )
    if actual_account_result is not None:
        residual_field = (
            "broker_minus_fee_adjusted_fill_pnl"
            if fee_complete
            else "broker_minus_actual_fill_pnl"
        )
        comparison.add_row(
            "Account change",
            format_usd(actual_account_result, signed=True),
            "—",
            "No simulator value · "
            f"{format_usd(totals[residual_field], signed=True)} unexplained residual**",
        )
    comparison.add_row(
        f"Entry · {result['entry_date']}",
        "Fills",
        f"{result['entry_time']} ET bar open",
        execution_context(
            totals["entry_execution_slippage_bps"],
            side="entry",
        ),
    )
    comparison.add_row(
        f"Exit · {result['exit_date']}",
        "Fills",
        "Opening auction",
        execution_context(
            totals["exit_execution_slippage_bps"],
            side="exit",
        ),
    )
    if abs(float(totals["quantity_pnl_impact"])) >= 0.005:
        comparison.add_row(
            "Quantity mismatch",
            "Exit quantities",
            "Entry quantities",
            f"{format_usd(totals['quantity_pnl_impact'], signed=True)} P&L",
        )
    CONSOLE.print(comparison)
    CONSOLE.print(
        "[dim]Gross P&L and execution differences exclude transaction costs. "
        "Entry price differences are actual fills versus minute-bar opens; positive "
        "means paid more. Exit differences are fills versus official opening auctions; "
        "positive means received more.[/dim]"
    )
    if fee_complete:
        CONSOLE.print(
            f"[dim]* Actual cost is the sum of {broker_fees['count']} Alpaca FEE "
            f"activities dated {broker_fees['activity_date']}. Bulk regulatory fees "
            "are account-day level and may not identify individual orders.[/dim]"
        )
    if actual_account_result is not None:
        CONSOLE.print(
            "[dim]** Account equity change is retained as a diagnostic; its residual "
            "can include rounding, timing, or unrelated account activity.[/dim]"
        )

    if show_symbol_breakdown:
        detail = Table(title="Execution differences by symbol")
        detail.add_column("Symbol")
        detail.add_column("Entry price Δ", justify="right")
        detail.add_column("Entry impact", justify="right")
        detail.add_column("Exit price Δ", justify="right")
        detail.add_column("Exit impact", justify="right")
        detail.add_column("Net model Δ", justify="right")
        for row in result["rows"]:
            detail.add_row(
                str(row["symbol"]),
                format_bps(row["entry_slippage_bps"]),
                format_usd(row["entry_execution_pnl_impact"], signed=True),
                format_bps(row["exit_slippage_bps"]),
                format_usd(row["exit_execution_pnl_impact"], signed=True),
                format_usd(row["actual_minus_simulator_gross_pnl"], signed=True),
            )
        CONSOLE.print(detail)
        CONSOLE.print(
            "[dim]Entry price Δ: actual fill vs minute-bar open; positive means "
            "paid more. Exit price Δ: actual fill vs official opening auction; "
            "positive means received more. P&L impacts sum to Actual − simulator.[/dim]"
        )
    for warning in list(result.get("warnings") or []) + list(
        ranking.get("warnings") or [] if isinstance(ranking, Mapping) else []
    ):
        CONSOLE.print(f"[yellow]Warning:[/yellow] {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile completed live fills with simulator minute/auction prices."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--entry-date", type=parse_day, default=None)
    parser.add_argument(
        "--since",
        type=parse_day,
        default=None,
        help="reconcile every closed session on or after YYYY-MM-DD",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--auctions-path", type=Path, default=Path(DEFAULT_AUCTIONS_PATH)
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--entry-time", type=parse_clock, default=None)
    parser.add_argument(
        "--liquidity-scheme",
        choices=("dollar_ema", "turnover_stability"),
        default=None,
        help="override missing or incorrect legacy ranking metadata",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=DEFAULT_TRANSACTION_COST_BPS,
        help="simulator assumption per side; actual fills remain gross",
    )
    parser.add_argument("--max-entry-staleness-minutes", type=float, default=10.0)
    parser.add_argument("--skip-ranking-replay", action="store_true")
    parser.add_argument(
        "--trading-url",
        default=None,
        help="Alpaca trading endpoint; defaults to WORK_DIR/effective_config.json",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=None,
        help="broker fee request timeout; defaults to the effective live config",
    )
    parser.add_argument(
        "--skip-broker-fees",
        action="store_true",
        help="do not query or use cached Alpaca fee activities",
    )
    parser.add_argument(
        "--show-symbol-breakdown",
        action="store_true",
        help="print per-symbol entry/exit price differences and P&L impacts",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.entry_date and args.since:
        parser.error("--entry-date and --since are mutually exclusive")
    if args.transaction_cost_bps < 0.0 or args.max_entry_staleness_minutes < 0.0:
        parser.error("cost and staleness values must be non-negative")
    if args.request_timeout_seconds is not None and args.request_timeout_seconds <= 0.0:
        parser.error("--request-timeout-seconds must be positive")
    _dataset_manifest(args.data_dir, "1Min")
    if not args.auctions_path.exists():
        parser.error(f"auction data does not exist: {args.auctions_path}")
    summaries = discover_closed_summaries(args.work_dir)
    if args.entry_date:
        selected_days = [args.entry_date] if args.entry_date in summaries else []
    elif args.since:
        selected_days = sorted(day for day in summaries if day >= args.since)
    else:
        selected_days = [max(summaries)] if summaries else []
    if not selected_days:
        parser.error("no matching closed live sessions were found")

    broker_client: AlpacaClient | None = None
    broker_unavailable_reason: str | None = None
    if args.skip_broker_fees:
        broker_unavailable_reason = "broker fee retrieval was disabled"
    else:
        try:
            if args.trading_url:
                trading_url = args.trading_url
                timeout = args.request_timeout_seconds or 30.0
            else:
                trading_url, effective_timeout = effective_broker_runtime(args.work_dir)
                timeout = args.request_timeout_seconds or effective_timeout
            key, secret = load_credentials()
            broker_client = AlpacaClient(
                key,
                secret,
                trading_url=trading_url,
                timeout_seconds=timeout,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            broker_unavailable_reason = str(error)

    output_dir = args.output_dir or args.work_dir / "reconciliations"
    failures = 0
    for entry_day in selected_days:
        summary_path, summary = summaries[entry_day]
        try:
            result = reconcile_execution(
                summary,
                args.data_dir,
                args.auctions_path,
                entry_minute_override=args.entry_time,
                transaction_cost_bps=args.transaction_cost_bps,
                max_entry_staleness_minutes=args.max_entry_staleness_minutes,
            )
            if args.skip_broker_fees:
                fee_summary = {
                    "status": "skipped",
                    "activity_date": result["exit_date"],
                    "reason": broker_unavailable_reason,
                }
                fee_warning = None
            else:
                fee_summary, fee_warning = broker_fees_for_session(
                    broker_client,
                    summary_path.parent / "fee_activities.json",
                    parse_day(str(result["exit_date"])),
                    unavailable_reason=broker_unavailable_reason,
                )
            attach_broker_fees(result, fee_summary)
            if fee_warning:
                result["warnings"].append(fee_warning)

            result["version"] = 2
            result["generated_at"] = datetime.now(tz=UTC).isoformat()
            result["live_summary_path"] = str(summary_path)
            if not args.skip_ranking_replay:
                ticks_path = args.work_dir / entry_day.isoformat() / "ticks.jsonl"
                try:
                    result["ranking_replay"] = replay_ranking(
                        summary,
                        ticks_path,
                        liquidity_scheme_override=args.liquidity_scheme,
                    )
                except (OSError, TypeError, ValueError) as error:
                    result["ranking_replay"] = {
                        "status": "unavailable",
                        "error": str(error),
                        "warnings": ["execution reconciliation is still complete"],
                    }
            json_path = output_dir / f"{entry_day.isoformat()}.json"
            csv_path = output_dir / f"{entry_day.isoformat()}.csv"
            _atomic_json(json_path, result)
            _atomic_csv(csv_path, result["rows"])
            print_result(result, show_symbol_breakdown=args.show_symbol_breakdown)
            CONSOLE.print(f"Wrote {json_path} and {csv_path}")
        except (OSError, TypeError, ValueError) as error:
            failures += 1
            CONSOLE.print(f"[red]{entry_day}: reconciliation failed:[/red] {error}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
