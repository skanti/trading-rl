"""Reconcile completed live fills with explicitly selected benchmark prices.

Missing or stale benchmarks exclude the whole session from matched comparisons.
Scheduled mode uses the archived entry schedule and selected exit time; forensic
mode uses the selected minute fields at each actual fill minute.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, date, datetime, time
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from rich.console import Console
from rich.table import Table

from .history import (
    DEFAULT_AUCTIONS_PATH,
    DEFAULT_DATA_DIR,
    EASTERN,
    _dataset_manifest,
)
from .backtest import (
    DEFAULT_NBBO_PATH,
    DEFAULT_EXIT_NBBO_PATH,
    ENTRY_PRICE_SOURCES,
    EXIT_PRICE_SOURCES,
    MINUTE_PRICE_COLUMNS,
    resolve_transaction_cost_bps,
)
from .broker_fees import broker_fees_for_session
from .reconciliation_prices import MissingBenchmarkData, load_benchmark_prices
from .live import (
    OPENING_AUCTION_CUTOFF,
    AlpacaClient,
    completed_liquidity_ranking,
    load_credentials,
)


DEFAULT_WORK_DIR = Path("/data/ppv1/live")
DEFAULT_SCHEDULE_TOLERANCE_MINUTES = 1.0
SCHEDULED_REPORTING_BENCHMARK = "scheduled_strategy"
ACTUAL_TIME_REPORTING_BENCHMARK = "actual_time_1_min"
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
    actual_time_simulator_net = totals.get("actual_time_simulator_net_pnl")
    if actual_time_simulator_net is not None:
        actual_time_net_difference = actual_net - float(actual_time_simulator_net)
        totals["actual_minus_actual_time_simulator_net_pnl"] = (
            actual_time_net_difference
        )
        totals["actual_minus_actual_time_simulator_net_bps"] = (
            actual_time_net_difference
            / float(totals["actual_entry_notional"])
            * 10_000.0
        )
    broker_pnl = totals.get("broker_equity_pnl")
    totals["broker_minus_fee_adjusted_fill_pnl"] = (
        float(broker_pnl) - actual_net if broker_pnl is not None else None
    )


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


def _fill_bar_timestamp(
    orders: Mapping[str, object], symbol: str, side: str
) -> datetime:
    """Return the Eastern timestamp of the 1-min bar containing a fill."""
    order = orders.get(symbol)
    if not isinstance(order, Mapping):
        raise ValueError(f"missing {side} order for {symbol}")
    filled_at = _timestamp(order.get("filled_at"), f"{symbol} {side} fill")
    return filled_at.astimezone(EASTERN).replace(second=0, microsecond=0)


def _order_timestamps(
    orders: Mapping[str, object],
    symbols: Sequence[str],
    side: str,
    field: str,
) -> tuple[dict[str, datetime], list[str]]:
    observed: dict[str, datetime] = {}
    missing: list[str] = []
    for symbol in symbols:
        order = orders.get(symbol)
        if not isinstance(order, Mapping):
            raise ValueError(f"missing {side} order for {symbol}")
        value = order.get(field)
        if not value:
            missing.append(symbol)
            continue
        observed[symbol] = _timestamp(value, f"{symbol} {side} {field}")
    return observed, missing


def _offset_range_text(low: float, high: float) -> str:
    if low >= 0.0:
        return f"{low:.1f}–{high:.1f} minutes after"
    if high <= 0.0:
        return f"{abs(high):.1f}–{abs(low):.1f} minutes before"
    return f"{abs(low):.1f} minutes before to {high:.1f} minutes after"


def execution_timing(
    entry_orders: Mapping[str, object],
    exit_orders: Mapping[str, object],
    symbols: Sequence[str],
    entry_day: date,
    exit_day: date,
    entry_minute: int,
    tolerance_minutes: float,
    *,
    exit_minute: int = 570,
    exit_price_source: str = "opening-auction",
) -> tuple[dict[str, object], list[str]]:
    """Classify whether live fills are comparable to the modeled schedule."""
    if tolerance_minutes < 0.0:
        raise ValueError("schedule tolerance must be non-negative")
    entry_target = datetime.combine(
        entry_day,
        time(entry_minute // 60, entry_minute % 60),
        tzinfo=EASTERN,
    )
    exit_target = datetime.combine(exit_day, time(exit_minute // 60, exit_minute % 60), tzinfo=EASTERN)
    auction_cutoff = datetime.combine(exit_day, OPENING_AUCTION_CUTOFF, tzinfo=EASTERN)
    entry_fills, missing_entry_fills = _order_timestamps(
        entry_orders, symbols, "entry", "filled_at"
    )
    exit_fills, missing_exit_fills = _order_timestamps(
        exit_orders, symbols, "exit", "filled_at"
    )
    exit_submissions, missing_exit_submissions = _order_timestamps(
        exit_orders, symbols, "exit", "submitted_at"
    )

    entry_offset_by_symbol = {
        symbol: (stamp.astimezone(EASTERN) - entry_target).total_seconds() / 60.0
        for symbol, stamp in entry_fills.items()
    }
    exit_offset_by_symbol = {
        symbol: (stamp.astimezone(EASTERN) - exit_target).total_seconds() / 60.0
        for symbol, stamp in exit_fills.items()
    }
    entry_offsets = list(entry_offset_by_symbol.values())
    exit_offsets = list(exit_offset_by_symbol.values())
    entry_exceeded = [
        symbol
        for symbol, offset in entry_offset_by_symbol.items()
        if abs(offset) > tolerance_minutes
    ]
    exit_exceeded = [
        symbol
        for symbol, offset in exit_offset_by_symbol.items()
        if abs(offset) > tolerance_minutes
    ]
    late_exit_submissions = [
        symbol
        for symbol, stamp in exit_submissions.items()
        if stamp.astimezone(EASTERN) >= auction_cutoff
    ]
    entry_fill_comparable = (
        not missing_entry_fills and bool(entry_offsets) and not entry_exceeded
    )
    exit_fill_comparable = (
        not missing_exit_fills and bool(exit_offsets) and not exit_exceeded
    )
    submitted_before_cutoff = (
        None
        if missing_exit_submissions
        else all(
            stamp.astimezone(EASTERN) < auction_cutoff
            for stamp in exit_submissions.values()
        )
    )
    opening_auction_comparable = bool(
        exit_fill_comparable and submitted_before_cutoff is True
    )

    timing: dict[str, object] = {
        "tolerance_minutes": float(tolerance_minutes),
        "schedule_comparable": bool(
            entry_fill_comparable and (opening_auction_comparable
                                       if exit_price_source == "opening-auction" else exit_fill_comparable)
        ),
        "entry": {
            "benchmark": "scheduled_minute_open",
            "scheduled_at": entry_target.isoformat(),
            "first_fill_at": min(entry_fills.values()).astimezone(EASTERN).isoformat()
            if entry_fills
            else None,
            "last_fill_at": max(entry_fills.values()).astimezone(EASTERN).isoformat()
            if entry_fills
            else None,
            "minimum_offset_minutes": min(entry_offsets) if entry_offsets else None,
            "maximum_offset_minutes": max(entry_offsets) if entry_offsets else None,
            "missing_fill_timestamp_count": len(missing_entry_fills),
            "exceeding_tolerance_count": len(entry_exceeded),
            "comparable": entry_fill_comparable,
            "actual_time_benchmark": {
                "alignment": "per_symbol_fill_minute",
                "price_source": "minute-open",
            },
        },
        "exit": {
            "benchmark": exit_price_source,
            "scheduled_at": exit_target.isoformat(),
            "auction_cutoff_at": auction_cutoff.isoformat(),
            "opening_cross_at": datetime.combine(exit_day, time(9, 30), tzinfo=EASTERN).isoformat(),
            "first_submission_at": min(exit_submissions.values())
            .astimezone(EASTERN)
            .isoformat()
            if exit_submissions
            else None,
            "last_submission_at": max(exit_submissions.values())
            .astimezone(EASTERN)
            .isoformat()
            if exit_submissions
            else None,
            "first_fill_at": min(exit_fills.values()).astimezone(EASTERN).isoformat()
            if exit_fills
            else None,
            "last_fill_at": max(exit_fills.values()).astimezone(EASTERN).isoformat()
            if exit_fills
            else None,
            "minimum_offset_minutes": min(exit_offsets) if exit_offsets else None,
            "maximum_offset_minutes": max(exit_offsets) if exit_offsets else None,
            "missing_submission_timestamp_count": len(missing_exit_submissions),
            "late_submission_count": len(late_exit_submissions),
            "missing_fill_timestamp_count": len(missing_exit_fills),
            "exceeding_tolerance_count": len(exit_exceeded),
            "submitted_before_auction_cutoff": submitted_before_cutoff,
            "opening_auction_comparable": opening_auction_comparable,
            "actual_time_benchmark": {
                "alignment": "per_symbol_fill_minute",
                "price_source": "minute-open",
            },
        },
    }

    warnings: list[str] = []
    if not entry_fill_comparable:
        details: list[str] = []
        if missing_entry_fills:
            details.append(
                f"{len(missing_entry_fills)}/{len(symbols)} symbols lack fill timestamps"
            )
        if entry_exceeded:
            details.append(
                f"{len(entry_exceeded)}/{len(symbols)} symbols exceed the "
                f"{tolerance_minutes:g}-min tolerance; fills span "
                f"{_offset_range_text(min(entry_offsets), max(entry_offsets))} "
                f"the scheduled {_clock_text(entry_minute)} ET entry"
            )
        warnings.append(
            "entry is off schedule: "
            + "; ".join(details)
            + "; actual-versus-simulator differences "
            "include timing drift, not only execution slippage"
        )

    if exit_price_source != "opening-auction" and not exit_fill_comparable:
        warnings.append(
            f"exit is off schedule for {exit_price_source} at {_clock_text(exit_minute)} ET: "
            f"{len(missing_exit_fills)} symbols lack fill timestamps; "
            f"{len(exit_exceeded)} exceed the {tolerance_minutes:g}-min tolerance"
        )
    if exit_price_source == "opening-auction" and not opening_auction_comparable:
        reasons: list[str] = []
        if missing_exit_submissions:
            reasons.append(
                f"{len(missing_exit_submissions)}/{len(symbols)} symbols lack "
                "submission timestamps"
            )
        if late_exit_submissions:
            reasons.append(
                f"{len(late_exit_submissions)}/{len(symbols)} orders were submitted "
                f"at or after the {OPENING_AUCTION_CUTOFF.strftime('%H:%M')} ET cutoff"
            )
        if missing_exit_fills:
            reasons.append(
                f"{len(missing_exit_fills)}/{len(symbols)} symbols lack fill timestamps"
            )
        if exit_exceeded:
            reasons.append(
                f"{len(exit_exceeded)}/{len(symbols)} symbols exceed the "
                f"{tolerance_minutes:g}-min tolerance; fills span "
                f"{_offset_range_text(min(exit_offsets), max(exit_offsets))} "
                "the 09:30 ET opening cross"
            )
        warnings.append(
            "exit is not opening-auction comparable: "
            + "; ".join(reasons)
            + "; a comparison with the auction would include timing drift"
        )
    return timing, warnings


def summary_execution_timing(
    summary: Mapping[str, object],
    *,
    entry_minute_override: int | None = None,
    schedule_tolerance_minutes: float = DEFAULT_SCHEDULE_TOLERANCE_MINUTES,
    exit_minute: int = 570,
    exit_price_source: str = "opening-auction",
) -> tuple[dict[str, object], list[str]]:
    """Classify a closed session's timing without requiring market data."""
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
    return execution_timing(
        entry_orders,
        exit_orders,
        symbols,
        entry_day,
        exit_day,
        entry_minute,
        schedule_tolerance_minutes,
        exit_minute=exit_minute,
        exit_price_source=exit_price_source,
    )


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
    nbbo_path: Path | None = None,
    entry_minute_override: int | None = None,
    transaction_cost_bps: float | None = None,
    max_entry_staleness_minutes: float = 1.0,
    schedule_tolerance_minutes: float = DEFAULT_SCHEDULE_TOLERANCE_MINUTES,
    actual_time_benchmark: bool = False,
    entry_price_source: str = "nbbo-ask",
    exit_price_source: str = "opening-auction",
    exit_minute: int = 570,
    exit_nbbo_path: Path | None = None,
    max_exit_staleness_minutes: float = 1.0,
) -> dict[str, object]:
    transaction_cost_bps = resolve_transaction_cost_bps(
        transaction_cost_bps, entry_price_source, exit_price_source,
    )
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
    timing, timing_warnings = summary_execution_timing(
        summary,
        entry_minute_override=entry_minute_override,
        schedule_tolerance_minutes=schedule_tolerance_minutes,
        exit_minute=exit_minute,
        exit_price_source=exit_price_source,
    )
    if entry_price_source not in ENTRY_PRICE_SOURCES or exit_price_source not in EXIT_PRICE_SOURCES:
        raise ValueError("unknown entry or exit price source")
    if exit_price_source == "opening-auction" and exit_minute != 570:
        raise ValueError("opening-auction requires a 09:30 exit")
    if actual_time_benchmark and (
        entry_price_source not in MINUTE_PRICE_COLUMNS or exit_price_source not in MINUTE_PRICE_COLUMNS
    ):
        raise ValueError("actual-time-minute-bar requires explicit minute-* entry and exit sources")
    timing["entry"]["benchmark"] = entry_price_source
    for side, source in (("entry", entry_price_source), ("exit", exit_price_source)):
        timing[side]["actual_time_benchmark"]["price_source"] = source
    calculate_actual_time = actual_time_benchmark
    entry_targets = {symbol: (entry_day, entry_minute) for symbol in symbols}
    exit_targets = {symbol: (exit_day, exit_minute) for symbol in symbols}
    if calculate_actual_time:
        for symbol in symbols:
            for side, orders, targets in (("entry", entry_orders, entry_targets), ("exit", exit_orders, exit_targets)):
                stamp = _fill_bar_timestamp(orders, symbol, side)
                targets[symbol] = (stamp.date(), stamp.hour * 60 + stamp.minute)
    entry_path = nbbo_path if entry_price_source == "nbbo-ask" else data_dir
    exit_path = (auctions_path if exit_price_source == "opening-auction"
                 else exit_nbbo_path if exit_price_source == "nbbo-bid" else data_dir)
    missing = []
    marks = {}
    for side, source, path, targets, max_age in (
        ("entry", entry_price_source, entry_path, entry_targets, max_entry_staleness_minutes),
        ("exit", exit_price_source, exit_path, exit_targets, max_exit_staleness_minutes),
    ):
        try:
            marks[side] = load_benchmark_prices(
                source, path, targets, max_staleness_minutes=max_age, split_path=auctions_path,
            )
        except MissingBenchmarkData as error:
            missing.append(f"{side}: {error}")
    if missing:
        raise MissingBenchmarkData(" | ".join(missing))

    rows: list[dict[str, object]] = []
    warnings = list(timing_warnings)
    quantity_mismatch_count = 0
    for symbol in symbols:
        entry_qty, actual_entry = _filled_order(entry_orders, symbol, "entry")
        exit_qty, actual_exit = _filled_order(exit_orders, symbol, "exit")
        entry_mark = marks["entry"][symbol]
        exit_mark = marks["exit"][symbol]
        simulated_entry = entry_mark.price
        comparable_entry = entry_mark.raw_price
        simulated_exit = exit_mark.price
        comparable_exit = exit_mark.raw_price
        staleness = entry_mark.staleness_minutes
        scheduled_entry_source = entry_price_source
        actual_time_entry_at = _fill_bar_timestamp(entry_orders, symbol, "entry") if calculate_actual_time else None
        actual_time_exit_at = _fill_bar_timestamp(exit_orders, symbol, "exit") if calculate_actual_time else None
        actual_time_entry = simulated_entry if calculate_actual_time else None
        actual_time_exit = simulated_exit if calculate_actual_time else None
        actual_time_entry_comparable = comparable_entry if calculate_actual_time else None
        actual_time_exit_comparable = comparable_exit if calculate_actual_time else None
        actual_time_entry_staleness = staleness if calculate_actual_time else None
        actual_time_exit_staleness = exit_mark.staleness_minutes if calculate_actual_time else None
        actual_time_entry_slippage = (actual_entry / comparable_entry - 1.0) * 10_000 if calculate_actual_time else None
        actual_time_exit_slippage = (actual_exit / comparable_exit - 1.0) * 10_000 if calculate_actual_time else None
        actual_time_simulated_return = None
        actual_time_simulated_pnl = None
        actual_time_simulated_cost = None
        actual_time_entry_execution_pnl_impact = None
        actual_time_exit_execution_pnl_impact = None

        actual_entry_notional = entry_qty * actual_entry
        actual_exit_notional = exit_qty * actual_exit
        actual_pnl = actual_exit_notional - actual_entry_notional
        actual_return = actual_pnl / actual_entry_notional
        simulated_return = simulated_exit / simulated_entry - 1.0
        simulated_pnl = actual_entry_notional * simulated_return
        simulated_exit_notional = actual_entry_notional + simulated_pnl
        entry_execution_pnl_impact = actual_entry_notional * (
            comparable_exit / actual_entry - 1.0 - simulated_return
        )
        exit_execution_pnl_impact = entry_qty * (actual_exit - comparable_exit)
        quantity_pnl_impact = (exit_qty - entry_qty) * actual_exit
        simulated_cost = (
            float(transaction_cost_bps)
            / 10_000.0
            * (actual_entry_notional + simulated_exit_notional)
        )
        if (
            actual_time_entry is not None
            and actual_time_entry_comparable is not None
            and actual_time_exit is not None
            and actual_time_exit_comparable is not None
        ):
            actual_time_simulated_return = actual_time_exit / actual_time_entry - 1.0
            actual_time_simulated_pnl = (
                actual_entry_notional * actual_time_simulated_return
            )
            actual_time_simulated_exit_notional = (
                actual_entry_notional + actual_time_simulated_pnl
            )
            actual_time_simulated_cost = (
                float(transaction_cost_bps)
                / 10_000.0
                * (actual_entry_notional + actual_time_simulated_exit_notional)
            )
            actual_time_entry_execution_pnl_impact = actual_entry_notional * (
                actual_time_exit_comparable / actual_entry
                - 1.0
                - actual_time_simulated_return
            )
            actual_time_exit_execution_pnl_impact = entry_qty * (
                actual_exit - actual_time_exit_comparable
            )
        if not math.isclose(entry_qty, exit_qty, rel_tol=1e-8, abs_tol=1e-8):
            quantity_mismatch_count += 1
        rows.append(
            {
                "symbol": symbol,
                "entry_quantity": entry_qty,
                "exit_quantity": exit_qty,
                "actual_entry_price": actual_entry,
                "simulator_entry_price": simulated_entry,
                "simulator_entry_source": scheduled_entry_source,
                "simulator_entry_price_comparable": comparable_entry,
                "entry_slippage_bps": (actual_entry / comparable_entry - 1.0)
                * 10_000.0,
                "entry_staleness_minutes": staleness,
                "actual_time_entry_bar_at": actual_time_entry_at.isoformat()
                if actual_time_entry_at is not None
                else None,
                "actual_time_entry_price": actual_time_entry,
                "actual_time_entry_price_comparable": actual_time_entry_comparable,
                "actual_time_entry_staleness_minutes": actual_time_entry_staleness,
                "actual_time_entry_slippage_bps": actual_time_entry_slippage,
                "actual_exit_price": actual_exit,
                "simulator_exit_price": simulated_exit,
                "simulator_exit_price_comparable": comparable_exit,
                "exit_slippage_bps": (actual_exit / comparable_exit - 1.0) * 10_000.0,
                "actual_time_exit_bar_at": actual_time_exit_at.isoformat()
                if actual_time_exit_at is not None
                else None,
                "actual_time_exit_price": actual_time_exit,
                "actual_time_exit_price_comparable": actual_time_exit_comparable,
                "actual_time_exit_staleness_minutes": actual_time_exit_staleness,
                "actual_time_exit_slippage_bps": actual_time_exit_slippage,
                "actual_time_simulator_gross_return": actual_time_simulated_return,
                "actual_time_simulator_gross_pnl": actual_time_simulated_pnl,
                "actual_time_simulator_transaction_cost": actual_time_simulated_cost,
                "actual_time_simulator_net_pnl": (
                    actual_time_simulated_pnl - actual_time_simulated_cost
                    if actual_time_simulated_pnl is not None
                    and actual_time_simulated_cost is not None
                    else None
                ),
                "actual_minus_actual_time_simulator_gross_pnl": (
                    actual_pnl - actual_time_simulated_pnl
                    if actual_time_simulated_pnl is not None
                    else None
                ),
                "actual_time_entry_execution_pnl_impact": (
                    actual_time_entry_execution_pnl_impact
                ),
                "actual_time_exit_execution_pnl_impact": (
                    actual_time_exit_execution_pnl_impact
                ),
                "exit_exchange": exit_mark.exchange,
                "simulator_exit_source": exit_price_source,
                "exit_staleness_minutes": exit_mark.staleness_minutes,
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

    if quantity_mismatch_count:
        warnings.append(
            f"entry and exit quantities differ for {quantity_mismatch_count}/"
            f"{len(symbols)} symbols; a corporate action or partial fill may require "
            "order-history reconstruction"
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
    actual_time_benchmarks_available = calculate_actual_time
    actual_time_entry_at_entry_quantities = (
        sum(
            float(row["entry_quantity"])
            * float(row["actual_time_entry_price_comparable"])
            for row in rows
        )
        if actual_time_benchmarks_available
        else None
    )
    actual_time_exit_at_entry_quantities = (
        sum(
            float(row["entry_quantity"])
            * float(row["actual_time_exit_price_comparable"])
            for row in rows
        )
        if actual_time_benchmarks_available
        else None
    )
    actual_time_simulator_gross_pnl = (
        sum(float(row["actual_time_simulator_gross_pnl"]) for row in rows)
        if actual_time_exit_at_entry_quantities is not None
        else None
    )
    actual_time_simulator_cost = (
        sum(float(row["actual_time_simulator_transaction_cost"]) for row in rows)
        if actual_time_exit_at_entry_quantities is not None
        else None
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
        "entry_price_source": entry_price_source,
        "exit_price_source": exit_price_source,
        "exit_time": _clock_text(exit_minute),
        "reporting_benchmark": ACTUAL_TIME_REPORTING_BENCHMARK if calculate_actual_time else SCHEDULED_REPORTING_BENCHMARK,
        "transaction_cost_bps_per_side": float(transaction_cost_bps),
        "symbols": symbols,
        "timing": timing,
        "warnings": warnings,
        "totals": {
            "entry_equity": entry_equity,
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
            "actual_time_exit_slippage_bps": (
                (
                    actual_exit_at_entry_quantities
                    / actual_time_exit_at_entry_quantities
                    - 1.0
                )
                * 10_000.0
                if actual_time_exit_at_entry_quantities is not None
                else None
            ),
            "actual_time_entry_slippage_bps": (
                (actual_entry_notional / actual_time_entry_at_entry_quantities - 1.0)
                * 10_000.0
                if actual_time_entry_at_entry_quantities is not None
                else None
            ),
            "actual_time_simulator_gross_pnl": actual_time_simulator_gross_pnl,
            "actual_time_simulator_gross_return_on_deployed": (
                actual_time_simulator_gross_pnl / actual_entry_notional
                if actual_time_simulator_gross_pnl is not None
                else None
            ),
            "actual_minus_actual_time_simulator_gross_pnl": (
                actual_gross_pnl - actual_time_simulator_gross_pnl
                if actual_time_simulator_gross_pnl is not None
                else None
            ),
            "actual_minus_actual_time_simulator_bps": (
                (actual_gross_pnl - actual_time_simulator_gross_pnl)
                / actual_entry_notional
                * 10_000.0
                if actual_time_simulator_gross_pnl is not None
                else None
            ),
            "actual_time_simulator_transaction_cost": actual_time_simulator_cost,
            "actual_time_simulator_net_pnl": (
                actual_time_simulator_gross_pnl - actual_time_simulator_cost
                if actual_time_simulator_gross_pnl is not None
                and actual_time_simulator_cost is not None
                else None
            ),
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


def _matches_archived_ranking(
    replayed: Sequence[tuple[str, float, int]],
    candidates: object,
) -> bool:
    """Return whether archived rank/score evidence identifies this replay."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return False
    comparable = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        try:
            rank = int(candidate.get("rank"))
            symbol = str(candidate.get("symbol") or "").upper()
            score = float(candidate.get("score"))
        except (TypeError, ValueError):
            continue
        if rank < 1 or rank > len(replayed) or not symbol or not math.isfinite(score):
            return False
        replayed_symbol, replayed_score, _ = replayed[rank - 1]
        if symbol != replayed_symbol or not math.isclose(
            score, replayed_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            return False
        comparable += 1
    # Several exact ranks and floating-point scores are strong enough to distinguish
    # the formulas without silently trusting one coincidental top name.
    return comparable >= 3


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
    configured_scheme = configuration.get("liquidity_scheme") or ranking.get(
        "liquidity_scheme"
    )
    scheme_value = liquidity_scheme_override or configured_scheme
    scheme_source = (
        "cli_override"
        if liquidity_scheme_override
        else "session_metadata"
        if configured_scheme
        else None
    )
    warnings: list[str] = []
    ema_span = int(
        configuration.get("ema_span") or ranking.get("ema_span_sessions") or 10
    )
    pipeline_version = ranking.get("ranking_pipeline_version")
    dispersion_span = (
        20 if pipeline_version is None or int(pipeline_version) <= 6 else ema_span
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

    def rank(scheme_name: str) -> list[tuple[str, float, int]]:
        return completed_liquidity_ranking(
            bars,
            sorted(session_days),
            entry_day,
            ema_span,
            min_history,
            minimum_trading_days,
            scheme_name,
            dispersion_span=dispersion_span,
        )

    replayed: list[tuple[str, float, int]]
    if scheme_value:
        scheme = str(scheme_value)
        replayed = rank(scheme)
    else:
        possible = {
            candidate_scheme: rank(candidate_scheme)
            for candidate_scheme in ("dollar_ema", "turnover_stability")
        }
        matches = [
            candidate_scheme
            for candidate_scheme, candidate_ranking in possible.items()
            if _matches_archived_ranking(candidate_ranking, ranking.get("candidates"))
        ]
        if len(matches) == 1:
            scheme = matches[0]
            replayed = possible[scheme]
            scheme_source = "archived_ranking_scores"
        else:
            scheme = "dollar_ema"
            replayed = possible[scheme]
            scheme_source = "legacy_default"
            warnings.append(
                "session metadata omitted the liquidity scheme and archived ranking "
                "scores could not identify it; assumed dollar_ema (override with "
                "--liquidity-scheme)"
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
        "liquidity_scheme_source": scheme_source,
        "ema_span": ema_span,
        "dispersion_span": dispersion_span,
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


def format_usd(value: object, *, signed: bool = False) -> str:
    amount = float(value)
    if signed:
        sign = "+" if amount >= 0 else "-"
        return f"{sign}${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def format_bps(value: object) -> str:
    return f"{float(value):+.2f} bps"


def simulator_price_difference_bps(actual: float, simulator: float) -> float:
    """Use actual fills as the price reference; positive means simulator is higher."""
    return (simulator / actual - 1) * 10_000


def pnl_comparison_context(
    pnl: object, *, label: str, bps: object | None = None
) -> str:
    difference = float(pnl)
    if abs(difference) < 0.005:
        explanation = f"Sim {label} = actual"
    else:
        comparison = "<" if difference > 0.0 else ">"
        explanation = (
            f"Sim {label} {comparison} actual by {format_usd(abs(difference))}"
        )
    if bps is not None:
        explanation += f" ({abs(float(bps)):.2f} bps)"
    return explanation


def _add_pnl_percentage_row(
    table: Table, label: str, actual: float | None, simulator: float, starting_equity: float | None,
) -> None:
    equity_available = starting_equity is not None and starting_equity > 0
    table.add_row(
        label,
        f"{actual / starting_equity:+.2%}" if equity_available and actual is not None else "Unavailable",
        f"{simulator / starting_equity:+.2%}" if equity_available else "Unavailable",
        f"Based on {format_usd(starting_equity)} starting equity" if equity_available else "Starting account equity unavailable",
    )


def _add_gross_pnl_rows(
    table: Table, actual: float, simulator: float, starting_equity: float | None,
) -> None:
    difference = actual - simulator
    equity_available = starting_equity is not None and starting_equity > 0
    difference_bps = difference / starting_equity * 10_000 if equity_available else None
    table.add_row(
        "Gross P&L",
        format_usd(actual, signed=True),
        format_usd(simulator, signed=True),
        pnl_comparison_context(difference, label="gross P&L", bps=difference_bps),
    )
    _add_pnl_percentage_row(table, "Gross P&L (%)", actual, simulator, starting_equity)


def execution_context(bps: object, *, side: str) -> str:
    price_difference = float(bps)
    if abs(price_difference) < 0.005:
        return f"Sim {side} price = actual (0.00 bps)"
    comparison = "<" if price_difference > 0.0 else ">"
    return f"Sim {side} price {comparison} actual by {abs(price_difference):.2f} bps"


def select_reporting_benchmark(
    result: dict[str, object], *, prefer_actual_time: bool
) -> str:
    """Select the headline benchmark while preserving both result families."""
    totals = result.get("totals") or {}
    actual_time_available = isinstance(totals, Mapping) and (
        totals.get("actual_time_simulator_gross_pnl") is not None
    )
    if prefer_actual_time and not actual_time_available:
        raise ValueError(
            "actual-time reconciliation requires complete per-symbol entry and exit "
            "1-min benchmarks"
        )
    benchmark = (
        ACTUAL_TIME_REPORTING_BENCHMARK
        if prefer_actual_time
        else SCHEDULED_REPORTING_BENCHMARK
    )
    result["reporting_benchmark"] = benchmark
    return benchmark


def _reported_total(
    result: Mapping[str, object], scheduled_field: str, actual_time_field: str
) -> float:
    totals = result.get("totals")
    if not isinstance(totals, Mapping):
        raise TypeError("reconciliation totals must be a mapping")
    field = (
        actual_time_field
        if result.get("reporting_benchmark") == ACTUAL_TIME_REPORTING_BENCHMARK
        else scheduled_field
    )
    return _number(totals.get(field), field)


def summarize_execution_prices(results: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Compare raw prices with identical entry-share weights on both sides.

    Pool notionals before calculating bps so the reported price averages reproduce
    the difference. Entry quantities isolate price effects from quantity changes.
    """
    prices: dict[str, float] = {}
    for side in ("entry", "exit"):
        observations = []
        for result in results:
            prefix = (
                "actual_time" if result.get("reporting_benchmark") == ACTUAL_TIME_REPORTING_BENCHMARK
                else "simulator"
            )
            impact_field = f"{'actual_time_' if prefix == 'actual_time' else ''}{side}_execution_pnl_impact"
            for row in result["rows"]:
                quantity = _number(row["entry_quantity"], "entry quantity")
                actual = _number(row[f"actual_{side}_price"], f"actual {side} price")
                simulated = _number(row[f"{prefix}_{side}_price_comparable"], f"comparable {side} price")
                if min(quantity, actual, simulated) <= 0:
                    raise ValueError("execution quantities and prices must be positive")
                impact = _number(row[impact_field], impact_field)
                observations.append((quantity, quantity * actual, quantity * simulated, impact))
        shares = math.fsum(value[0] for value in observations)
        if not shares:
            raise ValueError("execution price averages require filled trades")
        actual_notional = math.fsum(value[1] for value in observations)
        simulated_notional = math.fsum(value[2] for value in observations)
        prices[f"actual_{side}_price_average"] = actual_notional / shares
        prices[f"simulator_{side}_price_average"] = simulated_notional / shares
        prices[f"{side}_execution_slippage_bps"] = (actual_notional / simulated_notional - 1) * 10_000
        prices[f"{side}_execution_pnl_impact"] = math.fsum(value[3] for value in observations)
    deployed = math.fsum(float(result["totals"]["actual_entry_notional"]) for result in results)
    for side in ("entry", "exit"):
        prices[f"{side}_execution_pnl_impact_bps"] = prices[f"{side}_execution_pnl_impact"] / deployed * 10_000
    return prices


def _add_execution_price_rows(table: Table, prices: Mapping[str, float]) -> None:
    for side in ("entry", "exit"):
        difference = simulator_price_difference_bps(
            prices[f"actual_{side}_price_average"], prices[f"simulator_{side}_price_average"],
        )
        table.add_row(
            f"{side.capitalize()} price", "0.00 bps", format_bps(difference),
            "Simulator relative to actual",
        )


def summarize_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate reconciliations without letting small sessions dominate bps."""
    if not results:
        raise ValueError("at least one reconciliation result is required")

    def totals(result: Mapping[str, object]) -> Mapping[str, object]:
        value = result.get("totals")
        if not isinstance(value, Mapping):
            raise TypeError("reconciliation totals must be a mapping")
        return value

    def total(field: str, selected: Sequence[Mapping[str, object]] = results) -> float:
        return sum(_number(totals(result).get(field), field) for result in selected)

    def reported_total(scheduled_field: str, actual_time_field: str) -> float:
        return sum(
            _reported_total(result, scheduled_field, actual_time_field)
            for result in results
        )

    fee_confirmed = [
        result
        for result in results
        if isinstance(result.get("broker_fees"), Mapping)
        and result["broker_fees"].get("status") == "complete"  # type: ignore[index,union-attr]
        and totals(result).get("actual_broker_fee_cost") is not None
    ]
    account_observed = [
        result
        for result in results
        if totals(result).get("broker_equity_pnl") is not None
    ]
    residual_observed = [
        result
        for result in fee_confirmed
        if totals(result).get("broker_minus_fee_adjusted_fill_pnl") is not None
    ]
    ranking_observed = [
        result
        for result in results
        if isinstance(result.get("ranking_replay"), Mapping)
        and result["ranking_replay"].get("status") == "complete"  # type: ignore[index,union-attr]
    ]

    count = len(results)
    actual_time_benchmark_sessions = sum(
        result.get("reporting_benchmark") == ACTUAL_TIME_REPORTING_BENCHMARK
        for result in results
    )
    scheduled_entry_sources = sorted(
        {str(result.get("entry_price_source") or "unspecified") for result in results}
    )
    entry_notional = total("actual_entry_notional")
    if entry_notional <= 0.0:
        raise ValueError("aggregate entry notional must be positive")
    actual_gross = total("actual_gross_pnl")
    simulator_gross = reported_total(
        "simulator_gross_pnl", "actual_time_simulator_gross_pnl"
    )
    gross_difference = actual_gross - simulator_gross
    first_result = min(results, key=lambda result: str(result["entry_date"]))
    summary: dict[str, object] = {
        "sessions": count,
        "scheduled_entry_price_sources": scheduled_entry_sources,
        "scheduled_exit_price_sources": sorted({str(result.get("exit_price_source") or "unspecified") for result in results}),
        "trades": sum(len(result.get("symbols") or []) for result in results),
        "first_entry_date": min(str(result["entry_date"]) for result in results),
        "last_exit_date": max(str(result["exit_date"]) for result in results),
        "actual_entry_notional_total": entry_notional,
        "starting_equity": totals(first_result).get("entry_equity"),
        "actual_gross_pnl_total": actual_gross,
        "simulator_gross_pnl_total": simulator_gross,
        "actual_minus_simulator_gross_pnl_total": gross_difference,
        "actual_minus_simulator_gross_bps": gross_difference
        / entry_notional
        * 10_000.0,
        **summarize_execution_prices(results),
        "quantity_pnl_impact": total("quantity_pnl_impact"),
        "actual_time_benchmark_sessions": actual_time_benchmark_sessions,
        "fee_confirmed_sessions": len(fee_confirmed),
        "account_observed_sessions": len(account_observed),
        "residual_observed_sessions": len(residual_observed),
        "transaction_cost_bps_per_side": sorted(
            {
                _number(
                    result.get("transaction_cost_bps_per_side"),
                    "transaction cost bps per side",
                )
                for result in results
            }
        ),
    }
    # Gross, modeled costs, and net must cover the same sessions. Dropping
    # pending-fee sessions from net alone can hide losses and inflate returns.
    summary["simulator_transaction_cost_total"] = reported_total(
        "simulator_transaction_cost",
        "actual_time_simulator_transaction_cost",
    )
    simulator_net = reported_total("simulator_net_pnl", "actual_time_simulator_net_pnl")
    summary["simulator_net_pnl_total"] = simulator_net
    if fee_confirmed:
        summary["actual_broker_fee_cost_total"] = total(
            "actual_broker_fee_cost", fee_confirmed
        )
    if len(fee_confirmed) == count:
        actual_net = total("actual_net_pnl_after_broker_fees")
        net_difference = actual_net - simulator_net
        summary.update(
            {
                "actual_net_pnl_total": actual_net,
                "actual_minus_simulator_net_pnl_total": net_difference,
                "actual_minus_simulator_net_bps": net_difference
                / entry_notional
                * 10_000.0,
            }
        )
    if account_observed:
        summary["broker_equity_pnl_total"] = total(
            "broker_equity_pnl", account_observed
        )
    if residual_observed:
        summary["unexplained_residual_total"] = total(
            "broker_minus_fee_adjusted_fill_pnl", residual_observed
        )
    if ranking_observed:
        compared_symbols = sum(
            len(result.get("symbols") or []) for result in ranking_observed
        )
        overlap = sum(
            int(result["ranking_replay"]["overlap_count"])  # type: ignore[index]
            for result in ranking_observed
        )
        summary["ranking_observed_sessions"] = len(ranking_observed)
        summary["ranking_overlap_fraction"] = (
            overlap / compared_symbols if compared_symbols else 0.0
        )
    return summary


def print_overview(
    results: Sequence[Mapping[str, object]],
    *,
    skipped_sessions: Sequence[tuple[date, Sequence[str]]] = (),
    missing_sessions: Sequence[tuple[date, str]] = (),
) -> None:
    summary = summarize_results(results)
    sessions = int(summary["sessions"])
    fee_sessions = int(summary["fee_confirmed_sessions"])
    actual_time_sessions = int(summary["actual_time_benchmark_sessions"])
    scheduled_sources = summary["scheduled_entry_price_sources"]
    scheduled_entry_label = ", ".join(scheduled_sources)
    scheduled_exit_label = ", ".join(summary["scheduled_exit_price_sources"])
    comparison = Table(
        title="Reconciliation overview",
        caption=f"{summary['first_entry_date']} → {summary['last_exit_date']}",
        caption_justify="left",
    )
    comparison.add_column("Metric")
    comparison.add_column("Actual", justify="right")
    comparison.add_column("Simulator", justify="right")
    comparison.add_column("Difference / context", justify="right")
    _add_gross_pnl_rows(
        comparison,
        summary["actual_gross_pnl_total"],
        summary["simulator_gross_pnl_total"],
        summary["starting_equity"],
    )
    comparison.add_row(
        f"Costs ({fee_sessions}/{sessions} actual confirmed)",
        (
            f"-{format_usd(summary['actual_broker_fee_cost_total'])}"
            if summary.get("actual_broker_fee_cost_total")
            else "$0.00"
            if summary.get("actual_broker_fee_cost_total") == 0.0
            else "Unavailable"
        ),
        (
            f"-{format_usd(summary['simulator_transaction_cost_total'])}"
            if summary.get("simulator_transaction_cost_total")
            else "$0.00"
        ),
        "Actual: Fees*, Sim: "
        + "/".join(
            f"{value:g}bps" for value in summary["transaction_cost_bps_per_side"]
        )
        + " per side",
    )
    comparison.add_row(
        "Net P&L",
        format_usd(summary["actual_net_pnl_total"], signed=True)
        if summary.get("actual_net_pnl_total") is not None
        else "Unavailable",
        format_usd(summary["simulator_net_pnl_total"], signed=True),
        pnl_comparison_context(
            summary["actual_minus_simulator_net_pnl_total"],
            label="net P&L",
            bps=summary["actual_minus_simulator_net_bps"],
        )
        if summary.get("actual_minus_simulator_net_pnl_total") is not None
        else f"Actual fees confirmed for {fee_sessions}/{sessions} sessions",
    )
    _add_pnl_percentage_row(
        comparison, "Net P&L (%)",
        summary.get("actual_net_pnl_total"), summary["simulator_net_pnl_total"], summary["starting_equity"],
    )
    _add_execution_price_rows(comparison, summary)
    if abs(float(summary["quantity_pnl_impact"])) >= 0.005:
        comparison.add_row(
            "Quantity P&L impact", "—", "—",
            format_usd(summary["quantity_pnl_impact"], signed=True),
        )
    account_sessions = int(summary["account_observed_sessions"])
    if account_sessions:
        comparison.add_row(
            f"Account change ({account_sessions}/{sessions})",
            format_usd(summary["broker_equity_pnl_total"], signed=True),
            "—",
            "Total account equity change",
        )
    residual_sessions = int(summary["residual_observed_sessions"])
    if residual_sessions:
        comparison.add_row(
            f"Unexplained residual ({residual_sessions}/{sessions})",
            format_usd(summary["unexplained_residual_total"], signed=True),
            "—",
            "Account Δ − fee-adjusted fill P&L",
        )
    comparison.add_row(
        "Sessions",
        str(sessions),
        str(sessions),
        "100% matched baskets",
    )
    if skipped_sessions:
        skipped_contexts: list[str] = []
        for entry_day, warnings in skipped_sessions:
            entry_mismatch = any(
                warning.startswith("entry is off schedule") for warning in warnings
            )
            exit_mismatch = any(
                warning.startswith(("exit is not opening-auction comparable", "exit is off schedule"))
                for warning in warnings
            )
            if entry_mismatch and exit_mismatch:
                reason = "entry/exit schedule mismatch"
            elif entry_mismatch:
                reason = "entry schedule mismatch"
            elif exit_mismatch:
                reason = "exit schedule mismatch"
            else:
                reason = "schedule mismatch"
            skipped_contexts.append(f"{entry_day.isoformat()}: {reason}")
        displayed_context = "; ".join(skipped_contexts[:3])
        if len(skipped_contexts) > 3:
            displayed_context += f"; +{len(skipped_contexts) - 3} more"
        comparison.add_row(
            "Skipped sessions",
            str(len(skipped_sessions)),
            "—",
            displayed_context,
        )
    if missing_sessions:
        comparison.add_row("Missing-price sessions", str(len(missing_sessions)), "Excluded",
                           "Whole baskets excluded from both actual and simulated totals")
    comparison.add_row(
        "Trades",
        str(summary["trades"]),
        str(summary["trades"]),
        "100% matched trades",
    )
    comparison.add_section()
    comparison.add_row(
        "Entry price source",
        "Fills",
        (
            scheduled_entry_label
            if not actual_time_sessions
            else f"At fill minute: {scheduled_entry_label}"
            if actual_time_sessions == sessions
            else f"Per-session ({actual_time_sessions} actual-time)"
        ),
        "",
    )
    comparison.add_row(
        "Exit price source",
        "Fills",
        (
            scheduled_exit_label
            if not actual_time_sessions
            else f"At fill minute: {scheduled_exit_label}"
            if actual_time_sessions == sessions
            else f"Per-session ({actual_time_sessions} actual-time)"
        ),
        "",
    )
    if summary.get("ranking_observed_sessions"):
        ranking_sessions = int(summary["ranking_observed_sessions"])
        comparison.add_row(
            f"Ranking replay ({ranking_sessions}/{sessions})",
            "Actual basket",
            "Replayed basket",
            f"{float(summary['ranking_overlap_fraction']):.1%} symbol overlap",
        )
    CONSOLE.print(comparison)
    CONSOLE.print(
        "[dim]Dollar P&L values are cumulative totals. P&L (%) divides the corresponding dollar P&L "
        "by account equity before the first included entry. Only matched sessions contribute P&L. "
        "Price bps = (simulator / actual − 1) × 10,000, using raw prices and identical "
        "entry-share weights for both sides. Positive means the simulator price is higher. "
        "Gross P&L and execution differences exclude transaction costs.[/dim]"
    )
    if any(source in MINUTE_PRICE_COLUMNS and source != "minute-open"
           for source in [*scheduled_sources, *summary["scheduled_exit_price_sources"]]):
        CONSOLE.print("[dim]Minute high/low/close/VWAP model hypothetical fills over the selected minute; "
                      "these values are known only at its end.[/dim]")
    if actual_time_sessions:
        CONSOLE.print(
            f"[dim]The simulator uses the selected minute fields at each actual fill minute "
            f"for {actual_time_sessions}/{sessions} sessions.[/dim]"
        )
    CONSOLE.print(
        f"[dim]* Actual costs use Alpaca FEE activities for {fee_sessions}/{sessions} "
        "fee-confirmed sessions. Simulator costs and net P&L include all matched sessions. "
        "Actual net P&L is unavailable until fees are confirmed for every matched session.[/dim]"
    )


def _actual_time_bar_label(result: Mapping[str, object], side: str) -> str:
    stamps = sorted(
        _timestamp(
            row.get(f"actual_time_{side}_bar_at"), f"actual-time {side} bar"
        ).astimezone(EASTERN)
        for row in result.get("rows") or []
        if isinstance(row, Mapping) and row.get(f"actual_time_{side}_bar_at")
    )
    source = str(result.get(f"{side}_price_source") or "unspecified")
    if not stamps:
        return f"At fill minute: {source}"
    first = stamps[0]
    last = stamps[-1]
    if first == last:
        return f"{first:%Y-%m-%d %H:%M} ET {source}"
    if first.date() == last.date():
        return f"{first:%Y-%m-%d %H:%M}–{last:%H:%M} ET {source}"
    return f"{first:%Y-%m-%d %H:%M}–{last:%Y-%m-%d %H:%M} ET {source}"


def _scheduled_price_label(result: Mapping[str, object], side: str) -> str:
    source = str(result.get(f"{side}_price_source") or "unspecified")
    clock = str(result.get(f"{side}_time") or "unspecified")
    suffix = " (minute-wide hypothetical fill)" if source in MINUTE_PRICE_COLUMNS and source != "minute-open" else ""
    return f"{clock} ET {source}{suffix}"


def print_result(
    result: Mapping[str, object], *, show_symbol_breakdown: bool = False
) -> None:
    totals = result["totals"]
    use_actual_time = (
        result.get("reporting_benchmark") == ACTUAL_TIME_REPORTING_BENCHMARK
    )
    simulator_gross_pnl = _reported_total(
        result, "simulator_gross_pnl", "actual_time_simulator_gross_pnl"
    )
    actual_gross_pnl = float(totals["actual_gross_pnl"])
    ranking = result.get("ranking_replay") or {}
    ranking_text = "Unavailable"
    if isinstance(ranking, Mapping) and ranking.get("status") == "complete":
        ranking_text = (
            f"{ranking['overlap_count']}/{len(result['symbols'])} · "
            f"{ranking['liquidity_scheme']}"
        )

    comparison = Table(
        title="Live reconciliation",
        caption=f"{len(result['symbols'])} symbols",
        caption_justify="left",
    )
    comparison.add_column("Metric")
    comparison.add_column("Actual", justify="right")
    comparison.add_column("Simulator", justify="right")
    comparison.add_column("Explanation", justify="right")
    _add_gross_pnl_rows(
        comparison, actual_gross_pnl, simulator_gross_pnl, totals.get("entry_equity"),
    )
    modeled_cost = _reported_total(
        result,
        "simulator_transaction_cost",
        "actual_time_simulator_transaction_cost",
    )
    simulator_net_pnl = _reported_total(
        result, "simulator_net_pnl", "actual_time_simulator_net_pnl"
    )
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
        + f"Sim: {float(result['transaction_cost_bps_per_side']):g}bps per side",
    )
    actual_account_result = totals.get("broker_equity_pnl")
    comparison.add_row(
        "Net P&L",
        format_usd(totals["actual_net_pnl_after_broker_fees"], signed=True)
        if fee_complete
        else "Unavailable",
        format_usd(simulator_net_pnl, signed=True),
        pnl_comparison_context(
            float(totals["actual_net_pnl_after_broker_fees"]) - simulator_net_pnl,
            label="net P&L",
            bps=(
                (float(totals["actual_net_pnl_after_broker_fees"]) - simulator_net_pnl)
                / float(totals["actual_entry_notional"])
                * 10_000.0
            ),
        )
        if fee_complete
        else "Actual fees unavailable",
    )
    _add_pnl_percentage_row(
        comparison, "Net P&L (%)",
        totals["actual_net_pnl_after_broker_fees"] if fee_complete else None,
        simulator_net_pnl, totals.get("entry_equity"),
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
    timing = result.get("timing") or {}
    _add_execution_price_rows(comparison, summarize_execution_prices([result]))
    actual_time_exit_slippage = totals.get("actual_time_exit_slippage_bps")
    if (
        not use_actual_time
        and isinstance(timing, Mapping)
        and not timing.get("schedule_comparable")
        and actual_time_exit_slippage is not None
    ):
        comparison.add_row(
            "Off-schedule exit benchmark",
            "Fills",
            _actual_time_bar_label(result, "exit"),
            execution_context(actual_time_exit_slippage, side="exit"),
        )
    if abs(float(totals["quantity_pnl_impact"])) >= 0.005:
        comparison.add_row(
            "Quantity mismatch",
            "Exit quantities",
            "Entry quantities",
            f"{format_usd(totals['quantity_pnl_impact'], signed=True)} P&L",
        )
    comparison.add_section()
    comparison.add_row(
        "Entry price source",
        "Fills",
        (
            _actual_time_bar_label(result, "entry")
            if use_actual_time
            else _scheduled_price_label(result, "entry")
        ),
        str(result["entry_date"]),
    )
    comparison.add_row(
        "Exit price source",
        "Fills",
        (
            _actual_time_bar_label(result, "exit")
            if use_actual_time
            else _scheduled_price_label(result, "exit")
        ),
        str(result["exit_date"]),
    )
    comparison.add_row("Ranking replay", "Actual basket", "Replayed basket", ranking_text)
    CONSOLE.print(comparison)
    CONSOLE.print(
        "[dim]Gross P&L and execution differences exclude transaction costs. "
        "P&L (%) divides the corresponding dollar P&L by account equity before entry. "
        "Price bps = (simulator / actual − 1) × 10,000, using raw prices and identical "
        "entry-share weights for both sides. Positive means the simulator price is higher.[/dim]"
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
        detail.add_column("Side")
        detail.add_column("Actual price", justify="right")
        detail.add_column("Simulator price", justify="right")
        detail.add_column("Price Δ", justify="right")
        detail.add_column("P&L impact", justify="right")
        detail.add_column("Gross P&L Δ", justify="right")
        for row in result["rows"]:
            simulator_entry = float(row["actual_time_entry_price_comparable" if use_actual_time else "simulator_entry_price_comparable"])
            simulator_exit = float(row["actual_time_exit_price_comparable" if use_actual_time else "simulator_exit_price_comparable"])
            entry_bps = simulator_price_difference_bps(float(row["actual_entry_price"]), simulator_entry)
            exit_bps = simulator_price_difference_bps(float(row["actual_exit_price"]), simulator_exit)
            model_difference = row[
                "actual_minus_actual_time_simulator_gross_pnl"
                if use_actual_time
                else "actual_minus_simulator_gross_pnl"
            ]
            entry_impact = row[
                "actual_time_entry_execution_pnl_impact"
                if use_actual_time
                else "entry_execution_pnl_impact"
            ]
            exit_impact = row[
                "actual_time_exit_execution_pnl_impact"
                if use_actual_time
                else "exit_execution_pnl_impact"
            ]
            detail.add_row(
                str(row["symbol"]),
                "Entry",
                f"${float(row['actual_entry_price']):,.4f}",
                f"${simulator_entry:,.4f}",
                format_bps(entry_bps),
                format_usd(entry_impact, signed=True),
                "—",
            )
            detail.add_row(
                "", "Exit",
                f"${float(row['actual_exit_price']):,.4f}",
                f"${simulator_exit:,.4f}",
                format_bps(exit_bps),
                format_usd(exit_impact, signed=True),
                format_usd(model_difference, signed=True),
                end_section=True,
            )
        CONSOLE.print(detail)
        CONSOLE.print(
            "[dim]Price Δ uses actual fills as the reference; positive means the simulator price "
            "is higher. Entry + exit + quantity P&L impacts sum to "
            "Actual − simulator gross P&L.[/dim]"
        )


def print_warnings(
    results: Sequence[Mapping[str, object]],
    skipped_sessions: Sequence[tuple[date, Sequence[str]]],
    missing_sessions: Sequence[tuple[date, str]] = (),
) -> None:
    """Render all session warnings together after the reconciliation tables."""
    messages: list[tuple[str, str]] = []
    for entry_day, warnings in skipped_sessions:
        reason = " ".join(warnings) or "execution timing is not schedule comparable"
        messages.append((
            entry_day.isoformat(),
            f"skipped by strict schedule policy: {reason} Re-run with "
            "--reconciliation-mode actual-time-minute-bar --entry-price-source minute-open "
            "--exit-price-source minute-open to produce a forensic reconciliation.",
        ))
    messages.extend((day.isoformat(), f"skipped entire session: {reason}") for day, reason in missing_sessions)
    for result in results:
        warnings = list(result.get("warnings") or [])
        ranking = result.get("ranking_replay") or {}
        if isinstance(ranking, Mapping):
            warnings.extend(ranking.get("warnings") or [])
        messages.extend((str(result["entry_date"]), str(warning)) for warning in warnings)
    if messages:
        CONSOLE.print()
        for entry_day, warning in sorted(messages, key=lambda item: item[0]):
            CONSOLE.print(f"[yellow]Warning ({entry_day}):[/yellow] {warning}")


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
    parser.add_argument("--minute-bars-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--entry-price-source", choices=ENTRY_PRICE_SOURCES, default="nbbo-ask")
    parser.add_argument("--exit-price-source", choices=EXIT_PRICE_SOURCES, default="opening-auction")
    parser.add_argument("--exit-time", type=parse_clock, default=570)
    parser.add_argument("--exit-nbbo-path", type=Path, default=Path(DEFAULT_EXIT_NBBO_PATH))
    parser.add_argument("--max-exit-staleness-minutes", type=float, default=1.0,
                        help="NBBO age limit in minutes, capped at 1; minute sources require an exact bar")
    parser.add_argument(
        "--auctions-path", type=Path, default=Path(DEFAULT_AUCTIONS_PATH),
        help="opening-auction NPZ; its split ledger also converts minute prices to raw fill units",
    )
    parser.add_argument(
        "--nbbo-path",
        type=Path,
        default=Path(DEFAULT_NBBO_PATH),
        help="scheduled split-adjusted SIP NBBO data written by download_nbbo.py",
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
        default=None,
        help="additional simulator cost in bps per side (default: 0.0 for nbbo-ask → opening-auction; 1.0 otherwise)",
    )
    parser.add_argument("--max-entry-staleness-minutes", type=float, default=1.0,
                        help="NBBO age limit in minutes, capped at 1; minute sources require an exact bar")
    parser.add_argument(
        "--schedule-tolerance-minutes",
        type=float,
        default=DEFAULT_SCHEDULE_TOLERANCE_MINUTES,
        help="maximum fill-time deviation still considered on schedule (default: 1)",
    )
    parser.add_argument(
        "--reconciliation-mode",
        choices=("strict-schedule", "actual-time-minute-bar"),
        default="strict-schedule",
        help=(
            "strict-schedule skips off-schedule sessions; actual-time-minute-bar uses each "
            "symbol's selected minute fields at its entry and exit fill times; "
            "requires minute-* entry and exit sources "
            "(default: strict-schedule)"
        ),
    )
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
        "--refresh-broker-fees",
        action="store_true",
        help="refresh selected sessions' Alpaca fees even when their caches are fresh",
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
    parser.add_argument(
        "--show-session-details",
        action="store_true",
        help="print the reconciliation table for every individual session",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.transaction_cost_bps = resolve_transaction_cost_bps(
            args.transaction_cost_bps, args.entry_price_source, args.exit_price_source,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.entry_date and args.since:
        parser.error("--entry-date and --since are mutually exclusive")
    if (
        args.transaction_cost_bps < 0.0
        or args.max_entry_staleness_minutes < 0.0
        or args.max_exit_staleness_minutes < 0.0
        or args.schedule_tolerance_minutes < 0.0
    ):
        parser.error("cost, staleness, and tolerance values must be non-negative")
    if args.request_timeout_seconds is not None and args.request_timeout_seconds <= 0.0:
        parser.error("--request-timeout-seconds must be positive")
    if args.exit_price_source == "opening-auction" and args.exit_time != 570:
        parser.error("--exit-price-source opening-auction requires --exit-time 09:30")
    if args.reconciliation_mode == "actual-time-minute-bar" and (
        args.entry_price_source not in MINUTE_PRICE_COLUMNS or args.exit_price_source not in MINUTE_PRICE_COLUMNS
    ):
        parser.error("actual-time-minute-bar requires explicit minute-* entry and exit price sources")
    summaries = discover_closed_summaries(args.work_dir)
    if args.entry_date:
        selected_days = [args.entry_date] if args.entry_date in summaries else []
    elif args.since:
        selected_days = sorted(day for day in summaries if day >= args.since)
    else:
        selected_days = [max(summaries)] if summaries else []
    if not selected_days:
        parser.error("no matching closed live sessions were found")

    skipped_sessions: list[tuple[date, list[str]]] = []
    strict_schedule = args.reconciliation_mode == "strict-schedule"
    if strict_schedule:
        comparable_days: list[date] = []
        for entry_day in selected_days:
            _summary_path, summary = summaries[entry_day]
            try:
                timing, warnings = summary_execution_timing(
                    summary,
                    entry_minute_override=args.entry_time,
                    schedule_tolerance_minutes=args.schedule_tolerance_minutes,
                    exit_minute=args.exit_time,
                    exit_price_source=args.exit_price_source,
                )
            except (TypeError, ValueError) as error:
                parser.error(f"cannot classify {entry_day} schedule: {error}")
            if timing["schedule_comparable"]:
                comparable_days.append(entry_day)
            else:
                skipped_sessions.append((entry_day, warnings))
        selected_days = comparable_days
        if not selected_days:
            CONSOLE.print(
                f"No schedule-comparable sessions to reconcile; skipped "
                f"{len(skipped_sessions)} session"
                f"{'s' if len(skipped_sessions) != 1 else ''}."
            )
            print_warnings([], skipped_sessions)
            return

    if (args.entry_price_source in MINUTE_PRICE_COLUMNS or args.exit_price_source in MINUTE_PRICE_COLUMNS) and args.minute_bars_dir.exists():
        _dataset_manifest(args.minute_bars_dir, "1Min")

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
    missing_sessions: list[tuple[date, str]] = []
    completed_results: list[dict[str, object]] = []
    for entry_day in selected_days:
        summary_path, summary = summaries[entry_day]
        try:
            result = reconcile_execution(
                summary,
                args.minute_bars_dir,
                args.auctions_path,
                nbbo_path=args.nbbo_path,
                entry_minute_override=args.entry_time,
                transaction_cost_bps=args.transaction_cost_bps,
                max_entry_staleness_minutes=args.max_entry_staleness_minutes,
                schedule_tolerance_minutes=args.schedule_tolerance_minutes,
                actual_time_benchmark=not strict_schedule,
                entry_price_source=args.entry_price_source,
                exit_price_source=args.exit_price_source,
                exit_minute=args.exit_time,
                exit_nbbo_path=args.exit_nbbo_path,
                max_exit_staleness_minutes=args.max_exit_staleness_minutes,
            )
            select_reporting_benchmark(result, prefer_actual_time=not strict_schedule)
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
                    args.work_dir / entry_day.isoformat() / "fee_activities.json",
                    parse_day(str(result["exit_date"])),
                    unavailable_reason=broker_unavailable_reason,
                    force_refresh=args.refresh_broker_fees,
                )
            attach_broker_fees(result, fee_summary)
            if fee_warning:
                result["warnings"].append(fee_warning)

            result["version"] = 7
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
            completed_results.append(result)
            if args.show_session_details or args.show_symbol_breakdown:
                print_result(result, show_symbol_breakdown=args.show_symbol_breakdown)
        except MissingBenchmarkData as error:
            missing_sessions.append((entry_day, str(error)))
            _atomic_json(output_dir / f"{entry_day.isoformat()}.json", {
                "status": "skipped", "entry_date": entry_day.isoformat(),
                "entry_price_source": args.entry_price_source,
                "exit_price_source": args.exit_price_source,
                "reason": str(error),
            })
            (output_dir / f"{entry_day.isoformat()}.csv").unlink(missing_ok=True)
        except (OSError, TypeError, ValueError) as error:
            failures += 1
            CONSOLE.print(f"[red]{entry_day}: reconciliation failed:[/red] {error}")
    if completed_results:
        print_overview(completed_results, skipped_sessions=skipped_sessions, missing_sessions=missing_sessions)
    elif missing_sessions:
        CONSOLE.print(
            f"No sessions with complete benchmark prices to reconcile; "
            f"skipped {len(missing_sessions)} sessions."
        )
    print_warnings(completed_results, skipped_sessions, missing_sessions)
    if completed_results:
        CONSOLE.print(
            f"Wrote {len(completed_results)} reconciliation"
            f"{'s' if len(completed_results) != 1 else ''} to {output_dir}"
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
