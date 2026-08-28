"""Pure account-performance arithmetic for the dashboard snapshot.

Every function here is pure: callers pass already-fetched Alpaca payloads and get
plain dataclasses back, keeping the daemon's network and persistence code small.

One subtlety drives the parsing code. Alpaca returns ``timeframe=1D`` portfolio history
points stamped at UTC midnight of the day *after* the session, so a naive UTC read is
off by one calendar day. The session date is the timestamp's *Eastern* date, which is
what :func:`equity_series` uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Bucket keys, in the order they should be presented in a table.
BUCKET_ORDER = ("today", "week", "month", "year", "inception")
BUCKET_LABELS = {
    "today": "Today",
    "week": "Week to date",
    "month": "Month to date",
    "year": "Year to date",
    "inception": "Since inception"
}


def _float(value: object, default: float = 0.0) -> float:
    """Coerce an Alpaca string/number field to a finite float."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class EquityPoint:
    day: date
    equity: float
    profit_loss: float
    profit_loss_pct: float


@dataclass(frozen=True)
class PerformanceBucket:
    """Profit and loss over a window, anchored on the close before it opened."""

    key: str
    label: str
    start_day: date | None
    start_equity: float
    end_equity: float
    pnl: float
    pnl_pct: float
    sessions: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "start_day": self.start_day.isoformat() if self.start_day else None,
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "sessions": self.sessions
        }


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    entry_notional: float
    exit_notional: float
    pnl: float
    pnl_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_notional": self.entry_notional,
            "exit_notional": self.exit_notional,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct
        }


@dataclass(frozen=True)
class Statistics:
    max_drawdown: float
    max_drawdown_pct: float
    best_day: EquityPoint | None
    worst_day: EquityPoint | None
    win_rate: float
    winning_sessions: int
    losing_sessions: int
    sessions: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "best_day": _point_dict(self.best_day),
            "worst_day": _point_dict(self.worst_day),
            "win_rate": self.win_rate,
            "winning_sessions": self.winning_sessions,
            "losing_sessions": self.losing_sessions,
            "sessions": self.sessions
        }


def _point_dict(point: EquityPoint | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return {
        "day": point.day.isoformat(),
        "equity": point.equity,
        "profit_loss": point.profit_loss,
        "profit_loss_pct": point.profit_loss_pct
    }


def session_date(epoch_seconds: float) -> date:
    """Return the Eastern trading date for a portfolio-history timestamp."""
    moment = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    return moment.astimezone(EASTERN).date()


def history_period(created_at: str | datetime | None, now: datetime | None = None) -> str:
    """Choose the smallest whole-year period that reaches back past account inception.

    Alpaca back-fills flat equity for dates before the account existed, so asking for
    more history than necessary is harmless once :func:`equity_series` trims it, but a
    tighter period keeps the response small.
    """
    reference = now or datetime.now(tz=EASTERN)
    start: datetime | None = None
    if isinstance(created_at, datetime):
        start = created_at
    elif created_at:
        try:
            start = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            start = None
    if start is None:
        return "1A"
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    years = (reference - start).days / 365.25
    return f"{max(1, math.ceil(years + 0.05))}A"


def equity_series(
    history: Mapping[str, Any],
    *,
    since: date | None = None
) -> list[EquityPoint]:
    """Parse Alpaca's parallel portfolio-history arrays into sorted daily points.

    ``since`` drops points before account inception. Alpaca happily returns flat
    synthetic equity for dates predating the account, and charting those would invent
    a year of imaginary history.
    """
    timestamps = list(history.get("timestamp") or [])
    equities = list(history.get("equity") or [])
    profits = list(history.get("profit_loss") or [])
    percents = list(history.get("profit_loss_pct") or [])

    points: list[EquityPoint] = []
    for index, stamp in enumerate(timestamps):
        if index >= len(equities):
            break
        equity = equities[index]
        if equity is None:
            continue
        day = session_date(stamp)
        if since is not None and day < since:
            continue
        points.append(
            EquityPoint(
                day=day,
                equity=_float(equity),
                profit_loss=_float(profits[index]) if index < len(profits) else 0.0,
                profit_loss_pct=_float(percents[index]) if index < len(percents) else 0.0
            )
        )
    points.sort(key=lambda point: point.day)
    # A same-day duplicate can appear when a period boundary overlaps; keep the last.
    deduplicated: dict[date, EquityPoint] = {point.day: point for point in points}
    return [deduplicated[day] for day in sorted(deduplicated)]


def realized_equity_series(
    sessions: Sequence[Mapping[str, Any]],
    *,
    base_value: float,
    inception: date | None = None
) -> list[EquityPoint]:
    """Build strategy equity from completed basket results.

    The first point is the untraded baseline. Each later point uses the authoritative
    flat-account exit equity when available, otherwise applying gross fill P&L for a
    legacy basket. Open sessions are deliberately absent: live mark-to-market account
    equity is displayed separately by the dashboard.
    """
    closed: list[tuple[date, date, float, float]] = []
    entry_days: list[date] = []
    for session in sessions:
        if session.get("status") != "closed" or session.get("realized_pnl") is None:
            continue
        try:
            exit_day = date.fromisoformat(
                str(session.get("exit_date") or session.get("trading_day"))
            )
        except ValueError:
            continue
        raw_entry_day = session.get("entry_date") or session.get("trading_day")
        try:
            entry_day = date.fromisoformat(str(raw_entry_day))
            entry_days.append(entry_day)
        except ValueError:
            entry_day = exit_day
        closed.append(
            (
                exit_day,
                entry_day,
                _float(session["realized_pnl"]),
                _float(session.get("exit_equity")),
            )
        )

    baseline_day = inception or (min(entry_days) if entry_days else None)
    if baseline_day is None:
        return []

    balance = base_value
    points = [EquityPoint(baseline_day, balance, 0.0, 0.0)]
    realized_by_day: dict[date, list[tuple[date, float, float]]] = {}
    for exit_day, entry_day, pnl, exit_equity in sorted(closed):
        realized_by_day.setdefault(exit_day, []).append((entry_day, pnl, exit_equity))

    for exit_day in sorted(realized_by_day):
        for _entry_day, pnl, exit_equity in sorted(realized_by_day[exit_day]):
            # A flat-account snapshot includes fees and settlement rounding, making
            # it authoritative. Legacy sessions fall back to gross fill arithmetic.
            balance = exit_equity if exit_equity > 0.0 else balance + pnl
        cumulative_pnl = balance - base_value
        point = EquityPoint(
            day=exit_day,
            equity=balance,
            profit_loss=cumulative_pnl,
            profit_loss_pct=(cumulative_pnl / base_value) if base_value > 0.0 else 0.0,
        )
        if exit_day == baseline_day:
            points[0] = point
        else:
            points.append(point)
    return points


def inception_equity(history: Mapping[str, Any], series: Sequence[EquityPoint]) -> float:
    """Equity at the start of the requested series.

    Alpaca's ``base_value`` belongs to the account-lifetime portfolio history. It can
    predate a later deposit or this strategy's inception by months, so it is only a
    fallback when the filtered series has no usable equity point.
    """
    if series and series[0].equity > 0.0:
        return series[0].equity
    base = _float(history.get("base_value"))
    if base > 0.0:
        return base
    return 0.0


def strategy_inception_equity(
    sessions: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    series: Sequence[EquityPoint],
) -> float:
    """Return pre-entry equity for the earliest recorded strategy basket.

    The trading daemon persists an account snapshot immediately before each entry.
    That value is the precise baseline for a curve built by adding realized basket
    P&L. Older artifacts may not contain it, in which case the filtered Alpaca series
    is the best available fallback.
    """
    candidates: list[tuple[date, float]] = []
    for session in sessions:
        equity = _float(session.get("entry_equity"))
        if equity <= 0.0:
            continue
        raw_day = session.get("entry_date") or session.get("trading_day")
        try:
            entry_day = date.fromisoformat(str(raw_day))
        except ValueError:
            continue
        candidates.append((entry_day, equity))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return inception_equity(history, series)


def _baseline_before(series: Sequence[EquityPoint], boundary: date) -> EquityPoint | None:
    """Last session that closed strictly before ``boundary``."""
    candidate: EquityPoint | None = None
    for point in series:
        if point.day < boundary:
            candidate = point
        else:
            break
    return candidate


def bucket(
    series: Sequence[EquityPoint],
    boundary: date | None,
    *,
    key: str,
    end_equity: float,
    fallback_equity: float,
    baseline_is_first: bool = False
) -> PerformanceBucket:
    """Profit and loss from the close before ``boundary`` through ``end_equity``.

    ``boundary`` of ``None`` means "since inception", anchored on ``fallback_equity``.
    """
    if boundary is None:
        start_equity = fallback_equity
        start_day = series[0].day if series else None
        sessions = max(0, len(series) - 1) if baseline_is_first else len(series)
    else:
        anchor = _baseline_before(series, boundary)
        start_equity = anchor.equity if anchor is not None else fallback_equity
        start_day = anchor.day if anchor is not None else (series[0].day if series else None)
        sessions = sum(
            1
            for index, point in enumerate(series)
            if point.day >= boundary and (not baseline_is_first or index > 0)
        )

    pnl = end_equity - start_equity
    pnl_pct = (pnl / start_equity) if start_equity > 0.0 else 0.0
    return PerformanceBucket(
        key=key,
        label=BUCKET_LABELS.get(key, key.title()),
        start_day=start_day,
        start_equity=start_equity,
        end_equity=end_equity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        sessions=sessions
    )


def performance_table(
    series: Sequence[EquityPoint],
    account: Mapping[str, Any],
    history: Mapping[str, Any],
    now: datetime | None = None
) -> dict[str, PerformanceBucket]:
    """The five headline buckets, measured against live account equity.

    ``today`` uses Alpaca's own ``last_equity`` (equity at the previous close) rather
    than the series, so the number moves intraday instead of waiting for a new point.
    """
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)
    today = reference.date()

    end_equity = _float(account.get("equity"))
    if end_equity <= 0.0 and series:
        end_equity = series[-1].equity
    last_equity = _float(account.get("last_equity"))
    start_value = inception_equity(history, series)

    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    today_bucket = bucket(
        series,
        today,
        key="today",
        end_equity=end_equity,
        fallback_equity=last_equity if last_equity > 0.0 else start_value
    )
    if last_equity > 0.0:
        # Prefer Alpaca's authoritative previous close over the parsed series.
        pnl = end_equity - last_equity
        today_bucket = PerformanceBucket(
            key="today",
            label=BUCKET_LABELS["today"],
            start_day=today_bucket.start_day,
            start_equity=last_equity,
            end_equity=end_equity,
            pnl=pnl,
            pnl_pct=(pnl / last_equity) if last_equity > 0.0 else 0.0,
            sessions=1
        )

    return {
        "today": today_bucket,
        "week": bucket(series, week_start, key="week", end_equity=end_equity, fallback_equity=start_value),
        "month": bucket(series, month_start, key="month", end_equity=end_equity, fallback_equity=start_value),
        "year": bucket(series, year_start, key="year", end_equity=end_equity, fallback_equity=start_value),
        "inception": bucket(series, None, key="inception", end_equity=end_equity, fallback_equity=start_value)
    }


def realized_performance_table(
    series: Sequence[EquityPoint],
    now: datetime | None = None
) -> dict[str, PerformanceBucket]:
    """Performance buckets based only on closed strategy baskets."""
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)
    today = reference.date()
    end_equity = series[-1].equity if series else 0.0
    start_value = series[0].equity if series else 0.0
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    def realized_bucket(boundary: date | None, key: str) -> PerformanceBucket:
        return bucket(
            series,
            boundary,
            key=key,
            end_equity=end_equity,
            fallback_equity=start_value,
            baseline_is_first=True,
        )

    return {
        "today": realized_bucket(today, "today"),
        "week": realized_bucket(week_start, "week"),
        "month": realized_bucket(month_start, "month"),
        "year": realized_bucket(year_start, "year"),
        "inception": realized_bucket(None, "inception"),
    }


def statistics(series: Sequence[EquityPoint], *, baseline_is_first: bool = False) -> Statistics:
    """Drawdown, extremes and hit rate over an equity curve."""
    if not series:
        return Statistics(0.0, 0.0, None, None, 0.0, 0, 0, 0)

    peak = series[0].equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    for point in series:
        peak = max(peak, point.equity)
        drawdown = peak - point.equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = (drawdown / peak) if peak > 0.0 else 0.0

    # Day-over-day moves; the first point has no predecessor to compare against.
    moves: list[tuple[EquityPoint, float]] = []
    for index in range(1, len(series)):
        previous = series[index - 1]
        current = series[index]
        change = current.equity - previous.equity
        moves.append(
            (
                EquityPoint(
                    day=current.day,
                    equity=current.equity,
                    profit_loss=change,
                    profit_loss_pct=(change / previous.equity) if previous.equity > 0.0 else 0.0,
                ),
                change,
            )
        )
    traded = [(point, change) for point, change in moves if change != 0.0]
    winners = [point for point, change in traded if change > 0.0]
    losers = [point for point, change in traded if change < 0.0]

    best = max(traded, key=lambda item: item[1])[0] if traded else None
    worst = min(traded, key=lambda item: item[1])[0] if traded else None

    return Statistics(
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        best_day=best,
        worst_day=worst,
        win_rate=(len(winners) / len(traded)) if traded else 0.0,
        winning_sessions=len(winners),
        losing_sessions=len(losers),
        sessions=max(0, len(series) - 1) if baseline_is_first else len(series)
    )


def filled_notional(orders: Mapping[str, Mapping[str, Any]]) -> float:
    """Sum ``filled_qty * filled_avg_price`` across a map of order summaries.

    This is kept pure so snapshot and session calculations remain deterministic.
    """
    total = 0.0
    for order in orders.values():
        total += _float(order.get("filled_qty")) * _float(order.get("filled_avg_price"))
    return total


def closed_basket(position: Mapping[str, Any]) -> list[ClosedTrade]:
    """Per-symbol realized results for a strategy position that has been exited.

    Only symbols with both an entry and an exit fill are reported: a symbol that never
    filled has no result to show, and one still open has no exit price yet.
    """
    entry_orders: Mapping[str, Mapping[str, Any]] = position.get("entry_orders") or {}
    exit_orders: Mapping[str, Mapping[str, Any]] = position.get("exit_orders") or {}

    trades: list[ClosedTrade] = []
    for symbol in sorted(entry_orders):
        entry = entry_orders[symbol]
        exit_order = exit_orders.get(symbol)
        if exit_order is None:
            continue
        entry_qty = _float(entry.get("filled_qty"))
        exit_qty = _float(exit_order.get("filled_qty"))
        entry_price = _float(entry.get("filled_avg_price"))
        exit_price = _float(exit_order.get("filled_avg_price"))
        if entry_qty <= 0.0 or exit_qty <= 0.0:
            continue
        entry_notional = entry_qty * entry_price
        exit_notional = exit_qty * exit_price
        pnl = exit_notional - entry_notional
        trades.append(
            ClosedTrade(
                symbol=symbol,
                qty=exit_qty,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_notional=entry_notional,
                exit_notional=exit_notional,
                pnl=pnl,
                pnl_pct=(pnl / entry_notional) if entry_notional > 0.0 else 0.0
            )
        )
    return trades


def closed_basket_from_order_history(
    position: Mapping[str, Any], orders: Iterable[Mapping[str, Any]]
) -> list[ClosedTrade]:
    """Aggregate every partial exit fill for a closed strategy basket.

    Legacy trading state kept only the latest exit attempt for each symbol. When an
    order partially filled before being canceled, using that state alone understated
    proceeds. Alpaca's closed-order history retains all attempts, so the dashboard can
    reconstruct the weighted exit price without modifying trading state.
    """
    entry_date = str(position.get("entry_date") or "")
    if not entry_date:
        return closed_basket(position)
    prefix = f"olq-{entry_date.replace('-', '')}-x-"
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    seen_ids: set[str] = set()
    for order in orders:
        order_id = str(order.get("id") or order.get("client_order_id") or "")
        client_id = str(order.get("client_order_id") or "")
        symbol = str(order.get("symbol") or "")
        if (
            not symbol
            or not client_id.startswith(prefix)
            or str(order.get("side") or "").lower() != "sell"
            or order_id in seen_ids
        ):
            continue
        seen_ids.add(order_id)
        if _float(order.get("filled_qty")) > 0.0:
            by_symbol.setdefault(symbol, []).append(order)

    entry_orders: Mapping[str, Mapping[str, Any]] = position.get("entry_orders") or {}
    trades: list[ClosedTrade] = []
    for symbol in sorted(entry_orders):
        fills = by_symbol.get(symbol) or []
        entry_price = _float(entry_orders[symbol].get("filled_avg_price"))
        exit_qty = sum(_float(order.get("filled_qty")) for order in fills)
        exit_notional = sum(
            _float(order.get("filled_qty")) * _float(order.get("filled_avg_price"))
            for order in fills
        )
        if entry_price <= 0.0 or exit_qty <= 0.0:
            continue
        entry_notional = exit_qty * entry_price
        exit_price = exit_notional / exit_qty
        pnl = exit_notional - entry_notional
        trades.append(
            ClosedTrade(
                symbol=symbol,
                qty=exit_qty,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_notional=entry_notional,
                exit_notional=exit_notional,
                pnl=pnl,
                pnl_pct=(pnl / entry_notional) if entry_notional > 0.0 else 0.0,
            )
        )
    return trades or closed_basket(position)


def basket_totals(trades: Iterable[ClosedTrade]) -> dict[str, float]:
    """Aggregate a closed basket into the email's total row."""
    entry_total = 0.0
    exit_total = 0.0
    for trade in trades:
        entry_total += trade.entry_notional
        exit_total += trade.exit_notional
    pnl = exit_total - entry_total
    return {
        "entry_notional": entry_total,
        "exit_notional": exit_total,
        "pnl": pnl,
        "pnl_pct": (pnl / entry_total) if entry_total > 0.0 else 0.0
    }
