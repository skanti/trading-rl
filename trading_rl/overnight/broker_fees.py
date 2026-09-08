"""Shared parsing for Alpaca account-level fee activities."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
FEE_POSTING_GRACE_DAYS = 1
RECENT_FEE_DAYS = 7
RECENT_FEE_REFRESH = timedelta(hours=1)
HISTORICAL_FEE_REFRESH = timedelta(days=7)


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def safe_fee_activity(activity: Mapping[str, object]) -> dict[str, object]:
    """Keep auditable fee fields while stripping account identifiers."""
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
    """Summarize account-level fee activities booked for one exit trade date."""
    selected: list[dict[str, object]] = []
    for activity in activities:
        activity_type = str(activity.get("activity_type") or "").upper()
        activity_day = str(activity.get("date") or "")[:10]
        if activity_type != "FEE" or activity_day != exit_day.isoformat():
            continue
        safe = safe_fee_activity(activity)
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

    fetched = fetched_at or datetime.now(tz=UTC)
    summary: dict[str, object] = {
        "status": "observed",
        "source": "alpaca_account_activities",
        "scope": "all account FEE activities whose activity date equals the exit date",
        "activity_date": exit_day.isoformat(),
        "fetched_at": fetched.isoformat(),
        "count": len(selected),
        "net_amount": net_amount,
        "cost": -net_amount,
        "breakdown": breakdown,
        "activities": selected,
    }
    return classify_broker_fee_summary(summary, exit_day, as_of=fetched)


def classify_broker_fee_summary(
    summary: Mapping[str, object],
    exit_day: date,
    *,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Mark a fee observation pending until fees post or the grace period ends."""
    result = dict(summary)
    count = int(result.get("count") or 0)
    reference = as_of or datetime.now(tz=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    grace_end = exit_day + timedelta(days=FEE_POSTING_GRACE_DAYS)
    # Aging a cached empty response is not evidence that no fees were posted.
    observed = _timestamp(result.get("fetched_at"))
    confirmed = count > 0 or (
        observed is not None
        and min(observed, reference).astimezone(EASTERN).date() > grace_end
    )
    result["status"] = "complete" if confirmed else "pending"
    if confirmed:
        result.pop("reason", None)
    else:
        result["reason"] = "broker fees have not posted yet"
    return result


class FeeActivityClient(Protocol):
    def account_activities(
        self,
        activity_type: str,
        *,
        after: date,
        until: date,
    ) -> list[dict[str, object]]: ...


def _timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def load_fee_cache(
    path: Path,
    exit_day: date,
    *,
    as_of: datetime | None = None,
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("activity_date") != exit_day.isoformat()
        ):
            return None
        _number(payload["cost"], "cached broker fee cost")
        if int(payload["count"]) < 0:
            return None
        return classify_broker_fee_summary(payload, exit_day, as_of=as_of)
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fee_refresh_due(
    cached: Mapping[str, object] | None,
    exit_day: date,
    *,
    as_of: datetime,
) -> bool:
    if cached is None:
        return True
    checked = _timestamp(cached.get("last_checked_at") or cached.get("fetched_at"))
    if checked is None or checked > as_of:
        return True
    recent = (as_of.astimezone(EASTERN).date() - exit_day).days <= RECENT_FEE_DAYS
    interval = (
        RECENT_FEE_REFRESH
        if recent or cached.get("status") != "complete" or cached.get("refresh_warning")
        else HISTORICAL_FEE_REFRESH
    )
    return as_of - checked >= interval


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def cache_fee_activities(
    path: Path,
    exit_day: date,
    activities: Sequence[Mapping[str, object]],
    *,
    as_of: datetime,
) -> dict[str, object]:
    """Persist a successful observation, including unchanged and empty results."""
    fresh = summarize_broker_fees(activities, exit_day, fetched_at=as_of)
    cached = load_fee_cache(path, exit_day, as_of=as_of)
    if cached is not None and int(cached.get("count") or 0) > 0 and fresh["count"] == 0:
        fresh = dict(cached)
        fresh["refresh_warning"] = (
            "empty broker fee response; retaining previously observed fees"
        )
    fresh["last_checked_at"] = as_of.isoformat()
    _atomic_json(path, fresh)
    return fresh


def broker_fees_for_session(
    client: FeeActivityClient | None,
    cache_path: Path,
    exit_day: date,
    *,
    unavailable_reason: str | None = None,
    as_of: datetime | None = None,
    force_refresh: bool = False,
) -> tuple[dict[str, object], str | None]:
    """Use the durable cache, refreshing only due exit dates or explicit requests."""
    reference = as_of or datetime.now(tz=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    cached = load_fee_cache(cache_path, exit_day, as_of=reference)

    def warning(summary: Mapping[str, object]) -> str | None:
        if summary.get("refresh_warning"):
            return str(summary["refresh_warning"])
        if summary.get("status") == "pending":
            return "broker fees have not posted yet; reconciliation remains provisional"
        return None

    if not force_refresh and not fee_refresh_due(cached, exit_day, as_of=reference):
        assert cached is not None
        return cached, warning(cached)
    if client is None:
        reason = unavailable_reason or "broker fee retrieval is unavailable"
        if cached is not None:
            return cached, f"using cached broker fees because {reason}"
        return {
            "status": "unavailable",
            "activity_date": exit_day.isoformat(),
            "reason": reason,
        }, reason
    try:
        activities = client.account_activities(
            "FEE",
            after=exit_day - timedelta(days=1),
            until=exit_day + timedelta(days=4),
        )
        summary = cache_fee_activities(
            cache_path, exit_day, activities, as_of=reference
        )
        return summary, warning(summary)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # A failed request never advances last_checked_at or confirms empty fees.
        if cached is not None:
            return cached, f"broker fee refresh failed; using cache: {error}"
        return {
            "status": "unavailable",
            "activity_date": exit_day.isoformat(),
            "reason": str(error),
        }, f"broker fees are unavailable: {error}"
