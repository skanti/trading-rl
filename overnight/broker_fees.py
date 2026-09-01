"""Shared parsing for Alpaca account-level fee activities."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import math
import re
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
FEE_POSTING_GRACE_DAYS = 1


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
    confirmed = count > 0 or reference.astimezone(EASTERN).date() > grace_end
    result["status"] = "complete" if confirmed else "pending"
    if confirmed:
        result.pop("reason", None)
    else:
        result["reason"] = "broker fees have not posted yet"
    return result
