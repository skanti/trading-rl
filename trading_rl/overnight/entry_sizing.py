"""Pure broker-constrained sizing shared by live execution and decision replay."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .strategies import MAX_OVERNIGHT_EXPOSURE


def _float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric Alpaca field {field}={value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite Alpaca field {field}={value!r}")
    return result


def available_budget(account: Mapping[str, object], config) -> float:
    cash = max(0.0, _float(account.get("cash"), "account.cash"))
    buying_power_value = account.get("buying_power")
    stock_buying_power = (
        max(0.0, _float(buying_power_value, "account.buying_power"))
        if buying_power_value is not None
        else cash
    )
    # These are marginable US equities. ``non_marginable_buying_power`` tracks
    # settled dollars for assets such as crypto and can exclude same-day stock-sale
    # proceeds until T+1, even though Alpaca permits those proceeds to be reused for
    # equities immediately. Cap at cash to avoid borrowing, and at regular buying
    # power so account restrictions or other open orders are still respected.
    usable_cash = cash * (1.0 - config.cash_buffer_fraction)
    requested = (
        config.capital if config.capital is not None else cash * config.capital_fraction
    )
    budget = min(float(requested), usable_cash, stock_buying_power)
    if budget <= 0.0:
        raise RuntimeError("account has no cash available for the basket")
    if budget / config.top < 1.0:
        raise RuntimeError(
            "per-symbol notional would be below Alpaca's $1 fractional minimum"
        )
    return math.floor(budget * 100.0) / 100.0


def trend_vol_budget(
    account, config, exposure, positions, open_orders, selected_assets
):
    """Constrain requested equity exposure using current overnight account limits."""
    if any(
        account.get(flag)
        for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
    ):
        raise RuntimeError("account is blocked for trading")
    if account.get("status", "ACTIVE") != "ACTIVE":
        raise RuntimeError("account is not active")
    equity = _float(account.get("equity"), "account.equity")
    cash = max(0.0, _float(account.get("cash"), "account.cash"))
    if (
        equity <= 0
        or not math.isfinite(exposure)
        or not 0 < exposure <= config.risk_config.max_exposure
    ):
        raise RuntimeError("invalid equity or target exposure")
    allocated = min(
        equity,
        config.capital
        if config.capital is not None
        else equity * config.capital_fraction,
    )
    requested = allocated * exposure
    buying_power = max(0.0, _float(account.get("buying_power"), "account.buying_power"))
    multiplier = _float(account.get("multiplier"), "account.multiplier")
    can_borrow = multiplier >= 2 and all(
        bool(asset.get("marginable")) for asset in selected_assets
    )
    regt = (
        max(0.0, _float(account.get("regt_buying_power"), "account.regt_buying_power"))
        if can_borrow
        else cash
    )
    held = sum(
        abs(_float(position.get("market_value"), "position.market_value"))
        for position in positions
    )
    pending = 0.0
    for order in open_orders:
        if order.get("side") != "buy":
            continue
        if order.get("notional") is not None:
            pending += max(0.0, _float(order["notional"], "order.notional"))
        elif order.get("limit_price") is not None:
            quantity = max(
                0.0,
                _float(order.get("qty"), "order.qty")
                - _float(order.get("filled_qty") or 0, "order.filled_qty"),
            )
            pending += quantity * _float(order["limit_price"], "order.limit_price")
        else:
            raise RuntimeError(
                "cannot safely value an unrelated open buy order for overnight exposure"
            )
    limits = {
        "policy": requested,
        "buying_power": buying_power,
        "overnight_buying_power": regt,
        "account_exposure": max(0.0, MAX_OVERNIGHT_EXPOSURE * equity - held - pending),
    }
    if not can_borrow:
        limits["unborrowed_capital"] = cash
    # Higher broker maintenance requirements can constrain an otherwise marginable basket.
    requirements = [
        float(asset.get("maintenance_margin_requirement") or 0) / 100
        for asset in selected_assets
    ]
    if any(not math.isfinite(value) or value < 0 for value in requirements):
        raise RuntimeError("invalid asset maintenance margin requirement")
    if requirements and all(value > 0 for value in requirements):
        existing = max(
            0.0,
            _float(
                account.get("maintenance_margin") or 0, "account.maintenance_margin"
            ),
        )
        limits["maintenance_capacity"] = max(0.0, equity - existing) / float(
            np.mean(requirements)
        )
    budget = (
        math.floor(min(limits.values()) * (1 - config.cash_buffer_fraction) * 100) / 100
    )
    if budget / config.top < 1:
        raise RuntimeError(
            "constrained per-symbol notional is below the $1 fractional minimum"
        )
    return {
        "budget": budget,
        "allocated_equity": allocated,
        "account_equity": equity,
        "target_exposure": exposure,
        "effective_exposure": budget / allocated,
        "cash_buffer_fraction": config.cash_buffer_fraction,
        "notional_limits": limits,
        "binding_limit": min(limits, key=limits.get),
        "margin_eligible": can_borrow,
    }
