"""Offline replay of immutable entry decisions. No live, backtest or network imports."""

import copy
import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .entry_sizing import available_budget, trend_vol_budget
from .momentum import replay_allocation
from .portfolio import (
    basket_quantities,
    equal_notional,
    market_entry_order,
    select_entry_candidates,
)
from .risk_history import risk_signal, unit_return
from .strategies import RISK_STRATEGIES, STRATEGIES, allocation_slots, risk_config_type

REPLAY_VERSION = 2
SIZING_SETTINGS = (
    "top",
    "capital",
    "capital_fraction",
    "cash_buffer_fraction",
    "share_mode",
)
RANKING_SETTINGS = (
    "ema_span",
    "min_history_days",
    "minimum_trading_days",
    "liquidity_scheme",
)
ACCOUNT_FIELDS = (
    "cash",
    "equity",
    "buying_power",
    "regt_buying_power",
    "multiplier",
    "maintenance_margin",
    "status",
    "trading_blocked",
    "account_blocked",
    "trade_suspended_by_user",
)


def capture_decision_inputs(config, account, positions, orders, assets):
    """Whitelist inputs before submission, so replay needs no current broker state."""

    def project(row, fields):
        return {key: row[key] for key in fields if key in row}

    return copy.deepcopy(
        {
            "version": REPLAY_VERSION,
            "provenance": "entry_preflight",
            "strategy_name": config.strategy_name,
            "configuration": {
                key: getattr(config, key)
                for key in (*SIZING_SETTINGS, *RANKING_SETTINGS)
            },
            "account": project(account, ACCOUNT_FIELDS),
            "positions": [
                project(row, ("symbol", "market_value")) for row in positions
            ],
            "open_orders": [
                project(
                    row,
                    ("symbol", "side", "notional", "qty", "filled_qty", "limit_price"),
                )
                for row in orders
            ],
            "selected_assets": [
                project(row, ("symbol", "marginable", "maintenance_margin_requirement"))
                for row in assets
            ],
            "selection_exclusions_known": True,
        }
    )


def replay_risk(saved, parameters, entry_day, top, strategy_name="liquidity-trend-vol"):
    """Recalculate unit returns, lagged SPY trend and exposure from saved inputs."""
    if saved.get("parameters") != parameters:
        raise ValueError("saved risk parameters differ from the entry policy")
    config = risk_config_type(strategy_name)(**parameters)
    count = min(top, config.allocation_count) if strategy_name == "liquidity-momentum-focus" else top
    observations = copy.deepcopy(saved["observations"])
    for row in observations:
        names = row["symbols"]
        if len(names) not in (0, count) or len(set(names)) != len(names):
            raise ValueError(
                "risk basket must contain exactly top distinct symbols or be cash"
            )
        if strategy_name == "liquidity-momentum-focus":
            if row["entry_allowed"]:
                if replay_allocation(row["allocation"], config, top, row["entry_date"]) != names:
                    raise ValueError("risk momentum basket differs from its archived allocation")
            elif names or row["allocation"] is not None:
                raise ValueError("short session must have an empty modeled basket")
        if len(row["entry_prices"]) != len(names) or len(row["exit_prices"]) != len(
            names
        ):
            raise ValueError("risk basket price count differs from its membership")
        value = unit_return(row["entry_prices"], row["exit_prices"])
        if not math.isclose(
            value, row["unscaled_return"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("saved risk return differs from its archived prices")
        row["unscaled_return"] = value
    spy = saved["spy_history"]
    spy_dates = [row["date"] for row in spy]
    if len(spy_dates) != config.trend_window or len(set(spy_dates)) != len(spy_dates):
        raise ValueError("incomplete or duplicate SPY history")
    dates = sorted(
        set(
            spy_dates
            + [row["entry_date"] for row in observations]
            + [row["exit_date"] for row in observations]
            + [entry_day]
        )
    )
    if dates[-1] != entry_day or max(spy_dates) >= entry_day:
        raise ValueError("risk history includes future information")
    # Older sessions predate explicit source metadata and archived minute_open.
    # Preserve their actual basis even though new decisions use daily closes.
    source = saved.get("spy_trend_price_source", "minute-open-1559")
    field = "price" if "spy_trend_price_source" in saved else "minute_open"
    marks = {row["date"]: row[field] for row in spy}
    result = risk_signal(
        pd.to_datetime(dates),
        observations,
        np.array([marks.get(day, np.nan) for day in dates]),
        config,
        spy_trend_price_source=source,
    )
    for field in ("trade_date", "completed_exit_date", "spy_mark_date", "strong_trend"):
        if saved[field] != result[field]:
            raise ValueError(f"risk replay differs in {field}")
    for field in (
        "target_exposure",
        "annualized_volatility",
        "spy_previous_mark",
        "spy_trend_average",
    ):
        if not math.isclose(saved[field], result[field], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"risk replay differs in {field}")
    return result


def replay_entry_decision(summary):
    """Return an explicit match/mismatch/unavailable result, never use today's config."""
    position = summary.get("position") or {}
    name = position.get("strategy_name") or "liquidity-fixed"
    inputs = position.get("decision_inputs")
    if not inputs:
        return {
            "status": "unavailable",
            "strategy_name": name,
            "reason": "entry decision inputs were not archived; backfill from historical evidence",
        }
    try:
        if inputs["version"] not in (1, REPLAY_VERSION) or name not in STRATEGIES:
            raise ValueError("unsupported decision replay version or strategy")
        if inputs["strategy_name"] != name:
            raise ValueError("entry inputs and position disagree on strategy")
        settings = inputs["configuration"]
        config = SimpleNamespace(**settings)
        config.strategy_name = name
        config.risk_config = risk_config_type(name)(
            **(position.get("strategy_parameters") or {})
        )
        checks = {}
        risk = None
        if name in RISK_STRATEGIES:
            risk = replay_risk(
                position["risk_signal"],
                position["strategy_parameters"],
                position["entry_date"],
                config.top,
                name,
            )
            sizing = trend_vol_budget(
                inputs["account"],
                config,
                risk["target_exposure"],
                inputs["positions"],
                inputs["open_orders"],
                inputs["selected_assets"],
            )
            budget = sizing["budget"]
            saved = position["exposure_sizing"]
            checks["risk_exposure"] = math.isclose(
                risk["target_exposure"], saved["target_exposure"], abs_tol=1e-12
            )
            checks["broker_limits"] = sizing == saved
        else:
            budget = available_budget(inputs["account"], config)
            sizing = None
        checks["budget"] = math.isclose(
            budget, position["budget"], rel_tol=0, abs_tol=1e-8
        )
        per_symbol = equal_notional(budget, allocation_slots(config), round_to_cents=True) if budget > 0 else 0.0
        checks["per_symbol_notional"] = math.isclose(
            per_symbol, position["per_symbol_notional"], rel_tol=0, abs_tol=1e-8
        )
        checks["share_mode"] = config.share_mode == position.get(
            "share_mode", "fractional"
        )
        names = position["symbols"]
        ranking = position.get("ranking_snapshot") or summary.get("ranking") or {}
        held = {row["symbol"] for row in inputs["positions"]}
        pending = {row["symbol"] for row in inputs["open_orders"]}
        selected = select_entry_candidates(ranking, held, pending, config)
        checks["selected_symbols"] = names == selected
        checks["basket_size"] = (
            len(names) == (allocation_slots(config) if budget > 0 else 0) and len(set(names)) == len(names)
        )
        if name == "liquidity-momentum-focus":
            checks["cash_session"] = bool(position.get("cash_session")) == (risk["target_exposure"] == 0)
            checks["ranking_risk"] = ranking.get("risk_signal") == position.get("risk_signal")
        orders = {}
        if config.share_mode == "whole" and names:
            quantities = basket_quantities(
                [position["sizing_prices"][s] for s in names], budget, "whole"
            )
            targets = dict(zip(names, quantities.astype(int).tolist(), strict=True))
            checks["target_quantities"] = targets == position["target_quantities"]
            orders = {s: {"qty": str(q)} for s, q in targets.items() if q > 0}
        else:
            orders = {s: {"notional": f"{per_symbol:.2f}"} for s in names}
        # An unsubmitted plan is still replayable. Failed/partial fills are separate
        # execution outcomes; compare requested size, never filled size, here.
        saved_orders = position.get("entry_orders") or {}
        requested_matches = True
        for symbol, order in saved_orders.items():
            if symbol not in orders:
                requested_matches = False
                continue
            field = next(iter(orders[symbol]))
            if order.get(field) is not None:
                requested_matches &= math.isclose(
                    float(order[field]),
                    float(orders[symbol][field]),
                    rel_tol=0,
                    abs_tol=1e-8,
                )
        checks["submitted_sizes"] = requested_matches
        return {
            "status": "complete" if all(checks.values()) else "mismatch",
            "strategy_name": name,
            "provenance": inputs["provenance"],
            "checks": checks,
            "selection_exclusions_known": inputs.get(
                "selection_exclusions_known", False
            ),
            "budget": budget,
            "per_symbol_notional": per_symbol,
            "orders": orders,
            "order_payloads": {
                symbol: market_entry_order(
                    date.fromisoformat(position["entry_date"]), symbol, size
                )
                for symbol, size in orders.items()
            },
            "replayed_symbols": selected,
            "target_exposure": risk["target_exposure"] if risk else None,
            "spy_trend_price_source": risk["spy_trend_price_source"] if risk else None,
            "exposure_sizing": sizing,
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        return {"status": "unavailable", "strategy_name": name, "reason": str(error)}
