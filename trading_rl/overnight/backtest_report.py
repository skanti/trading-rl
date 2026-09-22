"""Shared metadata rows for single-strategy and multi-strategy backtest reports."""

from .execution_prices import MINUTE_PRICE_COLUMNS


def policy_setting(key: str, value) -> tuple[str, str]:
    labels = {
        "volatility_target": "Volatility target", "volatility_window": "Volatility window",
        "max_exposure": "Exposure cap", "warmup_exposure": "Warmup exposure",
        "trend_window": "SPY trend window", "weak_trend_multiplier": "Weak-trend multiplier",
        "risk_history_start": "Risk history start", "trend_buffer": "Trend buffer",
        "momentum_window": "SPY momentum confirmation", "excluded_entry_weekday": "Excluded entry weekday",
        "allocation_window": "Single-horizon momentum window", "allocation_count": "Stocks per momentum basket",
        "allocation_windows": "Momentum blend windows",
    }
    if value is None:
        text = "None"
    elif key in {"volatility_target", "trend_buffer"}:
        text = f"{value:.2%}"
    elif key in {"max_exposure", "warmup_exposure", "weak_trend_multiplier"}:
        text = f"{value:g}x"
    elif key == "excluded_entry_weekday":
        text = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")[value]
    elif key == "allocation_windows":
        text = ", ".join(map(str, value)) + " sessions"
    elif key == "allocation_count" and value == 0:
        text = "Rank-weighted shortlist"
    elif key in {"volatility_window", "trend_window", "momentum_window", "allocation_window"}:
        text = f"{value} sessions" if value else "Disabled"
    else:
        text = str(value)
    return labels.get(key, key.replace("_", " ").capitalize()), text


def comparison_metadata(summaries: list[dict]) -> dict[str, list[str]]:
    """Align metadata by label; benchmark semantics differ from overnight trading."""
    columns = [strategy_metadata(summary) for summary in summaries]
    first = summaries[0]
    benchmark = first["spy_buy_and_hold_metrics"]
    spy = {
        "Period": columns[0]["Period"],
        "Sessions / trades": f"{first['strategy_metrics']['periods']} / {int(benchmark['periods'] > 0)}",
        "Short sessions": "Remains invested",
        "Exposure policy": "1x continuous buy-and-hold",
        "Mean / maximum exposure": "1.00x / 1.00x",
        "Financing": "No borrowing",
        "Borrow drag": "0.00%/yr",
        "Cost": f"{first['transaction_cost_bps_per_side']:.2f} bps per side; entry/exit once",
        "Entry price source": columns[0]["Entry price source"] + "; first session only",
        "Exit price source": columns[0]["Exit price source"] + "; final exit and daily marks",
        "Position sizing": "Fractional buy-and-hold; continuously invested",
        "KPI sampling": columns[0]["KPI sampling"],
        "Annualization": "252 sessions/year",
        "Status": "Benchmark",
        "Unique symbols": "1",
        "Daily membership changes": "None",
        "Membership stability": "Fixed SPY holding",
        "Stale exit marks": "Not separately reported",
    }
    if "Capital basis" in columns[0]:
        spy["Capital basis"] = columns[0]["Capital basis"]
    if first.get("missing_benchmark_sessions", 0):
        spy["WARNING: benchmark gaps"] = (
            f"{first['missing_benchmark_sessions']} sessions unavailable; aggregate metrics not reported"
        )
    columns.append(spy)
    labels = list(dict.fromkeys(label for column in columns for label in column))
    policy_labels = list(dict.fromkeys(
        policy_setting(key, value)[0]
        for summary in summaries for key, value in summary.get("strategy_config", {}).items()
        if key != "leverage"
    ))
    if "SPY trend price source" in labels:
        policy_labels.append("SPY trend price source")
    # Keep risk settings together even when a fixed-exposure strategy is listed first.
    labels = [label for label in labels if label not in policy_labels]
    position = labels.index("Financing") if "Financing" in labels else labels.index("Cost")
    labels[position:position] = policy_labels
    return {label: [column.get(label, "—") for column in columns] for label in labels}


def _metric_text(summary: dict[str, object]) -> str:
    minimum_trading_days = int(summary["minimum_completed_trading_days"])
    if summary["liquidity_scheme"] == "dollar_ema":
        return (
            f"lagged log-dollar-volume EMA({summary['ema_span_sessions']}), "
            f"minimum {minimum_trading_days} completed trading days"
        )
    if summary["liquidity_scheme"] == "turnover_stability":
        return (
            f"lagged log-dollar-volume EMA({summary['ema_span_sessions']}) less its "
            f"{summary['ema_span_sessions']}-session dispersion, "
            f"minimum {minimum_trading_days} completed trading days"
        )
    raise ValueError(f"unknown liquidity scheme: {summary['liquidity_scheme']}")


def strategy_metadata(summary: dict) -> dict[str, str]:
    """Format run assumptions and diagnostics without changing numerical results."""
    strategy = summary["strategy_metrics"]
    details = {}

    def add(label, value):
        details[label] = value

    add(
        "Period",
        f"{summary['first_entry_date']} {summary['entry_time_eastern']} -> "
        f"{summary['last_exit_date']} {str(summary['exit_time_eastern']).split()[0]} Eastern",
    )
    add("Sessions / trades", f"{strategy['periods']} / {summary['trades']:,}")
    skipped_short = int(summary.get("skipped_short_entry_sessions", 0))
    add("Short sessions", f"{skipped_short} afternoon entries skipped")
    add("Liquidity shortlist", f"Top {summary['top']}")
    add(
        "Liquidity ranking",
        _metric_text(summary),
    )
    add("Exposure policy", summary.get("leverage_label", "Fixed exposure"))
    if "average_exposure" in summary:
        add("Mean / maximum exposure", f"{summary['average_exposure']:.2f}x / {summary['maximum_exposure']:.2f}x")
    for key, value in summary.get("strategy_config", {}).items():
        if key == "leverage":
            continue  # Already shown by the exposure policy.
        label, formatted = policy_setting(key, value)
        if key == "allocation_window" and summary["strategy_config"].get("allocation_windows"):
            formatted += " (unused with blend)"
        add(label, formatted)
    if summary.get("spy_trend_price_source"):
        add("SPY trend price source", summary["spy_trend_price_source"])
    if "margin_interest_rate" in summary:
        add("Financing", f"{summary['margin_interest_rate']:.2%} annual; calendar days / 360")
        add("Borrow drag", f"{summary['annual_borrow_drag']:.2%}/yr")
    add("Cost", f"{summary['transaction_cost_bps_per_side']:.2f} bps per side")
    source_descriptions = {
        **{
            source: f"Alpaca SIP minute-bar {source.removeprefix('minute-')}"
            for source in MINUTE_PRICE_COLUMNS
        },
        "nbbo-ask": "latest causal SIP ask",
        "nbbo-bid": "latest causal SIP bid",
        "opening-auction": "split-adjusted primary opening auction (Alpaca SIP condition O)",
    }
    for side in ("entry", "exit"):
        source = summary[f"{side}_price_source"]
        clock = str(summary[f"{side}_time_eastern"]).split()[0]
        timing = f"at {clock} Eastern"
        if source in MINUTE_PRICE_COLUMNS and source != "minute-open":
            hour, minute = map(int, clock.split(":"))
            end_minute = (hour * 60 + minute + 1) % (24 * 60)
            timing = (
                f"during {clock}–{end_minute // 60:02d}:{end_minute % 60:02d} Eastern "
                "(hypothetical fill)"
            )
        add(
            f"{side.title()} price source",
            f"{source}: {source_descriptions[source]} {timing}",
        )
    if summary.get("skipped_missing_prices", 0):
        add(
            "WARNING: missing prices",
            f"{summary['skipped_missing_prices']} symbol/date positions skipped across "
            f"{summary['missing_price_sessions']} sessions; original allocations held as cash. "
            "Missing exits are retrospective exclusions.",
        )
    add("Annualization", "252 sessions/year")
    if "experimental" in summary:
        add("Status", "Experimental" if summary["experimental"] else "Live-supported")
    if summary.get("budget") is None:
        add("Capital basis", "$1.00 normalized start; set --budget for dollar sizing")
    if summary.get("budget") is not None:
        add(
            "Position sizing",
            f"{summary['share_mode']} shares; "
            f"mean deployed ${float(summary['average_capital_deployed']):,.2f} "
            f"({float(summary['average_capital_utilization']):.2%}), "
            f"mean basket {float(summary['average_executed_basket_size']):.2f}/"
            f"{int(summary['basket_size'])}",
        )
        if summary["share_mode"] == "whole":
            add(
                "Whole-share effects",
                f"minimum utilization {float(summary['minimum_capital_utilization']):.2%}; "
                f"minimum basket {int(summary['minimum_executed_basket_size'])}/"
                f"{int(summary['basket_size'])}; skipped selections "
                f"{int(summary['skipped_selections']):,}; mean weight spread "
                f"{float(summary['average_position_weight_spread']):.2%}",
            )
    add(
        "KPI sampling",
        f"daily at {str(summary['exit_time_eastern']).split()[0]} Eastern (exit marks)",
    )
    if float(summary.get("leverage", 1.0)) > 1.0:
        unlevered = summary["unlevered_metrics"]
        add(
            "Leverage",
            f"{summary['leverage']:.2f}x at {summary['margin_interest_rate']:.2%} annual "
            f"(rate/360 per calendar day, mean hold "
            f"{summary['mean_holding_calendar_days']:.2f}d); "
            f"borrow drag {summary['annual_borrow_drag']:.2%}/yr",
        )
        add(
            "Unlevered comparison",
            f"return {unlevered['annualized_return']:.2%}, "
            f"Sharpe {unlevered['sharpe_zero_cash_rate']:.2f}, "
            f"max drawdown {unlevered['max_drawdown']:.2%}",
        )
        add(
            "Margin headroom",
            f"worst session leaves {summary['worst_session_margin_ratio']:.0%} equity "
            f"against a {summary['maintenance_margin']:.0%} floor; "
            f"that session breaches at {summary['margin_breach_leverage']:.2f}x",
        )
    # Tolerate summaries built before these keys existed rather than raising in display.
    duplicate_days = int(summary.get("sessions_with_two_classes_of_one_issuer", 0))
    unnamed = int(summary.get("symbols_without_an_issuer_name", 0))
    if "deduped_share_classes" not in summary:
        detail = None
    elif summary["deduped_share_classes"]:
        detail = "one share class per company"
        if duplicate_days:
            detail = f"FAILED: {duplicate_days} session(s) still hold two classes of one issuer"
        if unnamed:
            detail += f"; {unnamed} traded symbol(s) had no name in the security master"
    else:
        detail = "disabled"
        if duplicate_days:
            detail += f"; {duplicate_days} session(s) hold two classes of one issuer"
    if detail is not None:
        add("Share classes", detail)
    add("Unique symbols", f"{summary['unique_symbols_traded']:,}")
    add(
        "Daily membership changes",
        f"mean {summary['average_daily_membership_replacements']:.2f}, "
        f"maximum {summary['maximum_daily_membership_replacements']}",
    )
    add(
        "Membership stability",
        f"retention {summary['average_daily_membership_retention']:.2%}, "
        f"Jaccard {summary['average_daily_membership_jaccard']:.3f}",
    )
    add(
        "Stale exit marks",
        f"{summary['stale_exit_marks_over_10_minutes']} over 10 minutes; "
        f"maximum {summary['maximum_exit_staleness_minutes']:.1f} minutes",
    )
    return details
