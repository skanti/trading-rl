"""Render and deliver one email digest when a strategy basket closes."""

from __future__ import annotations

from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from html import escape
import json
import os
from pathlib import Path
import smtplib
import tempfile
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

import dashboard_metrics as performance
from dashboard_config import DashboardConfigError


def _money(value: object, *, signed: bool = False) -> str:
    amount = performance._float(value)
    if signed:
        return f"{'+' if amount >= 0.0 else '-'}${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def _percent(value: object, *, signed: bool = False) -> str:
    amount = performance._float(value) * 100.0
    prefix = "+" if signed and amount >= 0.0 else ""
    return f"{prefix}{amount:,.2f}%"


def _tone(value: object) -> str:
    amount = performance._float(value)
    if amount > 0.0:
        return "#047857"
    if amount < 0.0:
        return "#be123c"
    return "#475569"


def digest_key(state: Mapping[str, Any], trading_day: str) -> str | None:
    """Stable identity for today's completed basket, or ``None`` until it closes."""
    position = state.get("position") or {}
    if position.get("status") != "closed" or position.get("exit_date") != trading_day:
        return None
    entry_date = position.get("entry_date")
    completed_at = position.get("exit_completed_at")
    if not entry_date or not completed_at:
        return None
    return f"{entry_date}:{completed_at}"


def _mail_settings(config: DictConfig) -> tuple[str, int, str, str, list[str]]:
    values = {
        key: OmegaConf.select(config, key)
        for key in ("smtp.host", "smtp.port", "smtp.user", "smtp.password")
    }
    missing = [key for key, value in values.items() if value in (None, "")]
    recipients_value = OmegaConf.select(config, "notifications.recipients") or []
    recipients = [str(value).strip() for value in recipients_value if str(value).strip()]
    if not recipients:
        missing.append("notifications.recipients")
    if missing:
        raise DashboardConfigError(
            "digest email configuration is missing: " + ", ".join(missing)
        )
    return (
        str(values["smtp.host"]),
        int(values["smtp.port"]),
        str(values["smtp.user"]),
        str(values["smtp.password"]),
        recipients,
    )


def render_text(
    snapshot: Mapping[str, Any], state: Mapping[str, Any], config: DictConfig
) -> str:
    title = str(OmegaConf.select(config, "dashboard.title") or "Trading account")
    account = snapshot.get("account") or {}
    position = state.get("position") or {}
    lines = [
        title,
        f"Trading day {snapshot.get('trading_day')}",
        f"Basket {position.get('entry_date')} -> {position.get('exit_date')}: closed",
        "",
        f"{'Period':<18}{'P&L':>16}{'P&L %':>12}",
        "-" * 46,
    ]
    buckets = snapshot.get("performance") or {}
    for key in performance.BUCKET_ORDER:
        bucket = buckets.get(key)
        if not bucket:
            continue
        lines.append(
            f"{str(bucket.get('label') or key):<18}"
            f"{_money(bucket.get('pnl'), signed=True):>16}"
            f"{_percent(bucket.get('pnl_pct'), signed=True):>12}"
        )

    lines += ["", "Closed basket", "-" * 73]
    trades = list(snapshot.get("closed_basket") or [])
    if trades:
        lines.append(
            f"{'Symbol':<8}{'Qty':>12}{'Entry':>12}{'Exit':>12}{'P&L':>16}{'P&L %':>12}"
        )
        for trade in trades:
            lines.append(
                f"{str(trade.get('symbol') or ''):<8}"
                f"{performance._float(trade.get('qty')):>12,.4f}"
                f"{_money(trade.get('entry_price')):>12}"
                f"{_money(trade.get('exit_price')):>12}"
                f"{_money(trade.get('pnl'), signed=True):>16}"
                f"{_percent(trade.get('pnl_pct'), signed=True):>12}"
            )
    else:
        lines.append("No per-symbol fills were available.")

    today = buckets.get("today") or {}
    lines += [
        "",
        f"Equity: {_money(account.get('equity'))}",
        f"Cash: {_money(account.get('cash'))}",
        f"Today's account P&L: {_money(today.get('pnl'), signed=True)} "
        f"({_percent(today.get('pnl_pct'), signed=True)})",
    ]
    dashboard_url = str(OmegaConf.select(config, "dashboard.url") or "")
    if dashboard_url:
        lines += ["", f"Dashboard: {dashboard_url}"]
    return "\n".join(lines)


def render_html(
    snapshot: Mapping[str, Any], state: Mapping[str, Any], config: DictConfig
) -> str:
    title = escape(str(OmegaConf.select(config, "dashboard.title") or "Trading account"))
    account = snapshot.get("account") or {}
    position = state.get("position") or {}
    buckets = snapshot.get("performance") or {}
    performance_rows = "".join(
        "<tr>"
        f"<td>{escape(str(bucket.get('label') or key))}</td>"
        f"<td class='number' style='color:{_tone(bucket.get('pnl'))}'>"
        f"{_money(bucket.get('pnl'), signed=True)}</td>"
        f"<td class='number' style='color:{_tone(bucket.get('pnl'))}'>"
        f"{_percent(bucket.get('pnl_pct'), signed=True)}</td>"
        "</tr>"
        for key in performance.BUCKET_ORDER
        if (bucket := buckets.get(key))
    )
    trade_rows = "".join(
        "<tr>"
        f"<td>{escape(str(trade.get('symbol') or ''))}</td>"
        f"<td class='number'>{performance._float(trade.get('qty')):,.4f}</td>"
        f"<td class='number'>{_money(trade.get('entry_price'))}</td>"
        f"<td class='number'>{_money(trade.get('exit_price'))}</td>"
        f"<td class='number' style='color:{_tone(trade.get('pnl'))}'>"
        f"{_money(trade.get('pnl'), signed=True)}</td>"
        f"<td class='number' style='color:{_tone(trade.get('pnl'))}'>"
        f"{_percent(trade.get('pnl_pct'), signed=True)}</td>"
        "</tr>"
        for trade in snapshot.get("closed_basket") or []
    )
    if not trade_rows:
        trade_rows = "<tr><td colspan='6'>No per-symbol fills were available.</td></tr>"
    dashboard_url = escape(str(OmegaConf.select(config, "dashboard.url") or ""), quote=True)
    link = f"<p><a class='button' href='{dashboard_url}'>View dashboard</a></p>" if dashboard_url else ""
    return f"""<!doctype html>
<html><head><style>
body {{ background:#f8fafc; color:#0f172a; font-family:Arial,sans-serif; padding:24px }}
.card {{ background:white; border:1px solid #e2e8f0; border-radius:12px; margin:auto; max-width:720px; padding:28px }}
table {{ border-collapse:collapse; width:100%; margin-bottom:24px }}
th,td {{ border-bottom:1px solid #e2e8f0; padding:9px 10px; text-align:left }}
th {{ color:#64748b; font-size:12px; text-transform:uppercase }}
.number {{ text-align:right; font-variant-numeric:tabular-nums }}
.muted {{ color:#64748b }}
.button {{ background:#0f172a; border-radius:8px; color:white; display:inline-block; padding:10px 16px; text-decoration:none }}
</style></head><body><div class="card">
<h1>{title}</h1>
<p class="muted">Trading day {escape(str(snapshot.get('trading_day') or ''))} · equity {_money(account.get('equity'))}</p>
<p>Basket {escape(str(position.get('entry_date') or ''))} → {escape(str(position.get('exit_date') or ''))}: closed</p>
<h2>Performance</h2>
<table><thead><tr><th>Period</th><th class="number">P&amp;L</th><th class="number">P&amp;L %</th></tr></thead><tbody>{performance_rows}</tbody></table>
<h2>Closed basket</h2>
<table><thead><tr><th>Symbol</th><th class="number">Qty</th><th class="number">Entry</th><th class="number">Exit</th><th class="number">P&amp;L</th><th class="number">P&amp;L %</th></tr></thead><tbody>{trade_rows}</tbody></table>
{link}
</div></body></html>"""


def build_message(
    snapshot: Mapping[str, Any], state: Mapping[str, Any], config: DictConfig
) -> EmailMessage:
    _, _, sender, _, recipients = _mail_settings(config)
    today = (snapshot.get("performance") or {}).get("today") or {}
    title = str(OmegaConf.select(config, "dashboard.title") or "Trading account")
    subject = (
        f"{title} — {snapshot.get('trading_day')} — "
        f"{_money(today.get('pnl'), signed=True)} ({_percent(today.get('pnl_pct'), signed=True)})"
    )
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((title, sender))
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="trading-rl.local")
    message.set_content(render_text(snapshot, state, config))
    message.add_alternative(render_html(snapshot, state, config), subtype="html")
    return message


def send_digest(
    snapshot: Mapping[str, Any], state: Mapping[str, Any], config: DictConfig
) -> list[str]:
    host, port, user, password, recipients = _mail_settings(config)
    message = build_message(snapshot, state, config)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(message, from_addr=user, to_addrs=recipients)
    return recipients


def delivered_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(value) for value in payload.get("delivered", [])}


def mark_delivered(path: Path, key: str) -> None:
    keys = delivered_keys(path)
    keys.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as temporary:
        json.dump({"version": 1, "delivered": sorted(keys)}, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
