"""Daily digest email sent when the overnight basket is closed out.

The digest answers one question: what did the account do, and what did this morning's
exit realise? Numbers come from :mod:`baseline.performance`, so the table in the email
and the table on the dashboard are computed by the same code.

Nothing in here may raise into the trading daemon. Callers in
``live_overnight_liquidity`` wrap :func:`send_digest` in a bare ``except``; this module
keeps its own failure modes narrow and its side effects confined to one SMTP session.
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from baseline import performance
from baseline.dashboard_config import load_dashboard_config
from baseline.performance import (
    BUCKET_ORDER,
    ClosedTrade,
    EASTERN,
    PerformanceBucket,
    Statistics
)

LOGGER = logging.getLogger("overnight-liquidity-digest")

# Inline styles only: Gmail and Outlook strip or ignore <style> blocks unpredictably.
_INK = "#0f172a"
_MUTED = "#64748b"
_LINE = "#e2e8f0"
_GAIN = "#047857"
_LOSS = "#be123c"
_CANVAS = "#f8fafc"


@dataclass(frozen=True)
class Digest:
    title: str
    dashboard_url: str
    trading_day: date
    generated_at: datetime
    account: Mapping[str, Any]
    buckets: Mapping[str, PerformanceBucket]
    stats: Statistics
    trades: Sequence[ClosedTrade] = field(default_factory=tuple)
    totals: Mapping[str, float] = field(default_factory=dict)
    entry_date: str | None = None
    exit_date: str | None = None
    position_status: str | None = None

    @property
    def equity(self) -> float:
        return performance._float(self.account.get("equity"))

    @property
    def subject(self) -> str:
        today = self.buckets.get("today")
        if today is None:
            return f"{self.title} — {self.trading_day.isoformat()}"
        return (
            f"{self.title} — {self.trading_day.isoformat()} — "
            f"{_signed_money(today.pnl)} ({_signed_pct(today.pnl_pct)})"
        )


# --------------------------------------------------------------------------- format


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def _signed_pct(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:,.2f}%"


def _qty(value: float) -> str:
    # Fractional notional orders produce long decimals; six places is plenty.
    text = f"{value:,.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _tone(value: float) -> str:
    if value > 0:
        return _GAIN
    if value < 0:
        return _LOSS
    return _INK


# --------------------------------------------------------------------------- build


def build_digest(
    account: Mapping[str, Any],
    history: Mapping[str, Any],
    position: Mapping[str, Any] | None,
    config: DictConfig,
    now: datetime | None = None
) -> Digest:
    """Assemble the digest from already-fetched Alpaca payloads."""
    reference = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)
    created = account.get("created_at")
    inception = None
    if created:
        try:
            inception = datetime.fromisoformat(str(created).replace("Z", "+00:00")).astimezone(EASTERN).date()
        except ValueError:
            inception = None

    series = performance.equity_series(history, since=inception)
    buckets = performance.performance_table(series, account, history, now=reference)
    stats = performance.statistics(series)

    trades: Sequence[ClosedTrade] = ()
    totals: Mapping[str, float] = {}
    if position:
        trades = performance.closed_basket(position)
        totals = performance.basket_totals(trades)

    return Digest(
        title=str(OmegaConf.select(config, "dashboard.title") or "Trading account"),
        dashboard_url=str(OmegaConf.select(config, "dashboard.url") or ""),
        trading_day=reference.date(),
        generated_at=reference,
        account=account,
        buckets=buckets,
        stats=stats,
        trades=trades,
        totals=totals,
        entry_date=str(position.get("entry_date")) if position and position.get("entry_date") else None,
        exit_date=str(position.get("exit_date")) if position and position.get("exit_date") else None,
        position_status=str(position.get("status")) if position and position.get("status") else None
    )


def build_digest_live(
    client: Any,
    position: Mapping[str, Any] | None,
    config: DictConfig,
    now: datetime | None = None
) -> Digest:
    """Fetch what the digest needs from Alpaca and build it.

    ``client`` is a :class:`baseline.live_overnight_liquidity.AlpacaClient`; it is typed
    loosely so this module never has to import the trading script at call time.
    """
    account = client.account()
    period = performance.history_period(account.get("created_at"), now)
    history = client.portfolio_history(period=period, timeframe="1D")
    return build_digest(account, history, position, config, now=now)


# --------------------------------------------------------------------------- render


def render_text(digest: Digest) -> str:
    """Plain-text alternative. Column widths are fixed so it aligns in a mono font."""
    lines = [
        digest.title,
        f"Trading day {digest.trading_day.isoformat()}",
        "",
        f"{'Period':<18}{'P&L':>16}{'P&L %':>11}{'Equity':>16}",
        "-" * 61
    ]
    first = True
    for key in BUCKET_ORDER:
        bucket = digest.buckets.get(key)
        if bucket is None:
            continue
        # Every bucket ends on the same live equity, so print it once rather than
        # repeating one number down the column.
        equity = _money(bucket.end_equity) if first else ""
        lines.append(
            f"{bucket.label:<18}"
            f"{_signed_money(bucket.pnl):>16}"
            f"{_signed_pct(bucket.pnl_pct):>11}"
            f"{equity:>16}"
        )
        first = False

    lines += ["", "Closed this morning", "-" * 61]
    if digest.trades:
        lines.append(
            f"{'Symbol':<8}{'Qty':>12}{'Entry':>11}{'Exit':>11}{'P&L':>12}{'P&L %':>9}"
        )
        for trade in digest.trades:
            lines.append(
                f"{trade.symbol:<8}"
                f"{_qty(trade.qty):>12}"
                f"{_money(trade.entry_price):>11}"
                f"{_money(trade.exit_price):>11}"
                f"{_signed_money(trade.pnl):>12}"
                f"{_signed_pct(trade.pnl_pct):>9}"
            )
        if digest.totals:
            lines.append("-" * 63)
            lines.append(
                f"{'Total':<8}{'':>12}{'':>11}{'':>11}"
                f"{_signed_money(digest.totals.get('pnl', 0.0)):>12}"
                f"{_signed_pct(digest.totals.get('pnl_pct', 0.0)):>9}"
            )
    else:
        lines.append("No positions were held overnight.")

    stats = digest.stats
    lines += [
        "",
        "Account",
        "-" * 61,
        f"{'Equity':<24}{_money(digest.equity)}",
        f"{'Cash':<24}{_money(performance._float(digest.account.get('cash')))}",
        f"{'Max drawdown':<24}{_money(stats.max_drawdown)} ({_signed_pct(-abs(stats.max_drawdown_pct))})",
        f"{'Winning sessions':<24}{stats.winning_sessions} of "
        f"{stats.winning_sessions + stats.losing_sessions} ({stats.win_rate * 100:,.1f}%)"
    ]
    if digest.dashboard_url:
        lines += ["", f"View the dashboard: {digest.dashboard_url}"]
    lines += ["", f"Generated {digest.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"]
    return "\n".join(lines)


def _cell(content: str, *, align: str = "left", color: str = _INK, weight: str = "400") -> str:
    return (
        f'<td style="padding:10px 12px;border-bottom:1px solid {_LINE};'
        f'text-align:{align};color:{color};font-weight:{weight};'
        f'font-variant-numeric:tabular-nums;white-space:nowrap">{content}</td>'
    )


def _header_cell(content: str, *, align: str = "left") -> str:
    return (
        f'<th style="padding:8px 12px;border-bottom:2px solid {_LINE};'
        f'text-align:{align};color:{_MUTED};font-size:12px;font-weight:600;'
        f'letter-spacing:.04em;text-transform:uppercase">{content}</th>'
    )


def render_html(digest: Digest) -> str:
    """HTML alternative, table-based and inline-styled for email client compatibility."""
    table_open = (
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        'style="width:100%;border-collapse:collapse;font-size:14px">'
    )

    performance_rows = []
    for index, key in enumerate(BUCKET_ORDER):
        bucket = digest.buckets.get(key)
        if bucket is None:
            continue
        equity = _money(bucket.end_equity) if index == 0 else ""
        performance_rows.append(
            "<tr>"
            + _cell(bucket.label, weight="600")
            + _cell(_signed_money(bucket.pnl), align="right", color=_tone(bucket.pnl), weight="600")
            + _cell(_signed_pct(bucket.pnl_pct), align="right", color=_tone(bucket.pnl))
            + _cell(equity, align="right", color=_MUTED)
            + "</tr>"
        )

    if digest.trades:
        trade_rows = [
            "<tr>"
            + _cell(trade.symbol, weight="600")
            + _cell(_qty(trade.qty), align="right", color=_MUTED)
            + _cell(_money(trade.entry_price), align="right")
            + _cell(_money(trade.exit_price), align="right")
            + _cell(_signed_money(trade.pnl), align="right", color=_tone(trade.pnl), weight="600")
            + _cell(_signed_pct(trade.pnl_pct), align="right", color=_tone(trade.pnl))
            + "</tr>"
            for trade in digest.trades
        ]
        total_pnl = digest.totals.get("pnl", 0.0)
        # Notional totals sit under the columns they belong to: deployed under Entry,
        # realised under Exit.
        trade_rows.append(
            "<tr>"
            + _cell("Total", weight="700")
            + _cell("", align="right")
            + _cell(_money(digest.totals.get("entry_notional", 0.0)), align="right", color=_MUTED)
            + _cell(_money(digest.totals.get("exit_notional", 0.0)), align="right", color=_MUTED)
            + _cell(_signed_money(total_pnl), align="right", color=_tone(total_pnl), weight="700")
            + _cell(_signed_pct(digest.totals.get("pnl_pct", 0.0)), align="right", color=_tone(total_pnl), weight="700")
            + "</tr>"
        )
        trades_block = (
            table_open
            + "<thead><tr>"
            + _header_cell("Symbol")
            + _header_cell("Qty", align="right")
            + _header_cell("Entry", align="right")
            + _header_cell("Exit", align="right")
            + _header_cell("P&amp;L", align="right")
            + _header_cell("P&amp;L %", align="right")
            + "</tr></thead><tbody>"
            + "".join(trade_rows)
            + "</tbody></table>"
        )
    else:
        trades_block = (
            f'<p style="margin:0;padding:14px 16px;background:{_CANVAS};'
            f'border:1px solid {_LINE};border-radius:8px;color:{_MUTED}">'
            "No positions were held overnight.</p>"
        )

    stats = digest.stats
    held = ""
    if digest.entry_date and digest.exit_date:
        held = (
            f'<p style="margin:4px 0 0;color:{_MUTED};font-size:13px">'
            f"Basket entered {digest.entry_date}, exited {digest.exit_date}"
            + (f" ({digest.position_status})" if digest.position_status else "")
            + "</p>"
        )

    button = ""
    if digest.dashboard_url:
        button = (
            f'<p style="margin:28px 0 0"><a href="{digest.dashboard_url}" '
            f'style="display:inline-block;padding:11px 20px;background:{_INK};color:#ffffff;'
            'border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">'
            "View the dashboard &rarr;</a></p>"
        )

    return f"""<!doctype html>
<html><body style="margin:0;padding:24px;background:{_CANVAS};
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:{_INK}">
<div style="max-width:640px;margin:0 auto;background:#ffffff;border:1px solid {_LINE};
border-radius:12px;padding:28px">
  <h1 style="margin:0;font-size:20px;font-weight:700">{digest.title}</h1>
  <p style="margin:4px 0 0;color:{_MUTED};font-size:13px">
    Trading day {digest.trading_day.isoformat()} &middot; equity {_money(digest.equity)}
  </p>
  {held}

  <h2 style="margin:28px 0 8px;font-size:13px;font-weight:600;color:{_MUTED};
  letter-spacing:.04em;text-transform:uppercase">Performance</h2>
  {table_open}
    <thead><tr>
      {_header_cell("Period")}
      {_header_cell("P&amp;L", align="right")}
      {_header_cell("P&amp;L %", align="right")}
      {_header_cell("Equity", align="right")}
    </tr></thead>
    <tbody>{"".join(performance_rows)}</tbody>
  </table>

  <h2 style="margin:28px 0 8px;font-size:13px;font-weight:600;color:{_MUTED};
  letter-spacing:.04em;text-transform:uppercase">Closed this morning</h2>
  {trades_block}

  <h2 style="margin:28px 0 8px;font-size:13px;font-weight:600;color:{_MUTED};
  letter-spacing:.04em;text-transform:uppercase">Account</h2>
  {table_open}
    <tbody>
      <tr>{_cell("Cash")}{_cell(_money(performance._float(digest.account.get("cash"))), align="right")}</tr>
      <tr>{_cell("Max drawdown")}{_cell(
        f"{_money(stats.max_drawdown)} ({_signed_pct(-abs(stats.max_drawdown_pct))})",
        align="right", color=_LOSS if stats.max_drawdown > 0 else _INK)}</tr>
      <tr>{_cell("Winning sessions")}{_cell(
        f"{stats.winning_sessions} of {stats.winning_sessions + stats.losing_sessions}"
        f" ({stats.win_rate * 100:,.1f}%)", align="right")}</tr>
    </tbody>
  </table>

  {button}
  <p style="margin:24px 0 0;color:{_MUTED};font-size:12px">
    Generated {digest.generated_at.strftime("%Y-%m-%d %H:%M:%S %Z")}
  </p>
</div>
</body></html>"""


# ----------------------------------------------------------------------------- send


def build_message(digest: Digest, config: DictConfig) -> EmailMessage:
    """A ``multipart/alternative`` message ready to hand to an SMTP session."""
    recipients = [str(value) for value in OmegaConf.select(config, "notifications.recipients")]
    sender = str(OmegaConf.select(config, "smtp.user"))

    message = EmailMessage()
    message["Subject"] = digest.subject
    message["From"] = formataddr((digest.title, sender))
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="trading-rl.local")
    message.set_content(render_text(digest))
    message.add_alternative(render_html(digest), subtype="html")
    return message


def send_digest(digest: Digest, config: DictConfig) -> list[str]:
    """Send the digest over SMTP with STARTTLS. Returns the recipients addressed."""
    host = str(OmegaConf.select(config, "smtp.host"))
    port = int(OmegaConf.select(config, "smtp.port"))
    user = str(OmegaConf.select(config, "smtp.user"))
    password = str(OmegaConf.select(config, "smtp.password"))
    recipients = [str(value) for value in OmegaConf.select(config, "notifications.recipients")]

    message = build_message(digest, config)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(message, from_addr=user, to_addrs=recipients)
    LOGGER.info("digest emailed to %s", ", ".join(recipients))
    return recipients


# ------------------------------------------------------------------------------ cli


def _load_state_position(state_path: Path | None) -> Mapping[str, Any] | None:
    if state_path is None or not state_path.exists():
        return None
    import json

    state = json.loads(state_path.read_text())
    return state.get("position") or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render or send the overnight-liquidity digest email."
    )
    parser.add_argument("--config", default=None, help="path to dashboard/config.yaml")
    parser.add_argument(
        "--state-path",
        default=None,
        help="strategy state.json whose position supplies the closed-basket table",
    )
    parser.add_argument(
        "--preview",
        default=None,
        help="write the rendered HTML here and send nothing",
    )
    parser.add_argument(
        "--trading-url",
        default=os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets/v2"),
        help="Alpaca endpoint (default: $ALPACA_URL, else the paper endpoint)",
    )
    parser.add_argument("--send", action="store_true", help="actually send the email")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config = load_dashboard_config(args.config)

    # Imported here so `python -m baseline.reporting --preview` stays cheap to start.
    from baseline.live_overnight_liquidity import AlpacaClient, load_credentials

    key, secret = load_credentials()
    client = AlpacaClient(key, secret, args.trading_url)
    position = _load_state_position(Path(args.state_path).expanduser() if args.state_path else None)
    digest = build_digest_live(client, position, config)

    if args.preview:
        target = Path(args.preview).expanduser()
        target.write_text(render_html(digest))
        print(render_text(digest))
        print(f"\nHTML written to {target}")
    if args.send:
        send_digest(digest, config)
        print(f"Sent: {digest.subject}")
    if not args.preview and not args.send:
        print(render_text(digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
