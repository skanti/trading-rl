"""Small SMTP transport, independent of trading and dashboard execution."""

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from omegaconf import OmegaConf


def smtp_config_path(explicit: Path | None = None) -> Path:
    """Reuse the existing local SMTP configuration without importing its consumer."""
    if explicit is not None:
        return Path(explicit).expanduser()
    if os.environ.get("DASHBOARD_CONFIG"):
        return Path(os.environ["DASHBOARD_CONFIG"]).expanduser()
    return Path(__file__).resolve().parents[1] / "dashboard" / "config.yaml"


def send_text_email(
    recipient: str,
    subject: str,
    body: str,
    *,
    config_path: Path | None = None,
    timeout_seconds: float = 10.0,
) -> None:
    config = OmegaConf.load(smtp_config_path(config_path))
    values = {
        name: OmegaConf.select(config, f"smtp.{name}")
        for name in ("host", "port", "user", "password")
    }
    missing = [name for name, value in values.items() if value in (None, "")]
    if missing:
        raise ValueError("SMTP configuration is missing fields: " + ", ".join(missing))
    sender = str(values["user"])
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="trading-rl.local")
    message.set_content(body)
    with smtplib.SMTP(
        str(values["host"]), int(values["port"]), timeout=timeout_seconds
    ) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(sender, str(values["password"]))
        refused = smtp.send_message(message, from_addr=sender, to_addrs=[recipient])
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
