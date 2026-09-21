"""Failure callbacks that cannot replace the original ranking exception."""

import json
import logging
import os
import tempfile
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from trading_rl.notifications import send_text_email

LOGGER = logging.getLogger("overnight-liquidity-live")


@dataclass(frozen=True)
class RankingFailure:
    trade_date: date
    strategy_name: str
    occurred_at: str
    error_type: str
    error_message: str
    traceback: str
    log_path: Path


RankingFailureCallback = Callable[[RankingFailure], None]


@contextmanager
def notify_ranking_failure(
    callback: RankingFailureCallback | None,
    trade_date: date,
    strategy_name: str,
    log_path: Path,
):
    try:
        yield
    except Exception as error:
        if callback is not None:
            failure = RankingFailure(
                trade_date,
                strategy_name,
                datetime.now(UTC).isoformat(),
                type(error).__name__,
                str(error),
                traceback.format_exc(),
                log_path,
            )
            try:
                callback(failure)
            except Exception:
                LOGGER.exception(
                    "ranking failure callback failed; retaining original ranking error"
                )
        raise


class RankingFailureEmail:
    """Send once per strategy/session, remembering deliveries across restarts.

    A failed SMTP attempt is retried on the next ranking failure. No notification
    worker is started, and this callback is never part of entry preflight.
    """

    def __init__(
        self,
        recipient: str,
        state_path: Path,
        *,
        config_path: Path | None = None,
        timeout_seconds: float = 10.0,
        account_mode: str = "live",
    ):
        self.recipient = recipient
        self.state_path = state_path
        self.config_path = config_path
        self.timeout_seconds = timeout_seconds
        self.account_mode = account_mode
        self.delivered: set[str] = set()

    def __call__(self, failure: RankingFailure) -> None:
        key = f"{failure.trade_date}:{failure.strategy_name}:{self.recipient}"
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text())
                self.delivered.update(payload["delivered"])
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.warning("could not read ranking notification delivery history")
        if key in self.delivered:
            return
        send_text_email(
            self.recipient,
            f"[{self.account_mode.upper()}] Ranking failed: {failure.strategy_name} — {failure.trade_date}",
            f"Ranking preparation failed.\n\n"
            f"Session: {failure.trade_date}\nStrategy: {failure.strategy_name}\n"
            f"Occurred at: {failure.occurred_at}\n"
            f"Error: {failure.error_type}: {failure.error_message}\n\n"
            "The daemon retries during the ranking window. Entry requires a valid, "
            "timely ranking and risk signal. A manual rank command must be rerun.\n"
            "Only the first ranking failure per strategy/session is emailed; "
            "subsequent errors remain in the log.\n\n"
            f"Log: {failure.log_path}\n\nTraceback:\n{failure.traceback}",
            config_path=self.config_path,
            timeout_seconds=self.timeout_seconds,
        )
        # Remember success in memory even if persisting the marker fails.
        self.delivered.add(key)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.state_path.parent,
                prefix=self.state_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "delivered": sorted(self.delivered)}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        LOGGER.info("ranking failure notification sent to %s", self.recipient)
