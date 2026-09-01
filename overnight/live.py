"""Live/paper Alpaca execution for the causal overnight-liquidity basket.

The process ranks exchange-listed US company stocks from a bounded window of
*completed* daily bars, buys an equal-target basket late in the regular session,
and closes only those strategy-owned positions on the next trading morning.
Fractional mode additionally requires fractionable assets. State and deterministic
client order IDs make the workflow restartable and idempotent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time as time_module
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import requests
from rich.console import Console
from rich.table import Table

from backtest import (
    DEFAULT_SECURITY_MASTER_CACHE,
    basket_quantities,
    issuer_key,
    _security_symbol,
    causal_ema_log_liquidity,
    causal_turnover_stability,
    is_company_security,
    load_nasdaq_security_master,
)


EASTERN = ZoneInfo("America/New_York")
PAPER_TRADING_URL = "https://paper-api.alpaca.markets/v2"
DEFAULT_DATA_URL = "https://data.alpaca.markets/v2"
DEFAULT_WORK_DIR = Path("/data/ppv1/live")
DEFAULT_STATE_PATH = str(DEFAULT_WORK_DIR / "state.json")
DEFAULT_DAILY_BARS_DIR = Path("/data/ppv1/updates/bars_1day_2016-01-01")
DEFAULT_LIQUIDITY_SHORTLIST = (
    Path(__file__).resolve().parents[1] / "data" / "most_liquid.txt"
)
DEFAULT_SHORTLIST_SINCE = date(2022, 1, 1)
# One session in the daily top-N used to buy permanent candidacy, so the shortlist
# only ever grew. A trailing year keeps it tracking current liquidity; measured over
# the last 500 sessions, no window down to 125 ever dropped a name from the reserve.
DEFAULT_SHORTLIST_LOOKBACK_SESSIONS = 250
DAILY_BAR_ANNO = datetime(2010, 1, 1, tzinfo=UTC)
DAILY_BAR_COLUMNS = 8
DEFAULT_EXCHANGES = frozenset({"NASDAQ"})
# Nasdaq stops accepting market orders for the opening cross at 09:28, so an exit has
# to reach Alpaca before then. Submission opens early enough to absorb Alpaca's own
# queuing: sub-one-share orders are parked for a pre-open batch release around 09:15.
EXIT_SUBMISSION_OPEN = time(8, 0)
OPENING_AUCTION_CUTOFF = time(9, 28)
REGULAR_MARKET_OPEN = time(9, 30)
TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}
)
LOGGER = logging.getLogger("overnight-liquidity-live")
CONSOLE = Console()
RANKING_PIPELINE_VERSION = 6


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


class DailyLogHandler(logging.Handler):
    """Append logs to the current ET trading-day directory."""

    def __init__(self, work_dir: Path, fixed_day: date | None = None):
        super().__init__()
        self.work_dir = work_dir
        self.fixed_day = fixed_day

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = self.fixed_day or datetime.now(tz=EASTERN).date()
            path = self.work_dir / day.isoformat() / "live.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


class DailyArtifacts:
    """Persist auditable, credential-free inputs and outcomes for each day."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def directory(self, day: date) -> Path:
        path = self.work_dir / day.isoformat()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_ticks(
        self,
        day: date,
        bars_by_symbol: Mapping[str, Sequence[Mapping[str, object]]],
        *,
        feed: str,
        start: date | datetime,
        end: date | datetime,
    ) -> dict[str, object]:
        """Write the historical market bars consumed by that day's ranking.

        The filename follows the project's tick terminology, while every row
        explicitly records that the Alpaca source timeframe is ``1Day``.
        """
        path = self.directory(day) / "ticks.jsonl"
        row_count = 0
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            for symbol in sorted(bars_by_symbol):
                for bar in bars_by_symbol[symbol]:
                    record = {
                        "symbol": symbol,
                        "timeframe": "1Day",
                        "feed": feed,
                        **dict(bar),
                    }
                    temporary.write(json.dumps(record, sort_keys=True) + "\n")
                    row_count += 1
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        metadata = {
            "path": str(path),
            "rows": row_count,
            "symbols": len(bars_by_symbol),
            "timeframe": "1Day",
            "feed": feed,
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "written_at": _iso_now(),
        }
        _atomic_write_json(self.directory(day) / "ticks_summary.json", metadata)
        return metadata

    def write_summary(
        self,
        day: date,
        action: str,
        store: "StateStore",
        config: "StrategyConfig",
        *,
        error: str | None = None,
    ) -> None:
        state = store.load()
        position = dict(state.get("position") or {})
        entry_orders = position.get("entry_orders") or {}
        exit_orders = position.get("exit_orders") or {}

        def filled_notional(orders: Mapping[str, Mapping[str, object]]) -> float:
            total = 0.0
            for order in orders.values():
                try:
                    total += float(order.get("filled_qty") or 0.0) * float(
                        order.get("filled_avg_price") or 0.0
                    )
                except (TypeError, ValueError):
                    continue
            return total

        entry_notional = filled_notional(entry_orders)
        exit_notional = filled_notional(exit_orders)
        realized_pnl = exit_notional - entry_notional if exit_orders else None
        summary: dict[str, object] = {
            "version": 1,
            "strategy": "overnight_liquidity_long",
            "trading_day": day.isoformat(),
            "updated_at": _iso_now(),
            "last_action": action,
            "state_path": str(store.path),
            "configuration": {
                "top": config.top,
                "ema_span": config.ema_span,
                "minimum_trading_days": config.minimum_trading_days,
                "liquidity_lookback_calendar_days": config.lookback_calendar_days,
                "daily_bars_dir": str(config.daily_bars_dir),
                "liquidity_shortlist": str(config.liquidity_shortlist),
                "shortlist_since": config.shortlist_since.isoformat(),
                "shortlist_daily_top": config.shortlist_daily_top,
                "shortlist_lookback_sessions": config.shortlist_lookback_sessions,
                "liquidity_scheme": config.liquidity_scheme,
                "daily_overlap_days": config.daily_overlap_days,
                "feed": config.feed,
                "ranking_feed": config.feed,
                "quote_feed": config.quote_feed,
                "exchanges": sorted(config.exchanges),
                "capital": config.capital,
                "capital_fraction": config.capital_fraction,
                "cash_buffer_fraction": config.cash_buffer_fraction,
                "share_mode": config.share_mode,
                "quote_max_age_seconds": config.quote_max_age_seconds,
            },
            "ranking": state.get("ranking") or {},
            "position": position,
            "execution": {
                "entry_filled_notional": entry_notional,
                "exit_filled_notional": exit_notional,
                "realized_pnl_before_fees": realized_pnl,
                "realized_return_before_fees": (
                    realized_pnl / entry_notional
                    if realized_pnl is not None and entry_notional > 0.0
                    else None
                ),
            },
        }
        ticks_summary = self.directory(day) / "ticks_summary.json"
        if ticks_summary.exists():
            summary["market_data"] = json.loads(ticks_summary.read_text())
        if error is not None:
            summary["error"] = error
        _atomic_write_json(self.directory(day) / "summary.json", summary)


class AlpacaAPIError(RuntimeError):
    """An Alpaca REST request failed without exposing request credentials."""

    def __init__(self, method: str, url: str, status_code: int, message: str):
        super().__init__(
            f"Alpaca {method} {url} returned {status_code}: {message[:500]}"
        )
        self.status_code = int(status_code)


def parse_clock(value: str) -> time:
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "time must use 24-hour HH:MM format"
        ) from error
    return parsed.replace(second=0, microsecond=0)


def _combine(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=EASTERN)


def _iso_now(now: datetime | None = None) -> str:
    current = now or datetime.now(tz=EASTERN)
    return current.astimezone(ZoneInfo("UTC")).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _bar_session_date(value: str) -> date:
    return _parse_timestamp(value).astimezone(EASTERN).date()


def _float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric Alpaca field {field}={value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite Alpaca field {field}={value!r}")
    return result


def _chunks(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), int(size)):
        yield list(values[start : start + int(size)])


def _normalize_api_base(value: str, version: str = "v2") -> str:
    base = value.rstrip("/")
    if not base.rsplit("/", 1)[-1].startswith("v"):
        base = f"{base}/{version}"
    return base


def load_credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "set ALPACA_KEY and ALPACA_SECRET (or APCA_API_KEY_ID and APCA_API_SECRET_KEY)"
        )
    return key, secret


def load_data_credentials(trading_key: str, trading_secret: str) -> tuple[str, str]:
    """Load an optional market-data subscription, falling back to trading auth."""
    key = os.environ.get("ALPACA_DATA_KEY")
    secret = os.environ.get("ALPACA_DATA_SECRET")
    if not key and not secret:
        return trading_key, trading_secret
    if not key or not secret:
        raise RuntimeError(
            "set both ALPACA_DATA_KEY and ALPACA_DATA_SECRET, or neither"
        )
    return key, secret


def _completed_session_end(day: date) -> datetime:
    """Return an explicit timestamp safely beyond a completed session's daily bar."""
    return datetime.combine(day, time(23, 59, 59), tzinfo=EASTERN)


class AlpacaClient:
    """Small REST client with bounded retries and thread-local sessions."""

    def __init__(
        self,
        key: str,
        secret: str,
        trading_url: str = PAPER_TRADING_URL,
        data_url: str = DEFAULT_DATA_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        data_key: str | None = None,
        data_secret: str | None = None,
    ):
        if (data_key is None) != (data_secret is None):
            raise ValueError("data_key and data_secret must be provided together")
        self.trading_url = _normalize_api_base(trading_url)
        self.data_url = _normalize_api_base(data_url)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "trading-rl-overnight-liquidity/1",
        }
        self._data_headers = {
            **self._headers,
            "APCA-API-KEY-ID": data_key or key,
            "APCA-API-SECRET-KEY": data_secret or secret,
        }
        self._local = threading.local()

    def _session(self, *, data_credentials: bool = False) -> requests.Session:
        attribute = "data_session" if data_credentials else "trading_session"
        session = getattr(self._local, attribute, None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                self._data_headers if data_credentials else self._headers
            )
            setattr(self._local, attribute, session)
        return session

    def _request(
        self,
        method: str,
        base: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        payload: Mapping[str, object] | None = None,
        allow_not_found: bool = False,
        data_credentials: bool = False,
    ) -> Any:
        url = f"{base}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session(data_credentials=data_credentials).request(
                    method,
                    url,
                    params=params,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time_module.sleep(min(2**attempt, 8))
                continue
            if response.status_code == 404 and allow_not_found:
                return None
            if 200 <= response.status_code < 300:
                if not response.content:
                    return None
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else min(2**attempt, 8)
                    )
                    time_module.sleep(delay)
                    continue
            try:
                body = response.json()
                message = (
                    str(body.get("message", body))
                    if isinstance(body, dict)
                    else str(body)
                )
            except (ValueError, TypeError):
                message = response.text
            raise AlpacaAPIError(method, url, response.status_code, message)
        raise AssertionError("unreachable retry loop")

    def list_assets(self) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            self.trading_url,
            "assets",
            params={"status": "active", "asset_class": "us_equity"},
        )
        return list(result)

    def calendar(self, start: date, end: date) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            self.trading_url,
            "calendar",
            params={"start": start.isoformat(), "end": end.isoformat()},
        )
        return list(result)

    def clock(self) -> dict[str, Any]:
        return dict(self._request("GET", self.trading_url, "clock"))

    def account(self) -> dict[str, Any]:
        return dict(self._request("GET", self.trading_url, "account"))

    def positions(self) -> list[dict[str, Any]]:
        return list(self._request("GET", self.trading_url, "positions"))

    def position(self, symbol: str) -> dict[str, Any] | None:
        result = self._request(
            "GET",
            self.trading_url,
            f"positions/{quote(symbol, safe='')}",
            allow_not_found=True,
        )
        return dict(result) if result is not None else None

    def list_orders(self, status: str = "open") -> list[dict[str, Any]]:
        return list(
            self._request(
                "GET",
                self.trading_url,
                "orders",
                params={"status": status, "limit": 500, "direction": "desc"},
            )
        )

    def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        result = self._request(
            "GET",
            self.trading_url,
            "orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            allow_not_found=True,
        )
        return dict(result) if result is not None else None

    def order(self, order_id: str) -> dict[str, Any]:
        return dict(
            self._request("GET", self.trading_url, f"orders/{quote(order_id, safe='')}")
        )

    def submit_order(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return dict(self._request("POST", self.trading_url, "orders", payload=payload))

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", self.trading_url, f"orders/{quote(order_id, safe='')}")

    def historical_daily_bars(
        self,
        symbols: Sequence[str],
        start: date | datetime,
        end: date | datetime,
        feed: str,
        adjustment: str = "raw",
    ) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
        page_token: str | None = None
        while True:
            params: dict[str, object] = {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "feed": feed,
                "adjustment": adjustment,
                "limit": 10_000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            response = dict(
                self._request(
                    "GET",
                    self.data_url,
                    "stocks/bars",
                    params=params,
                    data_credentials=True,
                )
            )
            for symbol, bars in dict(response.get("bars") or {}).items():
                output.setdefault(str(symbol), []).extend(list(bars))
            page_token = response.get("next_page_token")
            if not page_token:
                break
        return output

    def latest_quotes(
        self, symbols: Sequence[str], feed: str
    ) -> dict[str, dict[str, Any]]:
        """Return the latest NBBO quote used to estimate whole-share buy quantities."""
        if not symbols:
            return {}
        response = dict(
            self._request(
                "GET",
                self.data_url,
                "stocks/quotes/latest",
                params={"symbols": ",".join(symbols), "feed": feed},
                data_credentials=True,
            )
        )
        return {
            str(symbol): dict(value)
            for symbol, value in dict(response.get("quotes") or {}).items()
        }


@dataclass(frozen=True)
class StrategyConfig:
    top: int
    ema_span: int
    min_history_days: int
    minimum_trading_days: int
    lookback_calendar_days: int
    daily_bars_dir: Path
    liquidity_shortlist: Path
    shortlist_since: date
    shortlist_daily_top: int
    shortlist_lookback_sessions: int | None
    liquidity_scheme: str
    daily_overlap_days: int
    feed: str
    quote_feed: str
    exchanges: frozenset[str]
    data_batch_size: int
    data_workers: int
    order_submit_workers: int
    capital: float | None
    capital_fraction: float
    cash_buffer_fraction: float
    fill_timeout_seconds: float
    poll_seconds: float
    entry_preflight_seconds: float
    share_mode: str
    quote_max_age_seconds: float


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "strategy": "overnight_liquidity_long"}
        state = json.loads(self.path.read_text())
        if state.get("version") != 1:
            raise ValueError(f"unsupported state version in {self.path}")
        return state

    def save(self, state: Mapping[str, object]) -> None:
        _atomic_write_json(self.path, state)


def eligible_assets(
    assets: Iterable[Mapping[str, object]],
    exchanges: frozenset[str],
    security_master: Mapping[str, Mapping[str, object]] | None = None,
    *,
    require_fractionable: bool = True,
) -> list[str]:
    symbols = {
        str(asset["symbol"])
        for asset in assets
        if asset.get("status") == "active"
        and bool(asset.get("tradable"))
        and (not require_fractionable or bool(asset.get("fractionable")))
        and asset.get("class") == "us_equity"
        and str(asset.get("exchange", "")).upper() in exchanges
    }
    if security_master is not None:
        symbols = {
            symbol
            for symbol in symbols
            if (record := security_master.get(_security_symbol(symbol))) is not None
            and is_company_security(dict(record))[0]
        }
    return sorted(symbols)


def _daily_bars_to_array(
    bars: Sequence[Mapping[str, object]], symbol: str
) -> np.ndarray:
    """Encode split-adjusted Alpaca daily bars in the shared int64 cache schema.

    An absent or non-positive VWAP is stored as ``0``, the schema's sentinel for
    "not reported", exactly as ``scripts/download_bars.py`` writes it. Substituting
    the close here instead would make the two writers disagree byte-for-byte on
    every zero-volume session, and the overlap comparison in ``_merge_daily_arrays``
    would then escalate those symbols to a full retained-history re-download on
    every rank, forever. Readers already resolve the sentinel: both
    ``dollar_volume_shortlist`` and ``completed_liquidity_ranking`` fall back to the
    close when VWAP is not positive.
    """
    rows: list[list[int]] = []
    for bar in bars:
        timestamp = bar.get("t")
        if not timestamp:
            continue
        parsed = _parse_timestamp(str(timestamp)).astimezone(UTC)
        seconds = int((parsed - DAILY_BAR_ANNO).total_seconds())
        close = _float(bar.get("c"), f"{symbol}.c")
        vwap_value = bar.get("vw")
        vwap = 0.0 if vwap_value is None else _float(vwap_value, f"{symbol}.vw")
        if vwap < 0.0:
            raise ValueError(f"negative daily VWAP for {symbol} at {timestamp}")
        values = [
            seconds,
            int(np.rint(_float(bar.get("o"), f"{symbol}.o") * 1000.0)),
            int(np.rint(_float(bar.get("h"), f"{symbol}.h") * 1000.0)),
            int(np.rint(_float(bar.get("l"), f"{symbol}.l") * 1000.0)),
            int(np.rint(close * 1000.0)),
            int(np.rint(_float(bar.get("v"), f"{symbol}.v"))),
            int(np.rint(_float(bar.get("n") or 0, f"{symbol}.n"))),
            int(np.rint(vwap * 1000.0)),
        ]
        if any(value < 0 for value in values):
            raise ValueError(f"negative daily-bar value for {symbol} at {timestamp}")
        rows.append(values)
    if not rows:
        return np.empty((0, DAILY_BAR_COLUMNS), dtype=np.int64)
    array = np.asarray(rows, dtype=np.int64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    if not (array[:-1, 0] < array[1:, 0]).all():
        raise ValueError(f"duplicate or unsorted daily timestamps for {symbol}")
    return array


def _merge_daily_arrays(base: np.ndarray, update: np.ndarray) -> np.ndarray | None:
    """Merge an identical overlap, returning ``None`` when history was revised."""
    for label, array in (("base", base), ("update", update)):
        if array.ndim != 2 or array.shape[1] != DAILY_BAR_COLUMNS or len(array) == 0:
            raise ValueError(f"invalid {label} daily-bar shape: {array.shape}")
        if not (array[:-1, 0] < array[1:, 0]).all():
            raise ValueError(f"{label} daily timestamps are not strictly increasing")
    first = int(update[0, 0])
    last = int(base[-1, 0])
    if first > last:
        return None
    overlap_end = min(last, int(update[-1, 0]))
    base_start = int(np.searchsorted(base[:, 0], first, side="left"))
    base_end = int(np.searchsorted(base[:, 0], overlap_end, side="right"))
    update_end = int(np.searchsorted(update[:, 0], overlap_end, side="right"))
    if not np.array_equal(base[base_start:base_end], update[:update_end]):
        return None
    if int(update[-1, 0]) <= last:
        return base
    return np.vstack((base[:base_start], update))


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[1] != DAILY_BAR_COLUMNS or len(array) == 0:
        raise ValueError(f"invalid daily-bar array for {path}: {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as temporary:
        np.save(temporary, array)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _array_session_date(seconds: int) -> date:
    return (DAILY_BAR_ANNO + timedelta(seconds=int(seconds))).astimezone(EASTERN).date()


def seed_missing_daily_cache(
    client: AlpacaClient,
    bars_dir: Path,
    symbols: Sequence[str],
    start: date,
    end: datetime,
    feed: str,
    batch_size: int,
    workers: int,
) -> dict[str, int]:
    """Add newly eligible companies to the broad cache from the shortlist epoch."""
    missing = sorted(
        symbol for symbol in symbols if not (bars_dir / f"{symbol}.npy").exists()
    )
    if not missing:
        return {"new_symbols_requested": 0, "new_symbols_added": 0}
    LOGGER.info(
        "downloading retained daily history for %d newly eligible companies",
        len(missing),
    )
    downloaded = _download_bars(
        client,
        missing,
        start,
        end,
        feed,
        batch_size,
        workers,
        adjustment="split",
    )
    added = 0
    for symbol in missing:
        array = _daily_bars_to_array(downloaded.get(symbol) or [], symbol)
        if len(array) == 0:
            LOGGER.warning(
                "no retained daily bars for newly eligible symbol %s", symbol
            )
            continue
        _atomic_save_array(bars_dir / f"{symbol}.npy", array)
        added += 1
    return {"new_symbols_requested": len(missing), "new_symbols_added": added}


def refresh_daily_cache(
    client: AlpacaClient,
    bars_dir: Path,
    symbols: Sequence[str],
    expected_session: date,
    end: datetime,
    feed: str,
    batch_size: int,
    workers: int,
    overlap_days: int,
) -> dict[str, int]:
    """Batch-refresh the broad split-adjusted cache and fail closed if it is stale."""
    if not symbols:
        raise RuntimeError(f"daily cache contains no symbols: {bars_dir}")
    start = expected_session - timedelta(days=overlap_days)
    downloaded = _download_bars(
        client,
        symbols,
        start,
        end,
        feed,
        batch_size,
        workers,
        adjustment="split",
    )
    downloaded_dates = {
        _bar_session_date(str(bar["t"]))
        for rows in downloaded.values()
        for bar in rows
        if bar.get("t")
    }
    if expected_session not in downloaded_dates:
        latest = max(downloaded_dates).isoformat() if downloaded_dates else "none"
        raise RuntimeError(
            f"daily SIP refresh is stale: expected completed session {expected_session}, "
            f"latest response session is {latest}"
        )
    if any(day > expected_session for day in downloaded_dates):
        raise RuntimeError(
            "daily SIP refresh unexpectedly included an incomplete future session"
        )

    revised: list[str] = []
    pending: dict[Path, np.ndarray] = {}
    for symbol in symbols:
        rows = downloaded.get(symbol) or []
        if not rows:
            continue
        cache_path = bars_dir / f"{symbol}.npy"
        if not cache_path.exists():
            raise RuntimeError(
                f"broad daily cache member disappeared during refresh: {cache_path}"
            )
        base = np.load(cache_path, allow_pickle=False)
        update = _daily_bars_to_array(rows, symbol)
        merged = _merge_daily_arrays(base, update)
        if merged is None:
            revised.append(symbol)
        else:
            pending[cache_path] = merged

    if revised:
        manifest_path = bars_dir / "_download_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"cannot repair revised split-adjusted history without {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        full_start = date.fromisoformat(str(manifest["since"])[:10])
        LOGGER.warning(
            "daily history changed for %d symbol(s); fully refreshing: %s",
            len(revised),
            ", ".join(revised),
        )
        replacements = _download_bars(
            client,
            revised,
            full_start,
            end,
            feed,
            batch_size,
            workers,
            adjustment="split",
        )
        replacement_arrays = {
            symbol: _daily_bars_to_array(replacements.get(symbol) or [], symbol)
            for symbol in revised
        }
        retry_symbols = [
            symbol
            for symbol, replacement in replacement_arrays.items()
            if len(replacement) == 0
            or _array_session_date(replacement[-1, 0]) < expected_session
        ]
        if retry_symbols:
            # Alpaca can omit an otherwise available symbol from a large,
            # deeply paginated multi-symbol response. Retry only those symbols
            # individually before treating the ranking data as stale.
            LOGGER.warning(
                "batch full-history response was stale for %d symbol(s); "
                "retrying individually: %s",
                len(retry_symbols),
                ", ".join(retry_symbols),
            )
            retried = _download_bars(
                client,
                retry_symbols,
                full_start,
                end,
                feed,
                1,
                workers,
                adjustment="split",
            )
            replacement_arrays.update(
                {
                    symbol: _daily_bars_to_array(retried.get(symbol) or [], symbol)
                    for symbol in retry_symbols
                }
            )
        for symbol in revised:
            replacement = replacement_arrays[symbol]
            if (
                len(replacement) == 0
                or _array_session_date(replacement[-1, 0]) < expected_session
            ):
                raise RuntimeError(f"full daily-history refresh is stale for {symbol}")
            pending[bars_dir / f"{symbol}.npy"] = replacement

    for cache_path, array in pending.items():
        _atomic_save_array(cache_path, array)
    manifest_path = bars_dir / "_download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        ticker_count = len(list(bars_dir.glob("*.npy")))
        manifest.update(
            {
                "updated_at": _iso_now(),
                "completed_through": expected_session.isoformat(),
                "ticker_count": ticker_count,
                "success_count": ticker_count,
                "failed_count": 0,
                "update_existing": True,
                "overlap_days": overlap_days,
            }
        )
        _atomic_write_json(manifest_path, manifest)
    return {
        "cache_symbols": len(symbols),
        "symbols_with_recent_bars": sum(
            bool(downloaded.get(symbol)) for symbol in symbols
        ),
        "updated_files": len(pending),
        "full_refreshes": len(revised),
    }


def dollar_volume_shortlist(
    bars_dir: Path, since: date, top: int, lookback_sessions: int | None = None
) -> tuple[list[str], int, list[str]]:
    """Return the union of each session's top-N stocks by split-adjusted dollar volume.

    ``since`` is the hard epoch floor. ``lookback_sessions`` additionally keeps only
    the most recent N completed sessions, so the union tracks current liquidity
    instead of ratcheting: without it, one session in the daily top-N in 2022 buys
    permanent candidacy. Pass ``None`` to union every session on or after ``since``.
    The cache holds only completed sessions, so the trailing slice stays causal.
    """
    if top < 1:
        raise ValueError("shortlist daily top must be positive")
    if lookback_sessions is not None and lookback_sessions < 1:
        raise ValueError("shortlist lookback sessions must be positive")
    since_seconds = int(
        (datetime.combine(since, time.min, tzinfo=UTC) - DAILY_BAR_ANNO).total_seconds()
    )
    paths: list[Path] = []
    excluded: list[str] = []
    int32_max = np.iinfo(np.int32).max
    for cache_path in sorted(bars_dir.glob("*.npy")):
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != DAILY_BAR_COLUMNS:
            raise ValueError(f"invalid daily cache file {cache_path}: {array.shape}")
        if any(
            np.max(array[:, index], initial=0) > int32_max for index in (1, 2, 3, 4, 7)
        ):
            excluded.append(cache_path.stem)
            continue
        paths.append(cache_path)
    timestamps: set[int] = set()
    for cache_path in paths:
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        timestamps.update(
            int(value) for value in array[array[:, 0] >= since_seconds, 0]
        )
    if not timestamps:
        raise RuntimeError(f"daily cache has no bars on or after {since}")
    ordered = np.asarray(sorted(timestamps), dtype=np.int64)
    if lookback_sessions is not None:
        ordered = ordered[-int(lookback_sessions) :]
    window_seconds = int(ordered[0])
    index = {int(value): row for row, value in enumerate(ordered)}
    values = np.zeros((len(ordered), len(paths)), dtype=np.float64)
    for column, cache_path in enumerate(paths):
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        rows = array[array[:, 0] >= window_seconds]
        row_indices = np.fromiter(
            (index[int(value)] for value in rows[:, 0]), dtype=np.int64, count=len(rows)
        )
        price_mills = np.where(rows[:, 7] > 0, rows[:, 7], rows[:, 4])
        values[row_indices, column] = (
            rows[:, 5].astype(np.float64) * price_mills.astype(np.float64) / 1000.0
        )
    daily_count = min(top, len(paths))
    partition = len(paths) - daily_count
    daily_top = np.argpartition(values, partition, axis=1)[:, partition:]
    selected = {
        int(column)
        for row, columns in enumerate(daily_top)
        for column in columns
        if values[row, column] > 0.0
    }
    return sorted(paths[column].stem for column in selected), len(ordered), excluded


def _atomic_write_symbols(path: Path, symbols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as temporary:
        temporary.write("".join(f"{symbol}\n" for symbol in symbols))
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def load_cached_daily_bars(
    bars_dir: Path, symbols: Sequence[str], start: date
) -> dict[str, list[dict[str, object]]]:
    start_seconds = int(
        (datetime.combine(start, time.min, tzinfo=UTC) - DAILY_BAR_ANNO).total_seconds()
    )
    output: dict[str, list[dict[str, object]]] = {}
    for symbol in symbols:
        array = np.load(bars_dir / f"{symbol}.npy", mmap_mode="r", allow_pickle=False)
        rows = array[array[:, 0] >= start_seconds]
        output[symbol] = [
            {
                "t": (DAILY_BAR_ANNO + timedelta(seconds=int(row[0]))).isoformat(),
                "c": float(row[4]) / 1000.0,
                "v": int(row[5]),
                "n": int(row[6]),
                "vw": float(row[7]) / 1000.0,
            }
            for row in rows
        ]
    return output


def completed_liquidity_ranking(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, object]]],
    session_dates: Sequence[date],
    trade_date: date,
    ema_span: int,
    min_history_days: int,
    minimum_trading_days: int | None = None,
    scheme: str = "dollar_ema",
) -> list[tuple[str, float, int]]:
    """Rank symbols using bars strictly before ``trade_date``.

    Dollar volume uses daily VWAP times volume, falling back to close times
    volume only when VWAP is unavailable. Missing sessions decay an existing
    EMA exactly like the backtest and do not count toward minimum history.

    ``scheme`` selects the ranking statistic, and both options call straight into
    the simulator's implementation so a live basket and a simulated one cannot
    drift apart.
    """
    if scheme not in ("dollar_ema", "turnover_stability"):
        raise ValueError(f"unsupported live liquidity scheme: {scheme}")
    completed = sorted({day for day in session_dates if day < trade_date})
    if not completed:
        raise ValueError("no completed sessions are available for ranking")
    symbols = sorted(bars_by_symbol)
    date_index = {day: index for index, day in enumerate(completed)}
    values = np.full((len(completed) + 1, len(symbols)), np.nan, dtype=np.float64)
    observations = np.zeros(len(symbols), dtype=np.int32)
    for column, symbol in enumerate(symbols):
        for bar in bars_by_symbol[symbol]:
            timestamp = bar.get("t")
            if not timestamp:
                continue
            bar_date = _bar_session_date(str(timestamp))
            row = date_index.get(bar_date)
            if row is None:
                continue
            volume = _float(bar.get("v"), "bar.v")
            price_value = bar.get("vw")
            price = _float(price_value, "bar.vw") if price_value is not None else 0.0
            if price <= 0.0:
                price = _float(bar.get("c"), "bar.c")
            if volume > 0.0 and price > 0.0:
                values[row, column] = price * volume
                observations[column] += 1
    score_matrix = (
        causal_turnover_stability(values, ema_span, min_history_days)
        if scheme == "turnover_stability"
        else causal_ema_log_liquidity(values, ema_span, min_history_days)
    )
    latest_scores = score_matrix[-1]
    required_sessions = (
        int(min_history_days)
        if minimum_trading_days is None
        else int(minimum_trading_days)
    )
    if required_sessions < 1:
        raise ValueError("minimum_trading_days must be positive")
    ranked = [
        (symbol, float(latest_scores[index]), int(observations[index]))
        for index, symbol in enumerate(symbols)
        if np.isfinite(latest_scores[index])
        and observations[index] >= required_sessions
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _download_bars(
    client: AlpacaClient,
    symbols: Sequence[str],
    start: date | datetime,
    end: date | datetime,
    feed: str,
    batch_size: int,
    workers: int,
    adjustment: str = "raw",
) -> dict[str, list[dict[str, Any]]]:
    batches = list(_chunks(symbols, batch_size))
    output: dict[str, list[dict[str, Any]]] = {}
    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                client.historical_daily_bars,
                batch,
                start,
                end,
                feed,
                adjustment,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            output.update(future.result())
            completed_count += 1
            if (
                completed_count == 1
                or completed_count % 10 == 0
                or completed_count == len(batches)
            ):
                LOGGER.info(
                    "downloaded daily bars for %d/%d symbol batches",
                    completed_count,
                    len(batches),
                )
    return output


def _calendar_dates(calendar: Sequence[Mapping[str, object]]) -> list[date]:
    return sorted(date.fromisoformat(str(session["date"])) for session in calendar)


def _next_session(client: AlpacaClient, current: date) -> date:
    sessions = _calendar_dates(
        client.calendar(current + timedelta(days=1), current + timedelta(days=10))
    )
    if not sessions:
        raise RuntimeError(f"Alpaca calendar has no next session after {current}")
    return sessions[0]


def _market_session_status(
    client: AlpacaClient, current: date
) -> tuple[bool, date | None]:
    """Return whether ``current`` is a session and the following session date."""
    sessions = _calendar_dates(client.calendar(current, current + timedelta(days=10)))
    is_session = current in sessions
    next_session = next((session for session in sessions if session > current), None)
    return is_session, next_session


def rank_for_day(
    client: AlpacaClient,
    store: StateStore,
    config: StrategyConfig,
    trade_date: date,
    now: datetime | None = None,
    artifacts: DailyArtifacts | None = None,
) -> dict[str, Any]:
    with store.locked():
        state = store.load()
        prior = state.get("ranking") or {}
        if (
            prior.get("trade_date") == trade_date.isoformat()
            and prior.get("ranking_pipeline_version") == RANKING_PIPELINE_VERSION
            and prior.get("liquidity_scheme") == config.liquidity_scheme
        ):
            LOGGER.info("ranking for %s already exists; reusing it", trade_date)
            return dict(prior)

        lookback_start = trade_date - timedelta(days=config.lookback_calendar_days)
        calendar = client.calendar(lookback_start, trade_date)
        sessions = _calendar_dates(calendar)
        completed = [day for day in sessions if day < trade_date]
        if trade_date not in sessions:
            raise RuntimeError(f"{trade_date} is not an Alpaca trading session")
        required_sessions = max(config.min_history_days, config.minimum_trading_days)
        if len(completed) < required_sessions:
            raise RuntimeError(
                f"only {len(completed)} completed sessions in bounded lookback; "
                "increase --liquidity-lookback-days"
            )

        bars_dir = config.daily_bars_dir
        manifest_path = bars_dir / "_download_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"daily-bar manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("timeframe") != "1Day" or manifest.get("adjustment") != "split":
            raise RuntimeError(
                "live ranking requires a split-adjusted 1Day cache; "
                f"found timeframe={manifest.get('timeframe')!r}, "
                f"adjustment={manifest.get('adjustment')!r}"
            )
        assets = client.list_assets()
        require_fractionable = config.share_mode == "fractional"
        alpaca_eligible = eligible_assets(
            assets,
            config.exchanges,
            require_fractionable=require_fractionable,
        )
        security_master_path = (
            artifacts.work_dir / "nasdaq_security_master.json"
            if artifacts is not None
            else Path(DEFAULT_SECURITY_MASTER_CACHE)
        )
        security_master = load_nasdaq_security_master(security_master_path)
        eligible_companies = set(
            eligible_assets(
                assets,
                config.exchanges,
                security_master,
                require_fractionable=require_fractionable,
            )
        )

        bars_end = _completed_session_end(completed[-1])
        cache_seed = seed_missing_daily_cache(
            client,
            bars_dir,
            sorted(eligible_companies),
            config.shortlist_since,
            bars_end,
            config.feed,
            config.data_batch_size,
            config.data_workers,
        )
        cache_symbols = sorted(cache_path.stem for cache_path in bars_dir.glob("*.npy"))
        if not cache_symbols:
            raise RuntimeError(f"daily cache contains no .npy files: {bars_dir}")

        LOGGER.info(
            "refreshing %d split-adjusted daily-cache symbols through %s in batches",
            len(cache_symbols),
            completed[-1],
        )
        cache_refresh = refresh_daily_cache(
            client,
            bars_dir,
            cache_symbols,
            completed[-1],
            bars_end,
            config.feed,
            config.data_batch_size,
            config.data_workers,
            config.daily_overlap_days,
        )
        shortlist, shortlist_sessions, incompatible_symbols = dollar_volume_shortlist(
            bars_dir,
            config.shortlist_since,
            config.shortlist_daily_top,
            config.shortlist_lookback_sessions,
        )
        _atomic_write_symbols(config.liquidity_shortlist, shortlist)

        symbols = sorted(set(shortlist) & eligible_companies)
        candidate_diagnostics = {
            **cache_seed,
            **cache_refresh,
            "shortlist_daily_top": config.shortlist_daily_top,
            "shortlist_lookback_sessions": config.shortlist_lookback_sessions,
            "shortlist_trading_days": shortlist_sessions,
            "shortlist_symbols": len(shortlist),
            "eligible_company_assets": len(eligible_companies),
            "filtered_candidate_symbols": len(symbols),
            "inactive_or_ineligible_shortlist_symbols": len(shortlist) - len(symbols),
            "int32_incompatible_symbols": len(incompatible_symbols),
        }
        if len(symbols) < config.top:
            raise RuntimeError(
                f"only {len(symbols)} eligible companies remain in the dollar-volume shortlist "
                f"for top={config.top}"
            )
        LOGGER.info(
            "ranking %d eligible companies from the %d-symbol historical dollar-volume "
            "shortlist, using %s through completed session %s",
            len(symbols),
            len(shortlist),
            lookback_start,
            completed[-1],
        )
        bars = load_cached_daily_bars(bars_dir, symbols, lookback_start)
        ticks_metadata = None
        if artifacts is not None:
            ticks_metadata = artifacts.write_ticks(
                trade_date,
                bars,
                feed=config.feed,
                start=lookback_start,
                end=bars_end,
            )
        ranking = completed_liquidity_ranking(
            bars,
            completed,
            trade_date,
            config.ema_span,
            config.min_history_days,
            config.minimum_trading_days,
            config.liquidity_scheme,
        )
        reserve_count = min(len(ranking), max(config.top * 3, config.top + 20))
        if reserve_count < config.top:
            raise RuntimeError(
                f"only {len(ranking)} stocks have {config.minimum_trading_days} completed bars"
            )
        result = {
            "trade_date": trade_date.isoformat(),
            "created_at": _iso_now(now),
            "completed_through": completed[-1].isoformat(),
            "lookback_start": lookback_start.isoformat(),
            "lookback_calendar_days": config.lookback_calendar_days,
            "ema_span_sessions": config.ema_span,
            "minimum_history_sessions": config.min_history_days,
            "minimum_completed_trading_days": config.minimum_trading_days,
            "feed": config.feed,
            "ranking_pipeline_version": RANKING_PIPELINE_VERSION,
            "candidate_method": (
                "most_liquid.txt daily top-N dollar-volume union over a trailing "
                "session window, then strictly lagged causal EMA(log1p(dollar volume))"
            ),
            "ranking_price": "split-adjusted daily VWAP, falling back to close",
            "daily_bars_dir": str(bars_dir),
            "liquidity_shortlist": str(config.liquidity_shortlist),
            "shortlist_since": config.shortlist_since.isoformat(),
            "shortlist_lookback_sessions": config.shortlist_lookback_sessions,
            "liquidity_scheme": config.liquidity_scheme,
            "candidate_diagnostics": candidate_diagnostics,
            "market_data_dump": ticks_metadata,
            "screened_candidate_symbols": symbols,
            "alpaca_eligible_asset_count": len(alpaca_eligible),
            "eligible_asset_count": len(symbols),
            "ranked_asset_count": len(ranking),
            # The issuer is resolved here, while the security master is already loaded,
            # and persisted with the ranking so the entry decision stays auditable and
            # survives a restart without a second lookup.
            "candidates": [
                {
                    "rank": index + 1,
                    "symbol": symbol,
                    "score": score,
                    "observations": observations,
                    "issuer": issuer_key(symbol, security_master),
                }
                for index, (symbol, score, observations) in enumerate(
                    ranking[:reserve_count]
                )
            ],
        }
        state["ranking"] = result
        state["updated_at"] = _iso_now(now)
        store.save(state)
        LOGGER.info(
            "ranking complete for %s: %d eligible ranked stocks; market data=%s",
            trade_date,
            len(ranking),
            ticks_metadata["path"] if ticks_metadata else "not persisted",
        )
        return result


def _print_ranking(ranking: Mapping[str, object], top: int) -> None:
    table = Table(title=f"Liquidity ranking for {ranking['trade_date']}")
    table.add_column("Rank", justify="right")
    table.add_column("Symbol")
    table.add_column("EMA log $ volume", justify="right")
    table.add_column("Sessions", justify="right")
    for candidate in list(ranking["candidates"])[:top]:
        table.add_row(
            str(candidate["rank"]),
            str(candidate["symbol"]),
            f"{float(candidate['score']):.4f}",
            str(candidate["observations"]),
        )
    CONSOLE.print(table)
    CONSOLE.print(
        f"Completed through {ranking['completed_through']}; "
        f"bounded lookback starts {ranking['lookback_start']}; feed={ranking['feed']}; "
        f"minimum history={ranking['minimum_completed_trading_days']} sessions"
    )
    CONSOLE.print(
        f"Dollar-volume shortlist: {ranking['candidate_diagnostics']['shortlist_symbols']} "
        f"historical members; {ranking['eligible_asset_count']} currently eligible companies; "
        f"{ranking['candidate_diagnostics']['cache_symbols']} broad-cache symbols refreshed"
    )


def _client_order_id(entry_date: date, side: str, symbol: str, attempt: int = 1) -> str:
    clean_symbol = "".join(
        character for character in symbol.upper() if character.isalnum()
    )
    marker = "e" if side == "buy" else "x"
    suffix = "" if attempt == 1 else f"-{attempt}"
    return f"olq-{entry_date:%Y%m%d}-{marker}-{clean_symbol}{suffix}"[:128]


def available_budget(account: Mapping[str, object], config: StrategyConfig) -> float:
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


def whole_share_order_plan(
    symbols: Sequence[str],
    quotes: Mapping[str, Mapping[str, object]],
    budget: float,
    reference: datetime,
    max_age_seconds: float,
) -> dict[str, object]:
    """Floor equal-notional targets using fresh asks, matching the simulator.

    Unused allocation is deliberately left as cash. A missing or stale quote aborts
    the complete plan so a live entry never guesses a quantity from old market data.
    """
    if not symbols:
        raise ValueError("whole-share sizing requires at least one symbol")
    prices: list[float] = []
    quote_times: dict[str, str] = {}
    normalized_reference = reference.astimezone(ZoneInfo("UTC"))
    for symbol in symbols:
        quote_row = quotes.get(symbol)
        if quote_row is None:
            raise RuntimeError(f"latest quote missing for {symbol}")
        ask = _float(quote_row.get("ap"), f"quote[{symbol}].ap")
        if ask <= 0.0:
            raise RuntimeError(f"latest ask is not positive for {symbol}")
        raw_timestamp = quote_row.get("t")
        if not raw_timestamp:
            raise RuntimeError(f"latest quote timestamp missing for {symbol}")
        quoted_at = _parse_timestamp(str(raw_timestamp))
        if quoted_at.tzinfo is None:
            raise RuntimeError(f"latest quote timestamp has no timezone for {symbol}")
        age_seconds = (normalized_reference - quoted_at).total_seconds()
        if age_seconds < -5.0:
            raise RuntimeError(f"latest quote timestamp is in the future for {symbol}")
        if age_seconds > max_age_seconds:
            raise RuntimeError(
                f"latest quote for {symbol} is {age_seconds:.1f}s old; "
                f"maximum is {max_age_seconds:.1f}s"
            )
        prices.append(ask)
        quote_times[symbol] = quoted_at.isoformat()

    quantities = basket_quantities(np.asarray(prices), budget, "whole")
    targets = {
        symbol: int(quantity)
        for symbol, quantity in zip(symbols, quantities, strict=True)
    }
    skipped = [symbol for symbol in symbols if targets[symbol] == 0]
    estimated_deployed = sum(
        targets[symbol] * price for symbol, price in zip(symbols, prices, strict=True)
    )
    if not any(targets.values()):
        raise RuntimeError("no selected stock is affordable as a whole share")
    return {
        "sizing_prices": dict(zip(symbols, prices, strict=True)),
        "sizing_quote_times": quote_times,
        "target_quantities": targets,
        "skipped_symbols": skipped,
        "estimated_deployed_notional": math.floor(estimated_deployed * 100.0) / 100.0,
    }


def _order_summary(order: Mapping[str, object]) -> dict[str, object]:
    return {
        key: order.get(key)
        for key in (
            "id",
            "client_order_id",
            "symbol",
            "side",
            "status",
            "qty",
            "notional",
            "filled_qty",
            "filled_avg_price",
            "time_in_force",
            "submitted_at",
            "filled_at",
        )
    }


def _submit_entry_order(
    client: AlpacaClient,
    symbol: str,
    payload: Mapping[str, object],
    *,
    recover_existing: bool,
) -> tuple[str, dict[str, Any], dict[str, object]]:
    """Submit one entry order, recovering deterministic IDs after uncertain failures."""
    started_at = datetime.now(tz=EASTERN)
    started_monotonic = time_module.monotonic()
    client_order_id = str(payload["client_order_id"])
    order = client.order_by_client_id(client_order_id) if recover_existing else None
    recovered = order is not None
    if order is None:
        try:
            order = client.submit_order(payload)
        except Exception:
            # A timed-out POST can still have reached Alpaca. Query its deterministic
            # ID before allowing the daemon to retry, preventing a duplicate order.
            order = client.order_by_client_id(client_order_id)
            if order is None:
                raise
            recovered = True
    completed_at = datetime.now(tz=EASTERN)
    telemetry: dict[str, object] = {
        "dispatch_started_at": _iso_now(started_at),
        "dispatch_completed_at": _iso_now(completed_at),
        "dispatch_duration_ms": (time_module.monotonic() - started_monotonic) * 1000.0,
        "recovered_by_client_order_id": recovered,
    }
    return symbol, dict(order), telemetry


def _exit_order_time_in_force(
    position: Mapping[str, object], now: datetime | None = None
) -> str:
    """Use the opening auction for whole shares and DAY only as post-open recovery."""
    if str(position.get("share_mode") or "fractional") != "whole":
        return "day"
    current = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)
    current_time = current.time().replace(tzinfo=None)
    if current_time < OPENING_AUCTION_CUTOFF:
        return "opg"
    if current_time < REGULAR_MARKET_OPEN:
        raise RuntimeError(
            "the 09:28 ET OPG cutoff has passed; whole-share exit will retry at market open"
        )
    return "day"


def _capture_flat_exit_account_snapshot(
    client: AlpacaClient,
    position: dict[str, Any],
    now: datetime | None = None,
) -> bool:
    """Persist authoritative net equity after the account is completely flat.

    Fill arithmetic does not include every brokerage fee or cent-level settlement
    adjustment. Account equity is therefore the final source of truth, but only when
    no unrelated position can contaminate this strategy basket's result.
    """
    try:
        if client.positions():
            LOGGER.info(
                "not recording exit account equity because the account still holds positions"
            )
            return False
        account = client.account()
    except Exception as error:  # noqa: BLE001 - a snapshot cannot hold up a completed exit
        LOGGER.warning("could not record the flat exit account snapshot: %s", error)
        return False

    position["exit_account_snapshot"] = {
        key: account.get(key)
        for key in (
            "cash",
            "equity",
            "buying_power",
            "regt_buying_power",
            "non_marginable_buying_power",
            "long_market_value",
            "short_market_value",
            "initial_margin",
            "maintenance_margin",
            "multiplier",
        )
    }
    position["exit_account_snapshot_at"] = _iso_now(now)
    return True


def _wait_for_orders(
    client: AlpacaClient,
    orders: Mapping[str, Mapping[str, object]],
    timeout_seconds: float,
    poll_seconds: float,
    *,
    cancel_on_timeout: bool = True,
) -> dict[str, dict[str, Any]]:
    """Poll orders for one reconciliation window.

    Entry orders are canceled when their fill window expires so a late entry cannot
    create an unmanaged overnight position. Exit orders are different: canceling a
    partially filled market sell and replacing it every 45 seconds creates avoidable
    churn. Callers closing positions therefore leave active orders working and resume
    monitoring the same Alpaca order on the next daemon pass.
    """
    latest = {symbol: dict(order) for symbol, order in orders.items()}
    deadline = time_module.monotonic() + timeout_seconds
    while True:
        active = []
        for symbol, order in latest.items():
            status = str(order.get("status", ""))
            if status not in TERMINAL_ORDER_STATUSES:
                active.append((symbol, str(order["id"])))
        if not active or time_module.monotonic() >= deadline:
            break
        time_module.sleep(
            min(poll_seconds, max(0.0, deadline - time_module.monotonic()))
        )
        for symbol, order_id in active:
            latest[symbol] = client.order(order_id)
    for symbol, order in list(latest.items()):
        if (
            cancel_on_timeout
            and str(order.get("status", "")) not in TERMINAL_ORDER_STATUSES
        ):
            try:
                client.cancel_order(str(order["id"]))
            except AlpacaAPIError as error:
                LOGGER.warning("could not cancel timed-out %s order: %s", symbol, error)
            latest[symbol] = client.order(str(order["id"]))
    return latest


def _select_unconflicted_candidates(
    ranking: Mapping[str, object],
    held_symbols: set[str],
    open_order_symbols: set[str],
    top: int,
    dedupe_share_classes: bool = True,
) -> list[str]:
    """Take the top ranked candidates, skipping conflicts and duplicate share classes.

    Alphabet lists as both GOOGL and GOOG, and the liquidity ranking scores them
    separately, so a basket can hold one company at double weight while reporting the
    nominal size. Candidates are walked in rank order, so the first class encountered is
    the more liquid one and the basket backfills from further down the reserve.

    A ranking written before issuers were recorded has no ``issuer`` field; such a
    candidate falls back to its own symbol, which disables deduping rather than failing.
    """
    selected: list[str] = []
    seen_issuers: set[str] = set()
    skipped_duplicates: list[str] = []
    for candidate in ranking["candidates"]:
        symbol = str(candidate["symbol"])
        if symbol in held_symbols or symbol in open_order_symbols:
            continue
        if dedupe_share_classes:
            issuer = str(candidate.get("issuer") or symbol)
            if issuer in seen_issuers:
                skipped_duplicates.append(symbol)
                continue
            seen_issuers.add(issuer)
        selected.append(symbol)
        if len(selected) == top:
            break
    if len(selected) != top:
        raise RuntimeError(
            f"only {len(selected)} ranked candidates remain after excluding existing positions/orders"
        )
    if skipped_duplicates:
        LOGGER.info(
            "skipped %d duplicate share class(es) so the basket holds distinct issuers: %s",
            len(skipped_duplicates),
            ", ".join(skipped_duplicates),
        )
    return selected


def enter_for_day(
    client: AlpacaClient,
    store: StateStore,
    config: StrategyConfig,
    trade_date: date,
    submit: bool,
    now: datetime | None = None,
    *,
    preflight_only: bool = False,
    dispatch_target: datetime | None = None,
    preflight_market_validated: bool = False,
) -> dict[str, Any]:
    if preflight_only and not submit:
        raise ValueError("preflight_only requires an authorized submit workflow")
    with store.locked():
        state = store.load()
        ranking = state.get("ranking") or {}
        previous = state.get("position") or {}
        resuming = (
            previous.get("entry_date") == trade_date.isoformat()
            and previous.get("status") != "closed"
        )
        if (
            ranking.get("trade_date") != trade_date.isoformat()
            or ranking.get("ranking_pipeline_version") != RANKING_PIPELINE_VERSION
        ):
            raise RuntimeError(
                f"no current ranking for {trade_date}; run the rank action first"
            )
        if resuming:
            if preflight_only:
                return dict(previous)
            if not submit and previous.get("status") == "planned":
                return dict(previous)
            LOGGER.info("entry workflow for %s already exists; resuming it", trade_date)
            selected = list(previous["symbols"])
            budget = float(previous["budget"])
            per_symbol = float(previous["per_symbol_notional"])
            position = dict(previous)
            share_mode = str(position.get("share_mode") or "fractional")
            if share_mode != config.share_mode:
                LOGGER.warning(
                    "resuming %s entry with persisted share mode %s instead of configured %s",
                    trade_date,
                    share_mode,
                    config.share_mode,
                )
        else:
            if previous and previous.get("status") not in (None, "closed"):
                raise RuntimeError(
                    f"strategy position from {previous.get('entry_date')} is still {previous.get('status')}"
                )
            preflight_started_at = now or datetime.now(tz=EASTERN)
            held_symbols = {str(item["symbol"]) for item in client.positions()}
            open_order_symbols = {
                str(item["symbol"]) for item in client.list_orders("open")
            }
            selected = _select_unconflicted_candidates(
                ranking, held_symbols, open_order_symbols, config.top
            )
            account = client.account()
            budget = available_budget(account, config)
            per_symbol = math.floor((budget / config.top) * 100.0) / 100.0
            share_mode = config.share_mode
            sizing: dict[str, object] = {}
            if share_mode == "whole":
                sizing = whole_share_order_plan(
                    selected,
                    client.latest_quotes(selected, config.quote_feed),
                    budget,
                    now or datetime.now(tz=EASTERN),
                    config.quote_max_age_seconds,
                )
            position = {
                "entry_date": trade_date.isoformat(),
                "exit_date": _next_session(client, trade_date).isoformat(),
                "status": "planned" if preflight_only or not submit else "entering",
                "share_mode": share_mode,
                "quote_feed": config.quote_feed if share_mode == "whole" else None,
                "symbols": selected,
                "budget": budget,
                "per_symbol_notional": per_symbol,
                **sizing,
                "entry_account_snapshot": {
                    key: account.get(key)
                    for key in (
                        "cash",
                        "equity",
                        "buying_power",
                        "regt_buying_power",
                        "non_marginable_buying_power",
                        "long_market_value",
                        "initial_margin",
                        "maintenance_margin",
                        "multiplier",
                    )
                },
                "entry_account_snapshot_at": _iso_now(now),
                "entry_preflight_started_at": _iso_now(preflight_started_at),
                "entry_preflight_completed_at": _iso_now(),
                "entry_dispatch_target_at": _iso_now(dispatch_target)
                if dispatch_target is not None
                else None,
                "entry_preflight_seconds": config.entry_preflight_seconds,
                "entry_preflight_market_validated": preflight_market_validated,
                "order_submit_workers": config.order_submit_workers,
                "entry_orders": {},
                "exit_orders": {},
                "created_at": _iso_now(now),
            }
            if submit:
                state["position"] = position
                state["updated_at"] = _iso_now(now)
                store.save(state)

        if preflight_only:
            LOGGER.info(
                "entry preflight prepared for %s: symbols=%d, orders=%d, target=%s",
                trade_date,
                len(selected),
                sum(
                    int(value) > 0
                    for value in (position.get("target_quantities") or {}).values()
                )
                if share_mode == "whole"
                else len(selected),
                position.get("entry_dispatch_target_at"),
            )
            return position

        if not submit:
            return position

        if share_mode == "whole":
            raw_targets = position.get("target_quantities")
            if not isinstance(raw_targets, Mapping):
                raise RuntimeError("whole-share entry state has no target quantities")
            target_quantities = {
                symbol: int(raw_targets.get(symbol, 0)) for symbol in selected
            }
            order_symbols = [
                symbol for symbol in selected if target_quantities[symbol] > 0
            ]
        else:
            target_quantities = {}
            order_symbols = selected

        payloads: dict[str, dict[str, object]] = {}
        for symbol in order_symbols:
            client_id = _client_order_id(trade_date, "buy", symbol)
            size = (
                {"qty": str(target_quantities[symbol])}
                if share_mode == "whole"
                else {"notional": f"{per_symbol:.2f}"}
            )
            payloads[symbol] = {
                "symbol": symbol,
                **size,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_id,
            }

        recover_existing = bool(position.get("entry_dispatch_started_at"))
        dispatch_started_at = datetime.now(tz=EASTERN)
        position["status"] = "entering"
        position.setdefault("entry_dispatch_started_at", _iso_now(dispatch_started_at))
        position["entry_dispatch_last_attempt_at"] = _iso_now(dispatch_started_at)
        target_value = position.get("entry_dispatch_target_at")
        if target_value and "entry_dispatch_lateness_ms" not in position:
            target_at = _parse_timestamp(str(target_value)).astimezone(EASTERN)
            position["entry_dispatch_lateness_ms"] = (
                dispatch_started_at - target_at
            ).total_seconds() * 1000.0
        state["position"] = position
        state["updated_at"] = _iso_now(dispatch_started_at)
        store.save(state)

        workers = max(
            1,
            min(
                int(
                    position.get("order_submit_workers") or config.order_submit_workers
                ),
                len(order_symbols),
            ),
        )
        results: dict[str, tuple[dict[str, Any], dict[str, object]]] = {}
        failures: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _submit_entry_order,
                    client,
                    symbol,
                    payloads[symbol],
                    recover_existing=recover_existing,
                ): symbol
                for symbol in order_symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, order, telemetry = future.result()
                except Exception as error:  # noqa: BLE001 - persist partial dispatch
                    failures[symbol] = error
                else:
                    results[symbol] = (order, telemetry)

        submitted: dict[str, dict[str, Any]] = {}
        for symbol in order_symbols:
            result = results.get(symbol)
            if result is None:
                continue
            order, telemetry = result
            submitted[symbol] = order
            previous_summary = position["entry_orders"].get(symbol, {})
            summary = {
                **previous_summary,
                **_order_summary(order),
            }
            for key, value in telemetry.items():
                summary.setdefault(key, value)
            summary["recovered_by_client_order_id"] = bool(
                previous_summary.get("recovered_by_client_order_id")
                or telemetry.get("recovered_by_client_order_id")
            )
            position["entry_orders"][symbol] = summary
        position["entry_dispatch_completed_at"] = _iso_now()
        state["position"] = position
        state["updated_at"] = _iso_now()
        store.save(state)
        if failures:
            failed = ", ".join(sorted(failures))
            first_error = failures[next(iter(failures))]
            raise RuntimeError(
                f"entry submission failed for: {failed}"
            ) from first_error

        final_orders = _wait_for_orders(
            client, submitted, config.fill_timeout_seconds, config.poll_seconds
        )
        filled_symbols = []
        for symbol, order in final_orders.items():
            position["entry_orders"][symbol] = {
                **position["entry_orders"].get(symbol, {}),
                **_order_summary(order),
            }
            if _float(order.get("filled_qty", 0), "order.filled_qty") > 0.0:
                filled_symbols.append(symbol)
        position["filled_symbols"] = filled_symbols
        position["status"] = "open" if filled_symbols else "entry_failed"
        position["entry_completed_at"] = _iso_now()
        state["position"] = position
        state["updated_at"] = _iso_now()
        store.save(state)
        LOGGER.info(
            "entry complete for %s: state=%s, filled=%d/%d",
            trade_date,
            position["status"],
            len(filled_symbols),
            len(order_symbols),
        )
        if not filled_symbols:
            raise RuntimeError("none of the basket entry orders filled")
        if len(filled_symbols) != len(order_symbols):
            LOGGER.warning(
                "only %d/%d submitted basket entries filled",
                len(filled_symbols),
                len(order_symbols),
            )
        return position


def exit_position(
    client: AlpacaClient,
    store: StateStore,
    config: StrategyConfig,
    submit: bool,
    now: datetime | None = None,
    wait_for_fill: bool = True,
) -> dict[str, Any]:
    with store.locked():
        state = store.load()
        position = dict(state.get("position") or {})
        if not position:
            raise RuntimeError("state contains no strategy position to close")
        if position.get("status") == "closed":
            if submit and not position.get("exit_account_snapshot"):
                if _capture_flat_exit_account_snapshot(client, position, now):
                    state["position"] = position
                    state["updated_at"] = _iso_now(now)
                    store.save(state)
            LOGGER.info("strategy position is already closed")
            return position
        entry_date = date.fromisoformat(str(position["entry_date"]))
        # Query every symbol for which this strategy actually submitted an
        # entry. This catches a late fill that arrived after a cancel response,
        # while never broadening the exit to unrelated account positions.
        entry_orders = position.setdefault("entry_orders", {})
        owned_symbols = list(
            dict.fromkeys(
                list(entry_orders.keys()) + list(position.get("filled_symbols") or [])
            )
        )
        for symbol in position.get("symbols") or []:
            if symbol in owned_symbols:
                continue
            recovered = client.order_by_client_id(
                _client_order_id(entry_date, "buy", symbol)
            )
            if recovered is not None:
                entry_orders[symbol] = _order_summary(recovered)
                owned_symbols.append(symbol)
        current_positions: dict[str, dict[str, Any]] = {}
        for symbol in owned_symbols:
            current = client.position(symbol)
            if (
                current is not None
                and _float(current.get("qty", 0), "position.qty") > 0.0
            ):
                current_positions[symbol] = current

        if not submit:
            return {
                **position,
                "status": "exit_plan",
                "exit_quantities": {
                    symbol: str(current["qty"])
                    for symbol, current in current_positions.items()
                },
            }
        position["status"] = "exiting"
        state["position"] = position
        state["updated_at"] = _iso_now(now)
        store.save(state)

        submitted: dict[str, dict[str, Any]] = {}
        attempts = position.setdefault("exit_attempts", {})
        for symbol, current in current_positions.items():
            attempt = max(1, int(attempts.get(symbol, 1)))
            order = None
            while attempt <= 20:
                client_id = _client_order_id(entry_date, "sell", symbol, attempt)
                order = client.order_by_client_id(client_id)
                status = str(order.get("status", "")) if order is not None else ""
                if (
                    order is None
                    or status not in TERMINAL_ORDER_STATUSES
                    or status == "filled"
                ):
                    break
                attempt += 1
            if attempt > 20:
                raise RuntimeError(f"too many exit attempts for {symbol}")
            if order is None:
                time_in_force = _exit_order_time_in_force(position, now)
                order = client.submit_order(
                    {
                        "symbol": symbol,
                        "qty": str(current["qty"]),
                        "side": "sell",
                        "type": "market",
                        "time_in_force": time_in_force,
                        "client_order_id": client_id,
                    }
                )
            attempts[symbol] = attempt
            submitted[symbol] = order
            position.setdefault("exit_orders", {})[symbol] = _order_summary(order)
            state["position"] = position
            store.save(state)

        # A queued order may have filled before the daemon's 09:30
        # reconciliation, leaving no current position to drive the loop above.
        # Refresh those completed orders so durable state records the fill.
        for symbol in set(owned_symbols) - set(current_positions):
            attempt = max(1, int(attempts.get(symbol, 1)))
            order = client.order_by_client_id(
                _client_order_id(entry_date, "sell", symbol, attempt)
            )
            if order is not None:
                position.setdefault("exit_orders", {})[symbol] = _order_summary(order)

        if not wait_for_fill:
            remaining = [
                symbol
                for symbol in owned_symbols
                if client.position(symbol) is not None
            ]
            position["remaining_symbols"] = remaining
            position["status"] = "exit_queued" if remaining else "closed"
            position["exit_queued_at"] = _iso_now(now)
            if not remaining:
                position["exit_completed_at"] = _iso_now(now)
                _capture_flat_exit_account_snapshot(client, position, now)
            state["position"] = position
            state["updated_at"] = _iso_now(now)
            store.save(state)
            LOGGER.info(
                "pre-open exit queued for entry %s: orders=%d, remaining=%d",
                entry_date,
                len(submitted),
                len(remaining),
            )
            return position

        final_orders = _wait_for_orders(
            client,
            submitted,
            config.fill_timeout_seconds,
            config.poll_seconds,
            cancel_on_timeout=False,
        )
        for symbol, order in final_orders.items():
            position["exit_orders"][symbol] = _order_summary(order)
        remaining = [
            symbol for symbol in owned_symbols if client.position(symbol) is not None
        ]
        position["remaining_symbols"] = remaining
        remaining_set = set(remaining)
        active_orders = [
            symbol
            for symbol, order in final_orders.items()
            if symbol in remaining_set
            if str(order.get("status", "")) not in TERMINAL_ORDER_STATUSES
        ]
        position["status"] = (
            "closed"
            if not remaining
            else "exiting"
            if active_orders
            else "exit_incomplete"
        )
        if not remaining:
            completed_at = datetime.now(tz=EASTERN)
            position["exit_completed_at"] = _iso_now(completed_at)
            _capture_flat_exit_account_snapshot(client, position, completed_at)
        state["position"] = position
        state["updated_at"] = _iso_now()
        store.save(state)
        if active_orders:
            LOGGER.info(
                "exit orders still working for entry %s after %.0fs: active=%d, remaining=%d",
                entry_date,
                config.fill_timeout_seconds,
                len(active_orders),
                len(remaining),
            )
        else:
            LOGGER.info(
                "exit complete for entry %s: state=%s, remaining=%d",
                entry_date,
                position["status"],
                len(remaining),
            )
        if remaining and not active_orders:
            raise RuntimeError(
                "positions remain after exit attempt: " + ", ".join(remaining)
            )
        return position


def _print_entry_plan(
    position: Mapping[str, object],
    submit: bool,
    console: Console = CONSOLE,
) -> None:
    if submit and position.get("status") == "planned":
        title = "PREPARED — overnight basket entry"
    else:
        title = (
            "Overnight basket entry" if submit else "DRY RUN — overnight basket entry"
        )
    table = Table(title=title)
    table.add_column("Symbol")
    share_mode = str(position.get("share_mode") or "fractional")
    table.add_column(
        "Quantity" if share_mode == "whole" else "Notional", justify="right"
    )
    if share_mode == "whole":
        table.add_column("Est. value", justify="right")
    table.add_column("Order status")
    orders = position.get("entry_orders") or {}
    quantities = position.get("target_quantities") or {}
    sizing_prices = position.get("sizing_prices") or {}
    symbols = list(position["symbols"])
    per_symbol = float(position["per_symbol_notional"])
    order_count = 0
    calculated_deployed = 0.0
    for symbol in symbols:
        order = orders.get(symbol, {})
        quantity = int(quantities.get(symbol, 0)) if share_mode == "whole" else None
        if share_mode == "whole":
            price = float(sizing_prices.get(symbol, 0.0))
            estimated_value = quantity * price
            calculated_deployed += estimated_value
            if quantity > 0:
                order_count += 1
            row = (
                str(symbol),
                str(quantity),
                f"${estimated_value:,.2f}",
                str(
                    order.get(
                        "status",
                        "skipped — below one share"
                        if quantity == 0
                        else "not submitted",
                    )
                ),
            )
        else:
            order_count += 1
            calculated_deployed += per_symbol
            row = (
                str(symbol),
                f"${per_symbol:,.2f}",
                str(order.get("status", "not submitted")),
            )
        table.add_row(*row)

    estimated_deployed = float(
        position.get("estimated_deployed_notional", calculated_deployed)
    )
    skipped_count = len(symbols) - order_count
    table.add_section()
    if share_mode == "whole":
        table.add_row(
            "TOTAL",
            "—",
            f"${estimated_deployed:,.2f}",
            f"{order_count} orders / {len(symbols)} selected; {skipped_count} skipped",
        )
    else:
        table.add_row(
            "TOTAL",
            f"${estimated_deployed:,.2f}",
            f"{order_count} orders / {len(symbols)} selected",
        )
    console.print(table)
    budget = float(position["budget"])
    deployment_percentage = (
        (estimated_deployed / budget * 100.0) if budget > 0.0 else 0.0
    )
    console.print(
        f"Budget ${budget:,.2f}; estimated basket ${estimated_deployed:,.2f} "
        f"({deployment_percentage:.1f}% of budget); exit session {position['exit_date']}; "
        f"state={position['status']}"
    )


def _print_exit(position: Mapping[str, object], submit: bool) -> None:
    table = Table(
        title="Overnight basket exit" if submit else "DRY RUN — overnight basket exit"
    )
    table.add_column("Symbol")
    table.add_column("Quantity", justify="right")
    table.add_column("Order status")
    quantities = position.get("exit_quantities") or {}
    orders = position.get("exit_orders") or {}
    for symbol in position.get("filled_symbols") or position.get("symbols") or []:
        order = orders.get(symbol, {})
        table.add_row(
            str(symbol),
            str(quantities.get(symbol, order.get("qty", "-"))),
            str(order.get("status", "not submitted")),
        )
    CONSOLE.print(table)
    CONSOLE.print(f"State: {position['status']}")


def _state_status(store: StateStore) -> None:
    with store.locked():
        state = store.load()
    table = Table(title="Live overnight liquidity state", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    ranking = state.get("ranking") or {}
    position = state.get("position") or {}
    table.add_row("State file", str(store.path))
    table.add_row("Ranking date", str(ranking.get("trade_date", "none")))
    table.add_row("Ranked through", str(ranking.get("completed_through", "none")))
    table.add_row("Position entry", str(position.get("entry_date", "none")))
    table.add_row("Position exit", str(position.get("exit_date", "none")))
    table.add_row("Position state", str(position.get("status", "none")))
    table.add_row("Owned symbols", ", ".join(position.get("filled_symbols") or []))
    CONSOLE.print(table)


def _is_paper_endpoint(url: str) -> bool:
    parsed = urlparse(_normalize_api_base(url))
    return parsed.scheme == "https" and parsed.hostname == "paper-api.alpaca.markets"


def _validate_live_clock(client: AlpacaClient, expected_date: date) -> None:
    clock = client.clock()
    market_now = _parse_timestamp(str(clock["timestamp"])).astimezone(EASTERN)
    if market_now.date() != expected_date:
        raise RuntimeError(
            f"Alpaca clock date {market_now.date()} does not match requested date {expected_date}"
        )
    if not bool(clock.get("is_open")):
        raise RuntimeError("Alpaca reports the US equity market is closed")


def _validate_exit_clock(client: AlpacaClient, expected_date: date) -> bool:
    """Validate an exit submission and report whether fills can be awaited now."""
    clock = client.clock()
    market_now = _parse_timestamp(str(clock["timestamp"])).astimezone(EASTERN)
    if market_now.date() != expected_date:
        raise RuntimeError(
            f"Alpaca clock date {market_now.date()} does not match requested date {expected_date}"
        )
    if not EXIT_SUBMISSION_OPEN <= market_now.time().replace(tzinfo=None) < time(16, 0):
        raise RuntimeError(
            f"exit submission must occur between {EXIT_SUBMISSION_OPEN:%H:%M} and 16:00 ET"
        )
    market_open = bool(clock.get("is_open"))
    if market_now.time().replace(tzinfo=None) >= time(9, 30) and not market_open:
        raise RuntimeError("Alpaca reports the US equity market is closed")
    return market_open


def _ranking_is_early_enough(
    ranking: Mapping[str, object], entry_at: datetime, minimum_lead_minutes: int
) -> bool:
    created_at = ranking.get("created_at")
    if not created_at:
        return False
    created = _parse_timestamp(str(created_at)).astimezone(EASTERN)
    return created <= entry_at - timedelta(minutes=minimum_lead_minutes)


def run_daemon(
    client: AlpacaClient,
    store: StateStore,
    config: StrategyConfig,
    ranking_time: time,
    entry_time: time,
    exit_time: time,
    minimum_ranking_lead_minutes: int,
    entry_grace_seconds: int,
    artifacts: DailyArtifacts,
) -> None:
    LOGGER.info(
        "daemon started: rank %s, enter %s, exit %s America/New_York",
        ranking_time.strftime("%H:%M"),
        entry_time.strftime("%H:%M"),
        exit_time.strftime("%H:%M"),
    )
    last_error_key: tuple[str, str] | None = None
    last_attempts: dict[str, datetime] = {}
    calendar_date: date | None = None
    is_trading_session = False
    next_trading_session: date | None = None

    def may_attempt(key: str, current: datetime) -> bool:
        previous = last_attempts.get(key)
        if previous is not None and (current - previous).total_seconds() < 60.0:
            return False
        last_attempts[key] = current
        return True

    while True:
        now = datetime.now(tz=EASTERN)
        today = now.date()
        sleep_seconds = config.poll_seconds
        try:
            if calendar_date != today:
                if not may_attempt("calendar", now):
                    time_module.sleep(sleep_seconds)
                    continue
                is_trading_session, next_trading_session = _market_session_status(
                    client, today
                )
                calendar_date = today
                if not is_trading_session:
                    next_label = (
                        next_trading_session.isoformat()
                        if next_trading_session is not None
                        else "unknown"
                    )
                    LOGGER.info(
                        "%s is not an Alpaca trading session; idling until %s",
                        today,
                        next_label,
                    )
            if not is_trading_session:
                last_error_key = None
                time_module.sleep(sleep_seconds)
                continue

            artifacts.directory(today)
            state = store.load()
            position = state.get("position") or {}
            exit_date_value = position.get("exit_date")
            queued_before_open = position.get(
                "status"
            ) == "exit_queued" and now < _combine(today, time(9, 30))
            if (
                exit_date_value
                and position.get("status") != "closed"
                and not queued_before_open
                and today >= date.fromisoformat(str(exit_date_value))
                and now >= _combine(today, exit_time)
                and now < _combine(today, time(16, 0))
                and may_attempt("exit", now)
            ):
                market_open = _validate_exit_clock(client, today)
                result = exit_position(
                    client,
                    store,
                    config,
                    submit=True,
                    now=now,
                    wait_for_fill=market_open,
                )
                _print_exit(result, True)
                artifacts.write_summary(today, "exit", store, config)
                entry_day = date.fromisoformat(str(result["entry_date"]))
                if entry_day != today:
                    artifacts.write_summary(entry_day, "exit", store, config)
            state = store.load()
            ranking = state.get("ranking") or {}
            rank_start = _combine(today, ranking_time)
            entry_at = _combine(today, entry_time)
            ranking_deadline = entry_at - timedelta(
                minutes=minimum_ranking_lead_minutes
            )
            if (
                rank_start <= now <= ranking_deadline
                and (
                    ranking.get("trade_date") != today.isoformat()
                    or ranking.get("ranking_pipeline_version")
                    != RANKING_PIPELINE_VERSION
                )
                and may_attempt("rank", now)
            ):
                result = rank_for_day(client, store, config, today, artifacts=artifacts)
                _print_ranking(result, config.top)
                artifacts.write_summary(today, "rank", store, config)
            state = store.load()
            ranking = state.get("ranking") or {}
            position = state.get("position") or {}
            entry_deadline = entry_at + timedelta(seconds=entry_grace_seconds)
            same_day_status = (
                position.get("status")
                if position.get("entry_date") == today.isoformat()
                else None
            )
            preflight_at = entry_at - timedelta(seconds=config.entry_preflight_seconds)
            preflight_due = preflight_at <= now < entry_at and same_day_status is None
            ranking_ready = (
                ranking.get("trade_date") == today.isoformat()
                and ranking.get("ranking_pipeline_version") == RANKING_PIPELINE_VERSION
            )
            ranking_early = ranking_ready and _ranking_is_early_enough(
                ranking, entry_at, minimum_ranking_lead_minutes
            )
            if preflight_due and ranking_early and may_attempt("entry-preflight", now):
                _validate_live_clock(client, today)
                prepared = enter_for_day(
                    client,
                    store,
                    config,
                    today,
                    submit=True,
                    now=now,
                    preflight_only=True,
                    dispatch_target=entry_at,
                    preflight_market_validated=True,
                )
                _print_entry_plan(prepared, True)
                artifacts.write_summary(today, "entry-preflight", store, config)

            # Preflight can take long enough to cross the target second. Refresh the
            # clock and state before deciding whether dispatch is due.
            now = datetime.now(tz=EASTERN)
            state = store.load()
            position = state.get("position") or {}
            same_day_status = (
                position.get("status")
                if position.get("entry_date") == today.isoformat()
                else None
            )
            new_entry_due = entry_at <= now <= entry_deadline and same_day_status in (
                None,
                "planned",
            )
            restart_due = same_day_status == "entering" and now < _combine(
                today, time(16, 0)
            )
            if (
                (new_entry_due or restart_due)
                and ranking_early
                and may_attempt("entry", now)
            ):
                # A planned entry already passed the live clock check during
                # preflight. Avoid another network round trip at the target second.
                if not position.get("entry_preflight_market_validated"):
                    _validate_live_clock(client, today)
                result = enter_for_day(
                    client,
                    store,
                    config,
                    today,
                    submit=True,
                    now=now,
                    dispatch_target=entry_at,
                )
                _print_entry_plan(result, True)
                artifacts.write_summary(today, "enter", store, config)
            elif (
                new_entry_due
                and not ranking_early
                and may_attempt("missing-ranking", now)
            ):
                LOGGER.error(
                    "entry skipped: no ranking completed at least %d minutes before %s",
                    minimum_ranking_lead_minutes,
                    entry_time.strftime("%H:%M"),
                )
            state = store.load()
            position = state.get("position") or {}
            if (
                position.get("entry_date") == today.isoformat()
                and position.get("status") == "planned"
            ):
                remaining = (entry_at - datetime.now(tz=EASTERN)).total_seconds()
                if 0.0 < remaining < sleep_seconds:
                    sleep_seconds = max(0.001, remaining)
            last_error_key = None
        except Exception as error:
            key = (today.isoformat(), str(error))
            if key != last_error_key:
                LOGGER.exception(
                    "scheduled action failed; the daemon will retry: %s", error
                )
                last_error_key = key
            artifacts.write_summary(today, "error", store, config, error=str(error))
        time_module.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank and trade the most-liquid company stocks overnight."
    )
    parser.add_argument(
        "action", choices=("run", "preview", "rank", "enter", "exit", "status")
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--entry-time",
        type=parse_clock,
        default=parse_clock("15:45"),
        help="ET time to open the basket. The selected names drift about 3.5 bps "
        "upward between 15:45 and 15:59 (t=2.66 over 500 sessions), so a later "
        "entry pays more for the same basket; see experiments.py",
    )
    parser.add_argument(
        "--exit-time",
        type=parse_clock,
        default=parse_clock("08:00"),
        help="ET time to submit the exit. Market orders reaching Alpaca before Nasdaq's "
        "09:28 cutoff fill at the official opening cross, so the value only needs to be "
        "early enough to absorb queuing delays -- sub-one-share orders are held for a "
        "pre-open batch release around 09:15",
    )
    parser.add_argument(
        "--ranking-time",
        type=parse_clock,
        default=parse_clock("14:00"),
        help="ET time to start ranking, well before entry. The rank reads only "
        "completed sessions strictly before the entry date, so starting earlier "
        "buys headroom for the daily-cache refresh without changing the result",
    )
    parser.add_argument(
        "--liquidity-scheme",
        choices=("dollar_ema", "turnover_stability"),
        default="turnover_stability",
        help="dollar_ema ranks on the lagged log-dollar-volume EMA; turnover_stability "
        "subtracts that name's own dispersion, demoting a stock that is only briefly "
        "enormous below one that trades heavily every session",
    )
    parser.add_argument("--minimum-ranking-lead-minutes", type=int, default=20)
    parser.add_argument("--entry-grace-seconds", type=int, default=75)
    parser.add_argument("--ema-span", type=int, default=10)
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument(
        "--minimum-trading-days",
        type=int,
        default=100,
        help="minimum completed daily bars required before a company can be ranked",
    )
    parser.add_argument(
        "--liquidity-lookback-days",
        type=int,
        default=180,
        help="bounded recent calendar-day window of completed daily bars",
    )
    parser.add_argument(
        "--daily-bars-dir",
        type=Path,
        default=DEFAULT_DAILY_BARS_DIR,
        help="broad split-adjusted 1Day cache refreshed before ranking",
    )
    parser.add_argument(
        "--liquidity-shortlist",
        type=Path,
        default=DEFAULT_LIQUIDITY_SHORTLIST,
        help="output file for the rebuilt historical dollar-volume shortlist",
    )
    parser.add_argument(
        "--shortlist-since",
        type=date.fromisoformat,
        default=DEFAULT_SHORTLIST_SINCE,
        metavar="YYYY-MM-DD",
        help="first session considered when rebuilding the shortlist (default: 2022-01-01)",
    )
    parser.add_argument(
        "--shortlist-daily-top",
        type=int,
        default=50,
        help="union each session's top-N stocks by dollar volume (default: 50)",
    )
    parser.add_argument(
        "--shortlist-lookback-sessions",
        type=int,
        default=DEFAULT_SHORTLIST_LOOKBACK_SESSIONS,
        help="union only the most recent N completed sessions, so the shortlist tracks "
        "current liquidity instead of accumulating every past top-N member "
        f"(default: {DEFAULT_SHORTLIST_LOOKBACK_SESSIONS}); 0 unions everything since "
        "--shortlist-since",
    )
    parser.add_argument(
        "--daily-overlap-days",
        type=int,
        default=30,
        help="overlap used to detect corrections and splits in daily bars (default: 30)",
    )
    parser.add_argument(
        "--feed",
        choices=("iex", "sip"),
        default="sip",
        help="feed used to refresh split-adjusted daily ranking bars (default: sip)",
    )
    parser.add_argument(
        "--quote-feed",
        choices=("iex", "sip"),
        default="iex",
        help="feed used for latest asks in whole-share sizing (default: iex)",
    )
    parser.add_argument("--exchanges", default=",".join(sorted(DEFAULT_EXCHANGES)))
    parser.add_argument("--data-batch-size", type=int, default=100)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument(
        "--order-submit-workers",
        type=int,
        default=8,
        help="maximum concurrent entry-order submissions (default: 8)",
    )
    capital = parser.add_mutually_exclusive_group()
    capital.add_argument(
        "--capital", type=float, default=None, help="maximum dollars deployed"
    )
    capital.add_argument(
        "--capital-fraction",
        type=float,
        default=1.0,
        help="fraction of cash requested for deployment (default: 1.0)",
    )
    parser.add_argument("--cash-buffer-fraction", type=float, default=0.02)
    parser.add_argument(
        "--share-mode",
        choices=("whole", "fractional"),
        default="whole",
        help="whole floors equal-notional targets to integer shares; fractional sends notionals",
    )
    parser.add_argument(
        "--quote-max-age-seconds",
        type=float,
        default=120.0,
        help="maximum latest-ask age accepted for whole-share sizing (default: 120)",
    )
    parser.add_argument("--fill-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--entry-preflight-seconds",
        type=float,
        default=10.0,
        help="prepare account checks, quotes, and sizing this many seconds before entry (default: 10)",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="root for YYYY-MM-DD logs, market-data dumps, and summaries",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="durable restart state; default: WORK_DIR/state.json",
    )
    parser.add_argument(
        "--trade-date", default=None, help="rank/enter date; default: today ET"
    )
    parser.add_argument(
        "--trading-url", default=os.environ.get("ALPACA_URL", PAPER_TRADING_URL)
    )
    parser.add_argument(
        "--data-url", default=os.environ.get("ALPACA_DATA_URL", DEFAULT_DATA_URL)
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="actually send orders; enter/exit are dry runs without this flag",
    )
    parser.add_argument(
        "--allow-live-endpoint",
        action="store_true",
        help="explicitly allow an endpoint other than paper-api.alpaca.markets",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> StrategyConfig:
    if (
        args.top < 1
        or args.ema_span < 1
        or args.min_history_days < 1
        or args.minimum_trading_days < 1
    ):
        parser.error(
            "top, ema-span, min-history-days, and minimum-trading-days must be positive"
        )
    if args.liquidity_lookback_days < 30:
        parser.error("liquidity-lookback-days must be at least 30")
    if args.shortlist_daily_top < 1 or args.daily_overlap_days < 1:
        parser.error("shortlist-daily-top and daily-overlap-days must be positive")
    if args.shortlist_lookback_sessions < 0:
        parser.error("shortlist-lookback-sessions must be zero or positive")
    if not 1 <= args.data_batch_size <= 200 or not 1 <= args.data_workers <= 16:
        parser.error("data-batch-size must be 1..200 and data-workers must be 1..16")
    if not 1 <= args.order_submit_workers <= 32:
        parser.error("order-submit-workers must be in [1, 32]")
    if args.capital is not None and args.capital <= 0.0:
        parser.error("capital must be positive")
    if not 0.0 < args.capital_fraction <= 1.0:
        parser.error("capital-fraction must be in (0, 1]")
    if not 0.0 <= args.cash_buffer_fraction < 1.0:
        parser.error("cash-buffer-fraction must be in [0, 1)")
    if (
        args.fill_timeout_seconds <= 0.0
        or args.poll_seconds <= 0.0
        or args.quote_max_age_seconds <= 0.0
    ):
        parser.error(
            "fill-timeout-seconds, poll-seconds, and quote-max-age-seconds must be positive"
        )
    if (
        args.minimum_ranking_lead_minutes < 0
        or args.entry_grace_seconds < 0
        or args.entry_preflight_seconds < 0.0
    ):
        parser.error(
            "minimum-ranking-lead-minutes, entry-grace-seconds, and "
            "entry-preflight-seconds must be non-negative"
        )
    if args.request_timeout_seconds <= 0.0:
        parser.error("request-timeout-seconds must be positive")
    regular_open = time(9, 30)
    regular_close = time(16, 0)
    if not regular_open <= args.entry_time < regular_close:
        parser.error("entry-time must be within regular US equity hours [09:30, 16:00)")
    if not EXIT_SUBMISSION_OPEN <= args.exit_time < regular_close:
        parser.error(
            "exit-time must be within the exit submission window "
            f"[{EXIT_SUBMISSION_OPEN:%H:%M}, 16:00)"
        )
    if args.share_mode == "whole" and args.exit_time >= OPENING_AUCTION_CUTOFF:
        parser.error(
            "--share-mode whole requires --exit-time before 09:28 for OPG exits"
        )
    lead = datetime.combine(date.min, args.entry_time) - datetime.combine(
        date.min, args.ranking_time
    )
    if lead < timedelta(minutes=args.minimum_ranking_lead_minutes):
        parser.error(
            "ranking-time must be at least minimum-ranking-lead-minutes before entry-time"
        )
    if args.action == "run" and not args.submit:
        parser.error(
            "the run action requires --submit; use enter/exit without it to preview orders"
        )
    if args.action == "preview" and args.submit:
        parser.error("the preview action never accepts --submit")
    if (
        args.submit
        and not _is_paper_endpoint(args.trading_url)
        and not args.allow_live_endpoint
    ):
        parser.error(
            "refusing non-paper order submission without --allow-live-endpoint"
        )
    exchanges = frozenset(
        value.strip().upper() for value in args.exchanges.split(",") if value.strip()
    )
    if not exchanges:
        parser.error("exchanges cannot be empty")
    return StrategyConfig(
        top=args.top,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        minimum_trading_days=args.minimum_trading_days,
        lookback_calendar_days=args.liquidity_lookback_days,
        daily_bars_dir=args.daily_bars_dir,
        liquidity_shortlist=args.liquidity_shortlist,
        shortlist_since=args.shortlist_since,
        shortlist_daily_top=args.shortlist_daily_top,
        shortlist_lookback_sessions=args.shortlist_lookback_sessions or None,
        liquidity_scheme=args.liquidity_scheme,
        daily_overlap_days=args.daily_overlap_days,
        feed=args.feed,
        quote_feed=args.quote_feed,
        exchanges=exchanges,
        data_batch_size=args.data_batch_size,
        data_workers=args.data_workers,
        order_submit_workers=args.order_submit_workers,
        capital=args.capital,
        capital_fraction=args.capital_fraction,
        cash_buffer_fraction=args.cash_buffer_fraction,
        fill_timeout_seconds=args.fill_timeout_seconds,
        poll_seconds=args.poll_seconds,
        entry_preflight_seconds=args.entry_preflight_seconds,
        share_mode=args.share_mode,
        quote_max_age_seconds=args.quote_max_age_seconds,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _validate_args(parser, args)
    preview_workspace = (
        tempfile.TemporaryDirectory(prefix="overnight-liquidity-preview.")
        if args.action == "preview"
        else None
    )
    work_dir = (
        Path(preview_workspace.name)
        if preview_workspace is not None
        else Path(args.work_dir).expanduser()
    )
    trade_date = (
        date.fromisoformat(args.trade_date)
        if args.trade_date
        else datetime.now(EASTERN).date()
    )
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    daily_log = DailyLogHandler(
        work_dir,
        fixed_day=trade_date if args.action != "run" else None,
    )
    daily_log.setFormatter(log_format)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), daily_log],
    )
    state_path = (
        work_dir / "state.json"
        if preview_workspace is not None
        else Path(args.state_path).expanduser()
        if args.state_path
        else work_dir / "state.json"
    )
    store = StateStore(state_path)
    artifacts = DailyArtifacts(work_dir)
    artifacts.directory(trade_date)
    if args.action == "status":
        _state_status(store)
        artifacts.write_summary(trade_date, "status", store, config)
        return
    key, secret = load_credentials()
    data_key, data_secret = load_data_credentials(key, secret)
    client = AlpacaClient(
        key,
        secret,
        args.trading_url,
        args.data_url,
        timeout_seconds=args.request_timeout_seconds,
        data_key=data_key,
        data_secret=data_secret,
    )
    try:
        if args.action == "rank":
            _print_ranking(
                rank_for_day(client, store, config, trade_date, artifacts=artifacts),
                config.top,
            )
            artifacts.write_summary(trade_date, "rank", store, config)
        elif args.action == "preview":
            ranking = rank_for_day(client, store, config, trade_date)
            _print_ranking(ranking, config.top)
            result = enter_for_day(client, store, config, trade_date, submit=False)
            _print_entry_plan(result, False)
            CONSOLE.print(
                "Preview complete: no orders submitted and live strategy state was not changed."
            )
        elif args.action == "enter":
            if args.submit:
                _validate_live_clock(client, trade_date)
                with store.locked():
                    ranking = store.load().get("ranking") or {}
                if not _ranking_is_early_enough(
                    ranking,
                    _combine(trade_date, args.entry_time),
                    args.minimum_ranking_lead_minutes,
                ):
                    raise RuntimeError(
                        "refusing entry because ranking was not completed by the configured lead deadline"
                    )
            result = enter_for_day(client, store, config, trade_date, args.submit)
            _print_entry_plan(result, args.submit)
            artifacts.write_summary(trade_date, "enter", store, config)
        elif args.action == "exit":
            market_open = True
            exit_now = datetime.now(tz=EASTERN)
            if args.submit:
                market_open = _validate_exit_clock(client, trade_date)
            result = exit_position(
                client,
                store,
                config,
                args.submit,
                now=exit_now,
                wait_for_fill=market_open,
            )
            _print_exit(result, args.submit)
            artifacts.write_summary(trade_date, "exit", store, config)
            entry_day = date.fromisoformat(str(result["entry_date"]))
            if entry_day != trade_date:
                artifacts.write_summary(entry_day, "exit", store, config)
        else:
            run_daemon(
                client,
                store,
                config,
                args.ranking_time,
                args.entry_time,
                args.exit_time,
                args.minimum_ranking_lead_minutes,
                args.entry_grace_seconds,
                artifacts,
            )
    except Exception as error:
        LOGGER.exception("%s action failed: %s", args.action, error)
        artifacts.write_summary(trade_date, "error", store, config, error=str(error))
        raise
    finally:
        if preview_workspace is not None:
            preview_workspace.cleanup()


if __name__ == "__main__":
    main()
