"""Live-only data refresh and persistence for causal trend/volatility signals.

No simulation, CLI or order-submission modules are imported here. Network calls
are explicit and use the caller's authenticated, retry-bounded read client.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .execution_prices import load_scheduled_nbbo_asks
from .history import (
    EASTERN,
    _manifest_fingerprint,
    _official_opening_auctions,
    company_universe_mask,
    exchange_universe_mask,
    load_daily_closes,
    load_daily_dollar_volume,
    load_primary_auction_exchange_mask,
    simulation_symbols,
)
from .ranking import build_issuer_map
from .risk_history import historical_baskets, risk_signal, unit_return


def risk_configuration(config):
    return {
        "pipeline_version": 3,
        "spy_trend_price_source": "daily-close",
        "strategy": config.strategy_name,
        "parameters": config.risk_config.as_dict(),
        "top": config.top,
        "ema_span": config.ema_span,
        "min_history_days": config.min_history_days,
        "minimum_trading_days": config.minimum_trading_days,
        "liquidity_scheme": config.liquidity_scheme,
        "entry_minute": config.entry_minute,
        "exchanges": sorted(config.exchanges),
        "feed": config.feed,
        "lookback_calendar_days": config.lookback_calendar_days,
        "shortlist_since": config.shortlist_since.isoformat(),
        "shortlist_daily_top": config.shortlist_daily_top,
        "shortlist_lookback_sessions": config.shortlist_lookback_sessions,
        "share_mode": config.share_mode,
        "daily_bars_dir": str(config.daily_bars_dir),
        "minute_bars_dir": str(config.risk_minute_bars_dir),
        "nbbo_path": str(config.risk_nbbo_path),
        "auctions_path": str(config.risk_auctions_path),
    }


def configuration_fingerprint(config):
    return hashlib.sha256(
        json.dumps(risk_configuration(config), sort_keys=True).encode()
    ).hexdigest()


def signal_is_current(signal, config, trade_date):
    if not isinstance(signal, dict):
        return False
    exposure = signal.get("target_exposure")
    return (
        signal.get("strategy") == config.strategy_name
        and signal.get("spy_trend_price_source") == "daily-close"
        and signal.get("trade_date") == trade_date.isoformat()
        and signal.get("completed_exit_date") == trade_date.isoformat()
        and signal.get("configuration_sha256") == configuration_fingerprint(config)
        and isinstance(exposure, (float, int))
        and not isinstance(exposure, bool)
        and np.isfinite(exposure)
        and 0 < exposure <= config.risk_config.max_exposure
        and len(signal.get("observations") or [])
        == config.risk_config.volatility_window
        and len(signal.get("spy_history") or []) == config.risk_config.trend_window
        and signal.get("parameters") == config.risk_config.as_dict()
        and all(
            all(key in row for key in ("symbols", "entry_prices", "exit_prices"))
            for row in signal["observations"]
        )
        and bool(signal.get("input_sha256"))
    )


def _pages(client, path, params, *, version="v2"):
    seen = set()
    base = client.data_url.rsplit("/", 1)[0] + "/" + version
    while True:
        response = client._request(
            "GET", base, path, params=params, data_credentials=True
        )
        yield response
        token = response.get("next_page_token")
        if not token:
            break
        if token in seen:
            raise ValueError(f"repeated pagination token for {path}")
        seen.add(token)
        params = {**params, "page_token": token}


def _stamp(day, minute):
    return datetime.combine(day, time(minute // 60, minute % 60), tzinfo=EASTERN)


class RiskPriceProvider:
    """Use raw archive prices plus a current split ledger, avoiding mixed bases."""

    def __init__(self, client, config, dates, symbols, now):
        self.client, self.config, self.now = client, config, now
        self.asks = {}
        if config.risk_nbbo_path.exists() and config.entry_minute == 945:
            _, _, rows = load_scheduled_nbbo_asks(
                config.risk_nbbo_path, dates, symbols, 945
            )
            self.asks = {
                (row.date.date(), row.symbol): float(row.raw_price)
                for row in rows.itertuples()
                if row.staleness_minutes <= 1
            }
        self.opens = {}
        if config.risk_auctions_path.exists():
            with np.load(config.risk_auctions_path, allow_pickle=False) as archive:
                if "raw_price" not in archive.files:
                    raise ValueError(
                        "risk auction archive requires raw_price to validate split bases"
                    )
            rows = _official_opening_auctions(config.risk_auctions_path, dates, symbols)
            self.opens = {
                (row.date.date(), row.symbol): float(row.raw_price)
                for row in rows.itertuples()
            }
        self.splits = []

    def refresh_splits(self, symbols, start, end):
        if not symbols:
            return
        params = {
            "symbols": ",".join(sorted(symbols)),
            "types": "forward_split,reverse_split",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "sort": "asc",
        }
        for page in _pages(self.client, "corporate-actions", params, version="v1"):
            actions = page.get("corporate_actions")
            if not isinstance(actions, dict):
                raise ValueError("missing corporate-actions response for risk history")  # noqa: TRY004
            for kind in ("forward_splits", "reverse_splits"):
                self.splits.extend(actions.get(kind, []))
        # Duplicate corporate-action pages must not apply one split twice.
        unique = {}
        for row in self.splits:
            key = (row["symbol"], row["ex_date"], row["old_rate"], row["new_rate"])
            old, new = float(row["old_rate"]), float(row["new_rate"])
            if not np.isfinite([old, new]).all() or min(old, new) <= 0:
                raise ValueError("invalid split ratio in risk history")
            unique[key] = row
        self.splits = list(unique.values())

    def ask(self, symbol, day):
        key = (day, symbol)
        if key not in self.asks:
            target = _stamp(day, self.config.entry_minute)
            if target > self.now:
                raise ValueError("risk history would require a future entry quote")
            params = {
                "symbols": symbol,
                "start": (target - timedelta(seconds=60)).isoformat(),
                "end": target.isoformat(),
                "feed": "sip",
                "sort": "desc",
                "limit": 1000,
            }
            for page in _pages(self.client, "stocks/quotes", params):
                for quote in page.get("quotes", {}).get(symbol, []):
                    stamp = pd.Timestamp(quote["t"]).to_pydatetime()
                    ask, bid = float(quote.get("ap", 0)), float(quote.get("bp", 0))
                    if (
                        0 <= (target - stamp).total_seconds() <= 60
                        and np.isfinite([ask, bid]).all()
                        and 0 < bid <= ask
                    ):
                        self.asks[key] = ask
                        return ask
            raise ValueError(f"missing fresh 15:45 SIP ask for {symbol} on {day}")
        return self.asks[key]

    def opening(self, symbol, day):
        key = (day, symbol)
        if key not in self.opens:
            start = _stamp(day, 570)
            if self.now <= start:
                raise ValueError(f"opening auction on {day} has not completed yet")
            end = min(self.now, _stamp(day, 960))
            params = {
                "symbols": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "feed": "sip",
                "limit": 10000,
                "sort": "asc",
            }
            found = []
            for page in _pages(self.client, "stocks/auctions", params):
                for record in page.get("auctions", {}).get(symbol, []):
                    if record.get("d") != day.isoformat():
                        continue
                    for auction in record.get("o", []):
                        price = float(auction.get("p", 0))
                        if (
                            auction.get("c") == "O"
                            and np.isfinite(price)
                            and price > 0
                            and pd.Timestamp(auction["t"]).to_pydatetime() <= self.now
                        ):
                            priority = {
                                "N": 0,
                                "Q": 0,
                                "P": 0,
                                "A": 0,
                                "T": 1,
                                "V": 2,
                            }.get(auction.get("x"), 3)
                            found.append((float(auction["s"]), -priority, price))
            if not found:
                raise ValueError(
                    f"missing primary opening auction for {symbol} on {day}"
                )
            self.opens[key] = max(found)[2]
        return self.opens[key]

    def adjusted_entry(self, symbol, entry_day, exit_day):
        value = self.ask(symbol, entry_day)
        for split in self.splits:
            if (
                split["symbol"] == symbol
                and entry_day.isoformat() < split["ex_date"] <= exit_day.isoformat()
            ):
                value *= float(split["old_rate"]) / float(split["new_rate"])
        return value

    def spy_marks(self, dates, trend_window):
        # rank_for_day refreshes the split-adjusted daily cache first, including
        # SPY. Use that same source as the backtester, with no extra minute API call.
        selected = dates[-trend_window - 1 : -1]
        if len(selected) != trend_window or any(
            day.date() >= self.now.astimezone(EASTERN).date() for day in selected
        ):
            raise ValueError("SPY trend requires completed previous daily bars")
        closes = load_daily_closes(self.config.daily_bars_dir / "SPY.npy", selected)
        if not np.isfinite(closes).all():
            missing = [str(day.date()) for day in selected[~np.isfinite(closes)]]
            raise ValueError("missing SPY daily close for " + ", ".join(missing))
        marks = np.full(len(dates), np.nan)
        marks[-trend_window - 1 : -1] = closes
        return marks


def prepare_live_risk(
    client, config, trade_date, security_master, output_dir, now=None
):
    """Refresh a finite risk snapshot; no access to positions or order endpoints."""
    now = now or datetime.now(EASTERN)
    if now.astimezone(EASTERN).date() < trade_date or now < _stamp(trade_date, 570):
        raise ValueError(
            f"risk signal for {trade_date} requires that morning's completed opening auction"
        )
    sessions = client.calendar(config.shortlist_since, trade_date)
    dates = pd.DatetimeIndex([row["date"] for row in sessions])
    if dates.empty or dates[-1].date() != trade_date:
        raise ValueError("risk calendar does not reach the requested trading session")
    if len(dates) <= max(
        config.risk_config.trend_window, config.risk_config.volatility_window
    ):
        raise ValueError("risk calendar has insufficient history")
    closes = [
        int(row["close"].split(":")[0]) * 60 + int(row["close"].split(":")[1])
        for row in sessions
    ]
    daily_manifest = _manifest_fingerprint(config.daily_bars_dir, "1Day")
    symbols = simulation_symbols(config.risk_minute_bars_dir, config.daily_bars_dir)
    company, _, _ = company_universe_mask(
        symbols, security_master, keep_unclassified=True
    )
    symbols = symbols[
        company & exchange_universe_mask(symbols, security_master, "nasdaq")
    ]
    positions = {day: row for row, day in enumerate(dates)}
    dollar = np.column_stack(
        [
            load_daily_dollar_volume(
                config.daily_bars_dir / f"{symbol}.npy", positions, len(dates)
            )
            for symbol in symbols
        ]
    )
    # Missing the last completed session must not masquerade as a zero-volume day.
    if not np.isfinite(dollar[-2]).any():
        raise ValueError("daily ranking inputs have not reached the previous session")
    exchange, _ = load_primary_auction_exchange_mask(
        config.risk_auctions_path, dates, symbols, "nasdaq"
    )
    rows, baskets = historical_baskets(
        dates,
        symbols,
        dollar,
        np.asarray(closes) > config.entry_minute,
        exchange,
        build_issuer_map(symbols, security_master),
        config.risk_config,
        top=config.top,
        ema_span=config.ema_span,
        min_history_days=config.min_history_days,
        minimum_trading_days=config.minimum_trading_days,
        liquidity_scheme=config.liquidity_scheme,
    )
    selected_symbols = {str(symbols[column]) for basket in baskets for column in basket}
    for row, basket in zip(rows, baskets, strict=True):
        if row == 0 or not np.isfinite(dollar[row - 1, basket]).all():
            raise ValueError(
                f"incomplete previous-session ranking inputs for {dates[row].date()}"
            )
    provider = RiskPriceProvider(
        client, config, dates[rows[0] :], np.array(sorted(selected_symbols)), now
    )
    provider.refresh_splits(selected_symbols, dates[rows[0]].date(), trade_date)
    observations = []
    for row, basket in zip(rows, baskets, strict=True):
        entry_day, exit_day = dates[row].date(), dates[row + 1].date()
        names = [str(symbols[column]) for column in basket]
        entries = [
            provider.adjusted_entry(symbol, entry_day, exit_day) for symbol in names
        ]
        exits = [provider.opening(symbol, exit_day) for symbol in names]
        observations.append(
            {
                "entry_date": str(entry_day),
                "exit_date": str(exit_day),
                "symbols": names,
                "entry_prices": entries,
                "exit_prices": exits,
                "unscaled_return": unit_return(entries, exits),
            }
        )
    marks = provider.spy_marks(dates, config.risk_config.trend_window)
    signal = risk_signal(dates, observations, marks, config.risk_config)
    signal["configuration_sha256"] = configuration_fingerprint(config)
    signal["created_at"] = now.isoformat()
    signal["spy_history"] = [
        {"date": str(day.date()), "price": float(mark)}
        for day, mark in zip(
            dates[-config.risk_config.trend_window - 1 : -1],
            marks[-config.risk_config.trend_window - 1 : -1],
            strict=True,
        )
    ]
    signal["split_actions"] = provider.splits
    signal["daily_manifest"] = daily_manifest
    signal["input_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "observations": observations,
                "spy_marks": marks[-config.risk_config.trend_window - 1 : -1].tolist(),
                "daily_volume": hashlib.sha256(dollar.tobytes()).hexdigest(),
                "daily_manifest": daily_manifest,
                "universe": symbols.tolist(),
                "security_master": security_master,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output_dir = Path(output_dir) / "risk" / config.strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{trade_date}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(signal, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)
    signal["path"] = str(path)
    return signal
