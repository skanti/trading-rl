"""Strict, shared configuration for the live overnight strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf


DEFAULT_LIVE_CONFIG_PATH = Path(__file__).with_name("config.yaml")
EFFECTIVE_CONFIG_FILENAME = "effective_config.json"

CONFIG_FIELDS = {
    "schedule": (
        "time_zone",
        "ranking_time",
        "entry_time",
        "exit_time",
        "minimum_ranking_lead_minutes",
        "entry_grace_seconds",
    ),
    "strategy": (
        "top",
        "liquidity_scheme",
        "ema_span",
        "min_history_days",
        "minimum_trading_days",
        "liquidity_lookback_days",
        "exchanges",
    ),
    "data": (
        "daily_bars_dir",
        "liquidity_shortlist",
        "shortlist_since",
        "shortlist_daily_top",
        "shortlist_lookback_sessions",
        "daily_overlap_days",
        "feed",
        "quote_feed",
        "data_batch_size",
        "data_workers",
    ),
    "execution": (
        "order_submit_workers",
        "capital",
        "capital_fraction",
        "cash_buffer_fraction",
        "share_mode",
        "quote_max_age_seconds",
        "fill_timeout_seconds",
        "poll_seconds",
        "entry_preflight_seconds",
    ),
    "runtime": (
        "work_dir",
        "state_path",
        "trade_date",
        "trading_url",
        "data_url",
        "request_timeout_seconds",
        "submit",
        "allow_live_endpoint",
        "log_level",
    ),
}


@dataclass
class ScheduleSettings:
    time_zone: str = MISSING
    ranking_time: str = MISSING
    entry_time: str = MISSING
    exit_time: str = MISSING
    minimum_ranking_lead_minutes: int = MISSING
    entry_grace_seconds: int = MISSING


@dataclass
class StrategySettings:
    top: int = MISSING
    liquidity_scheme: str = MISSING
    ema_span: int = MISSING
    min_history_days: int = MISSING
    minimum_trading_days: int = MISSING
    liquidity_lookback_days: int = MISSING
    exchanges: str = MISSING


@dataclass
class DataSettings:
    daily_bars_dir: str = MISSING
    liquidity_shortlist: str = MISSING
    shortlist_since: str = MISSING
    shortlist_daily_top: int = MISSING
    shortlist_lookback_sessions: int = MISSING
    daily_overlap_days: int = MISSING
    feed: str = MISSING
    quote_feed: str = MISSING
    data_batch_size: int = MISSING
    data_workers: int = MISSING


@dataclass
class ExecutionSettings:
    order_submit_workers: int = MISSING
    capital: float | None = MISSING
    capital_fraction: float = MISSING
    cash_buffer_fraction: float = MISSING
    share_mode: str = MISSING
    quote_max_age_seconds: float = MISSING
    fill_timeout_seconds: float = MISSING
    poll_seconds: float = MISSING
    entry_preflight_seconds: float = MISSING


@dataclass
class RuntimeSettings:
    work_dir: str = MISSING
    state_path: str | None = MISSING
    trade_date: str | None = MISSING
    trading_url: str = MISSING
    data_url: str = MISSING
    request_timeout_seconds: float = MISSING
    submit: bool = MISSING
    allow_live_endpoint: bool = MISSING
    log_level: str = MISSING


@dataclass
class LiveSettings:
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    data: DataSettings = field(default_factory=DataSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


def load_live_settings(path: Path | str = DEFAULT_LIVE_CONFIG_PATH) -> DictConfig:
    """Load a complete config, rejecting missing, unknown, and mistyped values."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"live config does not exist: {config_path}")
    schema = OmegaConf.structured(LiveSettings)
    loaded = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, loaded)
    missing = sorted(OmegaConf.missing_keys(merged))
    if missing:
        raise ValueError(f"live config is missing required values: {', '.join(missing)}")
    OmegaConf.resolve(merged)
    OmegaConf.set_readonly(merged, True)
    return merged


def argparse_defaults(settings: DictConfig) -> dict[str, Any]:
    """Flatten the structured sections onto the existing argparse destinations."""
    values: dict[str, Any] = {}
    for section in CONFIG_FIELDS:
        payload = OmegaConf.to_container(settings[section], resolve=True)
        if not isinstance(payload, dict):
            raise TypeError(f"live config section {section!r} must be a mapping")
        values.update(payload)
    return values


def effective_live_settings(values: Any) -> dict[str, dict[str, Any]]:
    """Rebuild nested, JSON-safe settings after argparse applied CLI overrides."""
    output: dict[str, dict[str, Any]] = {}
    for section, fields in CONFIG_FIELDS.items():
        section_values: dict[str, Any] = {}
        for name in fields:
            value = getattr(values, name)
            if isinstance(value, (date, time)):
                value = (
                    value.isoformat(timespec="minutes")
                    if isinstance(value, time)
                    else value.isoformat()
                )
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, (set, frozenset, tuple)):
                value = list(value)
            section_values[name] = value
        output[section] = section_values
    return output
