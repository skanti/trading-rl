"""Causal liquidity ranking shared by live execution, backtests, and exports.

Inputs are arrays and explicit eligibility constraints. This module performs no
file, network, order, execution-price, or position-sizing operations.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


# "Alphabet Inc. - Class C Capital Stock" and "Alphabet Inc. - Class A Common Stock"
# are one company listed twice. Ranking treats them as separate names, so a basket can
# end up holding the same issuer at double weight while reporting the nominal size.
_SHARE_CLASS_SUFFIX = re.compile(r"\s*[-\u2013]\s*Class\s+.*$", re.IGNORECASE)
# The leading ``\s+`` matters: without it a company whose name begins with a descriptor
# word ("American Airlines Group Inc. - Common Stock") would be erased to nothing.
_SECURITY_TYPE_SUFFIX = re.compile(
    r"\s*[-\u2013]?\s+(?:Common|Capital|Ordinary|Preferred)\s+(?:Stock|Shares)\b.*$",
    re.IGNORECASE,
)


def issuer_key(
    symbol: str, security_master: Mapping[str, Mapping[str, object]] | None
) -> str:
    """Collapse every listed share class of one company onto a single key.

    Falls back to the symbol itself when the security master has no name, so an unknown
    ticker is never silently merged with an unrelated one.
    """
    if not security_master:
        return symbol
    record = security_master.get(str(symbol).replace("-", ".").upper())
    name = str(record.get("name", "")) if record else ""
    if not name:
        return symbol
    trimmed = _SECURITY_TYPE_SUFFIX.sub("", _SHARE_CLASS_SUFFIX.sub("", name))
    trimmed = trimmed.strip().rstrip(" -\u2013,")
    return trimmed.casefold() or symbol


def build_issuer_map(
    symbols: Iterable[str], security_master: Mapping[str, Mapping[str, object]] | None
) -> dict[str, str]:
    """Map each sample id to its issuer key."""
    return {str(symbol): issuer_key(str(symbol), security_master) for symbol in symbols}


def causal_ema_log_liquidity(
    dollar_volume: np.ndarray,
    span: int,
    min_history_days: int,
) -> np.ndarray:
    """Return lagged EMA scores; row ``t`` can only use rows before ``t``.

    The EMA is applied to ``log1p(dollar_volume)``. This preserves liquidity
    ordering while preventing one exceptional volume day from dominating the
    ranking for weeks. Missing sessions decay an already-started EMA toward
    zero and do not count toward the minimum observed history.
    """
    values = np.asarray(dollar_volume, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("dollar_volume must have shape (dates, symbols)")
    if int(span) < 1:
        raise ValueError("span must be positive")
    if int(min_history_days) < 1:
        raise ValueError("min_history_days must be positive")
    if np.isfinite(values).any() and np.nanmin(values) < 0.0:
        raise ValueError("dollar volume must be non-negative")

    alpha = 2.0 / (int(span) + 1.0)
    dates, symbols = values.shape
    scores = np.full((dates, symbols), np.nan, dtype=np.float64)
    ema = np.zeros(symbols, dtype=np.float64)
    observations = np.zeros(symbols, dtype=np.int32)
    started = np.zeros(symbols, dtype=bool)
    for date_index in range(dates):
        eligible = observations >= int(min_history_days)
        scores[date_index, eligible] = ema[eligible]

        observed = np.isfinite(values[date_index]) & (values[date_index] > 0.0)
        transformed = np.zeros(symbols, dtype=np.float64)
        transformed[observed] = np.log1p(values[date_index, observed])
        continuing = started | observed
        first = observed & ~started
        ema[continuing] = (1.0 - alpha) * ema[continuing] + alpha * transformed[
            continuing
        ]
        ema[first] = transformed[first]
        observations[observed] += 1
        started |= observed
    return scores


def causal_turnover_stability(
    dollar_volume: np.ndarray,
    ema_span: int,
    min_history_days: int,
    *,
    dispersion_span: int | None = None,
) -> np.ndarray:
    """Rank on liquidity level less the dispersion of that same liquidity.

    Both terms are log dollars, so the subtraction needs no weighting to be
    meaningful: a name whose log turnover swings by 1.0 is docked exactly as much
    as a name with e times less turnover. The penalty demotes a stock that is only
    briefly enormous -- an earnings day, an index rebalance -- beneath one that
    trades heavily every session. That matters here because the basket is held
    through an entire overnight, which is when event risk actually pays out.

    Both terms are causal. The EMA is already lagged, and the dispersion is
    shifted one session, so row ``t`` sees only sessions strictly before ``t``.
    """
    # Production uses one horizon for both terms. The optional override exists only
    # so reconciliation can faithfully replay sessions produced by the former
    # fixed-20-session model.
    effective_dispersion_span = (
        int(ema_span) if dispersion_span is None else int(dispersion_span)
    )
    if effective_dispersion_span < 2:
        raise ValueError("turnover stability span must cover at least two sessions")
    level = causal_ema_log_liquidity(dollar_volume, ema_span, min_history_days)
    values = np.asarray(dollar_volume, dtype=np.float64)
    logged = np.log1p(np.where(np.isfinite(values) & (values > 0.0), values, np.nan))
    spread = (
        pd.DataFrame(logged)
        .rolling(
            effective_dispersion_span,
            min_periods=effective_dispersion_span // 2,
        )
        .std()
        .shift(1)
        .to_numpy()
    )
    # A name without enough history to measure dispersion is not penalised for it;
    # the separate minimum-history filter is what keeps such names out of a basket.
    return level - np.nan_to_num(spread, nan=0.0)


def causal_completed_trading_days(dollar_volume: np.ndarray) -> np.ndarray:
    """Count valid completed sessions strictly before every candidate date."""
    values = np.asarray(dollar_volume, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("dollar_volume must have shape (dates, symbols)")
    observed = np.isfinite(values) & (values > 0.0)
    cumulative = np.cumsum(observed, axis=0, dtype=np.int32)
    return cumulative - observed.astype(np.int32)


def liquidity_scores(
    dollar_volume: np.ndarray,
    scheme: str,
    ema_span: int,
    min_history_days: int,
    *,
    dispersion_span: int | None = None,
) -> np.ndarray:
    """Build causal scores for a liquidity scheme supported by live trading."""
    dollar = np.asarray(dollar_volume, dtype=np.float64)
    if dollar.ndim != 2:
        raise ValueError("dollar_volume must have shape (dates, symbols)")
    if scheme == "dollar_ema":
        return causal_ema_log_liquidity(dollar, ema_span, min_history_days)
    if scheme == "turnover_stability":
        return causal_turnover_stability(
            dollar, ema_span, min_history_days, dispersion_span=dispersion_span
        )
    raise ValueError(f"unknown liquidity scheme: {scheme}")


def ranked_indices(
    scores: np.ndarray,
    symbols: np.ndarray,
    *,
    completed_days: np.ndarray | None = None,
    minimum_trading_days: int = 1,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply history/eligibility rules, then sort by descending score and symbol."""
    scores = np.asarray(scores, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=str)
    if scores.ndim != 1 or symbols.shape != scores.shape:
        raise ValueError("scores and symbols must be matching vectors")
    if minimum_trading_days < 1:
        raise ValueError("minimum_trading_days must be positive")
    eligible = np.isfinite(scores)
    if completed_days is not None:
        completed_days = np.asarray(completed_days)
        if completed_days.shape != scores.shape:
            raise ValueError("completed_days must match scores")
        eligible &= completed_days >= minimum_trading_days
    if eligible_mask is not None:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)
        if eligible_mask.shape != scores.shape:
            raise ValueError("eligible_mask must match scores")
        eligible &= eligible_mask
    indices = np.flatnonzero(eligible)
    order = np.lexsort((symbols[indices], -scores[indices]))
    return indices[order]


def select_ranked_indices(
    ranked: np.ndarray,
    symbols: np.ndarray,
    top: int,
    *,
    issuers: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Take top-N in ranked order, backfilling duplicate company share classes."""
    if top < 1:
        raise ValueError("top must be positive")
    chosen: list[int] = []
    seen: set[str] = set()
    for index in ranked:
        name = str(symbols[index])
        key = issuers.get(name, name) if issuers is not None else name
        if issuers is not None and key in seen:
            continue
        seen.add(key)
        chosen.append(int(index))
        if len(chosen) == top:
            return np.asarray(chosen, dtype=np.int64)
    label = "distinct issuers" if issuers is not None else "symbols"
    raise ValueError(
        f"only {len(chosen)} causally eligible {label} are available for top={top}"
    )


def top_ranked_indices(
    scores: np.ndarray,
    symbols: np.ndarray,
    top: int,
    *,
    completed_days: np.ndarray | None = None,
    minimum_trading_days: int = 1,
    eligible_mask: np.ndarray | None = None,
    issuers: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Rank eligible candidates and select top-N with shared issuer deduplication."""
    ranked = ranked_indices(
        scores,
        symbols,
        completed_days=completed_days,
        minimum_trading_days=minimum_trading_days,
        eligible_mask=eligible_mask,
    )
    return select_ranked_indices(ranked, symbols, top, issuers=issuers)


def replay_strategy_selections(
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    dollar_volume: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    top: int,
    ema_span: int,
    min_history_days: int,
    minimum_trading_days: int,
    liquidity_scheme: str = "turnover_stability",
    issuers: Mapping[str, str] | None = None,
    dedupe_share_classes: bool = True,
    execution_exchange_mask: np.ndarray | None = None,
    entry_session_mask: np.ndarray | None = None,
    eligibility_mask: np.ndarray | None = None,
    reference_symbol: str = "SPY",
) -> pd.DataFrame:
    """Replay intended baskets using liquidity and explicit eligibility masks only.

    Exchange eligibility uses the following exit session, matching the backtest's
    historical listing-venue check. Price availability and sizing are caller concerns.
    """
    expected_shape = (len(dates), len(symbols))
    if dollar_volume.shape != expected_shape:
        raise ValueError("ranking arrays must match dates and symbols")
    for label, mask in (
        ("execution_exchange_mask", execution_exchange_mask),
        ("eligibility_mask", eligibility_mask),
    ):
        if mask is not None and mask.shape != expected_shape:
            raise ValueError(f"{label} must match the liquidity arrays")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("ranking dates must be unique and increasing")
    if entry_session_mask is None:
        entry_session_mask = np.ones(len(dates), dtype=bool)
    if entry_session_mask.shape != (len(dates),):
        raise ValueError("entry_session_mask must match dates")
    entries = np.flatnonzero(
        (dates >= start_date) & (dates < end_date) & entry_session_mask
    )
    if not entries.size:
        raise ValueError("the requested interval contains no entry sessions")
    if entries[-1] + 1 >= len(dates):
        raise ValueError("the final entry session has no following exit session")
    scores = liquidity_scores(
        dollar_volume, liquidity_scheme, ema_span, min_history_days
    )
    completed = causal_completed_trading_days(dollar_volume)
    records = []
    for index in entries:
        eligible = symbols != reference_symbol
        if execution_exchange_mask is not None:
            eligible &= execution_exchange_mask[index + 1]
        if eligibility_mask is not None:
            eligible &= eligibility_mask[index]
        try:
            selected = top_ranked_indices(
                scores[index],
                symbols,
                top,
                completed_days=completed[index],
                minimum_trading_days=minimum_trading_days,
                eligible_mask=eligible,
                issuers=issuers if dedupe_share_classes else None,
            )
        except ValueError as error:
            raise ValueError(
                f"cannot rank {dates[index]:%Y-%m-%d}: {error}; "
                f"each symbol needs {minimum_trading_days} completed sessions and "
                f"{min_history_days} liquidity observations. "
                "Check universe filters and provide earlier bar history or choose a later --since."
            ) from error
        records.extend(
            {
                "entry_date": dates[index].date().isoformat(),
                "exit_date": dates[index + 1].date().isoformat(),
                "rank": rank,
                "sample_id": str(symbols[chosen]),
                "liquidity_score": float(scores[index, chosen]),
            }
            for rank, chosen in enumerate(selected, 1)
        )
    return pd.DataFrame.from_records(records)
