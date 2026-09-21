"""Pure basket selection, sizing and unit-exposure returns."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from .history import REFERENCE_SYMBOL
from .ranking import select_ranked_indices, top_ranked_indices


def select_strategy_basket(
    symbols: np.ndarray,
    scores: np.ndarray,
    completed_days: np.ndarray,
    entry_prices: np.ndarray,
    entry_staleness: np.ndarray,
    *,
    top: int,
    minimum_trading_days: int,
    max_entry_staleness_minutes: int,
    entry_price_source: str,
    issuers: Mapping[str, str] | None = None,
    dedupe_share_classes: bool = True,
    execution_exchange_mask: np.ndarray | None = None,
    reference_symbol: str = REFERENCE_SYMBOL,
) -> np.ndarray:
    """Select intended members before execution availability or sizing drops trades."""
    eligible = symbols != str(reference_symbol)
    if execution_exchange_mask is not None:
        eligible &= execution_exchange_mask
    # Sparse NBBO coverage must never replace an intended member with a lower rank.
    if entry_price_source != "nbbo-ask":
        eligible &= (
            np.isfinite(entry_prices)
            & (entry_prices > 0.0)
            & (entry_staleness <= max_entry_staleness_minutes)
        )
    return top_ranked_indices(
        scores,
        symbols,
        top,
        completed_days=completed_days,
        minimum_trading_days=minimum_trading_days,
        eligible_mask=eligible,
        issuers=issuers if dedupe_share_classes else None,
    )


def basket_quantities(
    entry_prices: np.ndarray,
    budget: float,
    share_mode: str = "fractional",
    *,
    slots: int | None = None,
) -> np.ndarray:
    """Size one equal-notional basket in fractional or whole shares.

    Whole-share sizing deliberately rounds each name down independently. This never
    exceeds the budget and never redistributes a costly name's unused allocation into
    cheaper names, which would change the strategy's intended equal weighting. A stock
    priced above its per-name target therefore receives zero shares and its allocation
    remains cash. ``slots`` also reserves cash for selected names without prices;
    omitting it allocates across every supplied price, as live whole-share sizing does.
    """
    prices = np.asarray(entry_prices, dtype=np.float64)
    if prices.ndim != 1 or not prices.size:
        raise ValueError("entry_prices must be a non-empty one-dimensional array")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("entry_prices must be finite and positive")
    if not np.isfinite(budget) or float(budget) <= 0.0:
        raise ValueError("budget must be finite and positive")
    if share_mode not in {"fractional", "whole"}:
        raise ValueError("share_mode must be 'fractional' or 'whole'")

    slot_count = prices.size if slots is None else slots
    if slot_count < prices.size:
        raise ValueError("slots cannot be fewer than the priced basket members")
    target_notional = equal_notional(budget, slot_count)
    quantities = target_notional / prices
    return np.floor(quantities) if share_mode == "whole" else quantities


def unscaled_basket_return(entry_prices, exit_prices, top, cost_bps=0.0):
    """Equal slot weights; absent slots remain cash. Validate prices at the caller."""
    return float(
        np.sum(
            np.asarray(exit_prices) / np.asarray(entry_prices)
            - 1.0
            - 2.0 * cost_bps / 10000.0
        )
        / top
    )


def select_unconflicted_candidates(
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
    # Broker conflicts are live-specific eligibility constraints. Preserve the
    # archived ranking order and delegate issuer deduplication/top-N to the core.
    candidates = [
        candidate
        for candidate in ranking["candidates"]
        if str(candidate["symbol"]) not in held_symbols | open_order_symbols
    ]
    symbols = np.asarray([str(candidate["symbol"]) for candidate in candidates])
    issuers = {
        str(candidate["symbol"]): str(candidate.get("issuer") or candidate["symbol"])
        for candidate in candidates
    }
    try:
        selected = select_ranked_indices(
            np.arange(len(symbols)),
            symbols,
            top,
            issuers=issuers if dedupe_share_classes else None,
        )
    except ValueError as error:
        raise RuntimeError(
            "not enough ranked candidates remain after excluding existing positions/orders: "
            + str(error)
        ) from error
    return symbols[selected].tolist()


def client_order_id(entry_date: date, side: str, symbol: str, attempt: int = 1) -> str:
    clean_symbol = "".join(
        character for character in symbol.upper() if character.isalnum()
    )
    marker = "e" if side == "buy" else "x"
    suffix = "" if attempt == 1 else f"-{attempt}"
    return f"olq-{entry_date:%Y%m%d}-{marker}-{clean_symbol}{suffix}"[:128]


def market_entry_order(entry_date, symbol, size):
    """The deterministic broker payload used for submission and offline replay."""
    return {
        "symbol": symbol,
        **size,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": client_order_id(entry_date, "buy", symbol),
    }


def equal_notional(budget: float, slots: int, *, round_to_cents: bool = False) -> float:
    """Keep unfilled basket slots in cash; broker rounding is an explicit choice."""
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    if isinstance(slots, bool) or int(slots) != slots or slots < 1:
        raise ValueError("slots must be a positive integer")
    target = float(budget) / int(slots)
    return math.floor(target * 100) / 100 if round_to_cents else target


@dataclass(frozen=True)
class BasketReturns:
    missing_entry: np.ndarray
    missing_exit: np.ndarray
    unscaled_return: float

    @property
    def missing(self):
        return self.missing_entry | self.missing_exit


def basket_returns(
    entry_prices,
    exit_prices,
    *,
    slots,
    cost_bps=0.0,
    entry_staleness=None,
    exit_staleness=None,
    max_entry_age=0.0,
    max_exit_age=0.0,
    require_complete=False,
):
    """Validate one modeled interval and retain cash weights for unavailable slots.

    A unit-exposure risk observation ignores actual leverage, share rounding and
    broker financing. Live bootstrap and simulation use this same calculation.
    """
    entries, exits = (
        np.asarray(entry_prices, dtype=float),
        np.asarray(exit_prices, dtype=float),
    )
    if entries.ndim != 1 or entries.shape != exits.shape:
        raise ValueError(
            "entry and exit prices must have matching one-dimensional shapes"
        )
    if isinstance(slots, bool) or int(slots) != slots or slots < max(1, len(entries)):
        raise ValueError("slots must cover the complete intended basket")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")

    def missing(prices, staleness, limit):
        age = (
            np.zeros_like(prices)
            if staleness is None
            else np.asarray(staleness, dtype=float)
        )
        if age.shape != prices.shape:
            raise ValueError("price staleness must match prices")
        return (
            ~np.isfinite(prices)
            | (prices <= 0)
            | ~np.isfinite(age)
            | (age < 0)
            | (age > limit)
        )

    missing_entry = missing(entries, entry_staleness, max_entry_age)
    missing_exit = missing(exits, exit_staleness, max_exit_age)
    valid = ~(missing_entry | missing_exit)
    if require_complete and not valid.all():
        raise ValueError(
            "risk history requires complete positive entry and exit prices"
        )
    return BasketReturns(
        missing_entry,
        missing_exit,
        unscaled_basket_return(entries[valid], exits[valid], slots, cost_bps),
    )
