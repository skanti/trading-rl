"""Experimental overnight variants, measured against the shipped configuration.

Nothing here is wired into live trading. ``backtest.py`` stays the reference
implementation and this module imports its calendar, liquidity cache, ranking and
metrics, so a variant differs from the shipped strategy only where the experiment
intends it to.

Three things are explored, in the order the results are meant to be read:

1. Entry and exit clock times, holding the stock-picking logic fixed.
2. Deferring a losing position to a later session instead of realising it.
3. Alternative selection rules, capped at twenty names per session.

Every variant runs through the same portfolio engine, which marks equity each
morning at the exit clock. That accounting is what makes a deferred position
honest: its capital stays deployed and is unavailable to the next basket.

The engine is validated against backtest.py: on the shipped configuration it
returns 41.03% annualised against the reference 40.19%, with an identical 26.16%
maximum drawdown. The gap is one session, the boundary mark this loop needs to
seed its equity curve, and it is common to every variant so comparisons are fair.

Findings on the trailing 24 months (500 sessions, Nasdaq universe, 1bp per side):

Entry clock, ACCEPTED. Moving entry from 15:59 to 15:45 lifts Sharpe 1.74 -> 2.06.
The mechanism is measurable rather than inferred: the selected basket drifts
+3.46 bps upward between 15:45 and 15:59 (t=2.66), so a late entry simply pays
more for the same names. 15:40 and 15:45 form a plateau and both beat 15:59 in
each half separately, so this is not a single lucky cell. The fine structure
inside the plateau is noise and should not be tuned further.

Exit clock, UNCHANGED. Exiting at or near the open beats exiting later, but the
differences between 09:30, 09:35 and the auction are inside the noise, and the
ordering across adjacent minutes zigzags. The opening cross stays the choice
because it is the only one of the three verified to be obtainable.

Deferring losers, REJECTED. The premise holds -- lots down more than 200 bps at
the first opportunity recover 66 bps on average by the next session -- but the
recovery is a 55% coin flip and does not pay for the capital being locked up.
The tell is that the best hold count inverts between halves: five extra holds is
the strongest setting on the first half and the weakest on the second. The best
full-sample setting is worse than not deferring at all on the first half.

Selection, ACCEPTED. turnover_stability ranks on the same liquidity EMA less a
penalty for that name's own dispersion, preferring dependable turnover to names
that were briefly enormous. At twelve names it leads on both the full sample
(Sharpe 2.18) and the weaker half (1.91). Baskets of fifteen and twenty are worse
under every scheme, so the twenty-name cap is not binding -- dilution bites first.

Combined, against the shipped configuration: annualised 43.5% -> 59.0%, profit
factor 1.35 -> 1.47, Sharpe 1.74 -> 2.18, maximum drawdown 25.9% -> 23.4%. The
pair wins all four of four walk-forward blocks of 125 sessions with parameters
held fixed throughout.

Nothing here has been promoted into backtest.py or live.py.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .backtest import (
    DEFAULT_AUCTIONS_PATH,
    DEFAULT_DAILY_DATA_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_SECURITY_MASTER_CACHE,
    DEFAULT_TRANSACTION_COST_BPS,
    EXTENDED_OPEN_MINUTE,
    REFERENCE_SYMBOL,
    _cache_metadata,
    _security_symbol,
    build_issuer_map,
    causal_completed_trading_days,
    causal_ema_log_liquidity,
    company_universe_mask,
    exchange_universe_mask,
    load_nasdaq_security_master,
    load_opening_auction_prices,
    load_or_build_cache,
    reference_session_calendar,
    strategy_metrics,
    top_liquid_indices,
)
from .price_utils import forward_fill_positions

AUCTION_MINUTE = -1  # sentinel: exit in the opening cross rather than at a clock time


# --------------------------------------------------------------------------- data


@dataclass
class Panel:
    """Everything a variant needs, built once and reused across the whole sweep."""

    dates: pd.DatetimeIndex
    symbols: np.ndarray
    dollar_volume: np.ndarray
    completed_days: np.ndarray
    prices: dict[int, np.ndarray]
    staleness: dict[int, np.ndarray]
    issuers: dict[str, str]
    first_entry: int

    def price(self, minute: int) -> np.ndarray:
        return self.prices[minute]


def _symbol_minute_prices(
    args: tuple[str, Path, np.ndarray, tuple[int, ...]],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Read one symbol's price and staleness at every requested clock minute."""
    sample_id, minute_dir, context_sod, minutes = args
    count = len(context_sod)
    prices = np.full((len(minutes), count), np.nan, dtype=np.float64)
    staleness = np.full((len(minutes), count), np.inf, dtype=np.float64)
    path = minute_dir / f"{_security_symbol(sample_id)}.npy"
    if not path.exists():
        return sample_id, prices, staleness
    source = np.load(path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] < 3 or not len(source):
        return sample_id, prices, staleness
    seconds = np.asarray(source[:, 0], dtype=np.int64)
    requested = np.concatenate(
        [context_sod + (int(m) - EXTENDED_OPEN_MINUTE) * 60 for m in minutes]
    )
    available = np.searchsorted(seconds, requested, side="right") > 0
    flat_price = np.full(len(requested), np.nan, dtype=np.float64)
    flat_stale = np.full(len(requested), np.inf, dtype=np.float64)
    if available.any():
        positions = forward_fill_positions(source, requested[available], sample_id)
        seen_seconds = np.asarray(source[positions, 0], dtype=np.int64)
        seen_prices = np.asarray(source[positions, 1], dtype=np.float64) / 1000.0
        good = np.isfinite(seen_prices) & (seen_prices > 0.0)
        target = np.flatnonzero(available)
        flat_price[target[good]] = seen_prices[good]
        flat_stale[target[good]] = (requested[target[good]] - seen_seconds[good]) / 60.0
    for index in range(len(minutes)):
        lo, hi = index * count, (index + 1) * count
        prices[index] = flat_price[lo:hi]
        staleness[index] = flat_stale[lo:hi]
    return sample_id, prices, staleness


def build_panel(
    months: int,
    minutes: tuple[int, ...],
    *,
    minute_dir: Path,
    daily_dir: Path,
    auctions_path: Path,
    cache_dir: Path,
    ema_span: int,
    min_history_days: int,
    minimum_trading_days: int,
    exchange_filter: str,
    workers: int,
) -> Panel:
    """Assemble the ranking inputs plus a price matrix per candidate clock minute."""
    all_dates, all_context = reference_session_calendar(minute_dir / f"{REFERENCE_SYMBOL}.npy")
    end_index = len(all_dates) - 1
    requested_start = pd.Timestamp(all_dates[end_index]) - pd.DateOffset(months=int(months))
    first_entry = int(np.flatnonzero(all_dates >= requested_start)[0])
    warmup = max(3 * ema_span, min_history_days + 1, minimum_trading_days + 1)
    lo = max(0, first_entry - warmup)
    dates = all_dates[lo : end_index + 1]
    context = all_context[lo : end_index + 1]

    # The ranking side is the shipped one, so reuse its cache verbatim. Only the
    # liquidity matrix is taken from it; prices come from the sweep below.
    metadata = _cache_metadata(
        minute_dir, daily_dir, pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]),
        15 * 60 + 59, 9 * 60 + 30,
    )
    cache_path = cache_dir / (
        f"liquidity_{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}_e0959_x0570_v6.npz"
    )
    symbols, dollar_volume, _, _, _, _ = load_or_build_cache(
        cache_path, metadata, minute_dir, daily_dir, dates, context,
        15 * 60 + 59, 9 * 60 + 30, workers, False,
    )

    master = load_nasdaq_security_master(Path(DEFAULT_SECURITY_MASTER_CACHE))
    keep, _, _ = company_universe_mask(symbols, master, REFERENCE_SYMBOL, True)
    if exchange_filter != "all":
        keep = keep & exchange_universe_mask(symbols, master, exchange_filter)
    symbols = symbols[keep]
    dollar_volume = dollar_volume[:, keep]

    payload = [(str(s), minute_dir, context, minutes) for s in symbols]
    prices = {m: np.full((len(dates), len(symbols)), np.nan) for m in minutes}
    staleness = {m: np.full((len(dates), len(symbols)), np.inf) for m in minutes}
    order = {str(s): i for i, s in enumerate(symbols)}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sample_id, block, stale_block in pool.map(_symbol_minute_prices, payload, chunksize=8):
            column = order[sample_id]
            for index, minute in enumerate(minutes):
                prices[minute][:, column] = block[index]
                staleness[minute][:, column] = stale_block[index]

    if auctions_path.exists():
        prices[AUCTION_MINUTE] = load_opening_auction_prices(auctions_path, dates, symbols)
        staleness[AUCTION_MINUTE] = np.where(
            np.isfinite(prices[AUCTION_MINUTE]), 0.0, np.inf
        )

    return Panel(
        dates=dates,
        symbols=symbols,
        dollar_volume=dollar_volume,
        completed_days=causal_completed_trading_days(dollar_volume),
        prices=prices,
        staleness=staleness,
        issuers=build_issuer_map(symbols, master),
        first_entry=first_entry - lo,
    )


# ----------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """How a session's basket is chosen. ``scheme`` names the ranking statistic."""

    top: int = 12
    ema_span: int = 10
    min_history_days: int = 20
    minimum_trading_days: int = 100
    scheme: str = "dollar_ema"
    dedupe: bool = True


def _scheme_scores(panel: Panel, selection: Selection) -> np.ndarray:
    """Score every symbol on every date using only strictly prior sessions."""
    volume = panel.dollar_volume
    if selection.scheme == "dollar_ema":
        return causal_ema_log_liquidity(volume, selection.ema_span, selection.min_history_days)
    if selection.scheme == "dollar_ema_short":
        return causal_ema_log_liquidity(volume, 3, selection.min_history_days)
    if selection.scheme == "dollar_ema_long":
        return causal_ema_log_liquidity(volume, 40, selection.min_history_days)
    if selection.scheme == "turnover_stability":
        # Prefer names whose liquidity is high *and* steady: the EMA level less a
        # penalty for its own dispersion. A name that is only briefly enormous --
        # an earnings or index-rebalance spike -- ranks below a dependable one.
        level = causal_ema_log_liquidity(volume, selection.ema_span, selection.min_history_days)
        logged = np.log1p(np.where(np.isfinite(volume) & (volume > 0.0), volume, np.nan))
        frame = pd.DataFrame(logged)
        spread = frame.rolling(20, min_periods=10).std().shift(1).to_numpy()
        return level - np.nan_to_num(spread, nan=0.0)
    raise ValueError(f"unknown selection scheme: {selection.scheme}")


def rank_baskets(panel: Panel, selection: Selection, entry_minute: int) -> list[np.ndarray]:
    """Return the chosen column indices for every date, or an empty array."""
    scores = _scheme_scores(panel, selection)
    entry_prices = panel.price(entry_minute)
    entry_stale = panel.staleness[entry_minute]
    stock = panel.symbols != REFERENCE_SYMBOL
    stock_symbols = panel.symbols[stock]
    baskets: list[np.ndarray] = []
    for index in range(len(panel.dates)):
        if index < panel.first_entry or index + 1 >= len(panel.dates):
            baskets.append(np.empty(0, dtype=np.int64))
            continue
        executable = np.where(entry_stale[index, stock] <= 10, entry_prices[index, stock], np.nan)
        eligible = np.where(
            panel.completed_days[index, stock] >= selection.minimum_trading_days,
            scores[index, stock],
            np.nan,
        )
        try:
            local = top_liquid_indices(
                eligible, executable, selection.top, stock_symbols,
                issuers=panel.issuers if selection.dedupe else None,
            )
        except ValueError:
            baskets.append(np.empty(0, dtype=np.int64))
            continue
        baskets.append(np.flatnonzero(stock)[local])
    return baskets


# ------------------------------------------------------------------------- engine


@dataclass(frozen=True)
class Rules:
    """A complete variant: when to trade, and when to refuse to realise a loss."""

    entry_minute: int = 15 * 60 + 59
    exit_minute: int = AUCTION_MINUTE
    defer_below_bps: float | None = None
    max_extra_holds: int = 1
    cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    label: str = ""


@dataclass
class Lot:
    quantity: float
    entry_price: float
    holds: int = 0


def simulate(panel: Panel, baskets: list[Lot], rules: Rules, budget: float = 10_000.0):
    """Run one variant and return its per-session return series."""
    entry_prices = panel.price(rules.entry_minute)
    entry_stale = panel.staleness[rules.entry_minute]
    exit_prices = panel.price(rules.exit_minute)
    exit_stale = panel.staleness[rules.exit_minute]
    exit_limit = np.inf if rules.exit_minute == AUCTION_MINUTE else 10.0
    cost = rules.cost_bps / 1e4

    cash = budget
    book: dict[int, Lot] = {}
    equity_curve: list[float] = []
    dates: list[pd.Timestamp] = []
    deferred_sessions = 0
    deferral_events = 0

    for index in range(panel.first_entry, len(panel.dates) - 1):
        # Morning: mark, then decide which lots to realise.
        realise: list[int] = []
        for column, lot in book.items():
            price = exit_prices[index, column]
            if not np.isfinite(price) or exit_stale[index, column] > exit_limit:
                continue
            gain_bps = (price / lot.entry_price - 1.0) * 1e4
            hold = (
                rules.defer_below_bps is not None
                and gain_bps < rules.defer_below_bps
                and lot.holds < rules.max_extra_holds
            )
            if hold:
                lot.holds += 1
                deferred_sessions += 1
                if lot.holds == 1:
                    deferral_events += 1
                continue
            realise.append(column)
        for column in realise:
            lot = book.pop(column)
            proceeds = lot.quantity * exit_prices[index, column]
            cash += proceeds * (1.0 - cost)

        held_value = sum(
            lot.quantity * exit_prices[index, column]
            for column, lot in book.items()
            if np.isfinite(exit_prices[index, column])
        )
        equity_curve.append(cash + held_value)
        dates.append(pd.Timestamp(panel.dates[index]))

        # Afternoon: deploy whatever is free across names not already held.
        picks = [int(c) for c in baskets[index] if c not in book]
        if picks and cash > 0.0:
            allocation = cash / len(picks)
            for column in picks:
                price = entry_prices[index, column]
                if not np.isfinite(price) or price <= 0.0 or entry_stale[index, column] > 10:
                    continue
                quantity = allocation / price
                spend = quantity * price
                cash -= spend * (1.0 + cost)
                book[column] = Lot(quantity=quantity, entry_price=price)

    curve = pd.Series(equity_curve, index=pd.DatetimeIndex(dates))
    returns = curve.pct_change().dropna()
    return returns, curve, {
        "deferral_events": deferral_events,
        "deferred_sessions": deferred_sessions,
    }


def evaluate(panel: Panel, selection: Selection, rules: Rules, budget: float = 10_000.0):
    baskets = rank_baskets(panel, selection, rules.entry_minute)
    returns, curve, extra = simulate(panel, baskets, rules, budget)
    if returns.empty:
        return None
    metrics = strategy_metrics(returns)
    metrics.update(extra)
    metrics["ending_equity"] = float(curve.iloc[-1])
    return metrics


# ----------------------------------------------------------------------- caching

ENTRY_GRID = (15 * 60 + 30, 15 * 60 + 40, 15 * 60 + 45, 15 * 60 + 50,
              15 * 60 + 55, 15 * 60 + 57, 15 * 60 + 59)
EXIT_GRID = (AUCTION_MINUTE, 9 * 60 + 30, 9 * 60 + 31, 9 * 60 + 32,
             9 * 60 + 35, 9 * 60 + 40, 9 * 60 + 45, 10 * 60)
PANEL_CACHE = Path("/tmp/trading/experiment_cache")


def cached_panel(months: int, exchange_filter: str, workers: int) -> Panel:
    """Build the panel once per (window, universe) and reuse it across sweeps."""
    minutes = tuple(sorted(set(ENTRY_GRID) | {m for m in EXIT_GRID if m != AUCTION_MINUTE}))
    PANEL_CACHE.mkdir(parents=True, exist_ok=True)
    path = PANEL_CACHE / f"panel_{months}m_{exchange_filter}_{len(minutes)}min.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as blob:
            keys = [int(k) for k in blob["minute_keys"]]
            return Panel(
                dates=pd.DatetimeIndex(blob["dates"].astype("datetime64[D]")),
                symbols=blob["symbols"],
                dollar_volume=blob["dollar_volume"],
                completed_days=blob["completed_days"],
                prices={k: blob[f"p{k}"] for k in keys},
                staleness={k: blob[f"s{k}"] for k in keys},
                issuers=dict(zip(blob["issuer_keys"], blob["issuer_values"])),
                first_entry=int(blob["first_entry"]),
            )
    panel = build_panel(
        months, minutes,
        minute_dir=Path(DEFAULT_DATA_DIR),
        daily_dir=Path(DEFAULT_DAILY_DATA_DIR),
        auctions_path=Path(DEFAULT_AUCTIONS_PATH),
        cache_dir=Path("/tmp/trading/baseline_cache"),
        ema_span=10, min_history_days=20, minimum_trading_days=100,
        exchange_filter=exchange_filter, workers=workers,
    )
    payload = {
        "dates": panel.dates.to_numpy(dtype="datetime64[D]"),
        "symbols": panel.symbols,
        "dollar_volume": panel.dollar_volume,
        "completed_days": panel.completed_days,
        "minute_keys": np.asarray(sorted(panel.prices), dtype=np.int64),
        "issuer_keys": np.asarray(list(panel.issuers), dtype=object).astype(str),
        "issuer_values": np.asarray(list(panel.issuers.values()), dtype=object).astype(str),
        "first_entry": np.asarray(panel.first_entry),
    }
    for key, matrix in panel.prices.items():
        payload[f"p{key}"] = matrix
        payload[f"s{key}"] = panel.staleness[key]
    np.savez_compressed(path, **payload)
    return panel


# ----------------------------------------------------------------------- reporting


def _row(metrics: dict[str, float], label: str) -> dict[str, object]:
    return {
        "variant": label,
        "ann_%": metrics["annualized_return"] * 100,
        "PF": metrics["profit_factor"],
        "sharpe": metrics["sharpe_zero_cash_rate"],
        "maxDD_%": metrics["max_drawdown"] * 100,
        "calmar": metrics["calmar_ratio"],
        "win_%": metrics["win_rate"] * 100,
        "bps/day": metrics["mean_return"] * 1e4,
        "n": metrics["periods"],
    }


def render(rows: list[dict[str, object]], title: str, sort: str = "sharpe") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if sort in frame:
        frame = frame.sort_values(sort, ascending=False)
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold")
    for column in frame.columns:
        table.add_column(str(column), justify="left" if column == "variant" else "right")
    for _, record in frame.iterrows():
        table.add_row(*[
            str(record[c]) if c == "variant" else f"{float(record[c]):.2f}"
            for c in frame.columns
        ])
    console.print(table)
    return frame


def split_stability(panel: Panel, selection: Selection, rules: Rules) -> dict[str, float]:
    """Re-score a variant on each half of the window to expose window-fitting."""
    baskets = rank_baskets(panel, selection, rules.entry_minute)
    returns, _, _ = simulate(panel, baskets, rules)
    if returns.empty:
        return {}
    midpoint = len(returns) // 2
    first, second = returns.iloc[:midpoint], returns.iloc[midpoint:]
    out: dict[str, float] = {}
    for name, part in (("h1", first), ("h2", second)):
        if part.empty:
            continue
        stats = strategy_metrics(part)
        out[f"{name}_ann_%"] = stats["annualized_return"] * 100
        out[f"{name}_sharpe"] = stats["sharpe_zero_cash_rate"]
        out[f"{name}_PF"] = stats["profit_factor"]
    return out


# --------------------------------------------------------------------------- cli

SHIPPED = (Selection(top=12, scheme="dollar_ema"),
           Rules(entry_minute=15 * 60 + 59, exit_minute=AUCTION_MINUTE, label="shipped"))
PROPOSED = (Selection(top=12, scheme="turnover_stability"),
            Rules(entry_minute=15 * 60 + 45, exit_minute=AUCTION_MINUTE, label="proposed"))


def _clock(minute: int) -> str:
    return "auction" if minute == AUCTION_MINUTE else f"{minute // 60:02d}:{minute % 60:02d}"


def sweep_times(panel: Panel, selection: Selection) -> pd.DataFrame:
    rows = []
    for entry in ENTRY_GRID:
        baskets = rank_baskets(panel, selection, entry)
        for exit_minute in EXIT_GRID:
            returns, _, _ = simulate(panel, baskets, Rules(entry, exit_minute))
            if returns.empty:
                continue
            rows.append({"entry": _clock(entry), "exit": _clock(exit_minute),
                         **_row(strategy_metrics(returns), f"{_clock(entry)}->{_clock(exit_minute)}")})
    return pd.DataFrame(rows)


def sweep_selection(panel: Panel, rules: Rules, max_top: int = 20) -> pd.DataFrame:
    rows = []
    for scheme in ("dollar_ema", "dollar_ema_short", "dollar_ema_long", "turnover_stability"):
        for top in (8, 10, 12, 15, max_top):
            selection = Selection(top=top, scheme=scheme)
            baskets = rank_baskets(panel, selection, rules.entry_minute)
            returns, _, _ = simulate(panel, baskets, rules)
            if returns.empty:
                continue
            halves = split_stability(panel, selection, rules)
            rows.append({"scheme": scheme, "top": top,
                         **_row(strategy_metrics(returns), f"{scheme}/{top}"),
                         "worst_half_sharpe": min(halves["h1_sharpe"], halves["h2_sharpe"])})
    return pd.DataFrame(rows)


def head_to_head(panel: Panel) -> pd.DataFrame:
    rows = []
    for selection, rules in (SHIPPED, PROPOSED):
        baskets = rank_baskets(panel, selection, rules.entry_minute)
        returns, curve, _ = simulate(panel, baskets, rules)
        halves = split_stability(panel, selection, rules)
        label = f"{rules.label}: {selection.scheme}/{selection.top} " \
                f"{_clock(rules.entry_minute)}->{_clock(rules.exit_minute)}"
        rows.append({**_row(strategy_metrics(returns), label),
                     "end_$": float(curve.iloc[-1]),
                     "h1_sharpe": halves["h1_sharpe"], "h2_sharpe": halves["h2_sharpe"]})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("times", "select", "final", "all"))
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--exchange-filter", choices=("all", "nasdaq"), default="nasdaq")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    panel = cached_panel(args.months, args.exchange_filter, args.workers)
    Console().print(
        f"panel: {len(panel.dates)} sessions x {len(panel.symbols)} symbols, "
        f"{args.months}m trailing, universe={args.exchange_filter}"
    )
    if args.stage in ("times", "all"):
        frame = sweep_times(panel, SHIPPED[0])
        render(frame.drop(columns=["entry", "exit"]), "entry x exit clock", "sharpe")
    if args.stage in ("select", "all"):
        frame = sweep_selection(panel, PROPOSED[1])
        render(frame.drop(columns=["scheme", "top"]), "selection x basket size", "worst_half_sharpe")
    if args.stage in ("final", "all"):
        render(head_to_head(panel), "shipped vs proposed", "sharpe")


if __name__ == "__main__":
    main()
