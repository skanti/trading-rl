"""Build a symbol-pair/day index for relative-value training.

Two things make this different from :mod:`prepare_days`.

First, symbols are not tick-aligned. Coverage of extended hours differs from
name to name, so the two legs are joined on their timestamps rather than on
their row indices. Rows here therefore record *seconds*, which mean the same
thing to both legs, instead of per-symbol offsets.

Second, pairs are chosen from a trailing window that ends before the session
being traded ever opens. Ranking pairs with data from the day they are traded
on is the standard way to leak the answer into a pairs backtest.

The output deliberately mixes strongly co-moving pairs with randomly drawn
ones. The policy is never told which is which and never sees symbol identity,
so the only way it can learn to stand aside on an unrelated pair is to have
been shown unrelated pairs during training.

Symbols are partitioned into a training universe and a held-out universe before
any pair is formed, and pairs are only ever drawn inside one universe. Holding
symbols out after pairing does not work: a pair shares each of its legs with
many other pairs, so a name excluded from validation pairs still reaches
training inside a different pairing. Selecting within a partition also keeps
both splits well populated, which picking pairs whose *both* legs happen to be
held out does not - that leaves only a small fraction of the candidates.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from tqdm import tqdm

from prepare_days import FULL_SESSION_INTERVALS, RTH_CLOSE_MINUTE, RTH_OPEN_MINUTE


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("PAIRS")

PAIR_COLUMNS = (
    "sample_id_a",
    "sample_id_b",
    "date",
    "sod_sec",
    "eod_sec",
    "context_correlation",
    "selection",
    "universe",
)


def scan_symbol(
    npy_path: Path, anno: str, min_sec: int | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Complete regular sessions plus regular-session closes for one symbol.

    Both outputs come from a single pass. Sessions are grouped by locating the
    day boundaries of the already-sorted timestamps rather than by testing each
    day against every tick, which would be quadratic in the length of a
    multi-year file.
    """
    data = np.load(npy_path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{npy_path} must contain [seconds, price_mills, volume]")
    empty = pd.DataFrame(columns=("sample_id", "date", "sod_sec", "eod_sec"))
    secs = np.asarray(data[:, 0])
    if secs.size < 2 or not np.all(secs[:-1] < secs[1:]):
        return empty, np.empty(0, dtype=np.int64), np.empty(0)
    if min_sec is not None:
        offset = int(np.searchsorted(secs, min_sec, side="left"))
        if offset >= secs.size - 1:
            return empty, np.empty(0, dtype=np.int64), np.empty(0)
        data = data[offset:]
        secs = secs[offset:]

    dt = pd.to_datetime(secs, unit="s", origin=anno, utc=True).tz_convert("US/Eastern")
    minutes = np.asarray(dt.hour * 60 + dt.minute)
    in_session = (
        (minutes >= RTH_OPEN_MINUTE)
        & (minutes <= RTH_CLOSE_MINUTE)
        & (np.asarray(dt.second) == 0)
    )
    session_index = np.flatnonzero(in_session)
    if session_index.size == 0:
        return empty, np.empty(0, dtype=np.int64), np.empty(0)

    session_secs = secs[session_index].astype(np.int64)
    session_dates = np.asarray(dt.date)[session_index]
    session_minutes = minutes[session_index]
    closes = np.asarray(data[:, 1])[session_index].astype(np.float64) / 1000.0

    # Timestamps are sorted, so each calendar day is one contiguous run.
    boundaries = np.flatnonzero(session_dates[1:] != session_dates[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [session_dates.size]))

    rows = []
    for start, stop in zip(starts, stops):
        if stop - start != FULL_SESSION_INTERVALS + 1:
            continue
        if session_minutes[start] != RTH_OPEN_MINUTE or session_minutes[stop - 1] != RTH_CLOSE_MINUTE:
            continue
        day_secs = session_secs[start:stop]
        if not np.all(np.diff(day_secs) == 60):
            continue
        rows.append(
            (npy_path.stem, session_dates[start].isoformat(), int(day_secs[0]), int(day_secs[-1]))
        )
    table = pd.DataFrame(rows, columns=("sample_id", "date", "sod_sec", "eod_sec"))
    return table, session_secs, closes


def session_table(npy_path: Path, anno: str, min_sec: int | None = None) -> pd.DataFrame:
    """Complete regular sessions for one symbol, keyed by seconds."""
    return scan_symbol(npy_path, anno, min_sec)[0]


def trailing_correlations(
    returns: np.ndarray, valid: np.ndarray, min_observations: int
) -> np.ndarray:
    """Pairwise correlation of aligned return columns, ignoring absent names.

    ``returns`` is (ticks, symbols) and ``valid`` marks columns with complete
    coverage over the window. Columns without coverage come back as NaN so they
    are never selected.
    """
    output = np.full((returns.shape[1], returns.shape[1]), np.nan)
    if valid.sum() < 2 or returns.shape[0] < min_observations:
        return output
    block = returns[:, valid]
    block = block - block.mean(axis=0, keepdims=True)
    scale = np.sqrt((block**2).sum(axis=0))
    scale[scale == 0] = np.nan
    normalized = block / scale
    correlation = normalized.T @ normalized
    index = np.flatnonzero(valid)
    output[np.ix_(index, index)] = correlation
    return output


def build_pairs(
    data_dir: str,
    output_path: str,
    anno: str = "2010-01-01",
    window_size: int = 4096,
    rollout_size: int = FULL_SESSION_INTERVALS,
    date_from: str | None = None,
    date_to: str | None = None,
    correlated_per_day: int = 40,
    random_per_day: int = 25,
    min_correlation: float = 0.5,
    holdout_stride: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("*.npy"))
    if not paths:
        raise ValueError(f"no .npy files found in {data_dir}")
    if correlated_per_day < 0 or random_per_day < 0 or correlated_per_day + random_per_day < 1:
        raise ValueError("at least one pair per day must be requested")
    if holdout_stride < 2:
        raise ValueError("holdout_stride must be at least two, or no symbols remain for training")

    # Only history the trailing correlation window can reach is worth scanning.
    min_sec: int | None = None
    if date_from:
        lookback_days = 8 + window_size // FULL_SESSION_INTERVALS
        anchor = pd.Timestamp(date_from) - pd.Timedelta(days=2 * lookback_days + 14)
        min_sec = int((anchor - pd.Timestamp(anno)).total_seconds())

    sessions = []
    closes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in tqdm(paths, desc="index sessions"):
        table, session_secs, session_closes = scan_symbol(path, anno, min_sec)
        if table.empty:
            continue
        sessions.append(table)
        closes[path.stem] = (session_secs, session_closes)
    if not sessions:
        raise ValueError("no complete regular sessions found")
    sessions = pd.concat(sessions, ignore_index=True)
    sessions["date"] = pd.to_datetime(sessions["date"], format="%Y-%m-%d")
    if date_from:
        sessions = sessions[sessions["date"] >= pd.Timestamp(date_from)]
    if date_to:
        sessions = sessions[sessions["date"] <= pd.Timestamp(date_to)]
    if sessions.empty:
        raise ValueError("no complete sessions inside the requested date range")

    symbols = sorted(closes)
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    grid = np.unique(np.concatenate([secs for secs, _ in closes.values()]))
    prices = np.full((grid.size, len(symbols)), np.nan)
    for symbol, (secs, close) in closes.items():
        prices[np.searchsorted(grid, secs), symbol_index[symbol]] = close
    log_prices = np.log(prices)
    logger.info("Built regular-session grid, ticks=%d, symbols=%d", grid.size, len(symbols))

    rng = np.random.default_rng(seed)
    holdout_symbols = set(symbols[::holdout_stride])
    universes = {
        "holdout": np.array([symbol_index[s] for s in symbols if s in holdout_symbols]),
        "train": np.array([symbol_index[s] for s in symbols if s not in holdout_symbols]),
    }
    logger.info(
        "Partitioned symbols, train=%d, holdout=%d",
        universes["train"].size,
        universes["holdout"].size,
    )

    rows: list[tuple] = []
    upper = np.triu(np.ones((len(symbols), len(symbols)), dtype=bool), k=1)
    for date, group in tqdm(sessions.groupby("date"), desc="select pairs"):
        available = sorted(set(group.sample_id) & set(symbol_index))
        if len(available) < 2:
            continue
        sod_sec = int(group.sod_sec.iloc[0])
        eod_sec = int(group.eod_sec.iloc[0])
        if group.sod_sec.nunique() != 1 or group.eod_sec.nunique() != 1:
            raise ValueError(f"inconsistent session bounds on {date.date()}")

        # Strictly before the traded session opens.
        stop = int(np.searchsorted(grid, sod_sec))
        start = stop - int(window_size)
        if start < 0:
            continue
        returns = np.diff(log_prices[start:stop], axis=0)
        covered = np.zeros(len(symbols), dtype=bool)
        covered[[symbol_index[s] for s in available]] = True
        covered &= np.isfinite(returns).all(axis=0)
        correlation = trailing_correlations(returns, covered, min_observations=window_size // 2)

        for universe, members in universes.items():
            eligible = np.zeros(len(symbols), dtype=bool)
            eligible[members] = True
            eligible &= covered
            candidates = np.flatnonzero(eligible)
            if candidates.size < 2:
                continue
            within = upper & eligible[:, None] & eligible[None, :]
            scored = np.where(within, correlation, np.nan)

            selection: dict[tuple[int, int], str] = {}
            order = np.argsort(-np.nan_to_num(scored.ravel(), nan=-np.inf))
            for position in order[: correlated_per_day * 4]:
                i, j = divmod(int(position), len(symbols))
                if not np.isfinite(scored[i, j]) or scored[i, j] < min_correlation:
                    break
                selection[(i, j)] = "correlated"
                if len(selection) >= correlated_per_day:
                    break

            for _ in range(random_per_day * 4):
                if sum(1 for v in selection.values() if v == "random") >= random_per_day:
                    break
                i, j = rng.choice(candidates, size=2, replace=False)
                i, j = (int(i), int(j)) if i < j else (int(j), int(i))
                if (i, j) in selection or not np.isfinite(scored[i, j]):
                    continue
                selection[(i, j)] = "random"

            for (i, j), kind in selection.items():
                rows.append(
                    (
                        symbols[i],
                        symbols[j],
                        date.strftime("%Y-%m-%d"),
                        sod_sec,
                        eod_sec,
                        float(scored[i, j]),
                        kind,
                        universe,
                    )
                )

    pairs = pd.DataFrame(rows, columns=list(PAIR_COLUMNS))
    if pairs.empty:
        raise ValueError("no pair-days met the requested context and rollout sizes")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output, index=False)
    logger.info(
        "Wrote pair-days, rows=%d, correlated=%d, random=%d, dates=%d, train=%d, holdout=%d",
        len(pairs),
        int((pairs.selection == "correlated").sum()),
        int((pairs.selection == "random").sum()),
        pairs.date.nunique(),
        int((pairs.universe == "train").sum()),
        int((pairs.universe == "holdout").sum()),
    )
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--anno", default="2010-01-01")
    parser.add_argument("--window_size", type=int, default=4096)
    parser.add_argument("--rollout_size", type=int, default=FULL_SESSION_INTERVALS)
    parser.add_argument("--date_from", default=None)
    parser.add_argument("--date_to", default=None)
    parser.add_argument("--correlated_per_day", type=int, default=40)
    parser.add_argument("--random_per_day", type=int, default=25)
    parser.add_argument("--min_correlation", type=float, default=0.5)
    parser.add_argument("--holdout_stride", type=int, default=5,
                        help="every Nth symbol is reserved for validation pairs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = build_pairs(
        data_dir=args.data_dir,
        output_path=args.output_path,
        anno=args.anno,
        window_size=args.window_size,
        rollout_size=args.rollout_size,
        date_from=args.date_from,
        date_to=args.date_to,
        correlated_per_day=args.correlated_per_day,
        random_per_day=args.random_per_day,
        min_correlation=args.min_correlation,
        holdout_stride=args.holdout_stride,
        seed=args.seed,
    )
    print(f"wrote {len(result)} pair-days to {args.output_path}")
