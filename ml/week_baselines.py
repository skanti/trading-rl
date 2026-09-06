"""Rule-based weekly baselines over a trailing window.

Three strategies priced on the same session grid, the same forward-filled prints,
and the same round-trip cost the learned policy pays:

* ``A`` long an equal-weighted Mag7 basket from Monday's open to Friday's close;
* ``B`` the same schedule on SPY alone;
* ``C`` SPY bought once at the start and sold once at the end.

A and B stand aside over weekends and holidays; C does not. Comparing B with C
is therefore what prices the weekend exposure the intraweek policy structurally
refuses to take.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .week_dataset import forward_filled_prices


MAG7 = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")
WEEKS_PER_YEAR = 52.0


def week_schedule(
    days: pd.DataFrame, date_from: pd.Timestamp, date_to: pd.Timestamp, complete_only: bool
) -> pd.DataFrame:
    """Entry and exit timestamps for every week in the window.

    Entry is the first session's 09:30 and exit the last session's 16:00, so a
    holiday-shortened week is entered and exited on the sessions it actually has
    rather than being padded or silently skipped.
    """
    calendar = days.loc[:, ["date", "sod_sec", "eod_sec"]].drop_duplicates()
    if calendar.date.duplicated().any():
        raise ValueError("symbols disagree about one or more session boundaries")
    calendar = calendar[calendar.date.between(date_from, date_to)]
    weekday = calendar.date.dt.dayofweek
    calendar = calendar.assign(week_start=calendar.date - pd.to_timedelta(weekday, unit="D"))
    weeks = calendar.groupby("week_start").agg(
        sessions=("date", "size"),
        entry_sec=("sod_sec", "min"),
        exit_sec=("eod_sec", "max"),
        first_session=("date", "min"),
        last_session=("date", "max"),
    ).reset_index()
    if complete_only:
        weeks = weeks[weeks.sessions.eq(5)]
    return weeks.sort_values("week_start").reset_index(drop=True)


def performance(
    returns: np.ndarray, years: float, time_in_market: float, round_trips: int
) -> dict:
    """Summarize a weekly net-return series.

    ``profit_factor`` uses positive versus negative net weekly P&L, matching the
    definition the trainer applies to its per-timestep series. Drawdown is the
    worst peak-to-trough loss of the compounded equity curve, as a fraction of
    the peak. A round trip is one entry plus one exit, so it counts as two
    position changes under the policy's ``trades`` metric.
    """
    if not returns.size:
        raise ValueError("no weeks to summarize")
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    compounded = float(equity[-1] - 1.0)
    volatility = float(returns.std(ddof=1)) if returns.size > 1 else float("nan")
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    return {
        "weeks": int(returns.size),
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "round_trips": int(round_trips),
        "position_changes": int(2 * round_trips),
        "total_return_compounded": compounded,
        "total_return_simple_sum": float(returns.sum()),
        "cagr": float((1.0 + compounded) ** (1.0 / years) - 1.0) if years > 0 else float("nan"),
        "mean_weekly": float(returns.mean()),
        "stdev_weekly": volatility,
        "annualized_volatility": volatility * np.sqrt(WEEKS_PER_YEAR),
        "sharpe_annualized": float(returns.mean() / volatility * np.sqrt(WEEKS_PER_YEAR))
        if volatility and np.isfinite(volatility)
        else float("nan"),
        "max_drawdown": float(((peak - equity) / peak).max()),
        "hit_rate": float((returns > 0).mean()),
        "best_week": float(returns.max()),
        "worst_week": float(returns.min()),
        "time_in_market": time_in_market,
    }


def run(
    config_path: str,
    symbols: tuple[str, ...],
    reference_symbol: str,
    months: int,
    transaction_cost: float | None,
    complete_only: bool,
) -> tuple[pd.DataFrame, dict]:
    cfg = OmegaConf.load(config_path)
    data_dir = str(cfg.data.data_dir)
    cost = float(
        cfg.model.get("transaction_cost", 0.0) if transaction_cost is None else transaction_cost
    )
    round_trip = 2.0 * cost

    days = pd.read_csv(str(cfg.data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    date_to = days.date.max()
    date_from = date_to - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    weeks = week_schedule(days, date_from, date_to, complete_only)
    if not len(weeks):
        raise ValueError(f"no weeks between {date_from.date()} and {date_to.date()}")

    tradable = set(
        zip(days[days.is_tradable.astype(bool)].sample_id, days[days.is_tradable.astype(bool)].date)
    )
    entry = weeks.entry_sec.to_numpy(dtype=np.int64)
    exit_ = weeks.exit_sec.to_numpy(dtype=np.int64)
    frame = weeks.loc[:, ["week_start", "sessions", "first_session", "last_session"]].copy()

    legs: list[np.ndarray] = []
    for symbol in symbols:
        missing = [
            week
            for week, first, last in zip(weeks.week_start, weeks.first_session, weeks.last_session)
            if (symbol, first) not in tradable or (symbol, last) not in tradable
        ]
        if missing:
            raise ValueError(f"{symbol} is not tradable on {len(missing)} week boundaries")
        gross = forward_filled_prices(data_dir, symbol, exit_) / forward_filled_prices(
            data_dir, symbol, entry
        ) - 1.0
        frame[f"{symbol}_gross"] = gross
        legs.append(gross)

    # Every leg is a full round trip, so an equal-weighted basket pays one round
    # trip of cost in total rather than one per name.
    basket = np.mean(np.stack(legs, axis=0), axis=0)
    frame["A_mag7_weekly"] = basket - round_trip

    reference_entry = forward_filled_prices(data_dir, reference_symbol, entry)
    reference_exit = forward_filled_prices(data_dir, reference_symbol, exit_)
    frame["B_spy_weekly"] = reference_exit / reference_entry - 1.0 - round_trip

    # Held continuously, so each week runs from the previous exit; the product of
    # the series is exactly the single round trip from first entry to last exit.
    held = np.empty_like(reference_exit)
    held[0] = reference_exit[0] / reference_entry[0] - 1.0
    held[1:] = reference_exit[1:] / reference_exit[:-1] - 1.0
    frame["C_spy_hold"] = held
    frame.loc[0, "C_spy_hold"] -= cost
    frame.loc[len(frame) - 1, "C_spy_hold"] -= cost

    span_years = float((exit_[-1] - entry[0]) / (365.25 * 86_400))
    open_seconds = float((exit_ - entry).sum())
    summary = {
        "config": str(config_path),
        "window": f"{weeks.first_session.iloc[0].date()} to {weeks.last_session.iloc[-1].date()}",
        "span_years": span_years,
        "transaction_cost": cost,
        "round_trip_cost": round_trip,
        "complete_weeks_only": complete_only,
        "short_weeks": int((weeks.sessions != 5).sum()),
        "strategies": {
            "A_mag7_weekly": performance(
                frame.A_mag7_weekly.to_numpy(),
                span_years,
                open_seconds / (exit_[-1] - entry[0]),
                len(weeks) * len(symbols),
            ),
            "B_spy_weekly": performance(
                frame.B_spy_weekly.to_numpy(),
                span_years,
                open_seconds / (exit_[-1] - entry[0]),
                len(weeks),
            ),
            "C_spy_hold": performance(frame.C_spy_hold.to_numpy(), span_years, 1.0, 1),
        },
    }
    return frame, summary


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)
parser.add_argument("--symbols", default=",".join(MAG7))
parser.add_argument("--reference_symbol", default="SPY")
parser.add_argument("--months", type=int, default=12)
parser.add_argument("--transaction_cost", type=float, default=None)
parser.add_argument("--complete_weeks_only", action="store_true")
parser.add_argument("--output_path", default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    frame, summary = run(
        args.config_path,
        tuple(s.strip() for s in args.symbols.split(",") if s.strip()),
        args.reference_symbol,
        args.months,
        args.transaction_cost,
        args.complete_weeks_only,
    )
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2, default=str))
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, indent=2, default=str))
        frame.to_csv(Path(args.output_path).with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Session-level baselines, aggregated onto the same weekly buckets so every
# strategy shares one drawdown, Sharpe, and profit-factor convention.
# ---------------------------------------------------------------------------


def session_table(days: pd.DataFrame, date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    """One row per market session, with a week label and a lookback margin."""
    calendar = days.loc[:, ["date", "sod_sec", "eod_sec"]].drop_duplicates()
    calendar = calendar[calendar.date.le(date_to)].sort_values("date").reset_index(drop=True)
    weekday = calendar.date.dt.dayofweek
    calendar["week_start"] = calendar.date - pd.to_timedelta(weekday, unit="D")
    calendar["in_window"] = calendar.date.ge(date_from)
    return calendar


def weekly_from_sessions(
    week_labels: np.ndarray, session_returns: np.ndarray, weeks: np.ndarray
) -> np.ndarray:
    """Compound per-session net returns inside each week bucket."""
    out = np.zeros(weeks.size, dtype=np.float64)
    for index, week in enumerate(weeks):
        selected = session_returns[week_labels == week]
        out[index] = np.prod(1.0 + selected) - 1.0 if selected.size else 0.0
    return out


def extended_baselines(
    days: pd.DataFrame,
    data_dir: str,
    reference_symbol: str,
    symbols: tuple[str, ...],
    weeks: pd.DataFrame,
    cost: float,
    trend_window: int = 200,
    vol_window: int = 60,
    target_volatility: float = 0.12,
    max_leverage: float = 2.0,
) -> dict[str, np.ndarray]:
    """Weekly net-return series for the session-level and filtered rules."""
    date_from = weeks.first_session.min()
    date_to = weeks.last_session.max()
    sessions = session_table(days, date_from, date_to)
    opens = forward_filled_prices(data_dir, reference_symbol, sessions.sod_sec.to_numpy(np.int64))
    closes = forward_filled_prices(data_dir, reference_symbol, sessions.eod_sec.to_numpy(np.int64))

    window = sessions.in_window.to_numpy()
    labels = sessions.week_start.to_numpy()[window]
    week_index = weeks.week_start.to_numpy()

    # D/E: the two halves of buy-and-hold. Held together and cost-free they
    # reproduce it exactly, so this is a decomposition rather than a new bet.
    intraday = closes / opens - 1.0
    overnight = np.empty_like(closes)
    overnight[0] = np.nan
    overnight[1:] = opens[1:] / closes[:-1] - 1.0
    round_trip = 2.0 * cost

    # F: causal trend filter. The comparison uses the close before the week
    # opens and a moving average ending at that same close.
    frame = pd.Series(closes)
    average = frame.rolling(trend_window).mean().to_numpy()
    daily_return = np.concatenate(([np.nan], closes[1:] / closes[:-1] - 1.0))
    realized = (
        pd.Series(daily_return).rolling(vol_window).std().to_numpy() * np.sqrt(252.0)
    )

    def weekly_exposure(values: np.ndarray) -> np.ndarray:
        """Value from the last session strictly before each week's first session."""
        first = weeks.first_session.to_numpy()
        positions = np.searchsorted(sessions.date.to_numpy(), first, side="left") - 1
        if (positions < 0).any():
            raise ValueError("not enough history before the first evaluated week")
        return values[positions]

    trend_signal = weekly_exposure(closes) > weekly_exposure(average)
    trend_exposure = trend_signal.astype(np.float64)
    vol_estimate = weekly_exposure(realized)
    vol_exposure = np.clip(target_volatility / vol_estimate, 0.0, max_leverage)

    # C-style continuous holding, scaled by the weekly exposure decision.
    held = np.empty(len(weeks), dtype=np.float64)
    entry = weeks.entry_sec.to_numpy(np.int64)
    exit_ = weeks.exit_sec.to_numpy(np.int64)
    reference_entry = forward_filled_prices(data_dir, reference_symbol, entry)
    reference_exit = forward_filled_prices(data_dir, reference_symbol, exit_)
    held[0] = reference_exit[0] / reference_entry[0] - 1.0
    held[1:] = reference_exit[1:] / reference_exit[:-1] - 1.0

    def scaled(exposure: np.ndarray) -> np.ndarray:
        turnover = np.abs(np.diff(np.concatenate(([0.0], exposure, [0.0]))))
        charges = turnover[:-1] + np.concatenate((np.zeros(len(exposure) - 1), turnover[-1:]))
        return exposure * held - charges * cost

    # Equal-weighted and rebalanced weekly, exactly as the A basket is, so the
    # only thing separating H from A is that H stays invested over weekends.
    # Averaging prices instead of returns would build a price-weighted index and
    # make that comparison meaningless.
    per_leg = []
    for symbol in symbols:
        leg_entry = forward_filled_prices(data_dir, symbol, entry)
        leg_exit = forward_filled_prices(data_dir, symbol, exit_)
        leg = np.empty(len(weeks), dtype=np.float64)
        leg[0] = leg_exit[0] / leg_entry[0] - 1.0
        leg[1:] = leg_exit[1:] / leg_exit[:-1] - 1.0
        per_leg.append(leg)
    basket_held = np.mean(np.stack(per_leg, axis=0), axis=0)

    result = {
        "D_spy_intraday_only": weekly_from_sessions(
            labels, intraday[window] - round_trip, week_index
        ),
        "E_spy_overnight_only": weekly_from_sessions(
            labels, np.nan_to_num(overnight[window]) - round_trip, week_index
        ),
        "F_spy_trend_filter": scaled(trend_exposure),
        "G_spy_vol_target": scaled(vol_exposure),
        "H_mag7_hold": basket_held.copy(),
        "I_spy_hold_2x": 2.0 * held,
    }
    result["H_mag7_hold"][0] -= cost
    result["H_mag7_hold"][-1] -= cost
    result["I_spy_hold_2x"][0] -= 2.0 * cost
    result["I_spy_hold_2x"][-1] -= 2.0 * cost
    result["_exposure_trend"] = trend_exposure
    result["_exposure_vol"] = vol_exposure
    return result
