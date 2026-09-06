"""Greedy intraweek backtest for a trained checkpoint on a chosen symbol set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from . import model, utils
from .train import market_rewards, performance_metrics
from .week import (
    WEEK_FEATURE_NAMES,
    WEEK_SCALAR_NAMES,
    collect_week_rollout,
    validate_week_hours,
)
from .week_dataset import (
    WeekReferenceDataset,
    read_universe,
    ticks_per_session,
    week_targets,
)


MAG7 = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")


def hold_rewards(
    prices: torch.Tensor, context_ticks: int, steps: int, exposure: float, cost: float
) -> torch.Tensor:
    """Reward of holding a constant position for the whole week.

    Uses the same reward function as the policy, so entry and liquidation costs
    are charged identically and the two numbers are directly comparable.
    """
    positions = torch.full(
        (prices.shape[0], steps), int(exposure), dtype=torch.long, device=prices.device
    )
    rewards, _ = market_rewards(
        positions,
        prices[:, context_ticks : context_ticks + steps],
        prices[:, context_ticks + 1 : context_ticks + steps + 1],
        transaction_cost=cost,
        risk_penalty=0.0,
    )
    return rewards


@torch.no_grad()
def resolve_symbols(cfg, symbols: str | None, top: int | None) -> tuple[str, ...]:
    """Explicit list, an ``--top N`` slice, or whatever training validates on."""
    reference = str(cfg.data.reference_symbol)
    if symbols:
        return tuple(s.strip() for s in symbols.split(",") if s.strip())
    universe = cfg.data.get("val_universe", None)
    if top is not None:
        path = str(universe.path) if universe is not None else "../data/tickers_all.txt"
        return read_universe(path, top, reference)
    if universe is not None:
        return read_universe(str(universe.path), universe.get("size", None), reference)
    return MAG7


def evaluate(
    config_path: str,
    checkpoint_path: str | None,
    symbols: tuple[str, ...],
    device: str,
    date_from: str | None = None,
    batch_size: int = 8,
    date_to: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = OmegaConf.load(config_path)
    checkpoint_path = checkpoint_path or utils.load_most_recent_checkpoint(
        str(cfg.general.experiment_dir)
    )
    if not checkpoint_path:
        raise ValueError(f"no checkpoint found under {cfg.general.experiment_dir}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(state.get("feature_names", ())) != WEEK_FEATURE_NAMES or tuple(
        state.get("scalar_names", ())
    ) != WEEK_SCALAR_NAMES:
        raise ValueError("checkpoint is not an intraweek shifted-window MLP checkpoint")
    mlp_config = state["config"]["model"]["mlp"]
    actor = model.TradingActor(**mlp_config).to(device)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()

    context_ticks = int(cfg.data.context_ticks)
    steps = int(cfg.data.rollout_size)
    tick_minutes = int(cfg.data.tick_minutes)
    session_ticks = ticks_per_session(tick_minutes)
    cost = float(cfg.model.get("transaction_cost", 0.0))
    reference_symbol = str(cfg.data.reference_symbol)

    days = pd.read_csv(str(cfg.data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    date_from = pd.Timestamp(date_from or str(cfg.data.date_val))
    targets = week_targets(days)
    selected = targets.week_start.ge(date_from) & targets.sample_id.isin(symbols)
    if date_to is not None:
        # Inclusive of the week that starts on ``date_to``, so a caller names
        # the first and last Monday of the window rather than a boundary.
        selected &= targets.week_start.le(pd.Timestamp(date_to))
    targets = targets[selected].reset_index(drop=True)
    if not len(targets):
        raise ValueError(f"no weeks on or after {date_from.date()} for {sorted(symbols)}")
    absent = set(symbols).difference(targets.sample_id)
    if absent:
        print(f"note: {len(absent)} symbols have no week in the period: {sorted(absent)}")

    dataset = WeekReferenceDataset(
        days=days,
        data_dir=str(cfg.data.data_dir),
        reference_symbol=reference_symbol,
        context_days=int(cfg.data.context_days),
        tick_minutes=tick_minutes,
        targets=targets,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    rows: list[dict] = []
    all_rewards: list[torch.Tensor] = []
    for batch in loader:
        prices = batch["prices"].to(device=device, dtype=torch.float32)
        reference_prices = batch["reference_prices"].to(device=device, dtype=torch.float32)
        secs = batch["secs"].to(device=device, dtype=torch.long)
        validate_week_hours(
            secs, context_ticks, steps, tick_minutes, session_ticks, str(cfg.data.anno)
        )
        rollout = collect_week_rollout(
            actor,
            prices,
            reference_prices,
            context_ticks=context_ticks,
            rollout_size=steps,
            session_ticks=session_ticks,
            transaction_cost=cost,
            risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
            sampling="greedy",
            price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
        )
        all_rewards.append(rollout.rewards.cpu())
        long_hold = hold_rewards(prices, context_ticks, steps, 1.0, cost)
        short_hold = hold_rewards(prices, context_ticks, steps, -1.0, cost)
        boundaries = torch.arange(session_ticks - 1, steps, session_ticks, device=device)
        week_return = (
            prices[:, context_ticks + steps] / prices[:, context_ticks] - 1.0
        )
        spy_return = (
            reference_prices[:, context_ticks + steps] / reference_prices[:, context_ticks]
            - 1.0
        )
        equity = rollout.rewards.cumsum(dim=1)
        peak = torch.cat((torch.zeros_like(equity[:, :1]), equity), dim=1).cummax(dim=1).values[:, 1:]
        for i in range(prices.shape[0]):
            positions = rollout.positions[i]
            rows.append(
                {
                    "sample_id": batch["_id"][i],
                    "week_start": batch["week_start"][i],
                    "policy_return": rollout.rewards[i].sum().item(),
                    "long_hold_return": long_hold[i].sum().item(),
                    "short_hold_return": short_hold[i].sum().item(),
                    "stock_week_return": week_return[i].item(),
                    "spy_week_return": spy_return[i].item(),
                    "max_drawdown": (peak[i] - equity[i]).max().item(),
                    "trades": rollout.trades[i].float().sum().item(),
                    "transaction_cost": rollout.costs[i].sum().item(),
                    "long_fraction": positions.gt(0).float().mean().item(),
                    "short_fraction": positions.lt(0).float().mean().item(),
                    "flat_fraction": positions.eq(0).float().mean().item(),
                    "overnight_holds": positions.index_select(0, boundaries)
                    .ne(0)
                    .float()
                    .sum()
                    .item(),
                }
            )
    frame = pd.DataFrame(rows)
    rewards = torch.cat(all_rewards, dim=0)
    summary = {
        "checkpoint": str(checkpoint_path),
        "global_step": int(state.get("global_step", 0)),
        "symbols": sorted(set(frame.sample_id)),
        "weeks": sorted(set(frame.week_start)),
        "symbol_weeks": int(len(frame)),
        "date_from": str(date_from.date()),
        "date_to": None if date_to is None else str(pd.Timestamp(date_to).date()),
        **performance_metrics(rewards),
        "total_return": float(frame.policy_return.sum()),
        "mean_long_hold_return": float(frame.long_hold_return.mean()),
        "hit_rate": float((frame.policy_return > 0).mean()),
        "beats_long_hold": float((frame.policy_return > frame.long_hold_return).mean()),
        "mean_trades": float(frame.trades.mean()),
        "mean_overnight_holds": float(frame.overnight_holds.mean()),
        "mean_flat_fraction": float(frame.flat_fraction.mean()),
    }
    return frame, summary


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--symbols", default=None, help="explicit comma-separated list")
parser.add_argument("--top", type=int, default=None, help="top N of the ranked ticker list")
parser.add_argument("--date_from", default=None)
parser.add_argument("--date_to", default=None, help="last week_start to include")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--output_path", default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    frame, summary = evaluate(
        args.config_path,
        args.checkpoint,
        resolve_symbols(OmegaConf.load(args.config_path), args.symbols, args.top),
        args.device,
        args.date_from,
        args.batch_size,
        args.date_to,
    )
    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, default=str))
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, indent=2, default=str))
        frame.to_csv(Path(args.output_path).with_suffix(".csv"), index=False)
