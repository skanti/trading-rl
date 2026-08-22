"""Greedy full-session backtest for a trained trading checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import model
import utils
from train import collect_rollout, market_rewards, regular_session_mask, session_progress


@torch.no_grad()
def evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str | None,
    max_days: int | None,
    device: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, float | int | str]:
    cfg = OmegaConf.load(config_path)
    checkpoint_path = checkpoint_path or utils.load_most_recent_checkpoint(cfg.general.experiment_dir)
    if not checkpoint_path:
        raise ValueError(f"no checkpoint found under {cfg.general.experiment_dir}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor_cfg = state.get("config", {}).get("model", {}).get("actor")
    if actor_cfg is None:
        actor_cfg = OmegaConf.to_container(cfg.model.actor, resolve=True)
    actor = model.TradingActor(**actor_cfg).to(device)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()

    days = pd.read_csv(cfg.data.days_path)
    days.date = pd.to_datetime(days.date)
    date_from = date_from or str(cfg.data.date_val)
    days = days[days.date >= pd.Timestamp(date_from)]
    if date_to is not None:
        days = days[days.date <= pd.Timestamp(date_to)]
    days = days.sort_values(["date", "sample_id"])
    if max_days is not None:
        days = days.iloc[:max_days]
    n = actor.window_size
    day_rewards: list[float] = []
    step_rewards: list[np.ndarray] = []
    day_market_returns: list[float] = []
    positions: list[np.ndarray] = []
    trades: list[float] = []
    transaction_costs: list[float] = []
    risk_costs: list[float] = []
    long_rewards: list[float] = []
    short_rewards: list[float] = []
    opens: list[int] = []
    closes: list[int] = []
    reversals: list[int] = []

    for sample in tqdm(days.itertuples(index=False), total=len(days), desc="full-session backtest"):
        data = np.load(f"{cfg.data.data_dir}/{sample.sample_id}.npy", mmap_mode="r")
        sod_idx, eod_idx = int(sample.sod_idx), int(sample.eod_idx)
        start = sod_idx - n + 1
        if start < 0 or eod_idx <= sod_idx:
            continue
        segment = np.array(data[start : eod_idx + 1], copy=True)
        rollout_size = eod_idx - sod_idx
        if segment.shape[0] != n + rollout_size:
            continue
        prices = torch.as_tensor(segment[:, 1] / 1000.0, dtype=torch.float32, device=device).unsqueeze(0)
        volumes = torch.as_tensor(segment[:, 2], dtype=torch.float32, device=device).unsqueeze(0)
        secs = torch.as_tensor(segment[:, 0], dtype=torch.long, device=device).unsqueeze(0)
        if not regular_session_mask(secs[:, n - 1 :], cfg.data.anno).all():
            raise ValueError(f"{sample.sample_id} {sample.date} contains an out-of-session decision/exit tick")
        rollout = collect_rollout(
            actor,
            prices,
            volumes,
            session_progress(secs, cfg.data.anno),
            rollout_size,
            transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
            risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
            sampling="greedy",
            price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
        )
        day_rewards.append(rollout.rewards.sum().item())
        step_rewards.append(rollout.rewards[0].cpu().numpy())
        day_market_returns.append((prices[0, -1] / prices[0, n - 1] - 1.0).item())
        day_positions = rollout.positions[0].cpu().numpy()
        positions.append(day_positions)
        trades.append(rollout.trades.float().sum().item())
        transaction_costs.append(rollout.costs.sum().item())
        risk_costs.append(rollout.risk_costs.sum().item())
        previous_positions = np.pad(day_positions[:-1], (1, 0), constant_values=0)
        opens.append(int(((previous_positions == 0) & (day_positions != 0)).sum()))
        closes.append(int(((previous_positions != 0) & (day_positions == 0)).sum()))
        reversals.append(int((previous_positions * day_positions < 0).sum()))
        price_now = prices[:, n - 1 : -1]
        price_next = prices[:, n:]
        reward_kwargs = {
            "transaction_cost": float(cfg.model.get("transaction_cost", 0.0)),
            "risk_penalty": float(cfg.model.get("risk_penalty", 0.0)),
        }
        long_reward, _ = market_rewards(
            torch.full_like(price_now, model.MAX_POSITION, dtype=torch.long), price_now, price_next, **reward_kwargs
        )
        short_reward, _ = market_rewards(
            torch.full_like(price_now, -model.MAX_POSITION, dtype=torch.long), price_now, price_next, **reward_kwargs
        )
        long_rewards.append(long_reward.sum().item())
        short_rewards.append(short_reward.sum().item())

    if not day_rewards:
        raise ValueError("no evaluable validation sessions were found")
    all_positions = np.concatenate(positions)
    rewards = np.asarray(day_rewards)
    all_step_rewards = np.concatenate(step_rewards)
    market_returns = np.asarray(day_market_returns)
    long_rewards_array = np.asarray(long_rewards)
    short_rewards_array = np.asarray(short_rewards)
    cumulative = rewards.cumsum()
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = running_peak[1:] - cumulative
    reward_std = float(rewards.std())
    static_means = (0.0, float(long_rewards_array.mean()), float(short_rewards_array.mean()))
    gross_profit = float(np.clip(all_step_rewards, 0.0, None).sum())
    gross_loss = float(-np.clip(all_step_rewards, None, 0.0).sum())
    summary: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "date_from": date_from,
        "date_to": date_to or "latest",
        "days": len(day_rewards),
        "return": float(rewards.mean()),
        "total_return": float(rewards.sum()),
        "profit_factor": gross_profit / max(gross_loss, 1e-12),
        "mean_reward": float(rewards.mean()),
        "mean_gross_reward": float(rewards.mean() + np.mean(transaction_costs) + np.mean(risk_costs)),
        "mean_transaction_cost": float(np.mean(transaction_costs)),
        "mean_risk_cost": float(np.mean(risk_costs)),
        "median_reward": float(np.median(rewards)),
        "positive_day_fraction": float((rewards > 0).mean()),
        "reward_std": reward_std,
        "annualized_reward_sharpe": float(rewards.mean() / reward_std * np.sqrt(252.0)) if reward_std else 0.0,
        "maximum_cumulative_drawdown": float(drawdowns.max(initial=0.0)),
        "always_flat_mean_reward": 0.0,
        "always_long_mean_reward": static_means[1],
        "always_short_mean_reward": static_means[2],
        "excess_vs_best_static_mean_reward": float(rewards.mean() - max(static_means)),
        "mean_market_return": float(market_returns.mean()),
        "mean_trades": float(np.mean(trades)),
        "total_trades": int(np.sum(trades)),
        "mean_opens": float(np.mean(opens)),
        "mean_voluntary_closes": float(np.mean(closes)),
        "mean_reversals": float(np.mean(reversals)),
        "mean_abs_position": float(np.abs(all_positions).mean()),
        "long_fraction": float((all_positions > 0).mean()),
        "short_fraction": float((all_positions < 0).mean()),
        "flat_fraction": float((all_positions == 0).mean()),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="main.yaml")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--max_days", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--date_from")
    parser.add_argument("--date_to")
    parser.add_argument("--output_path")
    args = parser.parse_args()
    summary = evaluate_checkpoint(
        args.config_path,
        args.checkpoint_path,
        args.max_days,
        args.device,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_path:
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
