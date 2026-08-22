"""Train and verify the trading policy on online noisy Bezier markets."""

from __future__ import annotations

import argparse
import json
from itertools import chain

import numpy as np
import torch
from omegaconf import OmegaConf

import model
from dataset import OnlineBezierToyProvider
from train import collect_rollout, market_rewards, rollout_metrics, train_step


@torch.no_grad()
def evaluate(
    actor: model.TradingActor,
    provider: OnlineBezierToyProvider,
    batch_size: int,
    device: torch.device | str,
    seed: int = 10_007,
) -> dict[str, float]:
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    batch = provider.sample(batch_size, torch_device, generator)
    rollout = collect_rollout(
        actor,
        batch.prices,
        batch.volumes,
        batch.progress,
        provider.rollout_size,
        transaction_cost=1e-4,
        risk_penalty=1e-5,
        sampling="greedy",
        price_feature_scale=100.0,
    )
    active = batch.target_positions.ne(0)
    flat = ~active
    result = rollout_metrics(rollout)
    result["position_accuracy"] = rollout.positions.eq(batch.target_positions).float().mean().item()
    result["direction_accuracy"] = (
        rollout.positions[active].eq(batch.target_positions[active]).float().mean().item()
        if active.any()
        else 1.0
    )
    result["close_accuracy"] = rollout.positions[flat].eq(0).float().mean().item() if flat.any() else 1.0
    result["mean_abs_position"] = rollout.positions.abs().float().mean().item()
    oracle_rewards, _ = market_rewards(
        batch.target_positions,
        batch.prices[:, provider.window_size - 1 : -1],
        batch.prices[:, provider.window_size :],
        transaction_cost=1e-4,
        risk_penalty=1e-5,
    )
    result["oracle_reward"] = oracle_rewards.sum(dim=1).mean().item()
    result["three_anchor_fraction"] = batch.anchor_counts.eq(3).float().mean().item()
    return result


def run(updates: int, seed: int, device: str) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)
    window_size, rollout_size, batch_size = 32, 12, 64
    provider = OnlineBezierToyProvider(window_size, rollout_size)
    cfg = OmegaConf.create(
        {
            "model": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "transaction_cost": 1e-4,
                "risk_penalty": 1e-5,
                "ppo_clip": 0.2,
                "ppo_value_clip": 0.2,
                "max_grad_norm": 1.0,
                "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.01},
            },
            "data": {"rollout_size": rollout_size, "price_feature_scale": 100.0},
            "train": {"ppo_epochs": 2},
        }
    )
    actor = model.TradingActor(window_size, hidden_dim=64, depth=2).to(torch_device)
    critic = model.TradingCritic(window_size, hidden_dim=64, depth=2).to(torch_device)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=1e-3)

    before = evaluate(actor, provider, 512, torch_device)
    last_losses: dict[str, float] = {}
    for _ in range(updates):
        batch = provider.sample(batch_size, torch_device)
        _, last_losses = train_step(
            actor, critic, optimizer, batch.prices, batch.volumes, batch.progress, cfg
        )
    after = evaluate(actor, provider, 1024, torch_device)
    passed = (
        after["position_accuracy"] >= 0.85
        and after["direction_accuracy"] >= 0.90
        and after["reward"] > max(0.0, before["reward"])
    )
    summary: dict[str, object] = {
        "seed": seed,
        "updates": updates,
        "before": before,
        "after": after,
        "last_losses": last_losses,
        "passed": passed,
    }

    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("toy verification did not reach the required policy quality")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.updates, args.seed, args.device)
