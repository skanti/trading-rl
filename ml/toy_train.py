"""Train and verify the trading policy on a scale-randomized toy market."""

from __future__ import annotations

import argparse
import json
from itertools import chain

import numpy as np
import torch
from omegaconf import OmegaConf

import model
from train import collect_rollout, rollout_metrics, train_step


def make_toy_batch(
    batch_size: int,
    window_size: int,
    rollout_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create rising/falling episodes at unrelated absolute price scales."""
    ticks = window_size + rollout_size
    direction = torch.randint(0, 2, (batch_size,), device=device).float().mul(2).sub(1)
    # Symbol scales span two orders of magnitude. They carry no information
    # about direction and should disappear under relative-price normalization.
    base = torch.exp(torch.empty(batch_size, device=device).uniform_(np.log(8.0), np.log(800.0)))
    increments = direction[:, None] * 0.004 + torch.randn(batch_size, ticks, device=device) * 0.00015
    increments[:, 0] = 0.0
    # Halfway through the tradable horizon the signal disappears. After the
    # first observed flat tick, a risk-adjusted policy should explicitly close.
    flat_start = window_size + rollout_size // 2
    increments[:, flat_start:] = 0.0
    prices = base[:, None] * torch.exp(increments.cumsum(dim=1))
    volumes = torch.exp(torch.randn(batch_size, ticks, device=device) * 0.35 + 6.0)
    progress = torch.linspace(-1.0, 1.0, ticks, device=device).expand(batch_size, -1)
    return prices, volumes, progress, direction


@torch.no_grad()
def evaluate(
    actor: model.TradingActor,
    batch_size: int,
    window_size: int,
    rollout_size: int,
    device: torch.device | str,
) -> dict[str, float]:
    prices, volumes, progress, direction = make_toy_batch(batch_size, window_size, rollout_size, device)
    rollout = collect_rollout(
        actor,
        prices,
        volumes,
        progress,
        rollout_size,
        risk_penalty=9e-4,
        sampling="greedy",
        price_feature_scale=100.0,
    )
    trend_positions = rollout.positions[:, : rollout_size // 2]
    flat_positions = rollout.positions[:, rollout_size // 2 + 1 :]
    result = rollout_metrics(rollout)
    result["direction_accuracy"] = trend_positions.sign().eq(direction[:, None]).float().mean().item()
    result["close_accuracy"] = flat_positions.eq(0).float().mean().item()
    result["mean_bet_size"] = trend_positions.abs().float().mean().item()
    return result


def run(updates: int, seed: int, device: str) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)
    window_size, rollout_size, batch_size = 32, 12, 64
    cfg = OmegaConf.create(
        {
            "model": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "transaction_cost": 0.0,
                "risk_penalty": 9e-4,
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

    before = evaluate(actor, 512, window_size, rollout_size, torch_device)
    last_losses: dict[str, float] = {}
    for _ in range(updates):
        prices, volumes, progress, _ = make_toy_batch(batch_size, window_size, rollout_size, torch_device)
        _, last_losses = train_step(actor, critic, optimizer, prices, volumes, progress, cfg)
    after = evaluate(actor, 1024, window_size, rollout_size, torch_device)
    passed = (
        after["direction_accuracy"] >= 0.95
        and after["close_accuracy"] >= 0.90
        and after["reward"] > 0.005
        and after["mean_bet_size"] >= 4.0
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
