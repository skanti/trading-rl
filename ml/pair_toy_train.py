"""Train and verify the joint pair policy on online two-symbol toy markets.

The gates here are deliberately about behaviour rather than raw return: a pair
policy that simply doubles up on the common direction would earn more than a
single-symbol policy while learning nothing about the relationship between the
legs. What has to be demonstrated is that the policy reads the spread - it
takes the correct side of a stretched residual, and it backs away when the
residual is a random walk that will not come back.
"""

from __future__ import annotations

import argparse
import json
from itertools import chain

import numpy as np
import torch
from omegaconf import OmegaConf

from . import model
from .dataset import OnlinePairToyProvider
from .pair import (
    PAIR_FEATURE_DIM,
    PAIR_SCALAR_DIM,
    collect_pair_rollout,
    pair_rollout_metrics,
    pair_train_step,
)
from .pair_eval import (
    current_residual_z,
    score_positions,
    spread_behaviour,
    zscore_rule_positions,
)


@torch.no_grad()
def evaluate(
    actor: model.PairTradingActor,
    provider: OnlinePairToyProvider,
    batch_size: int,
    device: torch.device | str,
    cfg,
    seed: int = 10_007,
) -> dict[str, float]:
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    batch = provider.sample(batch_size, torch_device, generator)
    cost = float(cfg.model.transaction_cost)
    net_penalty = float(cfg.model.net_risk_penalty)
    gross_penalty = float(cfg.model.gross_risk_penalty)
    window, steps = provider.window_size, provider.rollout_size

    rollout = collect_pair_rollout(
        actor,
        batch.prices_a,
        batch.prices_b,
        batch.volumes_a,
        batch.volumes_b,
        batch.progress,
        steps,
        transaction_cost=cost,
        net_risk_penalty=net_penalty,
        gross_risk_penalty=gross_penalty,
        sampling="greedy",
    )
    residual_z, hedge_ratio = current_residual_z(
        batch.prices_a, batch.prices_b, batch.volumes_a, batch.volumes_b, batch.progress, window, steps
    )
    result = pair_rollout_metrics(rollout)
    result.update(spread_behaviour(rollout, residual_z, batch.is_mean_reverting))

    reference = dict(
        prices_a=batch.prices_a,
        prices_b=batch.prices_b,
        volumes_a=batch.volumes_a,
        volumes_b=batch.volumes_b,
        progress=batch.progress,
        window_size=window,
        rollout_size=steps,
        transaction_cost=cost,
        net_risk_penalty=net_penalty,
        gross_risk_penalty=gross_penalty,
        hedge_ratio=hedge_ratio,
    )
    rule_a, rule_b = zscore_rule_positions(residual_z)
    result["zscore_rule_reward"] = score_positions(rule_a, rule_b, **reference).rewards.sum(dim=1).mean().item()
    return result


def run(updates: int, seed: int, device: str) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)
    window_size, rollout_size, batch_size = 128, 24, 64
    provider = OnlinePairToyProvider(
        window_size,
        rollout_size,
        # A short window only supports a fast spread, so the half-lives are
        # scaled down with it; the ratio to the window length is what matters.
        half_life_min=4.0,
        half_life_max=24.0,
        spread_step_std=1.2e-3,
        # Over 24 ticks a Bezier trend is close to a straight line, so at this
        # scale the directional trade is almost free money and a policy that
        # took it would never need to look at the spread. Damping the shared
        # trend leaves the relative signal as the one worth learning, which is
        # what this gate exists to check.
        trend_std=0.002,
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "transaction_cost": 1e-4,
                "net_risk_penalty": 1e-5,
                "gross_risk_penalty": 1e-6,
                "ppo_clip": 0.2,
                "ppo_value_clip": 0.2,
                "max_grad_norm": 1.0,
                "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.01},
            },
            "data": {"rollout_size": rollout_size, "price_feature_scale": 100.0},
            "train": {"ppo_epochs": 2},
        }
    )
    actor = model.PairTradingActor(
        window_size, PAIR_FEATURE_DIM, hidden_dim=64, depth=2, scalar_dim=PAIR_SCALAR_DIM
    ).to(torch_device)
    critic = model.PairTradingCritic(
        window_size, PAIR_FEATURE_DIM, hidden_dim=64, depth=2, scalar_dim=PAIR_SCALAR_DIM
    ).to(torch_device)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=1e-3)

    before = evaluate(actor, provider, 512, torch_device, cfg)
    last_losses: dict[str, float] = {}
    for _ in range(updates):
        batch = provider.sample(batch_size, torch_device)
        _, last_losses = pair_train_step(
            actor,
            critic,
            optimizer,
            batch.prices_a,
            batch.prices_b,
            batch.volumes_a,
            batch.volumes_b,
            batch.progress,
            cfg,
        )
    after = evaluate(actor, provider, 1024, torch_device, cfg)

    # Regime discrimination is reported but not gated here. Separating an OU
    # spread from a random walk means resolving a difference in lag-one
    # autocorrelation of about 0.06, and over the 128-tick window this fast gate
    # uses the standard error of that estimate is roughly 0.03. The statistic is
    # barely resolvable at this scale whatever the policy does, so it is
    # verified at the full 4,096-tick window instead, where it is measurable.
    passed = (
        after["reward"] > max(0.0, before["reward"])
        and after["spread_direction_accuracy"] >= 0.60
        and after["reward"] > after["zscore_rule_reward"]
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
        raise RuntimeError("pair toy verification did not reach the required policy quality")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.updates, args.seed, args.device)
