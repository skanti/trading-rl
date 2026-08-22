"""Single-asset PPO observations augmented with an observation-only reference."""

from __future__ import annotations

import torch

import model
from train import (
    MarketRollout,
    build_market_features,
    generalized_advantages,
    market_rewards,
    ppo_update,
)


REFERENCE_FEATURE_NAMES = (
    "asset_relative_log_price",
    "reference_relative_log_price",
    "asset_log_return",
    "reference_log_return",
    "asset_normalized_log_volume",
    "reference_normalized_log_volume",
    "session_progress",
    "asset_position",
)
REFERENCE_FEATURE_DIM = len(REFERENCE_FEATURE_NAMES)


def build_reference_features(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    volumes: torch.Tensor,
    reference_volumes: torch.Tensor,
    progress: torch.Tensor,
    window_size: int,
    rollout_size: int,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """Build causal, scale-invariant asset and SPY windows without action history."""
    asset = build_market_features(
        prices, volumes, progress, window_size, rollout_size, price_feature_scale
    )
    reference = build_market_features(
        reference_prices,
        reference_volumes,
        progress,
        window_size,
        rollout_size,
        price_feature_scale,
    )
    return torch.stack(
        (
            asset[..., 0],
            reference[..., 0],
            asset[..., 1],
            reference[..., 1],
            asset[..., 2],
            reference[..., 2],
            asset[..., 3],
        ),
        dim=-1,
    )


@torch.no_grad()
def collect_reference_rollout(
    actor: model.TradingActor,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    volumes: torch.Tensor,
    reference_volumes: torch.Tensor,
    progress: torch.Tensor,
    rollout_size: int,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    sampling: str = "multinomial",
    price_feature_scale: float = 100.0,
) -> MarketRollout:
    """Trade only the asset while feeding both asset and reference observations."""
    n = actor.window_size
    steps = int(rollout_size)
    market = build_reference_features(
        prices,
        reference_prices,
        volumes,
        reference_volumes,
        progress,
        n,
        steps,
        price_feature_scale,
    )
    batch = prices.shape[0]
    position_history = torch.zeros(
        (batch, n + steps), dtype=torch.float32, device=prices.device
    )
    current_position = torch.zeros(batch, dtype=torch.long, device=prices.device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []

    for step in range(steps):
        action_window = position_history[:, step : step + n].unsqueeze(-1)
        state = torch.cat((market[:, step], action_window), dim=-1)
        action = actor.play(state, sampling=sampling)
        current_position = model.apply_action(current_position, action)
        states.append(state)
        actions.append(action)
        positions.append(current_position)
        position_history[:, n + step] = current_position.to(torch.float32) / model.MAX_POSITION

    states_tensor = torch.stack(states, dim=1)
    actions_tensor = torch.stack(actions, dim=1)
    positions_tensor = torch.stack(positions, dim=1)
    price_now = prices[:, n - 1 : n - 1 + steps]
    price_next = prices[:, n : n + steps]
    rewards, info = market_rewards(
        positions_tensor,
        price_now,
        price_next,
        transaction_cost=transaction_cost,
        risk_penalty=risk_penalty,
    )
    return MarketRollout(
        states=states_tensor,
        actions=actions_tensor,
        positions=positions_tensor,
        rewards=rewards,
        price_returns=info["returns"],
        costs=info["costs"],
        risk_costs=info["risk_costs"],
        trades=info["trades"],
        forced_closes=info["forced_closes"],
    )


def reference_train_step(
    actor: model.TradingActor,
    critic: model.TradingCritic,
    optimizer: torch.optim.Optimizer,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    volumes: torch.Tensor,
    reference_volumes: torch.Tensor,
    progress: torch.Tensor,
    cfg,
) -> tuple[MarketRollout, dict[str, float]]:
    rollout = collect_reference_rollout(
        actor,
        prices,
        reference_prices,
        volumes,
        reference_volumes,
        progress,
        rollout_size=int(cfg.data.rollout_size),
        transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
        price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
    )
    with torch.no_grad():
        distribution = actor.distribution(rollout.states)
        old_logprobs = distribution.log_prob(rollout.actions)
        old_values = critic(rollout.states)
        advantages, returns = generalized_advantages(
            rollout.rewards,
            old_values,
            float(cfg.model.gamma),
            float(cfg.model.gae_lambda),
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    losses = ppo_update(
        actor,
        critic,
        optimizer,
        rollout,
        old_logprobs,
        old_values,
        returns,
        advantages,
        cfg,
    )
    return rollout, losses
