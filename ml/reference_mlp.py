"""Autoregressive shifted-window MLP policy for an asset with an SPY reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from . import model
from .train import MarketRollout, generalized_advantages, market_rewards, ppo_update


MLP_REFERENCE_FEATURE_NAMES = (
    "asset_relative_log_price",
    "spy_relative_log_price",
    "asset_log_return",
    "spy_log_return",
    "asset_inventory",
    "previous_action_buy",
    "previous_action_nothing",
    "previous_action_sell",
)
MLP_REFERENCE_FEATURE_DIM = len(MLP_REFERENCE_FEATURE_NAMES)
MLP_REFERENCE_SCALAR_NAMES = ("time_to_close",)


def build_shifted_price_features(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    context_ticks: int,
    window_size: int,
    rollout_size: int,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """Build one causal, scale-invariant price window per trading decision."""
    if prices.shape != reference_prices.shape or prices.ndim != 2:
        raise ValueError("asset and SPY prices must share shape (batch, sequence)")
    if not torch.isfinite(prices).all() or not torch.isfinite(reference_prices).all():
        raise ValueError("prices must be finite")
    if (prices <= 0).any() or (reference_prices <= 0).any():
        raise ValueError("prices must be positive")

    context_ticks = int(context_ticks)
    window_size = int(window_size)
    steps = int(rollout_size)
    if context_ticks < 1 or window_size < 1 or steps < 1:
        raise ValueError("context, window, and rollout sizes must be positive")
    if window_size > context_ticks + 1:
        raise ValueError("MLP window cannot extend before the available context")
    required = context_ticks + steps + 1
    if prices.shape[1] < required:
        raise ValueError(f"rollout requires at least {required} price observations")

    # Decision zero is made from the window ending at ``context_ticks`` (09:30).
    start = context_ticks - window_size + 1
    stop = context_ticks + steps
    asset_windows = prices[:, start:stop].unfold(1, window_size, 1)
    spy_windows = reference_prices[:, start:stop].unfold(1, window_size, 1)
    if asset_windows.shape[1] != steps:
        raise AssertionError("shifted-window construction produced the wrong number of decisions")

    scale = float(price_feature_scale)
    asset_log = torch.log(asset_windows.float())
    spy_log = torch.log(spy_windows.float())
    asset_relative = (asset_log - asset_log[..., -1:]) * scale
    spy_relative = (spy_log - spy_log[..., -1:]) * scale
    asset_returns = F.pad(asset_log[..., 1:] - asset_log[..., :-1], (1, 0)) * scale
    spy_returns = F.pad(spy_log[..., 1:] - spy_log[..., :-1], (1, 0)) * scale
    return torch.stack(
        (asset_relative, spy_relative, asset_returns, spy_returns), dim=-1
    )


@torch.no_grad()
def collect_shifted_mlp_rollout(
    actor: model.TradingActor,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    context_ticks: int,
    rollout_size: int,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    sampling: str = "multinomial",
    price_feature_scale: float = 100.0,
) -> MarketRollout:
    """Sample actions sequentially and shift each result into the next window."""
    window_size = actor.window_size
    steps = int(rollout_size)
    market = build_shifted_price_features(
        prices,
        reference_prices,
        context_ticks,
        window_size,
        steps,
        price_feature_scale,
    )
    if actor.feature_dim != MLP_REFERENCE_FEATURE_DIM:
        raise ValueError(
            f"shifted reference actor requires feature_dim={MLP_REFERENCE_FEATURE_DIM}"
        )
    if actor.scalar_dim != len(MLP_REFERENCE_SCALAR_NAMES):
        raise ValueError("shifted reference actor requires scalar_dim=1")

    batch = prices.shape[0]
    device = prices.device
    # The historical portion is genuinely price-only: inventory and action
    # channels remain zero until the policy begins trading at 09:30.
    inventory_history = torch.zeros(
        (batch, window_size + steps), dtype=torch.float32, device=device
    )
    action_history = torch.zeros(
        (batch, window_size + steps, model.ACTION_DIM),
        dtype=torch.float32,
        device=device,
    )
    current_position = torch.zeros(batch, dtype=torch.long, device=device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    time_to_close = (
        torch.arange(steps, 0, -1, dtype=torch.float32, device=device)
        / float(steps)
    ).view(1, steps, 1).expand(batch, -1, -1)

    for step in range(steps):
        window = slice(step, step + window_size)
        state = torch.cat(
            (
                market[:, step],
                inventory_history[:, window].unsqueeze(-1),
                action_history[:, window],
            ),
            dim=-1,
        )
        action = actor.play(
            state, time_to_close[:, step], sampling=sampling
        )
        current_position = model.apply_action(current_position, action)
        states.append(state)
        actions.append(action)
        positions.append(current_position)

        next_index = window_size + step
        inventory_history[:, next_index] = current_position.to(torch.float32)
        action_history[:, next_index] = F.one_hot(
            action, num_classes=model.ACTION_DIM
        ).to(torch.float32)

    states_tensor = torch.stack(states, dim=1)
    actions_tensor = torch.stack(actions, dim=1)
    positions_tensor = torch.stack(positions, dim=1)
    context_ticks = int(context_ticks)
    price_now = prices[:, context_ticks : context_ticks + steps]
    price_next = prices[:, context_ticks + 1 : context_ticks + steps + 1]
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
        scalars=time_to_close,
    )


def shifted_mlp_train_step(
    actor: model.TradingActor,
    critic: model.TradingCritic,
    optimizer: torch.optim.Optimizer,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    cfg: DictConfig,
) -> tuple[MarketRollout, dict[str, float]]:
    rollout = collect_shifted_mlp_rollout(
        actor,
        prices,
        reference_prices,
        context_ticks=int(cfg.data.context_ticks),
        rollout_size=int(cfg.data.rollout_size),
        transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
        price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
    )
    with torch.no_grad():
        distribution = actor.distribution(rollout.states, rollout.scalars)
        old_logprobs = distribution.log_prob(rollout.actions)
        old_values = critic(rollout.states, rollout.scalars)
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
