"""Tokenized autoregressive PPO for one asset with SPY as a price reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions import Categorical

import model as trading_model
from gpt import CausalTradingTransformer, PairPriceTokenizer
from train import generalized_advantages, market_rewards


TOKEN_FEATURE_NAMES = (
    "joint_asset_spy_log_return_token",
    "asset_inventory",
    "previous_action",
)


@dataclass
class TokenRollout:
    token_ids: torch.Tensor
    inventory_ids: torch.Tensor
    previous_action_ids: torch.Tensor
    actions: torch.Tensor
    positions: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    price_returns: torch.Tensor
    costs: torch.Tensor
    risk_costs: torch.Tensor
    trades: torch.Tensor
    forced_closes: torch.Tensor
    context_ticks: int


@torch.no_grad()
def collect_token_rollout(
    transformer: CausalTradingTransformer,
    tokenizer: PairPriceTokenizer,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    context_ticks: int,
    rollout_size: int,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    sampling: str = "multinomial",
) -> TokenRollout:
    """Prime the 10-day cache once, then append one current-day tick at a time."""
    if prices.shape != reference_prices.shape or prices.ndim != 2:
        raise ValueError("asset and reference prices must share shape (batch, sequence)")
    context_ticks, steps = int(context_ticks), int(rollout_size)
    required = context_ticks + steps + 1
    if prices.shape[1] < required:
        raise ValueError(f"rollout requires at least {required} price observations")

    all_tokens = tokenizer.encode(prices, reference_prices)
    model_tokens = all_tokens[:, : context_ticks + steps]
    batch = prices.shape[0]
    device = prices.device
    inventory_ids = torch.full_like(model_tokens, 1)  # short/flat/long -> 0/1/2
    previous_action_ids = torch.full_like(model_tokens, 3)  # 3 means no prior command.
    transformer.setup_cache(batch, context_ticks + steps)
    transformer.reset_cache()
    transformer.cached_forward(
        model_tokens[:, :context_ticks],
        inventory_ids[:, :context_ticks],
        previous_action_ids[:, :context_ticks],
    )

    current_position = torch.zeros(batch, dtype=torch.long, device=device)
    actions: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    previous_action = torch.full(
        (batch,), 3, dtype=torch.long, device=device
    )
    for step in range(steps):
        sequence_index = context_ticks + step
        current_inventory = current_position + 1
        inventory_ids[:, sequence_index] = current_inventory
        previous_action_ids[:, sequence_index] = previous_action
        logits, value = transformer.cached_forward(
            model_tokens[:, sequence_index : sequence_index + 1],
            current_inventory.unsqueeze(1),
            previous_action.unsqueeze(1),
        )
        distribution = Categorical(logits=logits[:, 0])
        if sampling == "multinomial":
            action = distribution.sample()
        elif sampling in ("argmax", "greedy"):
            action = logits[:, 0].argmax(dim=-1)
        else:
            raise ValueError(f"unknown sampling mode: {sampling}")
        current_position = trading_model.apply_action(current_position, action)
        actions.append(action)
        positions.append(current_position)
        logprobs.append(distribution.log_prob(action))
        values.append(value[:, 0])
        previous_action = action

    actions_tensor = torch.stack(actions, dim=1)
    positions_tensor = torch.stack(positions, dim=1)
    old_values = torch.stack(values, dim=1)
    price_now = prices[:, context_ticks : context_ticks + steps]
    price_next = prices[:, context_ticks + 1 : context_ticks + steps + 1]
    rewards, info = market_rewards(
        positions_tensor,
        price_now,
        price_next,
        transaction_cost=transaction_cost,
        risk_penalty=risk_penalty,
    )
    return TokenRollout(
        token_ids=model_tokens,
        inventory_ids=inventory_ids,
        previous_action_ids=previous_action_ids,
        actions=actions_tensor,
        positions=positions_tensor,
        old_logprobs=torch.stack(logprobs, dim=1),
        old_values=old_values,
        rewards=rewards,
        price_returns=info["returns"],
        costs=info["costs"],
        risk_costs=info["risk_costs"],
        trades=info["trades"],
        forced_closes=info["forced_closes"],
        context_ticks=context_ticks,
    )


def transformer_ppo_update(
    transformer: CausalTradingTransformer,
    optimizer: torch.optim.Optimizer,
    rollout: TokenRollout,
    old_logprobs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: DictConfig,
) -> dict[str, float]:
    weights = cfg.model.loss_weights
    epochs = int(cfg.train.get("ppo_epochs", 4))
    clip = float(cfg.model.get("ppo_clip", 0.2))
    value_clip = float(cfg.model.get("ppo_value_clip", clip))
    max_grad_norm = float(cfg.model.get("max_grad_norm", 1.0))
    start = rollout.context_ticks
    stop = start + rollout.actions.shape[1]
    loss = policy_loss = value_loss = entropy_loss = torch.tensor(
        0.0, device=rollout.rewards.device
    )
    entropy = torch.tensor(0.0, device=rollout.rewards.device)

    for _ in range(epochs):
        logits, all_values = transformer(
            rollout.token_ids,
            rollout.inventory_ids,
            rollout.previous_action_ids,
        )
        distribution = Categorical(logits=logits[:, start:stop])
        logprobs = distribution.log_prob(rollout.actions)
        entropy = distribution.entropy().mean()
        values = all_values[:, start:stop]
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
        policy_loss = -torch.minimum(
            ratio * advantages, clipped_ratio * advantages
        ).mean()
        entropy_loss = -entropy
        clipped_values = rollout.old_values + (
            values - rollout.old_values
        ).clamp(-value_clip, value_clip)
        value_loss = 0.5 * torch.maximum(
            F.mse_loss(values, returns, reduction="none"),
            F.mse_loss(clipped_values, returns, reduction="none"),
        ).mean()
        loss = (
            float(weights.policy) * policy_loss
            + float(weights.value) * value_loss
            + float(weights.entropy) * entropy_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value": float(value_loss.item()),
        "loss_entropy": float(entropy_loss.item()),
        "entropy": float(entropy.item()),
    }


def token_train_step(
    transformer: CausalTradingTransformer,
    tokenizer: PairPriceTokenizer,
    optimizer: torch.optim.Optimizer,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    cfg: DictConfig,
) -> tuple[TokenRollout, dict[str, float]]:
    rollout = collect_token_rollout(
        transformer,
        tokenizer,
        prices,
        reference_prices,
        context_ticks=int(cfg.data.context_ticks),
        rollout_size=int(cfg.data.rollout_size),
        transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
    )
    with torch.no_grad():
        advantages, returns = generalized_advantages(
            rollout.rewards,
            rollout.old_values,
            float(cfg.model.gamma),
            float(cfg.model.gae_lambda),
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    losses = transformer_ppo_update(
        transformer,
        optimizer,
        rollout,
        rollout.old_logprobs,
        returns,
        advantages,
        cfg,
    )
    return rollout, losses
