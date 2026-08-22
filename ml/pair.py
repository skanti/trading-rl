"""Relative-value PPO machinery for a time-matched two-symbol pair.

The policy observes both legs at once and issues one joint command per tick.
Its inventory therefore spans four economically distinct trades: flat, a single
leg, a beta-hedged spread, and a doubled-up directional bet. The features and
the reward are built so that the spread is the cheapest of those to hold and
the doubled-up bet the most expensive, which is what pushes the policy to trade
the relationship between the legs rather than their common direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import model
from train import build_market_features, generalized_advantages, performance_metrics


PAIR_FEATURE_NAMES = (
    "relative_log_price_a",
    "relative_log_price_b",
    "log_return_a",
    "log_return_b",
    "normalized_log_volume_a",
    "normalized_log_volume_b",
    "spread_z",
    "spread_z_return",
    "session_progress",
    "position_a",
    "position_b",
)
PAIR_FEATURE_DIM = len(PAIR_FEATURE_NAMES)

# Window-level statistics appended once per decision instead of being broadcast
# across every tick of the window.
PAIR_SCALAR_NAMES = ("hedge_ratio", "return_correlation", "residual_autocorrelation", "residual_scale")
PAIR_SCALAR_DIM = len(PAIR_SCALAR_NAMES)

MAX_HEDGE_RATIO = 5.0


@dataclass
class PairRollout:
    states: torch.Tensor  # (batch, rollout, window, features)
    scalars: torch.Tensor  # (batch, rollout, scalars)
    actions: torch.Tensor  # joint categorical indices (batch, rollout)
    positions_a: torch.Tensor
    positions_b: torch.Tensor
    hedge_ratio: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    risk_costs: torch.Tensor
    trades: torch.Tensor
    net_exposure: torch.Tensor
    basket_pnl: torch.Tensor
    spread_pnl: torch.Tensor


@dataclass(frozen=True)
class PairWindowStatistics:
    """Causal, window-only estimates describing the relationship of two legs."""

    hedge_ratio: torch.Tensor  # (batch, rollout) sensitivity of leg B to leg A
    return_correlation: torch.Tensor
    residual_autocorrelation: torch.Tensor
    residual_scale: torch.Tensor  # in the same scaled log units as the features
    residual_z: torch.Tensor  # (batch, rollout, window)


def pair_window_statistics(
    log_returns_a: torch.Tensor,
    log_returns_b: torch.Tensor,
    relative_log_prices_a: torch.Tensor,
    relative_log_prices_b: torch.Tensor,
    eps: float = 1e-12,
) -> PairWindowStatistics:
    """Estimate the hedge ratio and the whitened spread from the window alone.

    Every statistic uses only ticks inside the observation window, so nothing
    here can see past the decision it informs.

    The hedge ratio is the cointegrating regression of leg B's log price on leg
    A's, not a regression of their returns. Regressing returns is badly biased
    here: the spread perturbs both legs tick by tick, so it acts as measurement
    error on the regressor and attenuates the slope toward zero. The resulting
    residual would then still carry a large share of the common factor, which is
    exactly the component the spread is supposed to have removed. Over a window
    long enough for the common factor to wander well beyond the spread's width,
    the level regression is dominated by genuine co-movement and recovers the
    factor sensitivity almost unbiased.

    Prices arrive as log ratios against the newest tick of the window rather
    than as raw log prices. Both give the same answer in exact arithmetic,
    since the difference is a per-window constant that centering removes, but
    the residual is around ``1e-3`` in log units while a raw log price sits
    near ``6.7``; forming it from raw levels in float32 would throw away most
    of its significant digits.
    """
    if log_returns_a.shape != log_returns_b.shape:
        raise ValueError("both legs must supply identically shaped return windows")
    if relative_log_prices_a.shape != relative_log_prices_b.shape:
        raise ValueError("both legs must supply identically shaped price windows")

    # The padded leading zero of the return window is not a real observation.
    returns_a, returns_b = log_returns_a[..., 1:], log_returns_b[..., 1:]
    centered_returns_a = returns_a - returns_a.mean(dim=-1, keepdim=True)
    centered_returns_b = returns_b - returns_b.mean(dim=-1, keepdim=True)
    return_covariance = (centered_returns_a * centered_returns_b).mean(dim=-1)
    return_variance_a = centered_returns_a.square().mean(dim=-1)
    return_variance_b = centered_returns_b.square().mean(dim=-1)
    correlation = return_covariance / (return_variance_a * return_variance_b).clamp_min(eps).sqrt()

    centered_levels_a = relative_log_prices_a - relative_log_prices_a.mean(dim=-1, keepdim=True)
    centered_levels_b = relative_log_prices_b - relative_log_prices_b.mean(dim=-1, keepdim=True)
    level_covariance = (centered_levels_a * centered_levels_b).mean(dim=-1)
    level_variance_a = centered_levels_a.square().mean(dim=-1)
    hedge_ratio = (level_covariance / level_variance_a.clamp_min(eps)).clamp(
        -MAX_HEDGE_RATIO, MAX_HEDGE_RATIO
    )

    residual = hedge_ratio.unsqueeze(-1) * centered_levels_a - centered_levels_b
    centered_residual = residual - residual.mean(dim=-1, keepdim=True)
    residual_scale = centered_residual.square().mean(dim=-1).clamp_min(eps).sqrt()
    residual_z = centered_residual / residual_scale.unsqueeze(-1)
    # Lag-one autocorrelation separates a mean-reverting spread from a random
    # walk; it is the only in-window statistic that distinguishes the regimes.
    autocorrelation = (centered_residual[..., 1:] * centered_residual[..., :-1]).mean(dim=-1) / (
        residual_scale.square().clamp_min(eps)
    )
    return PairWindowStatistics(
        hedge_ratio=hedge_ratio,
        return_correlation=correlation.clamp(-1.0, 1.0),
        residual_autocorrelation=autocorrelation.clamp(-1.5, 1.5),
        residual_scale=residual_scale,
        residual_z=residual_z,
    )


def build_pair_features(
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    window_size: int,
    rollout_size: int,
    price_feature_scale: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the joint observation for every decision in a pair rollout.

    Returns the per-tick feature windows without the two action channels, the
    per-decision scalar statistics, and the hedge ratio the reward needs to
    price residual market exposure. Both legs reuse the single-symbol feature
    builder so absolute price and symbol identity stay out of the observation.
    """
    n, t = int(window_size), int(rollout_size)
    leg_a = build_market_features(prices_a, volumes_a, progress, n, t, price_feature_scale)
    leg_b = build_market_features(prices_b, volumes_b, progress, n, t, price_feature_scale)

    # Channel 0 already holds each leg's log price relative to its newest tick,
    # scaled by the same constant for both legs, which leaves the hedge ratio
    # and the whitened residual unchanged.
    statistics = pair_window_statistics(leg_a[..., 1], leg_b[..., 1], leg_a[..., 0], leg_b[..., 0])
    residual_z = statistics.residual_z
    residual_z_return = F.pad(residual_z[..., 1:] - residual_z[..., :-1], (1, 0))

    market = torch.stack(
        (
            leg_a[..., 0],  # relative log price A
            leg_b[..., 0],  # relative log price B
            leg_a[..., 1],  # log return A
            leg_b[..., 1],  # log return B
            leg_a[..., 2],  # normalized log volume A
            leg_b[..., 2],  # normalized log volume B
            residual_z,
            residual_z_return,
            leg_a[..., 3],  # session progress, shared by both legs
        ),
        dim=-1,
    )
    scalars = torch.stack(
        (
            statistics.hedge_ratio,
            statistics.return_correlation,
            statistics.residual_autocorrelation,
            statistics.residual_scale,
        ),
        dim=-1,
    )
    return market, scalars, statistics.hedge_ratio


def pair_market_rewards(
    positions_a: torch.Tensor,
    positions_b: torch.Tensor,
    price_now_a: torch.Tensor,
    price_next_a: torch.Tensor,
    price_now_b: torch.Tensor,
    price_next_b: torch.Tensor,
    hedge_ratio: torch.Tensor,
    transaction_cost: float = 0.0,
    net_risk_penalty: float = 0.0,
    gross_risk_penalty: float = 0.0,
    max_position: int = model.MAX_POSITION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mark both legs to market and price turnover, market risk, and capital.

    ``net_risk_penalty`` charges the squared factor exposure
    ``exposure_a + hedge_ratio * exposure_b``, which is near zero for a hedged
    spread and largest for a doubled-up directional bet. ``gross_risk_penalty``
    charges deployed capital so a leg is only held when it earns its keep. Both
    legs are liquidated at the final price, as in the single-symbol reward.
    """
    shapes = {positions_a.shape, positions_b.shape, price_now_a.shape, price_next_a.shape,
              price_now_b.shape, price_next_b.shape, hedge_ratio.shape}
    if len(shapes) != 1:
        raise ValueError("positions, prices, and hedge_ratio must all have the same shape")

    exposure_a = positions_a.to(torch.float32) / float(max_position)
    exposure_b = positions_b.to(torch.float32) / float(max_position)
    returns_a = price_next_a / price_now_a.clamp_min(1e-12) - 1.0
    returns_b = price_next_b / price_now_b.clamp_min(1e-12) - 1.0

    previous_a = F.pad(exposure_a[:, :-1], (1, 0), value=0.0)
    previous_b = F.pad(exposure_b[:, :-1], (1, 0), value=0.0)
    turnover = (exposure_a - previous_a).abs() + (exposure_b - previous_b).abs()
    costs = turnover * float(transaction_cost)

    net_exposure = exposure_a + hedge_ratio * exposure_b
    gross_exposure = 0.5 * (exposure_a.abs() + exposure_b.abs())
    risk_costs = net_exposure.square() * float(net_risk_penalty) + gross_exposure * float(
        gross_risk_penalty
    )

    rewards = exposure_a * returns_a + exposure_b * returns_b - costs - risk_costs
    liquidation_cost = (exposure_a[:, -1].abs() + exposure_b[:, -1].abs()) * float(transaction_cost)
    rewards[:, -1] -= liquidation_cost
    costs[:, -1] += liquidation_cost

    # Splitting inventory into a basket leg and a spread leg,
    # ``exposure_a = basket + spread`` and ``exposure_b = basket - spread``,
    # makes the gross P&L split exactly in two:
    #   basket * (returns_a + returns_b) + spread * (returns_a - returns_b).
    # The first term is what a directional bet on the two names earns; the
    # second is what the relationship between them earns, and it is the only
    # part a single-symbol policy cannot reach.
    basket_exposure = 0.5 * (exposure_a + exposure_b)
    spread_exposure = 0.5 * (exposure_a - exposure_b)
    info = {
        "exposure_a": exposure_a,
        "exposure_b": exposure_b,
        "basket_exposure": basket_exposure,
        "spread_exposure": spread_exposure,
        "basket_pnl": basket_exposure * (returns_a + returns_b),
        "spread_pnl": spread_exposure * (returns_a - returns_b),
        "returns_a": returns_a,
        "returns_b": returns_b,
        "costs": costs,
        "risk_costs": risk_costs,
        "net_exposure": net_exposure,
        "gross_exposure": gross_exposure,
        "trades": exposure_a.ne(previous_a) | exposure_b.ne(previous_b),
        "spread_positions": (exposure_a * exposure_b).lt(0.0),
        "directional_positions": (exposure_a * exposure_b).gt(0.0),
        "single_leg_positions": exposure_a.eq(0.0) ^ exposure_b.eq(0.0),
    }
    return rewards, info


@torch.no_grad()
def collect_pair_rollout(
    actor: model.PairTradingActor,
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    rollout_size: int,
    transaction_cost: float = 0.0,
    net_risk_penalty: float = 0.0,
    gross_risk_penalty: float = 0.0,
    sampling: str = "multinomial",
    price_feature_scale: float = 100.0,
) -> PairRollout:
    """Run the joint policy sequentially, feeding both legs' actions forward."""
    n = actor.window_size
    t_steps = int(rollout_size)
    market, scalars, hedge_ratio = build_pair_features(
        prices_a, prices_b, volumes_a, volumes_b, progress, n, t_steps, price_feature_scale
    )
    batch = prices_a.shape[0]
    device = prices_a.device
    history_a = torch.zeros((batch, n + t_steps), dtype=torch.float32, device=device)
    history_b = torch.zeros((batch, n + t_steps), dtype=torch.float32, device=device)
    position_a = torch.zeros(batch, dtype=torch.long, device=device)
    position_b = torch.zeros(batch, dtype=torch.long, device=device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    positions_a: list[torch.Tensor] = []
    positions_b: list[torch.Tensor] = []

    for t in range(t_steps):
        action_windows = torch.stack(
            (history_a[:, t : t + n], history_b[:, t : t + n]), dim=-1
        )
        state = torch.cat((market[:, t], action_windows), dim=-1)
        action = actor.play(state, scalars[:, t], sampling=sampling)
        position_a, position_b = model.apply_pair_action(position_a, position_b, action)
        states.append(state)
        actions.append(action)
        positions_a.append(position_a)
        positions_b.append(position_b)
        history_a[:, n + t] = position_a.to(torch.float32) / model.MAX_POSITION
        history_b[:, n + t] = position_b.to(torch.float32) / model.MAX_POSITION

    stacked_a = torch.stack(positions_a, dim=1)
    stacked_b = torch.stack(positions_b, dim=1)
    rewards, info = pair_market_rewards(
        stacked_a,
        stacked_b,
        prices_a[:, n - 1 : n - 1 + t_steps],
        prices_a[:, n : n + t_steps],
        prices_b[:, n - 1 : n - 1 + t_steps],
        prices_b[:, n : n + t_steps],
        hedge_ratio,
        transaction_cost=transaction_cost,
        net_risk_penalty=net_risk_penalty,
        gross_risk_penalty=gross_risk_penalty,
    )
    return PairRollout(
        states=torch.stack(states, dim=1),
        scalars=scalars,
        actions=torch.stack(actions, dim=1),
        positions_a=stacked_a,
        positions_b=stacked_b,
        hedge_ratio=hedge_ratio,
        rewards=rewards,
        costs=info["costs"],
        risk_costs=info["risk_costs"],
        trades=info["trades"],
        net_exposure=info["net_exposure"],
        basket_pnl=info["basket_pnl"],
        spread_pnl=info["spread_pnl"],
    )


def pair_rollout_metrics(rollout: PairRollout) -> dict[str, float]:
    """Summarize a pair rollout, including which trade shape was held."""
    exposure_a = rollout.positions_a.float() / model.MAX_POSITION
    exposure_b = rollout.positions_b.float() / model.MAX_POSITION
    product = exposure_a * exposure_b
    both_flat = exposure_a.eq(0.0) & exposure_b.eq(0.0)
    return {
        **performance_metrics(rollout.rewards),
        "reward": rollout.rewards.sum(dim=1).mean().item(),
        "gross_reward": (rollout.rewards + rollout.costs + rollout.risk_costs).sum(dim=1).mean().item(),
        "transaction_cost": rollout.costs.sum(dim=1).mean().item(),
        "risk_cost": rollout.risk_costs.sum(dim=1).mean().item(),
        "gross_exposure": 0.5 * (exposure_a.abs() + exposure_b.abs()).mean().item(),
        "abs_net_exposure": rollout.net_exposure.abs().mean().item(),
        "trades": rollout.trades.float().sum(dim=1).mean().item(),
        "spread_fraction": product.lt(0.0).float().mean().item(),
        "directional_fraction": product.gt(0.0).float().mean().item(),
        "single_leg_fraction": (exposure_a.eq(0.0) ^ exposure_b.eq(0.0)).float().mean().item(),
        "flat_fraction": both_flat.float().mean().item(),
        "hedge_ratio": rollout.hedge_ratio.mean().item(),
        "basket_pnl": rollout.basket_pnl.sum(dim=1).mean().item(),
        "spread_pnl": rollout.spread_pnl.sum(dim=1).mean().item(),
    }


def pair_ppo_update(
    actor: model.PairTradingActor,
    critic: model.PairTradingCritic,
    optimizer: torch.optim.Optimizer,
    rollout: PairRollout,
    logprobs_old: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: DictConfig,
) -> dict[str, float]:
    """Clipped PPO over the joint action, matching the single-symbol update."""
    lw = cfg.model.loss_weights
    ppo_epochs = int(cfg.train.get("ppo_epochs", 4))
    ppo_clip = float(cfg.model.get("ppo_clip", 0.2))
    value_clip = float(cfg.model.get("ppo_value_clip", ppo_clip))
    max_grad_norm = float(cfg.model.get("max_grad_norm", 1.0))
    params = list(chain(actor.parameters(), critic.parameters()))

    loss = loss_policy = loss_value = loss_entropy = torch.tensor(0.0, device=rollout.states.device)
    entropy_mean = torch.tensor(0.0, device=rollout.states.device)
    for _ in range(ppo_epochs):
        dist = actor.distribution(rollout.states, rollout.scalars)
        logprobs = dist.log_prob(rollout.actions)
        entropy_mean = dist.entropy().mean()
        values = critic(rollout.states, rollout.scalars)

        ratio = torch.exp(logprobs - logprobs_old)
        clipped_ratio = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
        loss_policy = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        loss_entropy = -entropy_mean
        values_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
        value_unclipped = F.mse_loss(values, returns, reduction="none")
        value_clipped = F.mse_loss(values_clipped, returns, reduction="none")
        loss_value = 0.5 * torch.maximum(value_unclipped, value_clipped).mean()
        loss = lw.policy * loss_policy + lw.value * loss_value + lw.entropy * loss_entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_policy": float(loss_policy.item()),
        "loss_value": float(loss_value.item()),
        "loss_entropy": float(loss_entropy.item()),
        "entropy": float(entropy_mean.item()),
    }


def pair_train_step(
    actor: model.PairTradingActor,
    critic: model.PairTradingCritic,
    optimizer: torch.optim.Optimizer,
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    cfg: DictConfig,
) -> tuple[PairRollout, dict[str, float]]:
    rollout = collect_pair_rollout(
        actor,
        prices_a,
        prices_b,
        volumes_a,
        volumes_b,
        progress,
        rollout_size=int(cfg.data.rollout_size),
        transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        net_risk_penalty=float(cfg.model.get("net_risk_penalty", 0.0)),
        gross_risk_penalty=float(cfg.model.get("gross_risk_penalty", 0.0)),
        price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
    )
    with torch.no_grad():
        dist_old = actor.distribution(rollout.states, rollout.scalars)
        logprobs_old = dist_old.log_prob(rollout.actions)
        values_old = critic(rollout.states, rollout.scalars)
        advantages, returns = generalized_advantages(
            rollout.rewards, values_old, float(cfg.model.gamma), float(cfg.model.gae_lambda)
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = pair_ppo_update(
        actor, critic, optimizer, rollout, logprobs_old, values_old, returns, advantages, cfg
    )
    return rollout, losses
