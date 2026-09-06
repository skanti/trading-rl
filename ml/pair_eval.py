"""Benchmarks that isolate what the relative signal is actually worth.

A two-leg policy trades twice the instruments of a single-symbol policy, so
comparing its raw return against a one-leg number would reward it for nothing
more than deploying more capital. The reference points here all trade the same
two legs and are scored by the same pair reward, which leaves the use of the
relationship between the legs as the only difference between them:

``independent_leg_positions``
    the single-symbol policy run separately on each leg and combined. This is
    what the two symbols are worth without any relative information at all, and
    it is the benchmark the pair policy has to beat.
``zscore_rule_positions``
    the textbook pairs-trading rule on the same beta-neutral residual the
    policy observes. It shows what the relative signal gives up to a fixed rule,
    so any excess is what was learned rather than engineered.
``oracle_positions``
    the myopic optimum under the generating process, as an upper reference.

``spread_behaviour`` sits here too: it scores how the policy uses the residual
rather than what it earned, which is what separates a genuine relative-value
policy from one that happens to be long the same factor twice.
"""

from __future__ import annotations

import torch

from . import model
from .pair import (
    build_pair_features,
    pair_market_rewards,
    PairRollout,
)
from .train import build_market_features


def spread_behaviour(
    rollout,
    residual_z: torch.Tensor,
    is_mean_reverting: torch.Tensor,
    stretch: float = 1.0,
) -> dict[str, float]:
    """Measure whether the policy trades the residual the way it should.

    ``spread_direction_accuracy`` asks, over the ticks where the residual is
    stretched and the pair really does mean revert, how often the policy is
    positioned to profit from the residual coming back. ``regime_discrimination``
    contrasts how much spread exposure it takes on mean-reverting pairs against
    how much it takes on broken ones; a policy that cannot tell them apart
    scores zero.
    """
    exposure_a = rollout.positions_a.float()
    exposure_b = rollout.positions_b.float()
    # Positive when the policy is long the residual (long A against short B).
    spread_exposure = 0.5 * (exposure_a - exposure_b)
    reverting = is_mean_reverting.unsqueeze(-1).expand_as(residual_z)

    stretched = residual_z.abs() > stretch
    tradable = stretched & reverting & spread_exposure.ne(0.0)
    # A stretched residual is expected to fall, so the profitable side is short.
    correct = spread_exposure.sign().eq(-residual_z.sign())
    accuracy = correct[tradable].float().mean().item() if tradable.any() else float("nan")

    reverting_exposure = spread_exposure.abs()[reverting].mean().item() if reverting.any() else 0.0
    broken_exposure = spread_exposure.abs()[~reverting].mean().item() if (~reverting).any() else 0.0
    return {
        "spread_direction_accuracy": accuracy,
        "spread_exposure_mean_reverting": reverting_exposure,
        "spread_exposure_broken": broken_exposure,
        "regime_discrimination": reverting_exposure - broken_exposure,
    }


@torch.no_grad()
def independent_leg_positions(
    actor: model.TradingActor,
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    rollout_size: int,
    sampling: str = "greedy",
    price_feature_scale: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single-symbol policy on each leg with no knowledge of the other."""
    positions = []
    for prices, volumes in ((prices_a, volumes_a), (prices_b, volumes_b)):
        n, t_steps = actor.window_size, int(rollout_size)
        market = build_market_features(prices, volumes, progress, n, t_steps, price_feature_scale)
        batch = prices.shape[0]
        history = torch.zeros((batch, n + t_steps), dtype=torch.float32, device=prices.device)
        position = torch.zeros(batch, dtype=torch.long, device=prices.device)
        leg_positions = []
        for t in range(t_steps):
            state = torch.cat((market[:, t], history[:, t : t + n].unsqueeze(-1)), dim=-1)
            position = model.apply_action(position, actor.play(state, sampling=sampling))
            leg_positions.append(position)
            history[:, n + t] = position.to(torch.float32) / model.MAX_POSITION
        positions.append(torch.stack(leg_positions, dim=1))
    return positions[0], positions[1]


def step_limited_positions(targets: torch.Tensor) -> torch.Tensor:
    """Project a desired position path onto what the action space can reach.

    One command moves inventory by one step, so a direct reversal costs two
    ticks. A benchmark that flips sign in a single tick would be competing with
    an ability the policy does not have, which is why every reachable reference
    is passed through here first.
    """
    batch, steps = targets.shape
    reachable = torch.zeros_like(targets)
    position = torch.zeros(batch, dtype=targets.dtype, device=targets.device)
    for t in range(steps):
        step = (targets[:, t] - position).clamp(-1, 1)
        position = (position + step).clamp(-model.MAX_POSITION, model.MAX_POSITION)
        reachable[:, t] = position
    return reachable


def zscore_rule_positions(
    residual_z: torch.Tensor,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classic threshold rule on the beta-neutral residual.

    A high residual means leg A is rich against leg B, so the spread is sold by
    shorting A and buying B. The position is carried until the residual comes
    back inside ``exit_threshold``, which is the standard entry/exit hysteresis.
    The result is step-limited so the rule pays the same two-tick reversal cost
    the policy pays.
    """
    if entry_threshold <= exit_threshold:
        raise ValueError("entry_threshold must exceed exit_threshold")
    batch, steps = residual_z.shape
    desired = torch.zeros_like(residual_z, dtype=torch.long)
    state = torch.zeros(batch, dtype=torch.long, device=residual_z.device)
    for t in range(steps):
        z = residual_z[:, t]
        state = torch.where(z > entry_threshold, torch.full_like(state, -1), state)
        state = torch.where(z < -entry_threshold, torch.full_like(state, 1), state)
        state = torch.where(z.abs() < exit_threshold, torch.zeros_like(state), state)
        desired[:, t] = state
    positions_a = step_limited_positions(desired)
    return positions_a, step_limited_positions(-desired)


def oracle_positions(
    target_positions_a: torch.Tensor,
    target_positions_b: torch.Tensor,
    step_limited: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Myopic optimum implied by the generating process.

    This is an upper reference, not an attainable strategy: it reads the
    noise-free drift of both legs. Step-limiting it keeps it inside the action
    space, but it still sees the future.
    """
    if not step_limited:
        return target_positions_a, target_positions_b
    return step_limited_positions(target_positions_a), step_limited_positions(target_positions_b)


@torch.no_grad()
def score_positions(
    positions_a: torch.Tensor,
    positions_b: torch.Tensor,
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    window_size: int,
    rollout_size: int,
    transaction_cost: float,
    net_risk_penalty: float,
    gross_risk_penalty: float,
    price_feature_scale: float = 100.0,
    hedge_ratio: torch.Tensor | None = None,
) -> PairRollout:
    """Score any pair of position paths with the policy's own reward."""
    n, t = int(window_size), int(rollout_size)
    if hedge_ratio is None:
        _, _, hedge_ratio = build_pair_features(
            prices_a, prices_b, volumes_a, volumes_b, progress, n, t, price_feature_scale
        )
    rewards, info = pair_market_rewards(
        positions_a,
        positions_b,
        prices_a[:, n - 1 : n - 1 + t],
        prices_a[:, n : n + t],
        prices_b[:, n - 1 : n - 1 + t],
        prices_b[:, n : n + t],
        hedge_ratio,
        transaction_cost=transaction_cost,
        net_risk_penalty=net_risk_penalty,
        gross_risk_penalty=gross_risk_penalty,
    )
    # These reference paths never reach PPO, and a position path does not
    # determine a unique command sequence, so no actions are reconstructed.
    empty = torch.empty(0, device=prices_a.device)
    return PairRollout(
        states=empty,
        scalars=empty,
        actions=empty,
        positions_a=positions_a,
        positions_b=positions_b,
        hedge_ratio=hedge_ratio,
        rewards=rewards,
        costs=info["costs"],
        risk_costs=info["risk_costs"],
        trades=info["trades"],
        net_exposure=info["net_exposure"],
        basket_pnl=info["basket_pnl"],
        spread_pnl=info["spread_pnl"],
    )


def current_residual_z(
    prices_a: torch.Tensor,
    prices_b: torch.Tensor,
    volumes_a: torch.Tensor,
    volumes_b: torch.Tensor,
    progress: torch.Tensor,
    window_size: int,
    rollout_size: int,
    price_feature_scale: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual z-score at each decision tick, plus the hedge ratio used."""
    market, _, hedge_ratio = build_pair_features(
        prices_a,
        prices_b,
        volumes_a,
        volumes_b,
        progress,
        int(window_size),
        int(rollout_size),
        price_feature_scale,
    )
    # Channel 6 is spread_z; its last tick is the residual at the decision.
    return market[:, :, -1, 6], hedge_ratio
