"""Shifted-window MLP policy that trades one Mon--Fri week of 10-minute bars.

The intraday policy is liquidated every day at 16:00. This one holds inventory
across the four weeknights and is liquidated once, at Friday's close, so the
weekend is always flat while the reachable holding period grows from hours to
days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

import model
from reference_mlp import MLP_REFERENCE_FEATURE_DIM, build_shifted_price_features
from train import MarketRollout, generalized_advantages, market_rewards, ppo_update
from week_dataset import SESSIONS_PER_WEEK, ticks_per_session


WEEK_FEATURE_NAMES = (
    "asset_relative_log_price",
    "spy_relative_log_price",
    "asset_log_return",
    "spy_log_return",
    "asset_inventory",
    "previous_action_buy",
    "previous_action_nothing",
    "previous_action_sell",
)
WEEK_FEATURE_DIM = len(WEEK_FEATURE_NAMES)
# Two clocks and two week-to-date anchors. ``time_to_day_close`` reaches zero at
# each 16:00 decision, which is exactly the decision that chooses whether to
# carry inventory overnight; the window alone does not mark that boundary.
WEEK_SCALAR_NAMES = (
    "time_to_week_close",
    "time_to_day_close",
    "asset_week_to_date_return",
    "spy_week_to_date_return",
)
WEEK_SCALAR_DIM = len(WEEK_SCALAR_NAMES)

if WEEK_FEATURE_DIM != MLP_REFERENCE_FEATURE_DIM:
    raise AssertionError("week features must match the intraday reference layout")


def build_week_scalars(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    context_ticks: int,
    rollout_size: int,
    session_ticks: int,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """Per-decision clocks and Monday-anchored returns, shape (batch, steps, 4).

    Every value at step ``k`` is known at the tick the decision is made on, so
    nothing here looks ahead of the window the policy already sees.
    """
    context_ticks, steps = int(context_ticks), int(rollout_size)
    session_ticks = int(session_ticks)
    if session_ticks < 2:
        raise ValueError("a session must contain at least two grid points")
    if prices.shape != reference_prices.shape or prices.ndim != 2:
        raise ValueError("asset and reference prices must share shape (batch, sequence)")
    if prices.shape[1] < context_ticks + steps + 1:
        raise ValueError("price sequence is shorter than the configured week")

    device = prices.device
    index = torch.arange(steps, device=device, dtype=torch.float32)
    time_to_week_close = (float(steps) - index) / float(steps)
    offset = index.remainder(float(session_ticks))
    time_to_day_close = (float(session_ticks - 1) - offset) / float(session_ticks - 1)

    scale = float(price_feature_scale)
    decisions = slice(context_ticks, context_ticks + steps)
    asset_log = torch.log(prices[:, decisions].float())
    spy_log = torch.log(reference_prices[:, decisions].float())
    asset_week = (asset_log - asset_log[:, :1]) * scale
    spy_week = (spy_log - spy_log[:, :1]) * scale

    clocks = torch.stack((time_to_week_close, time_to_day_close), dim=-1)
    clocks = clocks.unsqueeze(0).expand(prices.shape[0], -1, -1)
    return torch.cat((clocks, asset_week.unsqueeze(-1), spy_week.unsqueeze(-1)), dim=-1)


def validate_week_hours(
    secs: torch.Tensor | np.ndarray,
    context_ticks: int,
    rollout_size: int,
    tick_minutes: int,
    session_ticks: int,
    anno: str = "2010-01-01",
) -> None:
    """Fail loudly if any traded tick leaves Mon--Fri regular hours.

    This is the check that makes the weekend constraint structural rather than
    incidental: the grid the policy acts on never contains a Saturday or Sunday,
    and its final priced tick is a Friday 16:00 close, where the rollout
    liquidates.
    """
    arr = secs.detach().cpu().numpy() if torch.is_tensor(secs) else np.asarray(secs)
    if arr.ndim != 2:
        raise ValueError("secs must have shape (batch, sequence)")
    context_ticks, steps = int(context_ticks), int(rollout_size)
    traded = arr[:, context_ticks : context_ticks + steps + 1]
    if traded.shape[1] != steps + 1:
        raise ValueError("secs is too short for the configured week")

    stamps = pd.to_datetime(traded.reshape(-1), unit="s", origin=anno, utc=True)
    stamps = stamps.tz_convert("US/Eastern")
    weekday = np.asarray(stamps.dayofweek).reshape(traded.shape)
    minutes = np.asarray(stamps.hour * 60 + stamps.minute).reshape(traded.shape)
    if (weekday > 4).any():
        raise ValueError("week rollouts must not contain weekend ticks")
    if ((minutes < 9 * 60 + 30) | (minutes > 16 * 60)).any():
        raise ValueError("week rollout ticks must remain inside regular market hours")
    if np.asarray(stamps.second).any():
        raise ValueError("week rollout ticks must land on exact minutes")
    if ((minutes - (9 * 60 + 30)) % int(tick_minutes)).any():
        raise ValueError(f"week rollout ticks must sit on the {int(tick_minutes)}-minute grid")
    if not (weekday[:, 0] == 0).all() or not (minutes[:, 0] == 9 * 60 + 30).all():
        raise ValueError("week rollouts must begin at Monday 09:30")
    if not (weekday[:, -1] == 4).all() or not (minutes[:, -1] == 16 * 60).all():
        raise ValueError("week rollouts must end at Friday 16:00")
    session_ticks = int(session_ticks)
    sessions, remainder = divmod(traded.shape[1], session_ticks)
    if remainder or sessions != SESSIONS_PER_WEEK:
        raise ValueError(
            f"a week must hold {SESSIONS_PER_WEEK} sessions of {session_ticks} ticks"
        )
    if (weekday[:, ::session_ticks] != np.arange(sessions)).any():
        raise ValueError("week sessions must run Monday through Friday in order")


@torch.no_grad()
def collect_week_rollout(
    actor: model.TradingActor,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    context_ticks: int,
    rollout_size: int,
    session_ticks: int,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    sampling: str = "multinomial",
    price_feature_scale: float = 100.0,
) -> MarketRollout:
    """Sample one command per 10-minute bar across a complete trading week."""
    window_size = actor.window_size
    steps = int(rollout_size)
    if actor.feature_dim != WEEK_FEATURE_DIM:
        raise ValueError(f"week actor requires feature_dim={WEEK_FEATURE_DIM}")
    if actor.scalar_dim != WEEK_SCALAR_DIM:
        raise ValueError(f"week actor requires scalar_dim={WEEK_SCALAR_DIM}")

    market = build_shifted_price_features(
        prices,
        reference_prices,
        context_ticks,
        window_size,
        steps,
        price_feature_scale,
    )
    scalars = build_week_scalars(
        prices,
        reference_prices,
        context_ticks,
        steps,
        session_ticks,
        price_feature_scale,
    )

    batch = prices.shape[0]
    device = prices.device
    # History channels stay zero through the context: the policy sees prices
    # there, not fabricated no-op commands it never issued.
    inventory_history = torch.zeros(
        (batch, window_size + steps), dtype=torch.float32, device=device
    )
    action_history = torch.zeros(
        (batch, window_size + steps, model.ACTION_DIM), dtype=torch.float32, device=device
    )
    current_position = torch.zeros(batch, dtype=torch.long, device=device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []

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
        action = actor.play(state, scalars[:, step], sampling=sampling)
        current_position = model.apply_action(current_position, action)
        states.append(state)
        actions.append(action)
        positions.append(current_position)

        next_index = window_size + step
        inventory_history[:, next_index] = current_position.to(torch.float32)
        action_history[:, next_index] = torch.nn.functional.one_hot(
            action, num_classes=model.ACTION_DIM
        ).to(torch.float32)

    states_tensor = torch.stack(states, dim=1)
    positions_tensor = torch.stack(positions, dim=1)
    context_ticks = int(context_ticks)
    price_now = prices[:, context_ticks : context_ticks + steps]
    price_next = prices[:, context_ticks + 1 : context_ticks + steps + 1]
    # ``market_rewards`` liquidates whatever is held after the final interval,
    # which here is Friday's 16:00 close. Nothing survives into the weekend.
    rewards, info = market_rewards(
        positions_tensor,
        price_now,
        price_next,
        transaction_cost=transaction_cost,
        risk_penalty=risk_penalty,
    )
    return MarketRollout(
        states=states_tensor,
        actions=torch.stack(actions, dim=1),
        positions=positions_tensor,
        rewards=rewards,
        price_returns=info["returns"],
        costs=info["costs"],
        risk_costs=info["risk_costs"],
        trades=info["trades"],
        forced_closes=info["forced_closes"],
        scalars=scalars,
    )


def overnight_metrics(positions: torch.Tensor, session_ticks: int) -> dict[str, float]:
    """Report how often inventory is actually carried between sessions."""
    session_ticks = int(session_ticks)
    steps = positions.shape[1]
    boundaries = torch.arange(
        session_ticks - 1, steps, session_ticks, device=positions.device
    )
    if not boundaries.numel():
        return {"overnight_fraction": 0.0, "overnight_holds": 0.0}
    held = positions.index_select(1, boundaries).ne(0).float()
    return {
        "overnight_fraction": held.mean().item(),
        "overnight_holds": held.sum(dim=1).mean().item(),
    }


def week_train_step(
    actor: model.TradingActor,
    critic: model.TradingCritic,
    optimizer: torch.optim.Optimizer,
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    cfg: DictConfig,
) -> tuple[MarketRollout, dict[str, float]]:
    rollout = collect_week_rollout(
        actor,
        prices,
        reference_prices,
        context_ticks=int(cfg.data.context_ticks),
        rollout_size=int(cfg.data.rollout_size),
        session_ticks=ticks_per_session(int(cfg.data.tick_minutes)),
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
