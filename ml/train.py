"""PPO training for an autoregressive, fixed-window MLP trading policy."""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

import model
import utils
from dataset import OnlineBezierToyProvider, make_dataloader


logger = logging.getLogger("RL")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
torch.set_float32_matmul_precision("high")

FEATURE_NAMES = ("relative_log_price", "log_return", "normalized_log_volume", "session_progress", "position")
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class MarketRollout:
    states: torch.Tensor  # (batch, rollout, window, features)
    actions: torch.Tensor  # categorical indices (batch, rollout)
    positions: torch.Tensor  # held inventory in {-1, 0, +1} (batch, rollout)
    rewards: torch.Tensor
    price_returns: torch.Tensor
    costs: torch.Tensor
    risk_costs: torch.Tensor
    trades: torch.Tensor
    forced_closes: torch.Tensor


def _market_datetimes(secs: torch.Tensor | np.ndarray, anno: str) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = secs.detach().cpu().numpy() if torch.is_tensor(secs) else np.asarray(secs)
    shape = arr.shape
    dt = pd.to_datetime(arr.reshape(-1), unit="s", origin=anno, utc=True).tz_convert("US/Eastern")
    minutes = np.asarray(dt.hour * 60 + dt.minute, dtype=np.float32).reshape(shape)
    return minutes, shape


def regular_session_mask(secs: torch.Tensor | np.ndarray, anno: str = "2010-01-01") -> torch.Tensor | np.ndarray:
    minutes, shape = _market_datetimes(secs, anno)
    mask = ((minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)).reshape(shape)
    if torch.is_tensor(secs):
        return torch.as_tensor(mask, dtype=torch.bool, device=secs.device)
    return mask


def session_progress(secs: torch.Tensor, anno: str = "2010-01-01") -> torch.Tensor:
    """Map 09:30..16:00 Eastern linearly to -1..1."""
    minutes, _ = _market_datetimes(secs, anno)
    progress = (minutes - (9 * 60 + 30)) / (6.5 * 60)
    progress = np.clip(progress, 0.0, 1.0) * 2.0 - 1.0
    return torch.as_tensor(progress, dtype=torch.float32, device=secs.device)


def build_market_features(
    prices: torch.Tensor,
    volumes: torch.Tensor,
    progress: torch.Tensor,
    window_size: int,
    rollout_size: int,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """Build scale-invariant market features for all decisions in a rollout.

    Price channels are log ratios, never absolute prices. Volume is centered
    and scaled independently inside each window. The returned tensor excludes
    action history; that is filled online by :func:`collect_rollout`.
    """
    if prices.shape != volumes.shape or prices.shape != progress.shape:
        raise ValueError("prices, volumes, and progress must have identical (batch, ticks) shapes")
    required = int(window_size) + int(rollout_size)
    if prices.ndim != 2 or prices.shape[1] < required:
        raise ValueError(f"need at least {required} ticks per sample")
    if (prices <= 0).any() or (volumes < 0).any():
        raise ValueError("prices must be positive and volumes non-negative")

    n, t = int(window_size), int(rollout_size)
    price_windows = prices.unfold(1, n, 1)[:, :t]
    volume_windows = volumes.unfold(1, n, 1)[:, :t]
    progress_windows = progress.unfold(1, n, 1)[:, :t]

    log_prices = torch.log(price_windows)
    relative_price = (log_prices - log_prices[..., -1:]) * float(price_feature_scale)
    log_returns = F.pad(log_prices[..., 1:] - log_prices[..., :-1], (1, 0)) * float(price_feature_scale)

    log_volume = torch.log1p(volume_windows)
    volume_mean = log_volume.mean(dim=-1, keepdim=True)
    volume_std = log_volume.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized_volume = (log_volume - volume_mean) / volume_std

    return torch.stack((relative_price, log_returns, normalized_volume, progress_windows), dim=-1)


def market_rewards(
    positions: torch.Tensor,
    price_now: torch.Tensor,
    price_next: torch.Tensor,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    max_position: int = model.MAX_POSITION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mark positions to market and charge proportional turnover.

    Positions are unit short, flat, or unit long. A final liquidation charge
    is always applied when the rollout ends non-flat.
    """
    if positions.shape != price_now.shape or positions.shape != price_next.shape:
        raise ValueError("positions, price_now, and price_next must have the same shape")
    exposure = positions.to(torch.float32) / float(max_position)
    returns = price_next / price_now.clamp_min(1e-12) - 1.0
    previous = F.pad(exposure[:, :-1], (1, 0), value=0.0)
    turnover = (exposure - previous).abs()
    costs = turnover * float(transaction_cost)
    risk_costs = exposure.square() * float(risk_penalty)
    rewards = exposure * returns - costs - risk_costs

    liquidation_cost = exposure[:, -1].abs() * float(transaction_cost)
    rewards[:, -1] -= liquidation_cost
    costs[:, -1] += liquidation_cost
    info = {
        "exposure": exposure,
        "returns": returns,
        "costs": costs,
        "risk_costs": risk_costs,
        "trades": exposure.ne(previous),
        "forced_closes": exposure[:, -1].ne(0.0),
        "end_positions": torch.zeros_like(exposure[:, -1]),
        "pnl": exposure * (price_next - price_now),
    }
    return rewards, info


@torch.no_grad()
def collect_rollout(
    actor: model.TradingActor,
    prices: torch.Tensor,
    volumes: torch.Tensor,
    progress: torch.Tensor,
    rollout_size: int,
    transaction_cost: float = 0.0,
    risk_penalty: float = 0.0,
    sampling: str = "multinomial",
    price_feature_scale: float = 100.0,
) -> MarketRollout:
    """Run the policy sequentially, feeding each action into future states."""
    n = actor.window_size
    t_steps = int(rollout_size)
    market = build_market_features(prices, volumes, progress, n, t_steps, price_feature_scale)
    b = prices.shape[0]
    position_history = torch.zeros((b, n + t_steps), dtype=torch.float32, device=prices.device)
    current_position = torch.zeros(b, dtype=torch.long, device=prices.device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []

    for t in range(t_steps):
        action_window = position_history[:, t : t + n].unsqueeze(-1)
        state = torch.cat((market[:, t], action_window), dim=-1)
        action = actor.play(state, sampling=sampling)
        current_position = model.apply_action(current_position, action)
        states.append(state)
        actions.append(action)
        positions.append(current_position)
        # At the next observed tick, this records the position that was held
        # over the just-completed price interval.
        position_history[:, n + t] = current_position.to(torch.float32) / model.MAX_POSITION

    states_tensor = torch.stack(states, dim=1)
    actions_tensor = torch.stack(actions, dim=1)
    positions_tensor = torch.stack(positions, dim=1)
    price_now = prices[:, n - 1 : n - 1 + t_steps]
    price_next = prices[:, n : n + t_steps]
    rewards, info = market_rewards(
        positions_tensor, price_now, price_next, transaction_cost=transaction_cost, risk_penalty=risk_penalty
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


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for a rollout that terminates with mandatory liquidation."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.shape[0], device=rewards.device)
    next_value = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + float(gamma) * next_value - values[:, t]
        next_advantage = delta + float(gamma) * float(gae_lambda) * next_advantage
        advantages[:, t] = next_advantage
        next_value = values[:, t]
    return advantages, advantages + values


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Kept as a useful diagnostic and backwards-compatible helper."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + float(gamma) * running
        returns[:, t] = running
    return returns


def ppo_update(
    actor: model.TradingActor,
    critic: model.TradingCritic,
    optimizer: optim.Optimizer,
    rollout: MarketRollout,
    logprobs_old: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: DictConfig,
) -> dict[str, float]:
    lw = cfg.model.loss_weights
    ppo_epochs = int(cfg.train.get("ppo_epochs", 4))
    ppo_clip = float(cfg.model.get("ppo_clip", 0.2))
    value_clip = float(cfg.model.get("ppo_value_clip", ppo_clip))
    max_grad_norm = float(cfg.model.get("max_grad_norm", 1.0))
    params = list(chain(actor.parameters(), critic.parameters()))

    loss = loss_policy = loss_value = loss_entropy = torch.tensor(0.0, device=rollout.states.device)
    entropy_mean = torch.tensor(0.0, device=rollout.states.device)
    for _ in range(ppo_epochs):
        dist = actor.distribution(rollout.states)
        logprobs = dist.log_prob(rollout.actions)
        entropy_mean = dist.entropy().mean()
        values = critic(rollout.states)

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


def prepare_batch(batch: dict, device: torch.device | str, anno: str) -> tuple[torch.Tensor, ...]:
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    volumes = batch["volumes"].to(device=device, dtype=torch.float32, non_blocking=True)
    secs = batch["secs"].to(device=device, dtype=torch.long, non_blocking=True)
    return prices, volumes, secs, session_progress(secs, anno)


def train_step(
    actor: model.TradingActor,
    critic: model.TradingCritic,
    optimizer: optim.Optimizer,
    prices: torch.Tensor,
    volumes: torch.Tensor,
    progress: torch.Tensor,
    cfg: DictConfig,
) -> tuple[MarketRollout, dict[str, float]]:
    rollout = collect_rollout(
        actor,
        prices,
        volumes,
        progress,
        rollout_size=int(cfg.data.rollout_size),
        transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
        price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
    )
    with torch.no_grad():
        dist_old = actor.distribution(rollout.states)
        logprobs_old = dist_old.log_prob(rollout.actions)
        values_old = critic(rollout.states)
        advantages, returns = generalized_advantages(
            rollout.rewards, values_old, float(cfg.model.gamma), float(cfg.model.gae_lambda)
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = ppo_update(actor, critic, optimizer, rollout, logprobs_old, values_old, returns, advantages, cfg)
    return rollout, losses


def performance_metrics(rewards: torch.Tensor) -> dict[str, float]:
    """Summarize net return paths used by both training and diagnostics.

    Profit factor is total positive net timestep P&L divided by the absolute
    total negative net timestep P&L. Drawdown is the worst peak-to-trough loss
    within any rollout, with every rollout starting from zero equity.
    """
    if rewards.ndim != 2 or not rewards.numel():
        raise ValueError("rewards must have non-empty shape (batch, time)")
    gross_profit = rewards.clamp_min(0.0).sum()
    gross_loss = -rewards.clamp_max(0.0).sum()
    profit_factor = gross_profit / gross_loss.clamp_min(1e-12)

    cumulative = rewards.cumsum(dim=1)
    cumulative_with_origin = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative), dim=1)
    running_peak = cumulative_with_origin.cummax(dim=1).values[:, 1:]
    max_drawdown = (running_peak - cumulative).amax()
    return {
        "return": rewards.sum(dim=1).mean().item(),
        "profit_factor": profit_factor.item(),
        "max_drawdown": max_drawdown.item(),
    }


def rollout_metrics(rollout: MarketRollout) -> dict[str, float]:
    exposure = rollout.positions.float() / model.MAX_POSITION
    return {
        **performance_metrics(rollout.rewards),
        "reward": rollout.rewards.sum(dim=1).mean().item(),
        "gross_reward": (rollout.rewards + rollout.costs + rollout.risk_costs).sum(dim=1).mean().item(),
        "transaction_cost": rollout.costs.sum(dim=1).mean().item(),
        "risk_cost": rollout.risk_costs.sum(dim=1).mean().item(),
        "position_abs": exposure.abs().mean().item(),
        "trades": rollout.trades.float().sum(dim=1).mean().item(),
        "forced_closes": rollout.forced_closes.float().mean().item(),
        "long_fraction": rollout.positions.gt(0).float().mean().item(),
        "short_fraction": rollout.positions.lt(0).float().mean().item(),
        "flat_fraction": rollout.positions.eq(0).float().mean().item(),
    }


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    exp_dir = cfg.general.experiment_dir
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device(cfg.model.device)
    n, t = int(cfg.data.window_size), int(cfg.data.rollout_size)
    use_toy = bool(cfg.data.get("use_toy", False))
    toy_provider: OnlineBezierToyProvider | None = None
    iterator = None
    if use_toy:
        toy_cfg = cfg.data.get("toy", {})
        toy_provider = OnlineBezierToyProvider(
            window_size=n,
            rollout_size=t,
            noise_std=float(toy_cfg.get("noise_std", 3e-4)),
            flat_return_threshold=float(toy_cfg.get("flat_return_threshold", 2.5e-4)),
        )
        logger.info(
            "Using online Bezier toy data, window_size=%d, rollout_size=%d, batch_size=%d",
            n,
            t,
            int(cfg.train.batch_size),
        )
    else:
        loader = make_dataloader(cfg.data, cfg.train, int(cfg.train.get("seed", 0)))
        iterator = utils.cycle(loader)

    actor = model.TradingActor(**OmegaConf.to_container(cfg.model.actor, resolve=True)).to(device)
    critic = model.TradingCritic(**OmegaConf.to_container(cfg.model.critic, resolve=True)).to(device)
    optimizer = optim.Adam(chain(actor.parameters(), critic.parameters()), lr=float(cfg.model.lr))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.train.steps_num), eta_min=float(cfg.model.get("eta_min", 1e-6))
    )
    logger.info("MLP model summary, trainable_params_num=%.2fM", sum(p.numel() for p in chain(actor.parameters(), critic.parameters())) / 1e6)

    csv_logger = utils.CSVLogger(exp_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(exp_dir)
    start_step = 0
    if checkpoint_path:
        logger.info("Loading checkpoint, checkpoint_path=%s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_use_toy = bool(state.get("config", {}).get("data", {}).get("use_toy", False))
        if checkpoint_use_toy != use_toy:
            raise ValueError(
                "checkpoint data source does not match data.use_toy; use a different experiment_name"
            )
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))
    elif cfg.model.get("pretrained_path", None):
        state = torch.load(cfg.model.pretrained_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)

    total_steps = int(cfg.train.steps_num)
    logger.info("Starting training, global_step=%d, total_steps=%d", start_step, total_steps)
    for step in tqdm(range(start_step, total_steps), desc=cfg.general.experiment_name):
        if toy_provider is not None:
            toy_batch = toy_provider.sample(int(cfg.train.batch_size), device)
            prices, volumes, progress = toy_batch.prices, toy_batch.volumes, toy_batch.progress
        else:
            if iterator is None:
                raise RuntimeError("market data iterator was not initialized")
            batch = next(iterator)
            prices, volumes, secs, progress = prepare_batch(batch, device, cfg.data.anno)
            if cfg.data.get("enforce_market_hours", True):
                action_secs = secs[:, n - 1 : n - 1 + t]
                reward_secs = secs[:, n : n + t]
                if not regular_session_mask(action_secs, cfg.data.anno).all():
                    raise ValueError("action ticks must remain inside regular market hours")
                if not regular_session_mask(reward_secs, cfg.data.anno).all():
                    raise ValueError("reward/exit ticks must remain inside regular market hours")

        rollout, losses = train_step(actor, critic, optimizer, prices, volumes, progress, cfg)
        scheduler.step()
        if (step + 1) % int(cfg.loop.log_interval) == 0:
            csv_logger.write(
                {
                    "step": step + 1,
                    "stage": 0,
                    "timestamp": time.time(),
                    "lr": scheduler.get_last_lr()[0],
                    **rollout_metrics(rollout),
                    **losses,
                }
            )
        if (step + 1) % int(cfg.loop.checkpoint_interval) == 0:
            utils.save_checkpoint(
                exp_dir,
                step + 1,
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )
            logger.info("Saved checkpoint, step=%d", step + 1)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

if __name__ == "__main__":
    main(OmegaConf.load(parser.parse_args().config_path))
