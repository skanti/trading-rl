import argparse
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from torch.distributions.categorical import Categorical
from tqdm import tqdm

import llama
import utils
from dataset import make_dataloader

logger = logging.getLogger("RL")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

torch.set_float32_matmul_precision("high")

ACTION_SHORT = 0
ACTION_HOLD = 1
ACTION_LONG = 2
ACTION_DIM = 3


@dataclass
class RolloutBatch:
    ctx: torch.Tensor
    seq: torch.Tensor
    ts: torch.Tensor
    price_now: torch.Tensor
    price_next: torch.Tensor
    secs_now: torch.Tensor
    secs_next: torch.Tensor
    batch_size: int
    rollout_size: int


class LlamaActor(nn.Module):
    def __init__(self, config: llama.ModelArgs, action_dim: int = ACTION_DIM):
        super().__init__()
        self.backbone = llama.Transformer(config)
        self.action_head = nn.Linear(config.dim, action_dim)
        nn.init.zeros_(self.action_head.bias)
        nn.init.normal_(self.action_head.weight, mean=0.0, std=0.02)

    def encode_last(self, ctx: torch.Tensor, seq: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        assert seq.ndim == 2
        assert ts.ndim == 3
        assert ctx.ndim == 2
        max_seq_length = ctx.shape[1] + seq.shape[1]
        dtype = self.backbone.output.weight.dtype
        self.backbone.setup_caches(
            max_batch_size=seq.shape[0],
            max_seq_length=max_seq_length,
            dtype=dtype,
        )
        x = torch.cat([self.backbone.ctx_embeddings(ctx), self.backbone.tok_embeddings(seq.long(), ts.long())], dim=1)
        hidden = self.backbone.forward_hidden(x)
        return hidden[:, -1]

    def forward(self, ctx: torch.Tensor, seq: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.encode_last(ctx=ctx, seq=seq, ts=ts))

    def distribution(self, ctx: torch.Tensor, seq: torch.Tensor, ts: torch.Tensor) -> Categorical:
        return Categorical(logits=self(ctx=ctx, seq=seq, ts=ts))


class LlamaCritic(nn.Module):
    def __init__(self, config: llama.ModelArgs):
        super().__init__()
        self.backbone = llama.Transformer(config)
        self.value_head = nn.Linear(config.dim, 1)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def encode_last(self, ctx: torch.Tensor, seq: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        assert seq.ndim == 2
        assert ts.ndim == 3
        assert ctx.ndim == 2
        max_seq_length = ctx.shape[1] + seq.shape[1]
        dtype = self.backbone.output.weight.dtype
        self.backbone.setup_caches(
            max_batch_size=seq.shape[0],
            max_seq_length=max_seq_length,
            dtype=dtype,
        )
        x = torch.cat([self.backbone.ctx_embeddings(ctx), self.backbone.tok_embeddings(seq.long(), ts.long())], dim=1)
        hidden = self.backbone.forward_hidden(x)
        return hidden[:, -1]

    def forward(self, ctx: torch.Tensor, seq: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encode_last(ctx=ctx, seq=seq, ts=ts)).squeeze(-1)


def make_model_args(cfg_model: DictConfig, key: str) -> llama.ModelArgs:
    cfg = cfg_model.get(key, None)
    if cfg is None:
        cfg = cfg_model.transformer
    if isinstance(cfg, DictConfig) and "transformer" in cfg:
        cfg = cfg.transformer
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw = dict(raw)
    min_block_size = int(cfg_model.ctx_size) + int(cfg_model.seq_size) * int(cfg_model.channels)
    raw["block_size"] = max(int(raw.get("block_size", min_block_size)), min_block_size)
    raw["vocab_size"] = int(raw.get("vocab_size", cfg_model.vocab_size))
    return llama.ModelArgs(**raw)


def batch_to_rollout(batch: dict, cfg_data: DictConfig, device: torch.device | str) -> RolloutBatch:
    seq_size = int(cfg_data.seq_size)
    channels = int(cfg_data.channels)
    rollout_size = int(cfg_data.rollout_size)
    context_tokens = seq_size * channels

    ctx = batch["ctx"].to(device=device, dtype=torch.float32, non_blocking=True)
    seq = batch["seq"].to(device=device, dtype=torch.long, non_blocking=True)
    ts = batch["ts"].to(device=device, dtype=torch.long, non_blocking=True)
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    secs = batch["secs"].to(device=device, dtype=torch.long, non_blocking=True)

    required_tokens = (seq_size + rollout_size) * channels
    required_prices = seq_size + rollout_size
    assert seq.shape[1] >= required_tokens, f"Need {required_tokens} tokens, got {seq.shape[1]}"
    assert prices.shape[1] >= required_prices, f"Need {required_prices} prices, got {prices.shape[1]}"

    seq_windows = []
    ts_windows = []
    for t in range(rollout_size):
        start = t * channels
        end = start + context_tokens
        seq_windows.append(seq[:, start:end])
        ts_windows.append(ts[:, start:end])

    b = seq.shape[0]
    seq_ctx = torch.stack(seq_windows, dim=1).reshape(b * rollout_size, context_tokens)
    ts_ctx = torch.stack(ts_windows, dim=1).reshape(b * rollout_size, context_tokens, ts.shape[-1])
    ctx_ctx = ctx[:, None, :].expand(b, rollout_size, ctx.shape[-1]).reshape(b * rollout_size, ctx.shape[-1])

    price_now = prices[:, seq_size - 1 : seq_size - 1 + rollout_size]
    price_next = prices[:, seq_size : seq_size + rollout_size]
    secs_now = secs[:, seq_size - 1 : seq_size - 1 + rollout_size]
    secs_next = secs[:, seq_size : seq_size + rollout_size]

    return RolloutBatch(
        ctx=ctx_ctx,
        seq=seq_ctx,
        ts=ts_ctx,
        price_now=price_now,
        price_next=price_next,
        secs_now=secs_now,
        secs_next=secs_next,
        batch_size=b,
        rollout_size=rollout_size,
    )


def regular_session_mask(secs: torch.Tensor | np.ndarray, anno: str = "2010-01-01") -> torch.Tensor | np.ndarray:
    is_tensor = torch.is_tensor(secs)
    device = secs.device if is_tensor else None
    arr = secs.detach().cpu().numpy() if is_tensor else np.asarray(secs)
    shape = arr.shape
    dt = pd.to_datetime(arr.reshape(-1), unit="s", origin=anno, utc=True).tz_convert("US/Eastern")
    minutes = np.asarray(dt.hour * 60 + dt.minute)
    mask = ((minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)).reshape(shape)
    if is_tensor:
        return torch.as_tensor(mask, dtype=torch.bool, device=device)
    return mask


def actions_to_positions(actions: torch.Tensor) -> torch.Tensor:
    assert actions.ndim == 2
    return actions.to(torch.float32) - 1.0


def market_rewards(
    actions: torch.Tensor,
    price_now: torch.Tensor,
    price_next: torch.Tensor,
    transaction_cost: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positions = actions_to_positions(actions)
    returns = (price_next - price_now) / price_now.clamp_min(1e-6)
    prev_positions = F.pad(positions[:, :-1], (1, 0), value=0.0)
    trade_cost = positions.sub(prev_positions).abs() * float(transaction_cost)
    rewards = positions * returns - trade_cost

    closing_cost = positions[:, -1].abs() * float(transaction_cost)
    rewards[:, -1] = rewards[:, -1] - closing_cost
    forced_closes = positions[:, -1].ne(0.0)

    info = {
        "positions": positions,
        "end_positions": torch.zeros_like(positions[:, -1]),
        "returns": returns,
        "trade_cost": trade_cost,
        "forced_closes": forced_closes,
        "trades": positions.sub(prev_positions).ne(0.0),
        "pnl": positions * (price_next - price_now),
    }
    return rewards, info


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    next_return = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        next_return = rewards[:, t] + gamma * next_return
        returns[:, t] = next_return
    return returns


def ppo_update(
    actor: LlamaActor,
    critic: LlamaCritic,
    optimizer: optim.Optimizer,
    rollout: RolloutBatch,
    actions: torch.Tensor,
    logprobs_old: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: DictConfig,
) -> dict[str, float]:
    lw = cfg.model.loss_weights
    ppo_epochs = int(cfg.train.get("ppo_epochs", 4))
    ppo_clip = float(cfg.model.get("ppo_clip", 0.1))
    ppo_value_clip = float(cfg.model.get("ppo_value_clip", ppo_clip))
    max_grad_norm = float(cfg.model.get("max_grad_norm", 1.0))

    actions_flat = actions.reshape(-1)
    logprobs_old_flat = logprobs_old.reshape(-1)
    values_old_flat = values_old.reshape(-1)
    returns_flat = returns.reshape(-1)
    advantages_flat = advantages.reshape(-1)

    loss_policy = torch.tensor(0.0, device=rollout.seq.device)
    loss_entropy = torch.tensor(0.0, device=rollout.seq.device)
    loss_value = torch.tensor(0.0, device=rollout.seq.device)
    loss = torch.tensor(0.0, device=rollout.seq.device)
    params = list(actor.parameters()) + list(critic.parameters())

    for _ in range(ppo_epochs):
        dist = actor.distribution(ctx=rollout.ctx, seq=rollout.seq, ts=rollout.ts)
        logprobs = dist.log_prob(actions_flat)
        entropy = dist.entropy()
        values = critic(ctx=rollout.ctx, seq=rollout.seq, ts=rollout.ts)

        ratio = torch.exp(logprobs - logprobs_old_flat)
        clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
        loss_policy = -torch.min(ratio * advantages_flat, clipped * advantages_flat).mean()
        loss_entropy = -entropy.mean()

        values_clipped = values_old_flat + torch.clamp(values - values_old_flat, -ppo_value_clip, ppo_value_clip)
        value_loss_unclipped = F.l1_loss(values, returns_flat, reduction="none")
        value_loss_clipped = F.l1_loss(values_clipped, returns_flat, reduction="none")
        loss_value = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        loss = loss_policy * lw.policy + loss_value * lw.value + loss_entropy * lw.entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

    return {
        "loss": loss.item(),
        "loss_policy": loss_policy.item(),
        "loss_entropy": loss_entropy.item(),
        "loss_value": loss_value.item(),
    }


def main(cfg: DictConfig) -> None:
    exp_dir = cfg.general.experiment_dir
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device(cfg.model.device)

    train_loader = make_dataloader(cfg_data=cfg.data, cfg_split=cfg.train, seed=int(cfg.train.get("seed", 0)))
    train_iter = utils.cycle(train_loader)

    actor = LlamaActor(make_model_args(cfg.model, "actor")).to(device)
    critic = LlamaCritic(make_model_args(cfg.model, "critic")).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params, lr=float(cfg.model.lr))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.steps_num),
        eta_min=float(cfg.model.get("eta_min", 1e-6)),
    )

    trainable_params_num = sum(p.numel() for p in params if p.requires_grad)
    logger.info(f"Model summary, trainable_params_num={trainable_params_num/1e6:0.2f}M")

    csv_logger = utils.CSVLogger(exp_dir=exp_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(exp_dir)

    global_step = 0
    if checkpoint_path:
        logger.info(f"Loading checkpoint, checkpoint_path={checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(state_dict["actor"], strict=True)
        critic.load_state_dict(state_dict["critic"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        global_step = int(state_dict.get("global_step", 0))
    elif cfg.model.get("pretrained_path", None):
        logger.info(f"Loading pretrained model, pretrained_path={cfg.model.pretrained_path}")
        state_dict = torch.load(cfg.model.pretrained_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(state_dict["actor"], strict=True)
        critic.load_state_dict(state_dict["critic"], strict=True)
    else:
        logger.info("No checkpoint found, training from scratch")

    logger.info(f"Starting training, global_step={global_step}")
    steps_end = global_step + int(cfg.train.steps_num)
    for step in tqdm(range(global_step, steps_end), desc=cfg.general.experiment_name):
        should_log = (step + 1) % int(cfg.loop.log_interval) == 0

        batch = next(train_iter)
        rollout = batch_to_rollout(batch=batch, cfg_data=cfg.data, device=device)

        if cfg.data.get("enforce_market_hours", True):
            assert regular_session_mask(rollout.secs_now, cfg.data.anno).all(), "Action ticks must be inside RTH"
            assert regular_session_mask(rollout.secs_next, cfg.data.anno).all(), "Exit/reward ticks must be inside RTH"

        with torch.no_grad():
            dist_old = actor.distribution(ctx=rollout.ctx, seq=rollout.seq, ts=rollout.ts)
            actions_flat = dist_old.sample()
            logprobs_old_flat = dist_old.log_prob(actions_flat)
            values_old_flat = critic(ctx=rollout.ctx, seq=rollout.seq, ts=rollout.ts)

        actions = actions_flat.reshape(rollout.batch_size, rollout.rollout_size)
        logprobs_old = logprobs_old_flat.reshape(rollout.batch_size, rollout.rollout_size)
        values_old = values_old_flat.reshape(rollout.batch_size, rollout.rollout_size)

        rewards, trade_info = market_rewards(
            actions=actions,
            price_now=rollout.price_now,
            price_next=rollout.price_next,
            transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
        )
        returns = discounted_returns(rewards=rewards, gamma=float(cfg.model.gamma))
        advantages = returns - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        losses = ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            rollout=rollout,
            actions=actions,
            logprobs_old=logprobs_old,
            values_old=values_old,
            returns=returns,
            advantages=advantages,
            cfg=cfg,
        )
        lr_scheduler.step()

        if should_log:
            positions = trade_info["positions"]
            metrics = {
                "step": step,
                "timestamp": time.time(),
                "lr": lr_scheduler.get_last_lr()[0],
                "reward": rewards.mean().item(),
                "pnl": trade_info["pnl"].sum(dim=1).mean().item(),
                "position_abs": positions.abs().mean().item(),
                "trades": trade_info["trades"].float().sum(dim=1).mean().item(),
                "forced_closes": trade_info["forced_closes"].float().mean().item(),
                **losses,
            }
            csv_logger.write(metrics)

        if (step + 1) % int(cfg.loop.checkpoint_interval) == 0:
            state_dict = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            utils.save_checkpoint(out_dir=exp_dir, global_step=step + 1, state_dict=state_dict)
            logger.info(f"Saving checkpoint, step={step + 1}")


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    main(cfg)
