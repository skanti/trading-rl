"""PPO trainer for the price-only, shifted-window asset/SPY MLP."""

from __future__ import annotations

import logging
import os
import time
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

from . import model, utils
from .reference_dataset import make_reference_dataloader
from .reference_mlp import (
    MLP_REFERENCE_FEATURE_DIM,
    MLP_REFERENCE_FEATURE_NAMES,
    MLP_REFERENCE_SCALAR_NAMES,
    collect_shifted_mlp_rollout,
    shifted_mlp_train_step,
)
from .train import regular_session_mask, rollout_metrics


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("REFERENCE_MLP")
MAX_MLP_PARAMETERS = 20_000_000


def prepare_price_batch(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    reference_prices = batch["reference_prices"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    secs = batch["secs"].to(device=device, dtype=torch.long, non_blocking=True)
    return prices, reference_prices, secs


def validate_market_hours(
    secs: torch.Tensor, context_ticks: int, steps: int, anno: str
) -> None:
    if not regular_session_mask(secs[:, context_ticks : context_ticks + steps], anno).all():
        raise ValueError("action ticks must remain inside regular market hours")
    if not regular_session_mask(
        secs[:, context_ticks + 1 : context_ticks + steps + 1], anno
    ).all():
        raise ValueError("reward/exit ticks must remain inside regular market hours")


def build_mlp_models(
    cfg: DictConfig, device: torch.device
) -> tuple[model.TradingActor, model.TradingCritic]:
    config = OmegaConf.to_container(cfg.model.mlp, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("model.mlp must be a mapping")
    if int(config.get("feature_dim", -1)) != MLP_REFERENCE_FEATURE_DIM:
        raise ValueError(f"model.mlp.feature_dim must be {MLP_REFERENCE_FEATURE_DIM}")
    actor = model.TradingActor(**config).to(device)
    critic_config = dict(config)
    critic_config.pop("action_dim", None)
    critic = model.TradingCritic(**critic_config).to(device)
    parameters = sum(parameter.numel() for parameter in chain(actor.parameters(), critic.parameters()))
    if parameters >= MAX_MLP_PARAMETERS:
        raise ValueError(
            f"MLP actor+critic have {parameters:,} parameters; they must remain below "
            f"{MAX_MLP_PARAMETERS:,}"
        )
    return actor, critic


@torch.no_grad()
def validation_metrics(
    actor: model.TradingActor,
    iterator,
    batches: int,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for _ in range(batches):
        prices, reference_prices, secs = prepare_price_batch(next(iterator), device)
        validate_market_hours(
            secs,
            int(cfg.data.context_ticks),
            int(cfg.data.rollout_size),
            str(cfg.data.anno),
        )
        rollout = collect_shifted_mlp_rollout(
            actor,
            prices,
            reference_prices,
            context_ticks=int(cfg.data.context_ticks),
            rollout_size=int(cfg.data.rollout_size),
            transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
            risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
            sampling="greedy",
            price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
        )
        for key, value in rollout_metrics(rollout).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / batches for key, value in totals.items()}


def main(cfg: DictConfig) -> None:
    if str(cfg.model.get("architecture", "mlp")).lower() != "mlp":
        raise ValueError("model.architecture must be 'mlp'")
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    device = torch.device(str(cfg.model.device))
    experiment_dir = str(cfg.general.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    context_ticks = int(cfg.data.context_ticks)
    rollout_size = int(cfg.data.rollout_size)
    window_size = int(cfg.model.mlp.window_size)
    if window_size > context_ticks + 1:
        raise ValueError("model.mlp.window_size exceeds available context plus 09:30")

    train_loader = make_reference_dataloader(cfg.data, cfg.train, int(cfg.train.seed))
    validation_loader = make_reference_dataloader(cfg.data, cfg.val, int(cfg.train.seed))
    train_iterator = utils.cycle(train_loader)
    validation_iterator = utils.cycle(validation_loader)
    actor, critic = build_mlp_models(cfg, device)
    parameters = list(chain(actor.parameters(), critic.parameters()))
    optimizer = optim.AdamW(
        parameters,
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.get("weight_decay", 0.01)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.steps_num),
        eta_min=float(cfg.model.get("eta_min", 1e-6)),
    )
    logger.info(
        "Shifted-window MLP summary, trainable_params_num=%.2fM, window_size=%d, "
        "feature_dim=%d, train_samples=%d, validation_samples=%d",
        sum(parameter.numel() for parameter in parameters) / 1e6,
        window_size,
        MLP_REFERENCE_FEATURE_DIM,
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )

    csv_logger = utils.CSVLogger(experiment_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(experiment_dir)
    start_step = 0
    if checkpoint_path:
        logger.info("Loading checkpoint, checkpoint_path=%s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            "actor" not in state
            or "critic" not in state
            or state.get("architecture") != "mlp"
            or tuple(state.get("feature_names", ())) != MLP_REFERENCE_FEATURE_NAMES
            or tuple(state.get("scalar_names", ())) != MLP_REFERENCE_SCALAR_NAMES
        ):
            raise ValueError("checkpoint is not a shifted-window MLP checkpoint")
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))

    total_steps = int(cfg.train.steps_num)
    validation_interval = int(cfg.loop.validation_interval)
    validation_batches = int(cfg.val.validation_batches)
    last_losses: dict[str, float] = {}
    logger.info(
        "Starting shifted-window MLP training, global_step=%d, total_steps=%d",
        start_step,
        total_steps,
    )
    for step in tqdm(range(start_step, total_steps), desc=str(cfg.general.experiment_name)):
        prices, reference_prices, secs = prepare_price_batch(next(train_iterator), device)
        if bool(cfg.data.get("enforce_market_hours", True)):
            validate_market_hours(
                secs, context_ticks, rollout_size, str(cfg.data.anno)
            )
        rollout, last_losses = shifted_mlp_train_step(
            actor, critic, optimizer, prices, reference_prices, cfg
        )
        scheduler.step()
        current_step = step + 1
        if current_step % int(cfg.loop.log_interval) == 0:
            csv_logger.write(
                {
                    "step": current_step,
                    "stage": 0,
                    "timestamp": time.time(),
                    "lr": scheduler.get_last_lr()[0],
                    **rollout_metrics(rollout),
                    **last_losses,
                }
            )
        if current_step % validation_interval == 0:
            metrics = validation_metrics(
                actor, validation_iterator, validation_batches, cfg, device
            )
            csv_logger.write(
                {
                    "step": current_step,
                    "stage": 1,
                    "timestamp": time.time(),
                    "lr": scheduler.get_last_lr()[0],
                    **metrics,
                    **{key: float("nan") for key in last_losses},
                }
            )
            logger.info(
                "Validation, step=%d, return=%.6f, profit_factor=%.3f, max_drawdown=%.6f",
                current_step,
                metrics["return"],
                metrics["profit_factor"],
                metrics["max_drawdown"],
            )
        if current_step % int(cfg.loop.checkpoint_interval) == 0:
            utils.save_checkpoint(
                experiment_dir,
                current_step,
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "feature_names": MLP_REFERENCE_FEATURE_NAMES,
                    "scalar_names": MLP_REFERENCE_SCALAR_NAMES,
                    "reference_symbol": str(cfg.data.reference_symbol),
                    "architecture": "mlp",
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )
