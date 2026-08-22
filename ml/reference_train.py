"""PPO training for one asset with SPY as an observation-only reference."""

from __future__ import annotations

import argparse
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

import model
import utils
from reference import (
    REFERENCE_FEATURE_DIM,
    REFERENCE_FEATURE_NAMES,
    collect_reference_rollout,
    reference_train_step,
)
from reference_dataset import make_reference_dataloader
from train import regular_session_mask, rollout_metrics, session_progress


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("REFERENCE_RL")
torch.set_float32_matmul_precision("high")


def prepare_reference_batch(
    batch: dict, device: torch.device, anno: str
) -> tuple[torch.Tensor, ...]:
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    volumes = batch["volumes"].to(device=device, dtype=torch.float32, non_blocking=True)
    reference_prices = batch["reference_prices"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    reference_volumes = batch["reference_volumes"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    secs = batch["secs"].to(device=device, dtype=torch.long, non_blocking=True)
    return prices, reference_prices, volumes, reference_volumes, secs, session_progress(secs, anno)


def validate_market_hours(secs: torch.Tensor, window: int, steps: int, anno: str) -> None:
    if not regular_session_mask(secs[:, window - 1 : window - 1 + steps], anno).all():
        raise ValueError("action ticks must remain inside regular market hours")
    if not regular_session_mask(secs[:, window : window + steps], anno).all():
        raise ValueError("reward/exit ticks must remain inside regular market hours")


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
        prepared = prepare_reference_batch(next(iterator), device, str(cfg.data.anno))
        prices, reference_prices, volumes, reference_volumes, secs, progress = prepared
        validate_market_hours(
            secs, int(cfg.data.window_size), int(cfg.data.rollout_size), str(cfg.data.anno)
        )
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
            sampling="greedy",
            price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
        )
        for key, value in rollout_metrics(rollout).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / batches for key, value in totals.items()}


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    device = torch.device(str(cfg.model.device))
    experiment_dir = str(cfg.general.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    window, steps = int(cfg.data.window_size), int(cfg.data.rollout_size)

    train_loader = make_reference_dataloader(cfg.data, cfg.train, int(cfg.train.seed))
    validation_loader = make_reference_dataloader(cfg.data, cfg.val, int(cfg.train.seed))
    train_iterator = utils.cycle(train_loader)
    validation_iterator = utils.cycle(validation_loader)

    actor = model.TradingActor(
        **OmegaConf.to_container(cfg.model.actor, resolve=True)
    ).to(device)
    critic = model.TradingCritic(
        **OmegaConf.to_container(cfg.model.critic, resolve=True)
    ).to(device)
    parameters = list(chain(actor.parameters(), critic.parameters()))
    optimizer = optim.Adam(parameters, lr=float(cfg.model.lr))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.steps_num),
        eta_min=float(cfg.model.get("eta_min", 1e-6)),
    )
    logger.info(
        "SPY-reference MLP summary, trainable_params_num=%.2fM, train_samples=%d, "
        "validation_samples=%d",
        sum(parameter.numel() for parameter in parameters) / 1e6,
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )

    csv_logger = utils.CSVLogger(experiment_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(experiment_dir)
    start_step = 0
    if checkpoint_path:
        logger.info("Loading checkpoint, checkpoint_path=%s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))

    last_losses: dict[str, float] = {}
    total_steps = int(cfg.train.steps_num)
    validation_interval = int(cfg.loop.validation_interval)
    validation_batches = int(cfg.val.validation_batches)
    logger.info("Starting SPY-reference training, global_step=%d, total_steps=%d", start_step, total_steps)
    for step in tqdm(range(start_step, total_steps), desc=str(cfg.general.experiment_name)):
        prepared = prepare_reference_batch(next(train_iterator), device, str(cfg.data.anno))
        prices, reference_prices, volumes, reference_volumes, secs, progress = prepared
        if bool(cfg.data.get("enforce_market_hours", True)):
            validate_market_hours(secs, window, steps, str(cfg.data.anno))
        rollout, last_losses = reference_train_step(
            actor,
            critic,
            optimizer,
            prices,
            reference_prices,
            volumes,
            reference_volumes,
            progress,
            cfg,
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
                    "feature_names": REFERENCE_FEATURE_NAMES,
                    "reference_symbol": str(cfg.data.reference_symbol),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)

if __name__ == "__main__":
    known, overrides = parser.parse_known_args()
    config = OmegaConf.load(known.config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    if int(config.model.actor.feature_dim) != REFERENCE_FEATURE_DIM:
        raise ValueError(f"reference actor requires feature_dim={REFERENCE_FEATURE_DIM}")
    main(config)
