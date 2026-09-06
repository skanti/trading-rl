"""PPO trainer for the intraweek, 10-minute asset/SPY MLP."""

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
from .train import rollout_metrics
from .week import (
    WEEK_METRIC_FIELDS,
    WEEK_FEATURE_DIM,
    WEEK_FEATURE_NAMES,
    WEEK_SCALAR_DIM,
    WEEK_SCALAR_NAMES,
    collect_week_rollout,
    overnight_metrics,
    validate_week_hours,
    week_metrics,
    week_train_step,
)
from .week_dataset import (
    make_week_dataloader,
    ticks_per_session,
    week_context_ticks,
    week_rollout_size,
)


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("WEEK_MLP")
torch.set_float32_matmul_precision("high")
MAX_MLP_PARAMETERS = 20_000_000


def build_week_models(
    cfg: DictConfig, device: torch.device
) -> tuple[model.TradingActor, model.TradingCritic]:
    config = OmegaConf.to_container(cfg.model.mlp, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("model.mlp must be a mapping")
    if int(config.get("feature_dim", -1)) != WEEK_FEATURE_DIM:
        raise ValueError(f"model.mlp.feature_dim must be {WEEK_FEATURE_DIM}")
    if int(config.get("scalar_dim", -1)) != WEEK_SCALAR_DIM:
        raise ValueError(f"model.mlp.scalar_dim must be {WEEK_SCALAR_DIM}")
    actor = model.TradingActor(**config).to(device)
    critic_config = dict(config)
    critic_config.pop("action_dim", None)
    critic = model.TradingCritic(**critic_config).to(device)
    parameters = sum(
        parameter.numel() for parameter in chain(actor.parameters(), critic.parameters())
    )
    if parameters >= MAX_MLP_PARAMETERS:
        raise ValueError(
            f"MLP actor+critic have {parameters:,} parameters; they must remain below "
            f"{MAX_MLP_PARAMETERS:,}"
        )
    return actor, critic


def prepare_week_batch(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    reference_prices = batch["reference_prices"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    secs = batch["secs"].to(device=device, dtype=torch.long, non_blocking=True)
    return prices, reference_prices, secs


def week_rollout_report(rollout, session_ticks: int) -> dict[str, float]:
    return {**rollout_metrics(rollout), **overnight_metrics(rollout.positions, session_ticks)}


@torch.no_grad()
def validation_metrics(
    actor: model.TradingActor,
    loader,
    batches: int | None,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Greedy metrics over the pinned validation universe.

    One full pass by default, so the number covers every held-out symbol-week
    and does not move with sampling. ``batches`` caps the pass if the universe
    is ever made large enough for that to matter.
    """
    session_ticks = ticks_per_session(int(cfg.data.tick_minutes))
    pooled: dict[str, list[torch.Tensor]] = {name: [] for name in WEEK_METRIC_FIELDS}
    for index, batch in enumerate(loader):
        if batches is not None and index >= int(batches):
            break
        prices, reference_prices, secs = prepare_week_batch(batch, device)
        validate_week_hours(
            secs,
            int(cfg.data.context_ticks),
            int(cfg.data.rollout_size),
            int(cfg.data.tick_minutes),
            session_ticks,
            str(cfg.data.anno),
        )
        rollout = collect_week_rollout(
            actor,
            prices,
            reference_prices,
            context_ticks=int(cfg.data.context_ticks),
            rollout_size=int(cfg.data.rollout_size),
            session_ticks=session_ticks,
            transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
            risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
            sampling="greedy",
            price_feature_scale=float(cfg.data.get("price_feature_scale", 100.0)),
        )
        # Collect the outcome tensors rather than per-batch summaries: profit
        # factor and drawdown are not means, so they must be computed once over
        # the pooled universe. Observation tensors are deliberately not kept.
        for name in WEEK_METRIC_FIELDS:
            pooled[name].append(getattr(rollout, name).cpu())
    if not pooled["rewards"]:
        raise ValueError("validation loader produced no batches")
    joined = {name: torch.cat(values, dim=0) for name, values in pooled.items()}
    return week_metrics(session_ticks=session_ticks, **joined)


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    device = torch.device(str(cfg.model.device))
    experiment_dir = str(cfg.general.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    tick_minutes = int(cfg.data.tick_minutes)
    session_ticks = ticks_per_session(tick_minutes)
    context_ticks = int(cfg.data.context_ticks)
    rollout_size = int(cfg.data.rollout_size)
    if context_ticks != week_context_ticks(int(cfg.data.context_days), tick_minutes):
        raise ValueError("data.context_ticks does not match context_days at this resolution")
    if rollout_size != week_rollout_size(tick_minutes):
        raise ValueError("data.rollout_size does not match a five-session week")
    window_size = int(cfg.model.mlp.window_size)
    if window_size > context_ticks + 1:
        raise ValueError("model.mlp.window_size exceeds available context plus Monday 09:30")

    train_loader = make_week_dataloader(cfg.data, cfg.train, int(cfg.train.seed))
    validation_loader = make_week_dataloader(cfg.data, cfg.val, int(cfg.train.seed))
    train_iterator = utils.cycle(train_loader)
    actor, critic = build_week_models(cfg, device)
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
        "Intraweek MLP summary, trainable_params_num=%.2fM, tick_minutes=%d, "
        "window_size=%d, context_ticks=%d, rollout_size=%d, train_weeks=%d, "
        "validation_symbol_weeks=%d",
        sum(parameter.numel() for parameter in parameters) / 1e6,
        tick_minutes,
        window_size,
        context_ticks,
        rollout_size,
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
            or tuple(state.get("feature_names", ())) != WEEK_FEATURE_NAMES
            or tuple(state.get("scalar_names", ())) != WEEK_SCALAR_NAMES
        ):
            raise ValueError("checkpoint is not an intraweek shifted-window MLP checkpoint")
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))

    total_steps = int(cfg.train.steps_num)
    validation_interval = int(cfg.loop.validation_interval)
    configured_batches = cfg.val.get("validation_batches", None)
    validation_batches = None if configured_batches is None else int(configured_batches)
    enforce_hours = bool(cfg.data.get("enforce_market_hours", True))
    last_losses: dict[str, float] = {}
    logger.info(
        "Starting intraweek MLP training, global_step=%d, total_steps=%d",
        start_step,
        total_steps,
    )
    for step in tqdm(range(start_step, total_steps), desc=str(cfg.general.experiment_name)):
        prices, reference_prices, secs = prepare_week_batch(next(train_iterator), device)
        if enforce_hours:
            validate_week_hours(
                secs,
                context_ticks,
                rollout_size,
                tick_minutes,
                session_ticks,
                str(cfg.data.anno),
            )
        rollout, last_losses = week_train_step(
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
                    **week_rollout_report(rollout, session_ticks),
                    **last_losses,
                }
            )
        if current_step % validation_interval == 0:
            metrics = validation_metrics(
                actor, validation_loader, validation_batches, cfg, device
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
                "Validation, step=%d, return=%.6f, profit_factor=%.3f, "
                "max_drawdown=%.6f, overnight_fraction=%.3f",
                current_step,
                metrics["return"],
                metrics["profit_factor"],
                metrics["max_drawdown"],
                metrics["overnight_fraction"],
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
                    "feature_names": WEEK_FEATURE_NAMES,
                    "scalar_names": WEEK_SCALAR_NAMES,
                    "reference_symbol": str(cfg.data.reference_symbol),
                    "tick_minutes": tick_minutes,
                    "architecture": "mlp",
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )
