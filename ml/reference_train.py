"""PPO training for a tokenized causal GPT using asset and SPY prices."""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

import utils
from gpt import CausalTradingTransformer, PairPriceTokenizer
from reference import TOKEN_FEATURE_NAMES, collect_token_rollout, token_train_step
from reference_dataset import make_reference_dataloader
from train import regular_session_mask, rollout_metrics


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("REFERENCE_GPT")
torch.set_float32_matmul_precision("high")
MAX_MODEL_PARAMETERS = 20_000_000


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
    if not regular_session_mask(
        secs[:, context_ticks : context_ticks + steps], anno
    ).all():
        raise ValueError("action ticks must remain inside regular market hours")
    if not regular_session_mask(
        secs[:, context_ticks + 1 : context_ticks + steps + 1], anno
    ).all():
        raise ValueError("reward/exit ticks must remain inside regular market hours")


@torch.no_grad()
def validation_metrics(
    transformer: CausalTradingTransformer,
    tokenizer: PairPriceTokenizer,
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
        rollout = collect_token_rollout(
            transformer,
            tokenizer,
            prices,
            reference_prices,
            context_ticks=int(cfg.data.context_ticks),
            rollout_size=int(cfg.data.rollout_size),
            transaction_cost=float(cfg.model.get("transaction_cost", 0.0)),
            risk_penalty=float(cfg.model.get("risk_penalty", 0.0)),
            sampling="greedy",
        )
        for key, value in rollout_metrics(rollout).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / batches for key, value in totals.items()}


def build_transformer(cfg: DictConfig, device: torch.device) -> CausalTradingTransformer:
    transformer = CausalTradingTransformer(
        **OmegaConf.to_container(cfg.model.transformer, resolve=True)
    ).to(device)
    parameters = transformer.parameter_count()
    if parameters >= MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"transformer has {parameters:,} parameters; it must remain below "
            f"{MAX_MODEL_PARAMETERS:,}"
        )
    return transformer


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    device = torch.device(str(cfg.model.device))
    experiment_dir = str(cfg.general.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    context_ticks = int(cfg.data.context_ticks)
    rollout_size = int(cfg.data.rollout_size)
    expected_context = int(cfg.data.context_days) * 16 * 60
    if context_ticks != expected_context:
        raise ValueError(
            f"context_ticks must be {expected_context} for "
            f"context_days={int(cfg.data.context_days)}"
        )
    if int(cfg.data.loader_window_size) != context_ticks + 1:
        raise ValueError("loader_window_size must retain the context plus the current open")
    required_prices = context_ticks + rollout_size + 1
    if int(cfg.model.transformer.max_seq_len) < required_prices - 1:
        raise ValueError("transformer max_seq_len is too short for the growing rollout")

    train_loader = make_reference_dataloader(cfg.data, cfg.train, int(cfg.train.seed))
    validation_loader = make_reference_dataloader(cfg.data, cfg.val, int(cfg.train.seed))
    train_iterator = utils.cycle(train_loader)
    validation_iterator = utils.cycle(validation_loader)
    tokenizer = PairPriceTokenizer(
        **OmegaConf.to_container(cfg.data.tokenizer, resolve=True)
    )
    transformer = build_transformer(cfg, device)
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.get("weight_decay", 0.01)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.steps_num),
        eta_min=float(cfg.model.get("eta_min", 1e-6)),
    )
    logger.info(
        "Causal GPT summary, trainable_params_num=%.2fM, vocab_size=%d, "
        "context_ticks=%d, model_tokens=%d, price_observations=%d, "
        "train_samples=%d, validation_samples=%d",
        transformer.parameter_count() / 1e6,
        tokenizer.vocab_size,
        context_ticks,
        required_prices - 1,
        required_prices,
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )

    csv_logger = utils.CSVLogger(experiment_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(experiment_dir)
    start_step = 0
    if checkpoint_path:
        logger.info("Loading checkpoint, checkpoint_path=%s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" not in state:
            raise ValueError("checkpoint predates the GPT architecture; use a new experiment name")
        transformer.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))

    last_losses: dict[str, float] = {}
    total_steps = int(cfg.train.steps_num)
    validation_interval = int(cfg.loop.validation_interval)
    validation_batches = int(cfg.val.validation_batches)
    logger.info("Starting causal-GPT training, global_step=%d, total_steps=%d", start_step, total_steps)
    for step in tqdm(range(start_step, total_steps), desc=str(cfg.general.experiment_name)):
        prices, reference_prices, secs = prepare_price_batch(next(train_iterator), device)
        if bool(cfg.data.get("enforce_market_hours", True)):
            validate_market_hours(
                secs, context_ticks, rollout_size, str(cfg.data.anno)
            )
        rollout, last_losses = token_train_step(
            transformer,
            tokenizer,
            optimizer,
            prices,
            reference_prices,
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
                transformer,
                tokenizer,
                validation_iterator,
                validation_batches,
                cfg,
                device,
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
            transformer.clear_cache()
            utils.save_checkpoint(
                experiment_dir,
                current_step,
                {
                    "model": transformer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "feature_names": TOKEN_FEATURE_NAMES,
                    "tokenizer": OmegaConf.to_container(cfg.data.tokenizer, resolve=True),
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
    main(config)
