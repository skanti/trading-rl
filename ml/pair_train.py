"""PPO training for the joint two-symbol relative-value policy.

Mirrors ``train.py`` but rolls out both legs under one joint action and logs
which trade shape the policy is holding, so a run that quietly degenerates into
a doubled-up directional bet is visible in the metrics rather than hidden
inside an aggregate return.
"""

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

from . import model, utils
from .dataset import OnlinePairToyProvider
from .pair import (
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    PAIR_SCALAR_DIM,
    PAIR_SCALAR_NAMES,
    pair_rollout_metrics,
    pair_train_step,
)
from .train import prepare_batch, regular_session_mask


logger = logging.getLogger("PAIR_RL")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
torch.set_float32_matmul_precision("high")


def build_models(cfg: DictConfig, device: torch.device) -> tuple[model.PairTradingActor, model.PairTradingCritic]:
    actor = model.PairTradingActor(**OmegaConf.to_container(cfg.model.actor, resolve=True)).to(device)
    critic = model.PairTradingCritic(**OmegaConf.to_container(cfg.model.critic, resolve=True)).to(device)
    return actor, critic


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    exp_dir = cfg.general.experiment_dir
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device(cfg.model.device)
    n, t = int(cfg.data.window_size), int(cfg.data.rollout_size)
    use_toy = bool(cfg.data.get("use_toy", False))

    toy_provider: OnlinePairToyProvider | None = None
    iterator = None
    if use_toy:
        toy_cfg = cfg.data.get("toy", {})
        toy_provider = OnlinePairToyProvider(
            window_size=n,
            rollout_size=t,
            return_noise_std=float(toy_cfg.get("return_noise_std", 3e-4)),
            flat_return_threshold=float(toy_cfg.get("flat_return_threshold", 2.5e-6)),
            spread_step_std=float(toy_cfg.get("spread_step_std", 2.2e-4)),
            half_life_min=float(toy_cfg.get("half_life_min", 20.0)),
            half_life_max=float(toy_cfg.get("half_life_max", 240.0)),
            broken_fraction=float(toy_cfg.get("broken_fraction", 0.35)),
            beta_min=float(toy_cfg.get("beta_min", 0.8)),
            beta_max=float(toy_cfg.get("beta_max", 1.25)),
        )
        logger.info(
            "Using online pair toy data, window_size=%d, rollout_size=%d, batch_size=%d",
            n,
            t,
            int(cfg.train.batch_size),
        )
    else:
        # Imported lazily so a toy run never needs the market pair index.
        from pair_dataset import make_pair_dataloader

        loader = make_pair_dataloader(cfg.data, cfg.train, int(cfg.train.get("seed", 0)))
        iterator = utils.cycle(loader)

    actor, critic = build_models(cfg, device)
    optimizer = optim.Adam(chain(actor.parameters(), critic.parameters()), lr=float(cfg.model.lr))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.train.steps_num), eta_min=float(cfg.model.get("eta_min", 1e-6))
    )
    logger.info(
        "Pair MLP summary, trainable_params_num=%.2fM",
        sum(p.numel() for p in chain(actor.parameters(), critic.parameters())) / 1e6,
    )

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
    logger.info("Starting pair training, global_step=%d, total_steps=%d", start_step, total_steps)
    for step in tqdm(range(start_step, total_steps), desc=cfg.general.experiment_name):
        if toy_provider is not None:
            batch = toy_provider.sample(int(cfg.train.batch_size), device)
            prices_a, prices_b = batch.prices_a, batch.prices_b
            volumes_a, volumes_b = batch.volumes_a, batch.volumes_b
            progress = batch.progress
        else:
            if iterator is None:
                raise RuntimeError("market pair iterator was not initialized")
            raw = next(iterator)
            prices_a, volumes_a, secs, progress = prepare_batch(
                {"prices": raw["prices_a"], "volumes": raw["volumes_a"], "secs": raw["secs"]},
                device,
                cfg.data.anno,
            )
            prices_b = raw["prices_b"].to(device=device, dtype=torch.float32, non_blocking=True)
            volumes_b = raw["volumes_b"].to(device=device, dtype=torch.float32, non_blocking=True)
            if cfg.data.get("enforce_market_hours", True):
                action_secs = secs[:, n - 1 : n - 1 + t]
                reward_secs = secs[:, n : n + t]
                if not regular_session_mask(action_secs, cfg.data.anno).all():
                    raise ValueError("action ticks must remain inside regular market hours")
                if not regular_session_mask(reward_secs, cfg.data.anno).all():
                    raise ValueError("reward/exit ticks must remain inside regular market hours")

        rollout, losses = pair_train_step(
            actor, critic, optimizer, prices_a, prices_b, volumes_a, volumes_b, progress, cfg
        )
        scheduler.step()
        if (step + 1) % int(cfg.loop.log_interval) == 0:
            csv_logger.write(
                {
                    "step": step + 1,
                    "stage": 0,
                    "timestamp": time.time(),
                    "lr": scheduler.get_last_lr()[0],
                    **pair_rollout_metrics(rollout),
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
                    "feature_names": PAIR_FEATURE_NAMES,
                    "scalar_names": PAIR_SCALAR_NAMES,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )
            logger.info("Saved checkpoint, step=%d", step + 1)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

if __name__ == "__main__":
    known, overrides = parser.parse_known_args()
    config = OmegaConf.load(known.config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    assert int(config.model.actor.feature_dim) == PAIR_FEATURE_DIM
    assert int(config.model.actor.scalar_dim) == PAIR_SCALAR_DIM
    main(config)
