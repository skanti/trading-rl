"""Supervised trainer for binary price-direction classifiers."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

import utils
from classify import (
    CLASSIFY_SCALAR_DIM,
    CLASSIFY_SCALAR_NAMES,
    RelativeDirectionClassifier,
    binary_metrics,
    build_classify_features,
    build_classify_scalars,
    classify_feature_names,
    classify_labels,
)
from classify_dataset import make_classify_dataloader


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("CLASSIFY")
torch.set_float32_matmul_precision("high")
MIN_PARAMETERS = 1_000_000
MAX_PARAMETERS = 2_000_000


def build_classifier(cfg: DictConfig, device: torch.device) -> RelativeDirectionClassifier:
    config = OmegaConf.to_container(cfg.model.mlp, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("model.mlp must be a mapping")
    feature_names = classify_feature_names(str(cfg.data.get("feature_mode", "relative")))
    if int(config.get("feature_dim", -1)) != len(feature_names):
        raise ValueError(f"model.mlp.feature_dim must be {len(feature_names)}")
    if int(config.get("scalar_dim", -1)) != CLASSIFY_SCALAR_DIM:
        raise ValueError(f"model.mlp.scalar_dim must be {CLASSIFY_SCALAR_DIM}")
    classifier = RelativeDirectionClassifier(**config).to(device)
    parameters = sum(parameter.numel() for parameter in classifier.parameters())
    if not MIN_PARAMETERS <= parameters <= MAX_PARAMETERS:
        raise ValueError(
            f"classifier has {parameters:,} parameters; expected "
            f"{MIN_PARAMETERS:,}..{MAX_PARAMETERS:,}"
        )
    return classifier


def prepare_batch(batch: dict, device: torch.device, cfg: DictConfig):
    prices = batch["prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    reference = batch["reference_prices"].to(device=device, dtype=torch.float32, non_blocking=True)
    anchor_progress = batch["anchor_progress"].to(device=device, dtype=torch.float32)
    weekday = batch["weekday"].to(device=device, dtype=torch.long)
    scalars = build_classify_scalars(anchor_progress, weekday)
    features = build_classify_features(
        prices,
        reference,
        str(cfg.data.get("feature_mode", "relative")),
        float(cfg.data.get("price_feature_scale", 100.0)),
    )
    labels = classify_labels(
        prices, reference, str(cfg.data.get("target_mode", "relative_direction"))
    )
    return features, scalars, labels


@torch.no_grad()
def validation_metrics(
    classifier: RelativeDirectionClassifier,
    loader,
    batches: int | None,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """One deterministic pass over the pinned validation universe.

    Logits are pooled before scoring: win rate is a mean and would survive
    per-batch averaging, but AUC and the confident-fifth accuracy would not.
    """
    classifier.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for index, batch in enumerate(loader):
        if batches is not None and index >= int(batches):
            break
        features, scalars, target = prepare_batch(batch, device, cfg)
        output = classifier(features, scalars)
        losses.append(F.binary_cross_entropy_with_logits(output, target, reduction="sum").cpu())
        logits.append(output.cpu())
        labels.append(target.cpu())
    classifier.train()
    if not logits:
        raise ValueError("validation loader produced no batches")
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    return {"loss": float(torch.stack(losses).sum() / labels.numel()), **binary_metrics(logits, labels)}


def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.train.get("seed", 0)))
    np.random.seed(int(cfg.train.get("seed", 0)))
    device = torch.device(str(cfg.model.device))
    experiment_dir = str(cfg.general.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    train_loader = make_classify_dataloader(cfg.data, cfg.train, int(cfg.train.seed))
    validation_loader = make_classify_dataloader(cfg.data, cfg.val, int(cfg.train.seed))
    train_iterator = utils.cycle(train_loader)
    classifier = build_classifier(cfg, device)
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.get("weight_decay", 0.01)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.steps_num),
        eta_min=float(cfg.model.get("eta_min", 1e-6)),
    )
    logger.info(
        "Direction classifier, params=%.2fM, window_size=%d, horizon_days=%d, "
        "anchor_minute=%s, target_minute=%s, train_samples=%d, validation_samples=%d",
        sum(parameter.numel() for parameter in classifier.parameters()) / 1e6,
        int(cfg.model.mlp.window_size),
        int(cfg.data.horizon_days),
        str(cfg.data.get("anchor_minute", "random")),
        str(cfg.data.get("target_minute", "same as anchor")),
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )

    csv_logger = utils.CSVLogger(experiment_dir)
    checkpoint_path = utils.load_most_recent_checkpoint(experiment_dir)
    start_step = 0
    if checkpoint_path:
        logger.info("Loading checkpoint, checkpoint_path=%s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        feature_names = classify_feature_names(str(cfg.data.get("feature_mode", "relative")))
        if tuple(state.get("feature_names", ())) != feature_names:
            raise ValueError("checkpoint feature layout does not match the experiment")
        if str(state.get("target_mode", "relative_direction")) != str(
            cfg.data.get("target_mode", "relative_direction")
        ):
            raise ValueError("checkpoint target mode does not match the experiment")
        classifier.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("global_step", 0))

    configured_batches = cfg.val.get("validation_batches", None)
    validation_batches = None if configured_batches is None else int(configured_batches)
    total_steps = int(cfg.train.steps_num)
    max_grad_norm = float(cfg.model.get("max_grad_norm", 1.0))
    logger.info("Starting classification, global_step=%d, total_steps=%d", start_step, total_steps)
    for step in tqdm(range(start_step, total_steps), desc=str(cfg.general.experiment_name)):
        features, scalars, labels = prepare_batch(next(train_iterator), device, cfg)
        logits = classifier(features, scalars)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        current_step = step + 1
        if current_step % int(cfg.loop.log_interval) == 0:
            csv_logger.write(
                {
                    "step": current_step,
                    "stage": 0,
                    "timestamp": time.time(),
                    "lr": scheduler.get_last_lr()[0],
                    "loss": float(loss.item()),
                    **binary_metrics(logits, labels),
                }
            )
        if current_step % int(cfg.loop.validation_interval) == 0:
            metrics = validation_metrics(
                classifier, validation_loader, validation_batches, cfg, device
            )
            csv_logger.write(
                {"step": current_step, "stage": 1, "timestamp": time.time(),
                 "lr": scheduler.get_last_lr()[0], **metrics}
            )
            logger.info(
                "Validation, step=%d, win_rate=%.4f (majority %.4f), confident=%.4f, "
                "auc=%.4f, loss=%.4f",
                current_step,
                metrics["win_rate"],
                metrics["majority_rate"],
                metrics["win_rate_confident"],
                metrics["auc"],
                metrics["loss"],
            )
        if current_step % int(cfg.loop.checkpoint_interval) == 0:
            utils.save_checkpoint(
                experiment_dir,
                current_step,
                {
                    "model": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "feature_names": classify_feature_names(
                        str(cfg.data.get("feature_mode", "relative"))
                    ),
                    "scalar_names": CLASSIFY_SCALAR_NAMES,
                    "target_mode": str(
                        cfg.data.get("target_mode", "relative_direction")
                    ),
                    "reference_symbol": str(cfg.data.reference_symbol),
                    "horizon_days": int(cfg.data.horizon_days),
                    "anchor_minute": cfg.data.get("anchor_minute", None),
                    "target_minute": cfg.data.get("target_minute", None),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
            )
