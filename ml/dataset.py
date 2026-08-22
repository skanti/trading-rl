"""Market-day loader for autoregressive trading rollouts."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime

import fsspec
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("DATASET")


@dataclass(frozen=True)
class BezierToyBatch:
    prices: torch.Tensor
    volumes: torch.Tensor
    progress: torch.Tensor
    target_positions: torch.Tensor
    anchor_counts: torch.Tensor
    latent_returns: torch.Tensor


class OnlineBezierToyProvider:
    """Generate fresh Bezier trends with integrated return noise in memory.

    Noise is added to log returns and cumulatively integrated into prices. IID
    innovations are therefore unpredictable from the current observation and
    do not create the mechanical mean reversion caused by IID price-level noise.
    """

    def __init__(
        self,
        window_size: int,
        rollout_size: int,
        return_noise_std: float = 3e-4,
        flat_return_threshold: float = 2.5e-4,
    ):
        if window_size < 2 or rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")
        if return_noise_std < 0 or flat_return_threshold < 0:
            raise ValueError("return_noise_std and flat_return_threshold must be non-negative")
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.return_noise_std = float(return_noise_std)
        self.flat_return_threshold = float(flat_return_threshold)

    @property
    def ticks(self) -> int:
        return self.window_size + self.rollout_size

    @staticmethod
    def _bernstein_matrix(points: torch.Tensor, degree: int) -> torch.Tensor:
        columns = []
        for i in range(degree + 1):
            coefficient = float(math.comb(degree, i))
            columns.append(coefficient * points.pow(i) * (1.0 - points).pow(degree - i))
        return torch.stack(columns, dim=-1)

    def _sample_curve(
        self,
        anchor_count: int,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        degree = anchor_count - 1
        anchor_steps = torch.randn(
            batch_size, anchor_count - 1, device=device, generator=generator
        ) * 0.035
        anchors = torch.cat(
            (torch.zeros(batch_size, 1, device=device), anchor_steps.cumsum(dim=1)), dim=1
        )
        anchor_u = torch.linspace(0.0, 1.0, anchor_count, device=device)
        controls = torch.linalg.solve(self._bernstein_matrix(anchor_u, degree), anchors.T).T
        sample_u = torch.linspace(0.0, 1.0, self.ticks, device=device)
        return controls @ self._bernstein_matrix(sample_u, degree).T

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> BezierToyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)
        anchor_counts = torch.randint(3, 5, (batch_size,), device=device, generator=generator)
        latent_log_prices = torch.empty(batch_size, self.ticks, device=device)
        for anchor_count in (3, 4):
            selected = anchor_counts.eq(anchor_count)
            count = int(selected.sum().item())
            if count:
                latent_log_prices[selected] = self._sample_curve(anchor_count, count, device, generator)

        latent_returns = latent_log_prices[:, 1:] - latent_log_prices[:, :-1]
        return_innovations = torch.randn(
            batch_size, self.ticks - 1, device=device, generator=generator
        ) * self.return_noise_std
        observed_log_returns = latent_returns + return_innovations
        observed_log_prices = torch.cat(
            (
                latent_log_prices[:, :1],
                latent_log_prices[:, :1] + observed_log_returns.cumsum(dim=1),
            ),
            dim=1,
        )
        base_log_price = torch.empty(batch_size, 1, device=device).uniform_(
            np.log(8.0), np.log(800.0), generator=generator
        )
        prices = torch.exp(base_log_price + observed_log_prices)

        future_returns = latent_returns[
            :, self.window_size - 1 : self.window_size - 1 + self.rollout_size
        ]
        target_positions = torch.where(
            future_returns.abs() > self.flat_return_threshold,
            future_returns.sign(),
            torch.zeros_like(future_returns),
        ).to(torch.long)

        volume_noise = torch.randn(
            batch_size, self.ticks, device=device, generator=generator
        ) * 0.35
        padded_returns = torch.nn.functional.pad(observed_log_returns, (1, 0))
        volumes = torch.exp(6.0 + volume_noise + 10.0 * padded_returns.abs())
        progress = torch.linspace(-1.0, 1.0, self.ticks, device=device).expand(batch_size, -1)
        return BezierToyBatch(prices, volumes, progress, target_positions, anchor_counts, latent_returns)


class MarketDayDataset(Dataset):
    """Return an N-tick context followed by T tradable price intervals.

    Input ``.npy`` files contain at least ``[seconds, price_mills, volume]``.
    Symbol identity and absolute price are intentionally not returned to the
    policy, preventing the easiest forms of symbol-specific memorization.
    ``eod_idx`` is treated as an inclusive index, as in the original dataset.
    """

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        window_size: int,
        rollout_size: int,
        should_augment: bool = False,
        require_full_session: bool = False,
        limit: int | None = None,
    ):
        self.data_dir = data_dir
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.should_augment = bool(should_augment)
        self.require_full_session = bool(require_full_session)
        if self.window_size < 2 or self.rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")

        filtered = days.copy()
        context_floor = (
            filtered["ctx_idx"].fillna(0).astype(np.int64)
            if "ctx_idx" in filtered
            else pd.Series(0, index=filtered.index, dtype=np.int64)
        )
        has_context = filtered["sod_idx"].astype(np.int64) >= context_floor + self.window_size - 1
        session_intervals = filtered["eod_idx"].astype(np.int64) - filtered["sod_idx"].astype(np.int64)
        has_rollout = session_intervals >= self.rollout_size
        if self.require_full_session:
            has_rollout &= session_intervals == self.rollout_size
        filtered = filtered[has_context & has_rollout]
        if limit is not None:
            filtered = filtered.iloc[:limit]
        self.days = filtered.reset_index(drop=True)
        removed = len(days) - len(self.days)
        if removed:
            logger.info(
                "Filtered unusable market days, removed=%d, retained=%d, full_session=%s",
                removed,
                len(self.days),
                self.require_full_session,
            )
        if not len(self.days):
            raise ValueError("no market days satisfy the configured context and rollout lengths")

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, idx: int) -> dict:
        sample = self.days.iloc[idx]
        sample_id = str(sample.sample_id)
        sample_path = f"{self.data_dir}/{sample_id}.npy"
        with fsspec.open(sample_path, "rb") as f:
            data = np.load(f)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"{sample_path} must contain [seconds, price_mills, volume]")

        n, t = self.window_size, self.rollout_size
        sod_idx, eod_idx = int(sample.sod_idx), int(sample.eod_idx)
        context_floor = int(sample.ctx_idx) if "ctx_idx" in sample.index and not pd.isna(sample.ctx_idx) else 0

        # The action at last_context_idx receives the preceding N ticks. Every
        # selected position must have a subsequent price inside RTH.
        last_context_low = max(sod_idx, context_floor + n - 1)
        last_context_high = eod_idx - t
        if last_context_high < last_context_low:
            raise ValueError(
                f"sample {sample_id} is too short for window_size={n}, rollout_size={t}: "
                f"sod_idx={sod_idx}, eod_idx={eod_idx}, ctx_idx={context_floor}"
            )
        if self.require_full_session:
            if last_context_low != sod_idx or last_context_high != sod_idx:
                raise ValueError(f"sample {sample_id} does not describe one complete market session")
            last_context_idx = sod_idx
        elif self.should_augment:
            last_context_idx = int(np.random.randint(last_context_low, last_context_high + 1))
        else:
            last_context_idx = last_context_low

        start = last_context_idx - n + 1
        stop = last_context_idx + t + 1
        segment = data[start:stop]
        if segment.shape[0] != n + t:
            raise ValueError(f"sample {sample_id} ended unexpectedly at index {stop - 1}")

        prices = segment[:, 1].astype(np.float32) / 1000.0
        volumes = segment[:, 2].astype(np.float32)
        secs = segment[:, 0].astype(np.int64)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"sample {sample_id} contains non-positive or non-finite prices")
        if not np.isfinite(volumes).all() or (volumes < 0).any():
            raise ValueError(f"sample {sample_id} contains negative or non-finite volumes")

        return {"_id": sample_id, "prices": prices, "volumes": volumes, "secs": secs}


# Preserve the old public name while changing its output to raw market arrays.
TokLoader = MarketDayDataset


def make_dataloader(cfg_data: DictConfig, cfg_split: DictConfig, seed: int) -> DataLoader:
    del seed  # DataLoader sampling uses PyTorch's process seed.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    with fsspec.open(cfg_data.days_path, "r") as f:
        days = pd.read_csv(f)
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")

    date_val = datetime.strptime(cfg_data.date_val, "%Y-%m-%d")
    if cfg_split.split == "val":
        # Validation remains bounded without exposing symbol identity to the
        # model or depending on the obsolete tokenizer mapping file.
        shortlist = set(sorted(days.sample_id.unique())[:32])
        mask = (days.date >= date_val) & days.sample_id.isin(shortlist)
    else:
        mask = days.date < date_val
    days = days[mask].reset_index(drop=True)
    rank_indices = np.array_split(np.arange(len(days)), world_size)[rank]
    days = days.iloc[rank_indices].reset_index(drop=True)
    logger.info("Samples found, samples_num=%d, split=%s", len(days), cfg_split.split)

    dataset = MarketDayDataset(
        days=days,
        data_dir=cfg_data.data_dir,
        window_size=int(cfg_data.window_size),
        rollout_size=int(cfg_data.rollout_size),
        limit=cfg_split.get("samples_num", None),
        should_augment=cfg_split.get("should_augment", False),
        require_full_session=cfg_data.get("require_full_session", False),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg_split.batch_size),
        num_workers=int(cfg_split.workers_num),
        shuffle=bool(cfg_split.get("should_augment", False)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(cfg_split.workers_num) > 0,
    )
