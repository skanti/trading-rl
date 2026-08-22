"""Market loader for time-matched two-symbol rollouts.

The two legs are joined on their timestamps at load time rather than on row
indices, because coverage of extended hours differs between symbols and a
shared row offset does not exist. Only ticks present in both legs survive the
join, so every observation the policy sees is genuinely simultaneous.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import fsspec
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("PAIR_DATASET")

# How much raw history to scan per leg, as a multiple of the ticks needed. The
# join discards ticks one leg is missing, so the scan has to start wider.
DEFAULT_SEARCH_FACTOR = 4


class MarketPairDataset(Dataset):
    """Return an N-tick joined context followed by T tradable intervals.

    Neither symbol identity nor absolute price leaves this class, matching the
    single-symbol loader. ``eod_sec`` is the inclusive last tick of the regular
    session, so a complete session spans ``rollout_size + 1`` ticks.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        data_dir: str,
        window_size: int,
        rollout_size: int,
        anno: str = "2010-01-01",
        search_factor: int = DEFAULT_SEARCH_FACTOR,
        limit: int | None = None,
    ):
        self.data_dir = data_dir
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.anno = str(anno)
        self.search_factor = int(search_factor)
        if self.window_size < 2 or self.rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")
        if self.search_factor < 1:
            raise ValueError("search_factor must be at least one")
        self.pairs = (pairs if limit is None else pairs.iloc[:limit]).reset_index(drop=True)
        if not len(self.pairs):
            raise ValueError("no pair-days were supplied")

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def ticks(self) -> int:
        return self.window_size + self.rollout_size

    def _load(self, sample_id: str) -> np.ndarray:
        with fsspec.open(f"{self.data_dir}/{sample_id}.npy", "rb") as f:
            data = np.load(f)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"{sample_id} must contain [seconds, price_mills, volume]")
        return data

    def _tail(self, data: np.ndarray, eod_sec: int, sample_id: str) -> np.ndarray:
        """Rows up to and including ``eod_sec``, limited to a search span."""
        secs = data[:, 0].astype(np.int64)
        stop = int(np.searchsorted(secs, eod_sec, side="right"))
        if stop == 0 or secs[stop - 1] != eod_sec:
            raise ValueError(f"sample {sample_id} has no tick at eod_sec={eod_sec}")
        start = max(0, stop - self.ticks * self.search_factor)
        return data[start:stop]

    def __getitem__(self, idx: int) -> dict:
        row = self.pairs.iloc[idx]
        sample_a, sample_b = str(row.sample_id_a), str(row.sample_id_b)
        sod_sec, eod_sec = int(row.sod_sec), int(row.eod_sec)

        tail_a = self._tail(self._load(sample_a), eod_sec, sample_a)
        tail_b = self._tail(self._load(sample_b), eod_sec, sample_b)
        shared, index_a, index_b = np.intersect1d(
            tail_a[:, 0].astype(np.int64), tail_b[:, 0].astype(np.int64), return_indices=True
        )
        if shared.size < self.ticks:
            raise ValueError(
                f"pair {sample_a}/{sample_b} on {row.date} shares only {shared.size} ticks, "
                f"needs {self.ticks}"
            )
        take = slice(shared.size - self.ticks, shared.size)
        segment_a = tail_a[index_a[take]]
        segment_b = tail_b[index_b[take]]
        secs = shared[take]

        # The last context tick must be the open, so the whole session is traded.
        session_secs = secs[self.window_size - 1 :]
        if session_secs[0] != sod_sec or session_secs[-1] != eod_sec:
            raise ValueError(
                f"pair {sample_a}/{sample_b} on {row.date} does not start the rollout at the open"
            )
        if not np.all(np.diff(session_secs) == 60):
            raise ValueError(
                f"pair {sample_a}/{sample_b} on {row.date} has gaps inside the regular session"
            )

        prices_a = segment_a[:, 1].astype(np.float32) / 1000.0
        prices_b = segment_b[:, 1].astype(np.float32) / 1000.0
        volumes_a = segment_a[:, 2].astype(np.float32)
        volumes_b = segment_b[:, 2].astype(np.float32)
        for name, prices in (("a", prices_a), ("b", prices_b)):
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError(f"pair {sample_a}/{sample_b} leg {name} has non-positive prices")
        for name, volumes in (("a", volumes_a), ("b", volumes_b)):
            if not np.isfinite(volumes).all() or (volumes < 0).any():
                raise ValueError(f"pair {sample_a}/{sample_b} leg {name} has invalid volumes")

        return {
            "_id": f"{sample_a}|{sample_b}|{row.date}",
            "prices_a": prices_a,
            "prices_b": prices_b,
            "volumes_a": volumes_a,
            "volumes_b": volumes_b,
            "secs": secs.astype(np.int64),
        }


def make_pair_dataloader(cfg_data: DictConfig, cfg_split: DictConfig, seed: int) -> DataLoader:
    """Split pair-days by date and by symbol universe.

    :mod:`prepare_pairs` already partitioned the symbols and drew every pair
    inside one universe, so the split here is a filter on that column plus the
    date cut. Splitting on date alone would not be enough: a pair shares each of
    its legs with many other pairs, so a symbol held out by date still reaches
    training inside a different pairing.
    """
    del seed  # DataLoader sampling uses PyTorch's process seed.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    with fsspec.open(cfg_data.pairs_path, "r") as f:
        pairs = pd.read_csv(f)
    pairs["date"] = pd.to_datetime(pairs["date"], format="%Y-%m-%d")
    if "universe" not in pairs:
        raise ValueError(
            f"{cfg_data.pairs_path} has no 'universe' column; rebuild it with prepare_pairs.py"
        )

    date_val = datetime.strptime(cfg_data.date_val, "%Y-%m-%d")
    wanted = "holdout" if cfg_split.split == "val" else "train"
    in_period = pairs.date >= date_val if cfg_split.split == "val" else pairs.date < date_val
    pairs = pairs[in_period & pairs.universe.eq(wanted)].reset_index(drop=True)
    rank_indices = np.array_split(np.arange(len(pairs)), world_size)[rank]
    pairs = pairs.iloc[rank_indices].reset_index(drop=True)
    logger.info(
        "Pair-days found, pairs_num=%d, split=%s, universe=%s, symbols=%d",
        len(pairs),
        cfg_split.split,
        wanted,
        len(set(pairs.sample_id_a) | set(pairs.sample_id_b)) if len(pairs) else 0,
    )

    dataset = MarketPairDataset(
        pairs=pairs,
        data_dir=cfg_data.data_dir,
        window_size=int(cfg_data.window_size),
        rollout_size=int(cfg_data.rollout_size),
        anno=str(cfg_data.get("anno", "2010-01-01")),
        limit=cfg_split.get("samples_num", None),
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
