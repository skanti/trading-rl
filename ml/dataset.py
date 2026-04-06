import logging
import os
from datetime import datetime
import time
from functools import partial

import einops
import fsspec
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader
from rich.logging import RichHandler

try:
    from . import tokenizer
except ImportError:
    import tokenizer

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("DATASET")


class TokLoader(Dataset):
    def __init__(
        self,
        days: dict,
        mapping: dict,
        data_dir: str,
        seq_size: int,
        ctx_size: int,
        channels: int,
        vocab_size: int,
        anno: str,
        rollout_size: int,
        should_augment: bool = False,
        limit: int | None = None,
    ):
        # data
        self.days = days
        if limit is not None:
            self.days = days[:limit]
        self.data_dir = data_dir
        self.seq_size = seq_size
        self.channels = channels
        self.rollout_size = rollout_size
        self.should_augment = should_augment

        # using mapping as weights
        self.weights = {k: 1.0 - i / len(mapping) for k, i in mapping.items()}

        # tokenizer
        self.tokenizer = tokenizer.Tokenizer(
            anno=anno,
            mapping=mapping,
            seq_size=seq_size,
            ctx_size=ctx_size,
            channels=channels,
            vocab_size=vocab_size,
        )

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, idx: int) -> dict:

        # get sample
        sample = self.days.iloc[idx]
        sample_id = sample.sample_id
        sample_path = f"{self.data_dir}/{sample_id}.npy"

        # get pos
        pos = self.weights[sample_id]
        days = (sample.date - self.tokenizer.anno).days

        # get size
        N = self.seq_size
        T = self.rollout_size
        sod_idx, eod_idx = int(sample.sod_idx), int(sample.eod_idx)
        ctx_idx = int(sample.ctx_idx) if "ctx_idx" in sample.index else 0

        # The first action sees the previous N ticks, and every action needs a
        # following price for reward. Keep the entire sampled path inside RTH.
        last_context_idx_low = max(sod_idx, ctx_idx + N - 1)
        last_context_idx_high = eod_idx - T
        assert last_context_idx_high >= last_context_idx_low, (
            f"Sample too short for seq_size={N}, rollout_size={T}, "
            f"sample_id={sample_id}, sod_idx={sod_idx}, eod_idx={eod_idx}, ctx_idx={ctx_idx}"
        )

        if self.should_augment:
            last_context_idx = np.random.randint(last_context_idx_low, last_context_idx_high + 1)
        else:
            last_context_idx = last_context_idx_low

        ctx_idx = last_context_idx - N + 1
        eod_idx = ctx_idx + N + T

        # make segment
        segment = np.arange(ctx_idx, eod_idx)
        assert segment.shape[0] == N + T

        # parse
        self.tokenizer.parse(sample_path=sample_path, segment=segment)

        # tokenize
        ctx, seq, ts = self.tokenizer.tokenize()
        assert seq.shape[0] == ((self.seq_size + self.rollout_size) * self.channels)

        # check date
        assert (
            self.tokenizer.dt_last.date() == sample.date.date()
        ), f"Date mismatch, sample_id={sample_id}, idx={idx}"

        out = {
            "_id": sample_id,
            "ts": ts,
            "seq": seq,
            "ctx": ctx,
            "prices": self.tokenizer.temp.astype(np.float32),
            "secs": self.tokenizer.secs.astype(np.int64),
            "pos": pos,
            "days": days,
        }
        return out


def make_dataloader(cfg_data: DictConfig, cfg_split: DictConfig, seed: int):
    # get world info
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # add barrier here
    if world_size > 1:
        torch.distributed.barrier()

    # load mapping
    with fsspec.open(cfg_data.mapping_path, "r") as f:
        mapping = f.read().splitlines()
        mapping = {x: i for i, x in enumerate(mapping)}

    # load days
    with fsspec.open(cfg_data.days_path, "r") as f:
        days = pd.read_csv(f)
        days.date = pd.to_datetime(days.date, format="%Y-%m-%d")

    # split val
    date_val = datetime.strptime(cfg_data.date_val, "%Y-%m-%d")
    if cfg_split.split == "val":
        shortlist = set(list(mapping.keys())[:32])
        m = (days.date >= date_val) & days.sample_id.isin(shortlist)
    else:
        m = days.date < date_val
    days = days[m].reset_index()

    # ### OVERFIT ###
    # m = (days.date == date_val) & (days.sample_id == "HG-MEWZ") # overfit
    # days = days[m].reset_index()
    # days = pd.concat(1000*[days])
    # ### OVERFIT ###

    # split
    days = np.array_split(days, world_size)[rank]
    samples_num = len(days)
    logger.info(f"Samples found, samples_num={samples_num}, split={cfg_split.split}")

    samples_num = len(days.sample_id)
    # create dataset
    dataset = TokLoader(
        days=days,
        mapping=mapping,
        data_dir=cfg_data.data_dir,
        seq_size=cfg_data.seq_size,
        ctx_size=cfg_data.ctx_size,
        channels=cfg_data.channels,
        vocab_size=cfg_data.vocab_size,
        anno=cfg_data.anno,
        rollout_size=cfg_data.rollout_size,
        limit=cfg_split.get("samples_num", None),
        should_augment=cfg_split.get("should_augment", False),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_split.batch_size,
        num_workers=cfg_split.workers_num,
        shuffle=cfg_split.get("should_augment", False),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg_split.workers_num > 0,
    )
    return dataloader
