from datetime import datetime
from pathlib import Path

import pandas as pd
import fsspec
import numpy as np
import einops


def rot13(text: str) -> str:
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cba = abc[::-1]
    return text.translate(str.maketrans(abc, cba))


class Tokenizer:
    def __init__(
        self,
        anno: str | datetime,
        mapping: dict,
        seq_size: int,
        ctx_size: int,
        channels: int,
        vocab_size: int,
    ):
        if isinstance(anno, str):
            anno = datetime.strptime(anno, "%Y-%m-%d")
        self.anno = anno
        self.mapping = mapping
        self.mapping_inv = {v: k for k, v in mapping.items()}
        self.seq_size = seq_size
        self.ctx_size = ctx_size
        self.channels = channels
        self.K = vocab_size

        assert self.channels == 2, f"Tokenizer now supports price/volume only, channels={channels}"
        self.vol_fac = 1.0

    def parse(self, sample_path: str, segment: np.ndarray, rot: bool = False) -> None:
        sample_id = Path(sample_path).stem

        with fsspec.open(sample_path, "rb") as f:
            data = np.load(f)

        data = data[segment]

        # check shape
        assert data.ndim == 2
        _, columns_num = data.shape
        assert columns_num >= 3, f"Expected at least [secs, price, volume] columns, got {columns_num}"

        # parse data
        secs, temp, vol = data[:, 0], data[:, 1], data[:, 2]

        # decode
        temp = temp / 1000

        if rot:
            sample_id = rot13(sample_id)

        # save
        self.sample_id = sample_id
        self.secs = secs
        self.temp = temp
        self.vol = vol

    def secs_to_ts(self, secs: np.ndarray) -> np.ndarray:
        dt = pd.to_datetime(secs, unit="s", origin=self.anno, utc=True).tz_convert("US/Eastern")
        minutes = dt.hour * 60 + dt.minute
        weekdays = dt.weekday
        days = dt.day
        months = dt.month
        years = dt.year - 2000
        ts = np.stack((minutes, weekdays, days, months, years), axis=1)
        return ts, dt

    def tokenize(self) -> tuple:
        # time steps
        ts, dt = self.quick_ts(self.secs)
        self.dt_first, self.dt_last = dt[0], dt[-1]

        # city
        city = self.mapping[self.sample_id]
        city = np.array([city])
        anno = pd.Timestamp(self.anno, tz="UTC")
        years = (dt[0] - anno).days
        years = np.array([years])
        op = np.array([self.temp[0]])

        # make ctx
        city_tok = city.astype(np.float32)
        op_tok = op.astype(np.float32)
        assert (op_tok >= 0).all()
        ctx = np.concatenate((city_tok, op_tok), axis=0)

        # tokenize temp
        K = self.K - 1  # avoid overfilling
        L = K // 2
        temp = (self.temp - op) / op * 10000
        temp = temp.clip(-L, L) + L
        temp_tok = np.round(temp).astype(np.int32)
        # tokenize vol
        vol = np.sqrt(self.vol) * self.vol_fac
        vol = np.clip(vol, 0, K)
        vol_tok = np.round(vol).astype(np.int32)

        # to sequence
        seq = np.stack((temp_tok, vol_tok), axis=1).flatten()

        # ctx
        assert ts.shape[0] == seq.shape[0]
        assert (seq >= 0).all() and (seq <= K).all()

        return ctx, seq, ts

    def quick_ts(self, secs: np.ndarray) -> np.ndarray:
        ts, dt = self.secs_to_ts(secs)

        # concat times
        ts = einops.repeat(ts, "s c -> (s n) c", n=self.channels)
        return ts, dt

    def detokenize(self, seq: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        # check
        assert seq.ndim == 2
        assert ctx.ndim == 2

        n = self.channels
        seq = einops.rearrange(seq, "b (s n) -> b s n", n=n).cpu().numpy()
        ctx = ctx.cpu().numpy()

        # split
        temp_tok, vol_tok = np.split(seq, n, axis=-1)

        # ctx
        city, first = ctx[:, 0], ctx[:, 1]

        # to value
        vol = np.square(vol_tok / self.vol_fac)
        first = einops.rearrange(first, "b -> b 1 1")
        L = (self.K - 1) // 2
        temp = (temp_tok - L) / 10000 * first + first
        seq1 = np.concatenate((temp * 1000, vol), axis=-1)
        return seq1.astype(np.int32)
