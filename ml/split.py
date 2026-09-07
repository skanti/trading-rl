import os
from functools import partial
from pathlib import Path
import itertools
from datetime import date, datetime, timedelta
import multiprocessing as mp
import argparse
import logging

from glob import glob
import numpy as np
from trading_rl.market_data.schema import validate_bar_columns
from tqdm import tqdm
import pandas as pd
from rich.logging import RichHandler

# pip install tqdm pyarrow pandas numpy

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("TRAIN")

ANNO = date(2010, 1, 1)


def rot13(text: str) -> str:
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cba = abc[::-1]
    return text.translate(str.maketrans(abc, cba))


def split_npy(npy_path: str, sod: pd.DataFrame, eod: pd.DataFrame) -> list:
    sample_id = Path(npy_path).stem
    if not os.path.exists(npy_path):
        return []
    data = np.load(npy_path)
    validate_bar_columns(data, "1Min", "minute bars")
    secs = data[:, 0]
    ticks_num = data.shape[0]
    assert (secs >= 0).all(), f"Negative secs, sample_id={sample_id}"
    is_sorted_strictly = np.all(secs[:-1] < secs[1:])
    assert is_sorted_strictly, f"Secs not strictly sorted, sample_id={sample_id}"

    # to dt
    dt = pd.to_datetime(secs, unit="s", origin=ANNO, utc=True).tz_convert("US/Eastern")

    # cut window
    idx_start = np.searchsorted(dt, sod, side="right") - 1
    idx_end = np.searchsorted(dt, eod, side="right") - 1

    segments = zip(idx_start, idx_end)
    seqs = []
    for i, (a, b) in enumerate(segments):
        today = days[i].date()
        assert today == sod[i].date()
        segment = b - a
        # few checks
        if segment <= 300:
            continue
        # get times
        win_seq = pd.to_datetime(dt[a:b])
        dt_first, dt_last = win_seq[0], win_seq[-1]
        # few checks
        if (sod[i] - dt_first).total_seconds() > 15 * 60:
            continue
        if (eod[i] - dt_last).total_seconds() > 15 * 60:
            continue

        # assert (win_seq.date == today).all(), f"Window error, sample_id={sample_id}"
        assert dt_first <= sod[i], f"Window must <= SOD, dt={dt_first}, sod={sod[i]}, sample_id={sample_id}"
        assert dt_last <= eod[i], f"Window must <= EOD, dt={dt_last}, eod={eod[i]}, sample_id={sample_id}"
        # append to seq
        today_str = today.strftime("%Y-%m-%d")
        seqs.append((sample_id, today_str, a, b))

    return seqs


parser = argparse.ArgumentParser()
parser.add_argument("--sample_ids", type=str, required=True)
parser.add_argument("--npy_dir", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--workers_num", type=int, default=16)

if __name__ == "__main__":
    args = parser.parse_args()

    today = datetime.now().date()

    since = date(2016, 1, 1)
    days = pd.date_range(since, today, freq="D")
    assert days[0].date() == since, f"First day does not match"
    assert days[-1].date() == today, f"Last day does not match"
    # Get schedule in ET
    sod = days + timedelta(hours=9, minutes=30, seconds=0)
    eod = days + timedelta(hours=16, minutes=0, seconds=0)
    sod = sod.tz_localize("US/Eastern")
    eod = eod.tz_localize("US/Eastern")

    if args.sample_ids.endswith(".txt"):
        with open(args.sample_ids, "r") as f:
            sample_ids = f.read().splitlines()
    else:
        sample_ids = [args.sample_ids]
    samples = [f"{args.npy_dir}/{x}.npy" for x in sample_ids]
    samples_num = len(samples)
    assert samples_num > 0, f"No samples found, npy_dir={args.npy_dir}"
    logger.info(f"Samples found, samples_num={samples_num}")

    fn = partial(split_npy, sod=sod, eod=eod)
    if args.workers_num == 1:
        data = list(map(fn, samples))
    else:
        with mp.Pool(args.workers_num) as pool:
            seqs = list(tqdm(pool.imap(fn, samples), total=samples_num))

    seqs = list(itertools.chain.from_iterable(seqs))
    seqs_num = len(seqs)
    logger.info(f"Data split, seqs_num={seqs_num}")
    np.random.shuffle(seqs)
    df = pd.DataFrame(seqs, columns=("sample_id", "date", "sod_idx", "eod_idx"))
    assert (df.sod_idx < df.eod_idx).all()
    df.to_csv(args.out_path, index=False)
    logger.info("Done")
