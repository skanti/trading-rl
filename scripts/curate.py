import re
import os
from pathlib import Path
from datetime import datetime, timedelta
import multiprocessing as mp
import argparse

from glob import glob
import numpy as np
from trading_rl.market_data.schema import BAR_INDEX, validate_bar_columns
from tqdm import tqdm
import pandas as pd
import pyarrow

# pip install tqdm pyarrow pandas numpy


def rot13(text: str) -> str:
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cba = abc[::-1]
    return text.translate(str.maketrans(abc, cba))


def load_npy(npy_path: str) -> np.ndarray:
    sample_id = Path(npy_path).stem
    name = rot13(sample_id)
    data = np.load(npy_path)
    validate_bar_columns(data, "1Min", "minute bars")
    secs = data[:, 0]
    is_sorted = np.all(secs[:-1] < secs[1:])

    temp = data[:, 1] / 1000
    vol = data[:, BAR_INDEX["volume"]]
    ticks_num = data.shape[0]

    assert secs[0] > 0, f"Negative secs, sample_id={sample_id}"

    anno = datetime(2010, 1, 1)
    dt = pd.to_datetime(secs, unit="s", origin=anno)

    df = pd.DataFrame(
        {
            "datetime": dt,
            "vol": vol,
            "temp": temp,
        }
    )
    assert df.shape[0] == ticks_num

    first_date = df.datetime.min()
    last_date = df.datetime.max()
    data100 = df[df.datetime > last_date - timedelta(days=100)]

    temp100 = data100.temp.mean()
    vol100 = data100.vol.mean()
    temp_max = df.temp.max()
    vol_max = df.vol.max()
    age = (last_date - first_date).days
    rain = (temp100 * vol100).mean()

    assert df.shape[0] == ticks_num

    ret = {
        "sample_id": sample_id,
        "name": name,
        "is_sorted": is_sorted,
        "first_date": first_date,
        "temp100": temp100,
        "vol100": vol100,
        "temp_max": temp_max,
        "vol_max": vol_max,
        "rain": rain,
        "age": age,
        "ticks": ticks_num,
        "latest": last_date,
    }
    return ret


parser = argparse.ArgumentParser()
parser.add_argument("--npy_dir", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--use_cache", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.out_path.endswith("_frames.csv")

    result = []
    for prefix in ["HG"]:
        samples = [f for f in glob(f"{args.npy_dir}/{prefix}-*.npy")]
        samples_num = len(samples)
        assert samples_num > 0, f"No samples found, npy_dir={args.npy_dir}"
        print(f"Samples found, samples_num={samples_num}")
        cache_path = f"/tmp/ppv1_{prefix}.cache"
        np.random.shuffle(samples)

        if args.use_cache and os.path.exists(cache_path):
            data = pd.read_feather(cache_path)
        else:
            # data = list(map(load_npy, samples))
            with mp.Pool(32) as pool:
                data = list(tqdm(pool.imap(load_npy, samples), total=samples_num))
            pd.DataFrame(data).to_feather(cache_path)

        data_num = len(data)
        print(f"Data loaded, samples_num={samples_num}")
        df = (
            pd.DataFrame(data)
            .sort_values(by="rain", ascending=False)
            .reset_index(drop=True)
        )

        rain = df.rain
        ticks = df.ticks

        # check activity
        m_ticks = ticks > 1e5
        m_sorted = df.is_sorted
        m_rain = rain > rain.quantile(0.1)
        m_vol_low = df.vol100 > df.vol100.quantile(0.1)
        m_temp_low = df.temp100 > 10  # df.temp100.quantile(0.1)
        m_temp_high = df.temp_max < 2000
        m_base = m_ticks & m_rain & m_vol_low & m_temp_low & m_temp_high & m_sorted
        mask = m_base
        df = df[mask].reset_index(drop=True)
        samples_num = len(df)
        df["weight"] = 1.0 - (df.index + 1) / samples_num
        result.append(df)
        print(f"Data filtered, samples_num={samples_num}")

    # collect
    df = (
        pd.concat(result).sort_values(by="rain", ascending=False).reset_index(drop=True)
    )
    samples_num = len(df)
    df = df[["sample_id", "ticks", "latest", "weight"]]
    df.to_csv(args.out_path, index=False)
    mapping_path = args.out_path.replace("_frames.csv", "_mapping.txt")
    breakpoint()
    df["sample_id"].to_csv(mapping_path, index=False, header=0)
    print(f"Saved, samples_num={samples_num}, out_path={args.out_path}")
