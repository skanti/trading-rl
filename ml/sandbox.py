import argparse
import json
from pathlib import Path
import time
from functools import partial
import logging
from datetime import date, datetime, timedelta, UTC
from zoneinfo import ZoneInfo
import multiprocessing as mp

import tabulate
from munch import Munch
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import numpy as np
from trading_rl.market_data.schema import validate_bar_columns
from rich.logging import RichHandler

from . import model

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("ANALYZE")

# pip install numpy tqdm pandas rich

ANNO = date(2010, 1, 1)


def pretty_print(df: pd.DataFrame, name: str | None = None) -> None:
    print("")
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid", floatfmt=".3f"))
    print("")


def sec2dt(sec: int) -> datetime:
    return ANNO + timedelta(seconds=int(sec))


def rot13(text: str) -> str:
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cba = abc[::-1]
    return text.translate(str.maketrans(abc, cba))


def load_npy(npy_path: str) -> pd.DataFrame:
    data = np.load(npy_path)
    validate_bar_columns(data, "1Min", "minute bars")

    assert data.ndim == 2
    secs = data[:, 0].astype(np.int32)
    temp = data[:, 1] / 1000
    dt = pd.to_datetime(secs, unit="s", origin=ANNO, utc=True).tz_convert("US/Eastern")
    df = Munch({"secs": secs, "temp": temp, "dt": dt})
    return df


class Heuristic:
    def __init__(self, window: int = 240, dip_thresh: float = 0.01):
        self.window = window
        self.dip_thresh = dip_thresh

    def play(
        self,
        temp: np.ndarray,
        secs: np.ndarray,
        sod: int,
        eod: int,
        **kwargs,
    ) -> int:
        N = len(temp)
        assert N == len(secs)

        # # oracle
        # j = temp.argmin()
        # return j, temp[j]

        for i in range(2, N):
            temp_hist, sec_hist = temp[:i], secs[:i]
            temp_now, sec_now = temp[i], secs[i]

            mask_window = (sec_now - sec_hist) // 60 <= self.window
            # if (eod - sec_now) < 60*30:
            #     break
            change = (temp_hist - temp_now) / temp_hist
            mask_change = change > self.dip_thresh
            # final mask
            mask = mask_change & mask_window
            if mask.any():
                j = mask.argmax()
                return i, temp[i]
        return None


def inference(
    symbol: str,
    agent: any,
    days: pd.DatetimeIndex,
    sod: pd.DatetimeIndex,
    eod: pd.DatetimeIndex,
    gt_dir: str,
) -> None:

    gt_path = f"{gt_dir}/{symbol}.npy"

    gt = load_npy(gt_path)

    # cut window segments
    idx_start = np.searchsorted(gt.dt, sod, side="right") - 1
    idx_end = np.searchsorted(gt.dt, eod, side="right") - 1
    segments = list(zip(idx_start, idx_end))

    anno = pd.to_datetime(ANNO).tz_localize("UTC")
    sod_s = (sod - anno).total_seconds()
    eod_s = (eod - anno).total_seconds()
    schedule = pd.date_range("09:30", "16:00", freq="1min").tz_localize("US/Eastern")

    # process segments
    res = []
    for i, (a, b) in enumerate(tqdm(segments)):
        today = days[i].date()
        assert today == sod[i].date()
        segment = b - a
        if segment == 0:
            continue
        if segment < 300:
            continue
        dt = gt.dt[a : b + 1]
        # checks
        dt_first, dt_last = dt[0], dt[-1]
        # few checks
        if (sod[i] - dt_first).total_seconds() > 15 * 60:
            continue
        if (eod[i] - dt_last).total_seconds() > 15 * 60:
            continue

        schedule = schedule.map(lambda x: x.replace(year=today.year, month=today.month, day=today.day))
        idx = np.searchsorted(dt, schedule, side="right") - 1
        assert len(idx) == len(schedule)
        assert (dt[idx] <= schedule).all(), (
            f"Latest match error, symbol={symbol}, today={today}"
        )
        temp = gt.temp[a + idx]
        secs = gt.secs[a + idx]
        dt = gt.dt[a + idx]

        # t0 = time.perf_counter()
        hit = agent.play(
            temp=temp[:240],
            secs=secs[:240],
            sod=sod[i],
            eod=eod[i],
            return_hit=True,
        )
        # t1 = time.perf_counter()
        # print(f"detect time, duration={t1 - t0:.4f}s")
        if hit is not None:
            # buy
            op = temp[0]
            idx_buy, temp_buy = hit
            dt_buy = dt[idx_buy].tz_convert("US/Eastern")
            assert temp_buy == temp[idx_buy]
            # sell
            idx_sell = -1  # today eod
            temp_sell = temp[idx_sell]
            dt_sell = dt[idx_sell].tz_convert("US/Eastern")
            assert dt_sell.date() == dt_buy.date(), f"Sell date error, symbol={symbol}, date={today}"
            wc = 2000
            qty = np.floor(wc / temp_buy)
            wc = qty * temp_buy
            pro = (temp_sell - temp_buy) * qty
            change = (temp_sell - temp_buy) / op * 100
            acc = pro >= 0
            res.append(
                {
                    "symbol": symbol,
                    "pro": pro,
                    "wc": wc,
                    "acc": acc,
                    "date": today,
                    "n": 1,
                    "dt_buy": dt_buy,
                    "temp_buy": temp_buy,
                    "dt_sell": dt_sell,
                    "temp_sell": temp_sell,
                    "pro": pro,
                }
            )
    # baseline
    temp_start, temp_end = gt.temp[idx_start[0]], gt.temp[idx_end[-1] - 1]
    pl_baseline = (temp_end - temp_start) / temp_start * 100
    baseline = {"symbol": symbol, "pl_baseline": pl_baseline}
    return pd.DataFrame(res), pd.Series(baseline)


parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=str, required=True)
parser.add_argument("--gt_dir", type=str, required=True)
parser.add_argument("--soy", type=str, default=None)
parser.add_argument("--eoy", type=str, default=None)
parser.add_argument("--workers_num", type=int, default=1)
parser.add_argument("--mode", type=str, default="month")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None)


def main() -> None:
    args = parser.parse_args()

    if Path(args.samples).exists():
        with open(args.samples, "r") as f:
            samples = set(f.read().splitlines())
    else:
        samples = [args.samples]

    samples_num = len(samples)

    today = datetime.now().date()

    since = date(2016, 1, 1)
    days = pd.date_range(since, today, freq="D")
    assert days[0].date() == since, f"First day does not match"
    assert days[-1].date() == today, f"Last day does not match"

    # create days
    if args.soy is None:
        soy = today
    else:
        soy = datetime.strptime(args.soy, "%Y-%m-%d").date()

    if args.eoy is None:
        eoy = today
    else:
        eoy = datetime.strptime(args.eoy, "%Y-%m-%d").date()
    days = (eoy - soy).days
    days = pd.to_datetime(np.arange(days + 1), unit="D", origin=soy)

    # Get schedule in ET
    sod = days + timedelta(hours=9, minutes=30, seconds=0)
    eod = days + timedelta(hours=16, minutes=0, seconds=0)

    # back to UTC
    sod = sod.tz_localize("US/Eastern")
    eod = eod.tz_localize("US/Eastern")
    soy_str, eoy_str = soy.strftime("%Y-%m-%d"), eoy.strftime("%Y-%m-%d")

    if args.config.endswith(".yaml"):
        cfg = OmegaConf.load(args.config)
        agent = model.Agent(**cfg.model.agent, batch_size=1).to("cuda")
        agent.load_checkpoint(args.checkpoint_path)
    else:
        agent = Heuristic()

    fn = partial(inference, agent=agent, days=days, sod=sod, eod=eod, gt_dir=args.gt_dir)
    if args.workers_num == 1:
        res = list(tqdm(map(fn, samples), total=samples_num))
    else:
        with mp.Pool(processes=args.workers_num) as pool:
            res = list(tqdm(pool.imap(fn, samples), total=samples_num))

    # summary
    df, baseline = zip(*res)
    df, baseline = pd.concat(df), pd.DataFrame(baseline)
    if len(df) == 0:
        logger.warning(f"No trades done, soy={soy_str}, eoy={eoy_str}")
        return
    df = df.sort_values(by=["dt_buy", "symbol"]).reset_index(drop=True)
    print(df)

    # # tmp
    # sec = np.mean(df.dt_buy.dt.hour * 3600 + df.dt_buy.dt.minute * 60 + df.dt_buy.dt.second)
    # avg_time = pd.to_datetime(sec, unit="s").time()

    df.set_index("date").to_csv(f"/tmp/sandbox_{soy_str}_{eoy_str}.csv")
    df = df.drop(columns=["dt_buy", "temp_buy", "dt_sell", "temp_sell"])

    df_symbol = df.drop(columns=["date"]).groupby("symbol").agg("sum")
    df_symbol = df_symbol.merge(baseline, on="symbol", how="left", suffixes=("", "_baseline"))

    wc = df.drop(columns=["symbol"]).groupby("date").agg("sum").wc.mean()
    pl = df.pro.sum() / wc
    pl_baseline = df_symbol.pl_baseline.mean()

    if (eoy - soy).days <= 14:
        period = "D"
    elif (eoy - soy).days < 365:
        period = "M"
    else:
        period = "Y"
    df_time = (
        df.assign(date=pd.to_datetime(df.date).dt.to_period(period)).drop(columns=["symbol"]).groupby("date").agg("sum")
    )
    df_time["pl"] = df_time.pro / wc * 100
    df_time["acc"] = df_time.acc / df_time.n * 100
    pretty_print(df_time)

    print("*******")
    summary = pd.DataFrame(
        [
            {
                "": "total",
                "pl_ours": pl * 100,
                "pro": df.pro.sum(),
                "wc_avg": wc,
                "acc": df.acc.mean() * 100,
                "n": str(df.n.sum()),
                "pl_baseline": pl_baseline,
                "from": soy_str,
                "until": eoy_str,
            }
        ]
    ).set_index("")
    pretty_print(summary)


if __name__ == "__main__":
    main()
