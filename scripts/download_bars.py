import os
import argparse
import re
import json
import pathlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tarfile
import logging
from datetime import datetime, timedelta, UTC
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("DOWNLOAD_BARS")

ANNO = datetime(2010, 1, 1, tzinfo=UTC)
REQUEST_TIMEOUT = (10, 90)
REQUEST_RETRIES = 10
ALPACA_REQUESTS_PER_MINUTE = 180
ALPACA_SIP_DELAY = timedelta(minutes=20)


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if delay:
            time.sleep(delay)


ALPACA_RATE_LIMITER = RateLimiter(ALPACA_REQUESTS_PER_MINUTE)


def save_as_npy(data: pd.DataFrame, out_path: str) -> bool:
    name = os.path.basename(out_path)
    required_columns = {"t", "o", "v", "n"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        logger.warning("Columns missing, name=%s, columns=%s", name, sorted(missing_columns))
        return False
    data["dt"] = pd.to_datetime(data["t"], utc=True)  # convert to datetime

    # load data
    assert data.dt.is_monotonic_increasing, f"Datetime not sorted, name={name}"

    # filter table
    m = data.dt >= ANNO
    data = data[m]
    if data.empty:
        logger.warning("No bars remain after timestamp filtering, name=%s", name)
        return False

    # convert to numpy
    secs = (data.dt - ANNO).dt.total_seconds()
    # Keep the established price-mills schema, but round instead of silently
    # truncating sub-mill values toward zero.
    price = np.rint(data.o * 1000)
    volume = data.v
    num = data.n
    if not all(np.isfinite(values).all() for values in (secs, price, volume, num)):
        logger.warning("Non-finite values, name=%s", name)
        return False

    # check smaller than int32_max
    if secs.max() > np.iinfo(np.int32).max:
        logger.warning(f"Secs too large, name={name}")
        return False
    if volume.max() > np.iinfo(np.int32).max:
        logger.warning(f"Volume too large, name={name}")
        return False
    if price.max() > np.iinfo(np.int32).max:
        logger.warning(f"Price too large, name={name}")
        return False
    if num.max() > np.iinfo(np.int32).max:
        logger.warning(f"Num too large, name={name}")
        return False

    if price.min() < 0:
        logger.warning(f"Price negative, name={name}")
        return False
    if num.min() < 0:
        logger.warning(f"Num negative, name={name}")
        return False

    # casting
    dtype = np.int32
    secs, price, volume, num = map(lambda x: x.astype(dtype), [secs, price, volume, num])

    # check
    dt_roundtrip = pd.to_datetime(secs, unit="s", origin=ANNO.date(), utc=True)
    assert (dt_roundtrip == data.dt).all(), f"Datetime encoding, name={name}, ANNO={ANNO}"

    # to matrix
    array = np.stack([secs, price, volume, num], axis=-1)
    assert array.ndim == 2

    # save
    # A process interruption must never leave a file that --skip_existing
    # mistakes for a complete ticker.
    tmp_path = f"{out_path}.part"
    with open(tmp_path, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return True


def request_json(
    session: requests.Session,
    url: str,
    params: dict,
    headers: dict,
    ticker: str,
) -> dict:
    """Request one page with bounded retries for rate limits and server errors."""
    for attempt in range(REQUEST_RETRIES):
        try:
            ALPACA_RATE_LIMITER.wait()
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                rate_limit_reset = response.headers.get("X-Ratelimit-Reset")
                if retry_after:
                    delay = float(retry_after)
                elif rate_limit_reset:
                    delay = max(float(rate_limit_reset) - time.time(), 1.0) + random.random()
                else:
                    delay = min(2**attempt, 60) + random.random()
                logger.warning(
                    "Transient response, ticker=%s, status=%d, retry=%d/%d, delay=%.1fs",
                    ticker,
                    response.status_code,
                    attempt + 1,
                    REQUEST_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"non-retryable HTTP {response.status_code} for ticker={ticker}"
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("response JSON must be an object")
            if "message" in payload and "bars" not in payload:
                raise RuntimeError(f"API error for ticker={ticker}: {payload['message']}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt + 1 == REQUEST_RETRIES:
                raise RuntimeError(f"request failed for {ticker} after {REQUEST_RETRIES} attempts") from exc
            delay = min(2**attempt, 60) + random.random()
            logger.warning(
                "Request error, ticker=%s, retry=%d/%d, delay=%.1fs, error=%s",
                ticker,
                attempt + 1,
                REQUEST_RETRIES,
                delay,
                type(exc).__name__,
            )
            time.sleep(delay)
    raise AssertionError("retry loop exited unexpectedly")


def clean_ticker(ticker: str) -> str:
    # clean ticker
    pattern = r"^[A-Z]{2}-"
    if re.match(pattern, ticker):
        ticker = ticker[3:]
    ticker = ticker.upper()
    # restore dot characters
    ticker = ticker.replace("-", ".")
    return ticker


def download_bars_alpaca(ticker: str, since: datetime) -> pd.DataFrame:
    # clean ticker
    ticker = clean_ticker(ticker)
    # set date
    dt_start = since
    # The subscription exposes delayed SIP data. Rounding the end up to the
    # end of the current day causes Alpaca to reject the entire historical
    # request because it includes restricted recent/future timestamps.
    dt_end = datetime.now(UTC) - ALPACA_SIP_DELAY
    dt_start = dt_start.replace(hour=0, minute=0, second=0, microsecond=0)
    headers = {
        "APCA-API-KEY-ID": os.environ["ALPACA_DATA_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_DATA_SECRET"],
        "accept": "application/json",
    }
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": ticker,
        "timeframe": "1Min",
        "limit": 10000,
        "adjustment": "split",
        "feed": "sip",
        "sort": "asc",
        "start": dt_start.isoformat(),
        "end": dt_end.isoformat(),
    }
    bars = []
    page_token = None
    with requests.Session() as session:
        while True:
            res = request_json(session, base_url, params, headers, ticker)
            page_bars = res.get("bars", {}).get(ticker, [])
            bars.extend(page_bars)
            next_page_token = res.get("next_page_token")
            if next_page_token is None:
                break
            if next_page_token == page_token:
                raise RuntimeError(f"pagination token repeated for ticker={ticker}")
            page_token = next_page_token
            params["page_token"] = next_page_token

    # to dataframe
    df = pd.DataFrame(bars)
    return df


def download_bars_polygon(ticker: str, since: int) -> pd.DataFrame:
    # clean ticker
    ticker = clean_ticker(ticker)
    # set date
    date_from = since
    date_to = datetime.now().date()
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str = date_to.strftime("%Y-%m-%d")
    headers = {"accept": "application/json"}
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date_from_str}/{date_to_str}"
    params = {
        "limit": 50000,
        "adjusted": True,
        "sort": "asc",
        "apiKey": os.environ["POLYGON_KEY"],
    }
    bars = []
    while True:
        res = requests.get(url, params=params, headers=headers)
        res = res.json()
        if res["resultsCount"] == 0:
            break
        bars.extend(res["results"])
        next_url = res.get("next_url")
        if next_url is None:
            break
        url = next_url
    # cleanup
    bars_clean = []
    for bar in bars:
        # convert to datetime
        ts_et = datetime.fromtimestamp(bar["t"] / 1000, tz=UTC)
        bars_clean.append(
            {
                "datetime": ts_et.strftime("%Y-%m-%d %H:%M:%S"),
                "price": bar["o"],
                "volume": bar["v"],
            }
        )
    df = pd.DataFrame(bars_clean)
    return df


def process_ticker(
    ticker: str, source: str, out_dir: str, since: datetime, skip_existing: bool = False
) -> bool:
    out_path = f"{out_dir}/{ticker}.npy"
    if skip_existing and os.path.exists(out_path):
        logger.info(f"Ticker exists already - skip, ticker={ticker}")
        return True
    try:
        if source == "alpaca":
            df = download_bars_alpaca(ticker, since)
        elif source == "polygon":
            df = download_bars_polygon(ticker, since)
        else:
            raise ValueError(f"Unknown source, source={source}")
        if len(df) == 0:
            logger.warning(f"No bars found, ticker={ticker}")
            return False
        return save_as_npy(data=df, out_path=out_path)
    except Exception:
        logger.exception("Ticker failed, ticker=%s", ticker)
        return False


def pack_to_archive(arc_dir: str) -> str:
    logger.info(f"Packing to archive, arc_dir={arc_dir}")
    name = os.path.basename(arc_dir)
    arc_path = f"{arc_dir}/{name}.tar"
    files = list(pathlib.Path(arc_dir).glob("*.npy"))
    files_num = len(files)
    if files_num == 0:
        logger.warning(f"No files found, arc_path={arc_path}")
        return None
    with tarfile.open(arc_path, "w") as tar:
        for npy_path in files:
            tar.add(npy_path, arcname=os.path.basename(npy_path))
    logger.info(f"Packed to archive, archive_path={arc_path}")
    return arc_path


def rot13(text: str) -> str:
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cba = abc[::-1]
    return text.translate(str.maketrans(abc, cba))


def main(
    source: str,
    tickers_path: str,
    out_dir: str,
    since: datetime,
    workers_num: int = 8,
    archive: bool = False,
    skip_existing: bool = False,
    rot: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()

    if tickers_path.endswith(".json"):
        with open(tickers_path, "r") as f:
            tickers = json.load(f)["sample_ids"]
    elif tickers_path.endswith((".csv", ".txt")):
        with open(tickers_path, "r") as f:
            tickers = f.read().splitlines()
    else:
        tickers = [tickers_path]

    if rot:
        tickers = [rot13(ticker) for ticker in tickers]
    tickers_num = len(tickers)
    assert tickers_num > 0, "No tickers found"
    logger.info(f"Tickers provided, tickers_num={tickers_num}")

    fn = partial(
        process_ticker,
        source=source,
        out_dir=out_dir,
        since=since,
        skip_existing=skip_existing,
    )
    if workers_num == 0:
        res = list(tqdm(map(fn, tickers), total=tickers_num, desc="Processing tickers"))
    else:
        with ThreadPoolExecutor(max_workers=workers_num) as pool:
            res = list(tqdm(pool.map(fn, tickers), total=tickers_num, desc="Processing tickers"))
    t1 = time.perf_counter()

    success_num = sum(res)
    failed_tickers = [ticker for ticker, succeeded in zip(tickers, res) if not succeeded]
    failed_path = pathlib.Path(out_dir) / "_failed_tickers.txt"
    failed_path.write_text("".join(f"{ticker}\n" for ticker in failed_tickers))
    logger.info(f"Downloading done, success_num={success_num}, tickers_num={tickers_num}")
    if failed_tickers:
        logger.warning("Tickers failed, failed_num=%d, manifest=%s", len(failed_tickers), failed_path)

    duration = t1 - t0
    if archive:
        pack_to_archive(out_dir)
    logger.info(f"Done, duration={duration:0.2f}s")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    required=True,
    choices=["alpaca", "polygon"],
    help="Source of data",
)
parser.add_argument("--tickers_path", type=str, required=True, help="File with tickers")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the dataset")
parser.add_argument("--days", type=int, default=None, help="Days to look back for data")
parser.add_argument("--since", type=str, default=None, help="Date to look back for data")
parser.add_argument("--workers_num", type=int, default=8, help="Number of workers to use")
parser.add_argument("--archive", action="store_true", help="Should pack into tar file?")
parser.add_argument("--skip_existing", action="store_true", help="Skip existing files?")
parser.add_argument("--rot", action="store_true", help="Apply rot13?")

if __name__ == "__main__":
    args = parser.parse_args()
    assert (
        args.days is not None or args.since is not None
    ), "Either --days or --since must be provided"
    if args.since is not None:
        dt_since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
    elif args.days is not None:
        dt_since = datetime.now(UTC) - timedelta(days=args.days)
    else:
        raise ValueError("Either --days or --since must be provided")

    main(
        source=args.source,
        tickers_path=args.tickers_path,
        out_dir=args.out_dir,
        since=dt_since,
        workers_num=args.workers_num,
        archive=args.archive,
        skip_existing=args.skip_existing,
        rot=args.rot,
    )
