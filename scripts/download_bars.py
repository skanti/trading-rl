import os
import argparse
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
DATASET_MANIFEST = "_download_manifest.json"
BAR_COLUMNS = {
    "1Min": ("seconds", "open_mills", "volume", "trades"),
    "1Day": (
        "seconds",
        "open_mills",
        "high_mills",
        "low_mills",
        "close_mills",
        "volume",
        "trades",
        "vwap_mills",
    ),
}


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


def dataframe_to_array(
    data: pd.DataFrame, name: str, timeframe: str = "1Min"
) -> np.ndarray | None:
    """Convert API bars to the compact integer on-disk schema."""
    required_columns = {"t", "o", "v", "n"}
    if timeframe == "1Day":
        required_columns.update({"h", "l", "c", "vw"})
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        logger.warning(
            "Columns missing, name=%s, columns=%s", name, sorted(missing_columns)
        )
        return None
    data = data.copy()
    data["dt"] = pd.to_datetime(data["t"], utc=True)  # convert to datetime

    # load data
    assert data.dt.is_monotonic_increasing, f"Datetime not sorted, name={name}"

    # filter table
    m = data.dt >= ANNO
    data = data[m]
    if data.empty:
        logger.warning("No bars remain after timestamp filtering, name=%s", name)
        return None

    # convert to numpy
    secs = (data.dt - ANNO).dt.total_seconds()
    # Keep the established price-mills schema, but round instead of silently
    # truncating sub-mill values toward zero.
    volume = data.v
    num = data.n
    prices = [np.rint(data.o * 1000)]
    if timeframe == "1Day":
        prices.extend(np.rint(getattr(data, field) * 1000) for field in ("h", "l", "c"))
        prices.append(np.rint(data.vw * 1000))
    values_to_check = [secs, *prices, volume, num]
    if not all(np.isfinite(values).all() for values in values_to_check):
        logger.warning("Non-finite values, name=%s", name)
        return None

    # Keep minute bars in their established compact format. Symbols whose
    # split-adjusted values cannot be represented safely are logged and skipped.
    # Daily aggregate volumes can legitimately exceed int32 (including NVDA),
    # so the distinct daily schema remains int64.
    dtype = np.int64 if timeframe == "1Day" else np.int32
    integer_max = np.iinfo(dtype).max
    if secs.max() > integer_max:
        logger.warning(f"Secs too large, name={name}")
        return None
    if volume.max() > integer_max:
        logger.warning(f"Volume too large, name={name}")
        return None
    if any(price.max() > integer_max for price in prices):
        logger.warning(f"Price too large, name={name}")
        return None
    if num.max() > integer_max:
        logger.warning(f"Num too large, name={name}")
        return None

    if any(price.min() < 0 for price in prices):
        logger.warning(f"Price negative, name={name}")
        return None
    if num.min() < 0:
        logger.warning(f"Num negative, name={name}")
        return None

    # casting
    secs, volume, num, *prices = map(
        lambda x: x.astype(dtype), [secs, volume, num, *prices]
    )

    # check
    dt_roundtrip = pd.to_datetime(secs, unit="s", origin=ANNO.date(), utc=True)
    assert (dt_roundtrip == data.dt).all(), (
        f"Datetime encoding, name={name}, ANNO={ANNO}"
    )

    # to matrix
    if timeframe == "1Day":
        open_price, high, low, close, vwap = prices
        array = np.stack(
            [secs, open_price, high, low, close, volume, num, vwap], axis=-1
        )
    else:
        (open_price,) = prices
        array = np.stack([secs, open_price, volume, num], axis=-1)
    assert array.ndim == 2
    if not (array[:-1, 0] < array[1:, 0]).all():
        logger.warning("Duplicate or unsorted timestamps, name=%s", name)
        return None
    return array


def save_array(array: np.ndarray, out_path: str) -> None:
    """Atomically replace one ticker file with a validated array."""
    if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
        raise ValueError(f"invalid bar array shape: {array.shape}")
    if not (array[:-1, 0] < array[1:, 0]).all():
        raise ValueError("bar timestamps must be strictly increasing")

    # save
    # A process interruption must never leave a file that --skip_existing
    # mistakes for a complete ticker.
    tmp_path = f"{out_path}.part"
    with open(tmp_path, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def save_as_npy(data: pd.DataFrame, out_path: str, timeframe: str = "1Min") -> bool:
    array = dataframe_to_array(data, os.path.basename(out_path), timeframe)
    if array is None:
        return False
    save_array(array, out_path)
    return True


def merge_bar_arrays(base: np.ndarray, update: np.ndarray) -> np.ndarray | None:
    """Merge an exactly matching overlap, or return None for a full refresh.

    Alpaca bars are split-adjusted. A new split—or any historical correction—can
    therefore rewrite values already stored locally. We only append an update when
    every row in the common timestamp range is byte-for-byte identical.
    """
    for label, array in (("base", base), ("update", update)):
        if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
            raise ValueError(f"invalid {label} bar array shape: {array.shape}")
        if not (array[:-1, 0] < array[1:, 0]).all():
            raise ValueError(f"{label} bar timestamps must be strictly increasing")
    if base.shape[1] != update.shape[1]:
        raise ValueError(
            f"base/update bar column mismatch: {base.shape[1]} != {update.shape[1]}"
        )

    update_first = int(update[0, 0])
    base_last = int(base[-1, 0])
    if update_first > base_last:
        return None

    overlap_end = min(base_last, int(update[-1, 0]))
    base_start_idx = int(np.searchsorted(base[:, 0], update_first, side="left"))
    base_end_idx = int(np.searchsorted(base[:, 0], overlap_end, side="right"))
    update_end_idx = int(np.searchsorted(update[:, 0], overlap_end, side="right"))
    base_overlap = base[base_start_idx:base_end_idx]
    update_overlap = update[:update_end_idx]
    if not np.array_equal(base_overlap, update_overlap):
        return None

    if int(update[-1, 0]) <= base_last:
        return base.copy()
    merged = np.vstack((base[:base_start_idx], update))
    if not (merged[:-1, 0] < merged[1:, 0]).all():
        raise ValueError("merged bar timestamps are not strictly increasing")
    return merged


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
                    delay = (
                        max(float(rate_limit_reset) - time.time(), 1.0)
                        + random.random()
                    )
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
                raise RuntimeError(
                    f"API error for ticker={ticker}: {payload['message']}"
                )
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt + 1 == REQUEST_RETRIES:
                raise RuntimeError(
                    f"request failed for {ticker} after {REQUEST_RETRIES} attempts"
                ) from exc
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


def storage_ticker(ticker: str) -> str:
    """Normalize a symbol for an unprefixed on-disk filename."""
    symbol = str(ticker).upper()
    if symbol.startswith("ST-"):
        return symbol[3:].replace("-", ".")
    return symbol


def clean_ticker(ticker: str) -> str:
    """Normalize a stored symbol for Alpaca's dot-separated API notation."""
    return storage_ticker(ticker)


def download_bars_alpaca(
    ticker: str, since: datetime, timeframe: str = "1Min"
) -> pd.DataFrame:
    # clean ticker
    ticker = clean_ticker(ticker)
    # set date
    dt_start = since
    # The subscription exposes delayed SIP data. Rounding the end up to the
    # end of the current day causes Alpaca to reject the entire historical
    # request because it includes restricted recent/future timestamps.
    if timeframe == "1Day":
        # Never persist a still-forming daily bar: it would differ on the next
        # update and falsely look like a historical correction requiring a full
        # symbol refresh. Daily downloads intentionally lag until the next NY day.
        dt_end = datetime.now(ZoneInfo("America/New_York")).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(UTC) - timedelta(microseconds=1)
    else:
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
        "timeframe": timeframe,
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


def download_bars_polygon(
    ticker: str, since: datetime, timeframe: str = "1Min"
) -> pd.DataFrame:
    # clean ticker
    ticker = clean_ticker(ticker)
    # set date
    date_from = since
    date_to = datetime.now().date()
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str = date_to.strftime("%Y-%m-%d")
    headers = {"accept": "application/json"}
    polygon_span = {"1Min": "minute", "1Day": "day"}[timeframe]
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/"
        f"{polygon_span}/{date_from_str}/{date_to_str}"
    )
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
        bars_clean.append(
            {
                "t": datetime.fromtimestamp(bar["t"] / 1000, tz=UTC).isoformat(),
                "o": bar["o"],
                "v": bar["v"],
                "n": bar.get("n", 0),
            }
        )
    df = pd.DataFrame(bars_clean)
    return df


def download_ticker(
    ticker: str, source: str, since: datetime, timeframe: str = "1Min"
) -> pd.DataFrame:
    if source == "alpaca":
        return download_bars_alpaca(ticker, since, timeframe)
    if source == "polygon":
        return download_bars_polygon(ticker, since, timeframe)
    raise ValueError(f"Unknown source, source={source}")


def process_ticker(
    ticker: str,
    source: str,
    out_dir: str,
    since: datetime,
    skip_existing: bool = False,
    update_existing: bool = False,
    overlap_days: int = 30,
    timeframe: str = "1Min",
) -> bool:
    ticker = storage_ticker(ticker)
    out_path = f"{out_dir}/{ticker}.npy"
    if skip_existing and os.path.exists(out_path):
        logger.info(f"Ticker exists already - skip, ticker={ticker}")
        return True
    try:
        if update_existing and os.path.exists(out_path):
            base = np.load(out_path)
            expected_columns = len(BAR_COLUMNS[timeframe])
            if base.ndim != 2 or base.shape[1] != expected_columns or len(base) == 0:
                raise ValueError(f"invalid existing bar array shape: {base.shape}")
            base_start = ANNO + timedelta(seconds=int(base[0, 0]))
            base_last = ANNO + timedelta(seconds=int(base[-1, 0]))
            update_since = max(base_start, base_last - timedelta(days=overlap_days))
            update_df = download_ticker(ticker, source, update_since, timeframe)
            if update_df.empty:
                logger.warning("No update bars found, ticker=%s", ticker)
                return False
            update_array = dataframe_to_array(
                update_df, os.path.basename(out_path), timeframe
            )
            if update_array is None:
                return False
            merged = merge_bar_arrays(base, update_array)
            if merged is not None:
                save_array(merged, out_path)
                logger.info(
                    "Incremental update complete, ticker=%s, old_rows=%d, new_rows=%d",
                    ticker,
                    len(base),
                    len(merged),
                )
                return True

            logger.warning(
                "Historical overlap changed; downloading full retained history, ticker=%s",
                ticker,
            )
            full_df = download_ticker(ticker, source, base_start, timeframe)
            if full_df.empty:
                logger.warning("No full-refresh bars found, ticker=%s", ticker)
                return False
            full_array = dataframe_to_array(
                full_df, os.path.basename(out_path), timeframe
            )
            if full_array is None:
                return False
            if int(full_array[-1, 0]) < int(base[-1, 0]):
                logger.error(
                    "Full refresh ends before existing data; preserving old file, ticker=%s",
                    ticker,
                )
                return False
            save_array(full_array, out_path)
            logger.info(
                "Full refresh complete, ticker=%s, old_rows=%d, new_rows=%d",
                ticker,
                len(base),
                len(full_array),
            )
            return True

        df = download_ticker(ticker, source, since, timeframe)
        if len(df) == 0:
            logger.warning(f"No bars found, ticker={ticker}")
            return False
        return save_as_npy(data=df, out_path=out_path, timeframe=timeframe)
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
    update_existing: bool = False,
    overlap_days: int = 30,
    timeframe: str = "1Min",
    rot: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = pathlib.Path(out_dir) / DATASET_MANIFEST
    if update_existing and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_source = str(existing_manifest.get("source", ""))
        existing_timeframe = str(existing_manifest.get("timeframe", ""))
        if existing_source and existing_source != source:
            raise ValueError(
                f"existing dataset source is {existing_source}, requested {source}"
            )
        if existing_timeframe and existing_timeframe != timeframe:
            raise ValueError(
                f"existing dataset timeframe is {existing_timeframe}, "
                f"requested {timeframe}"
            )

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
        update_existing=update_existing,
        overlap_days=overlap_days,
        timeframe=timeframe,
    )
    if workers_num == 0:
        res = list(tqdm(map(fn, tickers), total=tickers_num, desc="Processing tickers"))
    else:
        with ThreadPoolExecutor(max_workers=workers_num) as pool:
            res = list(
                tqdm(
                    pool.map(fn, tickers), total=tickers_num, desc="Processing tickers"
                )
            )
    t1 = time.perf_counter()

    success_num = sum(res)
    failed_tickers = [
        ticker for ticker, succeeded in zip(tickers, res) if not succeeded
    ]
    failed_path = pathlib.Path(out_dir) / "_failed_tickers.txt"
    failed_path.write_text("".join(f"{ticker}\n" for ticker in failed_tickers))
    logger.info(
        f"Downloading done, success_num={success_num}, tickers_num={tickers_num}"
    )
    if failed_tickers:
        logger.warning(
            "Tickers failed, failed_num=%d, manifest=%s",
            len(failed_tickers),
            failed_path,
        )

    manifest = {
        "updated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "timeframe": timeframe,
        "adjustment": "split",
        "columns": list(BAR_COLUMNS[timeframe]),
        "dtype": "int64" if timeframe == "1Day" else "int32",
        "since": since.isoformat(),
        "tickers_path": str(pathlib.Path(tickers_path).resolve()),
        "ticker_count": tickers_num,
        "success_count": success_num,
        "failed_count": len(failed_tickers),
        "update_existing": update_existing,
        "overlap_days": overlap_days,
    }
    manifest_part = manifest_path.with_suffix(f"{manifest_path.suffix}.part")
    manifest_part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_part, manifest_path)

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
parser.add_argument(
    "--out_dir", type=str, required=True, help="Directory to save the dataset"
)
parser.add_argument("--days", type=int, default=None, help="Days to look back for data")
parser.add_argument(
    "--since", type=str, default=None, help="Date to look back for data"
)
parser.add_argument(
    "--workers_num", type=int, default=8, help="Number of workers to use"
)
parser.add_argument(
    "--timeframe",
    choices=("1Min", "1Day"),
    default="1Min",
    help="bar timeframe (default: 1Min)",
)
parser.add_argument("--archive", action="store_true", help="Should pack into tar file?")
parser.add_argument("--skip_existing", action="store_true", help="Skip existing files?")
parser.add_argument(
    "--update_existing",
    action="store_true",
    help=(
        "incrementally extend existing files; automatically redownload a ticker's "
        "full retained history if its overlap changed"
    ),
)
parser.add_argument(
    "--overlap_days",
    type=int,
    default=30,
    help="calendar days compared during --update_existing (default: 30)",
)
parser.add_argument("--rot", action="store_true", help="Apply rot13?")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.skip_existing and args.update_existing:
        parser.error("--skip_existing and --update_existing are mutually exclusive")
    if args.overlap_days < 1:
        parser.error("--overlap_days must be positive")
    if args.days is None and args.since is None and args.update_existing:
        existing_manifest_path = pathlib.Path(args.out_dir) / DATASET_MANIFEST
        if existing_manifest_path.exists():
            existing_manifest = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            args.since = str(existing_manifest.get("since", ""))[:10] or None
    if args.days is None and args.since is None:
        parser.error("either --days or --since is required")
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
        update_existing=args.update_existing,
        overlap_days=args.overlap_days,
        timeframe=args.timeframe,
        rot=args.rot,
    )
