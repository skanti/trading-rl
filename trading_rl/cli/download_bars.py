import os
import argparse
import json
import pathlib
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
import tarfile
import logging
from datetime import datetime, timedelta, UTC
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from rich.logging import RichHandler

from ..market_data.bars import (
    BAR_COLUMNS,
    BAR_EPOCH,
    atomic_save_bar_array,
    download_alpaca_bars as download_alpaca_bar_pages,
    encode_alpaca_bars,
    merge_bar_arrays as merge_shared_bar_arrays,
    normalize_symbol,
)

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("DOWNLOAD_BARS")

ANNO = BAR_EPOCH
REQUEST_TIMEOUT = (10, 90)
REQUEST_RETRIES = 10
ALPACA_REQUESTS_PER_MINUTE = 180
ALPACA_BATCH_SIZE = 100
ALPACA_SIP_DELAY = timedelta(minutes=20)
DATASET_MANIFEST = "_download_manifest.json"


class DeferredDownloadSummary(logging.Filter):
    """Collapse repetitive per-ticker logs into a compact end-of-run summary."""

    DEFERRED_INFO_PREFIXES = (
        "Ticker exists already - skip",
        "Incremental update complete",
        "Full refresh complete",
    )

    def __init__(self, logger_name: str, example_limit: int = 5):
        super().__init__()
        self.logger_name = logger_name
        self.example_limit = example_limit
        self.counts: Counter[tuple[int, str]] = Counter()
        self.examples: dict[tuple[int, str], set[str]] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _category(message: str) -> str:
        if message.startswith("Alpaca batch failed; retrying"):
            return "Alpaca batch failed; split and retried"
        return message.split(",", 1)[0]

    @staticmethod
    def _example(message: str) -> str | None:
        for marker in ("ticker=", "name="):
            if marker in message:
                return message.split(marker, 1)[1].split(", ", 1)[0]
        return None

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != self.logger_name:
            return True
        message = record.getMessage()
        should_defer = record.levelno >= logging.WARNING or (
            record.levelno == logging.INFO
            and message.startswith(self.DEFERRED_INFO_PREFIXES)
        )
        if not should_defer:
            return True

        key = (record.levelno, self._category(message))
        example = self._example(message)
        with self.lock:
            self.counts[key] += 1
            if example:
                values = self.examples.setdefault(key, set())
                if len(values) < self.example_limit:
                    values.add(example)
        return False

    def write(self, destination: logging.Logger) -> None:
        with self.lock:
            counts = dict(self.counts)
            examples = {key: sorted(values) for key, values in self.examples.items()}

        if not counts:
            destination.info(
                "Download event summary: no warnings or per-ticker updates"
            )
            return

        info_count = sum(
            count
            for (level, _category), count in counts.items()
            if level < logging.WARNING
        )
        warning_count = sum(
            count
            for (level, _category), count in counts.items()
            if level == logging.WARNING
        )
        error_count = sum(
            count
            for (level, _category), count in counts.items()
            if level > logging.WARNING
        )
        destination.info(
            "Download event summary: updates=%d, warnings=%d, errors=%d",
            info_count,
            warning_count,
            error_count,
        )
        for (level, category), count in sorted(
            counts.items(), key=lambda item: (-item[0][0], item[0][1])
        ):
            sample = examples.get((level, category), [])
            example_text = f" (e.g. {', '.join(sample)})" if sample else ""
            destination.info(
                "  %s %s: %d%s",
                logging.getLevelName(level),
                category,
                count,
                example_text,
            )


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


def batches(values: list, size: int):
    """Yield bounded lists without requiring a second full copy of ``values``."""
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def dataframe_to_array(
    data: pd.DataFrame, name: str, timeframe: str = "1Min"
) -> np.ndarray | None:
    """Convert API bars through the shared compact on-disk encoder."""
    required_columns = {"t", "o", "v"}
    if timeframe == "1Day":
        required_columns.update({"h", "l", "c"})
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        logger.warning(
            "Columns missing, name=%s, columns=%s", name, sorted(missing_columns)
        )
        return None
    try:
        array = encode_alpaca_bars(data.to_dict("records"), name, timeframe)
    except ValueError as error:
        logger.warning("%s, name=%s", error, name)
        return None
    if len(array) == 0:
        logger.warning("No bars remain after timestamp filtering, name=%s", name)
        return None
    return array


def save_array(array: np.ndarray, out_path: str) -> None:
    """Atomically replace one ticker file using the shared storage primitive."""
    atomic_save_bar_array(pathlib.Path(out_path), array)


def save_as_npy(data: pd.DataFrame, out_path: str, timeframe: str = "1Min") -> bool:
    array = dataframe_to_array(data, os.path.basename(out_path), timeframe)
    if array is None:
        return False
    save_array(array, out_path)
    return True


def merge_bar_arrays(base: np.ndarray, update: np.ndarray) -> np.ndarray | None:
    """Merge through the shared exact-overlap implementation."""
    return merge_shared_bar_arrays(base, update)


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
    return normalize_symbol(ticker)


def clean_ticker(ticker: str) -> str:
    """Normalize a stored symbol for Alpaca's dot-separated API notation."""
    return storage_ticker(ticker)


def download_bars_alpaca_batch(
    tickers: list[str], since: datetime, timeframe: str = "1Min"
) -> dict[str, pd.DataFrame]:
    """Download one shared time range for several symbols.

    Alpaca's multi-symbol endpoint paginates across the combined result. Keeping
    the page loop here reduces request count without changing the per-symbol
    validation and atomic storage performed by ``process_ticker``.
    """
    cleaned = list(dict.fromkeys(clean_ticker(ticker) for ticker in tickers))
    if not cleaned:
        raise ValueError("at least one ticker is required")
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
    with requests.Session() as session:
        def request_page(params: dict[str, object]) -> dict[str, object]:
            request_label = ",".join(cleaned[:3])
            if len(cleaned) > 3:
                request_label += f",... ({len(cleaned)} symbols)"
            return request_json(session, base_url, params, headers, request_label)

        bars = download_alpaca_bar_pages(
            cleaned,
            dt_start,
            dt_end,
            timeframe,
            "sip",
            "split",
            request_page,
        )

    return {ticker: pd.DataFrame(rows) for ticker, rows in bars.items()}


def download_bars_alpaca(
    ticker: str, since: datetime, timeframe: str = "1Min"
) -> pd.DataFrame:
    """Retain the single-symbol interface for fallbacks and external callers."""
    symbol = clean_ticker(ticker)
    return download_bars_alpaca_batch([symbol], since, timeframe)[symbol]


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
    initial_df: pd.DataFrame | None = None,
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
            update_df = (
                initial_df
                if initial_df is not None
                else download_ticker(ticker, source, update_since, timeframe)
            )
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

        df = (
            initial_df
            if initial_df is not None
            else download_ticker(ticker, source, since, timeframe)
        )
        if len(df) == 0:
            logger.warning(f"No bars found, ticker={ticker}")
            return False
        return save_as_npy(data=df, out_path=out_path, timeframe=timeframe)
    except Exception:
        logger.exception("Ticker failed, ticker=%s", ticker)
        return False


def initial_request_since(
    ticker: str,
    out_dir: str,
    since: datetime,
    update_existing: bool,
    overlap_days: int,
) -> datetime:
    """Return the initial range needed for a ticker's incremental request."""
    out_path = pathlib.Path(out_dir) / f"{storage_ticker(ticker)}.npy"
    if not update_existing or not out_path.exists():
        return since
    try:
        base = np.load(out_path, mmap_mode="r")
        if base.ndim != 2 or len(base) == 0:
            return since
        base_start = ANNO + timedelta(seconds=int(base[0, 0]))
        base_last = ANNO + timedelta(seconds=int(base[-1, 0]))
        return max(base_start, base_last - timedelta(days=overlap_days))
    except (OSError, ValueError, IndexError):
        return since


def process_alpaca_batch(
    tasks: list[tuple[int, str, datetime]],
    out_dir: str,
    since: datetime,
    skip_existing: bool,
    update_existing: bool,
    overlap_days: int,
    timeframe: str,
) -> list[tuple[int, bool]]:
    """Download a symbol batch once, then apply normal per-symbol persistence.

    A rejected batch is bisected recursively. This preserves batching for healthy
    symbols while isolating an invalid or temporarily problematic symbol.
    """
    if not tasks:
        return []
    request_since = min(task[2] for task in tasks)
    try:
        frames = download_bars_alpaca_batch(
            [task[1] for task in tasks], request_since, timeframe
        )
    except Exception:
        if len(tasks) == 1:
            index, ticker, _ = tasks[0]
            logger.exception("Alpaca batch failed, ticker=%s", ticker)
            succeeded = process_ticker(
                ticker,
                source="alpaca",
                out_dir=out_dir,
                since=since,
                skip_existing=skip_existing,
                update_existing=update_existing,
                overlap_days=overlap_days,
                timeframe=timeframe,
            )
            return [(index, succeeded)]
        midpoint = len(tasks) // 2
        logger.warning(
            "Alpaca batch failed; retrying as groups of %d and %d symbols",
            midpoint,
            len(tasks) - midpoint,
        )
        return process_alpaca_batch(
            tasks[:midpoint],
            out_dir,
            since,
            skip_existing,
            update_existing,
            overlap_days,
            timeframe,
        ) + process_alpaca_batch(
            tasks[midpoint:],
            out_dir,
            since,
            skip_existing,
            update_existing,
            overlap_days,
            timeframe,
        )

    outcomes = []
    for index, ticker, _ in tasks:
        frame = frames.get(clean_ticker(ticker), pd.DataFrame())
        succeeded = process_ticker(
            ticker,
            source="alpaca",
            out_dir=out_dir,
            since=since,
            skip_existing=skip_existing,
            update_existing=update_existing,
            overlap_days=overlap_days,
            timeframe=timeframe,
            initial_df=frame,
        )
        outcomes.append((index, succeeded))
    return outcomes


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


def load_tickers(tickers_path: str) -> list[str]:
    """Load a symbol list from any existing file, including extensionless files."""
    path = pathlib.Path(tickers_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload["sample_ids"]
    elif path.is_file():
        values = path.read_text(encoding="utf-8").splitlines()
    elif path.suffix.lower() in {".csv", ".txt"} or os.sep in tickers_path:
        raise FileNotFoundError(f"ticker list does not exist: {tickers_path}")
    else:
        values = [tickers_path]
    tickers = [str(value).strip() for value in values if str(value).strip()]
    if not tickers:
        raise ValueError(f"ticker list is empty: {tickers_path}")
    return tickers


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
    batch_size: int = ALPACA_BATCH_SIZE,
    requests_per_minute: int = ALPACA_REQUESTS_PER_MINUTE,
) -> None:
    global ALPACA_RATE_LIMITER
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if requests_per_minute < 1:
        raise ValueError("requests per minute must be positive")
    ALPACA_RATE_LIMITER = RateLimiter(requests_per_minute)
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

    tickers = load_tickers(tickers_path)

    if rot:
        tickers = [rot13(ticker) for ticker in tickers]
    tickers_num = len(tickers)
    assert tickers_num > 0, "No tickers found"
    logger.info(f"Tickers provided, tickers_num={tickers_num}")
    if source == "alpaca":
        logger.info(
            "Alpaca download settings, timeframe=%s, batch_size=%d, workers=%d, "
            "requests_per_minute=%d",
            timeframe,
            batch_size,
            workers_num,
            requests_per_minute,
        )

    deferred_summary = DeferredDownloadSummary(logger.name)
    logger.addFilter(deferred_summary)
    if source == "alpaca" and batch_size > 1 and not skip_existing:
        tasks = [
            (
                index,
                ticker,
                initial_request_since(
                    ticker, out_dir, since, update_existing, overlap_days
                ),
            )
            for index, ticker in enumerate(tickers)
        ]
        # Similar start dates share batches so one stale/delisted symbol does not
        # expand the requested history for a batch of current symbols.
        tasks.sort(key=lambda task: task[2])
        task_batches = list(batches(tasks, batch_size))
        batch_fn = partial(
            process_alpaca_batch,
            out_dir=out_dir,
            since=since,
            skip_existing=skip_existing,
            update_existing=update_existing,
            overlap_days=overlap_days,
            timeframe=timeframe,
        )
        outcomes: list[tuple[int, bool]] = []
        if workers_num == 0:
            iterator = map(batch_fn, task_batches)
            with tqdm(total=tickers_num, desc="Processing tickers") as progress:
                for batch_outcomes in iterator:
                    outcomes.extend(batch_outcomes)
                    progress.update(len(batch_outcomes))
        else:
            with (
                ThreadPoolExecutor(max_workers=workers_num) as pool,
                tqdm(total=tickers_num, desc="Processing tickers") as progress,
            ):
                for batch_outcomes in pool.map(batch_fn, task_batches):
                    outcomes.extend(batch_outcomes)
                    progress.update(len(batch_outcomes))
        res = [False] * tickers_num
        for index, succeeded in outcomes:
            res[index] = succeeded
    else:
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
            res = list(
                tqdm(map(fn, tickers), total=tickers_num, desc="Processing tickers")
            )
        else:
            with ThreadPoolExecutor(max_workers=workers_num) as pool:
                res = list(
                    tqdm(
                        pool.map(fn, tickers),
                        total=tickers_num,
                        desc="Processing tickers",
                    )
                )
    t1 = time.perf_counter()
    logger.removeFilter(deferred_summary)

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
        "batch_size": batch_size if source == "alpaca" else 1,
        "requests_per_minute": requests_per_minute if source == "alpaca" else None,
    }
    manifest_part = manifest_path.with_suffix(f"{manifest_path.suffix}.part")
    manifest_part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_part, manifest_path)

    duration = t1 - t0
    if archive:
        pack_to_archive(out_dir)
    deferred_summary.write(logger)
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
parser.add_argument(
    "--batch_size",
    type=int,
    default=int(os.environ.get("ALPACA_BAR_BATCH_SIZE", ALPACA_BATCH_SIZE)),
    help="symbols per Alpaca request (default: 100; env: ALPACA_BAR_BATCH_SIZE)",
)
parser.add_argument(
    "--requests_per_minute",
    type=int,
    default=int(
        os.environ.get("ALPACA_REQUESTS_PER_MINUTE", ALPACA_REQUESTS_PER_MINUTE)
    ),
    help=(
        "client-side Alpaca request limit (default: 180; env: "
        "ALPACA_REQUESTS_PER_MINUTE)"
    ),
)
parser.add_argument("--rot", action="store_true", help="Apply rot13?")

def cli() -> None:
    """Parse command-line arguments and run the bar downloader."""
    args = parser.parse_args()
    if args.skip_existing and args.update_existing:
        parser.error("--skip_existing and --update_existing are mutually exclusive")
    if args.overlap_days < 1:
        parser.error("--overlap_days must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.requests_per_minute < 1:
        parser.error("--requests_per_minute must be positive")
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
        batch_size=args.batch_size,
        requests_per_minute=args.requests_per_minute,
    )


if __name__ == "__main__":
    cli()
