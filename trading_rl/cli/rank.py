"""Replay causal strategy baskets and export their historical symbol union."""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ..market_data.calendar import short_entry_dates
from ..overnight.history import (
    DEFAULT_DAILY_DATA_DIR,
    DEFAULT_SECURITY_MASTER_CACHE,
    REFERENCE_SYMBOL,
    _dataset_manifest,
    _parse_clock,
    _parse_day,
    _security_symbol,
    company_universe_mask,
    exchange_universe_mask,
    historical_window_bounds,
    load_daily_dollar_volume,
    load_nasdaq_security_master,
)
from ..overnight.ranking import build_issuer_map, replay_strategy_selections
from ..overnight.ranking_inputs import daily_ranking_calendar, daily_ranking_symbols


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    period = parser.add_mutually_exclusive_group()
    period.add_argument(
        "--since", type=_parse_day, help="first entry date (YYYY-MM-DD)"
    )
    period.add_argument(
        "--months",
        type=int,
        help="trailing calendar months (default: 12 when --since is omitted)",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_day,
        help="final exit date; defaults to the latest completed daily-bar session",
    )
    parser.add_argument(
        "--top", type=int, default=12, help="stocks per entry session (default: 12)"
    )
    parser.add_argument(
        "--liquidity-scheme",
        choices=("dollar_ema", "turnover_stability"),
        default="turnover_stability",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=10,
        help="liquidity EMA and turnover-stability dispersion span",
    )
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=100,
        help="minimum completed observed sessions before selection",
    )
    parser.add_argument(
        "--entry-time",
        type=_parse_clock,
        default=_parse_clock("15:45"),
        help="Eastern entry clock for excluding shortened sessions (default: 15:45)",
    )
    parser.add_argument(
        "--no-dedupe-share-classes", dest="dedupe_share_classes", action="store_false"
    )
    parser.add_argument(
        "--asset-filter", choices=("companies", "all"), default="companies"
    )
    parser.add_argument(
        "--exchange-filter",
        choices=("all", "nasdaq"),
        default="nasdaq",
        help="filter using the current security master (historical listing transfers are not applied)",
    )
    parser.add_argument(
        "--unclassified-asset-policy", choices=("keep", "exclude"), default="keep"
    )
    parser.add_argument(
        "--daily-bars-dir",
        type=Path,
        default=Path(DEFAULT_DAILY_DATA_DIR),
        help="split-adjusted daily bars used to rank completed-session liquidity",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="optional candidate shortlist; defaults to every symbol in the daily-bar store",
    )
    parser.add_argument(
        "--calendar-path",
        type=Path,
        help="saved calendar JSON with start, end, and sessions; defaults to Alpaca's calendar API",
    )
    parser.add_argument(
        "--security-master-cache",
        type=Path,
        default=Path(DEFAULT_SECURITY_MASTER_CACHE),
    )
    parser.add_argument("--security-master-max-age-days", type=int, default=7)
    parser.add_argument("--refresh-security-master", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/strategy_symbols.txt"),
        help="unique symbols, one per line (default: /tmp/strategy_symbols.txt)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="daily selection schedule; defaults to --output with a .csv suffix",
    )
    return parser


def write_strategy_export(
    selections: pd.DataFrame,
    symbols_path: Path,
    csv_path: Path,
) -> list[str]:
    """Write unique strategy symbols and an NBBO-compatible daily schedule."""
    if symbols_path.resolve() == csv_path.resolve():
        raise ValueError("symbol-list and selection-CSV output paths must differ")
    if selections.empty:
        raise ValueError("no strategy selections to export")
    unique_symbols = sorted(selections["sample_id"].unique())
    for path, contents in (
        (symbols_path, "".join(f"{symbol}\n" for symbol in unique_symbols)),
        (csv_path, selections.to_csv(index=False)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(contents)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    return unique_symbols


def export_rankings(args: argparse.Namespace, console: Console) -> None:
    csv_path = args.output_csv or args.output.with_suffix(".csv")
    if args.output.resolve() == csv_path.resolve():
        raise ValueError("symbol-list and selection-CSV output paths must differ")
    _dataset_manifest(args.daily_bars_dir, "1Day")
    symbols = daily_ranking_symbols(args.daily_bars_dir, args.symbols_file)
    master = {}
    if args.asset_filter == "companies" or args.exchange_filter != "all":
        master = load_nasdaq_security_master(
            args.security_master_cache,
            refresh=args.refresh_security_master,
            max_age_days=args.security_master_max_age_days,
        )
    if args.asset_filter == "companies":
        mask, _, _ = company_universe_mask(
            symbols,
            master,
            keep_unclassified=args.unclassified_asset_policy == "keep",
        )
        symbols = symbols[mask]
    if args.exchange_filter != "all":
        symbols = symbols[exchange_universe_mask(symbols, master, args.exchange_filter)]
    if not np.any(symbols != REFERENCE_SYMBOL):
        raise ValueError(
            "no tradable daily-bar candidates remain after universe filters"
        )
    all_dates, closes = daily_ranking_calendar(
        args.daily_bars_dir,
        symbols,
        end_date=args.end_date,
        calendar_path=args.calendar_path,
    )
    requested_start, final_exit, bounds = historical_window_bounds(
        all_dates,
        since=args.since,
        end_date=args.end_date,
        months=args.months,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        min_trading_days=args.min_trading_days,
    )
    dates = all_dates[bounds]
    shortened = short_entry_dates(
        {
            stamp.date(): closes[stamp.date()]
            for stamp in dates
            if requested_start <= stamp < final_exit
        },
        args.entry_time,
    )
    positions = {stamp: index for index, stamp in enumerate(dates)}

    def load_symbol(symbol: str) -> np.ndarray:
        return load_daily_dollar_volume(
            args.daily_bars_dir / f"{_security_symbol(symbol)}.npy",
            positions,
            len(dates),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        columns = list(
            track(
                executor.map(load_symbol, symbols),
                total=len(symbols),
                description="Loading daily liquidity",
                console=console,
            )
        )
    dollar_volume = np.column_stack(columns)
    selections = replay_strategy_selections(
        dates,
        symbols,
        dollar_volume,
        requested_start,
        final_exit,
        top=args.top,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        minimum_trading_days=args.min_trading_days,
        liquidity_scheme=args.liquidity_scheme,
        issuers=build_issuer_map(symbols, master),
        dedupe_share_classes=args.dedupe_share_classes,
        entry_session_mask=np.asarray(
            [stamp.date() not in shortened for stamp in dates]
        ),
    )
    unique = write_strategy_export(selections, args.output, csv_path)
    table = Table(title="Strategy ranking export")
    table.add_column("Result")
    table.add_column("Value", justify="right")
    for label, value in (
        ("Strategy", args.liquidity_scheme),
        ("Top per session", str(args.top)),
        ("Candidate stocks", str(int(np.count_nonzero(symbols != REFERENCE_SYMBOL)))),
        ("Entry sessions", f"{selections.entry_date.nunique():,}"),
        ("Daily selections", f"{len(selections):,}"),
        ("Unique symbols", f"{len(unique):,}"),
        ("Exchange filter", args.exchange_filter),
        (
            "Exchange metadata",
            "Current security master"
            if args.exchange_filter != "all"
            else "Not filtered",
        ),
        ("Shortened entries skipped", str(len(shortened))),
        ("First entry", selections.entry_date.min()),
        ("Last entry", selections.entry_date.max()),
        ("Last exit", selections.exit_date.max()),
    ):
        table.add_row(label, value)
    console.print(table)
    console.print(f"Symbols: {args.output}", markup=False)
    console.print(f"Daily selections: {csv_path}", markup=False)
    console.print(
        "Exported intended selections before execution skips or position sizing. SPY is added by the NBBO downloader."
    )
    if args.exchange_filter != "all":
        console.print(
            "Historical listing transfers are not applied; backtest auction eligibility may produce different baskets."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "months",
        "top",
        "ema_span",
        "min_history_days",
        "min_trading_days",
        "security_master_max_age_days",
        "workers",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.liquidity_scheme == "turnover_stability" and args.ema_span < 2:
        parser.error("--ema-span must be at least 2 for turnover stability")
    try:
        export_rankings(args, Console())
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
