"""Replay causal strategy baskets and export their historical symbol union."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ..market_data.calendar import auction_close_minutes
from ..overnight.history import (
    DEFAULT_AUCTIONS_PATH,
    DEFAULT_DAILY_DATA_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_SECURITY_MASTER_CACHE,
    REFERENCE_SYMBOL,
    _dataset_manifest,
    _parse_clock,
    _parse_day,
    _security_symbol,
    company_universe_mask,
    exchange_universe_mask,
    historical_window,
    load_daily_dollar_volume,
    load_nasdaq_security_master,
    load_primary_auction_exchange_mask,
    reference_session_calendar,
    simulation_symbols,
)
from ..overnight.ranking import build_issuer_map, replay_strategy_selections


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
        help="final exit date; defaults to the latest complete reference session",
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
        "--exchange-filter", choices=("all", "nasdaq"), default="nasdaq"
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
        "--minute-bars-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="candidate file inventory and SPY session calendar; stock minute prices are not read",
    )
    parser.add_argument(
        "--auctions-path",
        type=Path,
        default=Path(DEFAULT_AUCTIONS_PATH),
        help="official session closes and historical listing exchanges",
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
    _dataset_manifest(args.minute_bars_dir, "1Min")
    _dataset_manifest(args.daily_bars_dir, "1Day")
    known_closes = auction_close_minutes(args.auctions_path, None)
    all_dates, context_sod = reference_session_calendar(
        args.minute_bars_dir / f"{REFERENCE_SYMBOL}.npy",
        known_closes,
    )
    window = historical_window(
        all_dates,
        context_sod,
        args.auctions_path,
        since=args.since,
        end_date=args.end_date,
        months=args.months,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        min_trading_days=args.min_trading_days,
        entry_time=args.entry_time,
    )
    symbols = simulation_symbols(args.minute_bars_dir, args.daily_bars_dir)
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
    positions = {stamp: index for index, stamp in enumerate(window.dates)}

    def load_symbol(symbol: str) -> np.ndarray:
        return load_daily_dollar_volume(
            args.daily_bars_dir / f"{_security_symbol(symbol)}.npy",
            positions,
            len(window.dates),
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
    exchange_mask = None
    if args.exchange_filter != "all":
        exchange_mask, _ = load_primary_auction_exchange_mask(
            args.auctions_path,
            window.dates,
            symbols,
            args.exchange_filter,
        )
    selections = replay_strategy_selections(
        window.dates,
        symbols,
        dollar_volume,
        window.requested_start,
        window.end_date,
        top=args.top,
        ema_span=args.ema_span,
        min_history_days=args.min_history_days,
        minimum_trading_days=args.min_trading_days,
        liquidity_scheme=args.liquidity_scheme,
        issuers=build_issuer_map(symbols, master),
        dedupe_share_classes=args.dedupe_share_classes,
        execution_exchange_mask=exchange_mask,
        entry_session_mask=window.entry_session_mask,
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
        ("Shortened entries skipped", str(len(window.shortened_entries))),
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
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
