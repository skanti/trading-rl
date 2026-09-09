"""Export Alpaca's currently tradable company-stock universe."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import tempfile

from .live import (
    DEFAULT_EXCHANGES,
    PAPER_TRADING_URL,
    AlpacaClient,
    eligible_assets,
    load_credentials,
)
from .history import (
    DEFAULT_SECURITY_MASTER_CACHE,
    load_nasdaq_security_master,
)


DEFAULT_OUTPUT = str(Path(__file__).resolve().parents[2] / "data/nasdaq.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write Alpaca's active, tradable, fractionable company-stock universe, "
            "excluding ETFs and other non-company securities."
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--exchanges", default=",".join(sorted(DEFAULT_EXCHANGES)))
    parser.add_argument("--trading-url", default=os.environ.get("ALPACA_URL", PAPER_TRADING_URL))
    parser.add_argument("--security-master-cache", default=DEFAULT_SECURITY_MASTER_CACHE)
    parser.add_argument("--refresh-security-master", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="add the current universe to an existing output without removing historical symbols",
    )
    return parser


def merge_existing_symbols(
    current_symbols: list[str], output: Path, enabled: bool
) -> tuple[list[str], int]:
    current = set(current_symbols)
    if not enabled or not output.exists():
        return sorted(current), len(current)
    existing = {
        line.strip().upper()
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(existing | current), len(current - existing)


def main() -> None:
    args = build_parser().parse_args()
    exchanges = frozenset(
        exchange.strip().upper() for exchange in args.exchanges.split(",") if exchange.strip()
    )
    if not exchanges:
        raise ValueError("exchanges cannot be empty")

    key, secret = load_credentials()
    client = AlpacaClient(key, secret, trading_url=args.trading_url)
    assets = client.list_assets()
    security_master = load_nasdaq_security_master(
        Path(args.security_master_cache), refresh=args.refresh_security_master
    )
    symbols = eligible_assets(assets, exchanges, security_master)
    if not symbols:
        raise RuntimeError("Alpaca returned no eligible company stocks")

    selected = set(symbols)
    exchange_counts = Counter(
        str(asset.get("exchange", "UNKNOWN")).upper()
        for asset in assets
        if str(asset.get("symbol", "")).upper() in selected
    )
    output = Path(args.output)
    symbols, added = merge_existing_symbols(symbols, output, args.merge_existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write("\n".join(symbols) + "\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output)

    breakdown = ", ".join(
        f"{exchange}={count:,}" for exchange, count in sorted(exchange_counts.items())
    )
    print(f"wrote {len(symbols):,} symbols to {output.resolve()}")
    if args.merge_existing:
        print(f"added {added:,} newly available symbols; retained historical symbols")
    print(f"exchanges: {breakdown}")


if __name__ == "__main__":
    main()
