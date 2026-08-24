"""Export Alpaca's currently tradable company-stock universe."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import tempfile

from baseline.live_overnight_liquidity import (
    DEFAULT_EXCHANGES,
    PAPER_TRADING_URL,
    AlpacaClient,
    eligible_assets,
    load_credentials,
)
from baseline.overnight_liquidity import (
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
    return parser


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
    print(f"exchanges: {breakdown}")


if __name__ == "__main__":
    main()
