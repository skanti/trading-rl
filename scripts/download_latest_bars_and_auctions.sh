#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/overnight/.env}"
UPDATES_DIR="${UPDATES_DIR:-/data/ppv1/updates}"
BAR_SINCE="${BAR_SINCE:-2022-01-01}"
DAILY_BARS_DIR="${DAILY_BARS_DIR:-$UPDATES_DIR/bars_1day_$BAR_SINCE}"
MINUTE_BARS_DIR="${MINUTE_BARS_DIR:-$UPDATES_DIR/bars_1min_$BAR_SINCE}"
AUCTIONS_PATH="${AUCTIONS_PATH:-$UPDATES_DIR/alpaca_auctions_2022-01-01.npz}"
MASTER_PATH="${MASTER_PATH:-$REPO_DIR/data/master.txt}"
# Keep the mutable download universe beside the market-data stores, not in the
# tracked repository. MOST_LIQUID_PATH remains a compatibility override.
LIQUIDITY_CANDIDATES_PATH="${LIQUIDITY_CANDIDATES_PATH:-${MOST_LIQUID_PATH:-$UPDATES_DIR/liquidity_candidates.txt}}"
SHORTLIST_SINCE="${SHORTLIST_SINCE:-2022-01-01}"
SHORTLIST_DAILY_TOP="${SHORTLIST_DAILY_TOP:-50}"
SHORTLIST_LOOKBACK_SESSIONS="${SHORTLIST_LOOKBACK_SESSIONS:-250}"
AUCTION_START="${AUCTION_START:-2022-01-01}"
BAR_OVERLAP_DAYS="${BAR_OVERLAP_DAYS:-30}"
AUCTION_OVERLAP_DAYS="${AUCTION_OVERLAP_DAYS:-7}"
WORKERS="${WORKERS:-4}"
DAILY_BAR_BATCH_SIZE="${DAILY_BAR_BATCH_SIZE:-100}"
MINUTE_BAR_BATCH_SIZE="${MINUTE_BAR_BATCH_SIZE:-10}"
ALPACA_REQUESTS_PER_MINUTE="${ALPACA_REQUESTS_PER_MINUTE:-180}"
CALENDAR_SYMBOL="${CALENDAR_SYMBOL:-AAPL}"
LOCK_DIR="${LOCK_DIR:-/tmp/trading-rl-market-data-update.lock}"

usage() {
  cat <<'EOF'
Refresh the current eligible company-stock universe, update its split-adjusted
daily bars, rebuild the dollar-volume shortlist, update minute bars for the
shortlist plus SPY, and update auctions. Historical bar files are retained.

Usage:
  scripts/download_latest_bars_and_auctions.sh

Common environment overrides:
  PYTHON_BIN=/path/to/python
  ENV_FILE=/path/to/.env
  UPDATES_DIR=/data/ppv1/updates
  BAR_SINCE=2022-01-01
  DAILY_BARS_DIR=/data/ppv1/updates/bars_1day_2022-01-01
  MINUTE_BARS_DIR=/data/ppv1/updates/bars_1min_2022-01-01
  WORKERS=4
  DAILY_BAR_BATCH_SIZE=100
  MINUTE_BAR_BATCH_SIZE=10
  # Algo Trader Plus accounts may use 9000 (below Alpaca's documented 10000 RPM).
  ALPACA_REQUESTS_PER_MINUTE=180
  BAR_OVERLAP_DAYS=30
  AUCTION_OVERLAP_DAYS=7
  SHORTLIST_SINCE=2022-01-01
  SHORTLIST_DAILY_TOP=50
  SHORTLIST_LOOKBACK_SESSIONS=250
  LIQUIDITY_CANDIDATES_PATH=/data/ppv1/updates/liquidity_candidates.txt
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
  echo "Another market-data update appears to be running: $LOCK_DIR" >&2
  exit 1
fi

minute_symbols_tmp=""
cleanup() {
  if [[ -n "$minute_symbols_tmp" && -f "$minute_symbols_tmp" ]]; then
    rm -f -- "$minute_symbols_tmp"
  fi
  rmdir -- "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

check_failures() {
  local dataset_dir="$1"
  local failed_path="$dataset_dir/_failed_tickers.txt"
  if [[ -s "$failed_path" ]]; then
    echo "One or more downloads failed; see $failed_path" >&2
    sed 's/^/  /' "$failed_path" >&2
    exit 1
  fi
}

if [[ -f "$ENV_FILE" ]]; then
  log "Loading credentials from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
: "${ALPACA_DATA_KEY:?ALPACA_DATA_KEY must be set}"
: "${ALPACA_DATA_SECRET:?ALPACA_DATA_SECRET must be set}"
: "${ALPACA_KEY:?ALPACA_KEY must be set to refresh the asset master}"
: "${ALPACA_SECRET:?ALPACA_SECRET must be set to refresh the asset master}"

log "Refreshing the current eligible company-stock universe in $MASTER_PATH"
(
  cd -- "$REPO_DIR/overnight"
  "$PYTHON_BIN" universe.py \
    --output "$MASTER_PATH" \
    --refresh-security-master
)

log "Updating broad daily bars in $DAILY_BARS_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/download_bars.py" \
  --source alpaca \
  --timeframe 1Day \
  --tickers_path "$MASTER_PATH" \
  --out_dir "$DAILY_BARS_DIR" \
  --since "$BAR_SINCE" \
  --workers_num "$WORKERS" \
  --batch_size "$DAILY_BAR_BATCH_SIZE" \
  --requests_per_minute "$ALPACA_REQUESTS_PER_MINUTE" \
  --update_existing \
  --overlap_days "$BAR_OVERLAP_DAYS"
check_failures "$DAILY_BARS_DIR"

log "Rebuilding trailing-$SHORTLIST_LOOKBACK_SESSIONS-session top-$SHORTLIST_DAILY_TOP daily dollar-volume union"
"$PYTHON_BIN" "$SCRIPT_DIR/build_most_liquid.py" \
  --bars-dir "$DAILY_BARS_DIR" \
  --since "$SHORTLIST_SINCE" \
  --top "$SHORTLIST_DAILY_TOP" \
  --lookback-sessions "$SHORTLIST_LOOKBACK_SESSIONS" \
  --metric dollar-volume \
  --output "$LIQUIDITY_CANDIDATES_PATH"

# The minute universe is fully derived from the liquidity-prioritized shortlist,
# so it lives in a scratch file the exit trap removes. The durable record of what
# the store holds is _symbols.txt inside the store itself, written once the
# download succeeds.
minute_symbols_tmp="$(mktemp -t trading-rl-minute-symbols.XXXXXX)"
printf 'SPY\n' >"$minute_symbols_tmp"
while IFS= read -r symbol; do
  if [[ -n "$symbol" && "$symbol" != "SPY" ]]; then
    printf '%s\n' "$symbol" >>"$minute_symbols_tmp"
  fi
done <"$LIQUIDITY_CANDIDATES_PATH"

log "Updating liquidity-prioritized shortlist minute bars in $MINUTE_BARS_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/download_bars.py" \
  --source alpaca \
  --timeframe 1Min \
  --tickers_path "$minute_symbols_tmp" \
  --out_dir "$MINUTE_BARS_DIR" \
  --since "$BAR_SINCE" \
  --workers_num "$WORKERS" \
  --batch_size "$MINUTE_BAR_BATCH_SIZE" \
  --requests_per_minute "$ALPACA_REQUESTS_PER_MINUTE" \
  --update_existing \
  --overlap_days "$BAR_OVERLAP_DAYS"
check_failures "$MINUTE_BARS_DIR"

cp -- "$minute_symbols_tmp" "$MINUTE_BARS_DIR/_symbols.txt.part"
mv -- "$MINUTE_BARS_DIR/_symbols.txt.part" "$MINUTE_BARS_DIR/_symbols.txt"

calendar_path="$DAILY_BARS_DIR/$CALENDAR_SYMBOL.npy"
require_file "$calendar_path"
auction_end="$(
  "$PYTHON_BIN" - "$calendar_path" <<'PY'
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1])
bars = np.load(path, mmap_mode="r", allow_pickle=False)
if bars.ndim != 2 or bars.shape[1] < 1 or not len(bars):
    raise SystemExit(f"invalid daily calendar bars: {path}")
origin = datetime(2010, 1, 1, tzinfo=timezone.utc)
print((origin + timedelta(seconds=int(bars[-1, 0]))).date().isoformat())
PY
)"
if [[ ! "$auction_end" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Could not determine the latest completed daily-bar session: $auction_end" >&2
  exit 1
fi

log "Updating auctions through $auction_end"
if [[ -f "$AUCTIONS_PATH" && -f "${AUCTIONS_PATH%.npz}.json" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/download_auctions.py" \
    --update \
    --end "$auction_end" \
    --overlap-days "$AUCTION_OVERLAP_DAYS" \
    --symbols-file "$LIQUIDITY_CANDIDATES_PATH" \
    --output "$AUCTIONS_PATH"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/download_auctions.py" \
    --start "$AUCTION_START" \
    --end "$auction_end" \
    --symbols-file "$LIQUIDITY_CANDIDATES_PATH" \
    --output "$AUCTIONS_PATH"
fi

log "Market-data update complete through $auction_end"
