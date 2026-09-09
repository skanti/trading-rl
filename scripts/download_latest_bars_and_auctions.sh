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
NBBO_PATH="${NBBO_PATH:-$UPDATES_DIR/alpaca_nbbo_1545_2022-01-01.npz}"
NBBO_TARGETS_PATH="${NBBO_TARGETS_PATH:-}"
NBBO_RANK_SINCE="${NBBO_RANK_SINCE:-2023-01-01}"
NBBO_RANK_TOP="${NBBO_RANK_TOP:-12}"
NBBO_SYMBOLS_PATH="${NBBO_SYMBOLS_PATH:-$UPDATES_DIR/strategy_symbols_$NBBO_RANK_SINCE.txt}"
MASTER_PATH="${MASTER_PATH:-$REPO_DIR/data/master.txt}"
# Keep the mutable download universe beside the market-data stores, not in the
# tracked repository. MOST_LIQUID_PATH remains a compatibility override.
LIQUIDITY_CANDIDATES_PATH="${LIQUIDITY_CANDIDATES_PATH:-${MOST_LIQUID_PATH:-$UPDATES_DIR/liquidity_candidates.txt}}"
SHORTLIST_SINCE="${SHORTLIST_SINCE:-2022-01-01}"
SHORTLIST_DAILY_TOP="${SHORTLIST_DAILY_TOP:-20}"
# Historical downloads must retain former liquidity leaders as well as current ones.
# Zero unions every daily top-N list since SHORTLIST_SINCE.
SHORTLIST_LOOKBACK_SESSIONS="${SHORTLIST_LOOKBACK_SESSIONS:-0}"
AUCTION_START="${AUCTION_START:-2022-01-01}"
NBBO_START="${NBBO_START:-2022-01-01}"
BAR_OVERLAP_DAYS="${BAR_OVERLAP_DAYS:-30}"
AUCTION_OVERLAP_DAYS="${AUCTION_OVERLAP_DAYS:-7}"
NBBO_OVERLAP_DAYS="${NBBO_OVERLAP_DAYS:-7}"
NBBO_TARGET_TIME="${NBBO_TARGET_TIME:-15:45}"
WORKERS="${WORKERS:-8}"
DAILY_BAR_BATCH_SIZE="${DAILY_BAR_BATCH_SIZE:-100}"
MINUTE_BAR_BATCH_SIZE="${MINUTE_BAR_BATCH_SIZE:-1}"
ALPACA_REQUESTS_PER_MINUTE="${ALPACA_REQUESTS_PER_MINUTE:-180}"
LOCK_DIR="${LOCK_DIR:-/tmp/trading-rl-market-data-update.lock}"

usage() {
  cat <<'EOF'
Refresh the current eligible company-stock universe, update its split-adjusted
daily bars, rebuild the dollar-volume shortlist, update minute bars for that
shortlist plus SPY, and replay the strategy ranker. Then update auctions and
scheduled 15:45 NBBO for the ranker's symbol union plus SPY. No simulator run is
required. NBBO_TARGETS_PATH optionally supplies a trade CSV instead of ranking.
The auction/minute shortlist includes every daily top-20 symbol since 2022-01-01.
The NBBO shortlist replays strategy top-12 selections since 2023-01-01 by default.
Historical bar files are retained; new auction symbols are backfilled to the
existing dataset's start date.

Usage:
  scripts/download_latest_bars_and_auctions.sh

Common environment overrides:
  PYTHON_BIN=/path/to/python
  ENV_FILE=/path/to/.env
  UPDATES_DIR=/data/ppv1/updates
  BAR_SINCE=2022-01-01
  DAILY_BARS_DIR=/data/ppv1/updates/bars_1day_2022-01-01
  MINUTE_BARS_DIR=/data/ppv1/updates/bars_1min_2022-01-01
  WORKERS=8
  DAILY_BAR_BATCH_SIZE=100
  MINUTE_BAR_BATCH_SIZE=1
  # Algo Trader Plus accounts may use 9000 (below Alpaca's documented 10000 RPM).
  ALPACA_REQUESTS_PER_MINUTE=180
  BAR_OVERLAP_DAYS=30
  AUCTION_OVERLAP_DAYS=7
  NBBO_OVERLAP_DAYS=7
  NBBO_TARGET_TIME=15:45
  NBBO_TARGETS_PATH=/tmp/overnight_trades.csv
  NBBO_RANK_SINCE=2023-01-01
  NBBO_RANK_TOP=12
  NBBO_SYMBOLS_PATH=/data/ppv1/updates/strategy_symbols_2023-01-01.txt
  SHORTLIST_SINCE=2022-01-01
  SHORTLIST_DAILY_TOP=20
  SHORTLIST_LOOKBACK_SESSIONS=0  # all sessions since SHORTLIST_SINCE
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

if [[ "$SHORTLIST_LOOKBACK_SESSIONS" == "0" ]]; then
  log "Rebuilding top-$SHORTLIST_DAILY_TOP daily dollar-volume union over all sessions since $SHORTLIST_SINCE"
else
  log "Rebuilding trailing-$SHORTLIST_LOOKBACK_SESSIONS-session top-$SHORTLIST_DAILY_TOP daily dollar-volume union since $SHORTLIST_SINCE"
fi
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

log "Updating liquidity-prioritized shortlist minute bars through Alpaca's delayed SIP cutoff in $MINUTE_BARS_DIR"
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

if [[ -n "$NBBO_TARGETS_PATH" ]]; then
  if [[ ! -f "$NBBO_TARGETS_PATH" ]]; then
    echo "NBBO target trade CSV does not exist: $NBBO_TARGETS_PATH" >&2
    exit 1
  fi
  nbbo_target_args=(--targets-from-trades "$NBBO_TARGETS_PATH")
else
  log "Replaying strategy top-$NBBO_RANK_TOP selections since $NBBO_RANK_SINCE for the NBBO shortlist"
  "$PYTHON_BIN" -m trading_rl.cli.rank \
    --since "$NBBO_RANK_SINCE" \
    --top "$NBBO_RANK_TOP" \
    --daily-bars-dir "$DAILY_BARS_DIR" \
    --minute-bars-dir "$MINUTE_BARS_DIR" \
    --auctions-path "$AUCTIONS_PATH" \
    --output "$NBBO_SYMBOLS_PATH"
  nbbo_target_args=(--symbols-file "$NBBO_SYMBOLS_PATH")
fi

# Daily bars intentionally exclude the unfinished session, but today's opening
# auction becomes usable after the SIP delay. Do not derive this bound from the
# daily cache or same-day reconciliation will remain one session behind.
auction_end="$(
  "$PYTHON_BIN" - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo

print(datetime.now(ZoneInfo("America/New_York")).date().isoformat())
PY
)"
if [[ ! "$auction_end" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Could not determine the current New York date: $auction_end" >&2
  exit 1
fi

log "Updating auctions through the current New York date $auction_end"
"$PYTHON_BIN" "$SCRIPT_DIR/download_auctions.py" \
  --start "$AUCTION_START" \
  --refresh-requested-only \
  --end "$auction_end" \
  --overlap-days "$AUCTION_OVERLAP_DAYS" \
  --symbols-file "$LIQUIDITY_CANDIDATES_PATH" \
  --output "$AUCTIONS_PATH"

log "Updating scheduled $NBBO_TARGET_TIME ET SIP NBBO through $auction_end"
"$PYTHON_BIN" "$SCRIPT_DIR/download_nbbo.py" \
  --start "$NBBO_START" \
  --end "$auction_end" \
  --target-time "$NBBO_TARGET_TIME" \
  --overlap-days "$NBBO_OVERLAP_DAYS" \
  "${nbbo_target_args[@]}" \
  --auctions-path "$AUCTIONS_PATH" \
  --requests-per-minute "$ALPACA_REQUESTS_PER_MINUTE" \
  --output "$NBBO_PATH"

log "Market-data update complete through $auction_end"
