# Trading baselines

`overnight_liquidity.py` implements a point-in-time overnight baseline:

1. Read each stock's completed daily dollar volume as split-adjusted daily
   `VWAP * volume` (falling back to the daily close when VWAP is unavailable).
2. Smooth `log1p(dollar_volume)` with a causal EMA.
3. Before each 15:55 entry, rank using the EMA state through the previous
   session only. The current session never contributes to its own rank.
4. Equal-weight the selected stocks, then close them at 09:45 on the next
   trading session.

Transaction costs default to 1 basis point per side, or 2 basis points for a
complete entry-and-exit round trip. Override this with `--transaction-cost-bps`.
The potentially long per-symbol trade-frequency table is hidden by default;
include `--show-symbol-trade-frequency` when that breakdown is needed.

Every simulated stock must have at least 100 completed observed trading
sessions before it can be selected. The current session is not counted. Change
this causal listing-history filter with `--minimum-trading-days`; the default
excludes recent IPOs such as SPCX until they establish 100 sessions.

The log transform and default 20-session EMA keep earnings, index-rebalance,
and news-related volume spikes from dominating the ranking. Use `--ema-span 1`
to rank strictly by the previous completed session without smoothing.

The first run combines split-adjusted daily bars from
`/data/ppv1/updates/bars_1day_2016-01-01` with execution prices and intraday
activity from `/data/ppv1/updates/bars_1min_2016-01-01`, then writes a
date-by-symbol cache under `/tmp/trading/baseline_cache`. The daily bars drive
the causal liquidity ranking; minute bars are used only for entry/exit prices
and the optional Alpaca-style same-session activity schemes. Override the two
stores with `--daily-bars-dir` and `--data-dir`, respectively. Subsequent runs
with the same inputs and date range reuse the cache.

The simulator does not require a generated day-index CSV. It derives complete
New York sessions from `SPY.npy`, discovers the tradable universe from symbols
present in both bar stores, and locates activity and execution timestamps
directly in each minute array. Updating the bar directories therefore makes new
symbols and sessions available without rebuilding separate metadata.

To compare the original stable dollar-liquidity ranking with Alpaca's
real-time most-actives definitions in one run:

```bash
python -m baseline.overnight_liquidity \
  --top 10 \
  --months 12 \
  --ranking-time 15:15 \
  --entry-time 15:55 \
  --exit-time 09:35 \
  --liquidity-scheme compare \
  --transaction-cost-bps 1
```

The comparison includes four schemes on identical entry and exit prices:

- `dollar_ema`: lagged EMA of `log1p` completed-session dollar volume.
- `activity_union_ema`: each day, union the top 100 current-session symbols by
  share volume and trade count, remove non-company assets, then rerank the
  remaining candidates using their recent lagged dollar-volume EMA. The union
  is rebuilt from scratch daily with no prior-day carry-forward.
- `alpaca_volume`: current-session cumulative SIP share volume.
- `alpaca_trades`: current-session cumulative SIP trade count.

The two Alpaca modes match the ranking fields used by Alpaca's most-actives
screener. They include only complete one-minute bars strictly before
`--ranking-time`; the ranking-minute bar itself is excluded. The table reports
membership retention, Jaccard similarity, replacements, and unique symbols in
addition to return, profit factor, drawdown, volatility, and Sharpe. Higher
retention/Jaccard and lower replacement counts indicate a more stable universe.
Use one scheme name instead of `compare` to run it alone.

For the fast dynamic approximation directly:

```bash
python -m baseline.overnight_liquidity \
  --liquidity-scheme activity_union_ema \
  --activity-candidates 100 \
  --minimum-trading-days 100 \
  --top 10 \
  --months 12 \
  --ranking-time 15:15 \
  --entry-time 15:55 \
  --exit-time 09:35 \
  --transaction-cost-bps 1
```

By default, the simulator restricts the tradable universe to company equity. It
uses Nasdaq's current symbol directory to remove positively identified ETFs and
ETNs, funds and closed-end funds, BDCs, units, preferreds, rights and warrants,
debt instruments, royalty trusts, and acquisition-company/SPAC shells. SPY is
retained only as the benchmark; ADRs and common REIT equity remain eligible.

Historical symbols absent from today's directory are retained by default. An
absence may mean that a real company was acquired or delisted, so discarding it
would introduce survivorship bias. Use `--unclassified-asset-policy exclude`
for the strict current-directory interpretation, or `--asset-filter all` to
restore the unfiltered legacy universe. The downloaded security master is
cached for seven days at
`/tmp/trading/baseline_cache/nasdaq_security_master.json`.

Entry prices must have a print within 10 minutes of 15:55. A selected stock is
never removed using knowledge of whether it trades the following morning. If
no fresh print exists at 09:45, the backtest uses the latest causal mark up to
24 hours old and reports its staleness in both the trade CSV and summary. This
is preferable to silently replacing the stock with next-morning lookahead, but
such a stale mark is not evidence that a live order could have filled at 09:45.

### Incremental minute-bar updates

Existing `.npy` bar files can be extended without redownloading every ticker's full
history. Filenames use plain symbols such as `AAPL.npy` and `BRK.B.npy`; legacy
asset-class prefixes such as `ST-` are stripped when old input lists are encountered:

To refresh the broad daily store, rebuild the dollar-volume shortlist, update
the shortlist's minute bars, and extend auction data in one pass, run:

```bash
../scripts/download_latest_bars_and_auctions.sh
```

The script first merges Alpaca's currently available company stocks into
`data/master.txt`; it never removes historical or delisted symbols. It then loads
`ml/.env` by default and accepts environment overrides such as `PYTHON_BIN`,
`UPDATES_DIR`, `WORKERS`, and the overlap/shortlist settings shown by `--help`.
It obtains the auction end date from the refreshed daily bars, so a still-forming
market session is not requested.

```bash
.venv/bin/python ../scripts/download_bars.py \
  --source alpaca \
  --tickers_path /path/to/tickers.txt \
  --out_dir /data/ppv1/updates/bars \
  --since 2022-08-27 \
  --update_existing
```

For an existing ticker, the downloader fetches a 30-calendar-day overlap, compares
timestamps, split-adjusted open prices, volumes, and trade counts exactly, and appends
only when the overlap matches. Any difference—including the historical rewrite caused
by a new split—automatically escalates that ticker to a full retained-history refresh.
Other tickers remain incremental. `--since` supplies the start for new ticker files;
an existing file supplies its own retained start. Use `--overlap_days` to widen the
verification window.

Use `--timeframe 1Day` for complete daily bars. Daily files retain timestamp,
OHLC prices, volume, trade count, and VWAP; the exact column order is recorded in
`_download_manifest.json`. They use `int64` so extreme split-adjusted histories do
not overflow. The downloader excludes a still-forming New York session.
Future updates can reuse the manifest's start date:

```bash
.venv/bin/python ../scripts/download_bars.py \
  --source alpaca \
  --timeframe 1Day \
  --tickers_path /home/aavetisyan/dev/trading-rl/data/master.txt \
  --out_dir /data/ppv1/updates/bars_1day_2016-01-01 \
  --update_existing
```

### Opening-auction exit prices

For an OPG comparison, download Alpaca's SIP auction prints and ask the simulator
to replace the 09:30 minute-bar exit with the primary exchange's official opening
auction. The downloader retains both opening and closing records, but the simulator
selects the largest opening record with condition `O` for each symbol-date. Alpaca
can return smaller alternate-venue auctions and duplicate tape records alongside the
primary listing auction. The columnar NPZ stores consumer-facing prices and sizes
already adjusted with Alpaca's complete forward/reverse split ledger. It also embeds
the raw fields and ledger so every update can recompute all adjustments; the simulator
does not apply splits itself.

```bash
set -a
source .env
set +a
.venv/bin/python ../scripts/download_auctions.py \
  --start 2022-01-01 \
  --end 2026-08-27 \
  --symbols-from-trades /tmp/overnight_liquidity_48m_minute.csv \
  --output /data/ppv1/updates/alpaca_auctions_2022-01-01.npz
```

For later updates, reuse the same output and provide only a new end date:

```bash
.venv/bin/python ../scripts/download_auctions.py \
  --update \
  --end 2026-09-04 \
  --output /data/ppv1/updates/alpaca_auctions_2022-01-01.npz
```

The JSON manifest supplies the original start date and symbol universe. Update mode
redownloads and replaces only a seven-calendar-day overlap plus the new tail, which
captures late corrections without downloading years again. Split actions are small,
so their full retained history is refreshed and reapplied to every raw auction record
before the NPZ is atomically replaced. Pass `--overlap-days` to change the
overlap, or `--symbols`, `--symbols-file`, or `--symbols-from-trades` to add
symbols; only a newly added symbol receives a full-history download. Use
`--symbols-file data/most_liquid.txt` to align the auction universe with the
complete liquidity shortlist. Use the last completed trading date for `--end`
when the Alpaca plan does not permit querying the most recent SIP data.

Then run the comparison:

```bash
python -m baseline.overnight_liquidity \
  --top 12 \
  --months 24 \
  --budget 10000 \
  --share-mode fractional \
  --entry-time 15:59 \
  --exit-time 09:30 \
  --exit-price-source opening-auction \
  --transaction-cost-bps 1
```

There is no 15:59 auction. The entry therefore remains the 15:59 SIP minute-bar
open. Substituting the 16:00 closing auction would model a different entry time.

Use `--exchange-filter nasdaq` to remove non-Nasdaq candidates before ranking.
The top basket is then reranked from the remaining Nasdaq company universe; the
filter does not merely discard NYSE names after selection. SPY remains available
only as the benchmark. When the auction file is present, historical primary-auction
venues override the current directory for known symbol-dates, so a listing transfer
such as PLTR's 2024 NYSE-to-Nasdaq move is handled at the correct session.

```bash
python -m baseline.overnight_liquidity \
  --top 100 \
  --months 12 \
  --ema-span 20 \
  --min-history-days 20 \
  --transaction-cost-bps 1 \
  --output-csv /tmp/trading/liquidity_top100_overnight.csv \
  --summary-json /tmp/trading/liquidity_top100_overnight_summary.json
```

Change only `--top 50` to run the top-50 basket. The liquidity cache is shared.

### Fractional versus whole-share sizing

The default `--share-mode fractional` preserves the original exact equal-notional
simulation. Give it initial portfolio equity when comparing it with integer shares;
each exit's P&L is rolled into the next basket:

```bash
python -m baseline.overnight_liquidity \
  --top 12 \
  --months 12 \
  --budget 10000 \
  --share-mode fractional
```

Use `--share-mode whole` to floor every selected stock's target allocation to a whole
number of shares:

```bash
python -m baseline.overnight_liquidity \
  --top 12 \
  --months 12 \
  --budget 10000 \
  --share-mode whole
```

Rounding is conservative: unused dollars remain cash rather than being reassigned to
cheaper names. A stock whose entry price exceeds its equal-notional target receives zero
shares for that session. The report includes deployed capital, utilization, executed
basket size, skipped selections, and the spread between the largest and smallest
position weights.

To trade only liquidity ranks 51--100, excluding the 50 most-liquid stocks:

```bash
python -m baseline.overnight_liquidity \
  --top 100 \
  --exclude-top 50 \
  --months 12 \
  --ema-span 20 \
  --transaction-cost-bps 1
```

## Alpaca paper execution

`live_overnight_liquidity.py` applies the same causal liquidity idea to an
Alpaca account. By default it starts ranking at 15:00 ET, opens an equal-notional top-10
basket at 15:55, and submits its exit at 09:00 on the next trading session.
Before ranking, it refreshes every symbol already present in the broad
split-adjusted daily cache at `/data/ppv1/updates/bars_1day_2016-01-01` using
batched SIP requests with 30 days of overlap. An exact overlap is appended;
any changed bar (including a newly reflected split) triggers a full retained-history
refresh for that symbol. The rank step fails closed unless Alpaca returns the
immediately preceding completed session, and it never uses the unfinished entry-day bar.
Active eligible companies missing from the cache are first seeded with split-adjusted
history from the shortlist epoch, so new listings can enter later shortlist rebuilds.

The refreshed cache rebuilds `data/most_liquid.txt` as the union of every
session's top 50 stocks by `volume * VWAP` since 2022-01-01. Symbols whose
split-adjusted prices cannot fit the compact `int32` minute schema are excluded.
The live rank then considers every currently eligible company in this shortlist,
instead of relying on Alpaca's top-share-volume or top-trade-count activity feed.
Configure these locations and bounds with `--daily-bars-dir`,
`--liquidity-shortlist`, `--shortlist-since`, `--shortlist-daily-top`, and
`--daily-overlap-days`.

As in the simulator, a company must have at least 100 completed daily bars
strictly before the entry date. Configure this with `--minimum-trading-days`;
the 10-session EMA span remains separately configurable with `--ema-span`.

The default ranking `--feed sip` provides whole-market liquidity measurements.
Whole-share sizing separately defaults to real-time `--quote-feed iex`, so it does
not require recent SIP quote access. Use `--quote-feed sip` only with an Alpaca
real-time SIP subscription. Candidates are restricted to active, tradable, fractionable
company stocks on the major exchanges. The same Nasdaq security-master filter
used by the simulator strictly removes ETFs (including SPY and QQQ), funds,
units, preferreds, debt, SPAC shells, and unclassified current assets before
shortlist members are ranked. Dollar liquidity is `daily VWAP * volume`,
smoothed as a causal EMA of `log1p(dollar volume)`.

Credentials remain in environment variables and are never stored in the state
file. For the paper endpoint:

```bash
export ALPACA_URL="https://paper-api.alpaca.markets/v2"
export ALPACA_KEY="..."
export ALPACA_SECRET="..."
export ALPACA_DATA_KEY="..."       # optional separate market-data subscription
export ALPACA_DATA_SECRET="..."

python -m baseline.live_overnight_liquidity run \
  --top 10 \
  --ranking-time 15:00 \
  --entry-time 15:55 \
  --entry-preflight-seconds 10 \
  --order-submit-workers 8 \
  --exit-time 09:00 \
  --feed sip \
  --quote-feed iex \
  --capital-fraction 1.00 \
  --share-mode whole \
  --submit
```

`--capital-fraction` defaults to `1.0`. Basket sizing remains cash-only even on a
margin-enabled account. The requested
capital is capped by positive cash after `--cash-buffer-fraction` and by Alpaca's
regular stock `buying_power`. It deliberately does not use
`non_marginable_buying_power`, because that settlement-sensitive field can exclude
same-day stock-sale proceeds even though they are immediately reusable for equities.
The pre-entry account fields used for sizing are saved in strategy state and daily
summaries for auditability.

In persistent `run` mode, entry is a two-phase operation. By default, the daemon
checks the account and conflicts, fetches the basket's latest quotes, calculates
whole-share quantities, and durably records the plan 10 seconds before
`--entry-time`. At the target second it performs no quote or account refresh; it
dispatches the prepared orders with up to eight concurrent workers. Configure
these values with `--entry-preflight-seconds` and `--order-submit-workers`.
Deterministic client order IDs make a partial or uncertain concurrent dispatch
restart-safe, and the state records per-order dispatch timing for later analysis.
Alpaca has no batch request for unrelated equity orders, so every symbol remains
an individual `POST /v2/orders` request.

Live execution defaults to `--share-mode whole`. During entry preflight it requests the
latest ask for every selected stock from `--quote-feed iex`, rejects missing quotes or
quotes older than the default `--quote-max-age-seconds 120`, and floors each
equal-notional allocation to an integer quantity using the same rule as the simulator.
Unused dollars and any allocation too small to buy one share remain cash; they are not
redistributed to cheaper names. The quote prices, timestamps, target quantities, skipped
symbols, and estimated deployed notional are persisted in strategy state. Use
`--share-mode fractional` to retain the earlier notional-order behavior. Both modes use
the same fractionable-company universe and the same liquidity ranking; share mode affects
only sizing and the submitted order payload.

When both data variables are set, all market-data requests use those credentials while
account, position, calendar, and order requests keep
using the trading credentials. If neither data variable is set, market-data
requests fall back to the trading credentials. Batched daily-bar refreshes use
`adjustment=split`, end at an explicit timestamp on the last completed session,
and never include the unfinished ranking day. Current IEX quotes are used only
for whole-share sizing, never for liquidity ranking.

`run` is persistent and must stay running across the overnight holding period.
It requires `--submit`; one-shot `enter` and `exit` are dry runs unless that
flag is present. Useful one-shot operations are:

Use `preview` to rank immediately and print today's proposed entry basket in
one command. It uses isolated temporary state, never accepts `--submit`, and
cannot replace or suppress the live daemon's scheduled ranking:

```bash
python -m baseline.live_overnight_liquidity preview \
  --top 10 \
  --entry-time 15:59 \
  --exit-time 09:00 \
  --capital-fraction 0.95
```

Whole-share exits submitted before Alpaca's 09:28 cutoff use `market` + `opg`
and participate in the primary exchange's opening auction. The default 09:00
exit time leaves a safety margin before that cutoff. Fractional exits remain
`day` market orders because Alpaca does not support OPG for fractional shares.
The daemon records `exit_queued` without canceling working orders on the normal
fill timeout, then reconciles fills at 09:30. If an OPG order is rejected,
cancelled, or leaves shares behind, the post-open recovery uses a `day` market
order. A still-working or partially filled exit remains in place across
45-second reconciliation windows instead of being canceled and replaced.
Entry submissions still require an open regular session.

```bash
python -m baseline.live_overnight_liquidity rank
python -m baseline.live_overnight_liquidity enter
python -m baseline.live_overnight_liquidity status
python -m baseline.live_overnight_liquidity exit
```

The default work directory is `/data/ppv1/live`. Durable restart state is
written atomically to `/data/ppv1/live/state.json`, while every ET trading day
gets an audit directory such as:

```text
/data/ppv1/live/2026-08-24/
├── live.log
├── summary.json
├── ticks.jsonl
└── ticks_summary.json
```

`ticks.jsonl` contains the actual historical market records consumed by the
ranking and labels every row with its `1Day` timeframe and feed. It is not a
trade-by-trade exchange tick stream. `summary.json` is refreshed after ranking,
entry, exit, status, and failures; it includes the ranking, position and order
state, and realized fill-price P&L after exit. The daily log rolls over using
the America/New_York date. Use `--work-dir` to override the entire root or
`--state-path` to relocate only restart state.

The Nasdaq security-master cache is also kept under the work root.

Entry and exit order IDs are deterministic, so restarting the process does not
intentionally duplicate an order. It also excludes symbols with pre-existing
account positions or open orders and exits only the symbols recorded as owned
by this strategy. Sizing uses cash rather than margin buying power and leaves a
2% cash buffer by default; use `--capital 50000` for a fixed cap instead.

To export the current Alpaca-available company universe used by the live
strategy, run:

```bash
python -m baseline.export_alpaca_companies
```

This writes one symbol per line to `data/nasdaq.txt` at the repository root.
It includes active, tradable, fractionable company stocks on the configured
major exchanges and excludes ETFs and other non-company securities.
