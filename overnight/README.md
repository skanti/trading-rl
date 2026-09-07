# Overnight dollar-liquidity strategy

Install the repository once in editable mode from its root so every command and
cross-package import resolves through the `trading_rl` package:

```bash
uv pip install -e .
```

The installed commands include `trading-dashboard`, `trading-live`,
`trading-backtest`, `trading-reconcile`, `download-bars`,
`download-auctions`, and `download-nbbo`.
The existing `python overnight/live.py` and `python scripts/download_bars.py`
forms remain as compatibility launchers.

`backtest.py` simulates the strategy point-in-time:

1. Read each stock's completed daily dollar volume as split-adjusted daily
   `VWAP * volume` (falling back to the daily close when VWAP is unavailable).
2. Smooth `log1p(dollar_volume)` with a causal EMA, then subtract that series'
   dispersion over the same span so steady turnover outranks episodic turnover.
3. Before each 15:45 entry, rank using the EMA state through the previous
   session only. The current session never contributes to its own rank.
4. Equal-weight the selected stocks, then close them in the next session's
   opening auction.

Exits default to `--exit-price-source opening-auction` at `--exit-time 09:30`,
because that is the price a live order actually receives: Nasdaq routes any
market order reaching the broker before 09:28 into the opening cross. The
`minute` source prices the first consolidated print instead, which no order type
can target and which sits about 0.9 bps above the cross (36 months, t = -7.1),
so it flatters a backtest by roughly three annualized points.

Run the tests from this directory:

```bash
python -m unittest discover -s tests
```

Transaction costs default to 1 basis point per side, or 2 basis points for a
complete entry-and-exit round trip. Override this with `--transaction-cost-bps`.
The potentially long per-symbol trade-frequency table is hidden by default;
include `--show-symbol-trade-frequency` when that breakdown is needed.

Every simulated stock must have at least 100 completed observed trading
sessions before it can be selected. The current session is not counted. Change
this causal listing-history filter with `--minimum-trading-days`; the default
excludes recent IPOs such as SPCX until they establish 100 sessions.

The log transform and default 10-session EMA keep earnings, index-rebalance,
and news-related volume spikes from dominating the ranking. Turnover stability
uses that same span for its dispersion window, keeping the two horizons aligned.

The first run combines split-adjusted daily bars from
`/data/ppv1/updates/bars_1day_2022-01-01` with execution prices from
`/data/ppv1/updates/bars_1min_2022-01-01`, then writes a date-by-symbol cache
under `/tmp/trading/baseline_cache`. The daily bars drive the causal liquidity
ranking; minute bars are used only for entry/exit prices. Override the two stores
with `--daily-bars-dir` and `--data-dir`, respectively. Subsequent runs with the
same inputs and date range reuse the cache.

Use `--since YYYY-MM-DD` instead of `--months` to anchor the first eligible entry
session to a fixed date. The two options are mutually exclusive; for example:

```bash
python backtest.py --since 2024-01-01
```

The simulator does not require a generated day-index CSV. It derives complete
New York sessions from `SPY.npy`, discovers the tradable universe from symbols
present in both bar stores, and locates execution timestamps directly in each
minute array. Updating the bar directories therefore makes new symbols and
sessions available without rebuilding separate metadata.

The backtester supports the same two completed-session liquidity schemes as live
trading:

- `dollar_ema`: lagged EMA of `log1p` completed-session dollar volume.
- `turnover_stability`: that same EMA less the same-span dispersion of the
  log-dollar-volume series. Both terms are log dollars, so the subtraction needs
  no weighting: a name whose log turnover swings by 1.0 is docked as much as a
  name with `e` times less turnover. It demotes a stock that is only briefly
  enormous -- an earnings day, an index rebalance -- beneath one that trades
  heavily every session, which matters because the basket is held through an
  `live.py --liquidity-scheme` selects the same implementation, so a live basket
  and a simulated one cannot drift apart. This is the default in both; pass
  `--liquidity-scheme dollar_ema` for the level-only ranking.

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

Entry prices must have a print within 10 minutes of the entry time. A selected stock is
never removed using knowledge of whether it trades the following morning. If
no fresh print exists at the exit time, the backtest uses the latest causal mark
up to 24 hours old and reports its staleness in both the trade CSV and summary.
This is preferable to silently replacing the stock with next-morning lookahead,
but such a stale mark is not evidence that a live order could have filled there.
This fallback applies to `--exit-price-source minute`; the auction source has no
stale path, since a session either produced a condition-O cross or did not.

### Incremental minute-bar updates

Existing `.npy` bar files can be extended without redownloading every ticker's full
history. Filenames use plain symbols such as `AAPL.npy` and `BRK.B.npy`.
The bulk downloader and live ranking refresh both use
`trading_rl/market_data/bars.py` for
Alpaca request construction and pagination, compact bar encoding, exact-overlap
validation, and atomic array replacement. Their orchestration remains separate so
the live path can require a specific completed session and fail closed without
partially updating its cache.

To refresh the broad daily store, rebuild the dollar-volume shortlist, extend
auction data, and update shortlist minute bars in one pass, run:

```bash
../scripts/download_latest_bars_and_auctions.sh
```

The script first rewrites `data/master.txt` with Alpaca's currently active,
tradable, fractionable Nasdaq company stocks. Only that current universe is
refreshed, while historical and delisted `.npy` files already in the bar stores are
retained for backtests. It then loads `overnight/.env` by default and accepts
environment overrides such as `PYTHON_BIN`, `UPDATES_DIR`, `WORKERS`, and the
overlap/shortlist settings shown by `--help`. Auction updates use the current New
York date. An optional NBBO update runs last when `NBBO_TARGETS_PATH` names a
simulator trade CSV; targets still inside Alpaca's delayed-SIP window are deferred
and filled by the next overlap refresh.
The shortlist is ordered by consistent daily top-N appearances, then trailing
average liquidity, so its most persistently liquid symbols enter the concurrent
minute-download queue first.

```bash
python ../scripts/download_bars.py \
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
python ../scripts/download_bars.py \
  --source alpaca \
  --timeframe 1Day \
  --tickers_path /home/aavetisyan/dev/trading-rl/data/master.txt \
  --out_dir /data/ppv1/updates/bars_1day_2022-01-01 \
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
python ../scripts/download_auctions.py \
  --start 2022-01-01 \
  --end 2026-08-27 \
  --symbols-from-trades /tmp/overnight_liquidity_48m_minute.csv \
  --output /data/ppv1/updates/alpaca_auctions_2022-01-01.npz
```

For later updates, reuse the same output and provide only a new end date:

```bash
python ../scripts/download_auctions.py \
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
`--symbols-file /data/ppv1/updates/liquidity_candidates.txt` to align the auction
universe with the complete liquidity candidate set. Use the last completed
trading date for `--end` when the Alpaca plan does not permit querying the most
recent SIP data.

Then run the comparison:

```bash
python backtest.py \
  --top 12 \
  --months 24 \
  --budget 10000 \
  --share-mode fractional \
  --transaction-cost-bps 1
```

For a causal spread-aware entry, generate a normal minute-bar backtest CSV and use its
exact `(entry_date, sample_id)` basket as the NBBO download schedule:

```bash
trading-backtest \
  --since 2024-01-01 \
  --entry-price-source minute-bar \
  --output-csv /tmp/overnight_nbbo_targets.csv

download-nbbo \
  --targets-from-trades /tmp/overnight_nbbo_targets.csv \
  --output /data/ppv1/updates/alpaca_nbbo_1545_2024-01-01.npz
```

The downloader stores the latest valid SIP bid/ask at or before 15:45 ET, never a
later quote, with raw and split-adjusted fields. It requests only each session's
selected basket rather than crossing every symbol that appeared during the entire
window with every date. Basket rotation is therefore preserved: a multi-month run may
contain many unique symbols, but normally only that day's 12 targets plus the SPY
benchmark are queried in a single batch. Any target without a valid quote inside
`--lookback-seconds` (60 seconds by default) aborts the download before the dataset is
replaced. Version-1 broad-union NBBO files must be rebuilt once with the target CSV.

Use the resulting dataset in a backtest with `--entry-price-source nbbo-ask`; the
backward-compatible default remains `minute-bar`. The NBBO source requires
`--entry-time 15:45`. A buy is benchmarked at the ask, while the existing
transaction-cost assumption remains separately visible. Set `NBBO_TARGETS_PATH` to
the same trade CSV when running the all-in-one downloader wrapper; without it the
wrapper explicitly skips the optional NBBO update.

Afternoon entries are skipped on shortened sessions whose official close is at or
before the configured entry time. Historical runs read the close from the auction NPZ
(`--auctions-path`); the NBBO downloader uses the same file, and the live daemon uses
Alpaca's session-specific calendar close. The shortened date remains in the session
calendar so a position entered on the preceding full day can still exit at its normal
09:30 opening auction. Regenerate the minute-bar target CSV after changing this policy;
do not widen NBBO staleness to carry a 13:00 quote forward to 15:45.

`--entry-time` defaults to 15:45 rather than the close. The selected basket
drifts about 3.5 bps upward between 15:45 and 15:59 (t=2.66 over 500 sessions),
so a later entry pays more for the same names; 15:40 and 15:45 form a plateau and
both beat 15:59 in each half of the window separately. `experiments.py` records
the measurement.

`--exchange-filter` defaults to `nasdaq`, matching live execution, and removes
non-Nasdaq candidates before ranking. The top basket is then reranked from the
remaining Nasdaq company universe; the filter does not merely discard NYSE names
after selection. SPY remains available only as the benchmark. Pass
`--exchange-filter all` for the unrestricted universe.

When the auction file is present, historical primary-auction venues override the
current directory for known symbol-dates, so a listing transfer such as PLTR's
2024 NYSE-to-Nasdaq move is handled at the correct session. A venue is matched
against every SIP code for one listing market: Alpaca publishes a single Nasdaq
opening cross under both `Q` and `T`, and on sessions such as 2023-01-30 only the
`T` copy is present, so matching `Q` alone would drop the entire Nasdaq universe
for that date.

### Fractional versus whole-share sizing

The default `--share-mode fractional` preserves the original exact equal-notional
simulation. Give it initial portfolio equity when comparing it with integer shares;
each exit's P&L is rolled into the next basket:

```bash
python backtest.py \
  --top 12 \
  --months 12 \
  --budget 10000 \
  --share-mode fractional
```

Use `--share-mode whole` to floor every selected stock's target allocation to a whole
number of shares:

```bash
python backtest.py \
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

## Alpaca paper execution

`live.py` applies the same causal liquidity idea to an
Alpaca account. By default it starts ranking at 14:00 ET, opens an equal-notional top-12
basket at 15:45, and submits its exit at 08:00 on the next trading session.
The daemon checks Alpaca's market calendar once per New York date and idles on
weekends and exchange holidays instead of attempting scheduled actions.
Before ranking, it refreshes every symbol already present in the broad
split-adjusted daily cache at `/data/ppv1/updates/bars_1day_2022-01-01` using
batched SIP requests with 30 days of overlap. An exact overlap is appended;
any changed bar (including a newly reflected split) triggers a full retained-history
refresh for that symbol. The rank step fails closed unless Alpaca returns the
immediately preceding completed session, and it never uses the unfinished entry-day bar.
Active eligible companies missing from the cache are first seeded with split-adjusted
history from the shortlist epoch, so new listings can enter later shortlist rebuilds.

The market-data download script rebuilds
`/data/ppv1/updates/liquidity_candidates.txt` as the union of each session's top
50 stocks by `volume * VWAP`, over the most recent 250 completed sessions on or
after 2022-01-01. Symbols whose split-adjusted prices cannot fit the compact
`int32` minute schema are excluded. This downloader-owned file lets the minute-bar
and auction downloads follow the same universe.

The live rank independently recomputes the same candidate set in memory and
considers every currently eligible company in it, instead of relying on Alpaca's
top-share-volume or top-trade-count activity feed. It never reads or writes the
shared downloader artifact. Each rank only snapshots its own candidate set under
`/data/ppv1/live/YYYY-MM-DD/liquidity_candidates.txt`; that day's fresh nominal
top 12 is `ranking.ranked_top_symbols` in `summary.json`.
The longer ranked reserve remains in `ranking.candidates`, while
`position.symbols` records the actual basket after entry-time conflict and
duplicate-share-class filtering.

`--shortlist-lookback-sessions` bounds the union to a trailing window so the
shortlist tracks current liquidity. Without it, one session in the daily top 50
in 2022 bought permanent candidacy and the list only ever grew: unioning all
1,168 sessions since 2022-01-01 yields 724 candidates, of which 386 had not been
in a daily top 50 for a year. The trailing window is a candidate-set bound, not a
ranking change -- over the last 500 sessions no window down to 125 ever dropped a
name from the reserve the entry step draws on. Pass `0` to restore the
union-everything behaviour.

| `--shortlist-lookback-sessions` | shortlist size |
| --- | --- |
| `0` (all 1,168 sessions) | 724 |
| 750 | 592 |
| 500 | 500 |
| 250 (default) | 338 |
| 125 | 218 |

Configure these locations and bounds in `config.yaml`; the corresponding CLI options
remain available as one-run overrides.

As in the simulator, a company must have at least 100 completed daily bars
strictly before the entry date. Configure this with `--minimum-trading-days`;
the 10-session EMA span remains separately configurable with `--ema-span`.

The default ranking `--feed sip` provides whole-market liquidity measurements.
Whole-share sizing separately defaults to real-time `--quote-feed iex`, so it does
not require recent SIP quote access. Use `--quote-feed sip` only with an Alpaca
real-time SIP subscription. Candidates default to active, tradable Nasdaq company
stocks; use `--exchanges` to explicitly select a different venue set. Fractional
mode additionally requires Alpaca's `fractionable` flag, while whole-share mode
does not. The same Nasdaq security-master filter
used by the simulator strictly removes ETFs (including SPY and QQQ), funds,
units, preferreds, debt, SPAC shells, and unclassified current assets before
shortlist members are ranked. Dollar liquidity is `daily VWAP * volume`, falling
back to `daily close * volume` when VWAP is unavailable or zero, and is smoothed
as a causal EMA of `log1p(dollar volume)`.

`config.yaml` is the complete, validated source of truth for the live runner's
schedule, ranking, data, sizing, and runtime parameters. It is loaded with OmegaConf;
missing, unknown, or mistyped values fail before any broker call. Explicit CLI options
override YAML for operational exceptions, while the positional action (`run`, `status`,
and so on) remains a required command. The process prints the effective non-secret
settings at startup for a sanity check.

Credentials remain in environment variables and are never stored in YAML or the state
file. Endpoint values support OmegaConf environment interpolation. For a paper endpoint:

```bash
export ALPACA_URL="https://paper-api.alpaca.markets/v2"
export ALPACA_KEY="..."
export ALPACA_SECRET="..."
export ALPACA_DATA_KEY="..."       # optional separate market-data subscription
export ALPACA_DATA_SECRET="..."

python live.py run --submit
```

For the configured live endpoint, retain both explicit safety gates:

```bash
python live.py run --submit --allow-live-endpoint
```

Use `--config /path/to/alternate.yaml` to select another complete configuration.
For a temporary exception, a regular option has final precedence, for example
`python live.py status --entry-time 15:40 --top 10`.

Persistent `run` mode atomically records the merged values it actually consumed in
`WORK_DIR/effective_config.json`. This includes CLI overrides but no credentials. The
dashboard publisher prefers that active-runtime artifact and falls back to YAML when it
does not exist, so an override or an unapplied YAML edit cannot make the displayed
schedule drift from the running trader.

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

The shipped live configuration uses `share_mode: fractional`, preserving the current
deployment. In whole-share mode, entry preflight requests the
latest ask for every selected stock from `--quote-feed iex`, rejects missing quotes or
quotes older than the default `--quote-max-age-seconds 120`, and floors each
equal-notional allocation to an integer quantity using the same rule as the simulator.
Unused dollars and any allocation too small to buy one share remain cash; they are not
redistributed to cheaper names. The quote prices, timestamps, target quantities, skipped
symbols, and estimated deployed notional are persisted in strategy state. Both modes use
the same liquidity ranking. Fractional mode restricts the universe to Alpaca-fractionable
companies; whole mode can also rank non-fractionable companies. Share mode otherwise
affects sizing and the submitted order payload.

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
python live.py preview \
  --top 10 \
  --exit-time 08:00 \
  --capital-fraction 0.95
```

Whole-share exits submitted before Alpaca's 09:28 cutoff use `market` + `opg`
and participate in the primary exchange's opening auction. Nasdaq applies the same
09:28 cutoff to plain market orders, so a fractional `market` + `day` exit reaches
the cross too and fills at the Nasdaq Official Opening Price -- measured across 22
live fills on 2026-08-28 and 2026-08-31, every one matched the cross exactly, in
both share modes. The default 08:00
exit time leaves a safety margin before that cutoff. Fractional exits remain
`day` market orders because Alpaca does not support OPG for fractional shares.
The daemon records `exit_queued` without canceling working orders on the normal
fill timeout, then reconciles fills at 09:30. If an OPG order is rejected,
cancelled, or leaves shares behind, the post-open recovery uses a `day` market
order. A still-working or partially filled exit remains in place across
45-second reconciliation windows instead of being canceled and replaced.
Entry submissions still require an open regular session.

```bash
python live.py rank
python live.py enter
python live.py status
python live.py exit
```

The configured work directory is `/data/ppv1/live`. Durable restart state is
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
the America/New_York date. Change `runtime.work_dir` or `runtime.state_path` in YAML,
or use `--work-dir`/`--state-path` for a one-run override.

The Nasdaq security-master cache is also kept under the work root.

### Reconcile live sessions with the simulator

`reconcile_live_sessions.py` replays a completed basket using only information the
simulator can observe: the split-adjusted causal SIP NBBO ask at the scheduled 15:45
entry and the next session's primary condition-O opening auction. If a scheduled NBBO
snapshot is missing, it warns once with an aggregate count and falls back to the 1-min
bar open for only those symbols. It compares those prices
with the actual filled quantities and prices, separately reports the configured
transaction-cost assumption and the broker-equity residual, and re-runs the shared
liquidity ranker from that entry day's archived `ticks.jsonl`. The default console
report is one overview table across the selected period: dollar values are cumulative
totals and execution differences are deployed-capital-weighted basis points. Fee-based
actual net results use only fee-confirmed sessions and show their coverage. Entry-fill,
exit-fill, and quantity-mismatch P&L impacts remain available in the detailed artifacts.
Reconciliation also checks fill timestamps against the scheduled entry minute and the
09:30 opening cross, and verifies that exit orders were submitted before the 09:28
auction cutoff. A deviation beyond 1 min marks the session as not schedule
comparable. Strict schedule filtering is enabled by default: the whole session is
excluded from reconciliation and aggregate results, with a warning explaining whether
its entry, exit, or both were off schedule. Configure the tolerance with
`--schedule-tolerance-minutes`. Use
`--reconciliation-mode actual-time-minute-bar` to produce a forensic reconciliation
of a botched session (`actual-time` remains an alias). In that mode, the headline P&L and
execution comparison aligns every symbol's entry and exit to the open price of the
1-min bar containing that symbol's fill. The scheduled entry bar and opening auction
remain in the artifacts and detailed report as the scheduled-strategy counterfactual,
rather than being presented as ordinary execution slippage.
Actual-time mode never mixes benchmark types across a date range: if either actual-time
leg is unavailable for a session, that date fails reconciliation instead of falling
back to its scheduled prices.

With no date it reconciles the latest closed session. Select one session or a range with:

```bash
python reconcile_live_sessions.py --entry-date 2026-08-28
python reconcile_live_sessions.py --since 2026-08-01
python reconcile_live_sessions.py --entry-date 2026-08-28 --show-symbol-breakdown
python reconcile_live_sessions.py --since 2026-08-01 --show-session-details
```

Results are written to `WORK_DIR/reconciliations/YYYY-MM-DD.json` and `.csv`. Use
`--show-session-details` for the former per-session tables and
`--show-symbol-breakdown` for per-symbol execution attribution. Use
`--entry-time` or `--liquidity-scheme` only to reconstruct legacy summaries that lack
those fields; normal sessions retain their original schedule and ranking snapshot so a
later daemon restart or configuration change cannot rewrite the audit inputs.

By default the reconciler authenticates with the same Alpaca environment variables as
`live.py`, reads the trading endpoint from `WORK_DIR/effective_config.json`, and queries
`FEE` account activities around the exit date. Exit-date fees are cached in the entry
session's `fee_activities.json`, embedded in the reconciliation JSON, and used for the
actual net P&L. Bulk regulatory fees are account-day amounts and may not identify an
individual order, so the report labels that scope explicitly. Use `--skip-broker-fees`
for a fully offline gross reconciliation or `--trading-url` for an explicit endpoint.
The simulator's default one-basis-point cost per side remains a separate modeled
deduction. Account-equity residuals stay separate from confirmed Alpaca fee activities,
and gross actual-versus-simulator execution attribution always excludes costs.

Entry and exit order IDs are deterministic, so restarting the process does not
intentionally duplicate an order. It also excludes symbols with pre-existing
account positions or open orders and exits only the symbols recorded as owned
by this strategy. Sizing uses cash rather than margin buying power and leaves a
2% cash buffer by default; use `--capital 50000` for a fixed cap instead.

To export the current Alpaca-available company universe used by the live
strategy, run:

```bash
python universe.py
```

This writes one symbol per line to `data/nasdaq.txt` at the repository root.
It exports the conservative fractional-order universe: active, tradable,
fractionable Nasdaq company stocks by default, excluding ETFs and other
non-company securities. Use `--exchanges` to override that venue default.
