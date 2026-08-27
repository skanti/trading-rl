# Trading baselines

`overnight_liquidity.py` implements a point-in-time overnight baseline:

1. Sum each stock's regular-session dollar volume for every completed day.
2. Smooth `log1p(dollar_volume)` with a causal EMA.
3. Before each 15:55 entry, rank using the EMA state through the previous
   session only. The current session never contributes to its own rank.
4. Equal-weight the selected stocks, then close them at 09:45 on the next
   trading session.

Every simulated stock must have at least 100 completed observed trading
sessions before it can be selected. The current session is not counted. Change
this causal listing-history filter with `--minimum-trading-days`; the default
excludes recent IPOs such as SPCX until they establish 100 sessions.

The log transform and default 20-session EMA keep earnings, index-rebalance,
and news-related volume spikes from dominating the ranking. Use `--ema-span 1`
to rank strictly by the previous completed session without smoothing.

The first run scans the necessary minute-file slices in parallel and writes a
date-by-symbol cache under `/tmp/trading/baseline_cache`. Subsequent top-50 and
top-100 runs with the same date range reuse that cache.

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
basket at 15:55, and closes that basket at 09:35 on the next trading session.
The fast candidate screen unions Alpaca's current top 100 SIP symbols by share
volume, current top 100 by trade count, and the previous day's screened
universe. It then downloads only those candidates' completed daily bars from
the most recent 180 calendar days and reranks them by lagged EMA dollar volume.
It never downloads history back to 2016 and never uses the unfinished
entry-day bar. `--activity-candidates` can reduce the per-screen request below
100, at the cost of a less complete candidate funnel.

As in the simulator, a company must have at least 100 completed daily bars
strictly before the entry date. Configure this with `--minimum-trading-days`;
the 20-session EMA span remains separately configurable with `--ema-span`.

The default `sip` feed provides whole-market liquidity measurements and
requires an Alpaca SIP data subscription. Use `--feed iex` when running on a
free data plan. Candidates are restricted to active, tradable, fractionable
company stocks on the major exchanges. The same Nasdaq security-master filter
used by the simulator strictly removes ETFs (including SPY and QQQ), funds,
units, preferreds, debt, SPAC shells, and unclassified current assets before
historical bars are downloaded. Dollar liquidity is `daily VWAP * volume`,
smoothed as an EMA of `log1p(dollar volume)`.

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
  --exit-time 09:35 \
  --capital-fraction 0.90 \
  --submit
```

Basket sizing remains cash-only even on a margin-enabled account. The requested
capital is capped by positive cash after `--cash-buffer-fraction` and by Alpaca's
regular stock `buying_power`. It deliberately does not use
`non_marginable_buying_power`, because that settlement-sensitive field can exclude
same-day stock-sale proceeds even though they are immediately reusable for equities.
The pre-entry account fields used for sizing are saved in strategy state and daily
summaries for auditability.

When both data variables are set, SIP screener and historical-bar requests use
that subscription while account, position, calendar, and order requests keep
using the trading credentials. If neither data variable is set, market-data
requests fall back to the trading credentials. Daily-bar requests end at an
explicit timestamp on the last completed session and never include the
unfinished ranking day.

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

Exit submission times from 09:00 through 09:29 queue fractional `day` market
orders for the regular-session open. The daemon records `exit_queued` without
canceling those orders on the normal fill timeout, then reconciles their fills
at 09:30 and retries only when necessary. Once the market is open, a still-working
or partially filled exit is also left in place across 45-second reconciliation
windows instead of being canceled and replaced. Entry submissions still require
an open regular session.

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
