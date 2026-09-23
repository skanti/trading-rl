# Backtest strategies and variant comparisons

Select a strategy by its behavior:

```bash
trading-backtest --strategy liquidity-trend-vol --since 2026-01-01 --budget 10000
trading-backtest --strategy liquidity-fixed --since 2026-01-01 --budget 10000
```

Compare strategy families with one shared market-data load:

```bash
trading-backtest --strategy liquidity-momentum-blend,liquidity-fixed \
  --since 2023-01-01 --budget 10000
```

The table contains a result column for each strategy and one for SPY buy-and-hold.
Names can be separated by commas or spaces. Each strategy keeps its own defaults;
explicit basket/EMA overrides apply to all. Dates, capital and execution settings
are shared, and mismatched reporting calendars or benchmarks are rejected.
Combined CSV/JSON tables and an equity chart accompany separate run artifacts
under `/tmp/trading-backtests/candidate/comparisons/` or `--output-dir`.
`--strategy-config` and individual output-file flags require a single strategy.

`liquidity-momentum-focus` is the live and backtester default: momentum selection
within a liquidity shortlist, a SPY trend filter and volatility targeting. Select `--strategy liquidity-fixed` for
a liquidity-ranked basket with fixed exposure.

`--strategy liquidity-momentum-blend` selects the experimental blend of five- and
ten-session stock momentum. It reached 109.74% calendar CAGR / 19.40% minute-mark
drawdown in the 2023–September 2026 study, using reused historical data for selection.
It is an in-sample research preset, not approved for live trading. An extra 1 bp
per side lowers its CAGR to 96.77%. `liquidity-regime-vol` and
`liquidity-momentum-vol` retain earlier experimental families for comparisons.
All use shared execution, allocation and financing code. See
[research results and plugin instructions](../research/README.md).
`liquidity-momentum-focus` simplifies the blend to a single equal-weight top-three
basket ranked by ten-session momentum, with a 100-session SPY trend and
20-basket volatility history. These are the user-selected defaults, with a 35%
volatility target and 2x cap. Through September 23, 2026 their in-sample result is
120.39% calendar CAGR, 2.365 Sharpe and 21.12% minute-mark drawdown. The earlier
optimized 8/150/40 settings remain available as an explicit variant. See the
[full study and reproduction commands](../research/MOMENTUM_FOCUS.md).
Backtester-only plugins can be registered with `--strategy-plugin MODULE`; live
and reconciliation retain a separate approved strategy registry. The backtester
exposes `prepare_backtest(argv)` to prepare one reusable input panel for comparisons
without running a strategy or producing reports.
Names describe strategy families; parameter values belong in `--strategy-config`.
The live YAML selects `liquidity-momentum-focus`; `trading-live --strategy` also
accepts `liquidity-trend-vol` and `liquidity-fixed`. Existing processes retain their startup configuration until
the user restarts them.

`liquidity-trend-vol` uses the selected research rules: the usual top 12 distinct
Nasdaq issuers and turnover-stability ranking, a 35% annual volatility target,
2x maximum exposure, 20 completed unscaled net basket returns for volatility,
and a 100-session SPY trend average. It uses 1x while volatility warms up
(and when volatility is zero). Exposure is multiplied by 0.25 when the previous
session's split-adjusted SPY daily close is below its trailing average or the trend
history is incomplete. Both SPY inputs exclude the current session.

Ranking and SPY indicators use all available earlier history. Risk observations
start at the research inception, `2023-01-01`, independent of `--since`; changing
the report start resets capital and performance, not the signals. That inception
is a named configuration parameter, not a frozen data end date. Short sessions
remain zero-return observations. `liquidity-trend-vol` requires complete fresh prices
for selected baskets, including risk warmup. The full ranking-history requirement
still applies at inception; a later-starting dataset needs a later inception.

Entries default to stored 15:45 asks and exits to the next opening auction.
Additional costs default to zero for that pair and 1 bp per side otherwise.
Borrowing defaults to 6.75% annually for `liquidity-trend-vol`, charged only on actual
borrowing using calendar days / 360. `--margin-interest-rate 5` means 5%.
`--leverage` controls `liquidity-fixed`; `liquidity-trend-vol` exposure comes from its policy.
`liquidity-trend-vol` requires the exit clock before the entry clock so every risk
observation has completed before the next entry.

The strategies share ranking, sizing, financing, compounding and metric code.
Whole shares round down per position and financing uses actual deployed capital.
`liquidity-fixed` now includes early-close cash intervals in its metrics, initializes
rankings from all available history, and sizes leveraged trades before calculating
P&L. Its historical exchange check retains the exit-session convention used by
reconciliation; `liquidity-trend-vol` preserves research entry-session membership.
`exchange_membership_session` records this distinction. These conventions can
differ around exchange changes, so do not attribute those differences to risk
sizing. Current security-master classification also remains a limitation; a fully
point-in-time universe is a separate data improvement.

Strategy output defaults to a unique directory under
`/tmp/trading-backtests/candidate/<strategy>/`. `--output-dir` overrides it.
Runs write `trades.csv`, `portfolio.csv`, `minute_marks.csv`,
`summary.json`, and an equity chart. Summaries include resolved policy and CLI
parameters, data-manifest metadata, source hashes, input archive hashes, session
metrics and calendar CAGR. Minute-open audits reconcile to every modeled exit;
they carry stale prints and do not measure intraminute extremes or executable
bid-side drawdown. Historical research results and reports were archived under
`/tmp/trading-backtests/candidate/research-archive-*` before the research source
was removed. Temporary output needs copying to durable storage if it must survive
system cleanup.

## Live momentum-focus strategy

The shipped live configuration selects `liquidity-momentum-focus` with 10-session
momentum, a 100-session SPY average, 20 modeled basket returns, a 35% volatility
target and a 2x cap. Ranking first selects 12 liquid issuers, then chooses three
by momentum from completed daily closes. Equal momentum retains liquidity order.
A weak or incomplete SPY trend means zero exposure. The older trend/volatility
strategy retains its twelve-stock basket and 0.25 weak-trend multiplier. It keeps fractional equal-notional entries at 15:45 ET, a 2% sizing
buffer, and DAY market exits queued at 06:00 ET. For Nasdaq stocks, market orders
received by Alpaca before 09:28 ET receive the official opening price; fractional
DAY orders retain that price basis. State records the broker submission timestamp
and whether it precedes the cutoff. Missing timestamps remain unknown.

At ranking time, `live_risk.py` prepares the last 20 completed modeled overnight
returns through that morning's exit, including zero returns for short-session
cash intervals. Historical rankings use the shared scorer and historical venue
filter, followed by the same three-stock momentum selection. Trend-filter cash
days still observe the modeled three-stock return for volatility. Entry prices are SIP asks at 15:45; exits are primary opening auctions;
additional modeled costs are zero. Raw prices and a refreshed split ledger keep
each entry/exit pair on the same basis. Missing quotes and auction prints are
fetched through read-only data requests. SPY uses the previous 100 sessions'
split-adjusted daily closes from the same daily cache as the backtester. Ranking
seeds missing SPY daily history and refreshes it with the stock bars; SPY never
occupies a stock-shortlist slot. The current day's close is excluded. A missing
required input blocks new entries; exits do not
require risk data. Actual leveraged account returns never feed the risk estimate.

Backtests also default to daily closes. Use `--spy-trend-price-source minute-open-1559`
to reproduce the original research price basis for comparison. The volatility
window, overnight return inputs and execution prices stay the same. Run summaries
and new live snapshots identify `spy_trend_price_source`; new snapshots store each
SPY observation as `{date, price}`. Reconciliation continues to replay old snapshots
containing `minute_open` using their original prices, without rewriting past trades.
Changing the source invalidates a cached ranking signal and requires fresh risk
preparation. An already prepared entry keeps its saved plan.

Snapshots under `WORK_DIR/risk/<strategy>/YYYY-MM-DD.json` record inputs,
parameters, timestamps, configuration and data fingerprints. Ranking state also
holds the snapshot. A date or parameter change requires fresh preparation. Entry
preflight saves the risk snapshot, allocated equity, target and effective exposure,
broker limits, and exact order plan with the position. Retries reuse that plan
and deterministic order IDs. The current and historical momentum snapshots record
the full liquidity shortlist, lagged close endpoints, history dates and selected
symbols. Reconciliation recomputes selection, exposure and sizing from these saved
inputs. Existing fixed and trend/volatility positions keep their recorded semantics
and exit normally; historical sessions are not relabeled or backfilled as focus.

The ranking stage does the momentum and risk-history work. Preflight validates
those snapshots and checks current account limits before saving an immutable
order plan. A conflict in one of the chosen three blocks entry instead of silently
substituting another stock. Cash sessions save a closed, zero-order decision that
reconciliation can replay. The new pipeline fingerprint forces old ranking caches
to rebuild after restart; already prepared order plans remain unchanged.

Allocated capital is the smaller of account equity and `--capital`, or account
equity times `--capital-fraction`. Target notional is allocated capital times
policy exposure, constrained by regular and Reg T buying power, account-wide
2x exposure after existing holdings and open buys, and reported asset maintenance
requirements. The 2% buffer is applied after these limits. Cash accounts or baskets
containing a non-marginable asset are additionally capped at cash. Fractional
notionals are rounded down to cents. `liquidity-fixed` retains cash-only sizing.

`strategies.py`, `momentum.py`, `risk_history.py`, `portfolio.py`, and
`execution_prices.py` hold shared policies, momentum selection, risk calculations,
portfolio operations, and explicit price loaders. Live imports neither the
backtester nor research modules.
`BasketHistory` supplies the same lagged ranking features and basket selection to
simulation and live risk bootstrap. The shared return evaluator validates prices
and computes unit-exposure observations. Allocation uses an explicit slot count,
so missing executions retain their cash weight. Live, replay and simulation also
share equal-notional sizing and the policy's trailing volatility calculation.

The simulator retains continuous fractional sizing and its configured financing
and costs. Live applies cent rounding, the 2% buffer and observed broker limits;
those account constraints cannot be inferred from historical price arrays. The
fixed strategy's historical exit-day venue convention remains explicit. These
execution assumptions are separate from the shared strategy calculations.

Live execution imports no backtest, audit, plotting, or experiment runner, directly
or transitively. Import-boundary and simulator/live parity tests enforce this.

The user must restart `trading-live` to apply the code and YAML, preferably before
the 14:00 ET ranking stage. Restarting does not migrate or resize an existing
position. The next new basket uses the new policy once complete risk data is ready.
A preview before 09:30 cannot validate that day's morning auction yet.

## Exact live decision replay

New entries archive `position.decision_inputs` before submitting orders. This
versioned snapshot contains the sizing and ranking configuration, the broker account
fields used by the policy, existing holdings and open orders, and selected assets'
margin eligibility and maintenance requirements. The position also keeps its original
ranking, model returns with entry/exit prices, all 100 SPY marks, policy parameters,
and whole-share quotes where applicable. These inputs survive retries and exits.
Live and reconciliation share the same sizing, selection, order-ID and order-payload
functions; replay never consults today's account or configuration.

`trading-reconcile` now writes `decision_replay` with individual checks for risk
exposure, broker limits, ordered membership, budget, cent rounding, whole-share
quantities and requested order sizes. It also regenerates the complete broker
payloads. A mismatch or missing input is explicit. `ranking_replay` independently
recalculates scores from the session's archived daily bars, applying the recorded
position/order exclusions. The decision-only mode requires no market-data refresh
or broker credentials and includes off-schedule sessions:

```bash
trading-reconcile --since 2026-08-27 --decision-only \
  --output-dir /tmp/trading-backtests/candidate/replay-validation/decisions
```

The existing fill-based execution attribution remains available. A separate
`planned_strategy` result values the replayed order plan at scheduled benchmark
prices, so partial fills cannot silently change the modeled allocation. Its P&L is
gross; confirmed broker fees and equity residuals remain separate. Actual-time
forensic reports do not substitute delayed fills for the scheduled strategy.
Session `benchmark_paths.entry_nbbo` preserves historical entry clocks; explicit
`--nbbo-path` overrides it.

Historical migration is finite and backed up. It labels old sessions
`liquidity-fixed` and reconstructs only what their archived evidence supports:

```bash
python -m trading_rl.overnight.migrate_replay \
  --work-dir /data/ppv1/live --backup-dir /tmp/replay-backup
# Add --apply after inspecting the proposed changes.
```

The command preserves fills, account snapshots, budgets and statuses; repeated
runs are idempotent. Historical holdings/open orders were not recorded, so their
absence is marked unknown. Their recorded baskets and sizing are verified, rather
than retroactively applying trend/volatility targeting. The August 27 ranking had
been overwritten by the following session; its reconstruction from entry-day bars
and the unique formula matching its ordered basket is explicitly labeled.

## Parameter variants

Use `--strategy` for distinct named policies and `--strategy-config` for validated
JSON parameter overrides. Avoid a new strategy name for each numeric setting.
For example, save this as `research/variants/vol25.json` when starting a new
experiment:

```json
{"volatility_target": 0.25, "max_exposure": 2.0}
```

```bash
trading-backtest --strategy liquidity-trend-vol --strategy-config research/variants/vol25.json \
  --since 2026-01-01 --end-date 2026-09-18 --budget 10000 \
  --output-dir /tmp/trading-backtests/candidate/liquidity-trend-vol/vol25
```

Unknown keys and invalid values fail. All defaults are captured in the result.
Other policy fields are `volatility_window`, `warmup_exposure`, `trend_window`,
`weak_trend_multiplier`, and `risk_history_start`. Selection, price sources,
execution times, financing and costs retain their existing CLI flags.

## Recommended experiment framework

Keep this simulator as the accounting authority. Future experiment source and
versioned specifications belong in `research/`; results belong in `/tmp` by
default. The older `trading-experiments` command has its own historical simulator
and is not the runner for `liquidity-trend-vol`. New comparisons should call
`run_backtest` with the same immutable input arrays for every variant.

A finite batch runner should:

1. Expand a declared coarse parameter grid into immutable run specifications.
   Separate policy parameters, execution assumptions, data snapshot and evaluation
   windows. Hash the complete specification, source version and data snapshot to
   identify a run. Use separate output directories and record failed runs too.
2. Load a frozen data panel once, cache ranking features by their dependencies,
   and reuse it across variants. Cache keys must include data and universe
   fingerprints. Parallelize independent runs with bounded workers and read-only
   data; avoid repeatedly scanning raw bars or sharing mutable strategy state.
3. Use identical evaluation dates, warmup rules, prices, costs and universe
   conventions. Reject incomplete runs rather than removing favorable subsets of
   dates. Compare paired daily returns, calendar CAGR, minute and session drawdown,
   exposure, borrowing, turnover and results by year and market regime.
4. Declare chronological train/validation/test windows before selection. For
   repeated selection use rolling or expanding walk-forward evaluation, with
   trades overlapping split boundaries excluded from selection. Historical data
   may warm indicators; future validation returns may not choose parameters.
   The already explored 2023–2026 sample is not a new untouched holdout.
5. Retain every attempted variant. Use coarse sensitivity checks, higher costs,
   alternative fills, financing stress and block-bootstrap uncertainty on paired
   returns. Examine stable parameter neighborhoods rather than selecting one
   narrow peak. Repeated searches themselves increase overfitting risk; see
   [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

Start with run directories plus an aggregate CSV/Parquet index. When searching
many runs or coordinating workers becomes useful, add
[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking) for parameters,
metrics and artifacts. A local database can precede a shared tracking service.
MLflow tracks results; this backtester remains responsible for financial
accounting and causal validation. No tracking dependency or service is required
by this integration.
