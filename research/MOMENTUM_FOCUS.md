# Single-basket momentum experiment

`liquidity-momentum-focus` simplifies the momentum blend to one equal-weight
three-stock basket. Its user-selected defaults are **10/100/20**: ten trading sessions of
momentum, a 100-session SPY average, and twenty completed baskets for volatility.
These settings favor familiar horizons over the earlier optimized 8/150/40
configuration. They meet the original return target but miss the Sharpe and
drawdown targets. Results remain **in-sample**. At the user's request, this 10/100/20 configuration
is now the live and backtester default; promotion does not add independent validation.

```sh
trading-backtest --strategy liquidity-momentum-focus,liquidity-momentum-blend \
  --since 2023-01-01 --end-date 2026-09-23 --budget 10000
```

The standard comparison prints and saves a table, per-strategy trades and
summaries, minute-mark audits, and a chart including SPY buy-and-hold. The CLI
prints the chart's absolute path.

## Rules

1. Keep the shared top-twelve Nasdaq issuer shortlist: lagged log-dollar-volume
   EMA(10) minus its dispersion, at least 100 observed completed sessions, and
   one share class per company.
2. Rank those twelve by ten-session daily-close momentum, using only closes
   through the previous session. Buy the top three at equal weights. There is
   one basket and one momentum lookback, with no overlapping allocations.
3. When the previous SPY close is below its lagged 100-session SMA, multiply
   volatility-sized exposure by 0.1. This replaces the previous zero-exposure rule.
4. Target 35% annualized volatility using the last twenty completed
   modeled, unlevered basket returns. Cap exposure at 2x; use 1x during volatility
   warmup or zero measured volatility. Cash decisions continue observing the
   modeled basket, and early-close cash intervals contribute zero returns.
5. Enter at 15:45 SIP ask, exit at the next primary opening auction, and compound
   fractional positions. Charge 6.75% annual interest on actual borrowing using
   calendar days/360. Default additional execution cost is zero; the recorded
   ask already includes the entry spread.

The implementation reuses the existing momentum, risk, execution and accounting
code. `allocation_windows` must remain null and `allocation_count` positive;
the focus family always uses one equal-weight basket. Its default universe and
liquidity settings match the blend. The shared implementation is in
`trading_rl/overnight/momentum.py` and is registered for live, backtesting and
reconciliation. The ranking stage saves momentum inputs and modeled basket
history; reconciliation reproduces both traded and zero-exposure cash decisions.

## Results and limits

Requested inception: January 1, 2023; first entry January 3, 2023; final exit
September 23, 2026. Starting equity $10,000. Sharpe uses daily returns including
cash intervals, 252 sessions/year, and a zero cash rate. Drawdown below is the
full minute-open marked path, including financing and auction liquidation.

| Strategy | Calendar CAGR | Sharpe | Minute drawdown | Ending equity |
| --- | ---: | ---: | ---: | ---: |
| Current 10/100/20, N=3, weak multiplier 0.1 | 119.38% | 2.351 | 21.38% | $185,999 |
| Previous 10/100/20, N=3, weak multiplier 0 | 120.39% | 2.365 | 21.12% | $189,196 |
| Previous 10/100/20, N=4 | 114.29% | 2.291 | 21.16% | $170,447 |
| Earlier optimized focus 8/150/40 | 124.34% | 2.524 | 18.36% | $202,137 |
| Momentum blend | 112.27% | 2.340 | 19.40% | $164,527 |
| Optimized 8/150/40 + 0.25 bp/side | 120.75% | 2.477 | 18.39% | $190,340 |
| Optimized 8/150/40 + 1 bp/side | 110.30% | 2.336 | 18.46% | $158,917 |

The current default makes 2,775 stock trades versus 4,319 for the blend, and holds
three names on every traded interval. Use
`research/variants/momentum-focus-8d-spy150-vol40.json` to reproduce the earlier
optimized variant. Its 15.13% exit-only drawdown is smaller than its 18.36%
minute-mark drawdown; the full minute path is used for risk comparisons.

This study screened **4,104 configurations** adaptively over reused 2023–2026
data: 1,296 initial single-basket settings, 1,728 longer/fixed exposure settings,
and 1,080 additional momentum horizons. Only the eight-day, top-four,
150-day-trend, 40-basket-volatility settings passed the exit screen, at 35% and
40% volatility targets. The lower target was frozen before the exact minute
audit to leave more drawdown room. The screen and final engine agree within
1e-12, and the explicit 8/150/40 variant reproduces those audited daily portfolios.

This is **not independent validation**. With the other optimized parameters held
fixed, five-, seven-, and ten-day momentum yield Sharpe 2.378, 2.199, and 2.351.
The eight-day choice is sensitive, and an extra 0.25 bp/side already misses the
2.5 Sharpe target. The 2023 return was much weaker than 2024–2025. Aggregate
targets do not imply each year meets them or that future returns will do so.
Minute opens omit intraminute extremes and carry stale marks. The universe uses
current security-master classification; historical broker buying power and
market impact are not reproduced. These historical results do not establish future performance.

## Reproducing the earlier optimized study

Research source is in this repository; all generated files belong under
`/tmp/trading-backtests/candidate/single-momentum/`. Final metrics, annual returns
and horizon sensitivity for 8/150/40 are in its `final-report/` directory. The comparison
chart is in `comparison/`; exact paths are printed by the commands. Reports
include source, data and calendar fingerprints. Updated market data constitutes
a new experiment, so retain the input snapshots for exact numerical replay.

Prepare and screen the same grids:

```sh
study=/tmp/trading-backtests/candidate/single-momentum
python -m research.prepare_inputs --output-dir "$study" \
  --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-blend
python -m research.single_momentum_screen --input-dir "$study" \
  --output-dir "$study/screen"
python -m research.single_momentum_risk_screen --input-dir "$study" \
  --unit-dir "$study/screen" --output-dir "$study/risk-screen"
python -m research.single_momentum_screen --input-dir "$study" \
  --output-dir "$study/horizon-screen" --windows 4 6 8 12 --counts 2 3 4 5 6
```

Audit the frozen 8/150/40 configuration and stress costs. Explicit configuration
files preserve this historical experiment independently of the new defaults:

```sh
trading-backtest --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-vol --strategy-config research/variants/momentum-focus-8d-spy150-vol40.json \
  --output-dir "$study/focus-audit"
trading-backtest --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-focus --strategy-config research/variants/momentum-focus-8d-spy150-vol40.json \
  --output-dir "$study/comparison/1-liquidity-momentum-focus"
trading-backtest --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-blend --output-dir "$study/comparison/2-liquidity-momentum-blend"
trading-backtest --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-focus --transaction-cost-bps .25 \
  --strategy-config research/variants/momentum-focus-8d-spy150-vol40.json \
  --output-dir "$study/focus-quarter-bp"
trading-backtest --since 2023-01-01 --end-date 2026-09-23 --budget 10000 \
  --strategy liquidity-momentum-focus --transaction-cost-bps 1 \
  --strategy-config research/variants/momentum-focus-8d-spy150-vol40.json \
  --output-dir "$study/focus-one-bp"
python -m research.report_momentum_focus --study-dir "$study"
```

## Live parity verification

`research/verify_momentum_live.py` compares live risk preparation with a frozen
backtest and, when a live archive exists for that date, replays a hypothetical
entry plan and ranking using its broker snapshots. Its client permits only
read-only market-data requests; state and reports stay in the specified temporary
output directory. It never submits orders or changes live session archives.

```sh
python -m research.verify_momentum_live \
  --summary /tmp/trading-backtests/candidate/momentum-live-integration/backtest/summary.json \
  --output-dir /tmp/trading-backtests/candidate/momentum-live-integration \
  --date 2026-09-22
```

The N=1–12 basket-size comparison is reproducible with
`research/compare_momentum_counts.py`; its frozen results are under
`/tmp/trading-backtests/candidate/momentum-basket-size-20260923/`. N=3 is the new
user-selected default. The explicit `momentum-focus-10d-spy100-vol20.json` variant
retains N=4 to reproduce the previous default. Existing live decisions retain their
saved basket size. Changing N requires a daemon restart and fresh ranking/risk
preparation before new entries; an already prepared plan is reused unchanged.

The weak-trend multiplier comparison is reproducible with
`research/compare_weak_trend.py`; results are under
`/tmp/trading-backtests/candidate/weak-trend-multipliers-20260923/`. The current
0.1 setting trades all 925 eligible intervals in that study, versus 795 with zero;
eight early-close intervals still skip entry. Explicit historical variants retain
their original multipliers. The basket-size study also pins its original zero
multiplier. Saved live decisions always replay their archived policy parameters.
