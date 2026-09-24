# Experimental overnight strategies

Research source stays here. Generated results go under
`/tmp/trading-backtests/candidate/`. The promoted live and backtest default is `liquidity-momentum-focus`; its
implementation lives in shared production modules. Other experimental names
remain unavailable to the live daemon and reconciliation.

The newer [momentum-focus study](MOMENTUM_FOCUS.md) simplifies the blend to one
three-stock basket. Its user-selected defaults are 10-session momentum, a
100-session SPY trend and 20 closed baskets for volatility. Through September 23,
2026 these settings reached 119.38% calendar CAGR, 2.351 Sharpe and 21.38%
minute-mark drawdown with the current 0.1 weak-trend multiplier; they miss the original Sharpe and drawdown targets.
Select it explicitly with `--strategy liquidity-momentum-focus` in live or backtests. The
earlier optimized 8/150/40 configuration remains an explicit research variant.

## Selected experimental preset

```sh
trading-backtest --strategy liquidity-momentum-blend \
  --since 2023-01-01 --end-date 2026-09-18 --budget 10000
```

`liquidity-momentum-blend` meets the numerical historical target in the tested
period: **109.74% calendar CAGR and 19.40% minute-mark drawdown**. It was chosen
using reused 2023–2026 history after a substantial parameter search. This is an
**in-sample experimental result, not an independent validation or a forecast**.
The normal backtester default and live policy are unchanged.

The preset:

1. Forms the usual twelve-issuer Nasdaq liquidity shortlist, using the shared
   turnover-stability scorer with a ten-session EMA.
2. Ranks that shortlist separately by five-session and ten-session stock price
   momentum, using daily closes through the previous session only.
3. Allocates half the portfolio to each horizon's top four, equal-weighted within
   each half. An overlap receives 25% of the deployed basket; a name appearing in
   just one half receives 12.5%. The union contains four to eight names.
4. Targets 35% annualized volatility using the last twenty completed **weighted,
   unlevered modeled overnight basket returns**. Exposure is capped at 2x and is
   1x during volatility warmup or zero measured volatility.
5. Holds cash when the previous SPY close is below its previous 100-session moving
   average. Cash decisions still observe the modeled basket for future risk sizing.

The strategy uses the same execution, financing and compounding engine as the
shipped strategy: 15:45 ET SIP ask entries, next primary-opening-auction exits,
fractional quantities, and 6.75% annual margin interest on actual borrowing, charged
by calendar days/360. No short positions or weekday exclusions are used.

January 3, 2023–September 18, 2026, 930 intervals and $10,000 starting equity:

| Model | Calendar CAGR | Minute-mark drawdown | Ending equity |
| --- | ---: | ---: | ---: |
| Shipped liquidity-trend-vol | 77.42% | 19.79% | $83,760 |
| Experimental momentum blend | 109.74% | 19.40% | $155,776 |
| Blend, extra 0.25 bp per side | 106.42% | 19.58% | $146,827 |
| Blend, extra 1 bp per side | 96.77% | 20.13% | $122,942 |

Costs in the last two rows are **additional** to the observed ask/auction prices.
The result is sensitive to small execution costs. Minute-open marks miss intraminute
extremes and carry the last observed price when prints are absent. The existing
universe uses current security-master classification, not a complete historical
survivorship-free universe. Historical broker margin eligibility, actual buying
power and market impact are not reproduced by the simulator.

The current default security master and the archived study master produce identical
portfolios for this preset. Exact numerical reports, source/input fingerprints,
yearly returns and a comparison chart are under
`/tmp/trading-backtests/candidate/regime-optimization/final-report/`; detailed CLI
reports are in `blend-default-inputs/`, `liquidity-momentum-blend/`, and the cost
stress directories. Copy generated results elsewhere if they need durable storage.

## Strategy plugins and variants

Existing experimental families remain available for comparisons:

- `liquidity-regime-vol`: SPY trend, volatility sizing, optional trend buffer,
  momentum confirmation and excluded entry weekday. Its default six-stock preset
  failed the numerical target and is not an upgrade.
- `liquidity-momentum-vol`: configurable stock momentum allocations inside the
  liquidity shortlist. Its original 60-session rank-weighted preset reached
  93.75% CAGR / 27.81% drawdown and failed the requested criteria.
- `liquidity-momentum-blend`: the selected preset described above.

Use `--strategy-config config.json` for strict JSON overrides. For example:

```json
{
  "allocation_windows": [5, 10],
  "allocation_count": 4,
  "trend_window": 100,
  "volatility_window": 20,
  "volatility_target": 0.325,
  "weak_trend_multiplier": 0
}
```

`allocation_windows` averages separate horizon allocations. When null, the single
`allocation_window` applies. A positive `allocation_count` chooses equal-weighted
momentum leaders within each horizon; zero uses weights proportional to momentum
ranks across all shortlisted names. Optional `momentum_window` is a **SPY** trend
confirmation, separate from stock allocation lookbacks. `excluded_entry_weekday`
is null by default; 0 means Monday, 4 means Friday. Missing required prices or
momentum inputs fail the run instead of silently replacing names.

New plugins export `STRATEGY`, an experimental `StrategySpec`, from an importable
module. Register it with `--strategy-plugin research.my_variant`, then select its
name with `--strategy`. Builtin names cannot be replaced by external plugins.
The spec supplies config/policy classes, a label, default shortlist/EMA settings,
and whether daily stock closes are needed. Configs validate shared risk fields
and provide `as_dict()`. Policies accept `(config, spy_marks)` and implement
`exposure(row)` and `observe(unscaled_return)`.

Optional `prepare(PolicyContext)` receives dates, symbols and daily closes.
Optional `weights(row, selected_indices)` returns finite non-negative weights
summing to at most one; unused allocation stays cash. Shared sizing and risk-return
calculation consume the same weights. The engine enforces the exposure cap and
rejects negative exposure. Long/short simulation needs explicit execution support;
negative exposure is never silently priced as a long trade.

Plugins must enforce lagging and have causality tests: receiving historical arrays
is not a sandbox against lookahead. Source fingerprints include inherited plugin
implementations and config classes. Experiments never enter the approved live
strategy registry. The repository packages `research` with the CLI; after adding
it to an existing checkout, refresh its editable installation (`uv pip install
--no-deps -e .`, using the intended virtual environment).

## Research protocol and reproduction

The first screens used 2023–2024 entry sessions for training and 2025 entry sessions
for validation, excluding 2026 entries from selection. The selected rules then
failed the full-period return/drawdown combination. Follow-up work reused 2026,
so it is **no longer an untouched holdout**. The final preset was chosen from an
explicit full-period audit of a frozen 240-configuration momentum grid. The initial
rule requiring every development period to beat the shipped policy rejected this
preset: its 2025 result was slightly lower even though its overall performance
improved. Final selection is recorded as exploratory/in-sample, not retroactively
claimed to have passed that rule. Neighboring settings also met the numerical
full-period target, but they are not independent evidence after this search.

`frontier_screen` and `momentum_frontier` use conservative minute-drawdown bounds
from common-engine unit portfolios to avoid choosing on exit-only drawdown. Bounds
can overstate risk; every reported finalist is audited with the exact minute path.
Tests verify the bound covers actual paths, including financing and peak/trough
ordering. The final CLI's audit reconciles all liquidation endpoints within 1e-12
of starting session equity. Common-engine weighted observations match the research
screen across all 930 intervals.

Prepare one reusable panel through the production CLI's loader:

```sh
python -m research.prepare_inputs \
  --output-dir /tmp/trading-backtests/candidate/regime-optimization \
  --since 2023-01-01 --end-date 2026-09-18 --budget 10000
python -m research.allocation_screen \
  --input-dir /tmp/trading-backtests/candidate/regime-optimization \
  --output-dir /tmp/trading-backtests/candidate/regime-optimization/allocation
python -m research.momentum_frontier \
  --input-dir /tmp/trading-backtests/candidate/regime-optimization \
  --output-dir /tmp/trading-backtests/candidate/regime-optimization/momentum-frontier
python -m research.audit_momentum_frontier \
  --input-dir /tmp/trading-backtests/candidate/regime-optimization \
  --frontier-dir /tmp/trading-backtests/candidate/regime-optimization/momentum-frontier
```

The study also contains `regime_screen`, `basket_screen`, `calendar_screen`,
`short_diagnostic`, and report/evaluation runners. Their grids and periods are
explicit in source; all generated tables remain in `/tmp`. Use the study's exact
data snapshots and source fingerprints for numerical reproduction. A changed
input dataset or changed selection rule is a new experiment. Neither paper profits
nor passing software tests establish future trading performance; fresh prospective
validation remains necessary before any live promotion.

## Shorting and rejected hypotheses

Short exposures were considered first as optimistic sign-flip diagnostics, then
using actual SIP bid entries, whole-share sizing, auction exits and an assumed 3%
annual stock-borrow fee. The best scenario within the development exit-drawdown
limit produced about 83.8% development CAGR, versus 83.7% for a comparable long/cash
scenario. This did not justify adding shorts to the chosen strategy. Historical
borrow availability, dividend liabilities and historical short margin eligibility
are missing, so these remain scenarios, not executable-strategy validation.
See [Alpaca margin/borrow documentation](https://docs.alpaca.markets/us/docs/margin-and-short-selling)
and [fractional-order restrictions](https://docs.alpaca.markets/us/docs/fractional-trading).

Other tested ideas included trend windows/buffers, SPY momentum confirmation,
volatility windows, basket size, liquidity smoothing, inverse-volatility allocation,
short-horizon continuation/reversal, calendar exclusions and horizon blends.
The [New York Fed overnight-drift research](https://www.newyorkfed.org/research/staff_reports/sr917)
motivated examining reversal as a hypothesis; it does not validate this stock
strategy. Stock momentum, rather than reversal, produced the selected preset.
