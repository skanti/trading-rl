# Autoregressive PPO trading policies

This directory trains shifted-window MLP trading policies with PPO over three
inventory commands. `main.yaml` selects the intraweek policy: one rollout is a
complete Monday--Friday week of 10-minute bars, so inventory persists across the
four weeknights and is liquidated once, at Friday's close. A second policy,
described at the end, trades two time-matched symbols at once.

## Action semantics

The categorical action is a unit-free inventory command:

| Action index | Command | Position transition |
|---:|---|---|
| 0 | buy | short → flat → long |
| 1 | nothing | retain the current position |
| 2 | sell | long → flat → short |

Positions are bounded to `{-1, 0, +1}`. A repeated `buy` while already long or
`sell` while already short is a valid no-op: it adds no exposure, turnover, or
transaction cost. Reversing takes two decisions (close, then open the other
side). A configurable squared inventory penalty lets the policy prefer staying
flat. Every sampled rollout is liquidated at its final price, including the
corresponding cost, so nothing survives the end of the rollout — Friday's 16:00
close for the intraweek policy, 16:00 the same day for the intraday one.

## Policy input and model

Only active-asset and SPY prices are market inputs. SPY is observation-only: it
is present in the day index but `WeekReferenceDataset` excludes it from tradable
targets, so reward, turnover, risk, and liquidation come from the selected asset
alone. Both instruments are completed on the same expected grid and their
timestamps are then checked for exact equality. Volume, absolute price, and
symbol identity are excluded.

The first decision uses the window ending at Monday 09:30. After each sampled
command, the resulting inventory and a one-hot copy of that command are written
at the newest position of the next one-bar-shifted window. Historical state
channels are zero, so the context contains prices rather than fabricated no-op
actions.

Each window position carries eight features:

| # | feature |
|---:|---|
| 1--2 | relative log price, asset and SPY |
| 3--4 | one-bar log return, asset and SPY |
| 5 | active-asset inventory |
| 6--8 | previous-command indicators (buy / nothing / sell) |

The fixed flattened input coordinates already encode oldest-to-newest window
position, so the MLP adds no transformer-style positional encoding. Scalars that
carry a single number are appended once after the flattened window rather than
broadcast across every row, which would spend hundreds of inputs to say one
thing.

The actor and critic are 512 wide and four deep over `960 * 8 + 4` inputs,
9,447,428 parameters in total.

## Intraweek rollout

### Grid

| | intraweek (`main.yaml`) | intraday predecessor |
|---|---:|---:|
| bar | 10 min | 1 min |
| context | 960 ticks (10 extended sessions) | 9,600 ticks (10 extended sessions) |
| MLP window | 960 ticks (10 days) | 4,096 ticks (~4.3 days) |
| rollout | 199 decisions (five sessions) | 390 decisions (one session) |
| liquidation | Friday 16:00 only | every 16:00 |

A regular session contributes 40 grid points at ten-minute spacing, 09:30
through 16:00 inclusive, so a week holds 200 points and 199 of them are
decisions. The last point is Friday's close: it prices the final interval and
liquidates, and is never itself a decision. Each 04:00--19:59 extended session
contributes 96 context points, so the ten-day context is 960 ticks and the MLP
window spans the whole context rather than a slice of it.

### Bars are subsampled, not averaged

A ten-minute bar here is the last trade price at or before the ten-minute
boundary, produced by the same forward-fill the one-minute loader uses; volume
is the sum over the ten minutes ending at that boundary. Averaging the ten
constituent minute prices was rejected for two reasons. The average of minutes
`t..t+9` is not known at `t`, so a policy marked to market against it is reading
its own decision interval; and no order executes at the mean of a ten-minute
window, so the resulting P&L is not attainable. Subsampling keeps every quoted
price one the rollout could actually have traded at.

### The weekend is structural, not penalized

Nothing in the reward discourages a weekend position, because the grid makes one
unreachable. The 200 week ticks are regular-session ticks on five weekdays, the
final one is Friday 16:00, and `market_rewards` liquidates whatever is held
after the last interval — the same mechanism that closes the intraday policy at
16:00, applied once per week instead of once per day.
`week.validate_week_hours` re-checks every sampled batch: no tick may fall on a
Saturday or Sunday, every tick must sit on the ten-minute grid inside 09:30
through 16:00, the first must be a Monday 09:30 and the last a Friday 16:00, and
the five sessions must run Monday through Friday in order. It runs on training
and validation batches alike while `data.enforce_market_hours` is set.

The four weeknight gaps are real held intervals and are priced as such: the
09:30 print after an overnight hold is the next price, gap included. Metrics add
`overnight_fraction` (share of the four 16:00 decisions that carry inventory)
and `overnight_holds` (their mean count per week).

### Anchoring the price at Monday 09:30

The per-tick price channels stay anchored to the newest tick in the window. That
is what keeps the input stationary: the newest element is always zero and the
rest are log distances from it, so a Friday window and a Monday window are on
the same scale and the same weights read both. Re-anchoring all 960 window rows
to Monday 09:30 would break that. Coordinate `j` of the rolling window is always
a return over exactly `959 - j` bars ending now; against a fixed anchor its lag
depends on where the decision sits in the week, and the first layer has one
weight row per coordinate to encode one meaning. Measured on real weeks, the
newest coordinate under a Monday anchor is identically zero at Monday's open and
has a cross-sample std of 8.3 by Friday, a distribution shift inside a single
rollout. Overall input magnitude is unaffected — it is the per-coordinate
meaning that degrades.

Re-anchoring is also cheap. Consecutive windows differ by a pure constant equal
to minus the latest return, verified constant across all 959 shared ticks: a
median 0.135 against a median within-window spread of 2.36, and the return
channels do not move at all because a constant cancels in a first difference.
An MLP recomputes from scratch every step, so nothing is being invalidated; the
same choice would have broken the retired GPT path, whose KV cache needs a
tick's encoding to stay fixed once written.

Anchoring on the first context bar is worse still. Extended-hours coverage is
thin — a median 623 of 960 minute bars are missing on tradable symbol-days, and
97% of days are more than half missing — so that print is usually a stale carry
forward, and its error would enter all 7,684 inputs.

Week-to-date return is still worth stating explicitly, because it prices the
decision the policy is actually being asked to make, so it is supplied as two of
the four appended scalars:

| scalar | meaning |
|---|---|
| `time_to_week_close` | 1.0 at Monday 09:30, falling to 1/199 at the last decision |
| `time_to_day_close` | 1.0 at each 09:30, reaching 0.0 at each 16:00 |
| `asset_week_to_date_return` | `100 * log(P_t / P_monday_0930)` |
| `spy_week_to_date_return` | the same for SPY |

`time_to_day_close` is the one clock the window genuinely cannot supply: it hits
zero exactly on the decision that chooses whether to carry inventory overnight,
which is the only decision in the day whose next interval is seventeen hours
long. The intraday policy needed no such marker because it had one deadline.

### Weeks are all-or-nothing

A week is offered only when the exchange calendar holds five sessions, all five
are tradable for that symbol, they are contiguous in the symbol's own session
sequence, and ten earlier sessions exist to fill the context. Holiday weeks are
dropped rather than padded, because a shorter rollout would not share the fixed
policy input shape, and `scripts/split.py` already excludes scheduled 13:00
half-days from the calendar, so the weeks containing them are four-session weeks
and are dropped too. On the bundled 2026-08-22 snapshot that leaves 454 of 555
calendar weeks, and 769,027 training symbol-weeks against 7,772 validation
symbol-weeks.

Splitting is by whole week: a training week must end before `date_val`, and a
validation week must start on or after it, so no rollout straddles the boundary.
With the snapshot's last session on Friday 2026-08-21, the trailing four weeks
begin Saturday 2026-07-25 and the first validation week opens Monday 2026-07-27.

`model.gamma` is 0.999 rather than the intraday 0.99. At one decision per ten
minutes the smaller value discounts Friday's close to about 1e-9 seen from
Monday, which would make the multi-day holding period the experiment exists to
test invisible to the advantage estimate.

## Train on market data

Input `.npy` files use the eight-column OHLCV schema:
`[seconds, open_mills, high_mills, low_mills, close_mills, volume, trades, vwap_mills]`.
Loaders read volume from its named schema position and continue using open prices.
Legacy three/four-column files are rejected; re-download their full history into
a new store because the additional fields cannot be recovered locally. Build
the day/index CSV (including expected extended- and regular-session times) once,
then configure `DATA_DIR`, `EXP_DIR`, and the paths in [main.yaml](main.yaml):

```bash
python ../scripts/split.py \
  --sample_ids ../data/tickers_all.txt \
  --npy_dir "$DATA_DIR/ppv1/updates/bars_1min_2022-01-01" \
  --out_path "$DATA_DIR/ppv1/updates/full_2026-08-22_10d.csv" \
  --rollout_size 390
python train.py --config_path main.yaml
```

`train.py` dispatches on `data.rollout_mode`: `week` runs the intraweek trainer,
`session` runs the intraday one. The day index is shared — it is always built at
one-minute resolution, and the week loader subsamples it.

Missing bars in both the historical extended sessions and the traded regular
sessions are completed in memory by carrying the last price forward. Weekends,
holidays, and half days are absent from the session metadata rather than being
invented. Checkpoints contain the actor, critic, optimizer, scheduler, resolved
config, and input ordering, which is enough to resume training or construct the
same policy for greedy inference.

Training CSVs track net `return`, timestep `profit_factor`, worst
`max_drawdown`, and mean position changes in `trades` for every logged
batch. Backtest JSONs report `return`, `total_return`, `profit_factor`,
`maximum_cumulative_drawdown`, `mean_trades`, and `total_trades`. Profit factor
uses positive versus negative net timestep P&L after transaction and risk
costs; a trade is any change in the bounded position. Repeated commands at a
position boundary are therefore not trades.

The trainer evaluates greedy rollouts on the reserved trailing four weeks every
500 updates. These rows use `stage=1` in `metrics.csv`; training rows use
`stage=0`.

## Verify on toy data

The online toy provider samples three or four random log-price anchors, fits a
quadratic or cubic Bezier curve through them, samples a full price path from the
curve, and adds IID innovations to its log returns before integrating them into
price. This avoids the predictable mean reversion produced by independent
price-level noise. Absolute price is independently randomized over two orders
of magnitude. Training batches are generated in memory and are never written
to disk; evaluation uses a fixed seed and a held-out batch.

The toy provider remains a focused legacy MLP verification harness. The real
asset/SPY path selected by `main.yaml` does not mix synthetic paths into its
training or validation data.

```bash
python -m unittest discover -s tests -v
python toy_train.py --updates 400 --device cpu
```

The second command fails unless held-out position accuracy reaches 85%, active
direction accuracy reaches 90%, and reward is both positive and better than the
untrained policy. It prints the verification summary without writing generated
files. Flat accuracy remains in the report as a diagnostic rather than a gate:
with transaction costs, closing on every isolated low-slope tick is not always
the reward-maximizing behavior.

## Intraday predecessor

`reference_mlp.py` and `reference_mlp_train.py` keep the one-session policy the
intraweek experiment grew out of, reachable with `data.rollout_mode: session`.
It decides once per minute from a 4,096-minute window, appends a single
`time_to_close` scalar that starts at `1.0` at 09:30 and falls to `1/390` at
15:59, and is liquidated at every 16:00. The intraweek code reuses its window
builder and its trailing-validation split, so the two policies differ only in
clock, rollout span, and appended scalars.

## Checkpoint compatibility

Intraweek checkpoints record `feature_names` and `scalar_names` and are refused
by the intraday trainer, and vice versa: the two disagree on `scalar_dim`, so
loading one into the other would silently mismatch the input layout. Checkpoints
from the retired tokenized-GPT and 11-action policies are incompatible with both.


# Relative-value pair policy

A second policy trades two time-matched symbols at once. Every tick it issues
one joint command drawn from a 9-way head, and each leg keeps the single-symbol
inventory semantics, so the reachable inventories span four economically
distinct trades:

| `(position_a, position_b)` | Trade | Signal it exploits |
|---|---|---|
| `(0, 0)` | flat | no usable relationship, or a broken pair |
| `(+1, -1)` / `(-1, +1)` | spread | mean reversion of the beta-neutral residual |
| `(±1, 0)` / `(0, ±1)` | single leg | one leg has moved and the other has not |
| `(+1, +1)` / `(-1, -1)` | directional | the factor the two legs share |

A single 9-way head is used rather than two independent 3-way heads because the
value of a command on one leg depends on the command issued to the other:
"buy A" is only correct conditional on "sell B", and a factorized policy cannot
represent that.

## Observation

Eleven channels per tick, in the interleaved layout
`[price_a, price_b, actions, ...]`:

1. log price of A relative to the newest tick in the window;
2. the same for B;
3. one-tick log return of A;
4. the same for B;
5. window-normalized `log(1 + volume)` for A;
6. the same for B;
7. `spread_z`, the beta-neutral residual whitened over the window;
8. the one-tick change in `spread_z`;
9. regular-session progress from -1 to +1;
10. the normalized position held in A over the preceding interval;
11. the same for B.

Four window-level statistics are appended once per decision instead of being
broadcast across all 4,096 ticks: the hedge ratio, the leg-to-leg return
correlation, the lag-one autocorrelation of the residual, and the residual's
scale. Broadcasting them as extra channels would spend over sixteen thousand
inputs to carry four numbers.

The hedge ratio is the cointegrating regression of B's log price on A's, not a
regression of their returns. Regressing returns is badly biased here: the
spread perturbs both legs tick by tick, so it acts as measurement error on the
regressor and attenuates the slope. On the bundled toy that bias leaves roughly
a fifth of the common factor inside the "residual", which is more contamination
than the spread it is meant to isolate. The level regression recovers the
generating sensitivity to within about 0.04 and produces a residual that
correlates 0.92 with the true spread.

Every statistic is computed inside the observation window only. Absolute price
and symbol identity stay out, as in the single-symbol policy.

## Reward

Both legs are marked to market and charged proportional turnover, then two risk
terms are applied:

- `net_risk_penalty` charges the squared factor exposure
  `exposure_a + hedge_ratio * exposure_b`. This is near zero for a hedged
  spread and largest for a doubled-up directional bet, which is what prices
  market risk rather than merely capital.
- `gross_risk_penalty` charges deployed capital so a leg is held only when it
  earns its keep.

Both legs are liquidated at the final regular-session price. Note that a
round-trip pair trade pays four units of transaction cost, so at the configured
`1e-4` the spread has to clear 4bp before it is worth taking at all.

## Toy verification

`OnlinePairToyProvider` generates `log P_A = c + s/2` and
`log P_B = beta * c - s/2`. The common factor `c` is the same Bezier path with
integrated return noise the single-symbol provider uses, so it stays
unpredictable and leg A remains the marginal process the single-symbol policy
was trained on. The spread `s` is an Ornstein-Uhlenbeck process, and it is the
only forecastable component: a policy can only beat a single-leg trader by
reading the relationship between the legs.

The spread is parameterized by its per-step volatility rather than its
stationary width, so every pair shows the same leg correlation whatever its
half-life. A configurable fraction of pairs draw a random-walk spread instead,
which is untradeable and must be left alone. Because both regimes share a
per-step volatility, they can only be separated by the autocorrelation of the
residual — never by its scale.

```bash
python -m unittest discover -s tests -v
python pair_toy_train.py --updates 600 --device cuda:0
```

The second command gates on three things: the policy improves its reward, it
takes the correct side of a stretched residual at least 60% of the time, and it
beats the textbook z-score rule on the same residual. The gate damps the shared
trend, because over its short rollout a Bezier curve is close to a straight line
and a policy could score well on the directional trade without ever looking at
the spread.

Regime discrimination - holding less spread exposure on broken pairs than on
mean-reverting ones - is reported but deliberately not gated here, in the same
way `toy_train.py` reports flat accuracy without gating it. Separating an
Ornstein-Uhlenbeck spread from a random walk means resolving a difference in
lag-one autocorrelation of roughly 0.06, and over this gate's 128-tick window
the standard error of that estimate is about 0.03. The statistic is barely
resolvable at that scale whatever the policy does, so it is checked at the full
4,096-tick window by `pair_evaluate.py` instead.

## Train and evaluate

```bash
python pair_train.py --config_path pair.yaml
python pair_evaluate.py --config_path pair.yaml \
  --single_checkpoint "$EXP_DIR/trading/train-rl-mlp-4k-3action-bezier-integrated-noise/cp-0018500.ckpt" \
  --batch_size 32 --batches 8
```

`pair_evaluate.py` scores every strategy on identical price paths with the same
reward, so the numbers differ only in how the two legs are used:

- `pair_policy` — the learned joint policy;
- `baseline_two_legs` — the single-symbol policy run independently on each leg.
  This is the benchmark that matters, because it controls for the pair policy
  simply deploying twice the capital;
- `baseline_single_leg` — the same policy on leg A alone, which reproduces the
  published single-symbol KPI;
- `zscore_rule` — the textbook entry/exit rule on the same residual the policy
  observes, so any excess is what was learned rather than engineered;
- `oracle_myopic` — an upper reference that reads the noise-free drift of both
  legs. It is not attainable.

Every reachable reference is step-limited, because one command moves inventory
by one step and a reversal costs two ticks; a benchmark allowed to flip sign in
a single tick would be competing with an ability the policy does not have.

## Market pairs

```bash
python prepare_pairs.py \
  --data_dir "$DATA_DIR/ppv1/updates/bars_1min_2022-01-01" \
  --output_path "$DATA_DIR/ppv1/updates/ppv1_latest_pairs_4k.csv" \
  --window_size 4096 --date_from 2025-06-01
python pair_train.py --config_path pair.yaml data.use_toy=false \
  general.experiment_name=train-rl-pair-4k-market
```

Two things differ from the single-symbol pipeline.

Symbols are not tick-aligned — coverage of extended hours varies from name to
name — so the two legs are joined on their timestamps rather than on their row
indices, and only ticks present in both survive. Pair rows therefore record
seconds, which mean the same thing to both legs.

Pairs are ranked by correlation over a trailing window that ends before the
traded session opens. Ranking with data from the day being traded is the
standard way to leak the answer into a pairs backtest. The output deliberately
mixes strongly co-moving pairs with randomly drawn ones: the policy never sees
symbol identity, so the only way it can learn to stand aside on an unrelated
pair is to have been shown unrelated pairs while training.

Symbols are partitioned into a training universe and a held-out universe by
`--holdout_stride` *before* any pair is formed, and every pair is drawn inside
one universe. Holding symbols out after pairing does not work, because a pair
shares each of its legs with many other pairs and a name excluded from the
validation pairs still reaches training inside a different pairing. Selecting
within a partition also keeps both splits populated: taking only those pairs
whose *both* legs happen to be held out leaves about one in sixty-four of the
candidates. On the bundled snapshot the split gives 2,795 training pair-days
over 125 symbols and 487 validation pair-days over 24 symbols, with no symbol
and no date in common.

Note that `data.date_val` must fall inside the range `prepare_pairs.py` indexed,
or one of the two splits comes back empty.

### What the relative signal is worth on real pairs

Running only the fixed z-score rule over 400 held-out pair-days from the bundled
snapshot — no trained policy involved, so this measures the signal rather than
any model — separates cleanly by how the pair was selected:

| Held-out pairs | n | median return corr | best z-rule return | profit factor |
|---|---:|---:|---:|---:|
| correlated | 102 | 0.40 | +0.00285 | 1.26 |
| random | 298 | 0.03 | -0.00202 | 0.79 |

The relationship is worth something on pairs that actually co-move and is worth
less than nothing on pairs that do not, at every entry threshold tried. That is
the case for training on a mixture: a policy that cannot tell the two apart
gives back on the second group more than it makes on the first, and the pooled
number over all 400 pair-days is negative for exactly that reason.

Two caveats. With 102 correlated pair-days the t-statistic on that positive
number is only 0.78, so this is directional evidence and not a demonstrated
edge. And the median residual autocorrelation on real pairs is about 0.996 over
a 4,096-minute window, much closer to a random walk than the bundled toy's
Ornstein-Uhlenbeck spread: real intraday residuals revert far more weakly than
the toy assumes, so toy results should not be read as a forecast of market
results.
