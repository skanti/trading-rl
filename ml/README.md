# Tokenized causal-GPT PPO trading policy

This directory trains an autoregressive PPO policy that preserves the complete
causal sequence and issues one inventory command on every market tick. The default
context is the previous 10 trading days completed onto 04:00--19:59 Eastern
one-minute grids. The 9,600 historical tokens prime the transformer's KV cache
once. The current trading day then grows the sequence one token at a time from
09:30 through 15:59; the 16:00 price supplies the final reward and liquidation.

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
flat. Every sampled rollout is liquidated at its final regular-session price,
including the corresponding cost, and never carries an overnight position.

## Policy input and model

Only active-asset and SPY prices are market inputs. Each price series is first
converted to scale-free log returns. A mu-law quantizer maps each return to one
of 64 levels, and the pair is combined into one token:

`token = asset_level * 64 + spy_level`

That Cartesian product is exactly a 4,096-entry dictionary while retaining one
token per timestamp. The position held over the preceding interval is supplied
through a separate three-state short/flat/long embedding, and the previous
command through a buy/nothing/sell/no-prior-command embedding. This one-tick
shift keeps action history causal. Volume, absolute price, and symbol identity
are excluded.

The policy and value function share a GPT-2-style causal trunk with learned
position embeddings and separate three-logit policy and scalar value heads.
The default 6-layer, 256-wide model has about 8.3M parameters. Rollout uses a KV
cache, so earlier context is never re-tokenized or truncated as the trading day
grows. PPO optimization uses the equivalent full causal forward pass with a
small fixed entropy bonus and no entropy schedule.

## Verify on toy data

The online toy provider samples three or four random log-price anchors, fits a
quadratic or cubic Bezier curve through them, samples a full price path from the
curve, and adds IID innovations to its log returns before integrating them into
price. This avoids the predictable mean reversion produced by independent
price-level noise. Absolute price is independently randomized over two orders
of magnitude. Training batches are generated in memory and are never written
to disk; evaluation uses a fixed seed and a held-out batch.

The toy provider remains a focused legacy MLP verification harness. The default
`main.yaml` path is the tokenized real-market GPT experiment and does not mix
synthetic paths into its training or validation data.

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

## Train on market data

Input `.npy` files must contain at least `[seconds, price_mills, volume]`. Build
the day/index CSV (including expected extended- and regular-session times) once,
then set `data.use_toy: false`, choose a new experiment name, and configure
`DATA_DIR`, `EXP_DIR`, and the paths in [main.yaml](main.yaml):

```bash
python ../scripts/split.py \
  --sample_ids ../data/top500.txt \
  --npy_dir "$DATA_DIR/ppv1/updates/full_2026-08-22" \
  --out_path "$DATA_DIR/ppv1/updates/full_2026-08-22_top500_days_10d.csv" \
  --rollout_size 390
python train.py --config_path main.yaml
```

The configured 390-step rollout spans the complete 09:30–16:00 US/Eastern
session, so positions and autoregressive action history persist for the entire
day before mandatory close liquidation. Missing bars in both the historical
extended sessions and current regular session are completed in memory by
carrying the last price forward and assigning zero volume. Weekends, holidays,
and half days are absent from the session metadata rather than being invented.
Checkpoints contain the shared transformer, optimizer, scheduler, tokenizer,
resolved config, and input ordering, which is enough to resume training or
construct the same policy for greedy inference.

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

## Checkpoint compatibility

Checkpoints from the former MLP and 11-action policies are intentionally
incompatible with the shared GPT actor-critic. The default experiment name is
new so training starts from scratch.

## SPY-reference experiment

`main.yaml` trains the three-action, single-asset policy on real
market data while adding SPY as an observation-only reference. SPY is present
in `top500.txt` and the day index, but `MarketReferenceDataset` explicitly
excludes it from tradable targets. Reward, turnover, risk, and liquidation are
therefore calculated only from the selected asset.

Each tick contains the joint asset/SPY price-return token plus the asset's
position/action state. Both instruments are independently completed on the
same expected minute grid, then their timestamps are checked for exact equality.

The final four calendar weeks are reserved for validation. For this snapshot,
whose last session is 2026-08-21, validation begins on 2026-07-25 (the first
session is Monday, 2026-07-27). Greedy validation is run every 500 updates and
written to `metrics.csv` with `stage=1`; training rows use `stage=0`.

```bash
python train.py --config_path main.yaml
```

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
  --data_dir "$DATA_DIR/ppv1/updates/full_2026-08-22" \
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
