# MLP PPO trading policy

This directory trains an autoregressive PPO policy that uses the most recent
`N` ticks and issues one inventory command on every market tick. The default
window is 4,096 ticks; change both resolved `window_size` values by setting
`model.actor.window_size=8192` in the config for an 8,192-tick model.

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

## Policy input

Each tick contains five features:

1. log price relative to the newest price in the window;
2. one-tick log return;
3. window-normalized `log(1 + volume)`;
4. regular-session progress from -1 to +1;
5. the normalized position held over that tick's preceding interval.

Absolute price and symbol identity are excluded. During rollout, a selected
position is written into the next observation window. Consequently later
decisions condition on the actual preceding actions instead of evaluating all
ticks independently.

The actor and critic are separate Spider-style MLPs: the complete
`N × 5` window is flattened, passed through ReLU hidden layers, and mapped to
3 policy logits or one value. No transformer is used by the training path.
Real-market training uses standard clipped PPO with a small fixed entropy
bonus. There is no target-entropy floor or entropy schedule.

## Verify on toy data

The online toy provider samples three or four random log-price anchors, fits a
quadratic or cubic Bezier curve through them, samples a full price path from the
curve, and adds IID innovations to its log returns before integrating them into
price. This avoids the predictable mean reversion produced by independent
price-level noise. Absolute price is independently randomized over two orders
of magnitude. Training batches are generated in memory and are never written
to disk; evaluation uses a fixed seed and a held-out batch.

The normal trainer selects this provider when `data.use_toy: true`. Set it to
`false` to load the configured market-day files instead. Toy mode uses the same
configured window, rollout, model, PPO loop, metrics, and checkpoint path; only
the batch source and market-hours validation change. Give toy and market runs
different `general.experiment_name` values; the trainer rejects resuming a
checkpoint whose recorded data source does not match the flag.

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
the day/index CSV (including the inclusive regular-session `eod_idx`) once,
then set `data.use_toy: false`, choose a new experiment name, and configure
`DATA_DIR`, `EXP_DIR`, and the paths in [main.yaml](main.yaml):

```bash
python prepare_days.py \
  --data_dir "$DATA_DIR/ppv1/updates/ppv1_latest" \
  --output_path "$DATA_DIR/ppv1/updates/ppv1_latest_days_4k.csv" \
  --window_size 4096 --rollout_size 390
python train.py --config_path main.yaml
```

The configured 390-step rollout spans the complete 09:30–16:00 US/Eastern
session, so positions and autoregressive action history persist for the entire
day before mandatory close liquidation. Incomplete market days are excluded.
Checkpoints contain actor, critic, optimizer, scheduler, resolved config, and
feature ordering, which is enough to resume training or construct the same
policy for greedy inference.

Training CSVs track net `return`, timestep `profit_factor`, worst
`max_drawdown`, and mean position changes in `trades` for every logged
batch. Backtest JSONs report `return`, `total_return`, `profit_factor`,
`maximum_cumulative_drawdown`, `mean_trades`, and `total_trades`. Profit factor
uses positive versus negative net timestep P&L after transaction and risk
costs; a trade is any change in the bounded position. Repeated commands at a
position boundary are therefore not trades.

Backtest a checkpoint over complete held-out sessions (positions and action
history persist from the open until mandatory end-of-day liquidation):

```bash
python evaluate.py --config_path main.yaml --max_days 100 \
  --date_from 2025-12-01 --date_to 2025-12-05
```

Use one date range for checkpoint selection and keep a later range untouched
for the final test. For the bundled snapshot, December 1–5 supplies 50
validation symbol-days and December 8–9 supplies 20 final-test symbol-days.

## Checkpoint compatibility

Checkpoints from the former 11-action sized-position policy are intentionally
incompatible with this 3-action policy head. The default experiment name is
new so training starts from scratch instead of attempting to resume one of
those checkpoints.
