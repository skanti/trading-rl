# MLP PPO trading policy

This directory trains an autoregressive PPO policy that uses the most recent
`N` ticks and chooses a target position on every market tick. The default
window is 4,096 ticks; change both resolved `window_size` values by setting
`model.actor.window_size=8192` in the config for an 8,192-tick model.

## Action semantics

The categorical action is a target position, so opening, closing, reversing,
and resizing are all well-defined:

| Action index | Target | Meaning |
|---:|---:|---|
| 0–4 | -5…-1 | short, size 5…1 |
| 5 | 0 | close / stay flat |
| 6–10 | +1…+5 | long, size 1…5 |

Changing the target incurs proportional turnover cost. A configurable squared
inventory penalty makes the five sizes express risk-adjusted conviction rather
than always rewarding maximum leverage. Every sampled rollout
is liquidated at its final regular-session price, including the corresponding
cost, and never carries an overnight position.

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
11 policy logits or one value. No transformer is used by the training path.
Real-market training uses an annealed target-entropy floor so noisy early
rewards cannot collapse the 11-way policy before it explores long, short,
close, and sizing decisions. The default target decreases from 1.5 to 0.2,
allowing decisive execution after broad early exploration.

## Verify on toy data

The toy market randomizes absolute price over two orders of magnitude and
randomly generates rising or falling episodes followed by a flat regime. Its
known behavior is long/short size 5 while the signal is active, then close when
the signal disappears.

```bash
python -m unittest discover -s tests -v
python toy_train.py --updates 400 --device cpu
```

The second command fails unless held-out direction accuracy reaches 95%, close
accuracy reaches 90%, mean active bet size reaches 4, and reward is positive.
It prints the verification summary without writing generated files.

## Train on market data

Input `.npy` files must contain at least `[seconds, price_mills, volume]`. Build
the day/index CSV (including the inclusive regular-session `eod_idx`) once,
then configure `DATA_DIR`, `EXP_DIR`, and the paths in [main.yaml](main.yaml):

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
`max_drawdown`, and mean target-position changes in `trades` for every logged
batch. Backtest JSONs report `return`, `total_return`, `profit_factor`,
`maximum_cumulative_drawdown`, `mean_trades`, and `total_trades`. Profit factor
uses positive versus negative net timestep P&L after transaction and risk
costs; a trade is any change in target position, including a resize or
reversal.

Backtest a checkpoint over complete held-out sessions (positions and action
history persist from the open until mandatory end-of-day liquidation):

```bash
python evaluate.py --config_path main.yaml --max_days 100 \
  --date_from 2025-12-01 --date_to 2025-12-05
```

Use one date range for checkpoint selection and keep a later range untouched
for the final test. For the bundled snapshot, December 1–5 supplies 50
validation symbol-days and December 8–9 supplies 20 final-test symbol-days.

## Trained checkpoint

The selected 4k-window model is
`$EXP_DIR/trading/train-rl-mlp-4k-full-session/cp-0003000.ckpt`. It was chosen
only from the December 1–5 selection split, where its mean daily net reward was
`+0.001623`, annualized reward Sharpe was `4.53`, and maximum cumulative
drawdown was `0.0232` after the configured costs.

The subsequently opened December 8–9 final split was mixed: mean daily net
reward `-0.000312`, median `+0.000808`, 55% positive symbol-days, and maximum
cumulative drawdown `0.0278`. Gross reward remained positive, but transaction
and risk costs made mean net reward negative. These are research backtests over
a small temporal holdout, not evidence of live-trading profitability.
