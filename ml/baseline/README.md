# Trading baselines

`overnight_liquidity.py` implements a point-in-time overnight baseline:

1. Sum each stock's regular-session dollar volume for every completed day.
2. Smooth `log1p(dollar_volume)` with a causal EMA.
3. Before each 15:55 entry, rank using the EMA state through the previous
   session only. The current session never contributes to its own rank.
4. Equal-weight the selected stocks, then close them at 09:45 on the next
   trading session.

The log transform and default 20-session EMA keep earnings, index-rebalance,
and news-related volume spikes from dominating the ranking. Use `--ema-span 1`
to rank strictly by the previous completed session without smoothing.

The first run scans the necessary minute-file slices in parallel and writes a
date-by-symbol cache under `/tmp/trading/baseline_cache`. Subsequent top-50 and
top-100 runs with the same date range reuse that cache.

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
