# trading-rl

Market-data downloaders, backtesting, reconciliation, and live execution tools
for the overnight dollar-liquidity strategy.

Install the repository in editable mode for development:

```bash
uv pip install -e .
```

Install `.[dashboard]` to publish to Firebase, or `.[ml]` for the research
models. The core live, backtest, reconciliation, and downloader commands need
only the default dependencies.

The import package lives directly at `trading_rl/`; this project intentionally
does not use a `src/` directory. See [overnight/README.md](overnight/README.md)
for strategy behavior, data formats, and command examples.

Installed commands include `trading-dashboard`, `trading-dashboard-auth`,
`trading-live`, `trading-backtest`, `trading-rank`, `trading-reconcile`, `download-bars`,
`download-auctions`, and `download-nbbo`.

## Daemon management

The user starts and restarts all long-running processes, including live trading
and the dashboard/digest publisher. Agents must not launch or restart daemons as
part of code changes or deployments unless explicitly asked to do so. If a
restart is needed, identify the process and leave the restart to the user.
Do not stop user-managed processes unless explicitly requested.

These instructions also live in [AGENTS.md](AGENTS.md) for future coding agents.

## Strategy experiments

The `liquidity-trend-vol` strategy combines liquidity ranking, a SPY trend filter,
and volatility targeting (35% target and 2x cap by default):

```bash
trading-backtest --strategy liquidity-trend-vol --since 2026-01-01 --budget 10000
```

Live execution now defaults to `liquidity-trend-vol`, with shared risk logic and
no backtester imports. The user must restart `trading-live` to apply it; existing
positions keep their exit workflow.

Backtests default to `liquidity-trend-vol`. Use `--strategy liquidity-fixed` for fixed exposure and
`--strategy-config` for `liquidity-trend-vol` parameter variants. Its summaries,
trades, portfolio returns, minute audits and charts go to
`/tmp/trading-backtests/candidate/liquidity-trend-vol/`.
See [strategies and experiment design](overnight/STRATEGIES.md) for exact rules,
configuration, reproduction and the recommended comparison framework.
