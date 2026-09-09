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
