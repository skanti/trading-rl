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

Live execution and backtests default to `liquidity-momentum-focus`: shortlist the
12 most liquid Nasdaq issuers, select three by ten-session momentum, and target
35% volatility using 20 completed modeled baskets. Exposure is capped at 2x and
falls to zero below the 100-session SPY daily-close average.

```bash
trading-backtest --strategy liquidity-momentum-focus --since 2026-01-01 --budget 10000
```

Live, backtesting and reconciliation use shared strategy code. Restart
`trading-live` to apply the new default; existing positions keep their exit
workflow and saved plans retain their original strategy. Historical sessions
remain replayable with their recorded strategy and inputs.

Use `--strategy liquidity-trend-vol` or `--strategy liquidity-fixed` to select
the earlier strategies, and `--strategy-config` for backtest parameter variants.
Generated results default to `/tmp/trading-backtests/candidate/<strategy>/`.
See [strategies and experiment design](overnight/STRATEGIES.md) for exact rules,
configuration, reproduction and the recommended comparison framework.

Backtesting and historical ranking use Alpaca's official market calendar, cached
at `/data/ppv1/live/market_calendar.json`. Session dates and early closes no longer
depend on complete SPY minute bars. A morning-only final session can supply the
09:30 exit once that time has elapsed and the exit data is available; individual
missing prices remain subject to the existing execution-data checks.

The download pipeline refreshes this calendar before updating prices
(`CALENDAR_PATH` overrides its location). Backtests reuse the cache, fetching
missing coverage with Alpaca credentials when needed. Use `--refresh-calendar`
to refresh it explicitly, or `--calendar-path /path/to/calendar.json` for an
offline snapshot with `start`, `end`, and `sessions` fields. Insufficient or
invalid offline coverage is an error. Run summaries save the exact calendar
records and their hash so the session schedule is reproducible.
