# Trading dashboard

A Nuxt 4 single-page app on Firebase Hosting showing the live performance of the
overnight-liquidity Alpaca paper account: live broker equity, realized strategy equity,
day/week/month/year realized P&L, open positions, and a session-by-session trade log —
behind a login.

## How the data gets here

The browser never holds an Alpaca credential. A small daemon in this directory reads
Alpaca plus the trading process's JSON artifacts and publishes them to Firestore. The
trading process has no Firebase imports, flags, callbacks, or failure modes.

```
Alpaca API ─────────────────────┐
trading state.json ─────────────┤
overnight/config.yaml ─────────┴──▶ scripts/dashboard_daemon.py
                                      │
                                      └──▶ Firestore ──authed read──▶ Nuxt app
```

`accounts/paper` holds the snapshot; `accounts/paper/sessions/{YYYY-MM-DD}` holds one
document per completed basket, keyed by its entry trading day. Entry- and exit-day
copies of the same audit summary are collapsed into that one record. `firestore.rules`
allows authenticated reads and no browser writes at all — the publisher writes through
the Admin SDK, which bypasses rules.

The chart is dated by basket exit day and compounds strategy net P&L: closed-basket fill
P&L less confirmed Alpaca `FEE` account activities booked on the exit date. The daemon
caches those activities beside the live summary in
`<work-dir>/<entry-date>/fee_activities.json`, shared with `trading-reconcile`.
Dashboard polls use the cache: pending fees and exit dates within the last seven
calendar days are checked hourly; older confirmed dates are checked weekly for
corrections. Each request covers only a due session's exit-date window, rather than
the full account history. Successful checks update `last_checked_at` even when the
fees are unchanged. Use `trading-dashboard --once --refresh-broker-fees` to bypass
these intervals. Failed checks retain the cache, and an empty response cannot erase
previously observed fees. An empty cache becomes confirmed zero only after a
successful check beyond the posting grace period. Newly closed sessions appear
immediately using gross fill P&L with zero fees assumed. Performance buckets and digest
emails label these results as `provisional`; the final chart segment is dashed, and
provisional sessions remain excluded from risk statistics until fees post, normally the
next day. Today/WTD/MTD/YTD/inception performance includes provisional sessions.
Confirmed results then subtract the actual fees.
Account-equity change remains a separate reconciliation value, with any difference
shown as the unexplained residual. This keeps strategy performance independent of
deposits, settlement rounding, and Alpaca's delayed portfolio-history rollover.

The publisher reads the credential-free schedule, strategy, data, and execution
sections from the live runner's `effective_config.json` on every poll, falling back to
`overnight/config.yaml` when no active-runtime artifact exists. The frontend renders
pipeline times from that snapshot, including CLI overrides, so parameters no longer
need to be copied into dashboard config or baked into a new frontend build.

The schedule always shows a countdown beside the upcoming **Next market open**
milestone using the published exchange calendar, including when another milestone
is next or execution is still in progress. Queued exits display **Queued successfully** on the
exit-orders milestone while the market-open countdown continues. After the open,
queued positions display **Awaiting fills** until execution is confirmed. Countdown
updates run in the browser every 30 seconds and need no extra broker requests.

## Configuration

[`config.yaml`](./config.yaml) holds the Firebase project and web config plus the
dashboard login. Only code under `dashboard/` reads it.

**Alpaca credentials are not in it.** Export `ALPACA_KEY` / `ALPACA_SECRET` before
starting the dashboard daemon:

```bash
set -a && source ../overnight/.env && set +a
```

Only browser-safe values reach the bundle. `deploy.sh` checks that configured secrets
did not enter the generated static files.

## Local development

```bash
pnpm install
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
pnpm dev        # http://localhost:5001
```

Without a Firebase project configured the app serves a deterministic demo dataset and
treats you as signed in, so a fresh checkout is usable immediately.

```bash
pnpm test       # vitest
pnpm typecheck  # vue-tsc
pnpm lint       # eslint
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

## First-time Firebase setup

The project id is already set to `trading-dashboard-ccdd5` in `.firebaserc` and
`config.yaml`. These steps need your Google login, so run them yourself:

1. `pnpm dlx firebase-tools login`
2. Firebase console → **Firestore Database** → create in Native mode.
3. Firebase console → **Authentication → Sign-in method** → enable **Email/Password**.
4. Firebase console → **Project settings → Service accounts → Generate new private key**,
   saved to the ignored path in `config.yaml`
   (`dashboard/.secrets/service-account.json`). Never commit this file.
5. Create the login user from `config.yaml`:
   ```bash
   .venv/bin/trading-dashboard-auth
   ```

## Deploying

```bash
./deploy.sh                  # test + typecheck + lint, generate, deploy hosting & rules
./deploy.sh --skip-checks    # skip the gates
./deploy.sh --hosting-only   # leave firestore.rules alone
```

After the first deploy, update `dashboard.url` in `config.yaml` to the real site.

## Publishing snapshots

The dashboard daemon is intentionally separate from trading. Run it under the same
process supervisor as the trading daemon; it publishes immediately, then every two
minutes by default. After it observes a basket reach `closed` on its configured exit
day, it also sends one digest email using `smtp` and `notifications.recipients` from
`config.yaml`. A durable marker prevents duplicate messages across daemon restarts;
SMTP failures are logged and retried without affecting Firestore publishing.

```bash
set -a && source ../overnight/.env && set +a

.venv/bin/trading-dashboard --dry-run  # print one payload
.venv/bin/trading-dashboard --once     # publish once
.venv/bin/trading-dashboard            # keep publishing
```

Always use `dashboard/.venv` for this process. The ML environment intentionally has a
separate dependency set and is not supported for the Firebase daemon.

Use `--interval-seconds 60` to override the two-minute cadence and `--state-path` to
select a separate state file. Use `--no-email` to disable digests. A failed publish is
logged and retried on the next interval.

The daemon detects its mode directly from `--trading-url` or `ALPACA_URL`: the paper
endpoint reads `/data/ppv1/paper`, while the live endpoint reads `/data/ppv1/live`.
Use `--work-dir` to override that mapping. No extra live-mode flag is required. Only
the recognized HTTPS Alpaca paper and live hosts are accepted. Both modes publish to
the stable `accounts/current` document and record the detected mode in snapshot metadata.
When the mode changes, the publisher clears the old mode's derived session documents
before writing the new history. The frontend therefore remains account-agnostic and
does not need rebuilding when the publisher switches between paper and live.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/pages/` | `login`, overview (`index`), `positions`, `history` |
| `app/components/` | Stat tiles, performance/positions/sessions tables, the SVG equity chart |
| `app/composables/` | `useAuth`, `useRepository`, `useSnapshot` |
| `app/repositories/` | Firestore adapter and the offline demo adapter |
| `app/types/dashboard.ts` | Snapshot shapes; mirror of the publisher's output |
| `app/utils/` | Formatting and chart geometry (both unit tested) |
| `trading_rl/dashboard/daemon.py` | Independent Alpaca-to-Firestore polling daemon |
| `trading_rl/dashboard/metrics.py` | Pure snapshot performance calculations |
| `trading_rl/dashboard/config.py` | Dashboard-only configuration loader |
| `firestore.rules` | Authenticated read, no browser writes |
