# Trading dashboard

A Nuxt 4 single-page app on Firebase Hosting showing the live performance of the
overnight-liquidity Alpaca paper account: equity curve, day/week/month/year P&L, open
positions, and a session-by-session trade log — behind a login.

## How the data gets here

The browser never holds an Alpaca credential. The trading daemon does the fetching and
the writing, so there is no second process to supervise.

**Firestore publishing is currently off.** The daemon emails the digest and nothing
else; add `--dashboard` to start publishing.

```
Alpaca paper API
      │
      ▼
ml/baseline/live_overnight_liquidity.py  (run --submit)
      │
      ├──smtplib────────▶ digest email, after the morning exit          [on]
      │
      └──firebase-admin──▶ Firestore ──authed read──▶ this app          [--dashboard]
             on every rank / entry / exit, plus a 5-minute throttle
```

`accounts/paper` holds the snapshot; `accounts/paper/sessions/{YYYY-MM-DD}` holds one
document per trading day. `firestore.rules` allows authenticated reads and no browser
writes at all — the publisher writes through the Admin SDK, which bypasses rules.

## Configuration

[`config.yaml`](./config.yaml) holds the Firebase project and web config, the dashboard
login and SMTP settings. It is read by `ml/baseline/*` and by `nuxt.config.ts`.

**Alpaca credentials are not in it.** The publisher, the digest and the trading daemon
all read `ALPACA_KEY` / `ALPACA_SECRET` from the environment, as they always have:

```bash
cd ../ml && set -a && source .env && set +a
```

Only browser-safe values reach the bundle — never the SMTP or dashboard password.
`deploy.sh` greps the generated output for them and refuses to deploy if one appears.

## Local development

```bash
pnpm install
pnpm dev        # http://localhost:5001
```

Without a Firebase project configured the app serves a deterministic demo dataset and
treats you as signed in, so a fresh checkout is usable immediately.

```bash
pnpm test       # vitest
pnpm typecheck  # vue-tsc
pnpm lint       # eslint
```

## First-time Firebase setup

The project id is already set to `trading-dashboard-ccdd5` in `.firebaserc` and
`config.yaml`. These steps need your Google login, so run them yourself:

1. `pnpm dlx firebase-tools login`
2. Firebase console → **Firestore Database** → create in Native mode.
3. Firebase console → **Authentication → Sign-in method** → enable **Email/Password**.
4. Firebase console → **Project settings → Service accounts → Generate new private key**,
   saved to the path in `config.yaml` (`~/.config/trading-dashboard/service-account.json`).
   You can also paste the key JSON inline under `firebase.service_account` instead.
5. Create the login user from `config.yaml`:
   ```bash
   cd ../ml && .venv/bin/python -m baseline.provision_auth_user
   ```

## Deploying

```bash
./deploy.sh                  # test + typecheck + lint, generate, deploy hosting & rules
./deploy.sh --skip-checks    # skip the gates
./deploy.sh --hosting-only   # leave firestore.rules alone
```

After the first deploy, update `dashboard.url` in `config.yaml` so the digest email links
to the real site.

## Publishing snapshots

**No extra process is required.** When enabled, the trading daemon publishes to
Firestore itself: immediately after every rank, entry and exit, and every 5 minutes in
between.

```bash
cd ../ml && set -a && source .env && set +a

# Current setup: digest email only, no Firestore writes.
.venv/bin/python -m baseline.live_overnight_liquidity run --submit

# Once the dashboard is live:
.venv/bin/python -m baseline.live_overnight_liquidity run --submit --dashboard
```

| Flag | Effect |
| --- | --- |
| *(none)* | **Default** — digest email only; the publisher is never loaded |
| `--dashboard` | Publish snapshots to Firestore |
| `--publish-interval-seconds 300` | Cadence between strategy actions (default, needs `--dashboard`) |
| `--publish-interval-seconds 0` | Publish **only** on rank/entry/exit |
| `--no-email` | Do not send the exit digest |

Publishing and emailing are both wrapped end to end: a Firestore or SMTP failure is
logged and swallowed, never propagated into an entry or an exit.

`dashboard_publisher` also runs standalone, for a manual backfill or to check the
payload without waiting for the daemon:

```bash
.venv/bin/python -m baseline.dashboard_publisher --dry-run   # print, write nothing
.venv/bin/python -m baseline.dashboard_publisher             # one snapshot
.venv/bin/python -m baseline.dashboard_publisher --watch --interval-seconds 300
```

It refuses any endpoint other than `paper-api.alpaca.markets` unless started with
`--allow-live-endpoint`, so an unlucky `ALPACA_URL` cannot point it at real money.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/pages/` | `login`, overview (`index`), `positions`, `history` |
| `app/components/` | Stat tiles, performance/positions/sessions tables, the SVG equity chart |
| `app/composables/` | `useAuth`, `useRepository`, `useSnapshot` |
| `app/repositories/` | Firestore adapter and the offline demo adapter |
| `app/types/dashboard.ts` | Snapshot shapes; mirror of the publisher's output |
| `app/utils/` | Formatting and chart geometry (both unit tested) |
| `firestore.rules` | Authenticated read, no browser writes |
