## Architecture & Tech Stack
- Framework: Vue 3 with Nuxt v4. Strictly follow Nuxt 4 directory structures (the `app/` directory).
- Styling: Tailwind CSS via the Nuxt-UI module.
- Language: Strict TypeScript.
- Use the Vue Composition API exclusively (`<script setup lang="ts">`). Do not use the Options API.
- Keep components small and reusable. Strictly separate UI rendering from business logic.
- Database: Firebase Firestore, read-only from the browser.

## Data flow
- The browser never talks to Alpaca. The trading daemon (`live_overnight_liquidity run`) calls
  `dashboard_publisher.build_snapshot`/`publish` on every strategy action and on a throttle,
  writing one snapshot document plus a `sessions` subcollection; this app only reads.
- Publishing is **opt-in** (`--dashboard`) and currently off; the daemon emails the digest only.
  Run the app against the demo adapter, or publish once by hand with
  `python -m baseline.dashboard_publisher`, while it stays off.
- Both outbound side effects (Firestore publish, digest email) are wrapped in bare `except` on
  the daemon side. Trading correctness outranks telemetry — never let either raise.
- `dashboard/config.yaml` holds Firebase, SMTP and dashboard-login settings. `nuxt.config.ts`
  parses it at build time and copies only browser-safe values into `runtimeConfig.public`; the
  SMTP and dashboard passwords are never bundled (`deploy.sh` fails the build if they appear).
- Alpaca credentials live in the environment (`ml/.env`), never in `config.yaml`.
- Snapshot field shapes live in `app/types/dashboard.ts` and must track `build_snapshot`
  in `ml/baseline/dashboard_publisher.py`, which is their only writer.

## Design
- Rely on Nuxt-UI components for standard UI elements before writing custom Tailwind classes.
- Modern, clean, dark theme with a tonal design.
- Gains render emerald, losses rose, and a flat/zero value stays neutral slate — an untraded
  account must not read as a win. Use `toneClass()` from `app/utils/format.ts`.
- All figures use the `.numeric` class for tabular figures so columns align.
- Every table and chart needs a real empty state: the paper account starts flat at $100,000
  with no fills, and that is the first thing a new deploy will render.

## Testing Strategy
- **Vitest** over pure logic and adapters (`test/`), matching the reference project. Firestore is
  faked with `vi.mock('firebase/firestore', ...)` — no emulator.
- `./deploy.sh` gates every deploy on `pnpm test && pnpm typecheck && pnpm lint`.
