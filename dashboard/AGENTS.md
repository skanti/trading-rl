## Architecture & Tech Stack
- Framework: Vue 3 with Nuxt v4. Strictly follow Nuxt 4 directory structures (the `app/` directory).
- Styling: Tailwind CSS via the Nuxt-UI module.
- Language: Strict TypeScript.
- Use the Vue Composition API exclusively (`<script setup lang="ts">`). Do not use the Options API.
- Keep components small and reusable. Strictly separate UI rendering from business logic.
- Database: Firebase Firestore, read-only from the browser.

## Data flow
- The browser never talks to Alpaca. `scripts/dashboard_daemon.py` is a standalone polling
  process that reads Alpaca and the trading daemon's JSON artifacts, then writes one snapshot
  document plus a `sessions` subcollection. Nothing in `ml/baseline` imports dashboard code.
- The dashboard daemon retries its own failures. Trading must never supervise, import, or call it.
- `dashboard/config.yaml` holds Firebase and dashboard-login settings. `nuxt.config.ts` parses it
  at build time and copies only browser-safe values into `runtimeConfig.public`; configured
  secrets are never bundled (`deploy.sh` fails the build if they appear).
- Alpaca credentials live in the environment (`ml/.env`), never in `config.yaml`.
- Snapshot field shapes live in `app/types/dashboard.ts` and must track `build_snapshot`
  in `scripts/dashboard_daemon.py`, which is their only writer.

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
- Python daemon logic is covered by `python -m unittest discover -s scripts -p 'test_*.py'`.
- `./deploy.sh` gates every deploy on `pnpm test && pnpm typecheck && pnpm lint`.
