#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_CHECKS=1
DEPLOY_RULES=1

usage() {
  echo "Usage: ./deploy.sh [--skip-checks] [--hosting-only]" >&2
}

for arg in "$@"; do
  case "$arg" in
    --skip-checks) RUN_CHECKS=0 ;;
    --hosting-only) DEPLOY_RULES=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 2 ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_file package.json
require_file firebase.json
require_file .firebaserc
require_file config.yaml
require_file firestore.rules
require_command pnpm

if [[ ! -d node_modules ]]; then
  echo "node_modules is missing. Run pnpm install first." >&2
  exit 1
fi

if ! grep -q '"public"[[:space:]]*:[[:space:]]*".output/public"' firebase.json; then
  echo 'firebase.json must deploy hosting.public from ".output/public".' >&2
  exit 1
fi

if [[ "$RUN_CHECKS" -eq 1 ]]; then
  pnpm test
  pnpm typecheck
  pnpm lint
else
  echo "Skipping tests, typecheck, and lint."
fi

pnpm exec nuxi generate

if [[ ! -f .output/public/index.html ]]; then
  echo "Generate completed, but .output/public/index.html was not found." >&2
  exit 1
fi

# config.yaml carries the SMTP app password and the dashboard password, neither of
# which may reach the CDN. firebase.json ignores the file, but a bundling mistake would
# be silent, so fail loudly instead.
if [[ -f .output/public/config.yaml ]]; then
  echo "Refusing to deploy: config.yaml ended up inside .output/public." >&2
  exit 1
fi
for secret in \
  "$(sed -n 's/^  password: "\(.*\)"$/\1/p' config.yaml | head -1)" \
  "$(sed -n 's/^  password: "\(.*\)"$/\1/p' config.yaml | tail -1)"
do
  if [[ -n "$secret" ]] && grep -rqF -- "$secret" .output/public 2>/dev/null; then
    echo "Refusing to deploy: a secret from config.yaml is present in the bundle." >&2
    exit 1
  fi
done

TARGETS="hosting"
if [[ "$DEPLOY_RULES" -eq 1 ]]; then
  TARGETS="hosting,firestore:rules"
fi

pnpm dlx firebase-tools deploy --only "$TARGETS"
