#!/usr/bin/env bash
# Shared loader for remote/*.sh — sources .env from the repo root and validates required keys.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ECHO_ENV_FILE:-$REPO_ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found (copy .env.example to .env)" >&2; exit 1
fi
set -a; # shellcheck disable=SC1090
source "$ENV_FILE"; set +a
for key in ECHO_SSH_ALIAS ECHO_REMOTE_ROOT ECHO_REMOTE_PYTHON; do
  if [[ -z "${!key:-}" ]]; then echo "error: $key missing in $ENV_FILE" >&2; exit 1; fi
done
if [[ "$ECHO_REMOTE_ROOT" == "${ECHO_CO_TENANT_ROOT:-__none__}"* ]]; then
  echo "error: ECHO_REMOTE_ROOT overlaps the co-tenant project; refusing" >&2; exit 1
fi
rssh() { ssh -o ConnectTimeout=20 -o BatchMode=yes "$ECHO_SSH_ALIAS" "$@"; }
