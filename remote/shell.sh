#!/usr/bin/env bash
# Interactive shell in the project root with the venv activated.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec ssh -t "$ECHO_SSH_ALIAS" "cd '$ECHO_REMOTE_ROOT' && source .venv/bin/activate && exec bash -l"
