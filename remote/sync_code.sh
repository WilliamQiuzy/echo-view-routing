#!/usr/bin/env bash
# Push git-tracked files (code, configs, docs, patches) from the Mac to the server project root.
# Uses `git ls-files` so ONLY tracked paths are ever written; nothing on the server is deleted (server-only
# directories such as data/, runs/, checkpoints/, cache/, logs/, envs/, bin/, secrets/, third_party/<clones>/ are untouched).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rssh "mkdir -p '$ECHO_REMOTE_ROOT'"
git -C "$REPO_ROOT" ls-files -z | rsync -az --from0 --files-from=- "$REPO_ROOT/" "$ECHO_SSH_ALIAS:$ECHO_REMOTE_ROOT/"
echo "synced tracked files $REPO_ROOT -> $ECHO_SSH_ALIAS:$ECHO_REMOTE_ROOT (no deletions)"
