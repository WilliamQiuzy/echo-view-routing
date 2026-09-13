#!/usr/bin/env bash
# Pull small result artifacts (metrics, tables, plots, configs) from server runs/ to local runs/.
# Never pulls checkpoints, features, or data.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$REPO_ROOT/runs"
rsync -az --prune-empty-dirs \
  --include '*/' \
  --include '*.json' --include '*.csv' --include '*.md' --include '*.yaml' --include '*.png' --include '*.svg' --include '*.txt' \
  --exclude '*' \
  "$ECHO_SSH_ALIAS:$ECHO_REMOTE_ROOT/runs/" "$REPO_ROOT/runs/"
echo "pulled results -> $REPO_ROOT/runs"
