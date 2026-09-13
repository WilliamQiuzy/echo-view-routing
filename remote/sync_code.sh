#!/usr/bin/env bash
# Push the code tree (never data/runs/checkpoints) from the Mac to the server project root.
# Excluded paths are protected from --delete on the receiver, so server-side data is never removed.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rssh "mkdir -p '$ECHO_REMOTE_ROOT'"
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '.env' \
  --exclude 'data/' --exclude 'runs/' --exclude 'checkpoints/' --exclude 'cache/' --exclude 'logs/' \
  --exclude 'third_party/stfm/' --exclude 'demo/samples/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' --exclude '.DS_Store' \
  --exclude '*.tar' --exclude '*.npz' --exclude '*.pt' --exclude '*.pth' \
  "$REPO_ROOT/" "$ECHO_SSH_ALIAS:$ECHO_REMOTE_ROOT/"
echo "synced $REPO_ROOT -> $ECHO_SSH_ALIAS:$ECHO_REMOTE_ROOT"
