#!/usr/bin/env bash
# Launch a detached job on the server inside the project venv.
#   remote/run.sh <job-name> <module-or-script> [args...]
# Example: remote/run.sh train_b0 scripts/train_frame_encoder.py --config configs/frame_encoder.yaml
# Logs -> $ECHO_REMOTE_LOGS_ROOT/<job-name>.log ; pid -> <job-name>.pid
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 2 ]] || { echo "usage: $0 <job-name> <script> [args...]" >&2; exit 2; }
JOB="$1"; shift
LOGS="${ECHO_REMOTE_LOGS_ROOT:-$ECHO_REMOTE_ROOT/logs}"
printf -v ARGS '%q ' "$@"
rssh "cd '$ECHO_REMOTE_ROOT' && mkdir -p '$LOGS' && \
  if [[ -f '$LOGS/$JOB.pid' ]] && kill -0 \$(cat '$LOGS/$JOB.pid') 2>/dev/null; then echo 'job $JOB already running (pid '\$(cat '$LOGS/$JOB.pid')')'; exit 3; fi; \
  ECHO_GPU_MEM_FRACTION='${ECHO_GPU_MEM_FRACTION:-0.6}' PYTHONPATH='$ECHO_REMOTE_ROOT' \
  nohup '$ECHO_REMOTE_PYTHON' -u $ARGS > '$LOGS/$JOB.log' 2>&1 & echo \$! > '$LOGS/$JOB.pid'; \
  echo \"started $JOB pid \$(cat '$LOGS/$JOB.pid') log $LOGS/$JOB.log\""
