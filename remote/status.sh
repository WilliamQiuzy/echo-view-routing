#!/usr/bin/env bash
# One-screen status of the server: GPU, our jobs, downloads, disk, tail of a log.
#   remote/status.sh [job-name]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB="${1:-}"
LOGS="${ECHO_REMOTE_LOGS_ROOT:-$ECHO_REMOTE_ROOT/logs}"
rssh "echo '== GPU =='; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's#^#  #'; \
  echo '== our jobs =='; for p in '$LOGS'/*.pid; do [[ -f \"\$p\" ]] || continue; n=\$(basename \"\$p\" .pid); \
    if kill -0 \$(cat \"\$p\") 2>/dev/null; then echo \"  RUNNING \$n (pid \$(cat \"\$p\"))\"; else echo \"  stopped \$n\"; fi; done; \
  echo '== disk =='; df -h '$ECHO_REMOTE_ROOT' | tail -1; du -sh '$ECHO_REMOTE_ROOT'/data '$ECHO_REMOTE_ROOT'/runs '$ECHO_REMOTE_ROOT'/checkpoints 2>/dev/null; \
  if [[ -n '$JOB' ]]; then echo '== log $JOB =='; tail -n 30 '$LOGS/$JOB.log'; fi"
