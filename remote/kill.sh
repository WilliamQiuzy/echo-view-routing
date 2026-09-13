#!/usr/bin/env bash
# Stop one of OUR jobs by name. Verifies via /proc/<pid>/cwd that the process runs inside ECHO_REMOTE_ROOT
# before signalling, and also signals child python processes started by the job's shell wrapper.
#   remote/kill.sh <job-name>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 1 ]] || { echo "usage: $0 <job-name>" >&2; exit 2; }
LOGS="${ECHO_REMOTE_LOGS_ROOT:-$ECHO_REMOTE_ROOT/logs}"
rssh "p='$LOGS/$1.pid'; [[ -f \"\$p\" ]] || { echo 'no pid file for $1'; exit 1; }; pid=\$(cat \"\$p\"); \
  cwd=\$(readlink /proc/\$pid/cwd 2>/dev/null || true); \
  case \"\$cwd\" in '$ECHO_REMOTE_ROOT'*) ;; *) echo \"refusing: pid \$pid cwd=\$cwd is not under $ECHO_REMOTE_ROOT\"; exit 1;; esac; \
  kids=\$(pgrep -P \$pid || true); kill \$pid \$kids 2>/dev/null && echo \"stopped $1 (\$pid \$kids)\"; rm -f \"\$p\""
