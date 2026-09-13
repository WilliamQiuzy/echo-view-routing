#!/usr/bin/env bash
# Stop one of OUR jobs by name (only pids recorded under our logs dir; never touches the co-tenant).
#   remote/stop.sh <job-name>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 1 ]] || { echo "usage: $0 <job-name>" >&2; exit 2; }
LOGS="${ECHO_REMOTE_LOGS_ROOT:-$ECHO_REMOTE_ROOT/logs}"
rssh "p='$LOGS/$1.pid'; [[ -f \"\$p\" ]] || { echo 'no pid file for $1'; exit 1; }; \
  pid=\$(cat \"\$p\"); cmd=\$(ps -o cmd= -p \"\$pid\" || true); \
  case \"\$cmd\" in *'$ECHO_REMOTE_ROOT'*) kill \"\$pid\" && echo \"stopped $1 (\$pid)\";; *) echo \"refusing: pid \$pid is not one of ours: \$cmd\"; exit 1;; esac"
