#!/usr/bin/env bash
# Compatibility entrypoint: build the current aggregate-only result page.
# The legacy force-push publisher is retired; its exact prior working-tree
# bytes are preserved in legacy/publish_pages.before_external_tests.sh.txt.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != "--preview" || -z "${2:-}" ]]; then
  echo "Legacy video/force-push publication is retired." >&2
  echo "Build verified current results: demo/publish_pages.sh --preview OUTPUT_DIR" >&2
  echo "Current hosted results use the separately verified private Site." >&2
  exit 2
fi
cd "$REPO_ROOT"
.venv/bin/python scripts/build_external_results.py --out "$2"
