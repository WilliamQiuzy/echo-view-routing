#!/usr/bin/env bash
# Publish demo v2 (page + stream videos) to GitHub Pages on the gh-pages branch.
# Requires runs/_demo/streams/ pulled from the server and the latest ladder metrics in runs/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
.venv/bin/python demo/build_demo_v2.py >/dev/null
SITE="$(mktemp -d)"
python3 - "$REPO_ROOT/demo/demo_v2.html" "$SITE/index.html" <<'PY'
import sys
frag = open(sys.argv[1]).read()
i = frag.index('<div class="wrap">')
head, body = frag[:i], frag[i:]
open(sys.argv[2], "w").write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                             + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
PY
mkdir -p "$SITE/streams"
python3 - "$REPO_ROOT/runs/_demo/streams/streams.json" "$REPO_ROOT/runs/_demo/streams" "$SITE/streams" <<'PY'
import json, shutil, sys
d = json.load(open(sys.argv[1]))
for s in d["streams"]:
    shutil.copy(f"{sys.argv[2]}/{s['file']}", f"{sys.argv[3]}/{s['file']}")
print(len(d["streams"]), "videos copied")
PY
touch "$SITE/.nojekyll"
WT="$(mktemp -d)"
git worktree add -q -B gh-pages "$WT" 2>/dev/null || git worktree add -q "$WT" gh-pages
rm -rf "$WT"/* "$WT"/.nojekyll 2>/dev/null || true
cp -R "$SITE"/. "$WT"/
git -C "$WT" add -A
git -C "$WT" -c commit.gpgsign=false commit -q -m "pages: demo v2 $(date +%Y-%m-%d)" || echo "nothing to commit"
git -C "$WT" push -q -f origin gh-pages
git worktree remove --force "$WT"; rm -rf "$SITE"
echo "pushed gh-pages"
