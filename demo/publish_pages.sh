#!/usr/bin/env bash
# Publish the demo site (baselines page + ViT-family page + stream videos + figures) to GitHub Pages (gh-pages).
#   demo/publish_pages.sh                 build and force-push the gh-pages branch
#   demo/publish_pages.sh --preview DIR   build the identical site into DIR only (nothing is pushed)
# Requires runs/_demo/streams/ pulled from the server and the latest ladder metrics in runs/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PREVIEW=""
if [[ "${1:-}" == "--preview" ]]; then PREVIEW="${2:?usage: $0 --preview DIR}"; fi
.venv/bin/python demo/build_demo_v2.py >/dev/null
.venv/bin/python demo/build_vit_page.py >/dev/null
SITE="$(mktemp -d)"
wrap() {  # fragment -> standalone HTML document
python3 - "$1" "$2" <<'PY'
import sys
frag = open(sys.argv[1]).read()
i = frag.index('<div class="wrap">')
head, body = frag[:i], frag[i:]
open(sys.argv[2], "w").write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                             + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
PY
}
wrap "$REPO_ROOT/demo/demo_v2.html" "$SITE/index.html"
wrap "$REPO_ROOT/demo/vit.html" "$SITE/vit.html"
mkdir -p "$SITE/streams"
python3 - "$REPO_ROOT/runs/_demo/streams/streams.json" "$REPO_ROOT/runs/_demo/streams" "$SITE/streams" <<'PY'
import json, shutil, sys
d = json.load(open(sys.argv[1]))
for s in d["streams"]:
    shutil.copy(f"{sys.argv[2]}/{s['file']}", f"{sys.argv[3]}/{s['file']}")
print(len(d["streams"]), "videos copied")
PY
mkdir -p "$SITE/figures" && cp "$REPO_ROOT"/demo/figures/* "$SITE/figures/"
touch "$SITE/.nojekyll"
if [[ -n "$PREVIEW" ]]; then
  mkdir -p "$PREVIEW" && cp -R "$SITE"/. "$PREVIEW"/ && rm -rf "$SITE"
  echo "preview written to $PREVIEW (open $PREVIEW/index.html or $PREVIEW/vit.html)"; exit 0
fi
WT="$(mktemp -d)"
git worktree add -q -B gh-pages "$WT" 2>/dev/null || git worktree add -q "$WT" gh-pages
rm -rf "$WT"/* "$WT"/.nojekyll 2>/dev/null || true
cp -R "$SITE"/. "$WT"/
git -C "$WT" add -A
git -C "$WT" -c commit.gpgsign=false commit -q -m "pages: demo site $(date +%Y-%m-%d)" || echo "nothing to commit"
git -C "$WT" push -q -f origin gh-pages
git worktree remove --force "$WT"; rm -rf "$SITE"
echo "pushed gh-pages"
