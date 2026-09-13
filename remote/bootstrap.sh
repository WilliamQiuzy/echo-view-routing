#!/usr/bin/env bash
# One-time server setup (idempotent). Documents what was done by hand on 2026-09-13.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rssh "set -e; mkdir -p '$ECHO_REMOTE_ROOT'/{data/ev9v/raw,runs,logs,checkpoints,cache,third_party}; cd '$ECHO_REMOTE_ROOT';
  [ -x .venv/bin/python ] || python3 -m venv .venv;
  .venv/bin/pip install -q --upgrade pip;
  .venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130;
  .venv/bin/pip install -q -e '.[train,dev]';
  [ -d third_party/stfm ] || git clone -q https://github.com/bgx666/stfm.git third_party/stfm;
  git -C third_party/stfm checkout -q 532f60b2733a3442dcccb59099e00eba07bf293d;
  cd data/ev9v/raw;
  for f in train_labeled.txt validation_labeled.txt test_labeled.txt README.md; do [ -s \$f ] || curl -sSL -o \$f https://huggingface.co/datasets/bgx666/EV9V/resolve/main/\$f; done;
  for t in Videos Images; do [ -s \$t.tar ] || curl -sSL -C - -o \$t.tar https://huggingface.co/datasets/bgx666/EV9V/resolve/main/\$t.tar; done;
  cd ..; [ -d Videos ] || tar -xf raw/Videos.tar; [ -d Images ] || tar -xf raw/Images.tar;
  echo bootstrap-done"
