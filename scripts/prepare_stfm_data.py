#!/usr/bin/env python
"""Write data/stfm/EchoData/ (STFM's expected layout) from our manifest without copying pixels."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.ingest.stfm_layout import build_stfm_layout  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402


def main() -> None:
    ap = base_parser("Prepare STFM data layout"); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    out = build_stfm_layout(load_manifest(paths.manifest_path), paths.ev9v_images, paths.repo_root / cfg["stfm"]["data_path"], args.limit)
    print(f"STFM layout at {out}"); print("\n".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
