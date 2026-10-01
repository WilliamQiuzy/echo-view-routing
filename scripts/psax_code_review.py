#!/usr/bin/env python
"""Frame grids for the disputed EV9V parasternal short-axis codes (and PMPALA), for a reader to assign ASE levels.

For each code, `--per-code` random training clips (fixed seed) are shown as a row of `--frames` evenly spaced frames.
Writes <out>/psax_<code>.png. The EV9V documents disagree on these codes (docs/LABEL_MAPPING.md); the grids let a
clinician check them against the ASE / SHM level criteria (AV level: aortic valve and RVOT; MV level: mitral leaflets;
papillary level: both papillary muscles; apex: small LV cavity without papillary muscles; PA long axis: RVOT to the
pulmonary artery bifurcation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402

CODES = ["PASA", "PMASA", "PMVLSA", "PPMLSA", "PMPALA"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main() -> None:
    ap = argparse.ArgumentParser(description="EV9V PSAX code review grids")
    ap.add_argument("--out", required=True); ap.add_argument("--per-code", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); paths = load_paths()
    man = pd.read_csv(paths.manifest_path)
    man = man[(man["dataset"] == "EV9V") & (man["split"] == "train")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 15) if Path(FONT).is_file() else ImageFont.load_default()
    w, h, pad, head = 320, 240, 6, 24
    rng = np.random.default_rng(args.seed)
    for code in CODES:
        sub = man[man["raw_label"] == code]
        pick = sub.iloc[rng.choice(len(sub), size=min(args.per_code, len(sub)), replace=False)]
        img = Image.new("RGB", (args.frames * (w + pad) + pad, 34 + len(pick) * (h + head + pad)), "white")
        d = ImageDraw.Draw(img); d.text((pad, 8), f"EV9V code {code}: {len(pick)} random training clips", fill="black", font=font)
        for i, r in enumerate(pick.itertuples(index=False)):
            y = 34 + i * (h + head + pad)
            frames = sorted(Path(r.frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
            d.text((pad, y + 3), f"{r.video_id}  ({len(frames)} frames)", fill="black", font=font)
            for j, k in enumerate(np.linspace(0, len(frames) - 1, args.frames).round().astype(int)):
                img.paste(Image.open(frames[k]).convert("RGB").resize((w, h)), (pad + j * (w + pad), y + head))
        img.save(out / f"psax_{code}.png")
        print(code, len(sub), "clips in train; wrote", out / f"psax_{code}.png")


if __name__ == "__main__":
    main()
