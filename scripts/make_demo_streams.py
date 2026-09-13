#!/usr/bin/env python
"""Render demo streams (one native cine, one constructed stream per 2x2 cell, one four-fragment stream) as MP4
and decode each with every baseline at its frozen validation-selected threshold.

Writes runs/_demo/streams/<id>.mp4 and runs/_demo/streams/streams.json (per-method intervals in seconds).
Server-side (needs the feature cache, the JPEG frames and ffmpeg).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.compose.recipe_builder import Fragment, Pool, Recipe, make_pair_recipes  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.report import collect_metrics  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA  # noqa: E402
from echo_routing.ingest.frames import apply_gamma, list_frame_paths  # noqa: E402
from echo_routing.temporal.baselines.b_file import file_score  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.decode_select import route  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

FPS = 30
STEP = 3  # frames per 10 Hz sample


def render_mp4(recipe: Recipe, loader, images_root: Path, out: Path) -> float:
    """Concatenate the exact frames behind each fragment (3 frames per 10 Hz sample); returns duration in seconds."""
    tmp = Path(tempfile.mkdtemp(prefix="demo_stream_"))
    k = 0
    try:
        for fr in recipe.fragments:
            vf = loader(fr.video_id, "orig"); paths = list_frame_paths(images_root / fr.video_id)
            gamma = VARIANT_GAMMA[fr.variant]
            f0 = int(vf.frame_idx[fr.start]); f1 = min(int(vf.frame_idx[fr.end - 1]) + STEP, len(paths))
            for fi in range(f0, f1):
                with Image.open(paths[fi]) as im:
                    im = im.convert("RGB")
                    if gamma != 1.0:
                        im = apply_gamma(im, gamma)
                    im.save(tmp / f"{k:06d}.jpg", quality=92)
                k += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%06d.jpg"),
                        "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return k / FPS


def decode_all(stream, methods: list[str], taus: dict[str, float | None], cfg: dict, hz: float) -> dict:
    pol = cfg["policy"]; out = {}
    for m in methods:
        tau = taus.get(m)
        if m == "b_file":
            view, score = file_score(stream.prob, "mean")
            out[m] = {"tau": tau, "intervals": [{"start_s": 0.0, "end_s": stream.prob.shape[0] / hz, "view": int(view), "score": float(score),
                                                "accepted": bool(tau is not None and score >= tau)}]}
            continue
        labels, score_prob = decode_with_prob(m, stream.prob, cfg, stream.feat)
        ivs = route(score_prob, labels, tau if tau is not None else np.inf, method=pol["score"], min_len=int(pol["min_len"]))
        out[m] = {"tau": tau, "intervals": [{"start_s": iv.start / hz, "end_s": iv.end / hz, "view": int(iv.view), "score": float(iv.score),
                                            "accepted": bool(iv.accepted)} for iv in ivs]}
    return out


def main() -> None:
    ap = base_parser("Render and decode demo streams")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--mstcn-checkpoint", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); ont = load_ontology()
    if args.mstcn_checkpoint:
        cfg.setdefault("mstcn", {})["checkpoint"] = args.mstcn_checkpoint
    hz = float(cfg["sampling"]["target_hz"]); classes = list(ont.classes("family5"))
    out_dir = Path(args.out) if args.out else paths.runs_root / "_demo" / "streams"; out_dir.mkdir(parents=True, exist_ok=True)
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig")); test = idx[idx["split"] == "test"].reset_index(drop=True)
    raw_of = dict(zip(test["video_id"], test["raw_label"]))
    cache = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    metrics, _ = collect_metrics(paths.runs_root, args.ckpt_hash)
    taus = {m["method_id"]: (m.get("policy", {}).get("by_target", {}).get("0.05", {}) or {}).get("tau") for m in metrics}
    methods = [m for m in ["b_file", "b0", "b1", "b2", "b3", "b4"] if m in taus and (m != "b4" or args.mstcn_checkpoint)]
    rng = np.random.default_rng(args.seed)

    recipes: list[tuple[str, str, Recipe]] = []
    # 1) one untouched native cine (6-12 s), the longest such in the test split
    nat = test[(test["n_samples"] >= 60) & (test["n_samples"] <= 120)].sort_values("n_samples").iloc[-1]
    recipes.append(("native", "Untouched native cine (single view)",
                    Recipe("native_0000", "native", (Fragment(nat.video_id, int(nat.label_index), "orig", 0, int(nat.n_samples)),))))
    # 2) one constructed pair per 2x2 cell, fragments 2-4 s
    pool = Pool.from_index(test["video_id"], test["label_index"], test["n_samples"])
    for r in make_pair_recipes(pool, 1, rng, edit_variant=cfg["recipes"]["edit_variant"], min_len=20, max_len=40):
        recipes.append((r.cell, {"same_none": "Same view, two patients, hard join", "same_edit": "Same view, two patients, second half gamma-edited",
                                 "diff_none": "Different views, hard join", "diff_edit": "Different views, second half gamma-edited"}[r.cell], r))
    # 3) a four-fragment mixed stream: A4C -> A4C' (edited) -> PLAX -> PSAX
    want = [("A4C", "orig"), ("A4C", cfg["recipes"]["edit_variant"]), ("PLAX", "orig"), ("PSAX", "orig")]
    used = {f.video_id for _, _, r in recipes for f in r.fragments}; frags = []
    for fam, var in want:
        li = classes.index(fam)
        cands = test[(test["label_index"] == li) & (test["n_samples"] >= 30) & (~test["video_id"].isin(used))]
        row = cands.iloc[int(rng.integers(len(cands)))]; used.add(row.video_id)
        length = int(min(30, row.n_samples)); start = int(rng.integers(0, row.n_samples - length + 1))
        frags.append(Fragment(row.video_id, li, var, start, start + length))
    recipes.append(("multi", "Four fragments: A4C → A4C (edited) → PLAX → PSAX", Recipe("multi_0000", "multi", tuple(frags))))

    streams_out = []
    for kind, title, r in recipes:
        s = render(r, loader)
        mp4 = out_dir / f"{r.recipe_id}.mp4"
        dur = render_mp4(r, loader, paths.ev9v_images, mp4)
        truth = [{"start_s": sg.start / hz, "end_s": sg.end / hz, "label": int(sg.label)} for sg in labels_to_segments(s.labels)]
        streams_out.append({
            "id": r.recipe_id, "kind": kind, "cell": r.cell, "title": title, "file": mp4.name, "duration_s": dur, "hz": hz,
            "fragments": [{"video_id": f.video_id, "raw_label": raw_of.get(f.video_id), "family": classes[f.label], "variant": f.variant,
                           "start_s": None, "end_s": None, "samples": f.end - f.start} for f in r.fragments],
            "truth": truth, "joins_s": [float(j / hz) for j in s.joins], "semantic_s": [float(b / hz) for b in s.semantic_boundaries],
            "methods": decode_all(s, methods, taus, cfg, hz),
        })
        t = 0.0
        for fr in streams_out[-1]["fragments"]:
            fr["start_s"] = t; t += fr["samples"] / hz; fr["end_s"] = t
        print(f"{r.recipe_id}: {dur:.1f}s, {len(r.fragments)} fragments -> {mp4.name}", flush=True)
    (out_dir / "streams.json").write_text(json.dumps({"ckpt_hash": args.ckpt_hash, "classes": classes, "methods": methods, "taus": taus,
                                                        "streams": streams_out}, indent=1))
    print(f"wrote {out_dir / 'streams.json'}")


if __name__ == "__main__":
    main()
