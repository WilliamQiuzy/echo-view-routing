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
from echo_routing.compose.recipe_builder import Fragment, Pool, Recipe, make_pair_recipes, recipe_to_dict  # noqa: E402
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
EV_FAMILY = {"PLHLA": "PLAX", "PASA": "PSAX", "PMASA": "PSAX", "PMVLSA": "PSAX", "PPMLSA": "PSAX", "A4C": "A4C", "A5C": "A5C", "SC4C": "SC4C"}


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
        labels, score_prob = decode_with_prob(m, stream.prob, cfg, stream.feat, stream.recipe.recipe_id)
        ivs = route(score_prob, labels, tau if tau is not None else np.inf, method=pol["score"], min_len=int(pol["min_len"]))
        out[m] = {"tau": tau, "intervals": [{"start_s": iv.start / hz, "end_s": iv.end / hz, "view": int(iv.view), "score": float(iv.score),
                                            "accepted": bool(iv.accepted)} for iv in ivs]}
    return out


def main() -> None:
    ap = base_parser("Render and decode demo streams")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--mstcn-checkpoint", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--methods", default="b_file,b0,b1,b2,b3,b4", help="comma list of registry ids to decode (empty = none)")
    ap.add_argument("--pred-dirs", default=None, help="key=dir,... for lookup decoders (mstcn_official, asformer_official, stfm_windowed, echoviewclip_windowed, echoprime_windowed)")
    ap.add_argument("--skip-render", action="store_true", help="reuse existing MP4s")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); ont = load_ontology()
    if args.mstcn_checkpoint:
        cfg.setdefault("mstcn", {})["checkpoint"] = args.mstcn_checkpoint
    for kv in (args.pred_dirs or "").split(","):
        if kv:
            k, v = kv.split("=", 1); cfg.setdefault(k, {})["pred_dir"] = v
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
    methods = [m for m in args.methods.split(",") if m]
    rng = np.random.default_rng(args.seed)

    recipes: list[tuple[str, str, Recipe]] = []
    used: set[str] = set()
    fam_of = {c: classes.index(EV_FAMILY[c]) for c in EV_FAMILY}

    def pick(raw_code: str | None = None, family: str | None = None, lo: int = 25, hi: int = 100):
        """One unused test cine of a raw code (or any code of a family) with lo..hi samples."""
        cand = test[(test["n_samples"] >= lo) & (test["n_samples"] <= hi) & (~test["video_id"].isin(used))]
        if raw_code:
            cand = cand[cand["raw_label"] == raw_code]
        else:
            cand = cand[cand["label_index"] == classes.index(family)]
        if len(cand) == 0:
            cand = test[(~test["video_id"].isin(used)) & ((test["raw_label"] == raw_code) if raw_code else (test["label_index"] == classes.index(family)))]
        row = cand.iloc[int(rng.integers(len(cand)))]; used.add(row.video_id)
        return row

    def frag(row, variant="orig", length=None):
        n = int(row.n_samples); L = int(min(length or n, n)); start = int(rng.integers(0, n - L + 1)) if L < n else 0
        return Fragment(row.video_id, int(row.label_index), variant, start, start + L)

    # 1) one untouched native cine per raw view code of the five-family task
    for code in ["PLHLA", "PASA", "PMASA", "PMVLSA", "PPMLSA", "A4C", "A5C", "SC4C"]:
        r = pick(raw_code=code, lo=40, hi=110)
        recipes.append(("native", f"Single view · {EV_FAMILY[code]} ({code})", Recipe(f"native_{code}", "native", (frag(r),))))
    # 2) same-view joins (two patients), one per family; PLAX and A4C also with a gamma edit on the second half
    for fam in classes:
        a, b = pick(family=fam, lo=25, hi=60), pick(family=fam, lo=25, hi=60)
        recipes.append(("same_none", f"Same view, two patients · {fam}", Recipe(f"same_none_{fam}", "same_none", (frag(a, length=30), frag(b, length=30)))))
    for fam in ["PLAX", "A4C", "PSAX"]:
        a, b = pick(family=fam, lo=25, hi=60), pick(family=fam, lo=25, hi=60)
        recipes.append(("same_edit", f"Same view, second half gamma-edited · {fam}", Recipe(f"same_edit_{fam}", "same_edit", (frag(a, length=30), frag(b, cfg["recipes"]["edit_variant"], length=30)))))
    # 3) genuine view changes across several pairs, alternating hard join and edited second half
    for i, (fa, fb) in enumerate([("PLAX", "A4C"), ("A4C", "PSAX"), ("PSAX", "PLAX"), ("A4C", "A5C"), ("A5C", "SC4C"), ("SC4C", "A4C"), ("PLAX", "PSAX"), ("A5C", "A4C")]):
        a, b = pick(family=fa, lo=25, hi=60), pick(family=fb, lo=25, hi=60); edited = i % 2 == 1
        recipes.append(("diff_edit" if edited else "diff_none", f"View change · {fa} → {fb}{' (second half gamma-edited)' if edited else ''}",
                        Recipe(f"diff_{fa}_{fb}", "diff_edit" if edited else "diff_none", (frag(a, length=30), frag(b, cfg["recipes"]["edit_variant"] if edited else "orig", length=30)))))
    # 4) multi-fragment streams
    for k, seq in enumerate([["A4C", "A4C", "PLAX", "PSAX"], ["PSAX", "A5C", "A4C", "SC4C", "PLAX"], ["PLAX", "PLAX", "A4C", "A5C", "A4C", "PSAX"]]):
        frags = []
        for j, fam in enumerate(seq):
            r = pick(family=fam, lo=20, hi=60)
            frags.append(frag(r, cfg["recipes"]["edit_variant"] if (j % 3 == 1) else "orig", length=25))
        recipes.append(("multi", f"Multi-view stream · {' → '.join(seq)}", Recipe(f"multi_{k:02d}", "multi", tuple(frags))))

    streams_out = []
    for kind, title, r in recipes:
        s = render(r, loader)
        mp4 = out_dir / f"{r.recipe_id}.mp4"
        dur = (sum(f.end - f.start for f in r.fragments) / hz) if (args.skip_render and mp4.is_file()) else render_mp4(r, loader, paths.ev9v_images, mp4)
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
    (out_dir / "recipes.json").write_text(json.dumps([recipe_to_dict(r) for _, _, r in recipes], indent=1))
    (out_dir / "streams.json").write_text(json.dumps({"ckpt_hash": args.ckpt_hash, "classes": classes, "methods": methods, "taus": taus,
                                                        "streams": streams_out}, indent=1))
    print(f"wrote {out_dir / 'streams.json'}")


if __name__ == "__main__":
    main()
