#!/usr/bin/env python3
"""Distinct A51-P subset; parent all-marker qualification remains failed."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from scripts.prepare_direct_view_evidence import OUT as PARENT, PUB as REPORT, dump, sha, counts_for
from echo_routing.temporal import direct_view_evidence as dv

OUT = ROOT / "runs/a51_verified_direct_view_pilot_v1"
PUB = REPORT / "pilot"


def select():
    dest = OUT / "selection.private.json"
    if dest.exists():
        raise RuntimeError("pilot selection already frozen")
    mp, cp = PARENT / "source/private_metadata_marker.json", PARENT / "source/private_chronology.json"
    meta, chronology = json.loads(mp.read_text()), json.loads(cp.read_text())
    cr = {r["request_id"]: r for r in chronology["markers"]}
    assert len(cr) == len(meta["requests"]) == 707
    selected, excluded = [], []
    for r in meta["requests"]:
        axis = cr[r["id"]]
        assert r["role"] == axis["role"] and r["true_label"] == axis["true_label"]
        if axis["passed"]:
            selected.append({**r, "frames": axis["context_frames"], "native_pts_index": axis["native_pts_index"],
                             "context_duration_seconds": axis["context_duration_seconds"]})
        else:
            assert r["role"] == "fit"
            excluded.append({**r, "supervised_status": "UNKNOWN", "reason": axis["reason"]})
    assert len(selected) == 702 and len(excluded) == 5
    counts = counts_for(selected, meta["split"])
    numeric = {k: counts[k] >= minimum for k, minimum in dv.MINIMUMS.items()}
    assert all(numeric.values()), numeric
    assert counts["fit_requests"] == 569 and counts["diagnostic_requests"] == 133
    private = {"run_id": "A51-P", "original_metadata_sha256": sha(mp), "parent_chronology_sha256": sha(cp),
               "split": meta["split"], "requests": selected, "excluded_supervised_requests": excluded,
               "counts": counts, "original_full_source_passed": False}
    dump(dest, private)
    public = {"run_id": "A51-P", "original_full_source_passed": False, "original_markers": 707,
              "retained_markers": 702, "excluded_fit_supervised_requests": 5, "excluded_diagnostic_requests": 0,
              "patients_and_diagnostic_identities_unchanged": True, "partition_counts": counts,
              "numerical_minima": numeric, "selection_sha256": sha(dest),
              "parent_metadata_sha256": sha(mp), "parent_chronology_sha256": sha(cp),
              "source_models_or_scores": 0, "full_subset_binding_yet_passed": False}
    dump(PUB / "selection.json", public)
    print(json.dumps(public))


def bind():
    selection_file = OUT / "selection.private.json"
    freeze = json.loads((OUT / "selection_freeze.json").read_text())
    assert freeze["selection_sha256"] == sha(selection_file)
    if (OUT / "bound.private.json").exists():
        raise RuntimeError("verified subset already bound")
    private = json.loads(selection_file.read_text())
    chronology = json.loads((PARENT / "source/private_chronology.json").read_text())
    all_meta = json.loads((PARENT / "source/private_metadata_marker.json").read_text())
    by_video = defaultdict(list)
    for r in private["requests"]:
        by_video[(r["directory_label"], r["video"])].append(r)
    receipt = {"selection_sha256": sha(selection_file), "videos": [], "requests": []}
    cache_root = OUT / "pixel_cache"; cache_root.mkdir(exist_ok=True)
    for vi, media in enumerate(chronology["videos"]):
        key = media["folder"], media["video"]
        if key not in by_video:
            continue
        path = PARENT / "source/videos" / key[0] / key[1]
        assert sha(path) == media["sha256"]
        need = sorted({i for r in by_video[key] for i in r["frames"]})
        selected, pts, n = {}, [], 0
        with av.open(str(path)) as c:
            s = c.streams.video[0]; s.thread_type = "NONE"
            width, height, fps = s.width, s.height, float(s.average_rate)
            for i, frame in enumerate(c.decode(s)):
                n = i + 1
                value = Fraction(frame.pts) * Fraction(frame.time_base) * 30
                assert value.denominator == 1
                pts.append(value.numerator)
                if i in need:
                    selected[i] = np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize((224, 224), Image.Resampling.BILINEAR)).copy()
        assert pts == media["native_pts_indices"] and n == media["decoded_frames"]
        assert set(selected) == set(need)
        frames = np.stack([selected[i] for i in need])
        cache = cache_root / (hashlib.sha256((key[0] + "/" + key[1]).encode()).hexdigest() + ".npz")
        if cache.exists():
            raise RuntimeError("partial pilot cache retained; explicit recovery required")
        np.savez_compressed(cache, ordinals=np.asarray(need, np.int64), frames=frames)
        receipt["videos"].append({**media, "width": width, "height": height, "fps": fps,
                                  "cache": str(cache.relative_to(ROOT)), "cache_sha256": sha(cache), "cached_frames": len(need)})
        for r in by_video[key]:
            fr = np.stack([selected[i] for i in r["frames"]]); v = fr.astype(np.float32)
            labels = [z for z in all_meta["records"] if z["directory_label"] == key[0] and z["video"] == key[1]]
            context = [{q["true_label"] for q in labels if q["start"] + 1 <= i < q["end"] - 1} for i in r["frames"]]
            receipt["requests"].append({**r, "pixel_sha256": hashlib.sha256(fr.tobytes()).hexdigest(),
                "all_context_same_label": all(x == {r["true_label"]} for x in context),
                "clipped_slots": sum(i != j for i, j in zip(r["raw_frames"], r["frames"])),
                "geometry": [width, height, fps, n, float(v.mean()), float(v.std()),
                             float((v.max(-1) - v.min(-1)).mean()), float(np.abs(np.diff(v, axis=0)).mean())]})
        dump(OUT / "binding_progress.json", {"pid": os.getpid(), "complete_videos": vi + 1, "total_videos": len(chronology["videos"]),
             "cached_frames": sum(v["cached_frames"] for v in receipt["videos"]), "model_forwards": 0, "updated_epoch": time.time()})
    dv.check_duplicate_isolation(receipt["requests"])
    indexed = {r["id"]: r for r in receipt["requests"]}
    receipt["requests"] = [indexed[r["id"]] for r in private["requests"]]
    assert len(receipt["requests"]) == 702
    pixels = defaultdict(set)
    for r in receipt["requests"]:
        pixels[r["pixel_sha256"]].add(r["true_label"])
    assert all(len(v) == 1 for v in pixels.values()), "identical pixel requests have conflicting manual labels"
    fit = [r for r in receipt["requests"] if r["role"] == "fit"]
    x = np.asarray([r["geometry"] for r in fit], np.float64)
    y = np.asarray([r["true_label"] == "A5C" for r in fit], np.int64)
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-12] = 1.
    m = LogisticRegression(C=1., max_iter=1000, random_state=2026110151, solver="lbfgs").fit((x - mean) / scale, y)
    receipt["geometry_control"] = {"fit_mean": mean.tolist(), "fit_scale": scale.tolist(), "coef": m.coef_.tolist(),
                                   "intercept": m.intercept_.tolist(), "C": 1., "constant_prior": float(y.mean()), "diagnostic_scores_computed": False}
    context_keys = {(r["video"], i) for r in receipt["requests"] for i in r["frames"]}
    excluded_overlap = sum((r["video"], r["centre"]) in context_keys for r in private["excluded_supervised_requests"])
    dump(OUT / "bound.private.json", receipt)
    result = {"run_id": "A51-P", "subset_bound": True, "original_full_source_passed": False,
              "retained_requests": 702, "excluded_supervised_requests": 5,
              "excluded_marker_frames_occurring_in_retained_context": excluded_overlap,
              "cache_bytes": sum((ROOT / v["cache"]).stat().st_size for v in receipt["videos"]),
              "selected_videos": len(receipt["videos"]), "unique_selected_frames": sum(v["cached_frames"] for v in receipt["videos"]),
              "mixed_or_unknown_context_requests": sum(not r["all_context_same_label"] for r in receipt["requests"]),
              "clipped_slots": sum(r["clipped_slots"] for r in receipt["requests"]),
              "physical_context_duration_quantiles_seconds": np.quantile([r["context_duration_seconds"] for r in receipt["requests"]], [0, .25, .5, .75, 1]).tolist(),
              "bound_sha256": sha(OUT / "bound.private.json"), "geometry_fitted_fit_only": True,
              "source_diagnostic_predictions": 0, "model_forwards": 0}
    dump(PUB / "binding.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["select", "bind"]); args = p.parse_args()
    start, ok = time.time(), False
    try:
        {"select": select, "bind": bind}[args.stage](); ok = True
    finally:
        u = resource.getrusage(resource.RUSAGE_SELF)
        dump(OUT / (args.stage + "_process.json"), {"ok": ok, "pid": os.getpid(), "start_epoch": start, "end_epoch": time.time(),
             "wall_seconds": time.time() - start, "CPU_seconds": u.ru_utime + u.ru_stime, "GPU_job_wall_seconds": 0})
