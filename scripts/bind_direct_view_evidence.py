#!/usr/bin/env python3
"""A51 source ordinal/pixel binding. CPU-only, no model or heldout prediction."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from collections import Counter, defaultdict
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
from echo_routing.temporal import direct_view_evidence as dv
from scripts.prepare_direct_view_evidence import OUT, PUB, sha, dump, counts_for


def main():
    start = time.time()
    private = json.loads((OUT / "source/private_metadata_marker.json").read_text())
    acquisition = json.loads((OUT / "source/private_acquisition.json").read_text())
    if (OUT / "source/private_bound.json").exists():
        raise RuntimeError("source pixels already bound; no repeated source scan")
    by_video = defaultdict(list)
    for r in private["requests"]:
        by_video[(r["directory_label"], r["video"])].append(r)
    receipt = {"source_metadata_sha256": sha(OUT / "source/private_metadata_marker.json"),
               "acquisition_sha256": sha(OUT / "source/private_acquisition.json"),
               "videos": [], "requests": []}
    records = private["records"]
    pending = OUT / "source/binding_progress.json"
    cache_root = OUT / "source/pixel_cache"
    cache_root.mkdir(exist_ok=True)
    for vi, media in enumerate(acquisition["files"]):
        folder, video = media["folder"], media["video"]
        path = OUT / "source/videos" / folder / video
        assert sha(path) == media["sha256"]
        requests = by_video[(folder, video)]
        wanted = {0} | {x for r in requests for x in r["raw_frames"] if x >= 0}
        selected, all_pts, n, last = {}, [], 0, None
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            fps, header_n = float(stream.average_rate), stream.frames
            width, height = stream.width, stream.height
            for i, frame in enumerate(container.decode(stream)):
                n = i + 1
                if frame.pts is not None:
                    all_pts.append(float(frame.pts * frame.time_base))
                if i in wanted:
                    selected[i] = np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize((224, 224), Image.Resampling.BILINEAR)).copy()
                last = frame
            if last is None:
                raise ValueError("empty released video")
            selected[n - 1] = np.asarray(Image.fromarray(last.to_ndarray(format="rgb24")).resize((224, 224), Image.Resampling.BILINEAR)).copy()
        # AVI advertised count is not a decoded ordinal inventory. Preserve the
        # discrepancy; the protocol's N is the actual decoded count below.
        if all_pts and (len(all_pts) != n or any(b <= a for a, b in zip(all_pts, all_pts[1:]))):
            raise ValueError("released ordinal/PTS chronology is not monotone")
        if all_pts and (abs(all_pts[0]) > 1e-6 or max(abs((b - a) - 1. / 30.) for a, b in zip(all_pts, all_pts[1:])) > 1e-6):
            raise ValueError("released ordinal axis has an unexplained PTS offset or gap")
        intervals = [(r["start"], r["end"]) for r in records if r["directory_label"] == folder and r["video"] == video]
        dv.validate_video_axis(fps, n, intervals)
        bound_requests = []
        for r in requests:
            # Rebuild clipping from actual decoded N; never infer it from CSV.
            rebuilt = dv.marker_requests(r, n_frames=n)
            matched = [q for q in rebuilt if q["centre"] == r["centre"]]
            assert len(matched) == 1
            br = {**r, **matched[0]}
            br["role"] = r["role"]; br["id"] = r["id"]
            assert br["frames"][8] == br["centre"]
            fr = np.stack([selected[x] for x in br["frames"]])
            br["pixel_sha256"] = hashlib.sha256(fr.tobytes()).hexdigest()
            values = fr.astype(np.float32)
            chroma = values.max(-1) - values.min(-1)
            br["geometry"] = [width, height, fps, n, float(values.mean()), float(values.std()),
                              float(chroma.mean()), float(np.abs(np.diff(values, axis=0)).mean())]
            ref_labels = []
            for x in br["frames"]:
                supported = {v["true_label"] for v in records if v["video"] == video and v["directory_label"] == folder and v["start"] + 1 <= x < v["end"] - 1}
                ref_labels.append(next(iter(supported)) if len(supported) == 1 else "UNKNOWN")
            br["context_label_support"] = ref_labels
            br["all_context_same_label"] = all(x == br["true_label"] for x in ref_labels)
            bound_requests.append(br)
        need = sorted({x for r in bound_requests for x in r["frames"]})
        frames = np.stack([selected[x] for x in need])
        cache = cache_root / (hashlib.sha256((folder + "/" + video).encode()).hexdigest() + ".npz")
        if cache.exists():
            raise RuntimeError("partial binding cache retained; explicit recovery required")
        np.savez_compressed(cache, ordinals=np.asarray(need, np.int64), frames=frames)
        entry = {**media, "decoded_frames": n, "header_frames": header_n, "header_minus_decoded": header_n - n,
                 "fps": fps, "width": width, "height": height,
                 "cache": str(cache.relative_to(ROOT)), "cache_sha256": sha(cache),
                 "cached_frames": len(need), "selected_pixel_sha256": hashlib.sha256(frames.tobytes()).hexdigest()}
        receipt["videos"].append(entry)
        receipt["requests"].extend(bound_requests)
        dump(pending, {"pid": os.getpid(), "complete_videos": vi + 1, "total_videos": len(acquisition["files"]),
                       "decoded_frames": sum(v["decoded_frames"] for v in receipt["videos"]),
                       "new_cache_bytes": sum((ROOT / v["cache"]).stat().st_size for v in receipt["videos"]),
                       "model_forwards": 0, "updated_epoch": time.time()})
    dv.check_duplicate_isolation(receipt["requests"])
    # Stable source-request order is inherited from frozen metadata, not scan order.
    indexed = {r["id"]: r for r in receipt["requests"]}
    assert len(indexed) == len(private["requests"])
    receipt["requests"] = [indexed[r["id"]] for r in private["requests"]]
    fit = [r for r in receipt["requests"] if r["role"] == "fit"]
    X = np.asarray([r["geometry"] for r in fit], np.float64)
    y = np.asarray([r["true_label"] == "A5C" for r in fit], np.int64)
    mean, scale = X.mean(0), X.std(0)
    scale[scale < 1e-12] = 1.
    logistic = LogisticRegression(C=1., max_iter=1000, random_state=2026110151, solver="lbfgs").fit((X - mean) / scale, y)
    receipt["geometry_control"] = {"names": ["width", "height", "fps", "decoded_frames", "mean", "std", "chroma", "temporal_absolute_difference"],
                                   "fit_mean": mean.tolist(), "fit_scale": scale.tolist(),
                                   "coef": logistic.coef_.tolist(), "intercept": logistic.intercept_.tolist(),
                                   "C": 1., "constant_prior": float(y.mean()), "diagnostic_scores_computed": False}
    dump(OUT / "source/private_bound.json", receipt)
    counts = counts_for(receipt["requests"], private["split"])
    assert counts == private["counts"]
    checks = {k: True for k in dv.REQUIRED_CHECKS}
    checks["released_test_exclusion"] = not json.loads((PUB / "source_scope.json").read_text())["prior_protected_project_test_assignment_found"]
    gate = dv.source_qualified(counts, checks)
    out = {"passed": gate["passed"], "source_qualification": gate, "checks": checks,
           "private_bound_sha256": sha(OUT / "source/private_bound.json"), "selected_videos": len(receipt["videos"]),
           "decoded_frames": sum(v["decoded_frames"] for v in receipt["videos"]),
           "unique_selected_frames": sum(v["cached_frames"] for v in receipt["videos"]),
           "resolutions": sorted({(v["width"], v["height"]) for v in receipt["videos"]}),
           "fps_values": sorted({v["fps"] for v in receipt["videos"]}),
           "AVI_header_count_disagreements": sum(v["header_frames"] != v["decoded_frames"] for v in receipt["videos"]),
           "partition_counts": counts, "context_same_view_requests": sum(r["all_context_same_label"] for r in receipt["requests"]),
           "mixed_or_unknown_context_requests": sum(not r["all_context_same_label"] for r in receipt["requests"]),
           "clipped_context_slots": sum(r["clipped_slots"] for r in receipt["requests"]),
           "source_cache_bytes": sum((ROOT / v["cache"]).stat().st_size for v in receipt["videos"]),
           "source_geometry_fitted_on_fit_only": True, "source_diagnostic_scores_computed": False,
           "annotation_axis_scope": "Exact released video names, decoded ordinal inventory, chronology and convention-interior labels; historical annotation exporter itself is not independently reconstructed.",
           "new_anatomy_ground_truth": False, "model_forwards": 0,
           "wall_seconds": time.time() - start,
           "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    dump(PUB / "source_qualification.json", out)
    print(json.dumps(out))


if __name__ == "__main__":
    started = time.time()
    ok = False
    try:
        main()
        ok = True
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        dump(OUT / "binding_process.json", {"ok": ok, "pid": os.getpid(), "start_epoch": started,
             "end_epoch": time.time(), "wall_seconds": time.time() - started,
             "CPU_seconds": usage.ru_utime + usage.ru_stime, "GPU_job_wall_seconds": 0})
