#!/usr/bin/env python3
"""Independent A51-P source audit. No production selection/label helpers."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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

PARENT = ROOT / "runs/a51_discriminative_evidence_v1"
OUT = ROOT / "runs/a51_verified_direct_view_pilot_v1"
PUB = ROOT / "docs/results/2026-11-01_direct_view_evidence/pilot"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.time()
    selection = json.loads((OUT / "selection.private.json").read_text())
    bound = json.loads((OUT / "bound.private.json").read_text())
    meta = json.loads((PARENT / "source/private_metadata_marker.json").read_text())
    chronology = json.loads((PARENT / "source/private_chronology.json").read_text())
    assert bound["selection_sha256"] == sha(OUT / "selection.private.json")
    assert selection["parent_chronology_sha256"] == sha(PARENT / "source/private_chronology.json")
    assert selection["original_metadata_sha256"] == sha(PARENT / "source/private_metadata_marker.json")
    assert selection["split"] == meta["split"]
    fit, diagnostic = set(meta["split"]["fit"]), set(meta["split"]["diagnostic"])
    assert not fit & diagnostic
    annotations, request_lookup = defaultdict(list), {r["id"]: r for r in meta["requests"]}
    for r in meta["records"]:
        annotations[(r["directory_label"], r["video"])].append(r)
    chrono = {r["request_id"]: r for r in chronology["markers"]}
    good, bad = set(), set()
    # Independent explicit eight-hypothesis reference, including all five failures.
    for rid, r in request_lookup.items():
        axis = chrono[rid]
        passed = True
        for position in (r["centre"], axis["native_pts_index"]):
            for origin in (0, 1):
                for inclusive in (False, True):
                    if position is None:
                        passed = False; continue
                    values = set()
                    for a in annotations[(r["directory_label"], r["video"])]:
                        if a["start"] - origin <= position < a["end"] - origin + int(inclusive):
                            values.add(a["true_label"])
                    if values != {r["true_label"]}:
                        passed = False
        assert passed == axis["passed"]
        (good if passed else bad).add(rid)
    assert len(good) == 702 and len(bad) == 5
    assert {r["id"] for r in selection["requests"]} == good
    assert {r["id"] for r in selection["excluded_supervised_requests"]} == bad
    assert all(request_lookup[r]["role"] == "fit" for r in bad)
    assert {r["id"] for r in bound["requests"]} == good
    assert {r["id"] for r in bound["requests"] if r["role"] == "diagnostic"} == {r["id"] for r in meta["requests"] if r["role"] == "diagnostic"}
    by_video, cache = {}, {}
    for v in bound["videos"]:
        key = v["folder"], v["video"]
        media = PARENT / "source/videos" / key[0] / key[1]
        assert sha(media) == v["sha256"]
        cp = ROOT / v["cache"]
        assert sha(cp) == v["cache_sha256"]
        z = np.load(cp)
        assert z["frames"].dtype == np.uint8 and z["frames"].shape[1:] == (224, 224, 3)
        cache[key] = {int(i): f for i, f in zip(z["ordinals"], z["frames"])}
        by_video[key] = v
    video_hashes = [v["sha256"] for v in bound["videos"]]
    assert len(set(video_hashes)) == len(video_hashes)
    pixel_owners, context_keys = {}, set()
    for r in bound["requests"]:
        orig = request_lookup[r["id"]]
        assert r["patient"] == orig["patient"] and r["role"] == orig["role"] and r["true_label"] == orig["true_label"]
        key = r["directory_label"], r["video"]
        v = by_video[key]
        expected_frames = [max(0, min(v["decoded_frames"] - 1, r["centre"] + 3 * (i - 8))) for i in range(16)]
        assert r["frames"] == expected_frames == chrono[r["id"]]["context_frames"]
        pix = np.stack([cache[key][i] for i in expected_frames])
        digest = hashlib.sha256(pix.tobytes()).hexdigest()
        assert digest == r["pixel_sha256"]
        owner = r["patient"], r["role"], r["true_label"]
        assert digest not in pixel_owners or pixel_owners[digest] == owner
        pixel_owners[digest] = owner
        context_keys.update((r["video"], i) for i in expected_frames)
    # Fixed twelve unique videos per role, selected by frozen request digest.
    samples = []
    for role in ("fit", "diagnostic"):
        seen = set()
        for r in sorted([r for r in bound["requests"] if r["role"] == role], key=lambda r: r["id"]):
            key = r["directory_label"], r["video"]
            if key not in seen:
                samples.append(r); seen.add(key)
            if len(seen) == 12:
                break
        assert len(seen) == 12
    pixel_checks, decoded_axis_checks = 0, 0
    sample_by_video = defaultdict(list)
    for r in samples:
        sample_by_video[(r["directory_label"], r["video"])].append(r)
    for key, v in by_video.items():
        native_indices = []
        with av.open(str(PARENT / "source/videos" / key[0] / key[1])) as c:
            stream = c.streams.video[0]; stream.thread_type = "NONE"
            for ordinal, frame in enumerate(c.decode(stream)):
                value = Fraction(frame.pts) * Fraction(frame.time_base) * 30
                assert value.denominator == 1
                native_indices.append(value.numerator)
                for r in sample_by_video[key]:
                    if r["centre"] == ordinal:
                        processed = np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize((224, 224), Image.Resampling.BILINEAR))
                        assert np.array_equal(processed, cache[key][ordinal])
                        pixel_checks += 1
        assert native_indices == v["native_pts_indices"]
        decoded_axis_checks += 1
    assert pixel_checks == 24
    counts = {}
    for role in ("fit", "diagnostic"):
        rows = [r for r in bound["requests"] if r["role"] == role]
        counts[role + "_patients"] = len({r["patient"] for r in rows})
        counts[role + "_requests"] = len(rows)
        counts[role + "_disagreement_patients"] = len({r["patient"] for r in rows if r["true_label"] != r["directory_label"]})
        counts[role + "_discordant_requests"] = sum(r["true_label"] != r["directory_label"] for r in rows)
        for view in ("A4C", "A5C"):
            counts[role + "_videos_" + view] = len({r["video"] for r in rows if r["true_label"] == view})
            counts[role + "_directory_requests_" + view] = sum(r["directory_label"] == view for r in rows)
            counts[role + "_true_requests_" + view] = sum(r["true_label"] == view for r in rows)
    assert counts == selection["counts"]
    x = np.asarray([r["geometry"] for r in bound["requests"] if r["role"] == "fit"], float)
    geom = bound["geometry_control"]
    scale = x.std(0); scale[scale < 1e-12] = 1.
    assert np.max(np.abs(np.asarray(geom["fit_mean"]) - x.mean(0))) < 1e-12
    assert np.max(np.abs(np.asarray(geom["fit_scale"]) - scale)) < 1e-12
    assert geom["diagnostic_scores_computed"] is False
    overlap = sum((r["video"], r["centre"]) in context_keys for r in selection["excluded_supervised_requests"])
    rec = {"passed": True, "run_id": "A51-P", "original_full_source_passed": False,
           "original_marker_labels_reconstructed": 707, "retained_markers": 702, "excluded_fit_supervised_requests": 5,
           "unchanged_diagnostic_requests": 133, "all_retained_pixel_window_hashes_verified": 702,
           "complete_released_video_SHA_and_native_axis_checks": decoded_axis_checks,
           "raw_to_processed_pixel_checks": pixel_checks, "partition_counts": counts,
           "excluded_marker_frames_also_in_retained_context": overlap,
           "bound_sha256": sha(OUT / "bound.private.json"), "model_forwards": 0,
           "scope": "Source identity, tested export-axis label agreement, fixed exclusions/patients/contexts, all cache memberships and24pixel reconstructions. No anatomical truth or publisher benchmark reproduction.",
           "wall_seconds": time.time() - started,
           "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    PUB.mkdir(parents=True, exist_ok=True)
    (PUB / "independent_source.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
