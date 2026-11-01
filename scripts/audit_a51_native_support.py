#!/usr/bin/env python3
"""A51 original native and run-to-window label support; no model execution."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import scripts.train_risk_constrained_a5c as t44
from scripts.prepare_direct_view_evidence import OUT, PUB, dump, sha


def main():
    start = time.time()
    config = "configs/experiments/video_risk_constrained_a5c_v1.yaml"
    ctx = t44.Ctx(t44.load_cfg(SimpleNamespace(config=config, set=[])), config)
    result = {"folds": {}, "model_forwards": 0, "optimizer_updates": 0,
              "source": "Original A31 protocol-training-source pools and A44 source0 memberships only",
              "label_scope": "Native clip labels propagated to all frames; constructed reference positions inherit source clip codes. Not independently reviewed local anatomy."}
    private_sets = []
    for fold, expected in [(0, (200, 67)), (1, (200, 75))]:
        F = t44.Fold(ctx, 0, fold)
        counts, src, sessions = {}, set(), set()
        for label, name, n_expected in [(2, "A4C", expected[0]), (3, "A5C", expected[1])]:
            native = [i for i, seq in enumerate(F.seqs) if seq["kind"] == "native" and set(np.asarray(seq["labels"]).tolist()) == {label}]
            assert len(native) == n_expected
            windows, unique_raw_total, referenced_raw_total, edge_repeats = 0, 0, 0, 0
            retained, dimensions = [], Counter()
            for i in native:
                seq = F.seqs[i]
                sources = {r[0] for r in seq["refs"]}
                assert len(sources) == 1
                video = next(iter(sources)); src.add(video); sessions.add(t44.session_of(video))
                raw = [int(r[1]) for r in seq["refs"]]
                assert all(b >= a for a, b in zip(raw, raw[1:]))
                assert all(float(r[2]) == 1. for r in seq["refs"])
                paths = F.fp(video)
                first = min(max(0, raw[0]), len(paths) - 1)
                with Image.open(paths[first]) as im:
                    w, h = im.size
                dimensions[(w, h)] += 1
                rw, rh = (int(256 * w / h), 256) if w >= h else (256, int(256 * h / w))
                retained.append(224. * 224. / (rw * rh))
                rws = np.flatnonzero(F.W["seq"] == i)
                windows += len(rws)
                for row in rws:
                    positions = F.W["s"][row]
                    refs = [int(seq["refs"][int(p)][1]) for p in positions]
                    marker = int(F.W["marker"][row])
                    assert int(seq["labels"][int(positions[marker])]) == label
                    unique_raw_total += len(set(refs)); referenced_raw_total += len(refs)
                    edge_repeats += len(refs) - len(set(refs))
            counts[name] = {"native_videos": len(native), "original_requests": int(windows),
                            "raw_frame_references": int(referenced_raw_total), "within_request_unique_raw_sum": int(unique_raw_total),
                            "within_request_repeated_slots": int(edge_repeats),
                            "raw_dimensions_counts": [{"width": k[0], "height": k[1], "videos": v} for k, v in sorted(dimensions.items())],
                            "processed_crop_raw_area_fraction_median": float(np.median(retained)),
                            "processed_crop_raw_area_fraction_min": float(min(retained)),
                            "processed_crop_raw_area_fraction_max": float(max(retained)),
                            "independent_frame_anatomy_labels": 0}
        support = {}
        for key, idx in F.L.items():
            if key not in ("positives", "negatives"):
                continue
            mismatches, context_mixed, masses = 0, 0, []
            expected_label = 3 if key == "positives" else 2
            for run in idx:
                a, b = map(int, F.pk["offsets"][int(run):int(run) + 2])
                rows = F.pk["rows"][a:b] - F.off
                weights = F.pk["weights"][a:b]
                mass = 0.
                for row, weight in zip(rows, weights):
                    seq = F.seqs[int(F.W["seq"][row])]
                    positions = F.W["s"][row]
                    marker = int(F.W["marker"][row])
                    centre_label = int(seq["labels"][int(positions[marker])])
                    mass += float(weight) * (centre_label != expected_label)
                    mismatches += int(centre_label != expected_label)
                    context_mixed += int(any(int(seq["labels"][int(p)]) != expected_label for p in positions))
                masses.append(mass)
            support[key] = {"runs": len(idx), "membership_rows": int(sum(F.pk["offsets"][int(j) + 1] - F.pk["offsets"][int(j)] for j in idx)),
                            "rows_with_marker_reference_code_different_from_majority_run_target": mismatches,
                            "rows_with_any_context_code_different_from_run_target": context_mixed,
                            "runs_with_any_off_target_marker_mass": int(sum(x > 1e-12 for x in masses)),
                            "off_target_marker_weight_mass_mean_per_run": float(np.mean(masses)),
                            "off_target_marker_weight_mass_max_per_run": float(max(masses))}
        result["folds"][str(fold)] = {"native": counts, "native_distinct_recordings": len(src), "native_sessions": len(sessions),
                                         "A44_training_run_label_support": support,
                                         "pool_sha256": sha(ctx.a31.out / f"pool_{fold}.npy")}
        private_sets.append((src, sessions))
    assert not private_sets[0][0] & private_sets[1][0]
    assert not private_sets[0][1] & private_sets[1][1]
    result.update({"recordings_and_sessions_disjoint": True, "verified_patient_disjointness": False,
                   "wall_seconds": time.time() - start,
                   "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime})
    dump(PUB / "native_support.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
