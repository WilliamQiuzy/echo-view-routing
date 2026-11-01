#!/usr/bin/env python3
"""CPU-only all-marker conservative native-PTS versus ordinal binding check."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
from echo_routing.temporal.direct_view_evidence import robust_marker_label
from scripts.prepare_direct_view_evidence import OUT, PUB, sha, dump


def main():
    start = time.time()
    dest = OUT / "source/private_chronology.json"
    if dest.exists():
        raise RuntimeError("chronology check already sealed; no repeat or source rescue")
    meta_file = OUT / "source/private_metadata_marker.json"
    private = json.loads(meta_file.read_text())
    acq = json.loads((OUT / "source/private_acquisition.json").read_text())
    by_video, labels = defaultdict(list), defaultdict(list)
    for r in private["requests"]:
        by_video[(r["directory_label"], r["video"])].append(r)
    for r in private["records"]:
        labels[(r["directory_label"], r["video"])].append(r)
    rows, video_records, failure_counts, gap_hist = [], [], Counter(), Counter()
    for record in acq["files"]:
        key = record["folder"], record["video"]
        path = OUT / "source/videos" / key[0] / key[1]
        assert sha(path) == record["sha256"]
        indices, valid, errors = [], True, []
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            fps, header_n = float(s.average_rate), int(s.frames)
            if abs(fps - 30.) > .001:
                valid = False; errors.append("fps")
            for frame in c.decode(s):
                if frame.pts is None or frame.time_base is None:
                    valid = False; errors.append("missing_pts"); indices.append(None); continue
                position = Fraction(frame.pts) * Fraction(frame.time_base) * 30
                if position.denominator != 1:
                    valid = False; errors.append("noninteger_pts"); indices.append(None); continue
                indices.append(position.numerator)
        if not indices or indices[0] != 0:
            valid = False; errors.append("nonzero_or_absent_start")
        deltas = [b - a for a, b in zip(indices, indices[1:]) if a is not None and b is not None]
        if any(x <= 0 for x in deltas):
            valid = False; errors.append("nonmonotone")
        gap_hist.update(deltas)
        for r in by_video[key]:
            ordinal = r["centre"]
            native_index = indices[ordinal] if 0 <= ordinal < len(indices) else None
            passed = valid and native_index is not None and robust_marker_label(labels[key], ordinal, native_index, r["true_label"])
            reason = "passed" if passed else "chronology" if not valid else "missing_marker" if native_index is None else "label_axis_disagreement_or_ambiguity"
            if not passed:
                failure_counts[reason] += 1
            context = [max(0, min(len(indices) - 1, i)) for i in r["raw_frames"]]
            duration = None
            if context and indices[context[0]] is not None and indices[context[-1]] is not None:
                duration = (indices[context[-1]] - indices[context[0]]) / 30.
            rows.append({"request_id": r["id"], "patient": r["patient"], "role": r["role"], "true_label": r["true_label"],
                         "ordinal": ordinal, "native_pts_index": native_index, "passed": passed, "reason": reason,
                         "context_frames": context, "context_duration_seconds": duration})
        video_records.append({**record, "header_frames": header_n, "decoded_frames": len(indices), "native_pts_indices": indices,
                              "valid_chronology": valid, "errors": sorted(set(errors))})
    assert len(rows) == len(private["requests"]) == 707
    dump(dest, {"metadata_sha256": sha(meta_file), "videos": video_records, "markers": rows})
    by_role = {role: {"total": sum(r["role"] == role for r in rows),
                      "passed": sum(r["role"] == role and r["passed"] for r in rows),
                      "failed": sum(r["role"] == role and not r["passed"] for r in rows)} for role in ("fit", "diagnostic")}
    result = {"passed": not failure_counts, "exact_selected_videos": len(video_records), "exact_selected_markers": len(rows),
              "markers_passed": sum(r["passed"] for r in rows), "markers_failed": sum(not r["passed"] for r in rows),
              "failures": dict(failure_counts), "by_role": by_role,
              "videos_with_native_PTS_gaps": sum(any(b - a != 1 for a, b in zip(v["native_pts_indices"], v["native_pts_indices"][1:]) if a is not None and b is not None) for v in video_records),
              "header_decoded_count_disagreements": sum(v["header_frames"] != v["decoded_frames"] for v in video_records),
              "native_PTS_step_histogram": dict(gap_hist), "private_receipt_sha256": sha(dest),
              "scope": "All frozen markers tested under both exact-native-PTS and decoded-ordinal axes and all four annotation origin/endpoint hypotheses; no chosen offset or reselected marker.",
              "source_qualification_pass": not failure_counts, "source_or_target_models": 0, "GPU_forwards": 0,
              "wall_seconds": time.time() - start,
              "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    dump(PUB / "chronology_qualification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
