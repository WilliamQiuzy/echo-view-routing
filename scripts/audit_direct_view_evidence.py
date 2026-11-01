#!/usr/bin/env python3
"""Independent A51 metadata reconstruction. No production selection helpers."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/a51_discriminative_evidence_v1"
PUB = ROOT / "docs/results/2026-11-01_direct_view_evidence"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata"])
    parser.parse_args()
    start = time.time()
    repo = OUT / "source/repo"
    q1_file = OUT / "source/private_metadata.json"
    q2_file = OUT / "source/private_metadata_marker.json"
    q1, q2 = (json.loads(p.read_text()) for p in (q1_file, q2_file))
    rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    assert rev == q1["source_revision"] == q2["source_revision"]
    blobs = {}
    for name, expected in q1["metadata_sha256"].items():
        b = subprocess.check_output(["git", "-C", str(repo), "show", rev + ":data/metadata/" + name])
        assert hashlib.sha256(b).hexdigest() == expected
        blobs[name] = b
    meta = list(csv.DictReader(io.StringIO(blobs["Patients_and_videos.csv"].decode("utf-8-sig"))))
    ann = list(csv.DictReader(io.StringIO(blobs["Annotations_per_frame.csv"].decode("utf-8-sig"))))
    owners = {x["video_filename"]: (x["patient_id"], x["view"]) for x in meta}
    assert len(owners) == len(meta)
    assert q1["split"] == q2["split"] and q1["reserve"] == q2["reserve"]
    assert q2["q1_private_metadata_sha256"] == sha(q1_file)
    fs, hs = set(q2["split"]["fit"]), set(q2["split"]["diagnostic"])
    assert not fs & hs and not (fs | hs) & set(q2["reserve"]["reserved"])
    eligible_patients = sorted(fs | hs, key=lambda p: hashlib.sha256(("A51-source-v1|" + p).encode()).hexdigest())
    assert set(eligible_patients[::5]) == hs
    rebuilt, seen = [], set()
    for r in ann:
        name, folder = r["Video_FileName"], r["View_Folder"]
        patient, owner_folder = owners[name[:-4]]
        assert folder == owner_folder
        if patient not in fs | hs or folder not in ("A4C", "A5C") or r["Label"] not in ("A4C", "A5C"):
            continue
        start_i, end_i = int(r["Start_Frame"]), int(r["End_Frame"])
        centres = list(range(start_i + 1, end_i - 1, 15))
        if len(centres) > 8:
            centres = [centres[i * (len(centres) - 1) // 7] for i in range(8)]
        for centre in centres:
            frames = tuple(centre + 3 * (i - 8) for i in range(16))
            key = name, frames
            if key in seen:
                continue
            seen.add(key)
            for origin in (0, 1):
                for inclusive in (False, True):
                    assert start_i - origin <= centre < end_i - origin + int(inclusive)
            rebuilt.append((name, folder, patient, r["Label"], centre, frames))
    observed = [(r["video"], r["directory_label"], r["patient"], r["true_label"], r["centre"], tuple(r["frames"])) for r in q2["requests"]]
    assert rebuilt == observed
    totals = {}
    for role, patients in [("fit", fs), ("diagnostic", hs)]:
        rows = [r for r in rebuilt if r[2] in patients]
        totals[role + "_patients"] = len({r[2] for r in rows})
        totals[role + "_requests"] = len(rows)
        totals[role + "_disagreement_patients"] = len({r[2] for r in rows if r[1] != r[3]})
        totals[role + "_discordant_requests"] = sum(r[1] != r[3] for r in rows)
        for label in ("A4C", "A5C"):
            totals[role + "_videos_" + label] = len({r[0] for r in rows if r[3] == label})
            totals[role + "_directory_requests_" + label] = sum(r[1] == label for r in rows)
            totals[role + "_true_requests_" + label] = sum(r[3] == label for r in rows)
    assert totals == q2["counts"]
    manifest = json.loads((PUB / "marker_metadata_manifest.json").read_text())
    assert manifest["private_metadata_sha256"] == sha(q2_file)
    assert totals == manifest["partition_counts"]
    result = {"passed": True, "scope": "independent metadata/source/unchanged-patient/marker-label/count reconstruction only",
              "publisher_benchmark_split_reproduction": False, "q1_preserved_sha256": sha(q1_file),
              "q2_private_metadata_sha256": sha(q2_file), "all_reconstructed_requests": len(rebuilt),
              "partition_counts": totals, "source_numeric_gate_pass": manifest["metadata_minima_pass"],
              "pixels_or_model_execution": 0, "wall_seconds": time.time() - start,
              "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    dest = PUB / "independent_metadata.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
