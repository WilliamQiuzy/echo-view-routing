#!/usr/bin/env python3
"""A51 private source intake with public aggregate receipts; no model execution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal import direct_view_evidence as dv

OUT = ROOT / "runs/a51_discriminative_evidence_v1"
PUB = ROOT / "docs/results/2026-11-01_direct_view_evidence"
REVISION = "5e853ea4bb3eee8bb103fa44ad6ddb13fc63a5eb"
PINS = {
    "Annotations_per_frame.csv": "8e65f91588e9e3212cfbe0449854cb2e014a41cd388f89fc8f1c2ca8df5e4ac8",
    "Patients_and_videos.csv": "716f583c1f7af09cba8582813a3d7659cf79348f76fabe335f003dca37e8eddc",
}
ARCHIVES = {
    "A4C": ("5d20a547ea96e77e718bbcd40b8a0b0519b035da28dd1efbc3b3af94aae3b9b2", 789100831),
    "A5C": ("f8ed8dfa1f3ca88874aed085c6eef3fad99db81aad4af3720e121f2887743aa4", 639194286),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git_bytes(name):
    return subprocess.check_output(["git", "-C", str(OUT / "source/repo"), "show", REVISION + ":" + name])


def counts_for(requests, split):
    counts = {}
    for role, patients in split.items():
        rows = [r for r in requests if r["patient"] in patients]
        counts[role + "_patients"] = len({r["patient"] for r in rows})
        counts[role + "_requests"] = len(rows)
        counts[role + "_disagreement_patients"] = len({r["patient"] for r in rows if r["true_label"] != r["directory_label"]})
        counts[role + "_discordant_requests"] = sum(r["true_label"] != r["directory_label"] for r in rows)
        for label in dv.SOURCE_VIEWS:
            counts[role + "_videos_" + label] = len({r["video"] for r in rows if r["true_label"] == label})
            counts[role + "_directory_requests_" + label] = sum(r["directory_label"] == label for r in rows)
            counts[role + "_true_requests_" + label] = sum(r["true_label"] == label for r in rows)
    return counts


def metadata():
    if (OUT / "source/private_metadata.json").exists():
        raise RuntimeError("metadata already selected; never silently replace a cohort")
    repo = OUT / "source/repo"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    assert revision == REVISION
    blobs = {name: git_bytes("data/metadata/" + name) for name in PINS}
    for name, b in blobs.items():
        assert hashlib.sha256(b).hexdigest() == PINS[name]
    readme, license_ = git_bytes("README.md"), git_bytes("LICENSE")
    assert b"CC BY-NC 4.0" in readme and b"NonCommercial" in license_
    meta = list(csv.DictReader(io.StringIO(blobs["Patients_and_videos.csv"].decode("utf-8-sig"))))
    ann = list(csv.DictReader(io.StringIO(blobs["Annotations_per_frame.csv"].decode("utf-8-sig"))))
    records = dv.join_annotations(meta, ann)
    reserve = dv.reserve_publisher_candidates(records)
    patients = sorted({r["patient"] for r in records if r["directory_label"] in dv.SOURCE_VIEWS} - set(reserve["reserved"]))
    split = dv.patient_split(patients)
    requests, seen = [], set()
    for rec in records:
        if rec["patient"] not in patients:
            continue
        for r in dv.interval_requests(rec):
            key = (r["video"], tuple(r["frames"]))
            if key in seen:
                continue
            seen.add(key)
            r["role"] = "fit" if r["patient"] in split["fit"] else "diagnostic"
            r["id"] = hashlib.sha256(repr(key).encode()).hexdigest()
            requests.append(r)
    counts = counts_for(requests, split)
    assert max(counts.get("fit_directory_requests_" + c, 0) for c in dv.SOURCE_VIEWS) <= 2400
    private = {"source_revision": revision, "metadata_sha256": PINS, "records": records,
               "reserve": reserve, "split": split, "requests": requests, "counts": counts}
    dump(OUT / "source/private_metadata.json", private)
    public = {
        "stage": "metadata_only_no_pixels_or_models", "source_revision": revision,
        "source_url": "https://gitlab.com/JSarmientoA/echoUNAL", "license": "CC-BY-NC-4.0 research-only",
        "source_readme_sha256": hashlib.sha256(readme).hexdigest(), "source_license_sha256": hashlib.sha256(license_).hexdigest(),
        "publisher_code_sha256": hashlib.sha256(git_bytes("data/code/Classify.py")).hexdigest(),
        "metadata_sha256": PINS, "metadata_rows": len(meta), "annotation_rows": len(ann),
        "annotation_videos": len({r["video"] for r in records}),
        "metadata_views": dict(Counter(r["view"] for r in meta)),
        "interval_cross_table": [{"folder": a, "label": b, "intervals": n} for (a, b), n in sorted(Counter((r["directory_label"], r["true_label"]) for r in records).items())],
        "quarantined_patients_all_views": len(reserve["reserved"]), "own_partition_only": True,
        "publisher_benchmark_split_reproduced": False,
        "publisher_split_limitation": reserve["limitation"],
        "private_metadata_sha256": sha(OUT / "source/private_metadata.json"),
        "partition_counts": counts, "archives": ARCHIVES,
        "all_source_checks_passed": False, "no_anatomical_visibility_truth_claim": True,
    }
    dump(PUB / "metadata_manifest.json", public)
    print(json.dumps(public))


def download_archive(view):
    expected, length = ARCHIVES[view]
    dest = OUT / "source" / (view + ".zip")
    if dest.exists():
        if dest.stat().st_size == length and sha(dest) == expected:
            return dest
        raise RuntimeError("existing archive is not the frozen source; retain for failure audit")
    # Public GitLab LFS endpoint; presigned action URLs are never printed.
    payload = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": expected, "size": length}]}
    req = urllib.request.Request("https://gitlab.com/JSarmientoA/echoUNAL.git/info/lfs/objects/batch",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/vnd.git-lfs+json", "Accept": "application/vnd.git-lfs+json"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = json.load(response)
    action = data["objects"][0]["actions"]["download"]
    req = urllib.request.Request(action["href"], headers=action.get("header", {}))
    h, n, start = hashlib.sha256(), 0, time.time()
    part = dest.with_suffix(".zip.part")
    if part.exists():
        raise RuntimeError("partial download retained; explicit same-hash recovery required")
    with urllib.request.urlopen(req, timeout=60) as response, part.open("xb") as f:
        while b := response.read(2**20):
            f.write(b); h.update(b); n += len(b)
            if n > length or time.time() - start > 900:
                raise RuntimeError("source byte/wall bound exceeded")
    if n != length or h.hexdigest() != expected:
        raise RuntimeError("whole publisher archive SHA/size failure")
    part.replace(dest)
    return dest


def acquire():
    frozen = json.loads((OUT / "metadata_freeze.json").read_text())
    assert frozen["private_metadata_sha256"] == sha(OUT / "source/private_metadata.json")
    private = json.loads((OUT / "source/private_metadata.json").read_text())
    required = {(r["directory_label"], r["video"]) for r in private["requests"]}
    media = OUT / "source/videos"
    receipt = {"revision": REVISION, "archives": {}, "files": [], "quarantined_pixels_decoded": 0}
    for view in dv.SOURCE_VIEWS:
        archive = download_archive(view)
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            avi = [p for p in members if p.filename.lower().endswith(".avi") and not p.is_dir()]
            # Do not accept any path traversal, duplicate basename or huge zip bomb.
            names = [Path(p.filename).name for p in avi]
            if len(set(names)) != len(names) or any(".." in Path(p.filename).parts or Path(p.filename).is_absolute() for p in members):
                raise RuntimeError("unsafe or ambiguous archive inventory")
            if sum(p.file_size for p in members) > 1_500_000_000:
                raise RuntimeError("archive unpack bound")
            receipt["archives"][view] = {"sha256": sha(archive), "bytes": archive.stat().st_size, "released_avi_count": len(avi)}
            for item in avi:
                name = Path(item.filename).name
                if (view, name) not in required:
                    continue
                dest = media / view / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise RuntimeError("existing selected AVI requires receipt validation before resume")
                with z.open(item) as src, dest.open("xb") as f:
                    while b := src.read(2**20):
                        f.write(b)
                if dest.stat().st_size != item.file_size:
                    raise RuntimeError("extracted source size mismatch")
                receipt["files"].append({"folder": view, "video": name, "bytes": dest.stat().st_size, "sha256": sha(dest)})
        dump(OUT / "source/private_acquisition.json", receipt)
        archive.unlink()  # complete publisher hash and each retained AVI are now durable
    observed = {(r["folder"], r["video"]) for r in receipt["files"]}
    if observed != required:
        raise RuntimeError("fixed source members missing; no replacement allowed")
    hashes = [r["sha256"] for r in receipt["files"]]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("duplicate source videos; no silent dropping")
    public = {"stage": "complete_archives_hashed_selected_AVIs_retained_no_decode",
              "archives": receipt["archives"], "retained_videos": len(receipt["files"]),
              "retained_bytes": sum(r["bytes"] for r in receipt["files"]),
              "all_selected_members_present": observed == required,
              "distinct_full_video_sha256": len(set(hashes)),
              "private_receipt_sha256": sha(OUT / "source/private_acquisition.json"),
              "quarantined_pixels_decoded": 0, "model_forwards": 0}
    dump(PUB / "acquisition.json", public)
    print(json.dumps(public))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata", "acquire"])
    args = parser.parse_args()
    wall = time.time(); usage = resource.getrusage(resource.RUSAGE_SELF)
    ok = False
    try:
        {"metadata": metadata, "acquire": acquire}[args.stage]()
        ok = True
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        dump(OUT / (args.stage + "_process.json"), {"stage": args.stage, "ok": ok, "pid": os.getpid(),
             "start_epoch": wall, "end_epoch": time.time(), "wall_seconds": time.time() - wall,
             "CPU_seconds": after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime,
             "GPU_job_wall_seconds": 0})


if __name__ == "__main__":
    main()
