"""Writes the training data definitions of the release from the research repository's files:

  ev9v_recordings.json.gz   the recordings of the frame cache, in order: key, split, label, number of source frames
                            (goes to src/echoview/data/)
  plans/groups_s{0,1,2}.npz   the stage-2 and stage-3 sampling plans: groups of four neighbouring windows
  plans/windows_s{0,1,2}.npz  the stage-1 sampling plans: single windows, inside one recording or mixed

Runs inside the research repository (branch robust-upgrade), which is not part of this release:

    cd <research repository> && ECHO_INPUT=centre_crop_legacy python <this file> <research root> <output folder>

Recording names are replaced by keys (the first 16 hex digits of the SHA-256 of the EV9V folder name). Only what the
trainers read is kept: recording, frame number and gamma per slot, the marker slot, the label and the kind of window.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R))
from robust import cache  # noqa: E402

SOURCES = {"groups": "runs/video_sequence_objective_v1/manifests/VSO_s{seed}.npz",
           "windows": "runs/raw_video_context_v1/manifests/RV_s{seed}.npz"}


def recording_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


rows = cache.clip_rows(R)
train = {r["name"] for r in rows if r["split"] == "train"}
recordings = [{"key": recording_key(r["name"]), "split": r["split"], "label": int(r["label"]), "n_frames": int(r["n_frames"])} for r in rows]
if len({r["key"] for r in recordings}) != len(recordings):
    raise SystemExit("two recordings share a key")
document = {"format": "echo-view-routing recordings v1", "source": "EV9V (bgx666/EV9V, CC BY 4.0)",
            "recording_key": "first 16 hex digits of the SHA-256 of the EV9V recording folder name (UTF-8)",
            "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"], "recordings": recordings}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plans").mkdir(exist_ok=True)
payload = json.dumps(document, separators=(",", ":")).encode()
(OUT / "ev9v_recordings.json.gz").write_bytes(gzip.compress(payload, 9, mtime=0))
report = {"recordings": {s: sum(r["split"] == s for r in recordings) for s in ("train", "selection", "calibration")},
          "rows": int(rows[-1]["start"] + rows[-1]["count"]), "plans": {}}
for kind, pattern in SOURCES.items():
    for seed in (0, 1, 2):
        with np.load(R / pattern.format(seed=seed), allow_pickle=False) as z:
            names = [str(n) for n in z["videos"]]
            used = sorted(set(z["vid"].reshape(-1).tolist()))
            if any(names[i] not in train for i in used):
                raise SystemExit(f"{kind} plan of seed {seed} reads a recording outside the training split")
            arrays = {"recordings": np.array([recording_key(n) for n in names]), "vid": z["vid"].astype(np.int32),
                      "frame": z["frame"].astype(np.int32), "gamma": z["gamma"].astype(np.float32),
                      "marker": z["marker"].astype(np.int8), "target": z["target"].astype(np.int8), "kind": z["kind"].astype(np.int8)}
        target = OUT / "plans" / f"{kind}_s{seed}.npz"
        np.savez_compressed(target, **arrays)
        report["plans"][target.name] = {"updates": int(arrays["vid"].shape[0]), "windows_per_update": int(arrays["vid"].shape[1]),
                                        "recordings_used": len(used), "mixed_windows": int((arrays["kind"] == 1).sum()),
                                        "gammas": sorted({round(float(g), 3) for g in np.unique(arrays["gamma"])}),
                                        "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
(OUT / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report))
