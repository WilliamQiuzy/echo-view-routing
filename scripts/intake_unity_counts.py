#!/usr/bin/env python
"""post_a34_external_supervision_intake_v1: aggregate counts of the Unity Imaging labels actually held on the server.

  python scripts/intake_unity_counts.py --unity data/unity --out <json>

Joins labels-all.json, the official splits, the PNG cache and view_index_20220705.csv; counts label states per key for
A4C and A5C frames. Only aggregates are written (no file names or hashes). Label states follow the official
dataset_structure page: point / curve = coordinates; off = "definitely not in the image"; blurred = "might be in the
image, but there is no label available".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--unity", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); u = Path(a.unity)
    L = json.loads((u / "labels/labels/labels-all.json").read_text()); keys = json.loads((u / "labels/labels/keys.json").read_text())
    imgs = {f for _, _, fs in os.walk(u / "png-cache/png-cache") for f in fs if f.endswith(".png")}
    split = {ln.strip(): s for s in ("train", "tune", "ival") for ln in open(u / f"labels/labels/labels-{s}.txt") if ln.strip()}
    rows = list(csv.DictReader(open(u / "view_index_20220705.csv"))); views = defaultdict(set)
    for r in rows:
        views[r["file"]].add(r["view"])
    cine = lambda f: f.rsplit("-", 1)[0]  # noqa: E731
    held = [f for f in L if f in imgs]
    out = {"labels_entries": len(L), "png_files": len(imgs), "labels_with_png": len(held), "labels_without_png": len(L) - len(held),
           "splits_of_labelled_png": dict(Counter(split.get(f, "none") for f in held)), "label_states_all": dict(Counter(x.get("type") for f in held for x in L[f]["labels"].values())),
           "view_index": {"rows": len(rows), "unique_files": len(views), "files_listed_more_than_once": sum(1 for c in Counter(r["file"] for r in rows).values() if c > 1),
                          "files_with_conflicting_views": sum(1 for v in views.values() if len(v) > 1), "views_all": dict(Counter(next(iter(v)) for v in views.values() if len(v) == 1)),
                          "files_not_in_labelled_png": dict(Counter(next(iter(v)) for f, v in views.items() if f not in set(held) and len(v) == 1))},
           "views_of_labelled_png": dict(Counter(next(iter(views[f])) if len(views.get(f, ())) == 1 else "missing" for f in held)),
           "file_prefix": dict(Counter(f.split("-")[0] for f in held)), "plane_flags_in_keys": sorted({p for k in keys.values() for p in k if p.startswith("plane_")}),
           "tasks_naming_a5c": sorted({p for k in keys.values() for p in k if "a5c" in p}), "aortic_keys": [k for k in keys if k.startswith("ao")], "per_view": {}}
    for view in ("a4c", "a5c"):
        fs = [f for f in held if views.get(f) == {view}]; st = defaultdict(Counter)
        for f in fs:
            for k, x in L[f]["labels"].items():
                st[k][x.get("type")] += 1
        cines = Counter(cine(f) for f in fs)
        out["per_view"][view] = {"frames": len(fs), "cines": len(cines), "frames_per_cine_max": max(cines.values()) if cines else 0, "splits": dict(Counter(split.get(f, "none") for f in fs)),
                                 "cines_by_split": {s: len({cine(f) for f in fs if split.get(f) == s}) for s in ("train", "tune", "ival")},
                                 "keys_with_coordinates": {k: c["point"] + c["curve"] for k, c in st.items() if c["point"] + c["curve"] > 0},
                                 "aortic_key_states": {k: dict(st[k]) for k in st if k.startswith("ao")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True)); print(json.dumps(out["per_view"], indent=1)[:3000])


if __name__ == "__main__":
    main()
