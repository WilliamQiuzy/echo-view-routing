#!/usr/bin/env python
"""Round-1 blinded annotation pilot (task 4): reader package plus a separate key.

Reader package (<out>/package, zipped to <out>/annotation_pilot_<version>.zip): index.html (demo/annotator), clips.js
(anonymous id, frame count, fps), clips/<id>/000001.jpg... (frames in true temporal order) and the reader instructions.
Nothing in the package names the source clip, its label, any model output or why it was chosen. The key
(<out>/key/pilot_key.csv) maps ids to sources and strata and is never given to readers.

Composition (EV9V validation/test; discovery clips excluded; at most 2 clips per patient):
- random: 30 clips from the two-stage random sample (one clip per random patient), for prevalence;
- sustained: every clip either recognizer calls sustained/mixed at theta 0.5, topped up to 25 from theta 0.3;
- repeated: 15 flagged by both recognizers and 15 by one only;
- duplicates: 8 candidates shown again under new ids (test-retest), at least 20 positions from the first showing;
- practice: 2 training-split clips shown first and not scored.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402

ALPHABET = list("ACDEFHJKMNPRTUVWXY34679")  # no look-alike characters
REP, SUS = ("repeated", "mixed"), ("sustained", "mixed")
SIZES = {"random": 30, "sustained": 25, "repeated_both": 15, "repeated_one": 15, "duplicates": 8, "practice": 2}
MAX_PER_PATIENT, MIN_DUP_GAP = 2, 20
REPO = Path(__file__).resolve().parents[1]


def clip_table(screen_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(screen_csv)
    df = df[~df["discovery"]]
    w = df.pivot(index="video_id", columns="recognizer", values=["stratum", "contrast", "sides", "n_ret"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = df.drop_duplicates("video_id").set_index("video_id")[["session", "split", "label", "duration_s"]]
    t = meta.join(w)
    differ = lambda s: s.fillna("").map(lambda x: ">" in x and x.split(">")[0] != x.split(">")[1])  # noqa: E731
    t["sus05"] = t["stratum_B0"].isin(SUS) | t["stratum_EchoPrime"].isin(SUS)
    t["sus03"] = ((t["contrast_B0"].astype(float) >= 0.3) & differ(t["sides_B0"])) | \
                 ((t["contrast_EchoPrime"].astype(float) >= 0.3) & differ(t["sides_EchoPrime"]))
    t["sus03"] &= t["stratum_B0"] != "short"
    t["rep_both"] = t["stratum_B0"].isin(REP) & t["stratum_EchoPrime"].isin(REP)
    t["rep_one"] = (t["stratum_B0"].isin(REP) | t["stratum_EchoPrime"].isin(REP)) & ~t["rep_both"]
    return t


class Picker:
    def __init__(self, table: pd.DataFrame, rng: np.random.Generator) -> None:
        self.t, self.rng, self.rows, self.per_patient = table, rng, [], {}

    def take(self, pool: list[str], n: int, stratum: str) -> None:
        chosen = {r["video_id"] for r in self.rows}
        got = 0
        for vid in self.rng.permutation(sorted(pool)):
            if got >= n:
                break
            pat = patient_of(vid)
            if vid in chosen or self.per_patient.get(pat, 0) >= MAX_PER_PATIENT:
                continue
            self.rows.append({"video_id": vid, "pilot_stratum": stratum}); chosen.add(vid)
            self.per_patient[pat] = self.per_patient.get(pat, 0) + 1; got += 1


def anon_ids(n: int, rng: np.random.Generator) -> list[str]:
    """Distinct codes in generation order (a set would make the order depend on Python's string hashing)."""
    ids: list[str] = []
    while len(ids) < n:
        c = "".join(rng.choice(ALPHABET, 6))
        if c not in ids:
            ids.append(c)
    return ids


def order_with_gap(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Random order in which every duplicate is >= MIN_DUP_GAP positions from its first showing."""
    for _ in range(5000):
        perm = [rows[i] for i in rng.permutation(len(rows))]
        pos = {r["anon_id"]: i for i, r in enumerate(perm)}
        first = {r["video_id"]: pos[r["anon_id"]] for r in perm if not r["duplicate_of"]}
        if all(abs(pos[r["anon_id"]] - first[r["video_id"]]) >= MIN_DUP_GAP for r in perm if r["duplicate_of"]):
            return perm
    raise RuntimeError("could not place duplicates far enough apart")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the blinded annotation pilot")
    ap.add_argument("--screen-dir", required=True, help="folder with screen_ev9v.csv and screen_ev9v_summary.json")
    ap.add_argument("--out", required=True); ap.add_argument("--version", default="v1"); ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args(); paths = load_paths(); rng = np.random.default_rng(args.seed)
    sd, out = Path(args.screen_dir), Path(args.out)
    t = clip_table(sd / "screen_ev9v.csv")
    summary = json.loads((sd / "screen_ev9v_summary.json").read_text())
    pk = Picker(t, rng)
    pk.take(summary["random_sample"], SIZES["random"], "random")
    pk.take(list(t.index[t["sus05"]]), SIZES["sustained"], "sustained")
    n_sus = sum(r["pilot_stratum"] == "sustained" for r in pk.rows)
    pk.take(list(t.index[t["sus03"] & ~t["sus05"]]), SIZES["sustained"] - n_sus, "sustained")
    pk.take(list(t.index[t["rep_both"] & ~t["sus03"]]), SIZES["repeated_both"], "repeated_both")
    pk.take(list(t.index[t["rep_one"] & ~t["sus03"]]), SIZES["repeated_one"], "repeated_one")
    rows = [dict(r, duplicate_of="") for r in pk.rows]
    cand = [r for r in rows if r["pilot_stratum"] != "random"]
    dups = [dict(cand[i], duplicate_of="first showing") for i in rng.choice(len(cand), SIZES["duplicates"], replace=False)]
    man = pd.read_csv(paths.manifest_path)
    train = man[(man["dataset"] == "EV9V") & (man["split"] == "train") & man["family5_label"].notna() & man["duration_s"].between(3, 6)]
    practice = [{"video_id": v, "pilot_stratum": "practice", "duplicate_of": ""} for v in rng.choice(sorted(train["video_id"]), SIZES["practice"], replace=False)]
    ids = anon_ids(len(rows) + len(dups) + len(practice), rng)
    for r, i in zip(rows + dups + practice, ids):
        r["anon_id"] = i
    first_id = {r["video_id"]: r["anon_id"] for r in rows}
    for r in dups:
        r["duplicate_of"] = first_id[r["video_id"]]
    ordered = practice + order_with_gap(rows + dups, rng)

    pkg = out / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "clips").mkdir(parents=True)
    fdirs = man.set_index("video_id")["frames_dir"]; fps = man.set_index("video_id")["fps_playback"]
    clips = []
    for r in ordered:
        src = list_frame_paths(Path(fdirs[r["video_id"]])); dst = pkg / "clips" / r["anon_id"]; dst.mkdir()
        for k, p in enumerate(src, 1):
            shutil.copyfile(p, dst / f"{k:06d}.jpg")
        clips.append({"id": r["anon_id"], "n_frames": len(src), "fps": float(fps[r["video_id"]]), "practice": r["pilot_stratum"] == "practice"})
    (pkg / "clips.js").write_text("window.PILOT = " + json.dumps({"version": args.version, "clips": clips}, indent=0) + ";\n")
    shutil.copyfile(REPO / "demo" / "annotator" / "index.html", pkg / "index.html")
    shutil.copyfile(REPO / "docs" / "annotation" / "reader_instructions.md", pkg / "标注说明_reader_instructions.md")
    shutil.make_archive(str(out / f"annotation_pilot_{args.version}"), "zip", pkg)

    key = pd.DataFrame([{"order": i, **r} for i, r in enumerate(ordered)])
    key["patient"] = key["video_id"].map(patient_of)
    key = key.merge(t.drop(columns=["session"]), left_on="video_id", right_index=True, how="left")
    (out / "key").mkdir(parents=True, exist_ok=True)
    key.to_csv(out / "key" / "pilot_key.csv", index=False)
    print(key["pilot_stratum"].value_counts().to_string())
    print(f"clips {len(clips)} (scored {sum(not c['practice'] for c in clips)}), patients {key['patient'].nunique()}, "
          f"frames {sum(c['n_frames'] for c in clips)}; package {pkg}; key {out / 'key' / 'pilot_key.csv'}")


if __name__ == "__main__":
    main()
