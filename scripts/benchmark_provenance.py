#!/usr/bin/env python
"""Training-data overlap audit of every weight in the frozen EV9V test benchmark: for each component, the recordings
its training samples actually came from (actual manifests / lists), intersected with the official test split (888
clips), the benchmark's 800 five-family test clips and their sessions (file prefix without the clip index). External
pretraining whose data list is not available is reported as unknown, never as zero. Writes counts and evidence paths
only; identities stay on the server.

  python scripts/benchmark_provenance.py --config configs/experiments/frozen_external_benchmark_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.benchmark_manifest as bm  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402

PROTOCOL = "runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json"
MANIFESTS = {"V01 Ordered (A08's start)": "runs/raw_video_context_v1/manifests/RV_s{s}.npz", "A08 all_s{s} continuation": "runs/video_sequence_objective_v1/manifests/VSO_s{s}.npz",
             "A13 H head (A11 sealed manifest)": "runs/video_sustained_switch_v1/manifests/VSS_s{s}.npz"}


def manifest_recordings(path: Path) -> tuple[set, int]:
    with np.load(path) as z:
        vid, videos = z["vid"], z["videos"]
    used = videos[np.unique(vid)] if np.issubdtype(vid.dtype, np.integer) else np.unique(vid)
    return {str(v) for v in used}, int(np.prod(z_shape(path)[:2]))


def z_shape(path: Path) -> tuple:
    with np.load(path) as z:
        return z["c"].shape


def row(component: str, recs: set, samples: str, evidence: list[str], T888: set, T800: set, S_test: set, note: str = "") -> dict:
    rs = {patient_of(v) for v in recs}
    return {"component": component, "training_samples": samples, "training_recordings": len(recs), "training_sessions": len(rs),
            "test_recordings_official": len(T888), "test_recordings_benchmark": len(T800), "test_sessions": len(S_test),
            "recordings_in_official_test": len(recs & T888), "recordings_in_benchmark_test": len(recs & T800), "sessions_shared_with_test": len(rs & S_test),
            "status": "verified zero intersection" if not (recs & T888 or rs & S_test) else "intersection found", "evidence": evidence, "note": note}


def audit(cfg: dict) -> dict:
    paths = load_paths(); man = pd.read_csv(paths.manifest_path, usecols=["split", "video_id"], dtype=str); T888 = set(man.loc[man["split"] == "test", "video_id"])
    T800 = {x["id"] for x in bm.load(cfg)["items"] if x["kind"] == "native"}; S_test = {patient_of(v) for v in T888}; train = set(man.loc[man["split"] == "train", "video_id"]); out = []
    proto = json.loads((ROOT / PROTOCOL).read_text()); ptrain = set(proto["train"]["clips"])
    out.append(row("model-development protocol train allowlist (all A-series training draws from it)", ptrain, f"{len(ptrain)} clips", [PROTOCOL], T888, T800, S_test,
                   f"allowlist subset of the official train split: {ptrain <= train}"))
    for s in (0, 2):
        for name, pat in MANIFESTS.items():
            p = ROOT / pat.format(s=s); recs, n = manifest_recordings(p); w = z_shape(p)
            out.append(row(f"{name.replace('{s}', str(s))}, seed {s}", recs, f"{w[0]} updates x {w[1]} windows = {n} training windows", [pat.format(s=s)], T888, T800, S_test,
                           f"within the protocol allowlist: {recs <= ptrain}"))
    ctx = tl.Ctx(load_cfg(SimpleNamespace(config="configs/experiments/video_local_router_v1.yaml", set=[])), "configs/experiments/video_local_router_v1.yaml")
    pool = tl.pool(ctx); prec = {r[0] for q in pool for r in q["refs"]}
    for s in (0, 2):
        with np.load(ROOT / "runs/video_local_router_v1/features" / f"candidates_s{s}.npz") as z:
            fit = sorted({int(q) for q, f in zip(z["seq"], z["fit"]) if f}); nfit = int(z["fit"].sum())
        frec = {r[0] for i in fit for r in pool[i]["refs"]}
        out.append(row(f"A14 P router seed {s}: training pool (fit-set normalisation and fit zones)", frec, f"{nfit} fit zones from {len(fit)} pool sequences; pool {len(pool)} sequences, {len(prec)} recordings",
                       ["scripts/train_local_router.py pool()", f"runs/video_local_router_v1/features/candidates_s{s}.npz"], T888, T800, S_test, f"pool within the protocol allowlist: {prec <= ptrain}"))
    asf = json.loads((ROOT / "models_frozen/asformer_official/v2_frameorder/multi_train_79f4a41af6e3.json").read_text()); arec = {f["video_id"] for r in asf for f in r["fragments"]}
    out.append(row("ASFormer v2_frameorder training streams", arec, f"{len(asf)} training streams", ["models_frozen/asformer_official/v2_frameorder/multi_train_79f4a41af6e3.json"], T888, T800, S_test,
                   f"within the official train split: {arec <= train}"))
    out.append(row("B0 79f4a41af6e3 (ASFormer's feature encoder)", {v for v in train}, "official train split (loader rule: manifest split == train, five-family labels)",
                   ["runs/20260913-033411-b0_frame_encoder-b072725b-s0/config.resolved.yaml", "data/manifests/ev9v_native.csv"], T888, T800, S_test,
                   "derived from the training loader's split rule; no per-run clip list was saved"))
    for name, lst, strip in (("STFM numeric s100 training list", "data/stfm/EchoData/train.txt", None), ("EchoViewCLIP stage-1 training list", "data/echoviewclip/train.txt", ".mp4")):
        ids = set()
        for line in (ROOT / lst).read_text().splitlines():
            if line.strip():
                v = Path(line.split()[0].replace("\\", "/")).name; ids.add(v[: -len(strip)] if strip and v.endswith(strip) else v)
        out.append(row(name, ids, f"{len(ids)} clips", [lst], T888, T800, S_test, f"all names found in the EV9V manifest: {ids <= set(man['video_id'])}; within the official train split: {ids <= train}"))
    val = set(man.loc[man["split"] == "validation", "video_id"])
    out.append(row("EV9V validation split (not training: decoder, threshold and router choices; STFM / EchoViewCLIP early stopping)", val, f"{len(val)} clips", ["data/manifests/ev9v_native.csv"],
                   T888, T800, S_test, "supplementary; selection data, not gradient training"))
    ext = [{"component": "EchoPrime MViTv2-S video encoder (V01/A08 initialisation)", "status": "unknown", "evidence": ["third_party/EchoPrime/model_data/weights/echo_prime_encoder.pt"],
            "note": "pretrained by the EchoPrime authors on their own echo studies; no clip list available; overlap with EV9V cannot be verified"},
           {"component": "EchoPrime released view classifier", "status": "unknown", "evidence": ["models_frozen/echoprime_view_classifier/release_v1.0.0/view_classifier.pt"],
            "note": "released weights; training clip list not available; overlap with EV9V cannot be verified"},
           {"component": "CLIP ViT-B/16 (EchoViewCLIP initialisation)", "status": "unknown", "evidence": ["third_party/EchoViewCLIP"],
            "note": "OpenAI web-image pretraining; no list; EV9V file timestamps (2022) post-date it, which makes inclusion of these recordings implausible but is not a verification"},
           {"component": "ImageNet ResNet-18 (B0 and STFM backbone initialisation)", "status": "unknown", "evidence": ["runs/20260913-033411-b0_frame_encoder-b072725b-s0/config.resolved.yaml"],
            "note": "ImageNet-1k (2012) natural images; EV9V file timestamps (2022) post-date it; not verified against a list"}]
    return {"training_rows": out, "external_pretraining": ext, "unit_notes": "recording = clip id; session = file prefix without the clip index (703 groups); patient identities unavailable, so patient overlap is unknown",
            "test_definition": {"official_test_clips": len(T888), "benchmark_test_clips": len(T800), "excluded": "88 PMPALA clips (no family5 label)", "test_sessions": len(S_test)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--config", required=True)
    a = ap.parse_args(); cfg = load_cfg(SimpleNamespace(config=a.config, set=[])); fb = cfg["frozen_external_benchmark"]; t0, started = time.time(), vsh.now(); ok, note = False, ""
    try:
        rec = {**audit(cfg), "config_sha256": sha256_file(ROOT / a.config), "time": vsh.now()}
        for p in (ROOT / fb["outputs"]["root"] / "provenance.json", ROOT / fb["outputs"]["results_dir"] / "provenance.json"):
            p.parent.mkdir(parents=True, exist_ok=True); vsh.write_json(p, rec)
        for r in rec["training_rows"]:
            print(f"{r['component'][:70]:70s} rec {r['training_recordings']:5d} test∩ {r['recordings_in_official_test']} bench∩ {r['recordings_in_benchmark_test']} sess∩ {r['sessions_shared_with_test']} | {r['note'][:80]}")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        vsh.ledger_append(ROOT / fb["budget"]["ledger"], "provenance audit (CPU)", "cpu", started, time.time() - t0, ok, note)


if __name__ == "__main__":
    main()
