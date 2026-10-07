#!/usr/bin/env python
"""Check and connect the frozen feature sets of the L1 + ASRF round (plan Task 1;
configs/experiments/l1_asrf_integration_v1.yaml). No training and no score.

A (matched control) reads the existing E1 cache (cache_v2/features/16efc85be645). B reads one new standard cache per L1
seed, built here:
- train: the fan224 round's export (report_fan224_visual.py --phase train-features), checked file by file: 3,364
  recordings, orig and gamma090, 512-d fp16 features, frame indices equal to the E1 cache's; t_sec copied from the E1
  cache;
- validation: every 10 Hz sample of the protocol's validation clips in both variants, extracted here with the frozen L1
  weights through the legacy input;
- scores: logit = the frozen L1 head on the stored fp16 feature; prob = softmax(logit). The export's stored
  probabilities are a check only.
Then: parity of the online encoders against the stored features (E1: export reference fp32 and E1 cache; L1: the
export), and the A/B alignment of every training and validation stream (same recipes, sources, labels, joins, time
stamps; only features and scores differ). Results go to runs/l1_asrf_integration_v1/inputs.json (server).

  python scripts/prepare_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.features.cache import (VideoFeatures, load_video_features, read_index, save_video_features,  # noqa: E402
                                         variant_dir, write_index, write_meta)

NS = "l1_asrf_integration_v1"
VARIANTS = {"orig": 1.0, "gamma090": 0.9}
DIM, N_TRAIN = 512, 3364


# ----------------------------------------------------------------------------------------------- pure checks
def check_inputs(la: dict, root: Path) -> dict:
    out = {}
    for key in ("protocol", "e1_feature_manifest", "visual_export", "visual_reference", "probe", "fan224_config"):
        spec = la["inputs"][key]
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {key} {spec['path']}: sha256 {got[:12]} != {spec['sha256'][:12]}")
        out[key] = got
    return out


def check_export_file(z: dict, frame_idx: np.ndarray, dim: int = DIM) -> None:
    """One video of the L1 train export: both variants, (n, dim) fp16 features, (n, 5) probabilities, the cache's
    frame indices in the same order."""
    for v in VARIANTS:
        for part in ("feat", "prob"):
            if f"{v}_{part}" not in z:
                raise ValueError(f"missing {v}_{part}")
    if not np.array_equal(np.asarray(z["frame_idx"]), np.asarray(frame_idx)):
        raise ValueError("frame indices differ from the E1 cache (missing, extra or reordered frames)")
    n = len(frame_idx)
    for v in VARIANTS:
        f, p = np.asarray(z[f"{v}_feat"]), np.asarray(z[f"{v}_prob"])
        if f.shape != (n, dim) or f.dtype != np.float16:
            raise ValueError(f"{v} features {f.shape} {f.dtype}, expected ({n}, {dim}) float16")
        if p.shape != (n, 5):
            raise ValueError(f"{v} probabilities {p.shape}, expected ({n}, 5)")
        if not np.isfinite(f.astype(np.float32)).all():
            raise ValueError(f"{v} features are not finite")


def check_clip_sets(index: pd.DataFrame, proto: dict) -> None:
    """The cache holds every protocol train and validation clip and no other split."""
    need = set(proto["train"]["clips"]) | {c for h in ("selection", "calibration") for c in proto["validation"][h]["clips"]}
    have = set(index["video_id"])
    if need - have:
        raise ValueError(f"{len(need - have)} protocol clips missing")
    if set(index["split"]) - {"train", "validation"} or have - need:
        raise ValueError("rows outside the protocol's train and validation clips (no test row may enter)")


def index_order(e1_index: pd.DataFrame, rows: dict) -> list[str]:
    """The E1 index's video order; the new cache must hold exactly the same videos."""
    order = e1_index["video_id"].tolist()
    if set(order) != set(rows) or len(order) != len(rows):
        raise ValueError(f"the new cache holds {len(rows)} videos, the E1 index {len(order)}; they must be the same set")
    return order


def scores(head: torch.nn.Linear, feat16: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """logit = head(stored fp16 feature) in fp32; prob = softmax(logit)."""
    with torch.no_grad():
        lg = head(torch.from_numpy(np.asarray(feat16, np.float16).astype(np.float32)).to(device)).float()
        return lg.cpu().numpy(), torch.softmax(lg, -1).cpu().numpy()


def stream_signature(s) -> dict:
    """Everything of a rendered stream that must be identical between two feature sets."""
    return {"recipe": s.recipe.recipe_id, "fragments": [(f.video_id, f.label, f.variant, f.start, f.end) for f in s.recipe.fragments],
            "labels": np.asarray(s.labels, np.int64).tolist(), "joins": np.asarray(s.joins, int).tolist(),
            "semantic": np.asarray(s.semantic_boundaries, int).tolist(), "n": int(len(s.labels))}


def compare_streams(a: list, b: list) -> int:
    """A and B streams: equal signatures in the same order and equal lengths; the features must differ somewhere."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} against {len(b)} streams")
    differ = 0
    for x, y in zip(a, b):
        if stream_signature(x) != stream_signature(y):
            raise ValueError(f"stream {x.recipe.recipe_id} differs between the feature sets")
        if x.feat.shape != y.feat.shape:
            raise ValueError(f"stream {x.recipe.recipe_id}: feature shapes differ")
        differ += int(not np.array_equal(x.feat, y.feat))
    return differ


def cache_key(final_sha: str, seed: int) -> str:
    return hashlib.sha256(json.dumps({"model": f"L1_s{seed}", "final_sha256": final_sha, "input": "legacy", "frame_order": "numeric (cache_v2)"},
                                     sort_keys=True).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------- server steps
def l1_model(la: dict, fv: dict, paths, seed: int, device):
    from scripts.report_fan224_visual import load_trained
    spec = la["inputs"]["l1_models"][seed]; frozen = paths.repo_root / spec["frozen"]
    man = json.loads((frozen / "MANIFEST.json").read_text()); files = {f["file"]: f["sha256"] for f in man["files"]}
    if sha256_file(frozen / "final.pt") != spec["final_sha256"] or files.get("final.pt") != spec["final_sha256"]:
        raise SystemExit(f"L1_s{seed}: frozen final.pt differs from the config")
    run_man = json.loads((frozen / "run_manifest.json").read_text())
    if run_man["identity"] != spec["identity"] or run_man["arm"] != "L1" or int(run_man["seed"]) != seed or run_man["final_sha256"] != spec["final_sha256"]:
        raise SystemExit(f"L1_s{seed}: run manifest identity differs from the config")
    enc, head = load_trained(fv, paths, {"arm": "L1", "final_checkpoint": str(frozen / "final.pt")}, device)
    return enc, head, {"frozen": str(frozen), "final_sha256": spec["final_sha256"], "identity": spec["identity"], "transform": run_man.get("input_kind")}


def build_b_cache(la: dict, fv: dict, paths, seed: int, e1_idx: pd.DataFrame, e1_store, proto: dict, device, workers: int) -> tuple[dict, object, object]:
    """The B cache of one L1 seed; returns (record, the L1 encoder, the L1 head)."""
    from scripts.report_fan224_visual import extract
    from scripts.train_joint_visual_temporal import FramePaths
    from echo_routing.manifest import load_manifest
    enc, head, src = l1_model(la, fv, paths, seed, device); key = cache_key(src["final_sha256"], seed)
    root = paths.repo_root; export = root / la["inputs"]["l1_train_features"] / f"L1_s{seed}"
    meta = json.loads((export / "index.json").read_text())
    if meta.get("final_sha256") != src["final_sha256"] or meta.get("recordings") != N_TRAIN or meta.get("input") != "legacy":
        raise SystemExit(f"L1_s{seed}: the train export is incomplete or belongs to another model ({meta})")
    files = sorted(p.stem for p in export.glob("*.npz"))
    train = sorted(proto["train"]["clips"])
    if files != train:
        raise SystemExit(f"L1_s{seed}: export holds {len(files)} recordings, expected the {len(train)} protocol train clips")
    for v in VARIANTS:
        if variant_dir(paths.cache_root, key, v).exists():
            raise SystemExit(f"cache {key}/{v} exists; a new cache needs a new key")
    rows = {v: {} for v in VARIANTS}; prob_gap = 0.0; e1i = e1_idx.set_index("video_id")
    for vid in train:
        ref = e1_store.get(vid, "orig")
        with np.load(export / f"{vid}.npz") as z:
            zz = {k: z[k] for k in z.files}
        check_export_file(zz, ref.frame_idx)
        for v in VARIANTS:
            lg, pr = scores(head, zz[f"{v}_feat"], device); prob_gap = max(prob_gap, float(np.abs(pr - zz[f"{v}_prob"]).max()))
            path = save_video_features(variant_dir(paths.cache_root, key, v), VideoFeatures(vid, ref.frame_idx, ref.t_sec, zz[f"{v}_feat"], pr, lg))
            rows[v][vid] = {**e1i.loc[vid].to_dict(), "video_id": vid, "n_samples": int(len(ref.frame_idx)), "path": str(path)}
    val = sorted({c for h in ("selection", "calibration") for c in proto["validation"][h]["clips"]})
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in val})
    triples = [(vid, int(f), g) for vid in val for v, g in VARIANTS.items() for f in e1_store.get(vid, "orig").frame_idx]
    feats = extract(enc, triples, fp, "legacy", device, workers).astype(np.float16); k = 0
    for vid in val:
        ref = e1_store.get(vid, "orig"); n = len(ref.frame_idx)
        for v in VARIANTS:
            f16 = feats[k:k + n]; k += n; lg, pr = scores(head, f16, device)
            path = save_video_features(variant_dir(paths.cache_root, key, v), VideoFeatures(vid, ref.frame_idx, ref.t_sec, f16, pr, lg))
            rows[v][vid] = {**e1i.loc[vid].to_dict(), "video_id": vid, "n_samples": n, "path": str(path)}
    if k != len(feats):
        raise SystemExit("validation features misaligned")
    if prob_gap > float(la["b_cache"]["parity"]["export_prob_max_abs"]):
        raise SystemExit(f"L1_s{seed}: recomputed probabilities differ from the export by {prob_gap:.3e}")
    for v in VARIANTS:   # the E1 index's own row order: the training bank is built in index order (multi_bank)
        d = variant_dir(paths.cache_root, key, v); order = index_order(read_index(variant_dir(paths.cache_root, la["inputs"]["e1_cache_key"], v)), rows[v])
        write_index(d, [rows[v][vid] for vid in order])
        if read_index(d)["video_id"].tolist() != order:
            raise SystemExit(f"{key}/{v}: index order differs from the E1 cache")
        write_meta(d, {"ckpt_hash": key, "variant": v, "source": src, "train": "fan224 train export (fp16 features as stored)",
                       "validation": "extracted by prepare_l1_asrf.py with the frozen L1 weights, legacy input, fp32 inference, fp16 storage",
                       "scores": la["b_cache"]["scores"], "target_hz": 10.0})
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); check_clip_sets(idx, proto)
    e1_fman = json.loads((root / la["inputs"]["e1_feature_manifest"]["path"]).read_text())
    fman = {"encoder": {"name": f"L1_s{seed}: EchoViewCLIP stage-1 image encoder, block 11 + ln_post + proj and a linear head adapted on EV9V training frames (fan224 round)",
                        "cache_key": key, **src, "classes": e1_fman["encoder"]["classes"], "preprocessing": e1_fman["encoder"]["preprocessing"]},
            "feature_dim": DIM, "cache_root": str(paths.cache_root), "frame_order": e1_fman.get("frame_order"), "hz": e1_fman.get("hz"),
            "splits": ["train", "validation"], "test_split": "not extracted", "label_mapping": e1_fman.get("label_mapping")}
    out = root / "runs" / NS / "features" / f"feature_manifest_L1_s{seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fman, indent=1))
    return {"seed": seed, "cache_key": key, "feature_manifest": str(out), "feature_manifest_sha256": sha256_file(out),
            "train_recordings": len(train), "validation_recordings": len(val), "export_prob_max_abs_gap": prob_gap,
            "index_sha256": {v: sha256_file(variant_dir(paths.cache_root, key, v) / "index.csv") for v in VARIANTS}}, enc, head


@torch.no_grad()
def l1_parity(enc, la: dict, paths, seed: int, e1_store, proto: dict, device) -> dict:
    """The online L1 encoder against its stored train export on one training clip, both variants."""
    from scripts.train_joint_visual_temporal import FramePaths
    from echo_routing.manifest import load_manifest
    from echo_routing.temporal.joint_visual import load_frame_u8
    vid = sorted(proto["train"]["clips"])[0]; man = load_manifest(paths.manifest_path).set_index("video_id")
    fp = FramePaths({vid: str(man.loc[vid, "frames_dir"])}); paths_v = fp(vid)
    with np.load(paths.repo_root / la["inputs"]["l1_train_features"] / f"L1_s{seed}" / f"{vid}.npz") as z:
        stored = {v: z[f"{v}_feat"].astype(np.float32) for v in VARIANTS}; fidx = z["frame_idx"]
    n = min(len(fidx), 32); enc.eval(); out = {}
    for v, g in VARIANTS.items():
        u8 = torch.from_numpy(np.stack([load_frame_u8(paths_v[min(int(f), len(paths_v) - 1)], g) for f in fidx[:n]])).to(device)
        out[v] = float(np.abs(enc(u8).float().cpu().numpy() - stored[v][:n]).max())
    if max(out.values()) > float(la["b_cache"]["parity"]["stored_vs_online_fp16"]):
        raise SystemExit(f"L1_s{seed}: online features differ from the stored export {out}")
    return {"frames": n, "max_abs": out}


def alignment(cfg: dict, la: dict, paths, fmans: dict) -> dict:
    """Build the inputs of every feature set (protocol, bank digest, validation banks) and compare them with A."""
    from echo_routing.temporal.dev_run import Inputs
    root = paths.repo_root; proto = root / la["inputs"]["protocol"]["path"]
    a = Inputs(cfg, proto, root / la["inputs"]["e1_feature_manifest"]["path"], paths.cache_root); out = {}
    for name, fman in fmans.items():
        b = Inputs(cfg, proto, Path(fman), paths.cache_root)
        if a.labels != b.labels:
            raise SystemExit(f"{name}: clip labels differ")
        rec = {"train_streams": len(b.train.streams)}
        if len(a.train.streams) != len(b.train.streams):
            raise SystemExit(f"{name}: {len(b.train.streams)} training streams, A has {len(a.train.streams)}")
        for i, (x, y) in enumerate(zip(a.train.streams, b.train.streams)):
            if x.seq_id != y.seq_id or not np.array_equal(x.labels, y.labels) or not np.array_equal(x.joins, y.joins) or x.feat.shape != y.feat.shape:
                raise SystemExit(f"{name}: training stream {i} differs between A and B")
        rec["train_streams_feature_differ"] = int(sum(not np.array_equal(x.feat, y.feat) for x, y in zip(a.train.streams, b.train.streams)))
        for h in ("selection", "calibration"):
            ea, eb = a.halves[h], b.halves[h]
            if [s.seq_id for s in ea.natives] != [s.seq_id for s in eb.natives] or any(not np.array_equal(x.labels, y.labels) for x, y in zip(ea.natives, eb.natives)):
                raise SystemExit(f"{name}: {h} native clips differ")
            rec[f"{h}_pairs_feature_differ"] = compare_streams(list(ea.pairs), list(eb.pairs))
            rec[f"{h}_multi_feature_differ"] = compare_streams(list(ea.multi), list(eb.multi))
        for vid in a.labels:
            for v in VARIANTS:
                ra, rb = a.store.get(vid, v), b.store.get(vid, v)
                if not (np.array_equal(ra.frame_idx, rb.frame_idx) and np.array_equal(ra.t_sec, rb.t_sec)):
                    raise SystemExit(f"{name}: frame indices or time stamps differ")
        rec["recordings_checked"] = len(a.labels)
        out[name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); la = cfg["l1_asrf"]; paths = load_paths(); root = paths.repo_root
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    t0 = time.time(); inputs = check_inputs(la, root)
    from echo_routing.config import load_config
    from echo_routing.temporal.dev_data import FeatureStore
    from scripts.train_fan224_visual import build, parity, train_table
    fcfg = load_config(root / la["inputs"]["fan224_config"]["path"]); fv = fcfg["fan224"]
    proto = json.loads((root / la["inputs"]["protocol"]["path"]).read_text())
    e1_idx = read_index(variant_dir(paths.cache_root, la["inputs"]["e1_cache_key"], "orig")); check_clip_sets(e1_idx, proto)
    e1_store = FeatureStore(paths.cache_root, la["inputs"]["e1_cache_key"])
    table = train_table(fv, paths)
    from scripts.train_joint_visual_temporal import FramePaths
    e1_enc, _ = build(fv, paths, "L0", device)
    rep = {"inputs": inputs, "e1_parity": parity(e1_enc, fv, paths, table, FramePaths(table["frames_dir"]), device), "b": {}}
    del e1_enc; torch.cuda.empty_cache()
    fmans = {}
    for seed in [int(s) for s in la["seeds"]]:
        rec, enc, head = build_b_cache(la, fv, paths, seed, e1_idx, e1_store, proto, device, a.workers)
        rec["parity"] = l1_parity(enc, la, paths, seed, e1_store, proto, device)
        rep["b"][f"L1_s{seed}"] = rec; fmans[f"B_s{seed}"] = rec["feature_manifest"]
        print(json.dumps(rec), flush=True); del enc, head; torch.cuda.empty_cache()
    rep["alignment"] = alignment(cfg, la, paths, fmans); rep["seconds"] = time.time() - t0
    out = root / "runs" / NS / "inputs.json"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"alignment": rep["alignment"], "e1_parity": rep["e1_parity"], "seconds": rep["seconds"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
