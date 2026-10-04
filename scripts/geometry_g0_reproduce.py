#!/usr/bin/env python
"""G0 through the new extractor must reproduce E1 without retuning (geometry pretraining, start condition 4).

1. Raw embeddings: every clip and variant against the float32 raw E1 cache (frame indices equal; max abs difference).
2. Probe: the existing E1 probe (not retrained) on the new embeddings; logits against E1's formal cache, and whether
   any per-sample prediction changes.
3. Formal cache: written with that same probe (train_feature_probe.cache_arrays) under its own key, with a feature
   manifest. Its float16 features are compared with E1's.
4. ASRF: the round-2 E1 ASRF checkpoints of seeds 0-2 decode every validation sequence of both halves from the new
   cache; the decodes must equal the stored predictions.
B0 and B3 are reproduced by running scripts/calibrate_simple_decoders.py on the new manifest and comparing its outputs
with E1's round-2 run (--compare-simple). Writes <out>/reproduction.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, save_video_features, variant_dir, write_index, write_meta  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs, check_index  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode, inputs_of  # noqa: E402
from scripts.train_feature_probe import VARIANTS, cache_arrays, sha256_file  # noqa: E402

E1_MANIFEST = "runs/20261002-142832-e1_probe-38bbcf7c-s0/feature_manifest.json"
DEV_CONFIG = "configs/experiments/model_development_v2.yaml"


def load_probe(path: Path) -> torch.nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    probe = torch.nn.Linear(int(blob["in_dim"]), 5); probe.load_state_dict(blob["state_dict"]); return probe.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-raw", required=True); ap.add_argument("--old-raw", default="cache_v2/e1_raw/a74c7b233e79")
    ap.add_argument("--protocol", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--compare-simple", nargs=2, metavar=("NEW_RUN", "OLD_RUN"), default=None,
                    help="only compare two calibrate_simple_decoders run directories")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.compare_simple:
        new, old = (Path(p) for p in a.compare_simple); res = {}
        for f in sorted(p.name for p in old.glob("*.csv")):
            x, y = pd.read_csv(old / f), pd.read_csv(new / f)
            num = [c for c in x.columns if c in y.columns and x[c].dtype.kind in "fi"]
            res[f] = {"rows_equal": len(x) == len(y), "max_abs_numeric_diff": float(max([np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))) for c in num] or [0.0])),
                      "text_equal": all(x[c].astype(str).equals(y[c].astype(str)) for c in x.columns if c not in num and c in y.columns)}
        sel_old, sel_new = (json.loads((d / "selected.json").read_text())["selected"] for d in (old, new))
        res["selected_equal"] = sel_old == sel_new
        (out / "reproduction_simple.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); return
    paths = load_paths(); man = json.loads(Path(E1_MANIFEST).read_text()); e1_key = man["encoder"]["cache_key"]
    probe = load_probe(Path(man["encoder"]["checkpoint"])); proto_raw = Path(a.protocol).read_bytes(); proto = json.loads(proto_raw)
    new_raw, old_raw = Path(a.new_raw), Path(a.old_raw); emeta = json.loads((new_raw / "extract_meta.json").read_text())
    rep = {"raw": {}, "probe": {}, "cache": {}}
    e1_idx = read_index(variant_dir(paths.cache_root, e1_key, "orig")); clips = list(e1_idx["video_id"])
    key = "g0r" + hashlib.sha256(json.dumps({"raw": emeta["checkpoint_sha256"], "probe": man["encoder"]["sha256"]}, sort_keys=True).encode()).hexdigest()[:9]
    for var in VARIANTS:
        worst, idx_ok, changed, logit_d, fp16_diff, rows = 0.0, True, 0, 0.0, 0, []
        d = variant_dir(paths.cache_root, key, var); e1_dir = variant_dir(paths.cache_root, e1_key, var)
        for v in clips:
            zn, zo = np.load(new_raw / "raw" / var / f"{v}.npz"), np.load(old_raw / "raw" / var / f"{v}.npz")
            idx_ok &= bool(np.array_equal(zn["frame_idx"], zo["frame_idx"])); worst = max(worst, float(np.abs(zn["emb"] - zo["emb"]).max()))
            e1 = load_video_features(e1_dir / f"{v}.npz"); vf = cache_arrays(v, zn["frame_idx"], e1.t_sec, zn["emb"], probe)
            changed += int((vf.logit.argmax(1) != e1.logit.argmax(1)).sum()); logit_d = max(logit_d, float(np.abs(vf.logit - e1.logit).max()))
            path = save_video_features(d, vf); fp16_diff += int((load_video_features(path).feat != e1.feat).sum())
            r = e1_idx.set_index("video_id").loc[v]
            rows.append({"video_id": v, "split": r["split"], "raw_label": r["raw_label"], "label_index": int(r["label_index"]), "n_samples": vf.n, "path": str(path)})
        write_index(d, rows)
        write_meta(d, {"ckpt_hash": key, "variant": var, "visual_checkpoint_sha256": emeta["checkpoint_sha256"], "layer": emeta["layer"],
                       "preprocessing": emeta["preprocessing"], "probe": man["encoder"]["checkpoint"], "probe_sha256": man["encoder"]["sha256"],
                       "note": "G0 reproduction: new extractor, E1 probe reused without retraining"})
        check_index(read_index(d), proto)
        rep["raw"][var] = {"clips": len(clips), "frame_idx_equal": idx_ok, "max_abs_diff": worst}
        rep["probe"][var] = {"samples_with_changed_prediction": changed, "max_abs_logit_diff": logit_d}
        rep["cache"][var] = {"key": key, "float16_values_differing_from_e1": fp16_diff}
        print(var, rep["raw"][var], rep["probe"][var], rep["cache"][var], flush=True)
    new_man = {**man, "encoder": {**man["encoder"], "cache_key": key, "name": "G0: E1 through scripts/extract_encoder_features.py, E1 probe reused"},
               "raw_extraction": str(new_raw), "extract_meta": emeta}
    man_path = out / "feature_manifest.json"; man_path.write_text(json.dumps(new_man, indent=1, default=str)); rep["feature_manifest"] = str(man_path)
    cfg = load_config(ROOT / DEV_CONFIG); inputs = Inputs(cfg, Path(a.protocol), man_path, paths.cache_root); device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; rep["asrf"] = {}
    for k in (0, 1, 2):
        rd = Path(json.loads(Path(f"runs/md_round2/e1_s{k}_status.json").read_text())["M1"]["run_dir"])
        head, model = load_asrf(rd, ROOT, inputs.feature_dim, device); res = {}
        for h in HALVES:
            stored = np.load(rd / f"predictions_{h}.npz"); seqs = inputs_of(inputs.halves[h], inputs.store); bad = frames = 0
            for sid, (feat, _, _) in seqs.items():
                pred, _ = asrf_decode(head, model, feat, device); ref = stored[sid].astype(int)
                bad += int((pred != ref).sum()); frames += int(ref.size)
            res[h] = {"sequences": len(seqs), "frames": frames, "frames_differing": bad}
        rep["asrf"][f"seed{k}"] = {"run_dir": str(rd), **res}; print("ASRF seed", k, res, flush=True)
    (out / "reproduction.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
