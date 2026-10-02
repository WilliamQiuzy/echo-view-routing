#!/usr/bin/env python
"""Write the feature manifest of an encoder's cache and check it (2026-10-02 plan, Task 2).

Records the checkpoint (path, sha256, cache key), the encoder run, sampling, preprocessing, label mapping and the
per-variant index digests; checks that every protocol clip is cached with the protocol's 10 Hz sample count; and
re-extracts one short validation clip online with the same checkpoint to compare against the cache.
Writes <run_dir>/feature_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402
from scripts.model_development_protocol import sample_index  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import FRAME_ORDER_VERSION, load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash, load_checkpoint  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA, extract_video  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = base_parser("Feature manifest and cache check")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--encoder-run-dir", required=True)
    ap.add_argument("--protocol", required=True); ap.add_argument("--variants", default="orig,gamma090")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    key = checkpoint_hash(args.ckpt); model, meta = load_checkpoint(args.ckpt)
    proto = json.loads(Path(args.protocol).read_text())
    manifest = load_manifest(paths.manifest_path)
    expected = sample_index(manifest, float(cfg["sampling"]["target_hz"])).set_index("video_id")
    clips = list(proto["train"]["clips"]) + [c for h in ("selection", "calibration") for c in proto["validation"][h]["clips"]]
    variants = {}
    for var in args.variants.split(","):
        d = variant_dir(paths.cache_root, key, var); idx = read_index(d).set_index("video_id")
        missing = [c for c in clips if c not in idx.index]
        wrong_n = [c for c in clips if c in idx.index and int(idx.loc[c, "n_samples"]) != int(expected.loc[c, "n_samples"])]
        wrong_label = [c for c in clips if c in idx.index and int(idx.loc[c, "label_index"]) != int(expected.loc[c, "label_index"])]
        test_rows = int((idx["split"] == "test").sum())
        variants[var] = {"dir": str(d), "index_sha256": sha256_file(d / "index.csv"), "meta": json.loads((d / "meta.json").read_text()),
                         "n_cached": int(len(idx)), "n_protocol_clips": len(clips), "missing": missing[:10], "n_missing": len(missing),
                         "n_wrong_samples": len(wrong_n), "n_wrong_label": len(wrong_label), "n_test_rows": test_rows}
        if missing or wrong_n or wrong_label or test_rows:
            raise SystemExit(f"{var}: missing {len(missing)}, wrong n {len(wrong_n)}, wrong label {len(wrong_label)}, test rows {test_rows}")
    # online check on the shortest selection-half clip
    sel = proto["validation"]["selection"]["clips"]
    vid = min(sel, key=lambda v: int(expected.loc[v, "n_samples"]))
    row = manifest.set_index("video_id").loc[vid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = {}
    for var in args.variants.split(","):
        cached = load_video_features(variant_dir(paths.cache_root, key, var) / f"{vid}.npz")
        online = extract_video(model, device, Path(row["frames_dir"]), vid, int(row["n_frames"]), float(row["fps_playback"]),
                               float(cfg["sampling"]["target_hz"]), int(meta.get("image_size", cfg["preprocess"]["image_size"])),
                               VARIANT_GAMMA[var], batch_size=int(cfg["features"]["batch_size"]))
        checks[var] = {"video_id": vid, "n": int(cached.n), "frame_idx_equal": bool(np.array_equal(cached.frame_idx, online.frame_idx)),
                       "max_abs_prob_diff": float(np.abs(cached.prob - online.prob).max()),
                       "max_abs_feat_diff": float(np.abs(cached.feat - online.feat.astype(np.float16).astype(np.float32)).max()),
                       "argmax_agreement": float((cached.prob.argmax(1) == online.prob.argmax(1)).mean())}
    run_dir = new_run_dir(cfg, "feature_manifest")
    out = {"encoder": {"checkpoint": str(args.ckpt), "sha256": sha256_file(args.ckpt), "cache_key": key,
                       "run_dir": str(args.encoder_run_dir), "backbone": meta.get("backbone"), "task": meta.get("task"),
                       "classes": meta.get("classes"), "image_size": meta.get("image_size"), "epoch": meta.get("epoch")},
           "cache_root": str(paths.cache_root), "frame_order": FRAME_ORDER_VERSION, "hz": cfg["sampling"]["target_hz"],
           "splits": sorted({str(s) for v in variants.values() for s in ["train", "validation"]}), "test_split": "not extracted",
           "label_mapping": "family5_v1 (configs/labels.yaml)", "variants": variants, "online_check": checks,
           "protocol": str(args.protocol), "protocol_sha256": sha256_file(args.protocol)}
    (run_dir / "feature_manifest.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"run_dir": str(run_dir), "cache_key": key, "online_check": checks,
                      "variants": {k: {kk: v[kk] for kk in ("n_cached", "n_missing", "n_wrong_samples", "n_test_rows")} for k, v in variants.items()}}, indent=1))


if __name__ == "__main__":
    main()
