#!/usr/bin/env python
"""E1 classification probe and formal feature cache (round 2, Task 3).

1. Train a 5-class linear probe on the frozen EchoViewCLIP per-frame embeddings written by
   scripts/extract_echoviewclip_features.py (unedited frames). Recipe from configs/experiments/model_development_v2.yaml
   (probe): AdamW, lr 1e-3, weight decay 1e-4, batch 256, at most 20 epochs, patience 5, seed 0. Each epoch draws as
   many training clips as there are, class-balanced with replacement (as the v3 encoder), and 16 cached 10 Hz samples
   per clip, spaced with a random phase, with replacement when a clip has fewer than 16. The probe is selected on the
   selection half's native macro-F1 (every 10 Hz sample, clip majority vote); the calibration half never chooses it.
2. Write the formal cache for every variant: frame_idx and t_sec copied from the round-1 v3 cache and checked sample by
   sample, feat = the embedding (float16), logit = the frozen probe's output on the float32 embedding, prob =
   softmax(logit). The cache key binds the visual checkpoint, layer, preprocessing and probe digests.
3. Write feature_manifest.json (with feature_dim) and read one clip back through load_video_features.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from echo_routing.config import FRAME_ORDER_VERSION, load_paths  # noqa: E402
from echo_routing.features.cache import (VideoFeatures, load_video_features, read_index, save_video_features, variant_dir,  # noqa: E402
                                         write_index, write_meta)
from echo_routing.features.sampling import balanced_video_weights, spaced_indices  # noqa: E402
from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, check_index, digest  # noqa: E402

VARIANTS = ("orig", "gamma090")
NUM_CLASSES = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_draw(n_samples: list[int], labels: list[int], k: int, n_clips: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """(clip, sample) pairs of one epoch: clips class-balanced with replacement, k spaced samples per drawn clip, drawn
    with replacement when the clip holds fewer than k samples."""
    chosen = rng.choice(len(labels), size=n_clips, replace=True, p=balanced_video_weights(labels))
    out = []
    for c in chosen:
        n = int(n_samples[c])
        pos = spaced_indices(n, k, rng) if n >= k else rng.choice(n, size=k, replace=True)
        out += [(int(c), int(p)) for p in pos]
    return out


def clip_majority_f1(probe: nn.Module, embs: list[np.ndarray], labels: list[int]) -> float:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    with torch.no_grad():
        for e, y in zip(embs, labels):
            pred = probe(torch.from_numpy(e)).argmax(1).numpy()
            conf[y, int(np.bincount(pred, minlength=NUM_CLASSES).argmax())] += 1
    tp = np.diag(conf); den = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())


def train_probe(train_embs: list[np.ndarray], train_labels: list[int], sel_embs: list[np.ndarray], sel_labels: list[int],
                pc: dict, log) -> tuple[nn.Module, list[dict], int]:
    torch.manual_seed(int(pc["seed"])); np.random.seed(int(pc["seed"]))
    dim = int(train_embs[0].shape[1]); probe = nn.Linear(dim, NUM_CLASSES)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]))
    n_clips = int(pc.get("clips_per_epoch") or len(train_embs)); records, best, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(int(pc["max_epochs"])):
        rng = np.random.default_rng(int(pc["seed"]) + epoch)
        pairs = epoch_draw([e.shape[0] for e in train_embs], train_labels, int(pc["samples_per_clip"]), n_clips, rng)
        x = torch.from_numpy(np.stack([train_embs[c][p] for c, p in pairs])); y = torch.tensor([train_labels[c] for c, _ in pairs])
        order = torch.from_numpy(rng.permutation(len(pairs))); probe.train(); total = 0.0
        for i in range(0, len(order), int(pc["batch_size"])):
            b = order[i:i + int(pc["batch_size"])]
            loss = nn.functional.cross_entropy(probe(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss) * len(b)
        probe.eval(); f1 = clip_majority_f1(probe, sel_embs, sel_labels)
        records.append({"epoch": epoch, "train_loss": total / len(order), "selection_native_macro_f1": f1, "n_samples": len(order)})
        log(f"probe epoch {epoch}: loss {total / len(order):.4f}, selection native macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_epoch = f1, epoch; best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        elif epoch - best_epoch >= int(pc["patience"]):
            log(f"early stop: best epoch {best_epoch}"); break
    probe.load_state_dict(best_state); probe.eval()
    return probe, records, best_epoch


def cache_arrays(video_id: str, frame_idx: np.ndarray, t_sec: np.ndarray, emb: np.ndarray, probe: nn.Module) -> VideoFeatures:
    """Formal cache record: logit is the frozen probe on the float32 embedding and prob = softmax(logit)."""
    with torch.no_grad():
        logit = probe(torch.from_numpy(np.asarray(emb, np.float32))).numpy()
    prob = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
    return VideoFeatures(video_id, np.asarray(frame_idx, np.int32), np.asarray(t_sec, np.float32),
                         np.asarray(emb, np.float32), prob.astype(np.float32), logit.astype(np.float32))


def check_alignment(video_id: str, frame_idx: np.ndarray, v3: VideoFeatures) -> None:
    if not np.array_equal(np.asarray(frame_idx), v3.frame_idx):
        raise SystemExit(f"{video_id}: E1 frame indices do not match the round-1 v3 cache")


def main() -> None:
    ap = base_parser("E1 probe and feature cache")
    ap.add_argument("--raw-dir", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--v3-feature-manifest", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); pc = cfg["probe"]
    raw = Path(args.raw_dir); emeta = json.loads((raw / "extract_meta.json").read_text())
    proto_raw = Path(args.protocol).read_bytes(); proto = json.loads(proto_raw)
    if emeta["protocol_sha256"] != hashlib.sha256(proto_raw).hexdigest():
        raise SystemExit("raw embeddings were extracted for another protocol")
    v3man = json.loads(Path(args.v3_feature_manifest).read_text()); v3key = v3man["encoder"]["cache_key"]
    v3idx = read_index(variant_dir(paths.cache_root, v3key, "orig")); check_index(v3idx, proto)
    v3_order = list(v3idx["video_id"])            # the training bank is drawn in index order, so keep the v3 order
    v3idx = v3idx.set_index("video_id")
    run_dir = new_run_dir(cfg, "e1_probe"); log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True); log_lines.append(msg)

    def emb(v: str, var: str = "orig") -> np.ndarray:
        return np.load(raw / "raw" / var / f"{v}.npz")["emb"]

    train = list(proto["train"]["clips"]); sel = list(proto["validation"]["selection"]["clips"])
    probe, records, best_epoch = train_probe([emb(v) for v in train], [int(v3idx.loc[v, "label_index"]) for v in train],
                                             [emb(v) for v in sel], [int(v3idx.loc[v, "label_index"]) for v in sel], pc, log)
    probe_path = run_dir / "probe.pt"
    torch.save({"state_dict": probe.state_dict(), "in_dim": int(probe.in_features), "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"],
                "epoch": best_epoch, "recipe": dict(pc), "visual_checkpoint_sha256": emeta["checkpoint_sha256"], "layer": emeta["layer"]}, probe_path)
    probe_sha = sha256_file(probe_path)
    (run_dir / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    key = hashlib.sha256(json.dumps({"visual": emeta["checkpoint_sha256"], "layer": emeta["layer"], "preprocessing": emeta["preprocessing"],
                                     "probe": probe_sha}, sort_keys=True).encode()).hexdigest()[:12]
    wanted = set(train) | {c for h in HALVES for c in proto["validation"][h]["clips"]}
    clips = [v for v in v3_order if v in wanted]
    variants = {}
    for var in VARIANTS:
        d = variant_dir(paths.cache_root, key, var); rows = []
        for v in clips:
            z = np.load(raw / "raw" / var / f"{v}.npz")
            v3vf = load_video_features(variant_dir(paths.cache_root, v3key, var) / f"{v}.npz")
            check_alignment(v, z["frame_idx"], v3vf)                 # t_sec is then copied from the v3 record
            vf = cache_arrays(v, z["frame_idx"], v3vf.t_sec, z["emb"], probe)
            path = save_video_features(d, vf)
            rows.append({"video_id": v, "split": v3idx.loc[v, "split"], "raw_label": v3idx.loc[v, "raw_label"],
                         "label_index": int(v3idx.loc[v, "label_index"]), "n_samples": vf.n, "path": str(path)})
        write_index(d, rows)
        write_meta(d, {"ckpt_hash": key, "variant": var, "visual_checkpoint_sha256": emeta["checkpoint_sha256"], "layer": emeta["layer"],
                       "preprocessing": emeta["preprocessing"], "probe": str(probe_path), "probe_sha256": probe_sha, "target_hz": cfg["sampling"]["target_hz"]})
        idx = read_index(d); check_index(idx, proto)
        if list(idx["video_id"]) != clips:
            raise SystemExit(f"{var}: index order differs from the v3 index")
        bank = multi_bank(idx, "train", cfg)
        if digest([recipe_to_dict(r) for r in bank.recipes]) != proto["train"]["multi_bank"]["digest"]:
            raise SystemExit(f"{var}: the training bank rebuilt from the E1 index differs from the protocol")
        variants[var] = {"dir": str(d), "index_sha256": sha256_file(d / "index.csv"), "n_cached": int(len(idx)),
                         "n_test_rows": int((idx["split"] == "test").sum())}
        log(f"cache {var}: {len(idx)} clips -> {d}")
    check_vid = min(sel, key=lambda v: int(v3idx.loc[v, "n_samples"]))
    back = load_video_features(variant_dir(paths.cache_root, key, "orig") / f"{check_vid}.npz")
    expect = cache_arrays(check_vid, back.frame_idx, back.t_sec, emb(check_vid), probe)
    interface = {"video_id": check_vid, "n": back.n, "feat_shape": list(back.feat.shape), "prob_rows_sum_to_one": bool(np.allclose(back.prob.sum(1), 1, atol=1e-5)),
                 "prob_equals_softmax_logit": bool(np.allclose(back.prob, torch.softmax(torch.from_numpy(back.logit), 1).numpy(), atol=1e-6)),
                 "logit_matches_probe_on_raw": float(np.abs(back.logit - expect.logit).max()),
                 "feat_matches_raw_fp16": float(np.abs(back.feat - emb(check_vid)).max())}
    manifest = {"encoder": {"name": "E1: EchoViewCLIP stage-1 image encoder + linear probe", "checkpoint": str(probe_path), "sha256": probe_sha,
                            "cache_key": key, "run_dir": str(run_dir), "visual_checkpoint": emeta["checkpoint"],
                            "visual_checkpoint_sha256": emeta["checkpoint_sha256"], "visual_commit": emeta["commit"], "layer": emeta["layer"],
                            "preprocessing": emeta["preprocessing"], "probe_recipe": dict(pc), "probe_selected_epoch": best_epoch,
                            "probe_selection_native_macro_f1": records[best_epoch]["selection_native_macro_f1"],
                            "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]},
                "feature_dim": int(probe.in_features), "cache_root": str(paths.cache_root), "frame_order": FRAME_ORDER_VERSION,
                "hz": cfg["sampling"]["target_hz"], "splits": ["train", "validation"], "test_split": "not extracted",
                "label_mapping": "family5_v1 (configs/labels.yaml)", "variants": variants, "raw_extraction": str(raw),
                "extract_meta": emeta, "interface_check": interface, "frame_index_source": v3man["encoder"]["cache_key"],
                "protocol": str(args.protocol), "protocol_sha256": hashlib.sha256(proto_raw).hexdigest()}
    (run_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (run_dir / "probe_log.txt").write_text("\n".join(log_lines) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "cache_key": key, "probe_selected_epoch": best_epoch,
                      "probe_selection_native_macro_f1": manifest["encoder"]["probe_selection_native_macro_f1"], "interface": interface}, indent=1))


if __name__ == "__main__":
    main()
