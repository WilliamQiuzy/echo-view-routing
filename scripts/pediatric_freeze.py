#!/usr/bin/env python
"""Freeze the EchoNet-Pediatric external protocol before any prediction (task 2 of the 2026-10-01 round).

Decodes every listed AVI once to count its frames (no model is run), then writes, under
data/manifests/pediatric/<PROTOCOL_VERSION>/ on the server only:
- videos.csv: one row per video with view, label, first-token group, exam, clip number, fold, role, fps, frame count
  and the number of 10 Hz samples;
- recipes_<role>_<bank>.json for the `exam` and `same` stitch banks of each role;
- protocol.json: versions, frozen models and parameters, conditions, metrics, decision rules, and SHA-256 digests of
  the video table and of every recipe bank.
An existing frozen directory is never rewritten: the script recomputes the digests and stops if they differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.config import load_config  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3  # noqa: E402
from echo_routing.realvideo.pediatric import (CALIBRATION_FOLDS, PREPROCESS_VERSION, PROTOCOL_VERSION, VIDEO_ROOT,  # noqa: E402
                                              VIEW_TO_FAMILY5, digest_rows, exam_recipes, read_filelists,
                                              recipes_digest, same_view_recipes, timestamp_indices)

B0_CKPT = "checkpoints/20260913-033411-b0_frame_encoder-b072725b-s0/best.pt"
B0_HASH = "79f4a41af6e3"
MSTCN_MODEL = "models_frozen/mstcn_official/v2_frameorder/epoch-50.model"
EV9V_CONFIG = "runs/20260928-001145-baselines-1b027c50-s0/config.resolved.yaml"
HZ = 10.0
PERMUTATION_SEEDS = list(range(100, 110))   # the task-1 seeds, so EV9V and Pediatric use the same permutation function
BOOTSTRAP = {"unit": "first-token group (package)", "replicates": 2000, "seed": 20261001, "interval": "percentile 95%"}


def count_frames(path: str) -> tuple[float, int, int]:
    """(container fps, container frame count, frames actually decoded)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return float("nan"), 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)); n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n = 0
    while cap.grab():
        n += 1
    cap.release()
    return fps, n_meta, n


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_body(cfg: dict, videos: pd.DataFrame, banks: dict) -> dict:
    dec = cfg["decode"]
    return {
        "protocol_version": PROTOCOL_VERSION, "preprocess_version": PREPROCESS_VERSION,
        "frozen_before_prediction": True,
        "data": {"root": str(VIDEO_ROOT), "labels": {v: ["PLAX", "PSAX", "A4C", "A5C", "SC4C"][i] for v, i in VIEW_TO_FAMILY5.items()},
                 "output_space": "five family5 outputs kept; PLAX, A5C and SC4C predictions count as errors",
                 "groups": "first file-name token; the official folds never split a group; not verified as patient identity",
                 "exam": "first two file-name tokens", "clip_order": "third file-name token (acquisition number)"},
        "split": {"official": "10 folds (Split 0-9), no designated test fold", "calibration_folds": list(CALIBRATION_FOLDS),
                  "test_folds": sorted(set(range(10)) - set(CALIBRATION_FOLDS)),
                  "rule": "fixed before prediction; the main analysis uses the test folds only; calibration folds serve the optional secondary re-selection of decoding parameters"},
        "sampling": {"hz": HZ, "rule": "nearest frame to t = k/10 s at the AVI's own frame rate (CAP_PROP_FPS); frame count = frames actually decoded",
                     "preprocess": "OpenCV decode, BGR->RGB, letterbox to 224 (aspect-preserving, black pad), ImageNet normalisation, no gamma edit; bfloat16 autocast as in the EV9V extraction"},
        "models": {"b0": {"checkpoint": B0_CKPT, "hash": B0_HASH, "trained_on": "EV9V train split only (ImageNet-initialised ResNet-18)"},
                   "decoders": {"b1": dec["b1"], "b2": dec["b2"], "b3": dec["b3"], "b3a": dec.get("b3a", AUDITED_B3),
                                "source": f"{EV9V_CONFIG} (frozen on EV9V; b3a = audited setting chosen on EV9V true-order validation)"},
                   "b4": {"model": MSTCN_MODEL, "sha256": sha256_file(Path(MSTCN_MODEL)),
                          "note": "official MS-TCN retrained on correctly ordered EV9V B0 features; authors' constants"}},
        "main_methods": ["b2 (HMM)", "b3a (audited B3)", "b4 (MS-TCN v2)"], "secondary_methods": ["b0", "b1", "b3"],
        "conditions": {"true": "acquisition order", "random": {"seeds": PERMUTATION_SEEDS,
                       "rule": "echo_routing.realvideo.permutations.order_index(kind='random') per fragment; recipe id = video id for native clips"}},
        "banks": {k: {"n": len(v), "digest": recipes_digest(v)} for k, v in banks.items()},
        "bank_rules": {"native": "every video alone, whole clip",
                       "exam": "every exam with >= 2 clips: all its clips in clip-number order, whole clips, each video once",
                       "same": "inside one group and view: consecutive non-overlapping pairs in (exam, clip-number) order, each video at most once"},
        "metrics": {
            "classification": "true order, test folds: per-sample accuracy and confusion over the five outputs; clip majority vote; per-view recall; macro-F1 over the two present classes",
            "native": "per clip: internal cuts (label changes after decoding), samples whose decoded label differs from the clip label, error-run lengths, lag-1 autocorrelation of B0 probabilities",
            "stitched": "ladder implementation (echo_routing.evaluate.ladder.constructed_metrics): different-class boundary precision/recall/F1 at 0.5 s and 1.0 s; same-class join cut rate within 0.5 s",
            "effect": "random (mean over the 10 seeds) minus true, per method; per-seed effects reported as a range",
            "intervals": BOOTSTRAP},
        "decision_rules": {
            "transfer": "classification transfer counts as adequate if B0 clip majority-vote recall is >= 0.80 for both A4C and PSAX on the test folds; otherwise external validation is reported as limited and permutation results are descriptive only",
            "replication": "primary: random-minus-true native internal cuts per clip for b2 and b3a; replication in the EV9V direction requires both differences < 0 with group-bootstrap 95% intervals excluding 0",
            "no_rescue": "no filtering of clips by prediction quality, no re-split after seeing test results, no forced two-class output"},
        "secondary_optional": {"what": "re-select only decoding parameters on the calibration folds' exam streams",
                               "grid": {"b2.stay": [0.9, 0.95, 0.98, 0.99, 0.995], "b3.half_window": [5, 7, 9, 11],
                                        "b3.threshold_quantile": [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]},
                               "target": "different-class boundary recall >= 0.95 within 0.5 s on calibration; if unreachable, report infeasible and the test recall at the closest setting",
                               "status": "optional; reported in a separate column from direct transfer"},
        "videos_digest": digest_rows(videos),
        "counts": {
            "videos": videos.groupby(["role", "view"]).size().unstack().to_dict(),
            "groups": videos.groupby("role")["group"].nunique().to_dict(), "exams": videos.groupby("role")["exam"].nunique().to_dict(),
            "samples_p10_p50_p90": {r: [float(x) for x in g["n_samples"].quantile([0.1, 0.5, 0.9])] for r, g in videos.groupby("role")},
            "frame_count_mismatch": int((videos["n_frames"] != videos["n_frames_meta"]).sum())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the EchoNet-Pediatric external protocol")
    ap.add_argument("--out", default=f"data/manifests/pediatric/{PROTOCOL_VERSION}"); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); out = Path(args.out)
    if checkpoint_hash(Path(B0_CKPT)) != B0_HASH:
        raise SystemExit("B0 checkpoint hash mismatch")
    videos = read_filelists(VIDEO_ROOT)
    with ProcessPoolExecutor(args.workers) as ex:
        counts = list(ex.map(count_frames, videos["path"], chunksize=32))
    videos["fps"] = [c[0] for c in counts]; videos["n_frames_meta"] = [c[1] for c in counts]; videos["n_frames"] = [c[2] for c in counts]
    if (videos["n_frames"] <= 0).any() or videos["fps"].isna().any():
        raise SystemExit(f"{int((videos['n_frames'] <= 0).sum())} unreadable videos")
    videos["n_samples"] = [len(timestamp_indices(int(n), float(f), HZ)) for n, f in zip(videos["n_frames"], videos["fps"])]
    videos = videos.drop(columns=["path"])
    banks = {}
    for role in ("calibration", "test"):
        part = videos[videos["role"] == role]
        banks[f"{role}_exam"] = exam_recipes(part); banks[f"{role}_same"] = same_view_recipes(part)
    cfg = load_config(Path(EV9V_CONFIG))
    body = protocol_body(cfg, videos, banks)
    if (out / "protocol.json").exists():
        old = json.loads((out / "protocol.json").read_text())
        same = old["videos_digest"] == body["videos_digest"] and old["banks"] == body["banks"]
        raise SystemExit(f"{out} is already frozen; recomputed digests {'match' if same else 'DIFFER'}")
    out.mkdir(parents=True, exist_ok=False)
    videos.to_csv(out / "videos.csv", index=False)
    for k, v in banks.items():
        (out / f"recipes_{k}.json").write_text(json.dumps([recipe_to_dict(r) for r in v]))
    body["files_sha256"] = {p.name: sha256_file(p) for p in sorted(out.iterdir())}
    (out / "protocol.json").write_text(json.dumps(body, indent=1, default=str))
    for p in out.iterdir():
        p.chmod(0o444)
    summary = {k: body[k] for k in ("protocol_version", "videos_digest", "banks", "counts")}
    summary["protocol_sha256"] = sha256_file(out / "protocol.json")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
