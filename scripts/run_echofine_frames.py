#!/usr/bin/env python
"""A35 (configs/experiments/a5c_echofine_evidence_v1.yaml): frozen EchoFine classifier on a fixed frame plan.

STANDALONE: runs under envs/echofine (TensorFlow 2.13). Reads a plan written by scripts/diagnose_echofine_evidence.py
(recording frame paths, gamma), decodes each frame with A32's rule (RGB, echo_routing.ingest.frames.apply_gamma when
gamma != 1), resizes the whole frame with tf.image.resize(bilinear, antialias=False) to 224 x 224 in 0-255, and runs the
model with training=False (its own Rescaling(1/255) normalises). Writes the full 47-class softmax per plan row.

  --mode check   official EchoForge loader (single image, predict) against the batched path on fixed plan rows,
                 repeat inference, column order, sums, weight digest, throughput and memory; no classification readout
  --mode run     the one full pass over every plan row (resumable by chunk; the cache key is checked)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.ingest.frames import apply_gamma  # noqa: E402

FORGE = ROOT / "third_party/EchoForge_94131d9b"
WEIGHTS = ROOT / "models_frozen/echofine_classifier/hf_164aa2f3/TTE47_Classifier_Model.h5"
WEIGHTS_SHA = "e74c6581b7da9acf82d9c1545543a13777fee544dc8b3cebafd9f74587c132a2"
LOOKUP = FORGE / "echoforge/classification/models/EchoFine/datasplits/class_lookup.json"
PREPROCESSING = "RGB float32 0-255, tf.image.resize(size=(224, 224), method='bilinear', antialias=False), no crop / pad, model-internal Rescaling(1/255)"
CHUNK, BATCH = 4096, 128


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_frame(path: str, gamma: float) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB"); im = apply_gamma(im, gamma) if gamma != 1.0 else im
        return np.asarray(im, dtype=np.float32)


def preprocess(frames: np.ndarray) -> tf.Tensor:
    return tf.image.resize(tf.convert_to_tensor(frames), size=(224, 224), method="bilinear", antialias=False)


def weight_digest(model) -> str:
    h = hashlib.sha256()
    for w in model.weights:
        h.update(w.name.encode()); h.update(np.ascontiguousarray(w.numpy()).tobytes())
    return h.hexdigest()


def official_model():
    """EchoForge's own loader (build_echofine_classifier(pretrained=True)) with its cache pointed at the pinned file."""
    cache = Path(os.environ["ECHOFORGE_MODEL_DIR"]); link = cache / "thrive-2025__EchoFine_classifier" / "TTE47_Classifier_Model.h5"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(WEIGHTS)
    sys.path.insert(0, str(FORGE))
    from echoforge.classification.models.EchoFine.EchoFine_classifier import build_echofine_classifier  # noqa: E402
    return build_echofine_classifier(pretrained=True)


def decode_rows(plan: dict, rows: np.ndarray, pool: ThreadPoolExecutor) -> np.ndarray:
    return np.stack(list(pool.map(lambda i: load_frame(plan["path"][i], float(plan["gamma"][i])), rows)))


def key_of(plan_sha: str) -> str:
    return hashlib.sha256(json.dumps({"weights": WEIGHTS_SHA, "lookup": sha256(LOOKUP), "preprocessing": PREPROCESSING, "plan": plan_sha, "tf": tf.__version__,
                                      "keras": keras.__version__}, sort_keys=True).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", choices=("check", "run"), required=True)
    ap.add_argument("--threads", type=int, default=8); a = ap.parse_args()
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)      # full float32 (TF32 made outputs depend on batch composition by ~4e-4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); z = np.load(a.plan, allow_pickle=False); plan = {k: z[k] for k in z.files}; plan_sha = sha256(Path(a.plan)); n = len(plan["path"])
    if sha256(WEIGHTS) != WEIGHTS_SHA:
        raise SystemExit("weights differ from the pin")
    lookup = json.loads(LOOKUP.read_text()); names = [lookup[str(i)] for i in range(len(lookup))]
    if (names[7], names[13], names[14], len(names)) != ("a4ch-full", "a5ch-full", "a5ch-outflow", 47):
        raise SystemExit("class_lookup order differs from the protocol")
    model = tf.keras.models.load_model(WEIGHTS); d0 = weight_digest(model); pool = ThreadPoolExecutor(a.threads)
    dev = [d.name for d in tf.config.list_logical_devices("GPU")] or ["CPU"]; key = key_of(plan_sha)
    info = {"tf": tf.__version__, "keras": keras.__version__, "devices": dev, "tf32": tf.config.experimental.tensor_float_32_execution_enabled(), "xla_jit": tf.config.optimizer.get_jit(),
            "weights_sha256": WEIGHTS_SHA, "lookup_sha256": sha256(LOOKUP), "plan_sha256": plan_sha, "rows": int(n), "key": key, "preprocessing": PREPROCESSING,
            "model_input": str(model.input_shape), "model_output": str(model.output_shape), "first_layers": [l.__class__.__name__ for l in model.layers[:3]]}
    if a.mode == "check":
        rows = np.arange(32); x = preprocess(decode_rows(plan, rows, pool)); off = official_model()
        p_off = np.concatenate([off.predict(x[i:i + 1], verbose=0) for i in range(len(rows))]); p_b = model(x, training=False).numpy(); p_b2 = model(x, training=False).numpy()
        p_b16 = np.concatenate([model(x[i:i + 16], training=False).numpy() for i in range(0, len(rows), 16)])
        resc = [l for l in model.layers if l.__class__.__name__ == "Rescaling"] or [l for l in getattr(model.layers[0], "layers", []) if l.__class__.__name__ == "Rescaling"]
        info["check"] = {"official_vs_batched_max_abs": float(np.abs(p_off - p_b).max()), "argmax_equal": bool((p_off.argmax(1) == p_b.argmax(1)).all()), "repeat_max_abs": float(np.abs(p_b - p_b2).max()),
                         "batch16_vs_batch32_max_abs": float(np.abs(p_b - p_b16).max()), "row_sums": [float(p_b.sum(1).min()), float(p_b.sum(1).max())], "output_columns": int(p_b.shape[1]),
                         "input_range": [float(tf.reduce_min(x)), float(tf.reduce_max(x))], "rescaling_layers": [(l.name, float(l.scale), float(l.offset)) for l in resc],
                         "official_weights_equal": weight_digest(off) == d0}
        t0 = time.time(); rows = np.arange(min(n, 2048)); xs = decode_rows(plan, rows, pool); td = time.time() - t0; t1 = time.time()
        for b in range(0, len(rows), BATCH):
            _ = model(preprocess(xs[b:b + BATCH]), training=False).numpy()
        tg = time.time() - t1; info["throughput"] = {"frames": int(len(rows)), "decode_s": td, "model_s": tg, "projected_full_hours": (td + tg) / len(rows) * n / 3600}
        if dev != ["CPU"]:
            info["throughput"]["gpu_peak_mb"] = tf.config.experimental.get_memory_info(dev[0])["peak"] / 2 ** 20
        info["weights_unchanged"] = weight_digest(model) == d0; (out / "check.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1), flush=True)
        c = info["check"]
        if not (c["official_vs_batched_max_abs"] <= 1e-5 and c["argmax_equal"] and c["output_columns"] == 47 and abs(c["row_sums"][0] - 1) < 1e-4 and abs(c["row_sums"][1] - 1) < 1e-4 and info["weights_unchanged"] and c["official_weights_equal"]):
            raise SystemExit("an EchoFine check failed")
        return
    mm_path = out / "probs.npy"; done_path = out / "done_chunks.json"
    if mm_path.exists():
        rec = json.loads((out / "run.json").read_text())
        if rec["key"] != key:
            raise SystemExit("the stored probabilities belong to another key")
        mm = np.load(mm_path, mmap_mode="r+")
    else:
        mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, 47)); (out / "run.json").write_text(json.dumps(info, indent=1))
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set(); t0 = time.time(); k = 0
    for c0 in range(0, n, CHUNK):
        if c0 in done:
            continue
        rows = np.arange(c0, min(n, c0 + CHUNK)); xs = decode_rows(plan, rows, pool)
        for b in range(0, len(rows), BATCH):
            mm[rows[b:b + BATCH]] = model(preprocess(xs[b:b + BATCH]), training=False).numpy()
        mm.flush(); done.add(c0); done_path.write_text(json.dumps(sorted(done))); k += len(rows)
        print(f"{min(n, c0 + CHUNK)}/{n} frames; {time.time() - t0:.0f} s; device {dev[0]}", flush=True)
    del mm
    if weight_digest(model) != d0:
        raise SystemExit("weights changed during inference")
    rec = {**info, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "probs_sha256": sha256(mm_path), "weights_unchanged": True}
    (out / "run.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k_: rec[k_] for k_ in ("rows", "devices", "probs_sha256", "key")}), flush=True)


if __name__ == "__main__":
    main()
