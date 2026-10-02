"""One training run of a round-1 temporal head, end to end (shared by the M0-M4 entry scripts).

Loads the frozen protocol and the feature manifest, rebuilds the training bank (its digest must match the protocol),
trains with dev_train.train_head, then evaluates the selected checkpoint on both validation halves and writes
run_manifest.json, validation_metrics.json, train_log.jsonl, training_curves.png and per-half predictions with
content digests into the run directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from echo_routing.compose.banks import multi_bank
from echo_routing.compose.recipe_builder import recipe_to_dict
from echo_routing.compose.stream_assembler import render
from echo_routing.config import gpu_memory_fraction, load_paths
from echo_routing.evaluate.dev_metrics import evaluate_predictions
from echo_routing.features.cache import read_index, variant_dir
from echo_routing.temporal.dev_data import EvalSet, FeatureStore, TrainData, eval_set, stream_seq
from echo_routing.temporal.dev_train import Settings, decoded_digest, evaluate_head, time_decode, train_head

HALVES = ("selection", "calibration")
DECODED_LABEL_KEYS = ("labels", "local_labels", "frame_labels", "prob", "prepool_prob")
SOURCE_FILES = ("echo_routing/temporal/models/boundary_context.py", "echo_routing/temporal/losses/boundary_context.py",
                "echo_routing/temporal/dev_data.py", "echo_routing/temporal/dev_heads.py", "echo_routing/temporal/dev_train.py",
                "echo_routing/temporal/dev_run.py", "echo_routing/evaluate/dev_metrics.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_json(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_index(idx, proto: dict) -> None:
    """The feature cache must hold every protocol clip (train and both validation halves) and no test row."""
    need = set(proto["train"]["clips"]) | {c for h in HALVES for c in proto["validation"][h]["clips"]}
    missing = sorted(need - set(idx["video_id"]))
    if missing:
        raise SystemExit(f"feature cache lacks {len(missing)} protocol clips, e.g. {missing[:3]}")
    bad = sorted(set(idx["split"]) - {"train", "validation"})
    if bad:
        raise SystemExit(f"feature cache holds rows of split(s) {bad}; development runs must not read test")


def feature_dim_of(fman: dict, actual: int) -> int:
    """Input dimension of the temporal heads: declared by the feature manifest (round 2) and checked against the cache."""
    declared = fman.get("feature_dim")
    if declared is not None and int(declared) != int(actual):
        raise SystemExit(f"feature manifest declares dim {declared}, cache holds {actual}")
    return int(actual)


class Inputs:
    """Protocol, feature cache, training data and validation halves of one encoder."""

    def __init__(self, cfg: dict, protocol_path: Path, feature_manifest_path: Path, cache_root: Path) -> None:
        self.proto, self.proto_sha = load_json(protocol_path)
        self.fman, self.fman_sha = load_json(feature_manifest_path)
        self.protocol_path, self.feature_manifest_path = str(protocol_path), str(feature_manifest_path)
        key = self.fman["encoder"]["cache_key"]
        self.store = FeatureStore(cache_root, key)
        idx = read_index(variant_dir(cache_root, key, "orig"))
        check_index(idx, self.proto)
        self.labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
        bank = multi_bank(idx, "train", cfg)
        bank_digest = digest([recipe_to_dict(r) for r in bank.recipes])
        if bank_digest != self.proto["train"]["multi_bank"]["digest"]:
            raise SystemExit(f"training bank digest {bank_digest[:16]} != protocol {self.proto['train']['multi_bank']['digest'][:16]}")
        self.train = TrainData(tuple(stream_seq(render(r, self.store.get)) for r in bank.recipes),
                               tuple((v, self.labels[v]) for v in self.proto["train"]["clips"]), self.store)
        self.halves = {h: eval_set(self.proto, h, self.labels, self.store) for h in HALVES}
        self.cache_key = key
        self.feature_dim = feature_dim_of(self.fman, self.train.streams[0].feat.shape[1])

    def frame_majority(self, ev: EvalSet, hz: float, tol_s: float) -> dict:
        """The encoder's own per-sample argmax (B0) on one half: the reference for the native-F1 floor."""
        preds = {s.seq_id: self.store.get(s.seq_id, "orig").prob.argmax(1) for s in ev.natives}
        for s in (*ev.pairs, *ev.multi):
            preds[s.recipe.recipe_id] = s.prob.argmax(1)
        return evaluate_predictions(preds, ev, hz, tol_s)[0]


def plot_curves(records: list[dict], chosen: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k in records[0]["loss"]:
        axes[0].plot(ep, [r["loss"][k] for r in records], label=k)
    axes[0].set_title("training losses (per-update means)"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    for k, lab in (("native_macro_f1", "native clip macro-F1"), ("segment_f1_50_multi", "multi segment F1@50"),
                   ("boundary_recall_multi", "multi boundary recall")):
        axes[1].plot(ep, [r[k] for r in records], label=lab)
    axes[1].set_title("selection half"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    axes[2].plot(ep, [r["native_internal_cuts_per_clip"] for r in records], label="native cuts per clip")
    axes[2].plot(ep, [r["same_class_cut_rate_multi"] for r in records], label="same-class cut rate")
    for k in ("boundary_pos_rate_multi", "gate_mean_multi"):
        if k in records[0]:
            axes[2].plot(ep, [r[k] for r in records], label=k.replace("_", " "))
    axes[2].set_title("cuts and internal signals (selection half)"); axes[2].set_xlabel("epoch"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(chosen, color="grey", ls="--", lw=0.8)
    fig.suptitle(f"{title} (dashed: selected epoch {chosen})"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run(row: str, head, inputs: Inputs, cfg: dict, settings: Settings, run_dir: Path, ckpt_dir: Path, device,
        extra_manifest: dict | None = None, augment: Callable | None = None, log: Callable[[str], None] = print) -> dict:
    sel_cfg = cfg["selection"]; hz = float(cfg["protocol"]["hz"]); tol_s = float(sel_cfg["boundary_tolerance_s"])
    baseline = {h: inputs.frame_majority(inputs.halves[h], hz, tol_s) for h in HALVES}
    floor = baseline["selection"]["native_macro_f1"] - float(sel_cfg["native_f1_tolerance"])
    log(f"[{row}] encoder frame-majority native macro-F1: selection {baseline['selection']['native_macro_f1']:.4f}, "
        f"calibration {baseline['calibration']['native_macro_f1']:.4f}; checkpoint floor {floor:.4f}")
    res = train_head(head, inputs.train, inputs.halves["selection"], settings, ckpt_dir, floor, float(sel_cfg["tie_margin"]),
                     hz, tol_s, device, log=log, augment=augment)
    model = head.build().to(device)
    model.load_state_dict(torch.load(res["best"], map_location=device))
    out_metrics, preds_digest = {}, {}
    for h in HALVES:
        m, tables, decoded = evaluate_head(head, model, inputs.halves[h], hz, tol_s, device)
        out_metrics[h] = m; preds_digest[h] = decoded_digest(decoded)
        np.savez_compressed(run_dir / f"predictions_{h}.npz", **{k: np.asarray(v["labels"], np.int8) for k, v in decoded.items()})
        extra = {f"{k}|{f}": np.asarray(v[f], np.float16) for k, v in decoded.items() for f in v if f not in DECODED_LABEL_KEYS}
        if extra:
            np.savez_compressed(run_dir / f"diagnostics_{h}.npz", **extra)
        for name, df in tables.items():
            df.to_csv(run_dir / f"rows_{h}_{name}.csv", index=False)
    timing = time_decode(head, model, inputs.halves["calibration"], device)
    shutil.copyfile(Path(ckpt_dir) / "train_log.jsonl", run_dir / "train_log.jsonl")
    plot_curves(res["records"], res["chosen_epoch"], run_dir / "training_curves.png", f"{row} ({head.name}), seed {settings.seed}")
    cal = out_metrics["calibration"]
    manifest = {
        "row": row, "head": head.name, "seed": settings.seed, "settings": res["settings"], "updates": res["updates"],
        "epochs_run": len(res["records"]), "chosen_epoch": res["chosen_epoch"], "checkpoint_status": res["status"],
        "native_floor_selection": floor, "n_params": res["n_params"],
        "max_train_peak_gpu_memory_mb": max((r["peak_gpu_memory_mb"] or 0) for r in res["records"]),
        "train_seconds": sum(r["seconds_train"] for r in res["records"]),
        "recall_requirement_calibration": {"min": float(sel_cfg["boundary_recall_min"]), "value": cal["boundary_recall_multi"],
                                           "met": (cal["boundary_recall_multi"] or 0) >= float(sel_cfg["boundary_recall_min"])},
        "native_floor_calibration": {"floor": baseline["calibration"]["native_macro_f1"] - float(sel_cfg["native_f1_tolerance"]),
                                     "value": cal["native_macro_f1"],
                                     "met": cal["native_macro_f1"] >= baseline["calibration"]["native_macro_f1"] - float(sel_cfg["native_f1_tolerance"])},
        "timing": timing, "protocol": inputs.protocol_path, "protocol_sha256": inputs.proto_sha,
        "feature_manifest": inputs.feature_manifest_path, "feature_manifest_sha256": inputs.fman_sha, "cache_key": inputs.cache_key,
        "checkpoints": {"best": res["best"], "best_sha256": sha256_file(res["best"]), "last": res["last"], "last_sha256": sha256_file(res["last"])},
        "predictions_digest": preds_digest, "decoding": getattr(head, "decoding", "per-sample argmax"),
        "source_sha256": {f: sha256_file(Path(__file__).resolve().parents[2] / f) for f in SOURCE_FILES},
        **(extra_manifest or {}),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (run_dir / "validation_metrics.json").write_text(json.dumps({"method": out_metrics, "encoder_frame_majority": baseline,
                                                                 "chosen_epoch": res["chosen_epoch"]}, indent=1, default=str))
    log(f"[{row}] chosen epoch {res['chosen_epoch']} ({res['status']}); calibration: native F1 {cal['native_macro_f1']:.4f} "
        f"segF1 {cal['segment_f1_50_multi']:.4f} recall {cal['boundary_recall_multi']:.3f} cuts {cal['native_internal_cuts_per_clip']:.3f}")
    return manifest


def add_common_args(ap) -> None:
    ap.add_argument("--feature-manifest", required=True, help="feature_manifest.json of the encoder cache")
    ap.add_argument("--protocol", required=True, help="protocol.json of model_development_protocol.py")
    ap.add_argument("--lr", type=float, default=None, help="learning-rate trial (default temporal_training.lr)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")


def settings_of(args, cfg: dict) -> Settings:
    tc = dict(cfg["temporal_training"])
    if args.max_epochs:
        tc["max_epochs"] = int(args.max_epochs)
    return Settings.from_cfg(tc, seed=int(cfg.get("seed", tc["seed"])), lr=args.lr)


def prepare(args, cfg: dict, row: str, settings: Settings, run_dir: Path, log: Callable[[str], None] = print):
    """(inputs, ckpt_dir, device) for one run directory."""
    paths = load_paths()
    ckpt_dir = paths.checkpoints_root / Path(run_dir).name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    log(f"[{row}] run_dir={run_dir}\n[{row}] ckpt_dir={ckpt_dir}\n[{row}] {settings}")
    return inputs, ckpt_dir, device
