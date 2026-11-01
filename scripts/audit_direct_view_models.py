#!/usr/bin/env python3
"""Independent A51-P source state/journal/optimizer and cache-only metric audit."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

OUT = ROOT / "runs/a51_verified_direct_view_pilot_v1"
PUB = ROOT / "docs/results/2026-11-01_direct_view_evidence/pilot"
BASE = ROOT / "models_frozen/video_within_view_v1/all_s0/final.pt"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        values = state[key].detach().cpu().contiguous().numpy()
        for b in (key.encode(), str(values.dtype).encode(), str(values.shape).encode(), values.tobytes()):
            digest.update(b)
    return digest.hexdigest()


def metrics(labels, scores, discordant):
    positives = [i for i, y in enumerate(labels) if y]
    negatives = [i for i, y in enumerate(labels) if not y]
    tp = int(sum(scores[i] > 0 for i in positives)); fn = len(positives) - tp
    fp = int(sum(scores[i] > 0 for i in negatives)); tn = len(negatives) - fp
    correct_discord = int(sum((scores[i] > 0) == bool(labels[i]) for i in range(len(labels)) if discordant[i]))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(labels), "balanced_accuracy": .5 * (tp / len(positives) + tn / len(negatives)),
            "a4c_recall": tn / len(negatives), "a5c_recall": tp / len(positives), "a4c_fpr": fp / len(negatives),
            "discordant_accuracy": correct_discord / int(sum(discordant)),
            "AUC": sum(float(scores[i] > scores[j]) + .5 * float(scores[i] == scores[j]) for i in positives for j in negatives) / (len(positives) * len(negatives))}


def source_models():
    started = time.time()
    manifest = json.loads((PUB / "source_models_manifest.json").read_text())
    inputs = json.loads((PUB / "input_manifest.json").read_text())
    assert manifest["input_manifest_sha256"] == sha(PUB / "input_manifest.json")
    assert manifest["Base_A08_sha256"] == inputs["Base_A08_sha256"] == sha(BASE)
    bound = json.loads((OUT / "bound.private.json").read_text())
    assert sha(OUT / "bound.private.json") == inputs["bound_sha256"]
    fit = [r for r in bound["requests"] if r["role"] == "fit"]
    plan = np.load(OUT / "source_plan.npy")
    assert plan.shape == (600, 8) and set(plan.ravel()) == set(range(569))
    dirs = np.asarray([r["directory_label"] == "A5C" for r in fit])
    assert np.all(dirs[plan].sum(1) == 4)
    assert hashlib.sha256(plan.astype(np.int64).tobytes()).hexdigest() == inputs["source_plan_sha256"]
    # Rebuild exposure from the independent fixed RNG rule.
    rng = np.random.default_rng(2026110151); streams = []
    for label in (False, True):
        pool = np.flatnonzero(dirs == label); stream = []
        while len(stream) < 2400: stream += rng.permutation(pool).tolist()
        streams.append(np.asarray(stream[:2400]).reshape(600, 4))
    assert np.array_equal(np.concatenate(streams, 1), plan)
    base = torch.load(BASE, map_location="cpu", weights_only=True)
    initial = {"m." + n: t for n, t in base.items()}
    initial["h.fc.weight"] = torch.zeros((1, 512)); initial["h.fc.bias"] = torch.full((1,), -2.)
    initial_digest = tensor_hash(initial)
    results = {}
    for arm in ("L", "W"):
        entry = manifest["models"][arm]; rd = OUT / "source_train" / arm
        assert entry["identity"]["input_manifest_sha256"] == sha(PUB / "input_manifest.json")
        assert entry["identity"]["plan_sha256"] == inputs["source_plan_sha256"]
        for filename, key in [("final.pt", "bundle_sha256"), ("optimizer_final.pt", "optimizer_final_sha256"),
                              ("updates.private.jsonl", "journal_sha256"), ("fit_init.npy", "fit_init_sha256"), ("fit_final.npy", "fit_final_sha256")]:
            assert sha(rd / filename) == entry[key]
        bundle = torch.load(rd / "final.pt", map_location="cpu", weights_only=True)
        optimizer = torch.load(rd / "optimizer_final.pt", map_location="cpu", weights_only=False)
        assert optimizer["update"] == 600 and optimizer["identity"] == entry["identity"]
        assert "cpu_rng" in optimizer and "cuda_rng" in optimizer and len(optimizer["cuda_rng"]) == 1
        assert optimizer["cpu_rng"].dtype == torch.uint8
        all_final = {**{"m." + n: t for n, t in optimizer["model"].items()}, **{"h." + n: t for n, t in optimizer["head"].items()}}
        reconstructed = {**initial, **bundle["live"]}
        assert set(initial) == set(reconstructed) == set(all_final)
        assert tensor_hash(reconstructed) == tensor_hash(all_final) == bundle["full_state_sha256"] == entry["complete_full_state_sha256"]
        frozen = {n: t for n, t in initial.items() if n not in bundle["live"]}
        assert tensor_hash(frozen) == bundle["frozen_state_sha256"] == entry["frozen_state_sha256"]
        assert all(torch.equal(all_final[n], t) for n, t in frozen.items())
        assert all(n.startswith("m.classifier.") for n in frozen), "unexpected source-frozen body tensors"
        assert initial_digest == entry["initial_full_state_sha256"]
        groups = optimizer["opt"]["param_groups"]
        assert [g["name"] for g in groups] == ["encoder_trunk", "marker_stem", "projection", "scalar_head"]
        assert [g["base"] for g in groups] == [1e-5, 1e-4, 1e-4, 1e-3]
        assert all(g["weight_decay"] == 1e-4 for g in groups)
        opt_states = optimizer["opt"]["state"]
        assert len(opt_states) == entry["optimizer_state_tensors"]
        assert set(int(s["step"]) for s in opt_states.values()) == {600}
        for state in opt_states.values():
            assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
            assert (state["exp_avg_sq"] >= 0).all()
        journal = [json.loads(line) for line in (rd / "updates.private.jsonl").read_text().splitlines()]
        assert len(journal) == 600 and [r["u"] for r in journal] == list(range(600))
        assert sum(r["physical_forwards"] for r in journal) == sum(r["physical_backwards"] for r in journal) == 4800
        guard = json.loads((rd / "update_guard.json").read_text())
        assert guard == {"pending": None, "last_committed": 599}
        assert np.array_equal(np.load(rd / "fit_init.npy"), np.full(569, -2., np.float32))
        drift_groups = {k: 0. for k in ("encoder_trunk", "marker_stem", "projection", "scalar_head")}
        for name in bundle["live"]:
            diff = float((all_final[name].double() - initial[name].double()).pow(2).sum())
            group = "scalar_head" if name.startswith("h.") else "marker_stem" if name.startswith("m.marker_stem.") else "projection" if name.startswith("m.encoder.head.") else "encoder_trunk"
            drift_groups[group] += diff
            assert abs(diff ** .5 - entry["drift_by_tensor"][name]) < 1e-10
        assert all(v > 0 for v in drift_groups.values())
        assert all(journal[0]["groups"][k] == 0 for k in ("encoder_trunk", "marker_stem", "projection"))
        assert all(any(r["groups"][k] > 0 for r in journal[1:]) for k in drift_groups)
        results[arm] = {"updates": 600, "optimizer_counter": 600, "full_optimizer_RNG_verified": True,
                        "complete_state_verified": True, "frozen_old_5class_head_verified": True,
                        "true_training_forwards": 4800, "true_training_backwards": 4800,
                        "parameter_drift_L2_by_group": {k: v ** .5 for k, v in drift_groups.items()}}
    rec = {"passed": True, "models": results, "source_formal_updates": 1200,
           "same_initial_full_state": True, "complete_plan_coverage": 569, "source_diagnostic_forwards_in_audit": 0,
           "source_models_manifest_sha256": sha(PUB / "source_models_manifest.json"), "Base_A08_unchanged": sha(BASE) == inputs["Base_A08_sha256"],
           "wall_seconds": time.time() - started,
           "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    (PUB / "independent_source_models.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


def source_final():
    started = time.time()
    models = json.loads((PUB / "independent_source_models.json").read_text()); assert models["passed"]
    assert models["source_models_manifest_sha256"] == sha(PUB / "source_models_manifest.json")
    manifest = json.loads((PUB / "source_models_manifest.json").read_text())
    raw = np.load(OUT / "source_scores.private.npz")
    report = json.loads((PUB / "source_readout.json").read_text())
    data = json.loads((OUT / "bound.private.json").read_text())
    rows = [r for r in data["requests"] if r["role"] == "diagnostic"]
    y = np.asarray([r["true_label"] == "A5C" for r in rows])
    discord = np.asarray([r["true_label"] != r["directory_label"] for r in rows])
    assert len(rows) == 133 and sum(y) == 39 and sum(~y) == 94 and sum(discord) == 17
    for arm in ("G", "L", "W"):
        root = OUT / "source_cells"
        progress = json.loads((root / (arm + ".progress.json")).read_text())
        complete = json.loads((root / (arm + ".complete.json")).read_text())
        expected_sha = manifest["Base_A08_sha256"] if arm == "G" else manifest["models"][arm]["bundle_sha256"]
        assert progress["identity"] == complete["identity"]
        assert progress["identity"]["model_sha256"] == expected_sha and progress["identity"]["bound_sha256"] == sha(OUT / "bound.private.json")
        assert progress["done"] == complete["rows"] == 133 and progress["pending"] is None
        assert complete["sha256"] == sha(root / (arm + ".npy"))
        assert np.array_equal(np.load(root / (arm + ".npy")), raw[arm])
    control = data["geometry_control"]; features = np.asarray([r["geometry"] for r in rows], float)
    geometry = ((features - control["fit_mean"]) / control["fit_scale"]) @ np.asarray(control["coef"])[0] + control["intercept"][0]
    assert np.array_equal(geometry, raw["geometry"])
    p = control["constant_prior"]; assert np.array_equal(np.full(133, np.log(p / (1 - p))), raw["constant"])
    independent = {arm: metrics(y, raw[arm], discord) for arm in raw.files}
    for arm in raw.files:
        for key, value in report["metrics"][arm].items(): assert abs(independent[arm][key] - value) < 1e-12
    l, w = independent["L"], independent["W"]
    checks = {"L_balanced_accuracy": l["balanced_accuracy"] >= .8, "L_recall_A4C": l["a4c_recall"] >= .7,
              "L_recall_A5C": l["a5c_recall"] >= .7, "L_increment": l["balanced_accuracy"] - w["balanced_accuracy"] >= .05 - 1e-12,
              "A4C_protection": l["a4c_fpr"] <= w["a4c_fpr"],
              "image_over_geometry": l["balanced_accuracy"] - independent["geometry"]["balanced_accuracy"] >= .10 - 1e-12,
              "image_over_constant": l["balanced_accuracy"] - independent["constant"]["balanced_accuracy"] >= .10 - 1e-12,
              "discordant_increment": l["discordant_accuracy"] > w["discordant_accuracy"]}
    assert checks == report["source_advancement_gate"]["checks"]
    assert all(checks.values()) == report["source_advancement_gate"]["passed"]
    for arm in raw.files:
        class_recalls = []
        for label in (False, True):
            per_patient = []
            for patient in {r["patient"] for r in rows}:
                ids = [i for i, r in enumerate(rows) if r["patient"] == patient and y[i] == label]
                if ids: per_patient.append(float(np.mean((raw[arm][ids] > 0) == y[ids])))
            class_recalls.append(float(np.mean(per_patient)))
        independent[arm]["equal_patient_balanced_accuracy"] = float(np.mean(class_recalls))
    rec = {"passed": True, "cache_only_no_new_model_forwards": True, "source_diagnostic_requests": 133,
           "source_metrics": independent, "source_advancement_gate": {"passed": all(checks.values()), "checks": checks},
           "raw_cache_sha256": sha(OUT / "source_scores.private.npz"), "model_manifest_sha256": sha(PUB / "source_models_manifest.json"),
           "source_model_manifest_actual_git_freeze": report["source_model_freeze"],
           "scope": "Verified-subset source heldout view signal only; no target A5C recovery, complete-F1 or anatomical claim.",
           "wall_seconds": time.time() - started,
           "process_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    (PUB / "independent_source_final.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["source", "source_final"]); args = parser.parse_args()
    torch.set_num_threads(1)
    {"source": source_models, "source_final": source_final}[args.stage]()
