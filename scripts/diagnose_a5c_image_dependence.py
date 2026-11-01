#!/usr/bin/env python3
"""Prospective frozen-model image-dependence audit; never a training/evaluation runner."""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import torch
import echo_routing.temporal.image_dependence as imd
import echo_routing.temporal.joint_visual as jv
import echo_routing.vit.target_mvit as tm
import scripts.train_risk_constrained_a5c as t44

OUT = ROOT / "runs/a5c_image_dependence_v1"
PUB = ROOT / "docs/results/2026-10-31_a5c_image_dependence"
MODELS = ("A08", "C", "N")
sha = t44.sha256_file


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def ctx():
    p = "configs/experiments/video_risk_constrained_a5c_v1.yaml"
    return t44.Ctx(t44.load_cfg(SimpleNamespace(config=p, set=[])), p)


def sources_for(F):
    src, sess = [], []
    for i in range(len(F.pk["base"])):
        seq = F.seqs[int(F.pk["seq"][i])]
        refs = seq["refs"][int(F.pk["a"][i]):int(F.pk["b"][i])]
        counts = Counter(r[0] for r in refs)
        v = sorted(counts, key=lambda v: (-counts[v], v))[0]
        src.append(v); sess.append(t44.session_of(v))
    return np.array(src), np.array(sess)


def module_files():
    files = {}
    for m in list(sys.modules.values()):
        p = getattr(m, "__file__", None)
        if not p:
            continue
        p = Path(p).resolve()
        if p.is_relative_to(ROOT) and ".venv" not in p.parts and p.suffix == ".py":
            files[str(p.relative_to(ROOT))] = sha(p)
    for p in [ROOT / "scripts/diagnose_a5c_image_dependence.py", ROOT / "docs/superpowers/plans/2026-10-31-a5c-image-dependence.md"]:
        files[str(p.relative_to(ROOT))] = sha(p)
    return files


def prepare():
    if (OUT / "selection.json").exists():
        raise RuntimeError("selection already exists; no cohort replacement")
    c = ctx(); old = np.load(c.out / "readout_cache_s0.npz")
    cases, public, pins = [], {}, {}
    pins[str((c.out / "readout_cache_s0.npz").relative_to(ROOT))] = sha(c.out / "readout_cache_s0.npz")
    for h in (0, 1):
        F = t44.Fold(c, 0, h)
        y, base, pred = (old[f"{key}_heldout_f{h}"] for key in ("run_label", "run_pred_Base", "run_pred_N"))
        assert np.array_equal(y, F.pk["labels"]) and np.array_equal(base, F.pk["base"])
        src, sess = sources_for(F)
        groups = imd.outcome_groups(y, base, pred)
        public[str(h)] = {}
        for gi, g in enumerate(imd.GROUPS):
            selected, levels = imd.select_indices(groups[g], src, sess, [20261031, 51, h, gi], 8)
            public[str(h)][g] = {"denominator": len(groups[g]), "selected": len(selected),
                                  "selected_sessions": len(set(sess[selected])), "selected_recordings": len(set(src[selected])),
                                  "selection_levels": dict(Counter(levels))}
            for i, level in zip(selected, levels):
                rows, weights = t44.rc.members(F.pk, i, F.off)
                assert np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
                anchor = int(rows[np.flatnonzero(weights == weights.max())[0]])
                cases.append({"fold": h, "group": g, "run": i, "source": str(src[i]), "session": str(sess[i]),
                              "selection_level": level, "rows": rows.tolist(), "weights": weights.tolist(), "anchor": anchor})
        for p in [c.a31.out / f"pool_{h}.npy", c.a31.out / "data" / f"runs_s0_f{h}.npz"]:
            pins[str(p.relative_to(ROOT))] = sha(p)
    model_paths = [ROOT / "models_frozen/video_within_view_v1/all_s0/final.pt"]
    committed = json.loads((c.out / "weights_committed.json").read_text())
    for f in (0, 1):
        for a in ("C", "N"):
            key = f"{a}_s0_f{f}"
            for name, k in [("final.pt", "final"), ("head.pt", "head")]:
                p = c.out / "train" / key / name
                assert sha(p) == committed[key][k]
                model_paths.append(p)
    for p in model_paths:
        pins[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT / "selection.json", {"cases": cases})
    manifest = {"run_id": "a5c_image_dependence_v1", "mode": "outcome_conditioned_training_source_diagnostic",
                "selection_sha256": sha(OUT / "selection.json"), "groups": public, "case_count": len(cases),
                "all_original_denominators": {"A5C_missed_by_Base": 103, "correct_A4C": 1413},
                "pinned_files": pins, "executed_project_code": module_files(),
                "global_names": list(imd.GLOBAL_NAMES), "models": list(MODELS),
                "spatial_cells": 49, "spatial_replacements": ["mean", "blur"], "training_updates": 0,
                "torch": torch.__version__, "numpy": np.__version__, "OCR_available": False,
                "limits": ["No expert/anatomical truth", "No new heldout efficacy", "No private case export", "No training/threshold selection"]}
    dump(PUB / "input_freeze.json", manifest)
    dump(OUT / "input_freeze.json", manifest)
    print(json.dumps({"stage": "prepared", "cases": len(cases), "groups": public, "manifest_sha256": sha(OUT / "input_freeze.json")}), flush=True)


class Pixels:
    def __init__(self, F):
        self.F = F

    @lru_cache(maxsize=128)
    def frame(self, video, frame, gamma):
        paths = self.F.fp(video)
        path = paths[min(int(frame), len(paths) - 1)]
        raw = np.asarray(Image.open(path).convert("RGB"))
        im = Image.fromarray(raw)
        if abs(gamma - 1) > 1e-9:
            im = im.point(jv.gamma_lut(gamma))
        original = jv.crop_uint8(im)
        full, retained, rec = imd.letterbox_pair(raw, gamma)
        rec.pop("retained_mask")
        return original, full, retained, rec, raw, str(path)

    def window(self, row):
        F = self.F
        seq = F.seqs[int(F.W["seq"][row])]
        refs = [seq["refs"][int(i)] for i in F.W["s"][row]]
        frames = [self.frame(v, int(f), round(float(g), 6)) for v, f, g in refs]
        return tuple(np.stack([q[j] for q in frames]) for j in range(3)), [q[3] for q in frames], frames


def readout_weights(m, head):
    if head is None:
        return (m.classifier.weight[3] - m.classifier.weight[2]).detach(), (m.classifier.bias[3] - m.classifier.bias[2]).detach()
    return head.fc.weight[0].detach(), head.fc.bias[0].detach()


@torch.no_grad()
def predict(m, head, windows, marker, random_order, device):
    inp = tm.to_rgb(torch.from_numpy(np.stack(windows)).to(device))
    mk = tm.marker_planes(torch.full((len(windows),), int(marker), dtype=torch.long, device=device))
    e = m.embed(inp, mk)
    w, b = readout_weights(m, head)
    return torch.stack([e @ w + b, e @ w[random_order] + b], -1).float().cpu().numpy()


def private_contact(path, x, marker, frames, extras):
    raw = Image.fromarray(frames[marker][4]); raw.thumbnail((640, 480))
    canvas = Image.new("RGB", (max(672, raw.width), raw.height + 254))
    canvas.paste(raw, (0, 0))
    for j, pic in enumerate([x[marker], extras[0][marker], extras[1][marker]]):
        canvas.paste(Image.fromarray(pic), (224 * j, raw.height + 25))
    ImageDraw.Draw(canvas).text((2, raw.height + 4), "original input | full letterbox | retained footprint; no anatomical labels", fill="white")
    path.parent.mkdir(exist_ok=True, parents=True); canvas.save(path, quality=85)


def run():
    if (OUT / "scores.npz").exists() or (OUT / "run.started.json").exists():
        raise RuntimeError("diagnostic already started; no second readout or blind restart")
    freeze = json.loads((OUT / "freeze_receipt.json").read_text())
    manifest = json.loads((OUT / "input_freeze.json").read_text())
    assert freeze["manifest_sha256"] == sha(OUT / "input_freeze.json")
    assert manifest["selection_sha256"] == sha(OUT / "selection.json")
    for path, h in {**manifest["pinned_files"], **manifest["executed_project_code"]}.items():
        assert sha(ROOT / path) == h, "pinned file changed: " + path
    dump(OUT / "run.started.json", {"pid": os.getpid(), "started_epoch": time.time(), "freeze": freeze})
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2410))
    signal.alarm(3300)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    c, output, parity, metadata = ctx(), {}, {}, {}
    cases = json.loads((OUT / "selection.json").read_text())["cases"]
    old = np.load(c.out / "readout_cache_s0.npz")
    total, done, job_start = 0, 0, time.time()
    for h in (0, 1):
        cc = [q for q in cases if q["fold"] == h]
        rows = sorted({r for q in cc for r in q["rows"]})
        total += len(rows) * 11 * 3 + len({q["anchor"] for q in cc}) * 102 * 3
    for h in (0, 1):
        cc = [q for q in cases if q["fold"] == h]
        F = t44.Fold(c, 0, h); px = Pixels(F)
        rows = np.array(sorted({r for q in cc for r in q["rows"]}), int)
        anchors = np.array(sorted({q["anchor"] for q in cc}), int)
        output[f"rows_f{h}"] = rows; output[f"anchors_f{h}"] = anchors
        positions = {r: j for j, r in enumerate(old[f"rows_heldout_f{h}"])}
        O = t44.t31.outputs(c.a31, 0, h)
        expected = {"A08": O["logits"][rows, 3] - O["logits"][rows, 2],
                    **{a: old[f"scores_{a}_heldout_f{h}"][[positions[r] for r in rows]] for a in ("C", "N")}}
        metadata[str(h)] = {"rows": [], "anchors": []}
        for mi, model in enumerate(MODELS):
            if model == "A08":
                m, head = c.a31.a08(0, device), None
            else:
                m, head = t44.load_model(c, model, 0, 1 - h, device)
            m.eval(); m.requires_grad_(False)
            if head is not None:
                head.requires_grad_(False)
            before = t44.tet.state_digest(m)
            perm = np.random.default_rng([20261031, 51, h, mi]).permutation(512)
            order = torch.from_numpy(perm).to(device)
            global_s = np.empty((len(rows), 11, 2), np.float32)
            map_s = np.empty((len(anchors), 2, 49, 2), np.float32)
            time_s = np.empty((len(anchors), 4, 2), np.float32)
            max_error = 0.
            for j, row in enumerate(rows):
                (x, full, retained), rec, frames = px.window(int(row))
                marker = int(F.W["marker"][row])
                baseline = predict(m, head, [x], marker, order, device)[0]
                err = abs(float(baseline[0]) - float(expected[model][j])); max_error = max(max_error, err)
                if err > 1e-4:
                    dump(OUT / "parity_failure.json", {"fold": h, "model": model, "absolute_error": err})
                    raise RuntimeError("original-output parity failed")
                global_s[j, 0] = baseline
                variants = list(imd.global_variants(x, marker, full, retained, int(row) + 100000 * h))
                assert [v[0] for v in variants] == list(imd.GLOBAL_NAMES)
                for lo in range(1, len(variants), 4):
                    vv = [q[1] for q in variants[lo:lo + 4]]
                    global_s[j, lo:lo + len(vv)] = predict(m, head, vv, marker, order, device)
                done += len(variants)
                if mi == 0:
                    metadata[str(h)]["rows"].append({"row": int(row), "pixel_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                                                    "crop_retention_mean": float(np.mean([z["raw_crop_retained_fraction"] for z in rec])),
                                                    "native_shapes": sorted({tuple(z["raw_shape"]) for z in rec}), **imd.image_proxies(x)})
                if (j + 1) % 10 == 0 or j + 1 == len(rows):
                    progress(h, model, "global", j + 1, len(rows), done, total, job_start)
            parity[f"{model}_f{h}"] = max_error
            for j, row in enumerate(anchors):
                (x, full, retained), rec, frames = px.window(int(row))
                marker = int(F.W["marker"][row])
                for fi, fill in enumerate(("mean", "blur")):
                    repl = imd.replacement(x, fill)
                    grid = imd.grid_masks()
                    for lo in range(0, 49, 4):
                        vv = [imd.mask_window(x, mask, prepared=repl) for mask in grid[lo:lo + 4]]
                        map_s[j, fi, lo:lo + len(vv)] = predict(m, head, vv, marker, order, device)
                rep = imd.replacement(x, "mean")
                vv = []
                for b in range(4):
                    z = x.copy(); z[b * 4:(b + 1) * 4] = rep[b * 4:(b + 1) * 4]; vv.append(z)
                time_s[j] = predict(m, head, vv, marker, order, device)
                if mi == 0:
                    private_contact(OUT / "private_review" / f"fold{h}_case{j:02d}.jpg", x, marker, frames, (full, retained))
                    metadata[str(h)]["anchors"].append({"row": int(row), "marker": marker})
                done += 102
                if (j + 1) % 4 == 0 or j + 1 == len(anchors):
                    progress(h, model, "maps", j + 1, len(anchors), done, total, job_start)
            assert before == t44.tet.state_digest(m), "frozen model changed"
            output[f"global_{model}_f{h}"] = global_s
            output[f"maps_{model}_f{h}"] = map_s
            output[f"temporal_{model}_f{h}"] = time_s
            del m, head; torch.cuda.empty_cache()
        px.frame.cache_clear()
    np.savez_compressed(OUT / "scores.npz", **output)
    dump(OUT / "pixel_metadata.json", metadata)
    assert all(sha(ROOT / p) == h for p, h in manifest["pinned_files"].items())
    dump(OUT / "run_complete.json", {"pid": os.getpid(), "device": torch.cuda.get_device_name(0), "cases": len(cases),
                                     "parity_max_abs": parity, "frozen_models_unchanged": True,
                                     "training_updates": 0, "forward_windows": done, "expected_windows": total,
                                     "scores_sha256": sha(OUT / "scores.npz"), "pixel_metadata_sha256": sha(OUT / "pixel_metadata.json"),
                                     "gpu_process_wall_seconds": time.time() - job_start, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    progress(1, "all", "complete", done, total, done, total, job_start)


def progress(fold, model, stage, n, target, done, total, start):
    rec = {"pid": os.getpid(), "device": torch.cuda.get_device_name(0), "fold": fold, "model": model,
           "stage": stage, "stage_done": n, "stage_total": target, "forward_windows": done,
           "expected_windows": total, "elapsed_seconds": time.time() - start, "updated_epoch": time.time()}
    dump(OUT / "progress.json", rec)
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("prepare", "run"), required=True)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    start, cpu = time.time(), time.process_time()
    try:
        {"prepare": prepare, "run": run}[a.stage]()
    finally:
        with (OUT / "stages.jsonl").open("a") as f:
            f.write(json.dumps({"stage": a.stage, "pid": os.getpid(), "start_epoch": start, "end_epoch": time.time(),
                                "wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu}) + "\n")


if __name__ == "__main__":
    main()
