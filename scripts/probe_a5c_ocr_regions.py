#!/usr/bin/env python3
"""Fixed offline OCR-proposed input-region diagnostic, not anatomical supervision."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import scripts.diagnose_a5c_image_dependence as d
from echo_routing.temporal import image_dependence as op
from echo_routing.temporal import ocr_masks as om
from scripts.probe_a5c_mask_geometry import translated_masks
OUT = d.OUT


def timeout(_signum, _frame):
    raise TimeoutError("fixed OCR preparation wall allowance reached")


class Engine:
    def __init__(self, tool):
        self.tool, self.cache, self.calls = tool, {}, 0
        prefix = OUT / "ocr_tool/usr/lib/x86_64-linux-gnu"
        self.env = dict(os.environ, OMP_THREAD_LIMIT="1", LD_LIBRARY_PATH=str(prefix) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))

    def read(self, image):
        a = np.asarray(image)
        key = hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
        if key not in self.cache:
            tmp = OUT / "ocr_input_private.png"; Image.fromarray(a).save(tmp)
            try:
                cmd = [str(OUT / self.tool["binary_relative_to_run"]), str(tmp), "stdout", "--tessdata-dir",
                       str((OUT / self.tool["language_relative_to_run"]).parent), "--oem", "1", "--psm", "11", "-l", "eng", "tsv"]
                r = subprocess.run(cmd, env=self.env, capture_output=True, text=True, check=True, timeout=10)
                self.cache[key] = r.stdout; self.calls += 1
            finally:
                tmp.unlink(missing_ok=True)
        return self.cache[key]


def prepare():
    if (OUT / "ocr_input_manifest.json").exists():
        raise RuntimeError("OCR masks already frozen; no source replacement")
    frozen = json.loads((OUT / "ocr_source_freeze.json").read_text())
    assert d.sha(OUT / "ocr_tool_manifest.json") == frozen["tool_manifest_sha256"]
    assert d.sha(ROOT / "scripts/probe_a5c_ocr_regions.py") == frozen["driver_code_sha256"]
    assert d.sha(ROOT / "echo_routing/temporal/ocr_masks.py") == frozen["mask_code_sha256"]
    assert d.sha(OUT / "input_freeze.json") == frozen["parent_input_manifest_sha256"]
    assert d.sha(OUT / "scores.npz") == frozen["baseline_scores_sha256"]
    tool = json.loads((OUT / "ocr_tool_manifest.json").read_text())
    for p, expected in tool["members"].items(): assert d.sha(OUT / p) == expected
    assert tool["synthetic_fixed_control_pass"]
    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    start, cpu, children = time.time(), time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    child_start = children.ru_utime + children.ru_stime
    engine, c = Engine(tool), d.ctx()
    baseline = np.load(OUT / "scores.npz")
    pixel_meta = json.loads((OUT / "pixel_metadata.json").read_text())
    boxes, masks, coverage = {}, {}, {}
    for h in (0, 1):
        rows = baseline[f"rows_f{h}"]; F = d.t44.Fold(c, 0, h); px = d.Pixels(F)
        expected_pixels = {q["row"]: q["pixel_sha256"] for q in pixel_meta[str(h)]["rows"]}
        all_masks, raw_masks, input_masks = [], [], []
        boxes[str(h)] = {}
        counts = {"rows": len(rows), "raw_accepted_words": 0, "processed_accepted_words": 0,
                  "raw_detected_rows": 0, "raw_masks_surviving_original_crop": 0,
                  "processed_detected_rows": 0, "nonempty_input_masks": 0,
                  "exact_view_code_word_candidates": 0}
        for j, row in enumerate(rows):
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            used = time.process_time() - cpu + usage.ru_utime + usage.ru_stime - child_start
            assert used < 585, "OCR source CPU allowance nearly exhausted; no case replacement"
            (x, _, _), _, frames = px.window(int(row)); mk = int(F.W["marker"][row])
            assert hashlib.sha256(x.tobytes()).hexdigest() == expected_pixels[int(row)]
            raw = frames[mk][4]
            native_tsv = engine.read(raw)
            scaled = np.asarray(Image.fromarray(x[mk]).resize((672, 672), Image.Resampling.BILINEAR))
            input_tsv = engine.read(scaled)
            raw_boxes, input_boxes = om.boxes_from_tsv(native_tsv), om.boxes_from_tsv(input_tsv)
            a = om.native_to_input_mask(raw.shape[:2], raw_boxes)
            b = om.scaled_to_input_mask(input_boxes); mask = a | b
            boxes[str(h)][str(int(row))] = {"raw_shape": list(raw.shape[:2]), "marker": mk,
                                           "raw_tsv": native_tsv, "processed_tsv": input_tsv,
                                           "raw_boxes": raw_boxes, "processed_boxes": input_boxes,
                                           "input_pixel_sha256": expected_pixels[int(row)]}
            all_masks.append(mask); raw_masks.append(a); input_masks.append(b)
            counts["raw_accepted_words"] += len(raw_boxes); counts["processed_accepted_words"] += len(input_boxes)
            counts["raw_detected_rows"] += bool(raw_boxes); counts["raw_masks_surviving_original_crop"] += bool(a.any())
            counts["processed_detected_rows"] += bool(input_boxes); counts["nonempty_input_masks"] += bool(mask.any())
            counts["exact_view_code_word_candidates"] += sum(''.join(c for c in v['text'].upper() if c.isalnum()) in {'A4C','A5C','4CH','5CH'} for v in raw_boxes + input_boxes)
            if (j + 1) % 20 == 0 or j + 1 == len(rows):
                rec = {"stage": "offline_OCR_source", "fold": h, "rows_done": j + 1, "rows_total": len(rows), "OCR_calls": engine.calls,
                       "elapsed_seconds": time.time() - start, "updated_epoch": time.time(), "pid": os.getpid()}
                d.dump(OUT / "ocr_prepare_progress.json", rec); print(json.dumps(rec), flush=True)
        masks[f"rows_f{h}"] = rows
        masks[f"masks_f{h}"] = np.stack(all_masks)
        masks[f"raw_mapped_masks_f{h}"] = np.stack(raw_masks)
        masks[f"processed_masks_f{h}"] = np.stack(input_masks)
        areas = masks[f"masks_f{h}"].mean((1, 2))
        counts["input_area_fraction_mean"] = float(areas.mean()); counts["input_area_fraction_median"] = float(np.median(areas))
        counts["input_area_fraction_max"] = float(areas.max())
        coverage[str(h)] = counts; px.frame.cache_clear()
    np.savez_compressed(OUT / "ocr_masks.npz", **masks); d.dump(OUT / "ocr_boxes_private.json", boxes)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {"parent_input_manifest_sha256": frozen["parent_input_manifest_sha256"], "tool_manifest_sha256": frozen["tool_manifest_sha256"],
               "baseline_scores_sha256": frozen["baseline_scores_sha256"],
               "source_freeze": frozen, "masks_sha256": d.sha(OUT / "ocr_masks.npz"), "boxes_sha256": d.sha(OUT / "ocr_boxes_private.json"),
               "coverage": coverage, "OCR_calls": engine.calls, "logical_marker_views": sum(len(masks[f"rows_f{h}"]) for h in (0,1)) * 2,
               "confidence_min": 60, "alphanumeric_min": 2, "processed_upscale": 3, "raw_padding": 2, "processed_padding": 1,
               "model_forwards": 0, "training_updates": 0, "wall_seconds": time.time() - start,
               "process_CPU_seconds": time.process_time() - cpu + usage.ru_utime + usage.ru_stime - child_start,
               "limits": "OCR-proposed regions may include nontext. Marker-frame positions are reused in other time slots. No anatomical or view ground truth."}
    d.dump(OUT / "ocr_input_manifest.json", receipt); d.dump(d.PUB / "ocr_input_manifest.json", receipt)
    print(json.dumps({"stage": "OCR_masks_prepared", "coverage": coverage, "OCR_calls": engine.calls, "process_CPU_seconds": receipt["process_CPU_seconds"]}), flush=True)


def run():
    import torch
    if (OUT / "ocr_started.json").exists(): raise RuntimeError("OCR-region intervention already started")
    manifest = json.loads((OUT / "ocr_input_manifest.json").read_text())
    freeze = json.loads((OUT / "ocr_model_freeze.json").read_text())
    assert d.sha(OUT / "ocr_input_manifest.json") == freeze["manifest_sha256"]
    assert d.sha(OUT / "ocr_masks.npz") == manifest["masks_sha256"]
    assert d.sha(ROOT / "scripts/probe_a5c_ocr_regions.py") == manifest["source_freeze"]["driver_code_sha256"]
    assert d.sha(OUT / "scores.npz") == manifest["baseline_scores_sha256"]
    assert d.sha(ROOT / "echo_routing/temporal/ocr_masks.py") == manifest["source_freeze"]["mask_code_sha256"]
    budget = freeze["resource_preflight"]
    assert budget["forecast_wall_upper_seconds"] < budget["wall_limit_seconds"]
    signal.alarm(int(budget["wall_limit_seconds"]) - 15)
    resource.setrlimit(resource.RLIMIT_CPU, (int(budget["CPU_limit_seconds"]), int(budget["CPU_limit_seconds"]) + 5))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    original, masks = np.load(OUT / "scores.npz"), np.load(OUT / "ocr_masks.npz")
    n_nonempty = sum(int(masks[f"masks_f{h}"].any((1,2)).sum()) for h in (0,1))
    start, results, hashes, done, reused = time.time(), {}, {}, 0, 0
    d.dump(OUT / "ocr_started.json", {"pid": os.getpid(), "start_epoch": start, "nonempty_rows": n_nonempty})
    device = torch.device("cuda") if n_nonempty else None
    if device is not None: torch.cuda.set_device(0)
    c = d.ctx()
    for h in (0,1):
        rows = masks[f"rows_f{h}"]; assert np.array_equal(rows, original[f"rows_f{h}"])
        F = d.t44.Fold(c, 0, h); px = d.Pixels(F)
        results[f"rows_f{h}"] = rows; hashes[str(h)] = {}
        nonempty_rows = rows[masks[f"masks_f{h}"].any((1,2))]
        selected_hash = {int(nonempty_rows[0]),int(nonempty_rows[len(nonempty_rows)//2]),int(nonempty_rows[-1])} if len(nonempty_rows) else set()
        for mi, model in enumerate(d.MODELS):
            S = np.empty((len(rows), 2, 4, 2), np.float32)
            m, head, before = None, None, None
            if masks[f"masks_f{h}"].any():
                m, head = (c.a31.a08(0, device), None) if model == 'A08' else d.t44.load_model(c, model, 0, 1-h, device)
                m.eval(); m.requires_grad_(False)
                if head is not None: head.requires_grad_(False)
                before = d.t44.tet.state_digest(m)
                order = torch.from_numpy(np.random.default_rng([20261031,51,h,mi]).permutation(512)).to(device)
            for j, row in enumerate(rows):
                mask = masks[f"masks_f{h}"][j]
                if not mask.any():
                    S[j] = original[f"global_{model}_f{h}"][j,0]; reused += 8
                    continue
                (x, _, _), _, _ = px.window(int(row)); marker = int(F.W["marker"][row])
                positions = translated_masks(mask); row_hashes = {}
                for fi, fill in enumerate(('mean','blur')):
                    rep = op.replacement(x, fill)
                    vv = [op.mask_window(x, q, prepared=rep) for q in positions]
                    S[j,fi] = d.predict(m, head, vv, marker, order, device)
                    if mi==0 and int(row) in selected_hash:
                        for ki,v in enumerate(vv): row_hashes[f'{fill}_{ki}']=hashlib.sha256(v.tobytes()).hexdigest()
                if mi==0 and row_hashes: hashes[str(h)][str(int(row))]=row_hashes
                done += 8
                if (j+1)%20==0 or j+1==len(rows):
                    rec={'stage':'OCR_region_interventions','fold':h,'model':model,'rows_done':j+1,'rows_total':len(rows),'forward_windows':done,
                         'expected_new_forwards':n_nonempty*24,'unchanged_reused_slots':reused,'elapsed_seconds':time.time()-start,'updated_epoch':time.time(),
                         'pid':os.getpid(),'device':torch.cuda.get_device_name(0)}
                    d.dump(OUT/'ocr_progress.json',rec);print(json.dumps(rec),flush=True)
            if m is not None:
                assert before==d.t44.tet.state_digest(m);del m,head;torch.cuda.empty_cache()
            results[f'ocr_{model}_f{h}']=S
        px.frame.cache_clear()
    np.savez_compressed(OUT/'ocr_scores.npz',**results);d.dump(OUT/'ocr_intervention_hashes.json',hashes)
    d.dump(OUT/'ocr_complete.json',{'pid':os.getpid(),'forward_windows':done,'expected_new_forwards':n_nonempty*24,'unchanged_reused_slots':reused,
                                 'expected_logical_slots':sum(len(masks[f'rows_f{h}']) for h in (0,1))*24,'training_updates':0,'models_unchanged':True,
                                 'scores_sha256':d.sha(OUT/'ocr_scores.npz'),'intervention_hashes_sha256':d.sha(OUT/'ocr_intervention_hashes.json'),
                                 'wall_seconds':time.time()-start,'GPU_model_process':bool(n_nonempty)})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','run'),required=True)
    args=ap.parse_args();{'prepare':prepare,'run':run}[args.stage]()
