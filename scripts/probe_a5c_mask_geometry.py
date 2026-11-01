#!/usr/bin/env python3
"""Sequential toroidal location controls; planar seam confounds are retained."""
from __future__ import annotations

import numpy as np

OFFSETS = ((0, 0), (0, 112), (112, 0), (112, 112))


def translated_masks(mask):
    if np.asarray(mask).shape != (224, 224):
        raise ValueError("expected224 spatial mask")
    return [np.roll(mask, shift, axis=(0, 1)) for shift in OFFSETS]


def mask_properties(mask, actual):
    m, a = np.asarray(mask, bool), np.asarray(actual, bool)
    contact = int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum())
    inner_edges = int((m[1:] != m[:-1]).sum() + (m[:, 1:] != m[:, :-1]).sum())
    union = int((m | a).sum())
    return {"area_pixels": int(m.sum()), "empty": not bool(m.any()), "image_edge_contacts": contact,
            "periodic_boundary_edges": int((m != np.roll(m, 1, axis=0)).sum() + (m != np.roll(m, 1, axis=1)).sum()),
            "planar_boundary_edges": inner_edges + contact,
            "jaccard_with_actual": float((m & a).sum() / union) if union else None}


def main():
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import hashlib
    import json
    from pathlib import Path
    import resource
    import signal
    import sys
    import time
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import torch
    import scripts.diagnose_a5c_image_dependence as d
    from echo_routing.temporal import image_dependence as op
    out = d.OUT
    if (out / "geometry_started.json").exists():
        raise RuntimeError("shape probe already started; no duplicate")
    freeze = json.loads((out / "geometry_freeze.json").read_text())
    manifest = json.loads((out / "geometry_manifest.json").read_text())
    assert d.sha(out / "geometry_manifest.json") == freeze["manifest_sha256"]
    assert d.sha(out / "input_freeze.json") == manifest["parent_input_manifest_sha256"]
    assert d.sha(root / "scripts/probe_a5c_mask_geometry.py") == manifest["probe_code_sha256"]
    done_receipt = json.loads((out / "process_receipt.json").read_text())
    assert done_receipt["exit_code"] == 0
    complete = json.loads((out / "run_complete.json").read_text())
    assert complete["scores_sha256"] == d.sha(out / "scores.npz")
    limit = min(3300, int(3600 - done_receipt["full_process_wall_seconds"] - 60))
    assert limit > 300, "insufficient shared GPU allowance for supplement and closure"
    budget = manifest["resource_preflight"]
    assert budget["forecast_wall_upper_seconds"] < limit
    limit = min(limit, int(budget["probe_wall_limit_seconds"]))
    cpu_limit = int(budget["probe_CPU_limit_seconds"])
    assert cpu_limit >= 60 and budget["forecast_CPU_upper_seconds"] < cpu_limit
    signal.alarm(limit); resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_device(0)
    start, cpu = time.time(), time.process_time()
    d.dump(out / "geometry_started.json", {"pid": os.getpid(), "start_epoch": start, "limit_seconds": limit})
    old = np.load(out / "scores.npz"); c = d.ctx()
    meta = json.loads((out / "pixel_metadata.json").read_text())
    total = sum(len(old[f"rows_f{h}"]) for h in (0, 1)) * 24 * 3
    scored, results, hashes, mask_meta = 0, {}, {}, {}
    for h in (0, 1):
        F = d.t44.Fold(c, 0, h); px = d.Pixels(F); rows = old[f"rows_f{h}"]
        results[f"rows_f{h}"] = rows
        original_meta = {q["row"]: q for q in meta[str(h)]["rows"]}
        audit_rows = {int(rows[0]), int(rows[len(rows) // 2]), int(rows[-1])}
        hashes[str(h)] = {}
        mask_meta[str(h)] = {}
        for mi, model in enumerate(d.MODELS):
            m, head = (c.a31.a08(0, device), None) if model == "A08" else d.t44.load_model(c, model, 0, 1 - h, device)
            m.eval(); m.requires_grad_(False)
            if head is not None: head.requires_grad_(False)
            initial = d.t44.tet.state_digest(m)
            order = torch.from_numpy(np.random.default_rng([20261031, 51, h, mi]).permutation(512)).to(device)
            scores = np.empty((len(rows), 3, 2, 4, 2), np.float32)
            for j, row in enumerate(rows):
                (x, _, _), _, _ = px.window(int(row))
                assert hashlib.sha256(x.tobytes()).hexdigest() == original_meta[int(row)]["pixel_sha256"]
                masks = op.proxy_masks(x); marker = int(F.W["marker"][row])
                row_hashes = {}
                for ni, (name, mask) in enumerate(masks.items()):
                    positions = translated_masks(mask)
                    if mi == 0:
                        mask_meta[str(h)].setdefault(str(int(row)), {})[name] = [mask_properties(q, mask) for q in positions]
                    for fi, fill in enumerate(("mean", "blur")):
                        rep = op.replacement(x, fill)
                        vv = [op.mask_window(x, q, prepared=rep) for q in positions]
                        scores[j, ni, fi] = d.predict(m, head, vv, marker, order, device)
                        if mi == 0 and int(row) in audit_rows:
                            for ki, v in enumerate(vv): row_hashes[f"{name}_{fill}_{ki}"] = hashlib.sha256(v.tobytes()).hexdigest()
                if mi == 0 and int(row) in audit_rows: hashes[str(h)][str(int(row))] = row_hashes
                scored += 24
                if (j + 1) % 10 == 0 or j + 1 == len(rows):
                    rec = {"pid": os.getpid(), "device": torch.cuda.get_device_name(0), "stage": "shape_controls", "fold": h,
                           "model": model, "rows_done": j + 1, "rows_total": len(rows), "forward_windows": scored,
                           "expected_windows": total, "elapsed_seconds": time.time() - start, "updated_epoch": time.time()}
                    d.dump(out / "geometry_progress.json", rec); print(json.dumps(rec), flush=True)
            assert initial == d.t44.tet.state_digest(m)
            results[f"geometry_{model}_f{h}"] = scores
            del m, head; torch.cuda.empty_cache()
        px.frame.cache_clear()
    np.savez_compressed(out / "geometry_scores.npz", **results)
    d.dump(out / "geometry_input_hashes.json", hashes)
    d.dump(out / "geometry_mask_metadata.json", mask_meta)
    d.dump(out / "geometry_complete.json", {"pid": os.getpid(), "forward_windows": scored, "expected_windows": total,
                                           "gpu_process_wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu,
                                           "model_training_updates": 0, "models_unchanged": True,
                                           "scores_sha256": d.sha(out / "geometry_scores.npz"),
                                           "input_hashes_sha256": d.sha(out / "geometry_input_hashes.json"),
                                           "mask_metadata_sha256": d.sha(out / "geometry_mask_metadata.json")})


if __name__ == "__main__":
    main()
