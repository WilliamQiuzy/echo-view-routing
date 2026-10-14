#!/usr/bin/env python
"""Prepare the training labels of the video dense supervision round (docs/superpowers/plans/2026-10-13-video-dense-supervision.md,
Task 1; configs/experiments/video_dense_supervision_v1.yaml). CPU only; nothing trains and no development data is read.

1. Every pinned source is checked: the task sheet, the V01 and V02 configurations, the three V01 Ordered final weights, the
   RV and SC manifests with their indexes, the reference summaries and saved outputs, A01's configuration and features, and
   the server's environment including its torchvision MViT source.
2. The model geometry is read from the network: the final token grid must be (8, 7, 7) with query time stride 1, the RGB
   stem's time kernel, stride, padding and dilation must be (3, 2, 1, 1), and the anchors j * stride - padding +
   (kernel - 1) * dilation / 2 must equal the configured [0, 2, ..., 14]. Each seed's source state loads strictly.
3. For every seed, the SC window list (the only source of per-slot labels) and the RV window list must describe the same
   windows: every slot's recording, frame, gamma and sample index, every window's centre, target, kind, sequence and
   repeats, the video list, the RV marker (the middle target slot), RV's identity Ordered order, and the SC labels at the
   target slots against the target. Every recording must be a training clip of the protocol.
4. The aligned labels are the SC labels at the 8 anchor slots. The shuffled control permutes each window's 8 labels once
   with numpy default_rng([20261013, seed, update, window]); the multiset stays and nothing global is touched.
Writes runs/video_dense_supervision_v1/manifests/AUX_s{seed}.npz and index.json (server only), and the id-free
preparation.json (also in the results directory).

  python scripts/prepare_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.prepare_raw_video_context import allowlist, environment  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

PERM_BASE = 20261013
N_TIME = 8
SLOT_KEYS = ("vid", "frame", "gamma", "s")                # per slot (U, W, 16)
WINDOW_KEYS = ("c", "target", "kind", "seq", "repeats")   # per window (U, W)


# ----------------------------------------------------------------------------------------------- pure helpers
def stem_anchors(kernel: int, stride: int, padding: int, dilation: int, n_time: int = N_TIME, n_slots: int = tm.N_SLOTS) -> list[int]:
    """Input-slot centre of every stem time position: j * stride - padding + (kernel - 1) * dilation / 2."""
    out = [j * int(stride) - int(padding) + (int(kernel) - 1) * int(dilation) / 2 for j in range(int(n_time))]
    if any(x != int(x) for x in out) or any(not 0 <= x < n_slots for x in out):
        raise ValueError(f"stem time geometry {(kernel, stride, padding, dilation)} gives anchors {out} outside the {n_slots} integer slots")
    return [int(x) for x in out]


def pinned(root: Path, pin: dict) -> Path:
    path = Path(root) / pin["path"]
    if (got := sha256_file(path)) != pin["sha256"]:
        raise SystemExit(f"{pin['path']}: sha256 {got[:12]} differs from the pin {pin['sha256'][:12]}")
    return path


def check_allowlist(videos, train: set) -> None:
    outside = {str(v) for v in videos} - set(train)
    if outside:
        raise SystemExit(f"{len(outside)} manifest recordings are not training clips of the protocol")


def check_pairing(sc: dict, rv: dict) -> dict:
    """The SC and RV lists must describe the same windows; any difference raises ValueError (counts only in the message)."""
    for k in SLOT_KEYS + WINDOW_KEYS:
        a, b = np.asarray(sc[k]), np.asarray(rv[k])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"SC and RV differ in '{k}' ({int(np.sum(a != b)) if a.shape == b.shape else 'shape'} entries)")
    if np.asarray(sc["videos"]).tolist() != np.asarray(rv["videos"]).tolist():
        raise ValueError("SC and RV list different recordings")
    u, w = sc["c"].shape; bad_marker = bad_target = 0
    order_ok = np.array_equal(rv["order_ordered"], np.broadcast_to(np.arange(tm.N_SLOTS, dtype=rv["order_ordered"].dtype), rv["order_ordered"].shape))
    if not order_ok:
        raise ValueError("RV's Ordered order is not the identity")
    for i in range(u):
        for j in range(w):
            s, c = sc["s"][i, j], int(sc["c"][i, j])
            bad_marker += int(int(rv["marker"][i, j]) != tm.marker_slot(s, c))
            bad_target += int(np.any(sc["labels"][i, j][s == c] != sc["target"][i, j]))
    if bad_marker or bad_target:
        raise ValueError(f"{bad_marker} RV markers differ from the SC middle target slot; {bad_target} SC target-slot labels differ from the target")
    return {"windows": int(u * w), "updates": int(u), "windows_per_update": int(w), "mismatches": 0}


def aux_labels(labels: np.ndarray, anchors) -> np.ndarray:
    """(U, W, 16) per-slot labels -> (U, W, 8) labels at the anchor slots."""
    return np.ascontiguousarray(np.asarray(labels)[:, :, list(anchors)]).astype(np.int8)


def permutations(seed: int, n_updates: int, n_windows: int) -> np.ndarray:
    """(U, W, 8) one fixed permutation per window from its own NumPy generator; no global RNG is used."""
    out = np.empty((int(n_updates), int(n_windows), N_TIME), np.int8)
    for u in range(int(n_updates)):
        for w in range(int(n_windows)):
            out[u, w] = np.random.default_rng([PERM_BASE, int(seed), u, w]).permutation(N_TIME)
    return out


def shuffled_labels(aligned: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """New label j = aligned label perm[j] of the same window."""
    return np.take_along_axis(np.asarray(aligned), np.asarray(perm, dtype=np.int64), axis=2).astype(np.int8)


def label_summary(aligned: np.ndarray, shuffled: np.ndarray, perm: np.ndarray, kind: np.ndarray, s: np.ndarray, anchors) -> dict:
    """Counts of the label files: windows, permutation identities, windows whose labels really change, repeated anchors."""
    native, mixed = np.asarray(kind) == 0, np.asarray(kind) == 1
    ident = (np.asarray(perm) == np.arange(N_TIME)).all(-1); changed = (shuffled != aligned).any(-1)
    at = np.asarray(s)[:, :, list(anchors)]; dup = np.zeros(at.shape, bool)
    for j in range(1, at.shape[-1]):
        dup[..., j] = (at[..., j][..., None] == at[..., :j]).any(-1)
    multi_class = aligned.max(-1) != aligned.min(-1)
    return {"windows": int(aligned.shape[0] * aligned.shape[1]), "native_windows": int(native.sum()), "mixed_windows": int(mixed.sum()),
            "identity_permutations": int(ident.sum()), "identity_permutation_share": float(ident.mean()),
            "mixed_windows_with_two_or_more_anchor_classes": int((multi_class & mixed).sum()),
            "mixed_windows_with_changed_labels": int((changed & mixed).sum()),
            "mixed_changed_window_share": float((changed & mixed).sum() / max(1, mixed.sum())),
            "native_windows_with_changed_labels": int((changed & native).sum()),
            "changed_anchor_labels_mixed": int(((shuffled != aligned) & mixed[..., None]).sum()),
            "changed_anchor_label_share_mixed": float(((shuffled != aligned) & mixed[..., None]).sum() / max(1, mixed.sum() * N_TIME)),
            "repeated_anchor_share": float(dup.mean()), "windows_with_repeated_anchors": int(dup.any(-1).sum()),
            "anchor_label_counts": {str(k): int((aligned == k).sum()) for k in range(tm.NUM_CLASSES)}}


# ----------------------------------------------------------------------------------------------- the round's sources
def source_pins(vd: dict) -> dict[str, dict]:
    src = vd["source"]; out = {"raw_video_config": src["raw_video_config"], "readout_config": src["readout_config"],
                               "rv_index": src["rv_manifests"]["index"], "sc_index": src["sc_manifests"]["index"], "a01_config": src["a01"]["config"]}
    out |= {f"weights_s{k}": v for k, v in src["weights"].items()} | {f"rv_s{k}": v for k, v in src["rv_manifests"]["files"].items()}
    out |= {f"sc_s{k}": v for k, v in src["sc_manifests"]["files"].items()} | {k: v for k, v in src["reports"].items()}
    out |= {f"a01_features_s{k}": v for k, v in src["a01"]["features"].items()}
    for name in ("ordered_outputs", "continued_cls_outputs"):
        out |= {f"{name}_{h}_s{k}": v for h, per in src[name].items() for k, v in per.items()}
    return out


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def model_geometry(vd: dict) -> dict:
    """Token grid, query strides and stem time geometry of the torchvision encoder, and the anchors they give."""
    import echo_routing.vit.target_patch_readout as tp
    enc = tm.build_source_encoder(None); grid, strides = tp.token_grid(enc); conv = enc.conv_proj
    stem = [int(conv.kernel_size[0]), int(conv.stride[0]), int(conv.padding[0]), int(conv.dilation[0])]
    if tuple(grid) != (8, 7, 7) or any(s[0] != 1 for s in strides) or stem != list(vd["model"]["stem_time"]):
        raise SystemExit(f"grid {grid}, query time strides {sorted({s[0] for s in strides})} or stem {stem} differ from the plan")
    anchors = stem_anchors(*stem, grid[0])
    if anchors != list(vd["model"]["anchors"]):
        raise SystemExit(f"anchors {anchors} differ from the configured {vd['model']['anchors']}")
    return {"grid": list(grid), "query_time_strides": sorted({int(s[0]) for s in strides}), "stem_time": stem, "anchors": anchors}


def write_aux(path: Path, arrays: dict) -> str:
    """Write the AUX file once; an existing file must hold identical arrays (generation is deterministic)."""
    if path.is_file():
        old = load_npz(path)
        if set(old) != set(arrays) or any(not np.array_equal(old[k], arrays[k]) for k in arrays):
            raise SystemExit(f"{path.name} exists with other content; it is never overwritten")
        return sha256_file(path)
    tmp = path.with_name(path.stem + ".tmp.npz"); np.savez(tmp, **arrays); tmp.replace(path)
    return sha256_file(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); vd = cfg["video_dense_supervision"]; t0 = time.time()
    paths = load_paths(); root = paths.repo_root
    checked = {"task_sheet": sha256_file(pinned(root, vd["task_sheet"]))}
    for name, pin in source_pins(vd).items():
        checked[name] = sha256_file(pinned(root, pin))
    env = environment(); want = vd["source"]["environment"]
    if any(str(env[k]) != str(want[k]) for k in want):
        raise SystemExit(f"environment {env} differs from {want}")
    rv_cfg = load_config(root / vd["source"]["raw_video_config"]["path"])["raw_video_context"]
    train = set(allowlist(rv_cfg, root)); geom = model_geometry(vd); anchors = geom["anchors"]
    mdir = root / vd["outputs"]["root"] / "manifests"; mdir.mkdir(parents=True, exist_ok=True)
    rec = {"config": a.config, "checked": checked, "environment": env, "allowlist_clips": len(train), "geometry": geom, "seeds": {}}
    index = {"anchors": anchors, "perm_base": PERM_BASE, "seeds": {}}
    for s in [int(x) for x in vd["training"]["seeds"]]:
        sc = load_npz(pinned(root, vd["source"]["sc_manifests"]["files"][s])); rv = load_npz(pinned(root, vd["source"]["rv_manifests"]["files"][s]))
        pairing = check_pairing(sc, rv); check_allowlist(sc["videos"].tolist(), train); check_allowlist(rv["videos"].tolist(), train)
        state = torch.load(pinned(root, vd["source"]["weights"][s]), map_location="cpu", weights_only=True)
        base = tm.build_model(None, tm.initial_new_state(s)); base.load_state_dict(state, strict=True)
        aligned = aux_labels(sc["labels"], anchors); perm = permutations(s, *aligned.shape[:2]); shuffled = shuffled_labels(aligned, perm)
        if not np.array_equal(np.sort(aligned, -1), np.sort(shuffled, -1)):
            raise SystemExit(f"seed {s}: a shuffled window lost its label multiset")
        arrays = {"aligned": aligned, "perm": perm, "shuffled": shuffled, "anchors": np.asarray(anchors, np.int8)}
        sha = write_aux(mdir / f"AUX_s{s}.npz", arrays)
        summ = label_summary(aligned, shuffled, perm, sc["kind"], sc["s"], anchors)
        index["seeds"][str(s)] = {"file": f"AUX_s{s}.npz", "sha256": sha, "rv_sha256": checked[f"rv_s{s}"], "sc_sha256": checked[f"sc_s{s}"], "summary": summ}
        rec["seeds"][str(s)] = {"pairing": pairing, "recordings": int(len(sc["videos"])), "all_training_clips": True, "aux_sha256": sha,
                                "source_state_strict_load": True, "source_parameters": int(sum(p.numel() for p in base.parameters())),
                                "learned_marker_abs_max": float(state["marker_stem.weight"].abs().max()), "labels": summ}
        print(f"seed {s}: {pairing['windows']} windows paired; {summ['mixed_windows']} mixed, {summ['mixed_windows_with_changed_labels']} with changed shuffled labels; "
              f"repeated anchors {summ['repeated_anchor_share']:.4f}; AUX {sha[:12]}", flush=True)
    (mdir / "index.json").write_text(json.dumps(index, indent=1))
    rec["index_sha256"] = sha256_file(mdir / "index.json"); rec["pixel_digests"] = (
        "the RV manifests are pinned, so every window's pixel digest is the source's; training decodes every window and stops if its digest differs")
    rec["seconds"] = time.time() - t0
    ids = train | {session_of(v) for v in train}; assert_no_ids(rec, ids, "preparation")
    out = root / vd["outputs"]["root"]; res = root / vd["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    for p in (out / "preparation.json", res / "preparation.json"):
        p.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out / 'preparation.json'}")


if __name__ == "__main__":
    main()
