#!/usr/bin/env python
"""Single-frame controls for the 474-target context probe of the target-logit residual round (closing item 1 of the
lead's decision of 2026-10-06). No training and no tuning.

The probe (runs/b7_window_target/probe) keeps the m centre frames (m = 4, 8, 12 or 14 of 16) of each target's own
window and fills the other positions from a recording of another session, of the same view or of another view. Every
probe window holds the target sample at position 8 (offset 0). Two controls read only that frame:

- E1: the frozen E1 image encoder (fp32 export) with the frozen E1 probe, i.e. the untrained starting point of every arm;
- head: each R model's own trained frame head on its own encoder's embedding of the target frame (R with delta = 0).

Both are compared with each R model's final output on the probe windows (target-frame logits + correction), read as in
scripts/report_target_logit_residual.py. Per seed and centre share: accuracies, final - control with the session
bootstrap of the original probe (2,000 draws, seed 20261002), and the windows, source recordings and sessions that the
final output fixes or newly breaks relative to each control. The final-output accuracies must reproduce the round's
summary.json exactly, and the delta = 0 reading of every probe window must equal the head control.

  python scripts/probe_single_frame_controls.py --config configs/experiments/target_logit_residual_v1.yaml \
      --out docs/results/2026-10-06_target_logit_residual/probe_single_frame_controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.b7_window_target import session_ci  # noqa: E402
from scripts.report_center_supervision import extract  # noqa: E402
from scripts.report_target_logit_residual import Reader, load_new, verify_new  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402
from scripts.train_target_logit_residual import NS, build, prepare_inits  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402

TARGET_POS = 8
SHARES = (4, 8, 12, 14)
DRAWS, BOOT_SEED = 2000, 20261002
PROBE_DIR = "runs/b7_window_target/probe"
ROUND_SUMMARY = "docs/results/2026-10-06_target_logit_residual/summary.json"


# ----------------------------------------------------------------------------------------------- pure helpers
def load_probe(probe_dir: Path) -> tuple[dict, pd.DataFrame]:
    spec = json.loads((probe_dir / "windows.json").read_text())["windows"]
    meta = pd.read_csv(probe_dir / "meta.csv").dropna(subset=["id"]).reset_index(drop=True)
    return {k: [(str(v), int(f), float(g)) for v, f, g in w] for k, w in spec.items()}, meta


def target_refs(spec: dict, meta: pd.DataFrame) -> dict:
    """target -> the (video, frame, gamma) at window position 8; every window of a target must agree on it."""
    out = {}
    for k, t in zip(meta["id"], meta["target"]):
        w = spec[k]
        if len(w) != 16:
            raise ValueError(f"{k}: {len(w)} references, expected 16")
        ref = w[TARGET_POS]
        if ref[0] != t:
            raise ValueError(f"{k}: position {TARGET_POS} is not from the target recording")
        if out.setdefault(t, ref) != ref:
            raise ValueError(f"{t}: windows disagree on the target frame")
    return out


def block_key(condition: str, m) -> str:
    return "own window, 16 of 16" if condition == "own" else f"{condition}, {int(m)} of 16"


def changes(df: pd.DataFrame, control: str) -> dict:
    """Windows, source recordings (targets) and sessions the final output fixes or newly breaks against a control.
    For new errors, also how many name the context recording's view (only meaningful in other-view windows)."""
    out = {}
    for name, mask in (("fixed", df["final"] & ~df[control]), ("new", ~df["final"] & df[control])):
        x = df[mask]
        out[name] = {"windows": int(len(x)), "recordings": int(x["target"].nunique()), "sessions": int(x["session"].nunique())}
        if name == "new" and "names_context" in x:
            out[name]["names_context"] = int(x["names_context"].sum())
    return out


def compare_block(df: pd.DataFrame, controls: tuple[str, ...], draws: int = DRAWS, seed: int = BOOT_SEED) -> dict:
    res = {"windows": int(len(df)), "targets": int(df["target"].nunique()), "final_accuracy": float(df["final"].mean())}
    for c in controls:
        d = df.assign(d=df["final"].astype(float) - df[c].astype(float))
        res[c] = {"accuracy": float(df[c].mean()), "final_minus_control": float(d["d"].mean()),
                  "ci95": session_ci(d, "d", draws, seed), "changes": changes(df, c)}
    return res


def analyse(meta: pd.DataFrame, final_correct: np.ndarray, names_context: np.ndarray, e1: dict, head: dict) -> dict:
    for name, control in (("E1", e1), ("head", head)):
        if not meta["target"].isin(list(control)).all():
            raise ValueError(f"the {name} control misses probe targets")
    df = meta.assign(final=final_correct.astype(bool), names_context=names_context.astype(bool) & (meta["condition"] == "other view").to_numpy(),
                     e1=meta["target"].map(e1).astype(bool), head=meta["target"].map(head).astype(bool))
    out = {}
    for (cond, m), x in df.groupby(["condition", df["m"].fillna(16)], sort=True):
        out[block_key(cond, m)] = compare_block(x, ("head", "e1"))
    return out


def reproduces(meta: pd.DataFrame, final_correct: np.ndarray, stored: dict) -> float:
    """Largest gap between the recomputed and the stored same-view / other-view accuracies of the round's probe. As in
    report_target_logit_residual.probe, other-view windows count only for targets that have a same-view window."""
    df = meta.assign(final=final_correct.astype(bool)); gap = 0.0
    for m in SHARES:
        paired = set(df[(df["condition"] == "same view") & (df["m"] == m)]["target"])
        for cond, key in (("same view", "same_view_accuracy"), ("other view", "other_view_accuracy")):
            x = df[(df["condition"] == cond) & (df["m"] == m) & df["target"].isin(paired)]["final"]
            diff = abs(float(x.mean()) - float(stored[str(m)][key])) if len(x) else float("nan")
            if not np.isfinite(diff):
                raise ValueError(f"no comparable {cond} accuracy at {m} of 16")
            gap = max(gap, diff)
    return gap


# ----------------------------------------------------------------------------------------------- model reading
@torch.no_grad()
def probe_logits(reader: Reader, feats: np.ndarray, at: dict, spec: dict, ids: list, delta_zero: bool) -> np.ndarray:
    idx = np.array([[at[r] for r in spec[k]] for k in ids]); rel = np.tile(np.arange(16) - TARGET_POS, (len(ids), 1))
    return np.concatenate([reader.logits(feats[idx[i:i + 1024]], rel[i:i + 1024], delta_zero).float().cpu().numpy()
                           for i in range(0, len(ids), 1024)])


@torch.no_grad()
def head_logits(linear: torch.nn.Linear, feats: np.ndarray, device) -> np.ndarray:
    return linear(torch.from_numpy(feats).to(device)).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", default=f"runs/{NS}/status.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--details", default=f"runs/{NS}/report/probe_single_frame_controls.json"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs, rt = cfg["center_supervision"], cfg["target_logit_residual"]; paths = load_paths(); root = paths.repo_root
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    runs = verify_new(cfg, rt, cs, paths, root / a.status); stored = json.loads((root / ROUND_SUMMARY).read_text())["models"]
    spec, meta = load_probe(root / PROBE_DIR); tref = target_refs(spec, meta); ids = list(meta["id"])
    if meta["target"].nunique() != 474:
        raise SystemExit(f"expected the 474 probe targets, found {meta['target'].nunique()}")
    triples = sorted({r for k in ids for r in spec[k]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    targets = sorted(tref); target_triples = [tref[t] for t in targets]
    if int(meta.groupby("target")["label"].nunique().max()) != 1:
        raise SystemExit("a target carries more than one label")
    labels = meta.groupby("target")["label"].first(); label = np.array([int(labels[t]) for t in targets])
    row_target = np.searchsorted(np.array(targets), meta["target"].to_numpy())
    if not np.array_equal(np.array(targets)[row_target], meta["target"].to_numpy()):
        raise SystemExit("probe rows do not map onto the targets")

    # E1: the frozen export with the frozen probe (every arm's untrained start), on the target frames only
    pe = cs["inputs"]["probe"]
    if sha256_file(root / pe["path"]) != pe["sha256"]:
        raise SystemExit("the E1 probe differs from its recorded sha256")
    probe_file = torch.load(root / pe["path"], map_location="cpu", weights_only=True)
    if list(probe_file["classes"]) != list(cs["classes"]):
        raise SystemExit("the E1 probe's class order differs from the round's")
    probe_state = probe_file["state_dict"]
    init = root / "runs" / NS / "inits" / prepare_inits(cfg, paths)["0"]["file"]
    e1_model, _ = build(cs, paths, init, "R", device); e1_model.eval()
    fh = e1_model.head.frame_head
    if not (torch.equal(fh.weight.detach().cpu(), probe_state["weight"]) and torch.equal(fh.bias.detach().cpu(), probe_state["bias"])):
        raise SystemExit("the untrained frame head is not the E1 probe")
    e1_pred = head_logits(fh, extract(e1_model, target_triples, fp, device, a.workers), device).argmax(1)
    e1 = dict(zip(targets, (e1_pred == label).tolist()))
    del e1_model; torch.cuda.empty_cache()
    print(f"E1 target frame only: {np.mean(list(e1.values())):.4f} of {len(e1)} targets", flush=True)

    res = {"targets": len(targets), "sessions": int(meta["session"].nunique()),
           "targets_per_half": {h: int(x["target"].nunique()) for h, x in meta.groupby("half")},
           "e1_accuracy": float(np.mean(list(e1.values()))), "seeds": {}, "checks": {}}
    details = {"e1": {t: bool(v) for t, v in e1.items()}, "seeds": {}}
    for seed in rt["seeds"]:
        key = f"R_s{seed}"; model = load_new(cs, paths, runs[key], device); reader = Reader(model, "R", device)
        feats = extract(model, triples, fp, device, a.workers)
        final = probe_logits(reader, feats, at, spec, ids, False).argmax(1)
        dz = probe_logits(reader, feats, at, spec, ids, True)
        hl = head_logits(model.head.frame_head, feats[[at[r] for r in target_triples]], device)
        head_pred = dict(zip(targets, hl.argmax(1).tolist())); head = {t: head_pred[t] == int(labels[t]) for t in targets}
        hrow = hl[row_target]   # the head control's logits repeated for every probe window of the target
        dz_gap = float(np.abs(dz - hrow).max())
        if dz_gap > 1e-4 or not np.array_equal(dz.argmax(1), hrow.argmax(1)):
            raise SystemExit(f"{key}: delta = 0 on the probe windows differs from the head control ({dz_gap:.2e})")
        correct = final == meta["label"].to_numpy(); names_context = final == meta["context_label"].to_numpy()
        gap = reproduces(meta, correct, stored[key]["probe"])
        if gap > 1e-12:
            raise SystemExit(f"{key}: the final output does not reproduce the round's probe (gap {gap:.2e})")
        res["seeds"][str(seed)] = {"head_accuracy": float(np.mean(list(head.values()))), "blocks": analyse(meta, correct, names_context, e1, head)}
        res["checks"][key] = {"delta_zero_vs_head_max_abs_logit": dz_gap, "reproduces_round_probe_max_gap": gap}
        details["seeds"][key] = {"head": {t: bool(v) for t, v in head.items()},
                                 "final_wrong_windows": [k for k, c in zip(ids, correct) if not c]}
        b = res["seeds"][str(seed)]["blocks"][block_key("other view", SHARES[0])]
        print(f"{key}: head {res['seeds'][str(seed)]['head_accuracy']:.4f} | {SHARES[0]}/16 other view final {b['final_accuracy']:.4f} "
              f"(final - head {b['head']['final_minus_control']:+.4f}, final - E1 {b['e1']['final_minus_control']:+.4f})", flush=True)
        del model, feats; torch.cuda.empty_cache()
    out = root / a.out; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    det = root / a.details; det.parent.mkdir(parents=True, exist_ok=True); det.write_text(json.dumps({**res, "details": details}, indent=1))
    print(f"wrote {out} and {det}")


if __name__ == "__main__":
    main()
