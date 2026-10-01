#!/usr/bin/env python
"""Direct transfer to the frozen EchoNet-Pediatric test folds (task 2 of the 2026-10-01 round).

Nothing is tuned on Pediatric. B0 weights, decoder settings (EV9V config plus the audited B3) and the MS-TCN v2 model
are the EV9V-frozen ones named in the protocol, whose file digests are checked first. Conditions: true order and the
ten task-1 random seeds, applied inside every clip (natives) or every fragment (exam / same banks) with the same
function as on EV9V. Writes under --out (server only, since per-video rows carry file-name ids):
- classification.json: true-order transfer per sample and per clip (five outputs kept), test and calibration folds;
- natives.csv.gz: per test clip, condition, seed and method: deviations from the clip label, error runs, cuts;
- fragments.csv.gz: the ladder's per-fragment counts at the 0.5 s tolerance for both stitch banks;
- streams.csv.gz: per stream: class-aware segment counts at IoU 0.5 and predicted changes away from every join;
- boundary.csv: pooled boundary precision / recall / F1 per bank, condition, seed, method and tolerance;
- timing.json: B0 extraction seconds and per-method decoding seconds on the true-order test inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.compose.stream_assembler import Stream  # noqa: E402
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.evaluate.ladder import NativeCine, constructed_metrics  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.realvideo.pediatric import PREPROCESS_VERSION, PROTOCOL_VERSION, parse_name, recipe_group  # noqa: E402
from echo_routing.realvideo.permutations import error_runs, lag1_autocorrelation, order_index  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments  # noqa: E402
from scripts.pediatric_extract import verify_protocol  # noqa: E402
from scripts.pediatric_freeze import B0_HASH, EV9V_CONFIG, HZ, MSTCN_MODEL, PERMUTATION_SEEDS  # noqa: E402

METHODS = ("b0", "b1", "b2", "b3", "b3a", "b4")
CONDITIONS = [("true", 0)] + [("random", s) for s in PERMUTATION_SEEDS]
BANKS = ("exam", "same")
FEATURES_DIM = 2048   # the official MS-TCN input width; 512-d B0 features are zero-padded as in training
PRESENT = (1, 2)      # PSAX, A4C: the classes that occur in Pediatric
_STATE: dict = {}     # filled before forking the condition workers


def tag(kind: str, seed: int) -> str:
    return "true" if kind == "true" else f"{kind}_s{seed}"


def group_of(video_id: str) -> str:
    return parse_name(video_id + ".avi")[0]


def load_clips(feat_dir: Path, ids) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for v in ids:
        with np.load(feat_dir / f"{v}.npz") as z:
            out[v] = (z["prob"].astype(float), z["feat"].astype(np.float32))
    return out


def assemble(clips: dict, recipe, kind: str, seed: int) -> Stream:
    probs, feats, labels, joins, pos = [], [], [], [], 0
    for i, fr in enumerate(recipe.fragments):
        p, f = clips[fr.video_id]; p, f = p[fr.start:fr.end], f[fr.start:fr.end]
        o, _ = order_index(len(p), kind, seed, recipe.recipe_id, i)
        probs.append(p[o]); feats.append(f[o]); labels.append(np.full(len(p), fr.label, dtype=int))
        if i > 0:
            joins.append(pos)
        pos += len(p)
    lab = np.concatenate(labels); j = np.array(joins, dtype=int)
    sem = np.array([x for x in j if lab[x] != lab[x - 1]], dtype=int)
    return Stream(recipe, np.concatenate(probs), np.concatenate(feats), lab, j, sem)


def condition_inputs(clips: dict, test: pd.DataFrame, recipes: dict, kind: str, seed: int):
    natives = []
    for r in test.itertuples(index=False):
        p, f = clips[r.video_id]
        o, _ = order_index(len(p), kind, seed, r.video_id, 0)
        natives.append(NativeCine(r.video_id, int(r.label), p[o], f[o]))
    return natives, {b: [assemble(clips, rec, kind, seed) for rec in recipes[b]] for b in BANKS}


def load_mstcn(repo_root: Path, n_classes: int, device):
    sys.path.insert(0, str(repo_root / "third_party" / "ms-tcn"))
    from model import MultiStageModel  # official class (the seed patch touches main.py only)
    model = MultiStageModel(4, 10, 64, FEATURES_DIM, n_classes).to(device)
    model.load_state_dict(torch.load(repo_root / MSTCN_MODEL, map_location=device)); model.eval()
    return model


@torch.no_grad()
def predict_mstcn(model, items: list[tuple[str, np.ndarray]], out_dir: Path, n_classes: int, device) -> float:
    """Softmax of the last stage per item, saved as <out_dir>/<id>.npz; returns forward-pass seconds."""
    out_dir.mkdir(parents=True, exist_ok=True); spent = 0.0
    for sid, feat in items:
        x = np.zeros((FEATURES_DIM, feat.shape[0]), dtype=np.float32); x[: feat.shape[1]] = feat.T
        t0 = time.perf_counter()
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        prob = torch.softmax(model(xt, torch.ones(1, n_classes, xt.shape[2], device=device))[-1], dim=1)[0].T.cpu().numpy()
        spent += time.perf_counter() - t0
        np.savez_compressed(out_dir / f"{sid}.npz", prob=prob.astype(np.float32))
    return spent


def two_class_summary(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return {"n": int(y.size), "accuracy": float((y == yhat).mean()),
            "macro_f1_present": float(f1_score(y, yhat, labels=list(PRESENT), average="macro", zero_division=0)),
            "recall": {name: float((yhat[y == c] == c).mean()) for name, c in (("PSAX", 1), ("A4C", 2))},
            "confusion_rows_PSAX_A4C_cols_5": [[int(((y == c) & (yhat == k)).sum()) for k in range(5)] for c in PRESENT]}


def classification(clips: dict, videos: pd.DataFrame) -> dict:
    out = {}
    for role, part in videos.groupby("role"):
        y, yhat, cy, chat = [], [], [], []
        for r in part.itertuples(index=False):
            a = clips[r.video_id][0].argmax(1)
            y.append(np.full(a.size, r.label)); yhat.append(a); cy.append(int(r.label)); chat.append(int(np.bincount(a, minlength=5).argmax()))
        out[role] = {"per_sample": two_class_summary(np.concatenate(y), np.concatenate(yhat)), "per_clip": two_class_summary(cy, chat)}
    rec = out["test"]["per_clip"]["recall"]
    out["transfer_adequate"] = bool(min(rec.values()) >= 0.80)
    out["rule"] = "B0 clip majority-vote recall >= 0.80 for both A4C and PSAX on the test folds (frozen in the protocol)"
    return out


def native_rows(natives, cfg: dict, method: str, cond: str, seed: int) -> list[dict]:
    rows = []
    for c in natives:
        raw = c.prob.argmax(1); rr_raw = error_runs(raw, np.full(raw.size, c.label))
        lab = decode_with_prob(method, c.prob, cfg, c.feat, c.video_id)[0]
        rr = error_runs(lab, np.full(lab.size, c.label))
        rows.append({"condition": cond, "seed": seed, "method": method, "video_id": c.video_id, "group": group_of(c.video_id),
                     "label": c.label, "n": int(lab.size), "dev_raw": int(rr_raw.sum()), "runs_raw": int(rr_raw.size),
                     "dev_raw_in_runs_ge5": int(rr_raw[rr_raw >= 5].sum()), "lag1": lag1_autocorrelation(c.prob),
                     "dev_dec": int(rr.sum()), "runs_dec": int(rr.size), "max_run_dec": int(rr.max()) if rr.size else 0,
                     "cuts": int(boundaries_from_labels(lab).size), "majority": int(np.bincount(lab, minlength=5).argmax())})
    return rows


def stream_rows(streams, cfg: dict, method: str, bank: str, cond: str, seed: int, tol: float) -> list[dict]:
    rows = []
    for s in streams:
        lab = decode_with_prob(method, s.prob, cfg, s.feat, s.recipe.recipe_id)[0]
        pred_b = boundaries_from_labels(lab); true_seg = labels_to_segments(s.labels); pred_seg = labels_to_segments(lab)
        far = int(sum(1 for x in pred_b if not np.any(np.abs(s.joins - x) <= tol)))
        rows.append({"condition": cond, "seed": seed, "method": method, "bank": bank, "stream": s.recipe.recipe_id,
                     "group": recipe_group(s.recipe), "n": int(lab.size), "joins": int(s.joins.size),
                     "diff_joins": int(s.semantic_boundaries.size), "seg_true": len(true_seg), "seg_pred": len(pred_seg),
                     "seg_matched50": int(segment_f1(pred_seg, true_seg, 0.5)["matched"]), "pred_changes": int(pred_b.size),
                     "changes_away_from_joins": far, "frame_acc": float((lab == s.labels).mean())})
    return rows


def run_condition(task: tuple[str, int]):
    kind, seed = task; st = _STATE; cond = kind
    cfg = copy.deepcopy(st["cfg"]); cfg["mstcn_official"] = {"pred_dir": str(st["pred_root"] / tag(kind, seed))}
    natives, streams = condition_inputs(st["clips"], st["test"], st["recipes"], kind, seed)
    tol = cfg["policy"]["boundary_tolerances_s"][1] * HZ
    nat, frag, strm, bnd = [], [], [], []
    for m in METHODS:
        nat.extend(native_rows(natives, cfg, m, cond, seed))
        for b in BANKS:
            res = constructed_metrics(m, streams[b], HZ, cfg, tau=None, taus={})
            frag.extend({"condition": cond, "seed": seed, "method": m, "bank": b, "group": group_of(f["video_id"]), **f}
                        for f in res["_fragments"])
            for t, v in res["boundary"].items():
                bnd.append({"condition": cond, "seed": seed, "method": m, "bank": b, "tol_s": float(t),
                            **{k: v[k] for k in ("precision", "recall", "f1", "n_true", "n_pred", "matched")}})
            strm.extend(stream_rows(streams[b], cfg, m, b, cond, seed, tol))
    print(f"done {tag(kind, seed)}", flush=True)
    return nat, frag, strm, bnd


def decode_timing(natives, streams: dict, cfg: dict, mstcn_seconds: float) -> dict:
    n_samples = sum(c.prob.shape[0] for c in natives) + sum(s.prob.shape[0] for b in BANKS for s in streams[b])
    out = {"samples": int(n_samples)}
    for m in METHODS:
        t0 = time.perf_counter()
        for c in natives:
            decode_with_prob(m, c.prob, cfg, c.feat, c.video_id)
        for b in BANKS:
            for s in streams[b]:
                decode_with_prob(m, s.prob, cfg, s.feat, s.recipe.recipe_id)
        dt = time.perf_counter() - t0 + (mstcn_seconds if m == "b4" else 0.0)
        out[m] = {"seconds": round(dt, 3), "ms_per_1000_samples": round(1000 * dt / n_samples * 1000, 3),
                  "note": "MS-TCN forward passes included" if m == "b4" else "CPU decoding of B0 probabilities"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="EchoNet-Pediatric direct transfer")
    ap.add_argument("--protocol-dir", default=f"data/manifests/pediatric/{PROTOCOL_VERSION}")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-groups", type=int, default=None, help="smoke runs: first N test groups only")
    args = ap.parse_args(); paths = load_paths(); proto_dir = Path(args.protocol_dir); out = Path(args.out)
    proto = verify_protocol(proto_dir)
    import hashlib
    if hashlib.sha256((paths.repo_root / MSTCN_MODEL).read_bytes()).hexdigest() != proto["models"]["b4"]["sha256"]:
        raise SystemExit("MS-TCN model differs from the protocol")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(EV9V_CONFIG)); cfg["decode"].setdefault("b3a", dict(AUDITED_B3))
    if cfg["decode"]["b3a"] != proto["models"]["decoders"]["b3a"] or any(cfg["decode"][k] != proto["models"]["decoders"][k] for k in ("b1", "b2", "b3")):
        raise SystemExit("decoder settings differ from the protocol")
    videos = pd.read_csv(proto_dir / "videos.csv", dtype={"clip_no": str})
    feat_dir = paths.cache_root / "pediatric" / PREPROCESS_VERSION / "features" / B0_HASH
    clips = load_clips(feat_dir, videos["video_id"])
    test = videos[videos["role"] == "test"].reset_index(drop=True)
    recipes = {b: [recipe_from_dict(d) for d in json.loads((proto_dir / f"recipes_test_{b}.json").read_text())] for b in BANKS}
    if args.limit_groups:
        keep = set(test["group"].drop_duplicates().head(args.limit_groups))
        test = test[test["group"].isin(keep)].reset_index(drop=True)
        recipes = {b: [r for r in v if recipe_group(r) in keep] for b, v in recipes.items()}
    (out / "classification.json").write_text(json.dumps(classification(clips, videos), indent=1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mstcn(paths.repo_root, 5, device)
    pred_root = paths.cache_root / "pediatric" / PREPROCESS_VERSION / "b4" / B0_HASH / "mstcn_v2_frameorder"
    mstcn_seconds = {}
    for kind, seed in CONDITIONS:
        natives, streams = condition_inputs(clips, test, recipes, kind, seed)
        items = [(c.video_id, c.feat) for c in natives] + [(s.recipe.recipe_id, s.feat) for b in BANKS for s in streams[b]]
        mstcn_seconds[tag(kind, seed)] = predict_mstcn(model, items, pred_root / tag(kind, seed), 5, device)
        if kind == "true":
            true_inputs = (natives, streams)
    print("MS-TCN predictions written", flush=True)
    timing = {"b0_extraction": json.loads((feat_dir / "meta.json").read_text()).get("extraction_seconds"),
              "decoding_true_order_test": decode_timing(*true_inputs, {**cfg, "mstcn_official": {"pred_dir": str(pred_root / "true")}},
                                                        mstcn_seconds["true"]),
              "mstcn_forward_seconds": mstcn_seconds, "device": str(device)}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    _STATE.update(cfg=cfg, clips=clips, test=test, recipes=recipes, pred_root=pred_root)
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(run_condition, CONDITIONS, chunksize=1)
    pd.DataFrame([r for p in parts for r in p[0]]).to_csv(out / "natives.csv.gz", index=False)
    pd.DataFrame([r for p in parts for r in p[1]]).to_csv(out / "fragments.csv.gz", index=False)
    pd.DataFrame([r for p in parts for r in p[2]]).to_csv(out / "streams.csv.gz", index=False)
    pd.DataFrame([r for p in parts for r in p[3]]).to_csv(out / "boundary.csv", index=False)
    (out / "run.json").write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "videos_digest": proto["videos_digest"],
                                              "banks": proto["banks"], "test_videos": len(test), "conditions": [tag(*c) for c in CONDITIONS],
                                              "methods": METHODS, "limit_groups": args.limit_groups}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
