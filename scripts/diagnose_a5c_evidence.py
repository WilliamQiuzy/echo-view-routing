#!/usr/bin/env python
"""A32 (configs/experiments/a5c_source_external_evidence_v1.yaml): A5C error sources and frozen external recognition evidence.

  --stage plans      private plans of both A31 pools for the external runners, frame-order identity (CPU)
  --stage echoprime  EchoPrime's released view classifier on every unique pool reference, batched; parity against the
                     original per-stream runner path on two streams (GPU)
  --stage evc        the original scripts/run_windowed_echoviewclip.py in envs/echoviewclip on plan shards (GPU)
  --stage sets       A31 sets S1-S4, source concentration, label identity and window support (CPU)
  --stage readout    the fixed replacement rule for both external models on every A08 Base run; runs and full sequences (CPU)
  --stage package    the private review package, server only (CPU)
  --stage extras     per set: majority-source and true-class-source recordings across seeds; external S1 fixes by run type (CPU)
Training data only: A31's two fold pools (A15 large-fit sources). Nothing here trains, fits or searches.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.run_windowed_echoviewclip as evr  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.ingest.frames import apply_gamma, list_frame_paths  # noqa: E402
from echo_routing.temporal.windowed import ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, map_probs  # noqa: E402

A31_CONFIG = "configs/experiments/video_session_correction_v1.yaml"
SEEDS, FOLDS, ARMS = t31.SEEDS, t31.FOLDS, t31.ARMS
PLAX, PSAX, A4C, A5C, SC4C = range(5)
MODELS = ("EchoViewCLIP", "EchoPrime")
EVC = {"python": "envs/echoviewclip/bin/python", "runner": "scripts/run_windowed_echoviewclip.py", "config": "third_party/adapters/echoviewclip/ev9v_stage1.yaml",
       "ckpt": "models_frozen/echoviewclip_stage1/v1/best.pth", "sha256": "a74c7b233e79c67ca1d9b145f938283ed945378b4ccf3765d2bcc104f8dc0dfc", "labels": "data/echoviewclip/labels_ev9v.csv"}
EP = {"weights": "models_frozen/echoprime_view_classifier/release_v1.0.0/view_classifier.pt", "sha256": "046b76685de5295d151f74725d69be758c53aac3a028132e3f1f160ba5143954",
      "mean": [29.110628, 28.076836, 29.096405], "std": [47.989223, 46.456997, 47.20083]}
SHARDS = 3
now, write_json, ledger_append = t31.now, t31.write_json, t31.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["a5c_source_external_evidence"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG)


def private_ids(ctx: Ctx) -> set[str]:
    pools = [t31.load_pool(ctx.a31, f) for f in FOLDS]; rec = {r[0] for p in pools for q in p for r in q["refs"]}
    return rec | {session_of(v) for v in rec} | {q["id"] for p in pools for q in p if q["kind"] == "native"}


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    flat = json.loads(json.dumps(rec, default=float)); assert_no_ids(flat, private_ids(ctx), name)
    for p in (ctx.out / name, ctx.results / name):
        p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, job, kind, started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- plans (CPU)
def plans(ctx: Ctx) -> str:
    """Plans hold only stream ids, lengths and (recording, frame, gamma) references: no labels reach the runners."""
    root = load_paths().ev9v_images; rec = {"pools": {}}
    for f in FOLDS:
        seqs = t31.load_pool(ctx.a31, f); d = ctx.out / "plans" / f"pool_{f}"; d.mkdir(parents=True, exist_ok=True)
        streams = {q["id"]: {"n": len(q["labels"]), "refs": [[str(r[0]), int(r[1]), float(r[2])] for r in q["refs"]]} for q in seqs}
        (d / "plans.json").write_text(json.dumps({"frames_root": str(root), "hz": 10, "streams": streams}))
        ids = list(streams)
        for k in range(SHARDS):
            (d / f"shard_{k}.json").write_text(json.dumps({"frames_root": str(root), "hz": 10, "streams": {i: streams[i] for i in ids[k::SHARDS]}}))
        vids = sorted({r[0] for q in seqs for r in q["refs"]}); same = sum(evr.frame_paths(root / v) == list_frame_paths(root / v) for v in vids); fp = t31.t28.frame_paths(seqs)
        rec["pools"][str(f)] = {"streams": len(ids), "samples": int(sum(s["n"] for s in streams.values())), "recordings": len(vids), "runner_frame_order_equals_A08": int(same),
                                "runner_frame_dirs_equal_A08": int(sum(Path(fp.dirs[v]).resolve() == (root / v).resolve() for v in vids)),
                                "plans_sha256": sha256_file(d / "plans.json")}
        if same != len(vids) or rec["pools"][str(f)]["runner_frame_dirs_equal_A08"] != len(vids):
            raise SystemExit("the external runners' frames differ from A08's")
    write_json(ctx.out / "plans" / "record.json", rec); print(json.dumps(rec, indent=1)); return "plans of both pools"


# ----------------------------------------------------------------------------------------------- EchoPrime (GPU)
def echoprime_module():
    repo = ROOT / "third_party" / "EchoPrime"; cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import utils  # noqa: E402  (official: crop_and_scale, COARSE_VIEWS)
    os.chdir(cwd)
    if list(utils.COARSE_VIEWS) != ECHOPRIME_VIEWS:
        raise SystemExit("EchoPrime's view order differs from the mapping's")
    return utils


class EPFrames(Dataset):
    """One reference -> EchoPrime's input, by the original runner's code path (gamma only when != 1, crop_and_scale, mean/std)."""

    def __init__(self, refs: list[tuple], root: Path, crop) -> None:
        self.refs, self.root, self.crop, self.paths = refs, root, crop, {}
        self.mean, self.std = torch.tensor(EP["mean"]).reshape(3, 1, 1), torch.tensor(EP["std"]).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> torch.Tensor:
        v, fi, g = self.refs[i]
        if v not in self.paths:
            self.paths[v] = list_frame_paths(self.root / v)
        p = self.paths[v]
        with Image.open(p[min(int(fi), len(p) - 1)]) as im:
            im = im.convert("RGB"); im = apply_gamma(im, g) if g != 1.0 else im
            x = torch.as_tensor(self.crop(np.asarray(im)), dtype=torch.float).permute(2, 0, 1)
        return (x - self.mean) / self.std


def echoprime(ctx: Ctx, device, workers: int) -> str:
    import torchvision  # noqa: E402
    utils = echoprime_module(); w = ROOT / EP["weights"]
    if sha256_file(w) != EP["sha256"]:
        raise SystemExit("EchoPrime's weights differ from the pin")
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11); vc.load_state_dict(torch.load(w, map_location="cpu")); vc.to(device).eval()
    pools = {f: t31.load_pool(ctx.a31, f) for f in FOLDS}; refs = sorted({(str(r[0]), int(r[1]), float(r[2])) for f in FOLDS for q in pools[f] for r in q["refs"]}); root = load_paths().ev9v_images
    raw, t0 = [], time.time()
    with torch.no_grad():
        for k, x in enumerate(DataLoader(EPFrames(refs, root, utils.crop_and_scale), batch_size=256, num_workers=workers)):
            raw.append(torch.softmax(vc(x.to(device)).float(), dim=1).cpu().numpy())
            if k % 50 == 0:
                print(f"  {min(len(refs), (k + 1) * 256)}/{len(refs)} references, {time.time() - t0:.0f} s", flush=True)
    raw = np.concatenate(raw); fam = map_probs(raw, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5); at = {r: i for i, r in enumerate(refs)}; d = ctx.out / "echoprime"; d.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:
        idx = np.asarray([at[(str(r[0]), int(r[1]), float(r[2]))] for q in pools[f] for r in q["refs"]])
        np.savez(d / f"pool_{f}.npz", prob=fam[idx].astype(np.float32), raw=raw[idx].astype(np.float32), offsets=np.cumsum([0, *[len(q["labels"]) for q in pools[f]]]))
    # parity: the original runner's per-stream loop (64 frames per forward) on one constructed stream with a gamma edit and one native stream
    seqs = pools[0]; pick = [next(q for q in seqs if q["kind"] == "multi" and any(r[2] != 1.0 for r in q["refs"])), next(q for q in seqs if q["kind"] == "native")]; par = []
    ds = EPFrames([], root, utils.crop_and_scale)
    with torch.no_grad():
        for q in pick:
            xs = []
            for v, fi, g in q["refs"]:
                ds.refs = [(v, fi, g)]; xs.append(ds[0])
            ro = np.concatenate([torch.softmax(vc(torch.stack(xs[i:i + 64]).to(device)).float(), dim=1).cpu().numpy() for i in range(0, len(xs), 64)])
            rb = raw[[at[(str(r[0]), int(r[1]), float(r[2]))] for r in q["refs"]]]; par.append({"kind": q["kind"], "samples": len(xs), "max_abs": float(np.abs(ro - rb).max()), "argmax_equal": bool((ro.argmax(1) == rb.argmax(1)).all())})
    rec = {"unique_references": len(refs), "samples": {str(f): int(sum(len(q["labels"]) for q in pools[f])) for f in FOLDS}, "inference_seconds": time.time() - t0, "parity": par,
           "unmapped_mass": {"mean": float(1 - fam.sum(1).mean()), "p99": float(np.percentile(1 - fam.sum(1), 99))}, "tf32": {"cudnn": torch.backends.cudnn.allow_tf32, "matmul": torch.backends.cuda.matmul.allow_tf32}}
    write_json(d / "record.json", rec); print(json.dumps(rec, indent=1))
    if not all(p["max_abs"] <= 1e-4 and p["argmax_equal"] for p in par):
        raise SystemExit("batched EchoPrime outputs differ from the runner path")
    return f"{len(refs)} unique references"


# ----------------------------------------------------------------------------------------------- EchoViewCLIP (GPU, original runner)
def evc(ctx: Ctx, limit: int | None) -> str:
    if sha256_file(ROOT / EVC["ckpt"]) != EVC["sha256"]:
        raise SystemExit("EchoViewCLIP's weights differ from the pin")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}; procs = []; t0 = time.time()
    for f in FOLDS:
        out = ctx.out / "evc" / f"pool_{f}"; out.mkdir(parents=True, exist_ok=True)
        for k in range(SHARDS):
            cmd = [EVC["python"], EVC["runner"], "--plans", str(ctx.out / "plans" / f"pool_{f}" / f"shard_{k}.json"), "--ckpt", EVC["ckpt"], "--config", EVC["config"], "--out", str(out)] + (["--limit", str(limit)] if limit else [])
            log = open(ctx.out / "evc" / f"pool_{f}_shard_{k}{'_limit' if limit else ''}.log", "w"); procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print("runner PIDs:", [p.pid for p, _ in procs], flush=True)
    for p, log in procs:
        p.wait(); log.close()
    if any(p.returncode for p, _ in procs):
        raise SystemExit(f"a runner failed: {[p.returncode for p, _ in procs]}")
    done = {f: len(list((ctx.out / "evc" / f"pool_{f}").glob("*.npz"))) for f in FOLDS}; sec = time.time() - t0
    rec = {"limit_per_shard": limit, "outputs": done, "wall_seconds": sec, "time": now()}; print(json.dumps(rec), flush=True)
    write_json(ctx.out / "evc" / f"record{'_limit' if limit else ''}.json", rec); return f"{done} streams"


def evc_check(ctx: Ctx, seqs: list[dict], f: int) -> dict:
    """Shapes, finiteness, sums, nearest-centre structure and the class columns of every stream's output."""
    names = pd.read_csv(ROOT / EVC["labels"])["name"].tolist(); bad, steps = 0, []
    for q in seqs:
        with np.load(ctx.out / "evc" / f"pool_{f}" / f"{q['id']}.npz") as z:
            p, raw = z["prob"], z["raw"]
        n = len(q["labels"]); c = evr.window_centres(n, evr.STRIDE); near = np.abs(np.arange(n)[:, None] - c[None, :]).argmin(1)
        ok = p.shape == (n, 5) and raw.shape == (n, 9) and np.isfinite(raw).all() and np.allclose(raw.sum(1), 1, atol=1e-4) and np.allclose(p.sum(1), 1 - raw[:, 6], atol=1e-4)
        ok = ok and all(np.allclose(raw[near == j], raw[near == j][0]) for j in range(len(c)))
        bad += not ok; steps.append(len(c))
    return {"streams": len(seqs), "failed_checks": bad, "windows": int(sum(steps)), "label_rows_4_5": names[4:6], "raw9_4_5": evr.RAW9_ORDER[4:6],
            "family5_of_A4C_A5C": [evr.RAW9_TO_FAMILY5["A4C"], evr.RAW9_TO_FAMILY5["A5C"]]}


def external_probs(ctx: Ctx, seqs: list[dict], f: int) -> dict:
    with np.load(ctx.out / "echoprime" / f"pool_{f}.npz") as z:
        ep, off = z["prob"], z["offsets"]
    evp = {}
    for q in seqs:
        with np.load(ctx.out / "evc" / f"pool_{f}" / f"{q['id']}.npz") as z:
            evp[q["id"]] = z["prob"].astype(float)
    return {"EchoViewCLIP": evp, "EchoPrime": {q["id"]: ep[off[i]:off[i + 1]].astype(float) for i, q in enumerate(seqs)}}


# ----------------------------------------------------------------------------------------------- A31 runs and sets (CPU)
_LABELS: dict = {}


def clip_labels(ctx: Ctx) -> dict:
    """Training clips' family5 labels (A01's allowlisted training data), loaded once."""
    if not _LABELS:
        _LABELS.update(t31.vpr.training_data(ctx.a31.a15.a10.a07p).label)
    return _LABELS


def fold_data(ctx: Ctx, s: int, h: int) -> dict:
    """Held-out fold h under seed s: the pool, Base, its P runs and every A31 arm's held-out run prediction."""
    a = ctx.a31; seqs = t31.load_pool(a, h); R = t31.runs(a, s, h); pk = R["P"]
    with np.load(a.out / f"heldout_preds_s{s}_f{1 - h}.npz", allow_pickle=True) as z:
        hp = {k: z[k] for k in z.files}
    if not (np.array_equal(hp["labels"], pk["labels"]) and np.array_equal(hp["base"], pk["base"])):
        raise SystemExit("A31 held-out predictions do not line up with the runs")
    split_at = np.cumsum([len(q["labels"]) for q in seqs])[:-1]; base_preds = {q["id"]: b for q, b in zip(seqs, np.split(R["base_labels"], split_at))}
    src, tsrc, nsrc = [], [], []; lab = clip_labels(ctx)
    for i, x, y, t in zip(pk["seq"], pk["a"], pk["b"], pk["labels"]):
        c = Counter(r[0] for r in seqs[i]["refs"][x:y]); src.append({v: k / (y - x) for v, k in c.items()}); nsrc.append(len(c)); tsrc.append({v: w for v, w in src[-1].items() if lab[v] == t})
    kind = np.asarray([seqs[i]["kind"] for i in pk["seq"]])
    return {"seqs": seqs, "pk": pk, "arms": {x: hp[x] for x in ARMS}, "sessions": hp["sessions"], "base_preds": base_preds, "src": src, "true_src": tsrc, "n_sources": np.asarray(nsrc), "kind": kind,
            "length": pk["b"] - pk["a"], "pure": pk["purity"] >= 1.0, "key": [(int(i), int(x), int(y)) for i, x, y in zip(pk["seq"], pk["a"], pk["b"])]}


def set_masks(D: dict) -> dict:
    y, b, A = D["pk"]["labels"], D["pk"]["base"], D["arms"]; fe, te = A["F_E"] == A4C, A["T_E"] == A4C; ok5 = (y == A5C) & (b == A5C); s1 = (y == A5C) & (b == A4C); s4 = (y == A4C) & (b == A5C)
    return {"A5C_all": y == A5C, "A4C_all": y == A4C, "S1": s1, "S1_persistent": s1 & np.all([A[x] == A4C for x in ARMS], 0), "S1_fixed_by_no_arm": s1 & np.all([A[x] != A5C for x in ARMS], 0),
            "S2_F_E": ok5 & fe, "S2_T_E": ok5 & te, "S2_both": ok5 & fe & te, "S2": ok5 & (fe | te), "S3": ok5 & np.all([A[x] == A5C for x in ARMS], 0),
            "S4_F_E": s4 & (A["F_E"] == A4C), "S4_T_E": s4 & (A["T_E"] == A4C), "S4": s4 & ((A["F_E"] == A4C) | (A["T_E"] == A4C))}


def top_share(c: Counter, k: int, total: float):
    return float(sum(x for _, x in c.most_common(k)) / total) if total else None


def concentration(items: list[tuple[dict, int]], key: str, ref_w: Counter | None) -> dict:
    """Source weights under one attribution: "src" = every source by sample share (a multi-source run is split), "true_src" =
    only sources whose clip label is the run's reference class. Shares are of the set's total weight under that attribution."""
    w, uses = Counter(), Counter()
    for D, k in items:
        for v, x in D[key][k].items():
            w[v] += x; uses[v] += 1
    if not w:
        return {"recordings": 0}
    sw = Counter()
    for v, x in w.items():
        sw[session_of(v)] += x
    su = Counter(session_of(v) for v in w); tot = sum(w.values()); cum = np.cumsum(sorted(w.values(), reverse=True))
    out = {"weight": float(tot), "recordings": len(w), "sessions": len(su), "uses_per_recording_mean": float(sum(uses.values()) / len(w)), "uses_per_recording_max": int(max(uses.values())),
           "recordings_used_once": int(sum(u == 1 for u in uses.values())), "top_recording_share": {str(k): top_share(w, k, tot) for k in (1, 5, 10)}, "top_session_share": {str(k): top_share(sw, k, tot) for k in (1, 5, 10)},
           "top_session_share_by_unique_recordings": {str(k): top_share(su, k, len(w)) for k in (1, 5, 10)}, "recordings_for_half_the_weight": int(np.searchsorted(cum, tot / 2) + 1)}
    if ref_w is not None:
        rw = sum(ref_w.values()); top10 = [v for v, _ in w.most_common(10)]
        out["top10_recordings_share_in_reference_class_runs"] = float(sum(ref_w[v] for v in top10) / rw) if rw else None
    return out


def set_stats(items: list[tuple[dict, int]], ref_w: dict | None = None) -> dict:
    """items: (D, run index); ref_w: per attribution, the source weights of all runs of the set's true class."""
    n = len(items)
    if n == 0:
        return {"runs": 0}
    col = lambda key: np.asarray([D[key][k] for D, k in items])  # noqa: E731
    kind, L, pure, ns = col("kind"), col("length"), col("pure"), col("n_sources")
    return {"runs": n, "native": int((kind == "native").sum()), "constructed": int((kind == "multi").sum()), "pure": int(pure.sum()), "mixed": int((~pure).sum()), "short_le16": int((L <= 16).sum()),
            "long": int((L > 16).sum()), "single_source": int((ns == 1).sum()), "multi_source": int((ns > 1).sum()),
            **{f"by_{key}": concentration(items, key, None if ref_w is None else ref_w[key]) for key in ("src", "true_src")}}


def source_weight(items: list[tuple[dict, int]], key: str = "src") -> Counter:
    w = Counter()
    for D, k in items:
        for v, x in D[key][k].items():
            w[v] += x
    return w


def recording_error_profile(Ds: list[dict]) -> dict:
    """Per recording with true-A5C runs: the share of its A5C run weight that Base reads as A4C (are S1 errors recording-bound?)."""
    tot, wrong = Counter(), Counter()
    for D in Ds:
        y, b = D["pk"]["labels"], D["pk"]["base"]
        for k in np.flatnonzero(y == A5C):
            for v, x in D["true_src"][k].items():
                tot[v] += x; wrong[v] += x * (b[k] == A4C)
    r = np.asarray([wrong[v] / tot[v] for v in tot]); s1w = sum(wrong.values())
    return {"attribution": "true-class sources only", "A5C_recordings": len(tot), "with_any_A5C_read_as_A4C": int((r > 0).sum()), "at_least_half_read_as_A4C": int((r >= 0.5).sum()), "all_read_as_A4C": int((r >= 1 - 1e-9).sum()),
            "share_of_S1_weight_from_recordings_at_least_half_wrong": float(sum(wrong[v] for v in tot if wrong[v] / tot[v] >= 0.5) / s1w) if s1w else None,
            "share_of_S1_weight_from_recordings_all_wrong": float(sum(wrong[v] for v in tot if wrong[v] >= tot[v] - 1e-9) / s1w) if s1w else None}


def window_support(D: dict, W: dict, off: int) -> dict:
    """Per P run: weighted share of the 16 slots of its support windows outside the run / under another reference label, and centres inside."""
    pk = D["pk"]; o = pk["offsets"]; out = {"outside": np.zeros(len(pk["labels"])), "other_label": np.zeros(len(pk["labels"])), "centre_inside": np.zeros(len(pk["labels"])), "weights_sum_ok": True}
    for k in range(len(pk["labels"])):
        rows = pk["rows"][o[k]:o[k + 1]] - off; wt = pk["weights"][o[k]:o[k + 1]].astype(float); a, b = pk["a"][k], pk["b"][k]
        out["weights_sum_ok"] &= bool(abs(wt.sum() - (b - a)) < 1e-6 or abs(wt.sum() - 1) < 1e-6)
        S, L16, c = W["s"][rows], W["labels16"][rows], W["centre"][rows]; ww = wt / wt.sum()
        out["outside"][k] = float(ww @ ((S < a) | (S >= b)).mean(1)); out["other_label"][k] = float(ww @ (L16 != pk["labels"][k]).mean(1)); out["centre_inside"][k] = float(ww @ ((c >= a) & (c < b)))
    return out


def training_raw_codes(ctx: Ctx) -> dict:
    """Raw code -> label_index on the training rows only (other rows' labels are skipped unread, as in vsf.training_index)."""
    data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); path = data.dir / "index.csv"
    meta = pd.read_csv(path, usecols=["video_id", "split"], dtype=str); keep = set(np.flatnonzero(meta["split"].to_numpy() == "train").tolist())
    lab = pd.read_csv(path, usecols=["raw_label", "label_index"], skiprows=lambda i: i > 0 and (i - 1) not in keep)
    return {"pairs": {f"{r}->{i}": int(n) for (r, i), n in Counter(zip(lab["raw_label"], lab["label_index"].astype(int))).items()}, "columns": list(pd.read_csv(path, nrows=0).columns), "data": data}


def identity(ctx: Ctx) -> dict:
    lab = yaml.safe_load((ROOT / "configs/labels.yaml").read_text()); cls = lab["family5_classes"]; tr = training_raw_codes(ctx); data = tr.pop("data")
    fam = {code: (cls.index(v) if v else None) for code, v in lab["family5"].items()}
    mapping_ok = all(fam[k.split("->")[0]] == int(k.split("->")[1]) for k in tr["pairs"])
    nat_ok, multi_ok = 0, 0; nat_n, multi_n = 0, 0
    for f in FOLDS:
        for q in t31.load_pool(ctx.a31, f):
            if q["kind"] == "native":
                nat_n += 1; nat_ok += bool((np.asarray(q["labels"]) == data.label[q["id"]]).all())
            else:
                multi_n += 1; exp = np.concatenate([np.full(int(fr[4]) - int(fr[3]), data.label[fr[0]]) for fr in q["frags"]]); vids = [r[0] for r in q["refs"]]
                src = np.concatenate([[fr[0]] * (int(fr[4]) - int(fr[3])) for fr in q["frags"]])
                multi_ok += bool(len(exp) == len(q["labels"]) and (exp == np.asarray(q["labels"])).all() and list(src) == vids)
    return {"family5_v1": {"classes": cls, "A4C": cls.index("A4C"), "A5C": cls.index("A5C"), "raw_to_family5": fam}, "training_raw_code_pairs": tr["pairs"], "raw_codes_map_as_family5_v1": mapping_ok,
            "index_columns": tr["columns"], "structure_annotations": "none: the index has one clip-level code per recording and no LVOT or other structure fields",
            "A08_head_columns": "5 logits trained on label_index (family5_v1 order); A31 outputs have 5 columns",
            "native_labels_equal_clip_label": f"{nat_ok}/{nat_n}", "constructed_labels_equal_fragment_clip_labels_and_sources": f"{multi_ok}/{multi_n}",
            "external_maps": {"EchoViewCLIP": {"raw9": evr.RAW9_ORDER, "to_family5": evr.RAW9_TO_FAMILY5}, "EchoPrime": {"views": ECHOPRIME_VIEWS, "to_family5": ECHOPRIME_TO_FAMILY5}},
            "label_definitions": {k: lab["expansions"][src][k] for src in ("dataset_card", "stfm_readme") for k in ("A4C", "A5C")}}


def sets(ctx: Ctx) -> str:
    rec = {"identity": identity(ctx), "seeds": {}, "cross_seed": {}, "window_support": {}}; marks = {}
    for s in SEEDS:
        Ds = {h: fold_data(ctx, s, h) for h in FOLDS}; M = {h: set_masks(Ds[h]) for h in FOLDS}; marks[s] = (Ds, M)
        ref = {c: {key: source_weight([(Ds[h], k) for h in FOLDS for k in np.flatnonzero(M[h][f"{c}_all"])], key) for key in ("src", "true_src")} for c in ("A5C", "A4C")}
        r = {}
        for name in M[0]:
            rw = ref["A4C"] if name.startswith(("S4", "A4C")) else ref["A5C"]
            r[name] = {"both_folds": set_stats([(Ds[h], k) for h in FOLDS for k in np.flatnonzero(M[h][name])], rw), **{f"heldout_fold_{h}": set_stats([(Ds[h], k) for k in np.flatnonzero(M[h][name])]) for h in FOLDS}}
        rec["seeds"][str(s)] = {"sets": r, "recording_error_profile": recording_error_profile([Ds[h] for h in FOLDS])}
        ws = {}
        for h in FOLDS:
            O = t31.outputs(ctx.a31, s, h); W = t31.pool_W(Ds[h]["seqs"]); sup = window_support(Ds[h], W, int(O["offset"]))
            ws[f"heldout_fold_{h}"] = {"marker_at_centre": bool(all(W["s"][j][W["marker"][j]] == W["centre"][j] for j in range(len(W["seq"])))), "weights_sum_ok": sup["weights_sum_ok"],
                                       **{name: {k: float(sup[k][M[h][name]].mean()) if M[h][name].any() else None for k in ("outside", "other_label", "centre_inside")} for name in ("A5C_all", "S1", "S1_persistent", "S2", "S3", "S4")}}
        rec["window_support"][str(s)] = ws
    for name in ("S1", "S1_persistent", "S2", "S3", "S4"):
        c = {}
        for h in FOLDS:
            k0 = {marks[0][0][h]["key"][k] for k in np.flatnonzero(marks[0][1][h][name])}; k2 = {marks[2][0][h]["key"][k] for k in np.flatnonzero(marks[2][1][h][name])}
            r0 = set(source_weight([(marks[0][0][h], k) for k in np.flatnonzero(marks[0][1][h][name])], "true_src")); r2 = set(source_weight([(marks[2][0][h], k) for k in np.flatnonzero(marks[2][1][h][name])], "true_src"))
            c[f"heldout_fold_{h}"] = {"runs_s0": len(k0), "runs_s2": len(k2), "identical_runs_both_seeds": len(k0 & k2), "true_class_recordings_s0": len(r0), "true_class_recordings_s2": len(r2), "true_class_recordings_both_seeds": len(r0 & r2),
                                      "recording_jaccard": len(r0 & r2) / max(1, len(r0 | r2))}
        rec["cross_seed"][name] = c
    np.save(ctx.out / "sets_private.npy", np.array({s: {h: {"masks": marks[s][1][h], "key": marks[s][0][h]["key"]} for h in FOLDS} for s in SEEDS}, dtype=object), allow_pickle=True)
    publish(ctx, "sets.json", rec); print(json.dumps({s: {k: v["both_folds"].get("runs") for k, v in rec["seeds"][s]["sets"].items()} for s in rec["seeds"]}, indent=1)); return "sets and statistics"


# ----------------------------------------------------------------------------------------------- readout (CPU)
def run_counts(y: np.ndarray, b: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> dict:
    m = np.ones(len(y), bool) if mask is None else mask; wr = (b != y) & m; ok = (b == y) & m; fx, br = int((wr & (p == y)).sum()), int((ok & (p != y)).sum())
    return {"runs": int(m.sum()), "base_wrong": int(wr.sum()), "fixed": fx, "base_correct": int(ok.sum()), "broken": br, "net_fixed": fx - br, "accuracy_difference": (fx - br) / max(1, int(m.sum()))}


def change_table(y: np.ndarray, b: np.ndarray, p: np.ndarray) -> dict:
    """Fixed by (true <- Base), broken by (true -> new); all five classes."""
    fx = Counter(f"{int(t)}<-{int(c)}" for t, c, q in zip(y, b, p) if c != t and q == t); br = Counter(f"{int(t)}->{int(q)}" for t, c, q in zip(y, b, p) if c == t and q != t)
    return {"fixed": dict(sorted(fx.items())), "broken": dict(sorted(br.items())), "run_accuracy_by_class": {str(c): {"base": float((b[y == c] == c).mean()), "new": float((p[y == c] == c).mean()), "runs": int((y == c).sum())} for c in range(5)}}


def core_coverage(M: dict, y: np.ndarray, b: np.ndarray, p: np.ndarray) -> dict:
    cov = lambda m, tgt: {"runs": int(m.sum()), "to_target": int((m & (p == tgt)).sum())}  # noqa: E731
    ok5 = (y == A5C) & (b == A5C)
    return {"S1_fixed_to_A5C": cov(M["S1"], A5C), "S1_persistent_fixed_to_A5C": cov(M["S1_persistent"], A5C), "S2_kept_A5C": cov(M["S2"], A5C), "S3_kept_A5C": cov(M["S3"], A5C),
            "S4_fixed_to_A4C": cov(M["S4"], A4C), "correct_A5C_retained": cov(ok5, A5C), "A4C_read_as_A5C_fixed": cov((y == A4C) & (b == A5C), A4C), "A5C_read_as_A4C_fixed": cov((y == A5C) & (b == A4C), A5C)}


def error_overlap(y: np.ndarray, wrong: dict) -> dict:
    """Reference-assisted complementarity (diagnostic only): which runs each system gets wrong."""
    a = wrong["A08 Base"]; out = {}
    for k, w in wrong.items():
        if k != "A08 Base":
            out[k] = {"both_wrong": int((a & w).sum()), "A08_only_wrong": int((a & ~w).sum()), "other_only_wrong": int((~a & w).sum()), "either_wrong": int((a | w).sum()),
                      "A5C_both_wrong": int((a & w & (y == A5C)).sum()), "A5C_A08_only_wrong": int((a & ~w & (y == A5C)).sum()), "A5C_other_only_wrong": int((~a & w & (y == A5C)).sum())}
    e, p = wrong["EchoViewCLIP"], wrong["EchoPrime"]; out["EchoViewCLIP and EchoPrime"] = {"both_wrong": int((e & p).sum()), "all_three_wrong": int((e & p & a).sum()), "A5C_all_three_wrong": int((e & p & a & (y == A5C)).sum())}
    return out


def native_bootstrap(sess: np.ndarray, d: np.ndarray, s: int) -> list:
    us = np.unique(sess); gi = np.searchsorted(us, sess); rng = np.random.default_rng([20261024, 32, 77, s]); draws = rng.integers(0, len(us), (2000, len(us)))
    wts = np.stack([np.bincount(x, minlength=len(us)) for x in draws]).astype(float); per = np.bincount(gi, weights=d, minlength=len(us)); cnt = np.bincount(gi, minlength=len(us)).astype(float)
    return np.percentile((wts @ per) / (wts @ cnt), [2.5, 97.5]).tolist()


def native_clip_accuracy(seqs: list[dict], probs: dict) -> dict:
    """Each external model's own clip-level aggregation on native streams (mean probability, argmax): a training-data sanity check."""
    y = np.asarray([q["labels"][0] for q in seqs if q["kind"] == "native"]); p = np.asarray([probs[q["id"]].mean(0).argmax() for q in seqs if q["kind"] == "native"])
    return {str(c): {"clips": int((y == c).sum()), "correct": int((p[y == c] == c).sum())} for c in range(5)}


def readout(ctx: Ctx) -> str:
    a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"checks": {}, "runs": {}, "sequences": {}, "native_bootstrap": {}, "native_clip_accuracy": {}}; priv = {}
    ext = {h: external_probs(ctx, t31.load_pool(ctx.a31, h), h) for h in FOLDS}
    for h in FOLDS:
        seqs = t31.load_pool(ctx.a31, h); rec["checks"][f"EchoViewCLIP_pool_{h}"] = evc_check(ctx, seqs, h); rec["native_clip_accuracy"][f"pool_{h}"] = {m: native_clip_accuracy(seqs, ext[h][m]) for m in MODELS}
    A31R = {s: json.loads((ctx.a31.out / f"readout_s{s}.json").read_text()) for s in SEEDS}
    for s in SEEDS:
        cat = {k: [] for k in ("y", "b", "kind", "sess", *ARMS, *MODELS)}; masks = {}
        for h in FOLDS:
            D = fold_data(ctx, s, h); pk, seqs = D["pk"], D["seqs"]; M = set_masks(D); pr = {}
            for m in MODELS:
                P_ = ext[h][m]; pr[m] = np.asarray([P_[seqs[i]["id"]][x:y].mean(0).argmax() for i, x, y in zip(pk["seq"], pk["a"], pk["b"])])
                rl = {q["id"]: ir.relabel(D["base_preds"][q["id"]], P_[q["id"]]) for q in seqs}
                if not all(rl[seqs[i]["id"]][x] == pr[m][k] for k, (i, x, y) in enumerate(zip(pk["seq"], pk["a"], pk["b"]))):
                    raise SystemExit("the run rule and the sequence relabelling disagree")
                pr[m + "_seq"] = rl
            P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"); seq = {}
            for name, preds in (("A08 Base", D["base_preds"]), *[(m, pr[m + "_seq"]) for m in MODELS]):
                sr = t31.seq_rows(a8.cfg, P, preds, ev); seq[name] = {"sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr)}
            old = A31R[s]["models"][f"fit_f{1 - h}_heldout_f{h}"]["rows"]["Base"]
            rec["checks"][f"Base_sequence_equals_A31_s{s}_f{h}"] = bool(seq["A08 Base"]["sequence_row"] == old["sequence_row"] and seq["A08 Base"]["counts"] == old["counts"])
            rec["sequences"][f"s{s}_heldout_f{h}"] = seq
            for k, v in (("y", pk["labels"]), ("b", pk["base"]), ("kind", D["kind"]), ("sess", np.char.add(f"f{h}:", D["sessions"].astype(str))), *[(x, D["arms"][x]) for x in ARMS], *[(m, pr[m]) for m in MODELS]):
                cat[k].append(v)
            masks[h] = M; priv[(s, h)] = {m: pr[m] for m in MODELS}
        C = {k: np.concatenate(v) for k, v in cat.items()}; y, b = C["y"], C["b"]; Mc = {k: np.concatenate([masks[h][k] for h in FOLDS]) for k in masks[0]}
        r = {}
        for name in (*ARMS, *MODELS):
            p = C[name]; nat = C["kind"] == "native"
            r[name] = {"all": run_counts(y, b, p), "native": run_counts(y, b, p, nat), "constructed": run_counts(y, b, p, ~nat), "changes": change_table(y, b, p), "core": core_coverage(Mc, y, b, p)}
        r["error_overlap"] = error_overlap(y, {"A08 Base": b != y, **{m: C[m] != y for m in MODELS}})
        rec["runs"][str(s)] = r; nat = C["kind"] == "native"; wr = b != y
        for m in MODELS:
            d = ((wr & (C[m] == y)).astype(float) - ((~wr) & (C[m] != y)).astype(float))[nat]
            rec["native_bootstrap"][f"{m} - Base s{s}"] = {"unit": "session (native runs belong to one recording, so to one session exactly)", "sessions": int(len(np.unique(C["sess"][nat]))),
                                                          "accuracy_difference": float(d.mean()), "ci95": native_bootstrap(C["sess"][nat], d, s)}
    np.save(ctx.out / "readout_private.npy", np.array(priv, dtype=object), allow_pickle=True)
    if not all(v for k, v in rec["checks"].items() if k.startswith("Base_sequence")) or any(v["failed_checks"] for k, v in rec["checks"].items() if k.startswith("EchoViewCLIP")):
        publish(ctx, "readout.json", rec); raise SystemExit("a readout check failed")
    publish(ctx, "readout.json", rec)
    print(json.dumps({s: {k: v["all"] for k, v in rec["runs"][s].items() if "all" in v} for s in rec["runs"]}, indent=1)); return "runs and sequences"


# ----------------------------------------------------------------------------------------------- extras (CPU, descriptive)
def extras(ctx: Ctx) -> str:
    """Recordings counted three ways (any source / majority source / sources whose clip label is the run's true class),
    pooled over seeds, and where each external rule's S1 fixes and S3 breaks fall (native / constructed, short / long)."""
    data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); ro = np.load(ctx.out / "readout_private.npy", allow_pickle=True).item(); rec = {"recordings": {}, "by_run_type": {}}
    pooled = {}
    for s in SEEDS:
        for h in FOLDS:
            D = fold_data(ctx, s, h); M = set_masks(D); y = D["pk"]["labels"]
            for name in ("A5C_all", "S1", "S1_persistent", "S2", "S3", "S4"):
                p = pooled.setdefault(name, {"any": set(), "majority": set(), "true_class": set(), **{f"s{x}": set() for x in SEEDS}})
                for k in np.flatnonzero(M[name]):
                    src = D["src"][k]; p["any"] |= set(src); p["majority"].add(max(src, key=src.get)); tc = {v for v in src if data.label[v] == y[k]}; p["true_class"] |= tc; p[f"s{s}"] |= tc
            for m in MODELS:
                pr = ro[(s, h)][m]; L = D["length"]; nat = D["kind"] == "native"
                for name, tgt in (("S1", A5C), ("S3", A5C)):
                    for part, mk in (("native", nat), ("constructed", ~nat), ("short_le16", L <= 16), ("long", L > 16)):
                        c = rec["by_run_type"].setdefault(f"{m} s{s}", {}).setdefault(name, {}).setdefault(part, {"runs": 0, "to_A5C": 0})
                        c["runs"] += int((M[name] & mk).sum()); c["to_A5C"] += int((M[name] & mk & (pr == tgt)).sum())
    rec["recordings"] = {n: {k: len(v) for k, v in p.items()} for n, p in pooled.items()}
    rec["unmapped_mass"] = unmapped_mass(ctx)
    publish(ctx, "extras.json", rec); print(json.dumps(rec, indent=1)); return "extras"


def unmapped_mass(ctx: Ctx) -> dict:
    """Mean raw probability per view inside each set's runs (sample-weighted), for the classes each external map drops or
    keeps: EchoPrime's 11 views and EchoViewCLIP's PMPALA. Descriptive; the rule itself is unchanged."""
    out = {}
    for s in SEEDS:
        acc = {}
        for h in FOLDS:
            D = fold_data(ctx, s, h); M = set_masks(D); pk = D["pk"]; seqs = D["seqs"]; M["A4C_correct"] = (pk["labels"] == A4C) & (pk["base"] == A4C)
            with np.load(ctx.out / "echoprime" / f"pool_{h}.npz") as z:
                raw, off = z["raw"], z["offsets"]
            for name in ("S1", "S2", "S3", "S4", "A4C_correct"):
                a = acc.setdefault(name, {"samples": 0, "EchoPrime": np.zeros(11), "EchoViewCLIP_PMPALA": 0.0})
                for k in np.flatnonzero(M[name]):
                    i, x, y = int(pk["seq"][k]), int(pk["a"][k]), int(pk["b"][k]); a["samples"] += y - x; a["EchoPrime"] += raw[off[i] + x:off[i] + y].sum(0)
                    with np.load(ctx.out / "evc" / f"pool_{h}" / f"{seqs[i]['id']}.npz") as z:
                        a["EchoViewCLIP_PMPALA"] += float(z["raw"][x:y, 6].sum())
        out[f"s{s}"] = {n: {"samples": a["samples"], "EchoPrime": dict(zip(ECHOPRIME_VIEWS, (a["EchoPrime"] / max(1, a["samples"])).round(4).tolist())),
                            "EchoViewCLIP_PMPALA": round(a["EchoViewCLIP_PMPALA"] / max(1, a["samples"]), 4)} for n, a in acc.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("plans", "echoprime", "evc", "sets", "readout", "package", "extras"), required=True)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "plans":
        staged(ctx, "plans (CPU)", "cpu", lambda: plans(ctx))
    elif a.stage == "echoprime":
        staged(ctx, "EchoPrime inference and parity (GPU job wall time)", "gpu", lambda: echoprime(ctx, torch.device("cuda"), a.workers))
    elif a.stage == "evc":
        staged(ctx, f"EchoViewCLIP runner {'timing ' if a.limit else ''}(GPU job wall time)", "gpu", lambda: evc(ctx, a.limit))
    elif a.stage == "sets":
        staged(ctx, "sets and statistics (CPU)", "cpu", lambda: sets(ctx))
    elif a.stage == "readout":
        staged(ctx, "readout (CPU)", "cpu", lambda: readout(ctx))
    elif a.stage == "extras":
        staged(ctx, "extras (CPU)", "cpu", lambda: extras(ctx))
    else:
        import scripts.a5c_review_package as pkg
        staged(ctx, "review package (CPU)", "cpu", lambda: pkg.build(ctx))


if __name__ == "__main__":
    main()
