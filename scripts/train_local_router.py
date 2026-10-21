#!/usr/bin/env python
"""A14 (configs/experiments/video_local_router_v1.yaml): the fixed training pool, frozen A08 features on whole sequences,
Base (A08 + B3) and Alternative (A13 H) predictions, zones with full-sequence utility labels, the session split, and the
P/V router training (CPU).

  --prepare S [S ...]   pool, one frame decode, frozen A08 inference per seed (GPU), candidates and labels (CPU)
  --check               zone self-test, isolation, no-op/accept-all reproduction, optimiser, resume
  --train S [S ...]     P and V routers of the seeds (CPU)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.local_router as lr  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.train_frozen_switch as tf  # noqa: E402
import scripts.train_video_sustained_switch as tss  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A13_CONFIG = "configs/experiments/video_frozen_switch_v1.yaml"
A13_CHOICES = {"path": "runs/video_frozen_switch_v1/evaluation/choices.json", "sha256": "2907f89baa80a53817f2dd767274ca77e5efbf0a029d17c0e49dca3690b9708e"}
A08_CHOICES = {"path": "docs/results/2026-10-16_video_within_view/choices.json", "sha256": "b2a731e5f4393187ee9954f710821ac04938652f0a7d00e0381ad0b8b21cf27e"}
now, write_json, ledger_append, ledger_hours = tf.now, tf.write_json, tf.ledger_append, tf.ledger_hours


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vl = cfg, config, cfg["video_local_router"]; self.a13 = tf.Ctx(load_cfg(SimpleNamespace(config=A13_CONFIG, set=[])), A13_CONFIG)
        o = self.vl["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]; self.ledger = ROOT / self.vl["budget"]["ledger"]
        self.a10 = self.a13.a11.a10

    def base_params(self, s: int) -> dict:
        return json.loads(tf.vsh.pinned(A08_CHOICES).read_text())["choices"][f"all_s{s}"]["params"]

    def alt(self, s: int):
        ch = json.loads(tf.vsh.pinned(A13_CHOICES).read_text())["choices"][f"H_s{s}"]; man = json.loads((ROOT / "runs/video_frozen_switch_v1/train" / f"H_s{s}" / "run_manifest.json").read_text())
        p = Path(man["final_checkpoint"])
        if sha256_file(p) != man["final_sha256"] or json.loads(tf.vsh.pinned(A13_CHOICES).read_text())["heads"][f"H_s{s}"] != man["final_sha256"]:
            raise SystemExit(f"A13 H_s{s}: the head differs from its run record or the committed choices")
        h = tss.build_head(s); h.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); return h.eval(), float(ch["params"]["threshold"]), man["final_sha256"]


# ----------------------------------------------------------------------------------------------- pool
def pool(ctx: Ctx) -> list[dict]:
    seqs = vpr.training_data(ctx.a10.a07p).seqs; bank = [i for i, q in enumerate(seqs) if q["kind"] != "native"]
    pick = [bank[j] for j in np.random.default_rng([20261021, 1]).choice(len(bank), size=300, replace=False)]
    rng = np.random.default_rng([20261021, 2]); nat = []
    for c in range(5):
        cand = [i for i, q in enumerate(seqs) if q["kind"] == "native" and int(np.asarray(q["labels"])[0]) == c]; per, took = {}, 0
        for j in rng.permutation(len(cand)):
            i = cand[j]; sess = session_of(seqs[i]["id"])
            if per.get(sess, 0) >= 2:
                continue
            per[sess] = per.get(sess, 0) + 1; nat.append(i); took += 1
            if took == 40:
                break
    return [{"id": seqs[i]["id"], "kind": "multi" if seqs[i]["kind"] != "native" else "native", "labels": np.asarray(seqs[i]["labels"], int),
             "refs": [(str(r[0]), int(r[1]), float(r[2])) for r in seqs[i]["refs"]]} for i in sorted(pick) + sorted(nat)]


def per_sample_emb(W: dict, emb: np.ndarray, seqs: list[dict]) -> dict:
    out = {}
    for qi, q in enumerate(seqs):
        rows = np.flatnonzero(W["seq"] == qi); out[q["id"]] = emb[rows][rr.nearest_centre(len(q["labels"]), W["centre"][rows])]
    return out


def systems(probs: dict, edges: dict, seqs: list[dict], bparams: dict, thr: float) -> tuple[dict, dict]:
    return decode(bparams, probs), rsh.head_decode(probs, edges, seqs, thr)


def candidates(seqs: list[dict], probs: dict, emb: dict, base: dict, alt: dict, with_labels: bool = True) -> dict:
    rows = {"seq": [], "a": [], "b": [], "fp": [], "fv": [], "pos": [], "dtp": [], "dfp": [], "dfn": [], "kind": []}
    for qi, q in enumerate(seqs):
        b, a_ = base[q["id"]], alt[q["id"]]; zs = lr.zones(b, a_)
        if not zs:
            continue
        util = lr.zone_utilities(b, a_, q["labels"], zs) if with_labels else [{"positive": False, "dtp": 0, "dfp": 0, "dfn": 0}] * len(zs)
        for z, u in zip(zs, util):
            rows["seq"].append(qi); rows["a"].append(z[0]); rows["b"].append(z[1]); rows["kind"].append(q["kind"])
            rows["fp"].append(lr.features_p(probs[q["id"]], b, a_, z)); rows["fv"].append(lr.features_v(probs[q["id"]], emb[q["id"]], b, a_, z))
            rows["pos"].append(bool(u["positive"])); rows["dtp"].append(u["dtp"]); rows["dfp"].append(u["dfp"]); rows["dfn"].append(u["dfn"])
    if not rows["seq"]:
        raise SystemExit("no disagreement zone in the pool")
    return {k: np.asarray(v) for k, v in rows.items()}


def split(seqs: list[dict]) -> tuple[np.ndarray, int]:
    sess = sorted({session_of(r[0]) for q in seqs for r in q["refs"]} | {session_of(q["id"]) for q in seqs if q["kind"] == "native"})
    chk = {sess[i] for i in np.random.default_rng([20261021, 3]).permutation(len(sess))[: int(round(0.2 * len(sess)))]}
    out = []
    for q in seqs:
        if q["kind"] == "native":
            out.append(session_of(q["id"]) in chk)
        else:
            out.append(np.mean([session_of(r[0]) in chk for r in q["refs"]]) >= 0.5)
    mixed = sum(1 for q in seqs if q["kind"] == "multi" and 0 < np.mean([session_of(r[0]) in chk for r in q["refs"]]) < 1)
    return np.asarray(out, bool), int(mixed)


def prepare(ctx: Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0
    try:
        P = pool(ctx); is_check, mixed = split(P); refs = rw.half_refs(P); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(P, at, 5)
        orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
        vids = {r[0] for r in refs}
        if vids - set(prv.allowlist(ctx.a10.rv, ROOT)):
            raise SystemExit("a pool recording lies outside the training allowlist")
        t1 = time.time(); frames = rr.decode_frames(refs, FramePaths(prv.frame_dirs(ctx.a10.paths, ctx.a10.rv, vids)), workers); dsec = time.time() - t1
        device = torch.device("cuda"); tf.vsh.determinism(); fdir = ctx.out / "features"; fdir.mkdir(parents=True, exist_ok=True); rec = {"pool": {}, "seeds": {}}
        rec["pool"] = {"sequences": len(P), "multi": sum(q["kind"] == "multi" for q in P), "native": sum(q["kind"] == "native" for q in P), "windows": int(len(W["seq"])),
                       "refs": len(refs), "check_sequences": int(is_check.sum()), "decode_seconds": dsec, "streams_spanning_both_groups": mixed}
        for s in seeds:
            model, ssha = tf.frozen(ctx.a13, s, device); d0 = tf.path_digest(model); t2 = time.time(); lg, em, sec = rsh.infer(model, W, orders, frames, device); gsec += time.time() - t2
            if tf.path_digest(model) != d0 or model.training:
                raise SystemExit("the frozen path changed")
            del model; torch.cuda.empty_cache()
            probs = rr.per_sample(W, lg, P); embs = per_sample_emb(W, em, P); head, thr, hsha = ctx.alt(s)
            edges = rvs.edge_scores(head, W, lg, em, P, torch.device("cpu")); base, alt = systems(probs, edges, P, ctx.base_params(s), thr)
            cand = candidates(P, probs, embs, base, alt); fit = ~is_check[cand["seq"]] if cand["seq"].size else np.zeros(0, bool)
            np.savez(fdir / f"candidates_s{s}.npz", **cand, fit=fit, start_sha256=ssha, head_sha256=hsha)
            np.savez(fdir / f"pool_s{s}.npz", seq=W["seq"], centre=W["centre"], logits=lg, emb=em)
            short = cand["b"] - cand["a"] <= 16 if cand["seq"].size else np.zeros(0, bool)
            rec["seeds"][str(s)] = {"candidates": int(cand["seq"].size), "fit": int(fit.sum()), "check": int((~fit).sum()), "positive_fit": int((cand["pos"] & fit).sum()),
                                    "positive_check": int((cand["pos"] & ~fit).sum()), "sequences_with_zones": int(len(set(cand["seq"].tolist()))), "short_zones": int(short.sum()),
                                    "positive_short": int((cand["pos"] & short).sum()), "by_kind": {k: int((cand["kind"] == k).sum()) for k in ("multi", "native")},
                                    "positive_by_kind": {k: int((cand["pos"] & (cand["kind"] == k)).sum()) for k in ("multi", "native")},
                                    "sessions_with_zones": len({session_of(r[0]) for i in set(cand["seq"].tolist()) for r in P[i]["refs"]}),
                                    "dtp_sum_pos": int(cand["dtp"][cand["pos"]].sum()), "dfp_sum_pos": int(cand["dfp"][cand["pos"]].sum()), "head_sha256": hsha, "threshold": thr}
            print(f"seed {s}: {json.dumps(rec['seeds'][str(s)])}", flush=True)
        ids = {q["id"] for q in P} | vids | {session_of(v) for v in vids}; rec["finished"] = now(); flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        ledger_append(ctx.ledger, "prepare (frozen A08 pool inference, GPU)", "gpu", started, g, ok, note); ledger_append(ctx.ledger, "prepare (decode, candidates, CPU)", "cpu", started, total - g, ok, note)


# ----------------------------------------------------------------------------------------------- routers
class Router(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build(kind: str, d: int, seed: int) -> Router:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((4000 if kind == "P" else 5000) + seed); return Router(d)


def load_cand(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "features" / f"candidates_s{s}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fit_router(c: dict, kind: str, s: int, upto: int = 1350, state: dict | None = None, start: int = 0):
    X = c["fp" if kind == "P" else "fv"].astype(np.float32); fit = c["fit"]; y = c["pos"].astype(np.float32)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6; Xn = (X - mu) / sd
    npos, nneg = float(y[fit].sum()), float((1 - y[fit]).sum()); wpos, wneg = (npos + nneg) / (2 * max(1.0, npos)), (npos + nneg) / (2 * max(1.0, nneg))
    m = build(kind, X.shape[1], s); opt = torch.optim.AdamW(m.parameters(), lr=0.0, weight_decay=1e-4)
    if state is not None:
        m.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    idx_fit = np.flatnonzero(fit); draws = np.random.default_rng([20261021, 4, s]).integers(0, len(idx_fit), size=(1350, 128)); losses = []
    for u in range(start, upto):
        opt.param_groups[0]["lr"] = 1e-3 * lr_factor(u, 135, 1350); bi = idx_fit[draws[u]]; xb, yb = torch.from_numpy(Xn[bi]), torch.from_numpy(y[bi])
        w = torch.where(yb > 0.5, torch.full_like(yb, wpos), torch.full_like(yb, wneg)); loss = (w * F.binary_cross_entropy_with_logits(m(xb), yb, reduction="none")).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0); opt.step(); losses.append(float(loss.detach()))
    return m, opt, losses, {"mu": mu, "sd": sd, "wpos": wpos, "wneg": wneg}


def score(m: Router, X: np.ndarray, stats: dict) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(((X.astype(np.float32) - stats["mu"]) / stats["sd"]).astype(np.float32)))).numpy()


def train(ctx: Ctx, seeds: list[int]) -> None:
    for s in seeds:
        c = load_cand(ctx, s)
        for kind in ("P", "V"):
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                if int(c["pos"][c["fit"]].sum()) == 0:
                    raise SystemExit(f"seed {s}: no positive fit zone; preparation not met")
                m, _, losses, stats = fit_router(c, kind, s); sc = score(m, c["fp" if kind == "P" else "fv"], stats); y = c["pos"]
                def acc(mask):
                    return float(((sc[mask] >= 0.5) == y[mask]).mean()) if mask.any() else float("nan")
                def bal(mask):
                    pm, nm = mask & y, mask & ~y
                    return float(0.5 * ((sc[pm] >= 0.5).mean() if pm.any() else np.nan) + 0.5 * ((sc[nm] < 0.5).mean() if nm.any() else np.nan))
                rd = ctx.out / "routers"; rd.mkdir(parents=True, exist_ok=True); path = rd / f"{kind}_s{s}.pt"
                torch.save({"model": m.state_dict(), "mu": torch.from_numpy(stats["mu"]), "sd": torch.from_numpy(stats["sd"]), "d": int(stats["mu"].size)}, path)
                rec = {"kind": kind, "seed": s, "parameters": int(sum(p.numel() for p in m.parameters())), "loss_first": float(np.mean(losses[:135])), "loss_last": float(np.mean(losses[-135:])),
                       "fit_balanced_accuracy": bal(c["fit"]), "check_balanced_accuracy": bal(~c["fit"]), "fit_accuracy": acc(c["fit"]), "check_accuracy": acc(~c["fit"]),
                       "weights": {"pos": stats["wpos"], "neg": stats["wneg"]}, "sha256": sha256_file(path)}
                write_json(rd / f"{kind}_s{s}.json", rec); ok = True; note = json.dumps({k: rec[k] for k in ("loss_last", "fit_balanced_accuracy", "check_balanced_accuracy")}); print(f"[{kind}_s{s}] {note}", flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train router {kind}_s{s} (CPU)", "cpu", started, time.time() - t0, ok, note)


def check(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"zone_selftest": lr.selftest()}
    try:
        P = pool(ctx); refs = rw.half_refs(P); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(P, at, 5); s = seeds[0]
        with np.load(ctx.out / "features" / f"pool_s{s}.npz") as z:
            lg, em = z["logits"], z["emb"]
        probs = rr.per_sample(W, lg, P); embs = per_sample_emb(W, em, P); head, thr, _ = ctx.alt(s); edges = rvs.edge_scores(head, W, lg, em, P, torch.device("cpu"))
        base, alt = systems(probs, edges, P, ctx.base_params(s), thr)
        shuffled = [{**q, "labels": np.random.default_rng(7).permutation(q["labels"])} for q in P]
        c1 = candidates(P, probs, embs, base, alt, with_labels=False); c2 = candidates(shuffled, probs, embs, base, alt, with_labels=False)
        out["isolation"] = bool(np.array_equal(c1["a"], c2["a"]) and np.array_equal(c1["fp"], c2["fp"]) and np.array_equal(c1["fv"], c2["fv"]))
        noop = all(np.array_equal(lr.apply(base[q["id"]], alt[q["id"]], lr.zones(base[q["id"]], alt[q["id"]]), [False] * len(lr.zones(base[q["id"]], alt[q["id"]]))), base[q["id"]]) for q in P)
        allz = all(np.array_equal(lr.apply(base[q["id"]], alt[q["id"]], lr.zones(base[q["id"]], alt[q["id"]]), [True] * len(lr.zones(base[q["id"]], alt[q["id"]]))), alt[q["id"]]) for q in P)
        out["noop_equals_base"], out["all_equals_alternative"] = bool(noop), bool(allz)
        c = load_cand(ctx, s); m6, _, l6, _ = fit_router(c, "V", s, upto=6); m3, o3, l3, _ = fit_router(c, "V", s, upto=3)
        m33, _, l33, _ = fit_router(c, "V", s, upto=6, state={"model": m3.state_dict(), "opt": o3.state_dict()}, start=3)
        out["resume"] = {"loss_equal": l6[3:] == l33, "params_equal": all(torch.equal(a, b) for a, b in zip(m6.state_dict().values(), m33.state_dict().values()))}
        r = build("V", c["fv"].shape[1], s); opt = torch.optim.AdamW(r.parameters()); out["optimiser_only_router"] = [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in r.parameters()]
        out["parameters"] = {"P": int(sum(p.numel() for p in build("P", c["fp"].shape[1], s).parameters())), "V": int(sum(p.numel() for p in r.parameters()))}
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cap": float(ctx.vl["budget"]["gpu_hours_max"])}
        out["ok"] = bool(out["zone_selftest"]["ok"] and out["isolation"] and noop and allz and out["resume"]["loss_equal"] and out["resume"]["params_equal"] and out["optimiser_only_router"])
        ok = out["ok"]
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps(out, indent=1, default=str), flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", nargs="+", type=int); g.add_argument("--check", nargs="+", type=int); g.add_argument("--train", nargs="+", type=int)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.prepare, a.workers)
    if a.check:
        return check(ctx, a.check)
    train(ctx, a.train)


if __name__ == "__main__":
    main()
