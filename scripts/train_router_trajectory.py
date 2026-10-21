#!/usr/bin/env python
"""A17 (configs/experiments/video_router_trajectory_v1.yaml): candidate-conditioned local prediction trajectories over
A15's pinned zones, in true order (O) or shuffled inside each region (S), and the equal-capacity convolutional router
trained on CPU with A15's labels.

  --prepare            pins; per-sample probabilities from A15's window cache; Base/Alternative/52 features reproduced;
                       O and S trajectory tensors of the pool
  --check              tensor, permutation, padding, isolation, gradient, start, batch and resume checks
  --train S [S ...]    O and S routers of the seeds, with router-check logging at the fixed rule
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.local_router as lr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_discriminative_r1 as rd1  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A15_CONFIG = "configs/experiments/video_router_data_v1.yaml"
SEEDS, ARMS, CTX, C, CH = (0, 2), ("O", "S"), 16, 5, 18
SET_POOL, SET_SELECTION, SET_CALIBRATION = 0, 1, 2
now, write_json, ledger_append, ledger_hours = tl.now, tl.write_json, tl.ledger_append, tl.ledger_hours
torch.set_num_threads(1)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vt = cfg, config, cfg["video_router_trajectory"]
        self.a15 = trd.Ctx(load_cfg(SimpleNamespace(config=A15_CONFIG, set=[])), A15_CONFIG)
        o = self.vt["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]; self.ledger = ROOT / self.vt["budget"]["ledger"]


def verify_pins(ctx: Ctx) -> list[str]:
    p = ctx.vt["pinned"]; f = ctx.a15.out / "features"
    files = {"A15 preparation": (ROOT / p["A15_preparation"]["path"], p["A15_preparation"]["sha256"]), "A15 manifest": (ROOT / p["A15_pool_manifest"]["path"], p["A15_pool_manifest"]["sha256"]),
             "A15 choices": (ctx.a15.out / "evaluation" / "choices.json", p["A15_choices_sha256"]), "A15 screen": (ctx.a15.out / "evaluation" / "screen.json", p["A15_screen_sha256"]),
             "A16 choices": (ROOT / "runs/video_router_utility_v1/evaluation/choices.json", p["A16_choices_sha256"]),
             "A16 screen": (ROOT / "runs/video_router_utility_v1/evaluation/screen.json", p["A16_screen_sha256"])}
    for s in SEEDS:
        files.update({f"candidates_s{s}": (f / f"candidates_s{s}.npz", p["A15_candidates"][s]), f"preds_s{s}": (f / f"preds_s{s}.npz", p["A15_preds"][s]),
                      f"windows_s{s}": (f / f"windows_s{s}.npz", p["A15_windows"][s]), f"large-P_s{s}": (ctx.a15.out / "routers" / f"large-P_s{s}.pt", p["A15_large_P"][s]),
                      f"A16 U_s{s}": (ROOT / "runs/video_router_utility_v1/routers" / f"U_s{s}.pt", p["A16_U"][s])})
    for k, (path, sha) in files.items():
        if sha256_file(path) != sha:
            raise SystemExit(f"{k}: the pinned file differs from its pin")
    return sorted(files)


# ----------------------------------------------------------------------------------------------- trajectories
def regions(n: int, a: int, b: int) -> tuple[int, int, np.ndarray]:
    la, rb = max(0, a - CTX), min(n, b + CTX); t = np.arange(la, rb); return la, rb, np.where(t < a, 0, np.where(t < b, 1, 2)).astype(np.int64)


def trajectory(prob: np.ndarray, base: np.ndarray, alt: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """One zone's rows in true time order: 5 probabilities, Base and Alternative one-hots, region one-hot (18 channels)."""
    la, rb, reg = regions(prob.shape[0], a, b); eye = np.eye(C, dtype=np.float32)
    rows = np.concatenate([prob[la:rb].astype(np.float32), eye[np.asarray(base[la:rb], int)], eye[np.asarray(alt[la:rb], int)], np.eye(3, dtype=np.float32)[reg]], 1)
    return rows, reg


def shuffled(rows: np.ndarray, reg: np.ndarray, key: list[int]) -> np.ndarray:
    """S: the rows of each region permuted once by default_rng([*key, region]); whole rows move, regions never mix."""
    out = rows.copy()
    for r in (0, 1, 2):
        idx = np.flatnonzero(reg == r)
        if idx.size:
            out[idx] = rows[idx[np.random.default_rng([*key, r]).permutation(idx.size)]]
    return out


def tensors(c: dict, probs: list, base: list, alt: list, seed: int, set_code: int, local_seq=None) -> dict:
    """Ragged O/S rows of every zone in candidate order; local_seq maps a zone's sequence to its index in probs/base/alt."""
    rows_o, rows_s, reg, off = [], [], [], [0]
    for k in range(c["seq"].size):
        i = int(c["seq"][k]); j = i if local_seq is None else local_seq[i]; a, b = int(c["a"][k]), int(c["b"][k])
        r, g = trajectory(probs[j], base[j], alt[j], a, b); rows_o.append(r); rows_s.append(shuffled(r, g, [20261021, 17, seed, set_code, i, a, b])); reg.append(g); off.append(off[-1] + len(g))
    cat = lambda x, d: np.concatenate(x) if x else np.zeros(d, np.float32)  # noqa: E731
    return {"O": cat(rows_o, (0, CH)), "S": cat(rows_s, (0, CH)), "region": cat(reg, (0,)).astype(np.int64), "off": np.asarray(off, np.int64)}


def pool_inputs(ctx: Ctx, s: int, labels: list | None = None) -> dict:
    """Per-sample probabilities of every pool sequence from A15's window cache (A15's own mapping), with Base,
    Alternative and the 52 features rebuilt from them; `labels` replaces the references (isolation check only)."""
    pr = trd.load_preds(ctx.a15, s); ref = pr["labels"] if labels is None else labels; seqs = [{"id": str(i), "labels": ref[i], "kind": "x"} for i in range(len(ref))]
    with np.load(ctx.a15.out / "features" / f"windows_s{s}.npz") as z:
        W = {"seq": z["seq"], "centre": z["centre"]}; lg, em = z["logits"], z["emb"]; ident = (str(z["start_sha256"]), str(z["path_digest"]))
    prep = json.loads((ctx.a15.out / "preparation.json").read_text())["seeds"][str(s)]
    if ident != (prep["start_sha256"], prep["path_digest"]):
        raise SystemExit("the window cache identity differs from A15's preparation")
    probs = rr.per_sample(W, lg, seqs); head, thr, _ = ctx.a15.a14.alt(s); edges = rvs.edge_scores(head, W, lg, em, seqs, torch.device("cpu"))
    base, alt = tl.systems(probs, edges, seqs, ctx.a15.a14.base_params(s), thr)
    return {"seqs": seqs, "probs": [probs[q["id"]] for q in seqs], "base": [base[q["id"]] for q in seqs], "alt": [alt[q["id"]] for q in seqs], "pr": pr}


def prepare(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        pins = verify_pins(ctx); rec = {"config_sha256": sha256_file(ROOT / ctx.config), "pins_verified": pins, "seeds": {}}; td = ctx.out / "trajectories"; td.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            c = trd.load_cand(ctx.a15, s); I = pool_inputs(ctx, s); pr = I["pr"]
            same_base = all(np.array_equal(I["base"][i], pr["base"][i]) for i in range(len(pr["base"]))); same_alt = all(np.array_equal(I["alt"][i], pr["alt"][i]) for i in range(len(pr["alt"])))
            cand = trd.to_arrays(trd.candidates_p(I["seqs"], {q["id"]: I["probs"][i] for i, q in enumerate(I["seqs"])}, {q["id"]: I["base"][i] for i, q in enumerate(I["seqs"])},
                                                  {q["id"]: I["alt"][i] for i, q in enumerate(I["seqs"])}, with_labels=False))
            same_zones = bool(np.array_equal(cand["seq"], c["seq"]) and np.array_equal(cand["a"], c["a"]) and np.array_equal(cand["b"], c["b"])); same_fp = bool(same_zones and np.array_equal(cand["fp"], c["fp"]))
            if not (same_base and same_alt and same_zones and same_fp):
                raise SystemExit(f"seed {s}: the rebuilt inputs differ from A15 (base {same_base}, alt {same_alt}, zones {same_zones}, features {same_fp})")
            T = tensors(c, I["probs"], I["base"], I["alt"], s, SET_POOL); np.savez(td / f"pool_s{s}.npz", **T)
            lens = np.diff(T["off"]); zl = c["b"] - c["a"]
            rec["seeds"][str(s)] = {"rebuilt_equal_A15": {"base": same_base, "alternative": same_alt, "zones": same_zones, "features_bitwise": same_fp}, "zones": int(c["seq"].size),
                                    "rows": int(T["off"][-1]), "trajectory_length": {"min": int(lens.min()), "median": float(np.median(lens)), "max": int(lens.max())},
                                    "zone_length": {"min": int(zl.min()), "median": float(np.median(zl)), "max": int(zl.max())},
                                    "empty_left": int(((c["a"] - np.maximum(0, c["a"] - CTX)) == 0).sum()), "empty_right": int(sum(1 for k in range(c["seq"].size) if int(c["b"][k]) >= len(pr["labels"][int(c["seq"][k])])))}
            print(f"seed {s}: {json.dumps(rec['seeds'][str(s)])}", flush=True)
        rec["finished"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare trajectories (CPU)", "cpu", started, time.time() - t0, ok, note)


def load_t(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "trajectories" / f"pool_s{s}.npz") as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------- model
class TrajRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.c1 = nn.Conv1d(CH, 16, 5, padding=2); self.c2 = nn.Conv1d(16, 16, 5, padding=2)
        self.head = nn.Sequential(nn.Linear(100, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, region: torch.Tensor, p52: torch.Tensor) -> torch.Tensor:
        m = (region >= 0).to(x.dtype).unsqueeze(1); h = x.transpose(1, 2) * m
        h = F.gelu(self.c1(h)) * m; h = F.gelu(self.c2(h)) * m; pooled = []
        for r in range(3):
            mr = (region == r).to(x.dtype).unsqueeze(1); pooled.append((h * mr).sum(-1) / mr.sum(-1).clamp(min=1.0))
        return self.head(torch.cat([*pooled, p52], 1)).squeeze(-1)


def build(seed: int) -> TrajRouter:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6000 + seed); return TrajRouter()


def batch(T: dict, arm: str, idx) -> tuple[torch.Tensor, torch.Tensor]:
    off = T["off"]; L = int(max(off[k + 1] - off[k] for k in idx)); x = np.zeros((len(idx), L, CH), np.float32); r = np.full((len(idx), L), -1, np.int64)
    for j, k in enumerate(idx):
        a, b = int(off[k]), int(off[k + 1]); x[j, :b - a] = T[arm][a:b]; r[j, :b - a] = T["region"][a:b]
    return torch.from_numpy(x), torch.from_numpy(r)


def norm52(c: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = c["fp"].astype(np.float32); mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-6; return ((X - mu) / sd).astype(np.float32), mu, sd


def fit(T: dict, arm: str, c: dict, mask: np.ndarray, s: int, upto: int = trd.UPDATES, state: dict | None = None, start: int = 0):
    """A15 large-P's labels, balanced weights, batches and schedule, with the trajectory router."""
    Xn, mu, sd = norm52(c, mask); y = c["pos"].astype(np.float32); npos, nneg = float(y[mask].sum()), float((1 - y[mask]).sum())
    wpos, wneg = (npos + nneg) / (2 * max(1.0, npos)), (npos + nneg) / (2 * max(1.0, nneg)); m = build(s); opt = torch.optim.AdamW(m.parameters(), lr=0.0, weight_decay=1e-4)
    if state is not None:
        m.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    idx_fit = np.flatnonzero(mask); u0 = np.random.default_rng([20261021, 15, 8, s]).random((trd.UPDATES, trd.BATCH)); draws = np.minimum((u0 * len(idx_fit)).astype(np.int64), len(idx_fit) - 1)
    losses = []
    for u in range(start, upto):
        opt.param_groups[0]["lr"] = 1e-3 * lr_factor(u, trd.WARM, trd.UPDATES); bi = idx_fit[draws[u]]; xb, rb = batch(T, arm, bi); yb = torch.from_numpy(y[bi])
        w = torch.where(yb > 0.5, torch.full_like(yb, wpos), torch.full_like(yb, wneg)); loss = (w * F.binary_cross_entropy_with_logits(m(xb, rb, torch.from_numpy(Xn[bi])), yb, reduction="none")).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0); opt.step(); losses.append(float(loss.detach()))
    return m, opt, losses, {"mu": mu, "sd": sd, "wpos": wpos, "wneg": wneg, "n_fit": int(mask.sum()), "n_pos": int(npos), "n_neg": int(nneg)}


@torch.no_grad()
def score(m: TrajRouter, T: dict, arm: str, Xn: np.ndarray, chunk: int = 256) -> np.ndarray:
    n = len(T["off"]) - 1; out = []
    for k0 in range(0, n, chunk):
        idx = list(range(k0, min(n, k0 + chunk))); xb, rb = batch(T, arm, idx); out.append(torch.sigmoid(m(xb, rb, torch.from_numpy(Xn[idx]))).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def router_path(ctx: Ctx, arm: str, s: int) -> Path:
    return ctx.out / "routers" / f"{arm}_s{s}.pt"


def load_router(ctx: Ctx, arm: str, s: int):
    rec = json.loads((ctx.out / "routers" / f"{arm}_s{s}.json").read_text()); p = router_path(ctx, arm, s)
    if sha256_file(p) != rec["sha256"]:
        raise SystemExit(f"{arm}_s{s}: router weights differ from their record")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = build(s); m.load_state_dict(st_["model"]); m.eval()
    return m, st_["mu"].numpy(), st_["sd"].numpy(), rec["sha256"]


# ----------------------------------------------------------------------------------------------- router-check logging
def boundary_counts(preds: dict, refs: list, members, tol: float) -> dict:
    m = npd = ntr = 0
    for i in members:
        p, t = np.asarray(preds[i]), np.asarray(refs[i]); bp, bt = np.flatnonzero(np.diff(p) != 0) + 1, np.flatnonzero(np.diff(t) != 0) + 1
        m, npd, ntr = m + len(match_boundaries(bp, bt, tol)), npd + bp.size, ntr + bt.size
    return {"recall": m / max(1, ntr), "precision": m / max(1, npd)}


def check_log(c: dict, pr: dict, acc: np.ndarray, members: dict, tol: float) -> dict:
    """Accepted router-check edits by single-zone class (beneficial / zero / harmful), then whole-stream counts."""
    m = c["check"] & acc; zero = np.abs(c["du"]) < 1e-12
    out = {"accepted": int(m.sum()), "beneficial": int((m & c["pos"]).sum()), "zero": int((m & zero).sum()), "harmful": int((m & (c["du"] < -1e-12)).sum())}
    for kind, mem in members.items():
        routed = trd.routed_lists(c, pr, m, mem); r = trd.pooled_f1(routed, pr["labels"]); seqs = [{"id": i, "kind": "multi", "labels": pr["labels"][i]} for i in mem]
        sh = rd1.short_counts(seqs, routed) if kind == "multi" else {}
        out[kind] = {"f1": r["f1"], "tp": r["tp"], "pred": r["pred"], "ref": r["ref"], **({"fp_short": sh["fp_short"], "fn_short": sh["fn_short"]} if sh else {}),
                     "boundary": boundary_counts(routed, pr["labels"], mem, tol)}
    return out


def tolerance() -> float:
    import scripts.report_video_switch_head as rsh
    cfg = rsh.a08_ctx().cfg; return float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])


def train(ctx: Ctx, seeds: list[int]) -> None:
    ck = trd.check_kinds(ctx.a15); tol = tolerance()
    for s in seeds:
        c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); T = load_t(ctx, s); mask = c["large_fit"]; rd = ctx.out / "routers"; rd.mkdir(parents=True, exist_ok=True)
        a15 = torch.load(ctx.a15.out / "routers" / f"large-P_s{s}.pt", map_location="cpu", weights_only=True); Xn, mu, sd = norm52(c, mask)
        same_norm = bool(torch.equal(torch.from_numpy(mu), a15["mu"]) and torch.equal(torch.from_numpy(sd), a15["sd"]))
        base_log = {"Base": check_log(c, pr, np.zeros(c["seq"].size, bool), ck, tol), "Alternative": check_log(c, pr, np.ones(c["seq"].size, bool), ck, tol)}; write_json(rd / f"check_reference_s{s}.json", base_log)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                m, _, losses, stats = fit(T, arm, c, mask, s); wall = time.time() - t0; m.eval(); sc = score(m, T, arm, Xn); p = router_path(ctx, arm, s)
                torch.save({"model": m.state_dict(), "mu": torch.from_numpy(stats["mu"]), "sd": torch.from_numpy(stats["sd"])}, p)
                fpos, fneg = mask & c["pos"], mask & ~c["pos"]; fixed = sc > 0.5
                rec = {"arm": arm, "seed": s, "parameters": int(sum(x.numel() for x in m.parameters())), "updates": len(losses), "train_wall_seconds": wall, "threads": torch.get_num_threads(),
                       "fit": {k: stats[k] for k in ("n_fit", "n_pos", "n_neg", "wpos", "wneg")}, "normalisation_equals_A15_large_P": same_norm,
                       "loss_first": float(np.mean(losses[:trd.WARM])), "loss_last": float(np.mean(losses[-trd.WARM:])),
                       "fit_balanced_accuracy_at_fixed": float(0.5 * ((sc[fpos] > 0.5).mean() + (sc[fneg] <= 0.5).mean())), "check_fixed": check_log(c, pr, fixed, ck, tol), "sha256": sha256_file(p)}
                write_json(rd / f"{arm}_s{s}.json", rec); ok = True
                note = json.dumps({"loss_last": rec["loss_last"], "wall_s": round(wall, 1), "check_fixed_accepted": rec["check_fixed"]["accepted"]})
                print(f"[{arm}_s{s}] {json.dumps(rec, default=str)}", flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train router {arm}_s{s} (CPU, 1 thread)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- checks
def check(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {}
    try:
        out["pins"] = verify_pins(ctx); prep = json.loads((ctx.out / "preparation.json").read_text())
        out["rebuilt_equal_A15"] = {s: prep["seeds"][str(s)]["rebuilt_equal_A15"] for s in map(str, SEEDS)}
        s = 0; c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); T = load_t(ctx, s); I = pool_inputs(ctx, s); mask = c["large_fit"]
        T2 = tensors(c, I["probs"], I["base"], I["alt"], s, SET_POOL); out["tensors_rebuilt_equal"] = bool(all(np.array_equal(T[k], T2[k]) for k in ("O", "S", "region", "off")))
        same_members, same_sums, moved, rows_sorted = True, True, 0, lambda x: x[np.lexsort(x.T[::-1])]  # noqa: E731
        for k in range(c["seq"].size):
            a, b = int(T["off"][k]), int(T["off"][k + 1]); g = T["region"][a:b]
            for r in (0, 1, 2):
                o, sh = T["O"][a:b][g == r], T["S"][a:b][g == r]
                same_members &= bool(np.array_equal(rows_sorted(o), rows_sorted(sh))); same_sums &= bool(np.allclose(o.sum(0), sh.sum(0), atol=1e-5)); moved += int(not np.array_equal(o, sh))
        out["O_S_regions"] = {"same_members": same_members, "same_channel_sums": same_sums, "regions_reordered": moved}
        J = pool_inputs(ctx, s, labels=[np.random.default_rng(7).permutation(x) for x in pr["labels"]]); T3 = tensors(c, J["probs"], J["base"], J["alt"], s, SET_POOL)
        cJ = trd.to_arrays(trd.candidates_p(J["seqs"], {q["id"]: J["probs"][i] for i, q in enumerate(J["seqs"])}, {q["id"]: J["base"][i] for i, q in enumerate(J["seqs"])},
                                            {q["id"]: J["alt"][i] for i, q in enumerate(J["seqs"])}, with_labels=False))
        Xn0, _, _ = norm52(c, mask); mr = build(s).eval(); sc1, sc2 = score(mr, T, "O", Xn0), score(mr, T3, "O", Xn0)
        out["reference_isolation"] = bool(all(np.array_equal(T[k], T3[k]) for k in ("O", "S", "region", "off")) and np.array_equal(cJ["a"], c["a"]) and np.array_equal(cJ["fp"], c["fp"])
                                          and np.array_equal(sc1, sc2))
        prob = np.full((40, C), 0.2, np.float32); bse = np.zeros(40, int); alt_ = bse.copy(); alt_[0:4] = 3; alt_[36:40] = 2; alt_[10:30] = 1
        cases = {"empty_left": (0, 4), "empty_right": (36, 40), "long_zone": (10, 30)}; m = build(s).eval(); Xz = torch.zeros((1, 52)); res = {}
        for name, (a, b) in cases.items():
            r_, g = trajectory(prob, bse, alt_, a, b); Tq = {"O": r_, "region": g, "off": np.asarray([0, len(g)])}; xb, rb = batch(Tq, "O", [0])
            pad = torch.cat([xb, torch.zeros((1, 23, CH))], 1); rpad = torch.cat([rb, torch.full((1, 23), -1, dtype=torch.long)], 1)
            with torch.no_grad():
                d = float((m(xb, rb, Xz) - m(pad, rpad, Xz)).abs().max())
            res[name] = {"length": int(len(g)), "left": int((g == 0).sum()), "zone": int((g == 1).sum()), "right": int((g == 2).sum()), "padding_max_abs_diff": d}
        out["synthetic"] = {**res, "ok": bool(res["empty_left"]["left"] == 0 and res["empty_right"]["right"] == 0 and res["long_zone"]["zone"] == 20 and all(v["padding_max_abs_diff"] <= 1e-6 for v in res.values()))}
        r1, r2 = build(s), build(s); out["parameters"] = int(sum(p.numel() for p in r1.parameters()))
        out["identical_starts"] = all(torch.equal(a, b) for a, b in zip(r1.state_dict().values(), r2.state_dict().values()))
        opt = torch.optim.AdamW(r1.parameters()); out["optimiser_only_router"] = [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in r1.parameters()]
        Xn, _, _ = norm52(c, mask); k0 = int(np.flatnonzero(mask & (np.abs(c["du"]) < 1e-12))[0]); xb, rb = batch(T, "O", [k0]); r1.zero_grad()
        F.binary_cross_entropy_with_logits(r1(xb, rb, torch.from_numpy(Xn[[k0]])), torch.zeros(1)).backward(); out["zero_label_gradient_nonzero"] = bool(any(float(p.grad.abs().max()) > 0 for p in r1.parameters()))
        d1 = np.random.default_rng([20261021, 15, 8, s]).random((trd.UPDATES, trd.BATCH)); d2 = np.random.default_rng([20261021, 15, 8, s]).random((trd.UPDATES, trd.BATCH)); out["identical_batches"] = bool(np.array_equal(d1, d2))
        m6, _, l6, _ = fit(T, "O", c, mask, s, upto=6); m3, o3, _, _ = fit(T, "O", c, mask, s, upto=3)
        m33, _, l33, _ = fit(T, "O", c, mask, s, upto=6, state={"model": m3.state_dict(), "opt": o3.state_dict()}, start=3)
        out["resume"] = {"loss_equal": l6[3:] == l33, "params_equal": all(torch.equal(a, b) for a, b in zip(m6.state_dict().values(), m33.state_dict().values()))}
        zz = [lr.zones(pr["base"][i], pr["alt"][i]) for i in range(len(pr["base"]))]
        out["noop_equals_base"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [False] * len(zz[i])), pr["base"][i]) for i in range(len(zz)))
        out["all_equals_alternative"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [True] * len(zz[i])), pr["alt"][i]) for i in range(len(zz)))
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cpu_spent": ledger_hours(ctx.ledger, "cpu")}
        keys = ("tensors_rebuilt_equal", "reference_isolation", "identical_starts", "optimiser_only_router", "zero_label_gradient_nonzero", "identical_batches", "noop_equals_base", "all_equals_alternative")
        out["ok"] = bool(all(all(v.values()) for v in out["rebuilt_equal_A15"].values()) and all(out[k] for k in keys) and out["O_S_regions"]["same_members"] and out["O_S_regions"]["same_channel_sums"]
                         and out["synthetic"]["ok"] and out["parameters"] == 4385 and out["resume"]["loss_equal"] and out["resume"]["params_equal"]); ok = out["ok"]
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps(out, indent=1, default=str), flush=True)
        if not ok:
            raise SystemExit("a check failed")
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--train", nargs="+", type=int)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx)
    if a.check:
        return check(ctx)
    train(ctx, a.train)


if __name__ == "__main__":
    main()
