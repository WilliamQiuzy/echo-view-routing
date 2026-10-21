#!/usr/bin/env python
"""A16 (configs/experiments/video_router_utility_v1.yaml): count-based zone utilities on A15's pinned large-fit candidates
and the matched routers C (A15 large-P exactly) and U (|u|-weighted BCE on 1[u > 0]), trained on CPU.

  --prepare            pins, F_fit, utilities, cross-counts and the independent recount
  --check              synthetic utility and gradient tests, starts, batches, resume, no-op/accept-all, C/U code parity
  --train S [S ...]    C and U routers of the seeds, with router-check logging at the fixed rule
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
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

A15_CONFIG = "configs/experiments/video_router_data_v1.yaml"
SEEDS = (0, 2)
ARMS = ("C", "U")
now, write_json, ledger_append, ledger_hours = tl.now, tl.write_json, tl.ledger_append, tl.ledger_hours


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vu = cfg, config, cfg["video_router_utility"]
        self.a15 = trd.Ctx(load_cfg(SimpleNamespace(config=A15_CONFIG, set=[])), A15_CONFIG)
        o = self.vu["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]; self.ledger = ROOT / self.vu["budget"]["ledger"]


def verify_pins(ctx: Ctx, seeds) -> dict:
    p = ctx.vu["pinned_from_A15"]; f = ctx.a15.out / "features"; out = {}
    files = {"preparation": (ROOT / p["preparation"]["path"], p["preparation"]["sha256"]), "pool_manifest": (ROOT / p["pool_manifest"]["path"], p["pool_manifest"]["sha256"]),
             "A15 choices": (ctx.a15.out / "evaluation" / "choices.json", p["choices_sha256"]), "A15 screen": (ctx.a15.out / "evaluation" / "screen.json", p["screen_sha256"])}
    for s in seeds:
        files.update({f"candidates_s{s}": (f / f"candidates_s{s}.npz", p["candidates"][s]), f"preds_s{s}": (f / f"preds_s{s}.npz", p["preds"][s]),
                      f"windows_s{s}": (f / f"windows_s{s}.npz", p["windows"][s]), f"large-P_s{s}": (ctx.a15.out / "routers" / f"large-P_s{s}.pt", p["large_P"][s])})
    for k, (path, sha) in files.items():
        if sha256_file(path) != sha:
            raise SystemExit(f"{k}: the A15 file differs from its pin")
        out[k] = sha
    return out


# ----------------------------------------------------------------------------------------------- utilities
def counts_exposed(pred: np.ndarray, ref: np.ndarray) -> dict:
    """The evaluator-checked exposed matching (segment_objective_tables.match_pairs), independent of segment_f1."""
    p, t = labels_to_segments(np.asarray(pred)), labels_to_segments(np.asarray(ref)); return st._counts(p, t, st.match_pairs(p, t))


def fit_f1(pr: dict, members) -> dict:
    tp = npred = nref = 0
    for i in members:
        r = lr.seq_f1(pr["base"][i], pr["labels"][i]); tp, npred, nref = tp + r["tp"], npred + r["pred"], nref + r["ref"]
    return {"F_fit": 2 * tp / (npred + nref), "tp": tp, "pred": npred, "ref": nref, "streams": len(members)}


def utility(c: dict, F_fit: float) -> np.ndarray:
    if not np.array_equal(c["dfn"], -c["dtp"]):
        raise SystemExit("dFN differs from -dTP: the reference count is not fixed")
    u = 2.0 * c["dtp"] - F_fit * (c["dtp"] + c["dfp"])
    zero = (c["dtp"] == 0) & (c["dfp"] == 0)
    if np.abs(u[zero]).max(initial=0.0) != 0.0 or (np.abs(u[~zero]) == 0).any():
        raise SystemExit("u is zero exactly when dTP = dFP = 0 only")
    return np.where(zero, 0.0, u)


def u_weights(u: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """U's labels and weights: y = 1[u > 0], w = |u| / mean over the fit zones of |u| (u = 0 gives w = 0)."""
    mean = float(np.abs(u[fit]).mean())
    if mean == 0.0:
        raise SystemExit("mean |u| over the fit zones is 0: preparation not met")
    return (u > 0).astype(np.float32), (np.abs(u) / mean).astype(np.float32), mean


def classes(x: np.ndarray, zero: np.ndarray) -> np.ndarray:
    return np.where(zero, "zero", np.where(x > 0, "positive", "negative"))


def cross(c: dict, u: np.ndarray, mask: np.ndarray) -> dict:
    old = np.where(np.abs(c["du"]) < 1e-12, "zero", np.where(c["du"] > 0, "positive", "harmful")); new = classes(u, u == 0); out = {}
    for kind in ("multi", "native", "all"):
        m = mask & ((c["kind"] == kind) if kind != "all" else True)
        out[kind] = {f"old {a} / u {b}": int((m & (old == a) & (new == b)).sum()) for a in ("positive", "zero", "harmful") for b in ("positive", "zero", "negative")}
        out[kind]["u_sum_positive"] = float(u[m & (u > 0)].sum()); out[kind]["u_sum_negative"] = float(u[m & (u < 0)].sum())
        out[kind]["u_mean_abs_nonzero"] = float(np.abs(u[m & (u != 0)]).mean()) if (m & (u != 0)).any() else 0.0
        out[kind]["u_max"] = float(u[m].max()) if m.any() else 0.0; out[kind]["u_min"] = float(u[m].min()) if m.any() else 0.0
    return out


def recount(c: dict, pr: dict, idx: np.ndarray) -> dict:
    """Independent single-replacement recount of the given zones with the exposed matching."""
    bad = 0
    for k in idx:
        i = int(c["seq"][k]); b, a_, ref = pr["base"][i], pr["alt"][i], pr["labels"][i]; z = (int(c["a"][k]), int(c["b"][k]))
        c0, c1 = counts_exposed(b, ref), counts_exposed(lr.apply(b, a_, [z], [True]), ref)
        bad += int((c1["tp"] - c0["tp"], c1["fp"] - c0["fp"], c1["fn"] - c0["fn"]) != (int(c["dtp"][k]), int(c["dfp"][k]), int(c["dfn"][k])))
    return {"zones": int(len(idx)), "mismatches": bad}


def prepare(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        pins = verify_pins(ctx, SEEDS); grp, kinds = trd.seq_meta(ctx.a15); rec = {"config_sha256": sha256_file(ROOT / ctx.config), "pins_verified": sorted(pins), "seeds": {}}
        fit_multi = [int(i) for i in np.flatnonzero((grp == "fit") & (kinds == "multi"))]; ud = ctx.out / "utility"; ud.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); ff = fit_f1(pr, fit_multi)
            same = sum(lr.seq_f1(pr["base"][i], pr["labels"][i])["tp"] == counts_exposed(pr["base"][i], pr["labels"][i])["tp"]
                       and lr.seq_f1(pr["base"][i], pr["labels"][i])["pred"] == counts_exposed(pr["base"][i], pr["labels"][i])["pred"] for i in range(len(pr["base"])))
            u = utility(c, ff["F_fit"]); y, w, mean = u_weights(u, c["large_fit"])
            fitm, chk = c["large_fit"], c["check"]
            sample = np.concatenate([np.flatnonzero(fitm & (c["kind"] == "multi"))[:60], np.flatnonzero(fitm & (c["kind"] == "native")), np.flatnonzero(chk)[:40]])
            np.savez(ud / f"utility_s{s}.npz", u=u, y=y, w=w, large_fit=fitm, check=chk, F_fit=ff["F_fit"], mean_abs_u=mean)
            rec["seeds"][str(s)] = {"F_fit": ff, "mean_abs_u_fit": mean, "zones": {"large_fit": int(fitm.sum()), "check": int(chk.sum())},
                                    "u_classes": {g: {k: int((m & (classes(u, u == 0) == k)).sum()) for k in ("positive", "zero", "negative")} for g, m in (("large_fit", fitm), ("check", chk))},
                                    "weight_mass": {"positive": float(w[fitm & (u > 0)].sum()), "negative": float(w[fitm & (u < 0)].sum()), "zero_weight_zones": int((fitm & (w == 0)).sum())},
                                    "cross_large_fit": cross(c, u, fitm), "cross_check": cross(c, u, chk),
                                    "matching_check": {"sequences": len(pr["base"]), "segment_f1_equals_exposed_matching": int(same)}, "recount": recount(c, pr, sample)}
            r = rec["seeds"][str(s)]
            if r["matching_check"]["segment_f1_equals_exposed_matching"] != len(pr["base"]) or r["recount"]["mismatches"]:
                raise SystemExit(f"seed {s}: A15's counting differs from the exposed matching")
            print(f"seed {s}: F_fit {ff['F_fit']:.4f} mean|u| {mean:.4f} u classes {json.dumps(r['u_classes'])} recount {r['recount']}", flush=True)
        rec["finished"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare utilities (CPU)", "cpu", started, time.time() - t0, ok, note)


def load_u(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "utility" / f"utility_s{s}.npz") as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------- routers
def fit_weighted(c: dict, mask: np.ndarray, s: int, y: np.ndarray, w: np.ndarray, upto: int = trd.UPDATES, state: dict | None = None, start: int = 0):
    """trd.fit_router with per-zone labels and weights: the same normalisation, initialisation, batches, schedule and
    clipping; the loss is the batch mean of w * BCEWithLogits(output, y)."""
    X = c["fp"].astype(np.float32); mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-6; Xn = ((X - mu) / sd).astype(np.float32); yt, wt = y.astype(np.float32), w.astype(np.float32)
    m = tl.build("P", X.shape[1], s); opt = torch.optim.AdamW(m.parameters(), lr=0.0, weight_decay=1e-4)
    if state is not None:
        m.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    idx_fit = np.flatnonzero(mask); u0 = np.random.default_rng([20261021, 15, 8, s]).random((trd.UPDATES, trd.BATCH)); draws = np.minimum((u0 * len(idx_fit)).astype(np.int64), len(idx_fit) - 1)
    losses = []
    for u in range(start, upto):
        opt.param_groups[0]["lr"] = 1e-3 * lr_factor(u, trd.WARM, trd.UPDATES); bi = idx_fit[draws[u]]
        xb, yb, wb = torch.from_numpy(Xn[bi]), torch.from_numpy(yt[bi]), torch.from_numpy(wt[bi])
        loss = (wb * F.binary_cross_entropy_with_logits(m(xb), yb, reduction="none")).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0); opt.step(); losses.append(float(loss.detach()))
    return m, opt, losses, {"mu": mu, "sd": sd, "n_fit": int(mask.sum())}


def c_weights(c: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = c["pos"].astype(np.float32); npos, nneg = float(y[mask].sum()), float((1 - y[mask]).sum())
    wpos, wneg = (npos + nneg) / (2 * max(1.0, npos)), (npos + nneg) / (2 * max(1.0, nneg)); return y, np.where(c["pos"], wpos, wneg)


def edit_log(c: dict, pr: dict, u: np.ndarray, acc: np.ndarray, members: dict) -> dict:
    """Accepted edits on the router-check pool: utility signs and sizes, then actual net counts after combining."""
    m = c["check"] & acc; out = {"accepted": int(m.sum()), "u_positive": int((m & (u > 0)).sum()), "u_zero": int((m & (u == 0)).sum()), "u_negative": int((m & (u < 0)).sum()),
                                 "u_sum_positive": float(u[m & (u > 0)].sum()), "u_sum_negative": float(u[m & (u < 0)].sum())}
    for kind, mem in members.items():
        routed = trd.routed_lists(c, pr, m, mem); b = trd.pooled_f1({i: pr["base"][i] for i in mem}, pr["labels"]); r = trd.pooled_f1(routed, pr["labels"])
        out[kind] = {"f1": r["f1"], "base_f1": b["f1"], "net_dTP": r["tp"] - b["tp"], "net_dFP": (r["pred"] - r["tp"]) - (b["pred"] - b["tp"]), "accepted": int((m & np.isin(c["seq"], mem)).sum())}
    return out


def router_path(ctx: Ctx, arm: str, s: int) -> Path:
    return ctx.out / "routers" / f"{arm}_s{s}.pt"


def train(ctx: Ctx, seeds: list[int]) -> None:
    ck = trd.check_kinds(ctx.a15)
    for s in seeds:
        c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); U = load_u(ctx, s); mask = c["large_fit"]; rd = ctx.out / "routers"; rd.mkdir(parents=True, exist_ok=True)
        if not np.array_equal(U["large_fit"], mask):
            raise SystemExit("the utility file's fit mask differs from A15's")
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                if arm == "C":
                    m, _, losses, stats = trd.fit_router(c, mask, s)
                else:
                    m, _, losses, stats = fit_weighted(c, mask, s, U["y"], U["w"])
                sc = tl.score(m, c["fp"], stats); p = router_path(ctx, arm, s)
                torch.save({"model": m.state_dict(), "mu": torch.from_numpy(stats["mu"]), "sd": torch.from_numpy(stats["sd"]), "d": int(stats["mu"].size)}, p)
                fixed = sc > 0.5; u = U["u"]
                rec = {"arm": arm, "seed": s, "parameters": int(sum(x.numel() for x in m.parameters())), "updates": len(losses), "loss_first": float(np.mean(losses[:trd.WARM])),
                       "loss_last": float(np.mean(losses[-trd.WARM:])), "fit_accepted_at_fixed": int((fixed & mask).sum()),
                       "fit_u_positive_accepted": int((fixed & mask & (u > 0)).sum()), "fit_u_negative_accepted": int((fixed & mask & (u < 0)).sum()),
                       "check_fixed": edit_log(c, pr, u, fixed, ck), "sha256": sha256_file(p)}
                if arm == "C":
                    a15 = torch.load(ctx.a15.out / "routers" / f"large-P_s{s}.pt", map_location="cpu", weights_only=True); mA = tl.build("P", int(a15["d"]), s); mA.load_state_dict(a15["model"])
                    scA = tl.score(mA, c["fp"], {"mu": a15["mu"].numpy(), "sd": a15["sd"].numpy()})
                    rec["A15_large_P_parity"] = {"weights_bitwise": all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), a15["model"].values())),
                                                 "norm_bitwise": bool(torch.equal(torch.from_numpy(stats["mu"]), a15["mu"]) and torch.equal(torch.from_numpy(stats["sd"]), a15["sd"])),
                                                 "max_abs_score_diff": float(np.abs(sc - scA).max()), "sha_equal": rec["sha256"] == sha256_file(ctx.a15.out / "routers" / f"large-P_s{s}.pt")}
                write_json(rd / f"{arm}_s{s}.json", rec); ok = True
                note = json.dumps({"loss_last": rec["loss_last"], "check_fixed_accepted": rec["check_fixed"]["accepted"], "parity": rec.get("A15_large_P_parity")})
                print(f"[{arm}_s{s}] {json.dumps(rec, default=str)}", flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train router {arm}_s{s} (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- checks
def synthetic() -> dict:
    """Toy sequences with known counts: one replacement's dF sign equals the sign of 2 dTP - F dP with the set's own F,
    and dF = (2 dTP - F dP) / (P + R + dP) exactly."""
    ref0 = np.zeros(40, int); ref1 = np.array([0] * 30 + [3] * 6); other = np.array([1] * 30 + [2] * 30)
    cases = {"remove_false": (np.array([0] * 36 + [4] * 4), ref0, (36, 40)),        # Base ends in a false SC4C segment
             "lose_true": (ref1.copy(), ref1, (30, 36)),                           # the Alternative merges a short true A5C segment
             "recover_short": (np.zeros(36, int), ref1, (30, 36))}                 # the Alternative recovers it
    alts = {"remove_false": ref0.copy(), "lose_true": np.zeros(36, int), "recover_short": ref1.copy()}; out = {}
    want = {"remove_false": (0, -1), "lose_true": (-1, 0), "recover_short": (1, 0)}
    for name, (base, rr_, z) in cases.items():
        pool = [(other, other)]; b0 = [counts_exposed(p, r) for p, r in pool] + [counts_exposed(base, rr_)]
        TP, P, R = sum(x["tp"] for x in b0), sum(x["pred"] for x in b0), sum(x["ref"] for x in b0); Fb = 2 * TP / (P + R)
        after = counts_exposed(lr.apply(base, alts[name], [z], [True]), rr_); dTP, dP = after["tp"] - b0[-1]["tp"], after["pred"] - b0[-1]["pred"]
        Fa = 2 * (TP + dTP) / (P + dP + R); uval = 2 * dTP - Fb * dP
        out[name] = {"dTP": dTP, "dFP": dP - dTP, "dFN": after["fn"] - b0[-1]["fn"], "u": uval, "dF": Fa - Fb, "counts_as_designed": (dTP, dP - dTP) == want[name],
                     "sign_equal": bool(np.sign(Fa - Fb) == np.sign(uval)), "identity": bool(abs((Fa - Fb) - uval / (P + R + dP)) < 1e-12), "dFN_is_minus_dTP": after["fn"] - b0[-1]["fn"] == -dTP}
    out["directions"] = bool(out["remove_false"]["u"] > 0 and out["lose_true"]["u"] < 0 and out["recover_short"]["u"] > 0 and all(out[k]["counts_as_designed"] for k in cases))
    out["ok"] = bool(out["directions"] and all(v["sign_equal"] and v["identity"] and v["dFN_is_minus_dTP"] for k, v in out.items() if isinstance(v, dict)))
    return out


def check(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"synthetic_utility": synthetic(), "zone_selftest": lr.selftest()["ok"]}
    try:
        out["pins"] = sorted(verify_pins(ctx, SEEDS)); s = 0; c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); U = load_u(ctx, s); mask = c["large_fit"]
        u2 = np.where(c["check"], U["u"] * 10.0, U["u"]); y2, w2, mean2 = u_weights(u2, mask)
        out["weights_fit_only"] = bool(mean2 == float(U["mean_abs_u"]) and np.array_equal(w2[mask], U["w"][mask]))
        r = tl.build("P", 52, s); X = torch.from_numpy(((c["fp"].astype(np.float32) - c["fp"][mask].astype(np.float32).mean(0)) / (c["fp"][mask].astype(np.float32).std(0) + 1e-6)).astype(np.float32))
        zi = np.flatnonzero(mask & (U["w"] == 0))[:16]; loss = (torch.from_numpy(U["w"][zi]) * F.binary_cross_entropy_with_logits(r(X[zi]), torch.from_numpy(U["y"][zi]), reduction="none")).mean()
        r.zero_grad(); loss.backward(); out["zero_weight_zero_gradient"] = bool(all(float(p.grad.abs().max()) == 0.0 for p in r.parameters()))
        dirs = {}
        for name, k in (("positive_u", int(np.flatnonzero(mask & (U["u"] > 0))[0])), ("negative_u", int(np.flatnonzero(mask & (U["u"] < 0))[0]))):
            r = tl.build("P", 52, s); before = float(r(X[k:k + 1]).item()); opt = torch.optim.SGD(r.parameters(), lr=1e-2)
            loss = (torch.tensor([float(U["w"][k])]) * F.binary_cross_entropy_with_logits(r(X[k:k + 1]), torch.tensor([float(U["y"][k])]), reduction="none")).mean()
            opt.zero_grad(); loss.backward(); opt.step(); dirs[name] = float(r(X[k:k + 1]).item()) - before
        out["cost_directions"] = {**dirs, "ok": dirs["positive_u"] > 0 and dirs["negative_u"] < 0}
        rc, ru = tl.build("P", 52, s), tl.build("P", 52, s); out["identical_starts"] = all(torch.equal(a, b) for a, b in zip(rc.state_dict().values(), ru.state_dict().values()))
        opt = torch.optim.AdamW(ru.parameters()); out["optimiser_only_router"] = [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in ru.parameters()]
        yc, wc = c_weights(c, mask); m1, _, l1, _ = trd.fit_router(c, mask, s, upto=40); m2, _, l2, _ = fit_weighted(c, mask, s, yc, wc, upto=40)
        out["same_code_path_C"] = {"losses_equal": l1 == l2, "weights_bitwise": all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))}
        m6, _, l6, _ = fit_weighted(c, mask, s, U["y"], U["w"], upto=6); m3, o3, _, _ = fit_weighted(c, mask, s, U["y"], U["w"], upto=3)
        m33, _, l33, _ = fit_weighted(c, mask, s, U["y"], U["w"], upto=6, state={"model": m3.state_dict(), "opt": o3.state_dict()}, start=3)
        out["resume_U"] = {"loss_equal": l6[3:] == l33, "params_equal": all(torch.equal(a, b) for a, b in zip(m6.state_dict().values(), m33.state_dict().values()))}
        zz = [lr.zones(pr["base"][i], pr["alt"][i]) for i in range(len(pr["base"]))]
        out["noop_equals_base"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [False] * len(zz[i])), pr["base"][i]) for i in range(len(zz)))
        out["all_equals_alternative"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [True] * len(zz[i])), pr["alt"][i]) for i in range(len(zz)))
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cpu_spent": ledger_hours(ctx.ledger, "cpu")}
        keys = ("zone_selftest", "weights_fit_only", "zero_weight_zero_gradient", "identical_starts", "optimiser_only_router", "noop_equals_base", "all_equals_alternative")
        out["ok"] = bool(out["synthetic_utility"]["ok"] and all(out[k] for k in keys) and out["cost_directions"]["ok"] and out["same_code_path_C"]["losses_equal"]
                         and out["same_code_path_C"]["weights_bitwise"] and out["resume_U"]["loss_equal"] and out["resume_U"]["params_equal"]); ok = out["ok"]
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
