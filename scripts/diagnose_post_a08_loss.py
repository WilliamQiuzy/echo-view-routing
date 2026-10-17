#!/usr/bin/env python
"""Post-A08 loss diagnostic (docs/results/2026-10-17_post_a08_loss_diagnostic/protocol.md, frozen before this code ran).

At each seed's V01 Ordered start weights, the first 16 updates of A07's pinned training manifest (both groups) are
forwarded in train mode (A08's dropout and RNG rule) and in eval mode. No optimiser exists, nothing steps and no
checkpoint is written. A08's CE, S_A and S_M are reproduced with A08's module and with an independent numpy
reimplementation. The penalty is decomposed by edge type, class role, truncation and probability bin. The gradients of
CE and of 0.15*S with respect to the current-endpoint logits are compared by norm, cosine and their effect on the
true-class margin, and attributed to the softmax-coupled class terms. Train mode also gives parameter gradients (the new
parameters, as in warm-up, and the encoder as a start-point counterfactual). Aggregates go to the results directory;
per-window records stay on the server.

  python scripts/diagnose_post_a08_loss.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.train_video_sequence_objective as vso  # noqa: E402
import scripts.train_video_within_view as tw  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.train_joint_visual_temporal import params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.temporal.losses import segment_objective as so  # noqa: E402
from echo_routing.temporal.losses import within_view as wv  # noqa: E402

CFG = "configs/experiments/video_within_view_v1.yaml"
OUT, RES = ROOT / "runs/post_a08_loss_diagnostic", ROOT / "docs/results/2026-10-17_post_a08_loss_diagnostic"
UPDATES, COEF, ZERO, TRUNC = 16, wv.COEFFICIENT, 1e-12, wv.TRUNCATION
BINS = [0.0, 1e-6, 1e-4, 1e-2, 0.5, 1.0 + 1e-12]
BIN_NAMES = ["[0,1e-6)", "[1e-6,1e-4)", "[1e-4,1e-2)", "[1e-2,0.5)", "[0.5,1]"]
REF = {"ce": 0.11502886059133743, "L_A": 1.0649314014820694, "L_M": 1.0971463198637705}   # A08 checks.json, train mode, seed 0, update 0
ROLES = {"same": ("label", "other"), "cross": ("current", "previous", "rest")}


def numpy_values(ell: np.ndarray, lab: np.ndarray) -> dict:
    """Independent reimplementation from the float64 log-probabilities: d, S_A, S_M and CE with loops."""
    G, T, Cn = ell.shape; d = np.zeros((G, T - 1, Cn)); per_a, per_m = [], []
    for n in range(G):
        sel = 0; tot_m = 0.0
        for t in range(1, T):
            for c in range(Cn):
                d[n, t - 1, c] = min((ell[n, t, c] - ell[n, t - 1, c]) ** 2, TRUNC)
            if lab[n, t] == lab[n, t - 1]:
                sel += 1; tot_m += d[n, t - 1].sum()
        per_a.append(d[n].sum() / ((T - 1) * Cn)); per_m.append(tot_m / (Cn * sel) if sel else 0.0)
    ce = float(np.mean([-ell[n, t, lab[n, t]] for n in range(G) for t in range(T)]))
    return {"d": d, "S_A": float(np.mean(per_a)), "S_M": float(np.mean(per_m)), "ce": ce}


def weights(lab: np.ndarray) -> dict:
    """Each edge's weight in 0.15*S_A and 0.15*S_M (per class element)."""
    G, T = lab.shape; m = (lab[:, 1:] == lab[:, :-1]).astype(float); cnt = m.sum(1, keepdims=True)
    return {"A": np.full((G, T - 1), COEF / ((T - 1) * 5 * G)), "M": COEF * m / (5 * np.maximum(cnt, 1) * G), "m": m}


def closed_form(ell: np.ndarray, d_raw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """g[n,t,:] = sum_c 2 w delta_c (e_c - p) at current endpoints t >= 1; zero where clamped; position 0 gets nothing."""
    G, T, Cn = ell.shape; p = np.exp(ell); g = np.zeros_like(ell)
    for n in range(G):
        for t in range(1, T):
            for c in range(Cn):
                if d_raw[n, t - 1, c] <= TRUNC:
                    e = np.zeros(Cn); e[c] = 1.0; g[n, t] += 2 * w[n, t - 1] * (ell[n, t, c] - ell[n, t - 1, c]) * (e - p[n, t])
    return g


def role_of(lab: np.ndarray, n: int, t: int, c: int) -> tuple[str, str]:
    cur, prev = int(lab[n, t]), int(lab[n, t - 1])
    if cur == prev:
        return "same", ("label" if c == cur else "other")
    return "cross", ("current" if c == cur else "previous" if c == prev else "rest")


def analyse(z: torch.Tensor, lab: np.ndarray) -> dict:
    """One update's (8, 5) logits (requires_grad) and (2, 4) labels: values, decompositions and logit gradients."""
    z3 = z.view(2, 4, 5); runs = [so.reference_runs(x) for x in lab]
    ce, _ = so.objective("frame", z3, runs, np.ones(5)); ell_t = wv.log_probabilities(z3); d_t = wv.edge_penalties(ell_t); mask = wv.equal_label_mask(lab)
    sa, sm = wv.all_neighbours(d_t), wv.equal_label(d_t, mask)
    g_ce, = torch.autograd.grad(ce, z, retain_graph=True); g_a, = torch.autograd.grad(COEF * sa, z, retain_graph=True); g_m, = torch.autograd.grad(COEF * sm, z, retain_graph=True)
    ell = ell_t.detach().cpu().numpy(); zz = z3.detach().double().cpu().numpy(); nv = numpy_values(ell, lab)
    raw = (ell[:, 1:] - ell[:, :-1]) ** 2; w = weights(lab); g = {"CE": g_ce.view(2, 4, 5).double().cpu().numpy(), "A": g_a.view(2, 4, 5).double().cpu().numpy(),
                                                                    "M": g_m.view(2, 4, 5).double().cpu().numpy()}
    rec = {"ce": float(ce), "S_A": float(sa), "S_M": float(sm), "L_A": float(ce + COEF * sa), "L_M": float(ce + COEF * sm),
           "numpy_max_diff": max(abs(nv["ce"] - float(ce)), abs(nv["S_A"] - float(sa)), abs(nv["S_M"] - float(sm)), float(np.abs(nv["d"] - d_t.detach().cpu().numpy()).max())),
           "closed_form_max_diff": max(float(np.abs(closed_form(ell, raw, w[x]) - g[x]).max()) for x in ("A", "M")),
           "position0_penalty_grad_max": max(float(np.abs(g[x][:, 0]).max()) for x in ("A", "M")),
           "groups": {"native": {"selected": int(w["m"][0].sum()), "edges": 3, "empty": int(w["m"][0].sum() == 0)},
                      "constructed": {"selected": int(w["m"][1].sum()), "edges": 3, "empty": int(w["m"][1].sum() == 0)}}, "elements": [], "windows": []}
    p = np.exp(ell)
    for n in range(2):
        for t in range(1, 4):
            etype, _ = role_of(lab, n, t, 0)
            for c in range(5):
                _, role = role_of(lab, n, t, c); dv = min(raw[n, t - 1, c], TRUNC)
                rec["elements"].append({"kind": ("native", "constructed")[n], "edge": etype, "role": role, "clamped": bool(raw[n, t - 1, c] > TRUNC), "p_cur": float(p[n, t, c]),
                                        "A": float(w["A"][n, t - 1] * dv), "M": float(w["M"][n, t - 1] * dv)})
            y = int(lab[n, t]); k = int(max((c for c in range(5) if c != y), key=lambda c: zz[n, t, c]))
            win = {"kind": ("native", "constructed")[n], "edge": etype, "position": t, "margin": float(zz[n, t, y] - zz[n, t, k]), "p_true": float(p[n, t, y])}
            dm = lambda v: float(-(v[y] - v[k]))  # noqa: E731 - first-order margin change along the descent direction
            win["CE"] = {"norm": float(np.linalg.norm(g["CE"][n, t])), "dm": dm(g["CE"][n, t])}
            for x in ("A", "M"):
                gx = g[x][n, t]; nx, nc = float(np.linalg.norm(gx)), win["CE"]["norm"]
                cos = float(gx @ g["CE"][n, t] / (nx * nc)) if nx >= ZERO and nc >= ZERO else None
                terms = {}
                if raw is not None:
                    for c in range(5):
                        if raw[n, t - 1, c] <= TRUNC:
                            e = np.zeros(5); e[c] = 1.0; gc = 2 * w[x][n, t - 1] * (ell[n, t, c] - ell[n, t - 1, c]) * (e - p[n, t])
                            role = role_of(lab, n, t, c)[1]; terms[role] = terms.get(role, np.zeros(5)) + gc
                win[x] = {"norm": nx, "zero": nx < ZERO, "cos": cos, "dm": dm(gx), "terms": {r: {"norm": float(np.linalg.norm(v)), "dm": dm(v)} for r, v in terms.items()}}
            rec["windows"].append(win)
    cur = [(n, t) for n in range(2) for t in range(1, 4)]
    for x in ("A", "M"):
        a = np.concatenate([g[x][n, t] for n, t in cur]); b = np.concatenate([g["CE"][n, t] for n, t in cur]); na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        rec[f"batch_{x}"] = {"ratio": na / nb if nb >= ZERO else None, "cos": float(a @ b / (na * nb)) if na >= ZERO and nb >= ZERO else None}
    return rec


def param_grads(model, z: torch.Tensor, lab: np.ndarray) -> dict:
    """Gradients of CE, 0.15*S_A and 0.15*S_M and of the totals w.r.t. the new parameters and the encoder (start point only)."""
    z3 = z.view(2, 4, 5); ce, _ = so.objective("frame", z3, [so.reference_runs(x) for x in lab], np.ones(5))
    d = wv.edge_penalties(wv.log_probabilities(z3)); mask = wv.equal_label_mask(lab)
    losses = {"CE": ce, "A": COEF * wv.all_neighbours(d), "M": COEF * wv.equal_label(d, mask)}
    groups = {"new": list(model.new_parameters()), "encoder": list(model.encoder_parameters())}; flat = {}
    for name, loss in losses.items():
        gs = torch.autograd.grad(loss, groups["new"] + groups["encoder"], retain_graph=True, allow_unused=True)
        k = len(groups["new"]); flat[name] = {"new": torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).double() for x, p in zip(gs[:k], groups["new"])]),
                                              "encoder": torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).double() for x, p in zip(gs[k:], groups["encoder"])])}
    out = {}
    for grp in ("new", "encoder"):
        c = flat["CE"][grp]; nc = float(c.norm())
        for x in ("A", "M"):
            v = flat[x][grp]; nv = float(v.norm()); tot = c + v
            out[f"{grp}_{x}"] = {"ratio": nv / nc if nc >= ZERO else None, "cos": float(v @ c / (nv * nc)) if nv >= ZERO and nc >= ZERO else None,
                                 "cos_total_ce": float(tot @ c / (float(tot.norm()) * nc)) if nc >= ZERO else None}
    return out


def flags(model) -> dict:
    return {"training": bool(model.training), "requires_grad": [bool(p.requires_grad) for p in model.parameters()]}


def run_seed(ctx, seed: int, device) -> dict:
    m, msha = ctx.manifest(seed); state, ssha = ctx.source_state(seed); upd = vso.Updates(m, ctx.frame_paths())
    model = tw.make_model(state, device); before = {"params": params_digest(model.named_parameters()), **flags(model)}
    out = {"manifest_sha256": msha, "start_weights_sha256": ssha, "train": [], "eval": [], "params": [],
           "inputs": {"group_seq": m["group_seq"][:UPDATES].tolist(), "group_start": m["group_start"][:UPDATES].tolist()}}
    model.set_encoder_trainable(True); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model.train()
    for u in range(UPDATES):                                         # train mode: A08's dropout and RNG rule, updates in order
        b = upd(u); lab = b[3].numpy().astype(np.int64).reshape(2, 4)
        z = model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))
        rec = analyse(z, lab); rec["update"] = u; rec["params"] = param_grads(model, z, lab); out["train"].append(rec)
        del z; model.zero_grad(set_to_none=True)
    model.eval()
    for u in range(UPDATES):                                         # eval mode: no dropout, no stochastic depth
        b = upd(u); lab = b[3].numpy().astype(np.int64).reshape(2, 4)
        with torch.no_grad():
            z0 = model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))
        rec = analyse(z0.detach().clone().requires_grad_(True), lab); rec["update"] = u; out["eval"].append(rec)
    for p_, f in zip(model.parameters(), before["requires_grad"]):
        p_.requires_grad_(f)
    model.train(before["training"]); after = {"params": params_digest(model.named_parameters()), **flags(model)}
    out["state_unchanged"] = bool(before == after); del model; torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------------------------- aggregation (no ids)
def q(xs) -> dict | None:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": float(np.median(xs)), "q25": float(np.percentile(xs, 25)), "q75": float(np.percentile(xs, 75)), "mean": float(np.mean(xs))} if xs else None


def aggregate_mode(recs: list[dict]) -> dict:
    out = {"updates": len(recs), "groups": {k: {"groups": len(recs), "selected_edges": sum(r["groups"][k]["selected"] for r in recs),
                                                "edges": 3 * len(recs), "empty_groups": sum(r["groups"][k]["empty"] for r in recs)} for k in ("native", "constructed")}}
    out["values"] = {k: q([r[k] for r in recs]) for k in ("ce", "S_A", "S_M", "L_A", "L_M")}
    out["scalar"] = {x: {"penalty_share": q([COEF * r[f"S_{x}"] / r[f"L_{x}"] for r in recs]), "penalty_over_ce": q([COEF * r[f"S_{x}"] / r["ce"] for r in recs])} for x in ("A", "M")}
    out["checks"] = {"numpy_max_diff": max(r["numpy_max_diff"] for r in recs), "closed_form_max_diff": max(r["closed_form_max_diff"] for r in recs),
                     "position0_penalty_grad_max": max(r["position0_penalty_grad_max"] for r in recs)}
    els = [e for r in recs for e in r["elements"]]; dec = {}
    for x in ("A", "M"):
        tot = sum(e[x] for e in els) or 1.0; dec[x] = {"total": tot}
        for et, roles in ROLES.items():
            for role in roles:
                dec[x][f"{et}/{role}"] = sum(e[x] for e in els if e["edge"] == et and e["role"] == role) / tot
        dec[x]["clamped_share_of_penalty"] = sum(e[x] for e in els if e["clamped"]) / tot
        dec[x]["bins"] = {bn: {role: sum(e[x] for e in els if lo <= e["p_cur"] < hi and e["role"] == role) / tot for role in ("label", "other", "current", "previous", "rest")}
                          for bn, lo, hi in zip(BIN_NAMES, BINS[:-1], BINS[1:])}
    out["loss_decomposition"] = dec
    out["elements"] = {"n": len(els), "clamped": sum(e["clamped"] for e in els), "by_edge": {et: sum(e["edge"] == et for e in els) for et in ROLES},
                       "bins": {bn: sum(lo <= e["p_cur"] < hi for e in els) for bn, lo, hi in zip(BIN_NAMES, BINS[:-1], BINS[1:])}}
    wins = [w for r in recs for w in r["windows"]]; grad = {}
    for x in ("A", "M"):
        sub = {}
        for name, sel in (("all", wins), ("same", [w for w in wins if w["edge"] == "same"]), ("cross", [w for w in wins if w["edge"] == "cross"]),
                          ("native", [w for w in wins if w["kind"] == "native"]), ("constructed", [w for w in wins if w["kind"] == "constructed"])):
            nz = [w for w in sel if not w[x]["zero"]]
            sub[name] = {"windows": len(sel), "zero_norm": len(sel) - len(nz), "ratio": q([w[x]["norm"] / w["CE"]["norm"] for w in sel if w["CE"]["norm"] >= ZERO]),
                         "share_ratio_above_1": float(np.mean([w[x]["norm"] > w["CE"]["norm"] for w in sel])) if sel else None,
                         "cos": q([w[x]["cos"] for w in nz]), "conflict_share": float(np.mean([w[x]["cos"] is not None and w[x]["cos"] < 0 for w in sel])) if sel else None,
                         "opposes_margin_share": float(np.mean([w[x]["dm"] < 0 for w in sel])) if sel else None,
                         "dominant_opposing_share": float(np.mean([w[x]["dm"] < 0 and abs(w[x]["dm"]) > w["CE"]["dm"] for w in sel])) if sel else None,
                         "dm_ratio": q([w[x]["dm"] / w["CE"]["dm"] for w in sel if w["CE"]["dm"] > ZERO])}
            roles = sorted({r for w in sel for r in w[x]["terms"]})
            sub[name]["terms"] = {r: {"norm_over_penalty": q([w[x]["terms"][r]["norm"] / w[x]["norm"] for w in nz if r in w[x]["terms"]]),
                                      "dm_sum": float(sum(w[x]["terms"][r]["dm"] for w in sel if r in w[x]["terms"])),
                                      "opposes_margin_share": float(np.mean([w[x]["terms"][r]["dm"] < 0 for w in sel if r in w[x]["terms"]]))} for r in roles}
            sub[name]["dm_sum"] = {"CE": float(sum(w["CE"]["dm"] for w in sel)), x: float(sum(w[x]["dm"] for w in sel))}
        grad[x] = sub
    out["logit_gradients"] = grad
    out["batch"] = {x: {"ratio": q([r[f"batch_{x}"]["ratio"] for r in recs]), "cos": q([r[f"batch_{x}"]["cos"] for r in recs])} for x in ("A", "M")}
    out["margins"] = q([w["margin"] for w in wins]); out["p_true"] = q([w["p_true"] for w in wins])
    if "params" in recs[0]:
        out["parameter_gradients"] = {k: {"ratio": q([r["params"][k]["ratio"] for r in recs]), "cos": q([r["params"][k]["cos"] for r in recs]),
                                          "cos_total_ce": q([r["params"][k]["cos_total_ce"] for r in recs])} for k in recs[0]["params"]}
    return out


def reading(train: dict) -> dict:
    out = {}
    for x in ("A", "M"):
        med = [train[s]["logit_gradients"][x]["all"]["ratio"]["median"] for s in train]; dom = [train[s]["logit_gradients"][x]["all"]["dominant_opposing_share"] for s in train]
        r = "R1" if all(v <= 1 for v in med) else ("R3" if all(v > 0.5 for v in dom) else "R2")
        out[x] = {"reading": r, "median_ratio_by_seed": med, "dominant_opposing_share_by_seed": dom}
    return out


def later_training() -> dict:
    s = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())["training"]; out = {}
    for key, es in s.items():
        x = "A" if key.startswith("all") else "M"
        out[key] = [{"epoch": e["epoch"], "ce": e["ce"], f"S_{x}": e[f"S_{x}"], "penalty_over_ce": COEF * e[f"S_{x}"] / e["ce"], "same": e["same_label_penalty"],
                     "different": e["different_label_penalty"]} for e in es]
    return out


def main() -> None:
    t0, started = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx = tw.Ctx(load_cfg(SimpleNamespace(config=CFG, set=[])), CFG); tw.determinism(); device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); g0 = time.time(); raw = {s: run_seed(ctx, s, device) for s in ctx.seeds}; gsec = time.time() - g0
    r0 = raw[0]["train"][0]; gate = {k: {"got": r0[k], "want": v, "diff": abs(r0[k] - v)} for k, v in REF.items()}
    res = {"protocol": "docs/results/2026-10-17_post_a08_loss_diagnostic/protocol.md", "reproduction_gate": gate, "gate_ok": all(v["diff"] <= 1e-12 for v in gate.values()),
           "state_unchanged": {str(s): raw[s]["state_unchanged"] for s in raw}, "inputs": {str(s): {"manifest_sha256": raw[s]["manifest_sha256"], "start_weights_sha256": raw[s]["start_weights_sha256"],
                                                                                                   "updates": f"0-{UPDATES - 1}"} for s in raw},
           "train": {str(s): aggregate_mode(raw[s]["train"]) for s in raw}, "eval": {str(s): aggregate_mode(raw[s]["eval"]) for s in raw}}
    res["readings"] = reading(res["train"]); res["later_training"] = later_training(); res["seconds"] = {"gpu": gsec, "total": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    (OUT / "private_windows.json").write_text(json.dumps({str(s): {"inputs": raw[s]["inputs"], **{mode: [{k: v for k, v in r.items() if k in ("update", "windows", "groups")} for r in raw[s][mode]]
                                                                                                  for mode in ("train", "eval")}} for s in raw}, default=float))
    (RES / "diagnostic.json").write_text(json.dumps(res, indent=1, default=float))
    led = OUT / "ledger.jsonl"; tw.ledger_append(led, "diagnostic run (GPU forwards and gradients at the V01 start points)", "gpu", started, gsec, bool(res["gate_ok"]), "")
    tw.ledger_append(led, "diagnostic run (CPU part: loading, decomposition, aggregation)", "cpu", started, time.time() - t0 - gsec, bool(res["gate_ok"]), "")
    print(json.dumps({"gate": gate, "gate_ok": res["gate_ok"], "state_unchanged": res["state_unchanged"], "readings": res["readings"], "gpu_seconds": gsec}, indent=1, default=float))


if __name__ == "__main__":
    main()
