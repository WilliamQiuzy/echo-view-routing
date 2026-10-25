#!/usr/bin/env python
"""A34 (configs/experiments/video_native_bag_evidence_v1.yaml): whole native recordings as bags; a copied A08 head trained
with the mean-probability bag CE (M, r = 1) or the normalised power-mean bag CE (P, r = 4). CPU only.

  --stage bags     bag tables of both pools (all native entries, nearest-centre weights) and the fixed batch plans
  --stage check    objective against NumPy / A29, invariances, Init/r1 = Base, binding, resume, hashes, timing
  --stage train    8 heads (source seeds 0 / 2 x fit folds 0 / 1 x M / P), 1,350 updates each, with fit-fold monitors
  --stage readout  after the weights are committed: one held-out readout of Init, M and P under r1 and r4
The A08 + B3 path only supplies Base and the runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                  # CPU only in this round

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.bag_objective as bo  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG = "configs/experiments/video_session_correction_v1.yaml"
SEEDS, FOLDS, ARMS, READOUTS = (0, 2), (0, 1), {"M": 1.0, "P": 4.0}, {"r1": 1.0, "r4": 4.0}
U, WARM, BATCH = 1350, 135, 16
now, write_json, ledger_append = t31.now, t31.write_json, t31.ledger_append
torch.set_num_threads(1)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["native_bag_evidence"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG)


def staged(ctx: Ctx, job: str, fn) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, job, "cpu", started, time.time() - t0, ok, note)


def private_ids(ctx: Ctx) -> set:
    pools = [t31.load_pool(ctx.a31, f) for f in FOLDS]; rec = {r[0] for p in pools for q in p for r in q["refs"]}
    return rec | {session_of(v) for v in rec} | {q["id"] for p in pools for q in p}


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    flat = json.loads(json.dumps(rec, default=float)); assert_no_ids(flat, private_ids(ctx), name)
    for p in (ctx.out / name, ctx.results / name):
        p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)


# ----------------------------------------------------------------------------------------------- bags
def bag_table(ctx: Ctx, f: int) -> dict:
    """Every native entry of pool f: its request windows (fold-local rows) with nearest-centre sample weights."""
    seqs = t31.load_pool(ctx.a31, f); W = t31.pool_W(seqs); near = t27.nearest_rows(W, seqs); rows, wts, ix, y, n, ids = [], [], [], [], [], []
    for i, q in enumerate(seqs):
        if q["kind"] != "native":
            continue
        r, w = ir.interval_rows(near[i], 0, len(q["labels"])); k = len(y)
        rows.append(r); wts.append(w); ix.append(np.full(len(r), k)); y.append(int(q["labels"][0])); n.append(len(q["labels"])); ids.append(q["id"])
    return {"rows": np.concatenate(rows), "w": np.concatenate(wts), "ix": np.concatenate(ix), "y": np.asarray(y), "n_samples": np.asarray(n), "ids": ids,
            "offsets": np.cumsum([0, *[len(r) for r in rows]]), "seqs": seqs, "W": W}


def gather(B: dict, sel: np.ndarray) -> tuple:
    o = B["offsets"]; rows = np.concatenate([B["rows"][o[k]:o[k + 1]] for k in sel]); w = np.concatenate([B["w"][o[k]:o[k + 1]] for k in sel])
    ix = np.concatenate([np.full(o[k + 1] - o[k], j) for j, k in enumerate(sel)])
    return torch.from_numpy(rows), torch.from_numpy(w), torch.from_numpy(ix), torch.from_numpy(B["y"][sel])     # float64 weights: exact normalisation


def plan(s: int, f: int, n_bags: int) -> np.ndarray:
    return np.random.default_rng([20261025, 34, s, f]).integers(0, n_bags, (U, BATCH))


def bags(ctx: Ctx) -> str:
    rec = {}
    for f in FOLDS:
        B = bag_table(ctx, f); sess = [session_of(v) for v in B["ids"]]; wpb = np.diff(B["offsets"])
        rec[f"pool_{f}"] = {"bags": len(B["y"]), "per_class": np.bincount(B["y"], minlength=5).tolist(), "recordings": len(set(B["ids"])), "sessions": len(set(sess)),
                            "samples_per_bag": {"min": int(B["n_samples"].min()), "median": float(np.median(B["n_samples"])), "max": int(B["n_samples"].max())},
                            "windows_per_bag": {"min": int(wpb.min()), "median": float(np.median(wpb)), "max": int(wpb.max())},
                            "weights_sum_to_one": bool(np.allclose(np.bincount(B["ix"], weights=B["w"]), 1.0)),
                            "samples_per_bag_by_class_median": [float(np.median(B["n_samples"][B["y"] == c])) for c in range(5)]}
        for s in SEEDS:
            P_ = plan(s, f, len(B["y"])); np.save(ctx.out / f"plan_s{s}_f{f}.npy", P_); rec[f"pool_{f}"][f"plan_s{s}_sha256"] = hashlib.sha256(P_.tobytes()).hexdigest()
    publish(ctx, "bags.json", rec); print(json.dumps(rec, indent=1)); return "bag tables"


# ----------------------------------------------------------------------------------------------- heads
def emb_of(ctx: Ctx, s: int, f: int) -> dict:
    O = t31.outputs(ctx.a31, s, f); return {"emb": torch.from_numpy(O["emb"]), "logits": O["logits"], "seq": O["seq"], "centre": O["centre"], "offset": int(O["offset"])}


def head_and_opt(ctx: Ctx, s: int):
    h = ir.copied_head(*ctx.a31.a27.init(s)); return h, t31.opt_of([("classifier", list(h.parameters()), 1e-3)])


def one_step(h, opt, emb: torch.Tensor, B: dict, sel: np.ndarray, r: float, u: int) -> dict:
    t31.sched(opt, u, {"updates": U, "warmup": WARM}); rows, w, ix, y = gather(B, sel); loss = bo.bag_loss(h(emb[rows]), w, ix, y, r).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(h.parameters(), t31.CLIP)); opt.step()
    return {"u": int(u), "loss": float(loss.detach()), "grad_norm": gn}


def monitor(h, emb: torch.Tensor, B: dict, r_own: float, init_pred: np.ndarray | None) -> dict:
    """Fit-bag readings: CE under both r, own-r margin, change against Init/r1 bag predictions, gradient concentration."""
    rows, w, ix = torch.from_numpy(B["rows"]), torch.from_numpy(B["w"]), torch.from_numpy(B["ix"]); y = torch.from_numpy(B["y"]); n = len(B["y"])
    z = h(emb[rows]).detach().requires_grad_(True); out = {}
    for name, r in READOUTS.items():
        out[f"ce_{name}"] = float(bo.bag_loss(z.detach().double(), w.double(), ix, y, r).mean())
    lg = bo.bag_log_g(z.double(), w.double(), ix, n, r_own); lb = lg - torch.logsumexp(lg, -1, keepdim=True)
    other = lb.detach().clone(); other[torch.arange(n), y] = -torch.inf; out["margin_own_r"] = float((lb.detach()[torch.arange(n), y] - other.max(1).values).mean())
    bo.bag_loss(z, w, ix, y, r_own).sum().backward(); g = z.grad.norm(dim=1).numpy(); share = g / np.maximum(np.bincount(B["ix"], weights=g, minlength=n)[B["ix"]], 1e-30)
    top = np.zeros(n); np.maximum.at(top, B["ix"], share); eff = 1.0 / np.maximum(np.bincount(B["ix"], weights=share ** 2, minlength=n), 1e-30)
    out["grad_top_window_share_mean"], out["grad_effective_windows_mean"] = float(top.mean()), float(eff.mean())
    pred = bo.bag_scores(z.detach(), w, ix, n, 1.0).argmax(1).numpy(); out["bag_accuracy_r1"] = float((pred == B["y"]).mean())
    if init_pred is not None:
        wr = init_pred != B["y"]; out["vs_init_r1"] = {"init_wrong": int(wr.sum()), "fixed": int((wr & (pred == B["y"])).sum()), "broken": int((~wr & (pred != B["y"])).sum())}
        out["init_wrong_subgroup_ce_own_r"] = float(bo.bag_loss(z.detach().double(), w.double(), ix, y, r_own).numpy()[wr].mean()) if wr.any() else None
    return out


def train(ctx: Ctx) -> None:
    for s in SEEDS:
        for f in FOLDS:
            B = bag_table(ctx, f); E = emb_of(ctx, s, f)["emb"]; P_ = np.load(ctx.out / f"plan_s{s}_f{f}.npy")
            h0, _ = head_and_opt(ctx, s); init_pred = bo.bag_scores(h0(E[torch.from_numpy(B["rows"])]).detach(), torch.from_numpy(B["w"]), torch.from_numpy(B["ix"]), len(B["y"]), 1.0).argmax(1).numpy()
            for arm, r in ARMS.items():
                t0, started, ok, note = time.time(), now(), False, ""
                try:
                    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); h, o = head_and_opt(ctx, s); before = monitor(h, E, B, r, init_pred); log = []
                    for u in range(U):
                        log.append(one_step(h, o, E, B, P_[u], r, u))
                        if (u + 1) % 270 == 0:
                            print(f"[{arm}_s{s}_f{f}] update {u + 1}/{U}: loss {np.mean([x['loss'] for x in log[-270:]]):.4f}  {time.time() - t0:.1f} s", flush=True)
                    after = monitor(h, E, B, r, init_pred); torch.save({k: v.detach().clone() for k, v in h.state_dict().items()}, rd / "final.pt")
                    (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                    man = {"arm": arm, "r": r, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "plan_sha256": hashlib.sha256(P_.tobytes()).hexdigest(),
                           "monitor_before": before, "monitor_after": after, "seconds": time.time() - t0}
                    write_json(rd / "run_manifest.json", man); print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256", "seconds")}), flush=True); ok = True
                except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                    note = f"{type(e).__name__}: {str(e)[:200]}"; raise
                finally:
                    ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- check
def check(ctx: Ctx) -> str:
    res = {"config_sha256": sha256_file(ROOT / ctx.config)}; shas0 = {}
    for s in SEEDS:
        for f in FOLDS:
            shas0[f"s{s}_f{f}"] = sha256_file(ctx.a31.out / f"outputs_s{s}_f{f}.npz"); E = emb_of(ctx, s, f); h, _ = head_and_opt(ctx, s); B = bag_table(ctx, f); W, seqs = B["W"], B["seqs"]
            if not (np.array_equal(E["seq"], W["seq"]) and np.array_equal(E["centre"], W["centre"])):
                raise SystemExit("A31 outputs are not in this window order")
            z = h(E["emb"]).detach().double(); R = t31.runs(ctx.a31, s, f); pk = R["P"]; split_at = np.cumsum([len(q["labels"]) for q in seqs])[:-1]
            base = dict(zip([q["id"] for q in seqs], np.split(R["base_labels"], split_at))); ps = rr.per_sample(W, z.float().numpy(), seqs)
            seq_eq = all(np.array_equal(bo.relabel_r(base[q["id"]], ps[q["id"]], 1.0, ir.runs), base[q["id"]]) for q in seqs)
            loc = torch.from_numpy(pk["rows"] - E["offset"]); rix = torch.from_numpy(np.repeat(np.arange(len(pk["labels"])), np.diff(pk["offsets"])))
            run_eq = bool(np.array_equal(bo.bag_scores(z[loc], torch.from_numpy(pk["weights"]), rix, len(pk["labels"]), 1.0).argmax(1).numpy(), pk["base"]))
            res[f"s{s}_f{f}"] = {"head_vs_cached_logits_max_abs": float(np.abs(z.numpy() - E["logits"]).max()), "init_r1_sequences_equal_base": bool(seq_eq), "init_r1_runs_equal_base": run_eq}
    # objective on real bags (seed 0, fold 0)
    B = bag_table(ctx, 0); E = emb_of(ctx, 0, 0)["emb"]; h, _ = head_and_opt(ctx, 0); g = np.random.default_rng(1); sel = g.choice(len(B["y"]), 20, replace=False); rows, w, ix, y = gather(B, sel)
    z = h(E[rows]).detach().double(); o = B["offsets"]; mx = 0.0
    for r in (1.0, 4.0):
        got = bo.bag_loss(z, w.double(), ix, y, r).numpy()
        for j, k in enumerate(sel):
            zz = z[ix == j].numpy(); mx = max(mx, abs(got[j] - bo.numpy_loss(zz, B["w"][o[k]:o[k + 1]], int(B["y"][k]), r)))
    res["numpy_max_abs"] = mx; res["r1_vs_A29_pooled_ce_max_abs"] = float((bo.bag_loss(z, w.double(), ix, y, 1.0) - t29.pooled_ce(z, w.double(), ix, y)).abs().max())
    one = (ix == 0); res["single_window_vs_ce_max_abs"] = float(max(abs(float(bo.bag_loss(z[one][:1], torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.long), y[:1], r)[0]) - float(torch.nn.functional.cross_entropy(z[one][:1], y[:1]))) for r in (1.0, 4.0)))
    zp = torch.cat([z, z[:3]]); wp = torch.cat([w.double(), torch.zeros(3, dtype=torch.float64)]); ip = torch.cat([ix, ix[:3]]); perm = torch.randperm(len(w), generator=torch.Generator().manual_seed(2))
    ref = bo.bag_loss(z, w.double(), ix, y, 4.0)
    res["invariance_max_abs"] = {"zero_weight_padding": float((bo.bag_loss(zp, wp, ip, y, 4.0) - ref).abs().max()), "permutation": float((bo.bag_loss(z[perm], w.double()[perm], ix[perm], y, 4.0) - ref).abs().max()),
                                 "duplicate_split": float((bo.bag_loss(torch.cat([z, z[:1]]), torch.cat([w.double()[:1] / 2, w.double()[1:], w.double()[:1] / 2]), torch.cat([ix, ix[:1]]), y, 4.0) - ref).abs().max())}
    fin = []
    for zz in (torch.zeros(6, 5, dtype=torch.float64), torch.tensor([[40.0, -40, -40, -40, -40]] * 3 + [[-40.0, 40, -40, -40, -40]] * 3, dtype=torch.float64)):
        zz = zz.clone().requires_grad_(True); bo.bag_loss(zz, torch.full((6,), 1 / 6, dtype=torch.float64), torch.zeros(6, dtype=torch.long), torch.tensor([1]), 4.0).sum().backward(); fin.append(bool(torch.isfinite(zz.grad).all()))
    res["synthetic_gradients_finite"] = fin
    # binding, real updates, identical starts and batches, resume
    P_ = plan(0, 0, len(B["y"])); hm, om = head_and_opt(ctx, 0); hp, op = head_and_opt(ctx, 0)
    res["identical_init"] = all(torch.equal(a, b) for a, b in zip(hm.state_dict().values(), hp.state_dict().values())); res["identical_plans"] = bool(np.array_equal(P_, np.load(ctx.out / "plan_s0_f0.npy")))
    res["optimizer_binds_exactly"] = sorted(id(p) for gr in om.param_groups for p in gr["params"]) == sorted(id(p) for p in hm.parameters()); w0 = hm.weight.detach().clone()
    for u in range(6):
        one_step(hm, om, E, B, P_[u], 4.0, u)
    res["update_moves_weights"] = float((hm.weight.detach() - w0).abs().max())
    ha, oa = head_and_opt(ctx, 0)
    for u in range(3):
        one_step(ha, oa, E, B, P_[u], 4.0, u)
    st = {"h": {k: v.clone() for k, v in ha.state_dict().items()}, "o": oa.state_dict()}; hb, ob = head_and_opt(ctx, 0); hb.load_state_dict(st["h"]); ob.load_state_dict(st["o"])
    for u in range(3, 6):
        one_step(hb, ob, E, B, P_[u], 4.0, u)
    res["resume_3_plus_3_vs_6_max_abs"] = float(max((a - b).abs().max() for a, b in zip(hb.state_dict().values(), hm.state_dict().values())))
    t0 = time.time()
    for u in range(50):
        one_step(hp, op, E, B, P_[u], 4.0, u)
    spu = (time.time() - t0) / 50; res["timing"] = {"seconds_per_update": spu, "projected_training_hours_8_heads": 8 * U * spu * 1.2 / 3600}
    res["a31_outputs_unchanged"] = all(sha256_file(ctx.a31.out / f"outputs_s{s}_f{f}.npz") == shas0[f"s{s}_f{f}"] for s in SEEDS for f in FOLDS); res["outputs_sha256"] = shas0
    cells = [res[f"s{s}_f{f}"] for s in SEEDS for f in FOLDS]
    res["ok"] = bool(all(c["head_vs_cached_logits_max_abs"] <= 1e-4 and c["init_r1_sequences_equal_base"] and c["init_r1_runs_equal_base"] for c in cells) and res["numpy_max_abs"] < 1e-9
                     and res["r1_vs_A29_pooled_ce_max_abs"] < 1e-12 and res["single_window_vs_ce_max_abs"] < 1e-12 and max(res["invariance_max_abs"].values()) < 1e-12 and all(fin)
                     and res["identical_init"] and res["identical_plans"] and res["optimizer_binds_exactly"] and res["update_moves_weights"] > 0 and res["resume_3_plus_3_vs_6_max_abs"] == 0.0 and res["a31_outputs_unchanged"])
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps(res, indent=1)[:4000])
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{spu:.4f} s/update"


# ----------------------------------------------------------------------------------------------- readout (once, after the weights are committed)
def load_head(ctx: Ctx, arm: str, s: int, f: int):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    h, _ = head_and_opt(ctx, s); h.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return h.eval()


def tables(z: torch.Tensor, r: float, D: dict) -> dict:
    """Held-out whole bags, P runs and complete sequences of one model under readout r."""
    import scripts.diagnose_a5c_evidence as d32  # noqa: E402
    B, pk, seqs, W = D["B"], D["pk"], D["seqs"], D["W"]
    bp = bo.bag_scores(z[torch.from_numpy(B["rows"])], torch.from_numpy(B["w"]), torch.from_numpy(B["ix"]), len(B["y"]), r).argmax(1).numpy(); conf = np.zeros((5, 5), int); np.add.at(conf, (B["y"], bp), 1)
    f1 = [2 * conf[c, c] / max(1, conf[c].sum() + conf[:, c].sum()) for c in range(5)]
    pp = bo.bag_scores(z[D["loc"]], torch.from_numpy(pk["weights"]), D["rix"], len(pk["labels"]), r).numpy(); pred = pp.argmax(1); y, b = pk["labels"], pk["base"]; M = D["sets"]; ok5 = (y == 3) & (b == 3)
    runs_ = {**t31.run_table(pp, pk, seqs, D["sess"]), "changes": d32.change_table(y, b, pred), "correct_A5C_kept": {"runs": int(ok5.sum()), "kept": int((ok5 & (pred == 3)).sum())},
             "A32_S1_fixed": {"runs": int(M["S1"].sum()), "fixed": int((M["S1"] & (pred == 3)).sum())}, "A32_S2_kept": {"runs": int(M["S2"].sum()), "kept": int((M["S2"] & (pred == 3)).sum())}}
    ps = rr.per_sample(W, z.float().numpy(), seqs); preds = {q["id"]: bo.relabel_r(D["base"][q["id"]], ps[q["id"]], r, ir.runs) for q in seqs}; sr = t31.seq_rows(D["a8cfg"], D["P"], preds, D["ev"])
    return {"whole_bags": {"bags": int(len(B["y"])), "accuracy": float((bp == B["y"]).mean()), "macro_f1": float(np.mean(f1)), "confusion": conf.tolist(), "A5C_read_as_A4C": int(conf[3, 2]), "A4C_read_as_A5C": int(conf[2, 3])},
            "runs": runs_, "sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr), "pred": pred}


def readout(ctx: Ctx) -> str:
    if (ctx.out / "readout.json").exists():
        raise SystemExit("the held-out readout exists; it runs once")
    a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); sp = np.load(ROOT / "runs/a5c_source_external_evidence_v1/sets_private.npy", allow_pickle=True).item(); rec = {"cells": {}, "oof": {}}; priv = {}
    for s in SEEDS:
        acc = {}
        for f in FOLDS:
            h = 1 - f; E = emb_of(ctx, s, h); B = bag_table(ctx, h); R = t31.runs(ctx.a31, s, h); pk = R["P"]; seqs, W = B["seqs"], B["W"]; split_at = np.cumsum([len(q["labels"]) for q in seqs])[:-1]
            if [(int(i), int(a), int(b)) for i, a, b in zip(pk["seq"], pk["a"], pk["b"])] != sp[s][h]["key"]:
                raise SystemExit("A32 sets are not on these runs")
            P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}
            D = {"B": B, "pk": pk, "seqs": seqs, "W": W, "sets": sp[s][h]["masks"], "sess": np.load(ctx.a31.out / "data" / f"run_sessions_s{s}_f{h}.npy", allow_pickle=True), "a8cfg": a8.cfg, "P": P,
                 "ev": bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"), "base": dict(zip([q["id"] for q in seqs], np.split(R["base_labels"], split_at))),
                 "loc": torch.from_numpy(pk["rows"] - E["offset"]), "rix": torch.from_numpy(np.repeat(np.arange(len(pk["labels"])), np.diff(pk["offsets"])))}
            heads = {"Init": head_and_opt(ctx, s)[0], **{arm: load_head(ctx, arm, s, f) for arm in ARMS}}; cell = {}
            with torch.no_grad():
                for name, hd in heads.items():
                    z = hd(E["emb"]).double()
                    for ro, r in READOUTS.items():
                        t = tables(z, r, D); priv[(s, f, name, ro)] = t.pop("pred"); cell[f"{name}/{ro}"] = t
            base_sr = t31.seq_rows(a8.cfg, P, D["base"], D["ev"]); cell["Base"] = {"sequence_row": base_sr["row"], "counts": rsh.counts_of(seqs, base_sr)}
            rec["cells"][f"s{s}_heldout_f{h}"] = cell; acc[h] = (pk, D["sess"], [seqs[i]["kind"] for i in pk["seq"]])
        y = np.concatenate([acc[h][0]["labels"] for h in FOLDS]); b = np.concatenate([acc[h][0]["base"] for h in FOLDS]); wr = b != y; ok5 = (y == 3) & (b == 3); oof = {}
        for name in ("Init", *ARMS):
            for ro in READOUTS:
                p = np.concatenate([priv[(s, 1 - h, name, ro)] for h in FOLDS]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum())
                c = [rec["cells"][f"s{s}_heldout_f{h}"][f"{name}/{ro}"] for h in FOLDS]; bs = [rec["cells"][f"s{s}_heldout_f{h}"]["Base"] for h in FOLDS]
                oof[f"{name}/{ro}"] = {"runs": int(len(y)), "base_wrong": int(wr.sum()), "fixed": fx, "broken": br, "net_fixed": fx - br, "correct_A5C_kept": int((ok5 & (p == 3)).sum()), "correct_A5C": int(ok5.sum()),
                                       "A5C_read_as_A4C_fixed": int(((y == 3) & (b == 2) & (p == 3)).sum()), "A4C_read_as_A5C_fixed": int(((y == 2) & (b == 3) & (p == 2)).sum()),
                                       "segment_f1_50_multi_mean": float(np.mean([x["sequence_row"]["segment_f1_50_multi"] for x in c])), "segment_f1_minus_base_mean": float(np.mean([x["sequence_row"]["segment_f1_50_multi"] - q["sequence_row"]["segment_f1_50_multi"] for x, q in zip(c, bs)])),
                                       "native_macro_f1_minus_base_mean": float(np.mean([x["sequence_row"]["native_macro_f1"] - q["sequence_row"]["native_macro_f1"] for x, q in zip(c, bs)])),
                                       "missed_short_minus_base": int(sum(x["counts"]["fn_short"] - q["counts"]["fn_short"] for x, q in zip(c, bs))), "short_fp_minus_base": int(sum(x["counts"]["fp_short"] - q["counts"]["fp_short"] for x, q in zip(c, bs))),
                                       "whole_bag_accuracy": [x["whole_bags"]["accuracy"] for x in c]}
        kind = np.concatenate([np.asarray(acc[h][2]) for h in FOLDS]); sess = np.concatenate([np.char.add(f"f{h}:", acc[h][1].astype(str)) for h in FOLDS]); nat = kind == "native"
        pP, pM = (np.concatenate([priv[(s, 1 - h, a, "r4")] for h in FOLDS]) for a in ("P", "M")); d = ((pP == y).astype(float) - (pM == y).astype(float))[nat]
        us = np.unique(sess[nat]); gi = np.searchsorted(us, sess[nat]); g = np.random.default_rng([20261025, 34, 77, s]); dr = g.integers(0, len(us), (2000, len(us))); wts = np.stack([np.bincount(x, minlength=len(us)) for x in dr]).astype(float)
        per = np.bincount(gi, weights=d, minlength=len(us)); cnt = np.bincount(gi, minlength=len(us)).astype(float)
        oof["native_P_minus_M_r4_run_accuracy"] = {"sessions": int(len(us)), "difference": float(d.mean()), "ci95_descriptive": np.percentile((wts @ per) / (wts @ cnt), [2.5, 97.5]).tolist()}
        rec["oof"][f"s{s}"] = oof
    o = rec["oof"]; rule = {}
    for s in SEEDS:
        x = o[f"s{s}"]; P4, M4 = x["P/r4"], x["M/r4"]
        rule[f"s{s}"] = {"P_minus_M_r4_net_positive": P4["net_fixed"] - M4["net_fixed"] > 0, "P_r4_vs_Base_net_positive": P4["net_fixed"] > 0, "P_r4_segment_f1_above_Base": P4["segment_f1_minus_base_mean"] > 0,
                         "correct_A5C_kept_ge_Base_minus_5": P4["correct_A5C_kept"] >= P4["correct_A5C"] - 5, "missed_short_le_Base_plus_3": P4["missed_short_minus_base"] <= 3}
    rec["decision"] = {"rule": rule, "next_step_rule_met": all(all(v.values()) for v in rule.values())}
    np.save(ctx.out / "readout_private.npy", np.array({str(k): v for k, v in priv.items()}, dtype=object), allow_pickle=True); publish(ctx, "readout.json", rec)
    print(json.dumps({s: {k: {kk: v[kk] for kk in ("fixed", "broken", "net_fixed", "correct_A5C_kept", "segment_f1_minus_base_mean")} for k, v in o[s].items() if "/" in k} for s in o}, indent=1)); print(json.dumps(rec["decision"], indent=1))
    return "held-out readout"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("bags", "check", "train", "readout"), required=True)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        return train(ctx)
    staged(ctx, f"{a.stage} (CPU)", {"bags": lambda: bags(ctx), "check": lambda: check(ctx), "readout": lambda: readout(ctx)}[a.stage])


if __name__ == "__main__":
    main()
