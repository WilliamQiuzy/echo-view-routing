#!/usr/bin/env python
"""Checks before the queue of the video dense supervision round (plan Tasks 1-3; run through
scripts/train_video_dense_supervision.py --check). Training data only; no development data is read.

1. Pairing: every seed's DenseMViT loads the source state strictly with the TargetMViT names, counts and digest (no
   parameter added); the grid, query strides, stem time geometry, anchors and the real norm-output shape; the label files.
2. The 16 check windows of the seed-0 manifest (V01's rule): pixels equal the manifest digests, the marker is the middle
   target slot, the target is the manifest's, the aligned labels are the SC labels at the anchors and the shuffled labels
   are the aligned ones under the window's fixed permutation.
3. Eval: the CLS logits equal the source TargetMViT's (maximum difference 0); the time logits equal a recomputation from
   the source model's norm output (<= 1e-5); the outputs do not change when the labels do.
4. Hooks never outlive a forward, also after an exception; frozen and unfrozen gradients of the aligned loss reach the
   marker, the shared head and projection, and the encoder only when unfrozen.
5. Arm pairing in train mode (same state and RNG): identical CLS and time logits before the loss and identical torch RNG
   states after the step; on native windows the aligned and shuffled losses and gradients are identical (CPU, one thread).
6. 100-update smokes per arm (encoder frozen; no score gate); V01's rule for reproducibility and resume (6 updates and
   3 + 3 within loss 1e-4 and parameters 1e-5) and for the unfreeze boundary on a short schedule.
7. Timing per arm in the unfrozen state on formal batches, the projected round cost against the 8-hour cap, and the
   input-token figure.
Writes runs/video_dense_supervision_v1/checks/check.json, its id-free copy in the results directory and a ledger line.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.video_dense_supervision as vds
import scripts.prepare_video_dense_supervision as pd
import scripts.train_video_dense_supervision as tv
from echo_routing.evaluate.failure_units import session_of
from scripts.check_raw_video_context import check_windows, group_diff, max_diff
from scripts.prepare_fold_output_correction import assert_no_ids
from scripts.train_joint_visual_temporal import params_digest

SEED = 0
GROUP = tv.GROUP


# ----------------------------------------------------------------------------------------------- models and inputs
def source_model(state: dict, device) -> tm.TargetMViT:
    m = tm.build_model(None, tm.initial_new_state(0)); m.load_state_dict(state, strict=True)
    return m.to(device).eval()


def batches(inp: tuple, device):
    u8, slots, target = inp[:3]
    return tm.to_rgb(u8.to(device)), tm.marker_planes(slots.to(device)), target.to(device)


def pairing(ctx, device) -> dict:
    out = {}
    for s in ctx.seeds:
        state, _ = ctx.source_state(s); base = source_model(state, "cpu"); m = tv.make_model(state, "cpu"); aux, asha = ctx.aux(s)
        same = (list(m.state_dict()) == list(base.state_dict()) and sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in base.parameters())
                and params_digest(m.named_parameters()) == params_digest(base.named_parameters()))
        multiset = bool(np.array_equal(np.sort(aux["aligned"], -1), np.sort(aux["shuffled"], -1)))
        perm_ok = bool(np.array_equal(pd.shuffled_labels(aux["aligned"], aux["perm"]), aux["shuffled"]))
        out[str(s)] = {"same_start_no_new_parameters": bool(same), "parameters": int(sum(p.numel() for p in m.parameters())), "aux_sha256": asha,
                       "label_multiset_kept": multiset, "shuffled_equals_permuted_aligned": perm_ok, "marker_abs_max": float(state["marker_stem.weight"].abs().max())}
    state = ctx.source_state(SEED)[0]; m = tv.make_model(state, "cpu"); base = source_model(state, device); shape = {}
    h = base.encoder.norm.register_forward_hook(lambda _m, _i, o: shape.__setitem__("x", list(o.shape)))
    try:
        with torch.no_grad():
            base(torch.zeros(2, 3, 16, 224, 224, device=device), tm.marker_planes(torch.tensor([8, 9], device=device)))
    finally:
        h.remove()
    ok = all(v["same_start_no_new_parameters"] and v["label_multiset_kept"] and v["shuffled_equals_permuted_aligned"] and v["marker_abs_max"] > 0 for v in out.values())
    return {"seeds": out, "grid": list(m.grid), "query_strides": [list(x) for x in m.query_strides], "stem_time": list(m.stem_time), "anchors": m.anchors,
            "norm_output_shape": shape["x"], "ok": bool(ok and list(m.grid) == [8, 7, 7] and m.anchors == list(vds.ANCHORS) and shape["x"] == [2, 393, 768])}


def input_checks(ctx, m: dict, pairs: list, inp: tuple) -> dict:
    aux, _ = ctx.aux(SEED); sc = pd.load_npz(pd.pinned(ctx.root, ctx.vd["source"]["sc_manifests"]["files"][SEED]))
    fails = {"digest": 0, "marker_slot": 0, "target": 0, "aligned_labels": 0, "shuffled_labels": 0}
    for k, (u, w) in enumerate(pairs):
        fails["digest"] += int(not np.array_equal(inp[3][k].numpy(), m["digest_ordered"][u, w]))
        fails["marker_slot"] += int(int(inp[1][k]) != tm.marker_slot(m["s"][u, w], int(m["c"][u, w])))
        fails["target"] += int(int(inp[2][k]) != int(m["target"][u, w]))
        fails["aligned_labels"] += int(not np.array_equal(aux["aligned"][u, w], sc["labels"][u, w][list(vds.ANCHORS)]))
        want = np.random.default_rng([pd.PERM_BASE, SEED, u, w]).permutation(8)
        fails["shuffled_labels"] += int(not np.array_equal(aux["shuffled"][u, w], aux["aligned"][u, w][want]))
    return {"windows": len(pairs), "failures": fails, "same_inputs_for_every_arm": "one decoded batch feeds all three arms", "ok": not any(fails.values())}


# ----------------------------------------------------------------------------------------------- equivalence and hooks
@torch.no_grad()
def equivalence(ctx, inp: tuple, device) -> dict:
    state, _ = ctx.source_state(SEED); base = source_model(state, device); m = tv.make_model(state, device).eval(); rgb, mk, _ = batches(inp, device)
    d_cls = d_time = 0.0; repeat = True
    for i in (0, 8):
        seen = {}
        h = base.encoder.norm.register_forward_hook(lambda _m, _i, o: seen.__setitem__("x", o.clone()))
        try:
            want_cls = base(rgb[i:i + 8], mk[i:i + 8])
        finally:
            h.remove()
        want_time = base.classifier(base.encoder.head(seen["x"][:, 1:].reshape(8, 8, 7, 7, 768).mean((2, 3))))
        z_cls, z_time = m.forward_with_time(rgb[i:i + 8], mk[i:i + 8]); again = m.forward_with_time(rgb[i:i + 8], mk[i:i + 8])
        d_cls = max(d_cls, float((z_cls - want_cls).abs().max())); d_time = max(d_time, float((z_time - want_time).abs().max()))
        repeat &= bool(torch.equal(z_cls, again[0]) and torch.equal(z_time, again[1]))
    c = ctx.vd["checks"]
    return {"cls_max_abs": d_cls, "time_max_abs": d_time, "labels_never_enter_the_forward": repeat,
            "ok": d_cls <= float(c["cls_parity_max_abs"]) and d_time <= float(c["time_recompute_max_abs"]) and repeat}


def hooks_and_grads(ctx, inp: tuple, device) -> dict:
    state, _ = ctx.source_state(SEED); rgb, mk, y = batches(inp, device); aux, _ = ctx.aux(SEED)
    m = tv.make_model(state, device).eval(); blk = m.encoder.blocks[3]; blk.forward = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("check"))
    try:
        m.forward_with_time(rgb[:4], mk[:4]); raised = False
    except RuntimeError:
        raised = True
    finally:
        del blk.forward
    clean = not m.encoder.norm._forward_hooks and not m.encoder.conv_proj._forward_hooks
    lab = torch.from_numpy(aux["aligned"][0].astype(np.int64)).to(device)   # any labels serve the gradient check
    m.train(); m.set_encoder_trainable(False); m.zero_grad(set_to_none=True); z, t = m.forward_with_time(rgb[:8], mk[:8]); vds.dense_loss(z, t, y[:8], lab, 0.5)[0].backward()
    frozen = {"marker": float(m.marker_stem.weight.grad.abs().max()), "head": float(m.classifier.weight.grad.abs().max()),
              "encoder_with_grad": int(sum(p.grad is not None for p in m.encoder_parameters()))}
    m.set_encoder_trainable(True); m.zero_grad(set_to_none=True); z, t = m.forward_with_time(rgb[:8], mk[:8]); vds.dense_loss(z, t, y[:8], lab, 0.5)[0].backward()
    g = [p.grad for p in m.encoder_parameters()]
    unfrozen = {"projection": float(m.encoder.head[-1].weight.grad.abs().max()), "none": sum(x is None for x in g),
                "nonfinite": sum(x is not None and not bool(torch.isfinite(x).all()) for x in g)}
    ok = raised and clean and frozen["marker"] > 0 and frozen["head"] > 0 and frozen["encoder_with_grad"] == 0 and unfrozen["projection"] > 0 and unfrozen["nonfinite"] == 0
    return {"exception_raised": raised, "hooks_clean": clean, "frozen": frozen, "unfrozen": unfrozen, "ok": bool(ok)}


def arm_pairing(ctx, m_rv: dict, pairs: list, inp: tuple, device) -> dict:
    """Train mode, same state and RNG: identical logits before the loss and RNG states after the step across the arms
    (GPU, formal batch); on native windows aligned and shuffled losses and gradients identical (CPU, one thread)."""
    state, _ = ctx.source_state(SEED); aux, _ = ctx.aux(SEED); o = ctx.o; res = {}
    batch = (0, inp[0][:8], inp[1][:8], inp[2][:8], inp[3][:8]); us = [p[0] for p in pairs[:8]]; ws = [p[1] for p in pairs[:8]]
    for arm in vds.ARMS:
        m = tv.make_model(state, device); opt = tv.make_optimizer(m, o); lrs = tv.apply_schedule(m, opt, 0, ctx.warmup, ctx.total, o)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        rec = tv.train_step(m, opt, batch, tv.arm_labels(aux, arm)[us, ws], aux["aligned"][us, ws], tv.arm_lambda(ctx.vd, arm), lrs, o, device, keep=True)
        res[arm] = (rec["outputs"][0], rec["outputs"][1], torch.get_rng_state(), torch.cuda.get_rng_state_all(), rec["loss"])
        del m, opt; torch.cuda.empty_cache()
    ref = res["center_only"]
    same = all(torch.equal(ref[0], res[a][0]) and torch.equal(ref[1], res[a][1]) and torch.equal(ref[2], res[a][2])
               and all(torch.equal(x, y) for x, y in zip(ref[3], res[a][3])) for a in vds.ARMS)
    native = [k for k, (u, w) in enumerate(pairs) if int(m_rv["kind"][u, w]) == 0][:2]
    n = torch.get_num_threads(); torch.set_num_threads(1); out = []
    try:
        rgb, mk, y = (x.cpu() for x in batches(tuple(t[native] for t in inp[:3]), "cpu"))
        for arm in ("aligned_dense", "shuffled_dense"):
            m = tv.make_model(state, "cpu"); m.train(); m.set_encoder_trainable(False); m.zero_grad(set_to_none=True)
            lab = torch.from_numpy(tv.arm_labels(aux, arm)[[pairs[k][0] for k in native], [pairs[k][1] for k in native]].astype(np.int64))
            torch.manual_seed(SEED); z, t = m.forward_with_time(rgb, mk); loss, _ = vds.dense_loss(z, t, y, lab, 0.5); loss.backward()
            out.append((float(loss.detach()), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
    finally:
        torch.set_num_threads(n)
    native_same = out[0][0] == out[1][0] and out[0][1].keys() == out[1][1].keys() and all(torch.equal(out[0][1][k], out[1][1][k]) for k in out[0][1])
    return {"logits_and_rng_identical_across_arms": bool(same), "losses": {a: res[a][4] for a in vds.ARMS}, "native_windows": len(native),
            "native_aligned_and_shuffled_identical": bool(native_same), "ok": bool(same and native_same and len(native) == 2)}


# ----------------------------------------------------------------------------------------------- short training runs
class Updates:
    """Update u's decoded batch (digest-checked) and its labels, cached."""

    def __init__(self, m: dict, aux: dict, fp) -> None:
        self.m, self.aux, self.fp, self.cache = m, aux, fp, {}

    def __call__(self, u: int):
        if u not in self.cache:
            out = tv.rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(tv.WINDOWS)])
            tv.check_digests(self.m, GROUP, u, out[3]); self.cache[u] = (u, *out)
        return self.cache[u]


def run_updates(ctx, upd: Updates, arm: str, us: list[int], warmup: int, total: int, device, state_file=None) -> dict:
    state, _ = ctx.source_state(SEED); model = tv.make_model(state, device); opt = tv.make_optimizer(model, ctx.o); lam = tv.arm_lambda(ctx.vd, arm)
    labels = tv.arm_labels(upd.aux, arm); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if state_file is not None:
        tv.load_resume(state_file, model, opt, "check")
    enc0 = params_digest(model.encoder.named_parameters()); recs = []
    for u in us:
        lrs = tv.apply_schedule(model, opt, u, warmup, total, ctx.o)
        s = tv.train_step(model, opt, upd(u), labels[u], upd.aux["aligned"][u], lam, lrs, ctx.o, device); s["encoder_digest"] = params_digest(model.encoder.named_parameters())
        recs.append(s)
    return {"model": model, "opt": opt, "records": recs, "encoder_start": enc0}


def roundtrip(ctx, run: dict, tmp: Path, device) -> dict:
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all(); tv.save_resume(tmp, run["model"], run["opt"], 0, "check", [])
    state, _ = ctx.source_state(SEED); model = tv.make_model(state, device); opt = tv.make_optimizer(model, ctx.o)
    torch.manual_seed(12345); torch.cuda.manual_seed_all(12345); tv.load_resume(tmp, model, opt, "check"); tmp.unlink(missing_ok=True)
    a, b, sa, sb = run["model"].state_dict(), model.state_dict(), run["opt"].state_dict(), opt.state_dict()
    weights = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    optim = sa["param_groups"] == sb["param_groups"] and sa["state"].keys() == sb["state"].keys() and all(
        torch.equal(torch.as_tensor(sa["state"][i][f]).cpu(), torch.as_tensor(sb["state"][i][f]).cpu()) for i in sa["state"] for f in sa["state"][i])
    rng = torch.equal(torch.get_rng_state(), cpu) and all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), cuda))
    return {"weights": bool(weights), "optimiser": bool(optim), "rng": bool(rng), "ok": bool(weights and optim and rng)}


def reproducibility(ctx, upd: Updates, arm: str, device, tmp: Path) -> dict:
    c = ctx.vd["checks"]; us = list(range(int(c["reproducibility"]["updates"])))
    a = run_updates(ctx, upd, arm, us, ctx.warmup, ctx.total, device); b = run_updates(ctx, upd, arm, us, ctx.warmup, ctx.total, device)
    half = run_updates(ctx, upd, arm, us[:3], ctx.warmup, ctx.total, device); tv.save_resume(tmp, half["model"], half["opt"], 3, "check", [])
    rest = run_updates(ctx, upd, arm, us[3:], ctx.warmup, ctx.total, device, state_file=tmp); tmp.unlink(missing_ok=True)
    la, lb, lc = ([r["loss"] for r in x] for x in (a["records"], b["records"], half["records"] + rest["records"]))
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lb)), "params_max_abs": max_diff(a["model"], b["model"])}
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lc)), "params_max_abs": max_diff(a["model"], rest["model"])}
    enc_same = all(r["encoder_digest"] == a["encoder_start"] for r in a["records"])
    ok = (rep["loss_max_abs"] <= float(c["reproducibility"]["loss_abs"]) and rep["params_max_abs"] <= float(c["reproducibility"]["params_abs"])
          and res["loss_max_abs"] <= float(c["resume"]["loss_abs"]) and res["params_max_abs"] <= float(c["resume"]["params_abs"]) and enc_same)
    return {"losses": la, "reproducibility": rep, "resume": res, "encoder_unchanged_in_warmup": enc_same, "ok": bool(ok)}


def boundary(ctx, upd: Updates, arm: str, device, tmp: Path) -> dict:
    b = ctx.vd["checks"]["boundary"]; w, t, k = int(b["warmup"]), int(b["total"]), int(b["resume_at"]); us = list(range(t))
    d = run_updates(ctx, upd, arm, us, w, t, device); f = run_updates(ctx, upd, arm, us, w, t, device)
    half = run_updates(ctx, upd, arm, us[:k], w, t, device); tv.save_resume(tmp, half["model"], half["opt"], k, "check", [])
    e = run_updates(ctx, upd, arm, us[k:], w, t, device, state_file=tmp); tmp.unlink(missing_ok=True)
    want = [tv.schedule(u, w, t, ctx.o) for u in us]; got = [(r["lr_encoder"], r["lr_new"]) for r in d["records"]]
    frozen = [r["encoder_digest"] == d["encoder_start"] for r in d["records"]]
    ld, le, lf = ([r["loss"] for r in x] for x in (d["records"], half["records"] + e["records"], f["records"]))
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, le)), "by_group": group_diff(d["model"], e["model"])}
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, lf)), "by_group": group_diff(d["model"], f["model"])}
    trip = roundtrip(ctx, d, tmp, device)
    bound = {"encoder": float(b["adam_step_bound"]) * sum(x[0] for x in want), "new": float(b["adam_step_bound"]) * sum(x[1] for x in want)}
    within = all(x["by_group"][g]["max_abs"] <= bound[g] for x in (res, rep) for g in bound)
    tol = float(ctx.vd["checks"]["resume"]["loss_abs"])
    ok = got == want and all(frozen[:w]) and not any(frozen[w:]) and trip["ok"] and res["loss_max_abs"] <= tol and rep["loss_max_abs"] <= tol and within
    return {"lrs": got, "encoder_unchanged_after_update": frozen, "state_roundtrip": trip, "resume": res, "repeat": rep, "parameter_bound": bound,
            "within_bound": within, "ok": bool(ok)}


def smoke(ctx, inp: tuple, pairs: list, arm: str, device) -> dict:
    sc = ctx.vd["checks"]["smoke"]; state, _ = ctx.source_state(SEED); aux, _ = ctx.aux(SEED); m = tv.make_model(state, device); m.set_encoder_trainable(False)
    opt = torch.optim.AdamW(m.new_parameters(), lr=float(sc["lr_new"]), betas=tuple(float(x) for x in ctx.o["betas"]), eps=float(ctx.o["eps"]),
                            weight_decay=float(ctx.o["weight_decay"]))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); enc0 = params_digest(m.encoder.named_parameters()); new0 = [p.detach().clone() for p in m.new_parameters()]
    rgb, mk, y = batches(inp, device); lam = tv.arm_lambda(ctx.vd, arm)
    lab = torch.from_numpy(tv.arm_labels(aux, arm)[[p[0] for p in pairs], [p[1] for p in pairs]].astype(np.int64)).to(device)
    losses, times, grads, t0 = [], [], [], time.time()
    for i in range(int(sc["updates"])):
        sl = slice(0, 8) if i % 2 == 0 else slice(8, 16); m.train()
        z, t = m.forward_with_time(rgb[sl], mk[sl]); loss, parts = vds.dense_loss(z, t, y[sl], lab[sl], lam); opt.zero_grad(set_to_none=True); loss.backward()
        grads.append(float(torch.nn.utils.clip_grad_norm_(m.new_parameters(), float(ctx.o["grad_clip"])))); opt.step(); losses.append(float(loss.detach())); times.append(parts["time"])
    m.eval()
    with torch.no_grad():
        outs = [m.forward_with_time(rgb[i:i + 8], mk[i:i + 8]) for i in (0, 8)]
    pred = torch.cat([o[0].argmax(1) for o in outs]); tpred = torch.cat([o[1].argmax(-1) for o in outs])
    enc_same = params_digest(m.encoder.named_parameters()) == enc0; changed = all(not torch.equal(p.detach(), q) for p, q in zip(m.new_parameters(), new0) if p.ndim > 1)
    finite = bool(np.isfinite(losses).all() and np.isfinite(grads).all())
    return {"arm": arm, "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])), "time_loss_first10": float(np.mean(times[:10])),
            "time_loss_last10": float(np.mean(times[-10:])), "grad_norm_first10": float(np.mean(grads[:10])), "grad_norm_last10": float(np.mean(grads[-10:])),
            "cls_accuracy_16": float((pred == y).float().mean()), "time_accuracy_own_labels": float((tpred == lab).float().mean()),
            "encoder_unchanged": enc_same, "marker_and_head_changed": changed, "seconds": time.time() - t0, "ok": finite and enc_same and changed}


def timing(ctx, m: dict, aux: dict, fp, arm: str, device, workers: int) -> dict:
    tc = ctx.vd["checks"]["timing"]; n_w, n_t = int(tc["warmup_updates"]), int(tc["timed_updates"]); state, _ = ctx.source_state(SEED)
    model = tv.make_model(state, device); opt = tv.make_optimizer(model, ctx.o); model.set_encoder_trainable(True); lam = tv.arm_lambda(ctx.vd, arm); labels = tv.arm_labels(aux, arm)
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"]); lrs = (float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"]))
    torch.manual_seed(SEED); torch.cuda.reset_peak_memory_stats(device); stamps = []; torch.cuda.synchronize(); prev = time.time()
    for batch in tv.loader(m, GROUP, fp, list(range(n_w + n_t)), workers):
        u = int(batch[0]); tv.check_digests(m, GROUP, u, batch[4]); tv.train_step(model, opt, batch, labels[u], aux["aligned"][u], lam, lrs, ctx.o, device)
        torch.cuda.synchronize(); now = time.time(); stamps.append(now - prev); prev = now
    timed = stamps[n_w:]
    return {"arm": arm, "seconds_per_update_mean": float(np.mean(timed)), "seconds_per_update_max": float(np.max(timed)),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20, "hours_per_run": float(np.mean(timed)) * ctx.total / 3600}


def token_figure(inp: tuple, labels: np.ndarray, k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames, slot = inp[0][k].numpy(), int(inp[1][k]); names = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
    fig = plt.figure(figsize=(17, 4.6)); gs = fig.add_gridspec(2, 16, height_ratios=[1.25, 1])
    for i in range(16):
        a = fig.add_subplot(gs[0, i]); a.imshow(frames[i][::4, ::4]); a.set_xticks([]); a.set_yticks([]); a.set_title(f"slot {i}", fontsize=7)
        for sp in a.spines.values():
            sp.set_edgecolor("#d62728" if i == slot else ("#4c72b0" if i in vds.ANCHORS else "#bbbbbb")); sp.set_linewidth(3 if i == slot else (1.5 if i in vds.ANCHORS else 0.5))
    b = fig.add_subplot(gs[1, :]); b.axis("off")
    for t, a_ in enumerate(vds.ANCHORS):
        b.text(t / 8 + 0.01, 0.55, f"time token {t}\nanchor slot {a_}\naligned label {names[int(labels[t])]}", fontsize=8, va="center")
    fig.suptitle(f"One mixed-view training window (red: marker slot {slot}; blue: the 8 anchor slots). The final norm holds 1 + 8 x 7 x 7 tokens: the CLS token "
                 "gives the output, and each time position's 49 tokens are averaged and read by the shared head for the auxiliary loss.", fontsize=9, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def run_checks(ctx, device, workers: int) -> None:
    """Every attempt, finished or not, adds its GPU hours to the round's ledger."""
    t0 = time.time(); started = tv.now(); ok = False; note = ""
    try:
        ok = checks(ctx, device, workers, t0)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tv.ledger_append(ctx.ledger, "check", started, (time.time() - t0) / 3600, ok, note)
    if not ok:
        raise SystemExit("a hard check failed; see check.json")


def checks(ctx, device, workers: int, t0: float) -> bool:
    out = ctx.out / "checks"; out.mkdir(parents=True, exist_ok=True); res_dir = ctx.root / ctx.vd["outputs"]["results_dir"]; res_dir.mkdir(parents=True, exist_ok=True)
    m, _ = ctx.manifest(SEED); aux, _ = ctx.aux(SEED); fp = ctx.frame_paths(); rec: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = check_windows(m); pairs = [(c["u"], c["w"]) for c in chosen]; inp = tv.rvc.window_inputs(m, m["order_ordered"], fp, pairs)
        rec["windows"] = [{**c, "kind": int(m["kind"][c["u"], c["w"]]), "marker": int(m["marker"][c["u"], c["w"]]), "target_slots": int(m["n_target_slots"][c["u"], c["w"]]),
                           "aligned_label_classes": int(len(set(aux["aligned"][c["u"], c["w"]].tolist())))} for c in chosen]
        rec["pairing"] = pairing(ctx, device); rec["inputs"] = input_checks(ctx, m, pairs, inp); rec["equivalence"] = equivalence(ctx, inp, device)
        rec["hooks_and_gradients"] = hooks_and_grads(ctx, inp, device); rec["arm_pairing"] = arm_pairing(ctx, m, pairs, inp, device)
        print("pairing, inputs, equivalence, hooks, gradients and arm pairing done", flush=True)
        rec["smoke"] = {a: smoke(ctx, inp, pairs, a, device) for a in vds.ARMS}; print("smokes done", flush=True)
        upd = Updates(m, aux, fp)
        rec["reproducibility"] = {a: reproducibility(ctx, upd, a, device, out / "tmp_resume.pt") for a in vds.ARMS}
        rec["boundary"] = {a: boundary(ctx, upd, a, device, out / "tmp_resume.pt") for a in vds.ARMS}; print("reproducibility, resume and boundary done", flush=True)
        rec["timing"] = {a: timing(ctx, m, aux, fp, a, device, workers) for a in vds.ARMS}
    rec["nondeterministic_operator_warnings"] = sorted({str(w.message).split("\n")[0][:300] for w in caught if "determinis" in str(w.message).lower()})
    spent_before = tv.ledger_hours(ctx.ledger); spent = (time.time() - t0) / 3600; hours = max(t["hours_per_run"] for t in rec["timing"].values()); b = ctx.vd["budget"]
    rec["budget"] = {"ledger_hours_before": spent_before, "check_hours_this_run": spent, "hours_per_run_max": hours, "runs": len(ctx.vd["queue"]),
                     "evaluation_allowance_hours": float(b["evaluation_allowance_hours"]), "cap_hours": float(b["gpu_hours_max"])}
    rec["budget"]["projected_total_hours"] = spent_before + spent + len(ctx.vd["queue"]) * hours + float(b["evaluation_allowance_hours"])
    rec["budget"]["ok"] = rec["budget"]["projected_total_hours"] <= rec["budget"]["cap_hours"]
    parts = ("pairing", "inputs", "equivalence", "hooks_and_gradients", "arm_pairing", "budget")
    rec["ok"] = bool(all(rec[p]["ok"] for p in parts) and all(v["ok"] for v in rec["smoke"].values()) and all(v["ok"] for v in rec["reproducibility"].values())
                     and all(v["ok"] for v in rec["boundary"].values()))
    rec["identities"] = {tv.run_key(a, s): tv.identity(ctx, a, s, ctx.manifest(s)[1], ctx.aux(s)[1], ctx.vd["source"]["weights"][s]["sha256"])
                         for s in ctx.seeds for a in vds.ARMS}
    rec["code"] = tv.code_digest(); rec["seconds"] = time.time() - t0
    mixed = [i for i, c in enumerate(chosen) if int(m["kind"][c["u"], c["w"]]) == 1]
    k = next((i for i in mixed if len(set(aux["aligned"][chosen[i]["u"], chosen[i]["w"]].tolist())) > 1), mixed[0])
    token_figure(inp, aux["aligned"][chosen[k]["u"], chosen[k]["w"]], k, res_dir / "token_supervision.png")
    ids = {str(v) for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}; assert_no_ids(rec, ids, "check")
    for p in (out / "check.json", res_dir / "checks.json"):
        p.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v.get("ok") for k, v in rec.items() if isinstance(v, dict) and "ok" in v}, indent=1)); print(json.dumps(rec["budget"], indent=1))
    print(f"check ok: {rec['ok']}")
    return rec["ok"]
