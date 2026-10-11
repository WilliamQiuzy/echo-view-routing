#!/usr/bin/env python
"""Checks before the queue of the target-anchored readout round (plan Tasks 1-2; run through
scripts/train_target_patch_readout.py --check). Training data only; no development data is read.

1. Pairing: the three readouts of every seed load the same Ordered state strictly, with equal names, counts and digests,
   no added parameter and the learned marker; the token grid, the query strides and the real norm-output shape.
2. The 16 check windows of the seed-0 manifest (manifest order and coverage): Ordered pixels equal the manifest digests,
   the marker slot is the middle target slot, and the three readouts receive identical inputs.
3. Continued CLS equals the source model (eval, maximum difference 0). Global and Target patches equal an independent
   recomputation from the source model's norm output (768-d readout and 512-d output, maximum difference <= 1e-5).
4. Hooks never outlive a forward, also after an exception; gradients reach the marker and the head through the frozen
   encoder, the encoder and projection receive none while frozen and finite ones after unfreezing.
5. 100-update smokes per readout (encoder frozen, constant lr 1e-4 for the marker stem and head; no score gate).
6. The source round's rule for reproducibility and resume: two 6-update runs and 3 + 3 updates on the formal schedule
   within loss 1e-4 and parameters 1e-5; a short schedule across the unfreeze boundary with an exact state round-trip,
   losses within 1e-4 and drift within 2.1 x each group's summed learning rate.
7. Timing per readout in the unfrozen state on formal batches (5 warm-up, 20 timed updates, decoding and hooks included),
   the projected round cost against the 12-hour cap, and the input-token figure.
Writes runs/target_patch_readout_v1/checks/check.json, its id-free copy and the check ledger.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.target_patch_readout as tp
import scripts.train_raw_video_context as rvc
import scripts.train_target_patch_readout as tr
from echo_routing.evaluate.failure_units import session_of
from scripts.check_raw_video_context import check_windows, group_diff, max_diff
from scripts.prepare_fold_output_correction import assert_no_ids
from scripts.train_joint_visual_temporal import params_digest

SEED = 0
GROUP = tr.GROUP


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
        state, _ = ctx.source_state(s); base = source_model(state, "cpu"); keys = list(base.state_dict()); n = sum(p.numel() for p in base.parameters())
        ok = True
        for r in tp.READOUTS:
            m = tr.make_model(state, r, "cpu")
            ok &= list(m.state_dict()) == keys and sum(p.numel() for p in m.parameters()) == n and params_digest(m.named_parameters()) == params_digest(base.named_parameters())
        out[str(s)] = {"same_start_no_new_parameters": bool(ok), "parameters": int(n), "marker_abs_max": float(state["marker_stem.weight"].abs().max())}
    state = ctx.source_state(SEED)[0]; m = tr.make_model(state, "target_patches", "cpu"); base = source_model(state, device); shape = {}
    h = base.encoder.norm.register_forward_hook(lambda _m, _i, o: shape.__setitem__("x", list(o.shape)))   # the plain source path
    try:
        with torch.no_grad():
            base(torch.zeros(2, 3, 16, 224, 224, device=device), tm.marker_planes(torch.tensor([8, 9], device=device)))
    finally:
        h.remove()
    return {"seeds": out, "grid": list(m.grid), "query_strides": [list(x) for x in m.query_strides], "stem_time": list(m.stem_time),
            "norm_output_shape": shape["x"], "ok": all(v["same_start_no_new_parameters"] and v["marker_abs_max"] > 0 for v in out.values())
            and list(m.grid) == [8, 7, 7] and all(x[0] == 1 for x in m.query_strides) and shape["x"] == [2, 393, 768]}


def input_checks(m: dict, pairs: list, inp: tuple) -> dict:
    fails = {"digest": 0, "marker_slot": 0, "target": 0}
    for k, (u, w) in enumerate(pairs):
        fails["digest"] += int(not np.array_equal(inp[3][k].numpy(), m["digest_ordered"][u, w]))
        fails["marker_slot"] += int(int(inp[1][k]) != tm.marker_slot(m["s"][u, w], int(m["c"][u, w])))
        fails["target"] += int(int(inp[2][k]) != int(m["target"][u, w]))
    return {"windows": len(pairs), "failures": fails, "same_inputs_for_every_readout": "one decoded batch feeds all three readouts (Ordered order only)",
            "ok": not any(fails.values())}


# ----------------------------------------------------------------------------------------------- equivalence
def own_weights(slots: torch.Tensor) -> torch.Tensor:
    """Independent recomputation: time slot t covers input slots 2t-1, 2t and 2t+1."""
    w = torch.zeros(len(slots), 8)
    for i, k in enumerate(slots.tolist()):
        cover = [t for t in range(8) if abs(2 * t - int(k)) <= 1]
        w[i, cover] = 1.0 / len(cover)
    return w


@torch.no_grad()
def equivalence(ctx, inp: tuple, device) -> dict:
    state, _ = ctx.source_state(SEED); base = source_model(state, device); rgb, mk, _ = batches(inp, device); out = {}
    cls = tr.make_model(state, "cls", device).eval()
    out["cls_max_abs"] = float(max((cls(rgb[i:i + 8], mk[i:i + 8]) - base(rgb[i:i + 8], mk[i:i + 8])).abs().max() for i in (0, 8)))
    for r in ("global_patches", "target_patches"):
        model = tr.make_model(state, r, device).eval(); d768 = d512 = dout = 0.0
        for i in (0, 8):
            seen, head_in = {}, {}
            h1 = base.encoder.norm.register_forward_hook(lambda _m, _i, o: seen.__setitem__("x", o.clone()))
            try:
                base.embed(rgb[i:i + 8], mk[i:i + 8])
            finally:
                h1.remove()
            p = seen["x"][:, 1:].reshape(8, 8, 7, 7, 768).mean((2, 3))
            vec = p.mean(1) if r == "global_patches" else (own_weights(inp[1][i:i + 8]).to(device)[:, :, None] * p).sum(1)
            h2 = model.encoder.head.register_forward_pre_hook(lambda _m, a: head_in.__setitem__("x", a[0][:, 0].clone() if a[0].ndim == 3 else a[0].clone()))
            try:
                emb = model.embed(rgb[i:i + 8], mk[i:i + 8])
            finally:
                h2.remove()
            d768 = max(d768, float((head_in["x"] - vec).abs().max())); d512 = max(d512, float((emb - base.encoder.head(vec)).abs().max()))
            dout = max(dout, float((model.classifier(emb) - base.classifier(base.encoder.head(vec))).abs().max()))
        out[r] = {"readout_768_max_abs": d768, "embedding_512_max_abs": d512, "logits_max_abs": dout}
    tol = float(ctx.tp["checks"]["readout_recompute_max_abs"])
    out["ok"] = out["cls_max_abs"] <= float(ctx.tp["checks"]["cls_parity_max_abs"]) and all(out[r]["readout_768_max_abs"] <= tol and out[r]["embedding_512_max_abs"] <= tol
                                                                                           for r in ("global_patches", "target_patches"))
    return out


def hooks_and_grads(ctx, inp: tuple, device) -> dict:
    state, _ = ctx.source_state(SEED); rgb, mk, y = batches(inp, device); out = {}
    for r in tp.READOUTS:
        m = tr.make_model(state, r, device).eval()
        with torch.no_grad():
            a = m(rgb[:4], mk[:4]); b = m(rgb[:4], mk[:4])
        blk = m.encoder.blocks[3]; blk.forward = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("check"))
        try:
            m(rgb[:4], mk[:4]); raised = False
        except RuntimeError:
            raised = True
        finally:
            del blk.forward
        clean = not m.encoder.norm._forward_hooks and not m.encoder.conv_proj._forward_hooks
        m.train(); m.set_encoder_trainable(False); m.zero_grad(set_to_none=True); F.cross_entropy(m(rgb[:8], mk[:8]), y[:8]).backward()
        frozen = {"marker": float(m.marker_stem.weight.grad.abs().max()), "head": float(m.classifier.weight.grad.abs().max()),
                  "encoder_with_grad": int(sum(p.grad is not None for p in m.encoder_parameters()))}
        m.set_encoder_trainable(True); m.zero_grad(set_to_none=True); F.cross_entropy(m(rgb[:8], mk[:8]), y[:8]).backward()
        g = [p.grad for p in m.encoder_parameters()]
        unfrozen = {"projection": float(m.encoder.head[-1].weight.grad.abs().max()), "none": sum(x is None for x in g),
                    "nonfinite": sum(x is not None and not bool(torch.isfinite(x).all()) for x in g)}
        out[r] = {"repeat_equal": bool(torch.equal(a, b)), "exception_raised": raised, "hooks_clean": clean, "frozen": frozen, "unfrozen": unfrozen,
                  "ok": bool(torch.equal(a, b)) and raised and clean and frozen["marker"] > 0 and frozen["head"] > 0 and frozen["encoder_with_grad"] == 0
                  and unfrozen["projection"] > 0 and unfrozen["none"] == 0 and unfrozen["nonfinite"] == 0}
        del m; torch.cuda.empty_cache()
    out["ok"] = all(v["ok"] for v in out.values())
    return out


# ----------------------------------------------------------------------------------------------- short training runs
class Updates:
    def __init__(self, m: dict, fp) -> None:
        self.m, self.fp, self.cache = m, fp, {}

    def __call__(self, u: int):
        if u not in self.cache:
            out = rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(rvc.WINDOWS)])
            rvc.check_digests(self.m, GROUP, u, out[3]); self.cache[u] = (u, *out)
        return self.cache[u]


def run_updates(ctx, upd: Updates, readout: str, us: list[int], warmup: int, total: int, device, state_file=None) -> dict:
    state, _ = ctx.source_state(SEED); model = tr.make_model(state, readout, device); opt = rvc.make_optimizer(model, ctx.o)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if state_file is not None:
        rvc.load_resume(state_file, model, opt, "check")
    enc0 = params_digest(model.encoder.named_parameters()); recs = []
    for u in us:
        lrs = rvc.apply_schedule(model, opt, u, warmup, total, ctx.o); s = rvc.train_step(model, opt, upd(u), lrs, ctx.o, device)
        s["encoder_digest"] = params_digest(model.encoder.named_parameters()); recs.append(s)
    return {"model": model, "opt": opt, "records": recs, "encoder_start": enc0}


def roundtrip(ctx, run: dict, readout: str, tmp: Path, device) -> dict:
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all(); rvc.save_resume(tmp, run["model"], run["opt"], 0, "check", [])
    state, _ = ctx.source_state(SEED); model = tr.make_model(state, readout, device); opt = rvc.make_optimizer(model, ctx.o)
    torch.manual_seed(12345); torch.cuda.manual_seed_all(12345); rvc.load_resume(tmp, model, opt, "check"); tmp.unlink(missing_ok=True)
    a, b, sa, sb = run["model"].state_dict(), model.state_dict(), run["opt"].state_dict(), opt.state_dict()
    weights = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    optim = sa["param_groups"] == sb["param_groups"] and sa["state"].keys() == sb["state"].keys() and all(
        torch.equal(torch.as_tensor(sa["state"][i][f]).cpu(), torch.as_tensor(sb["state"][i][f]).cpu()) for i in sa["state"] for f in sa["state"][i])
    rng = torch.equal(torch.get_rng_state(), cpu) and all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), cuda))
    return {"weights": bool(weights), "optimiser": bool(optim), "rng": bool(rng), "ok": bool(weights and optim and rng)}


def reproducibility(ctx, upd: Updates, readout: str, device, tmp: Path) -> dict:
    c = ctx.tp["checks"]; us = list(range(int(c["reproducibility"]["updates"])))
    a = run_updates(ctx, upd, readout, us, ctx.warmup, ctx.total, device); b = run_updates(ctx, upd, readout, us, ctx.warmup, ctx.total, device)
    half = run_updates(ctx, upd, readout, us[:3], ctx.warmup, ctx.total, device); rvc.save_resume(tmp, half["model"], half["opt"], 3, "check", [])
    rest = run_updates(ctx, upd, readout, us[3:], ctx.warmup, ctx.total, device, state_file=tmp); tmp.unlink(missing_ok=True)
    la, lb, lc = ([r["loss"] for r in x] for x in (a["records"], b["records"], half["records"] + rest["records"]))
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lb)), "params_max_abs": max_diff(a["model"], b["model"])}
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lc)), "params_max_abs": max_diff(a["model"], rest["model"])}
    enc_same = all(r["encoder_digest"] == a["encoder_start"] for r in a["records"])
    ok = (rep["loss_max_abs"] <= float(c["reproducibility"]["loss_abs"]) and rep["params_max_abs"] <= float(c["reproducibility"]["params_abs"])
          and res["loss_max_abs"] <= float(c["resume"]["loss_abs"]) and res["params_max_abs"] <= float(c["resume"]["params_abs"]) and enc_same)
    return {"losses": la, "reproducibility": rep, "resume": res, "encoder_unchanged_in_warmup": enc_same, "ok": bool(ok)}


def boundary(ctx, upd: Updates, readout: str, device, tmp: Path) -> dict:
    b = ctx.tp["checks"]["boundary"]; w, t, k = int(b["warmup"]), int(b["total"]), int(b["resume_at"]); us = list(range(t))
    d = run_updates(ctx, upd, readout, us, w, t, device); f = run_updates(ctx, upd, readout, us, w, t, device)
    half = run_updates(ctx, upd, readout, us[:k], w, t, device); rvc.save_resume(tmp, half["model"], half["opt"], k, "check", [])
    e = run_updates(ctx, upd, readout, us[k:], w, t, device, state_file=tmp); tmp.unlink(missing_ok=True)
    want = [rvc.schedule(u, w, t, ctx.o) for u in us]; got = [(r["lr_encoder"], r["lr_new"]) for r in d["records"]]
    frozen = [r["encoder_digest"] == d["encoder_start"] for r in d["records"]]
    ld, le, lf = ([r["loss"] for r in x] for x in (d["records"], half["records"] + e["records"], f["records"]))
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, le)), "by_group": group_diff(d["model"], e["model"])}
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, lf)), "by_group": group_diff(d["model"], f["model"])}
    trip = roundtrip(ctx, d, readout, tmp, device)
    bound = {"encoder": float(b["adam_step_bound"]) * sum(x[0] for x in want), "new": float(b["adam_step_bound"]) * sum(x[1] for x in want)}
    within = all(x["by_group"][g]["max_abs"] <= bound[g] for x in (res, rep) for g in bound)
    tol = float(ctx.tp["checks"]["resume"]["loss_abs"])
    ok = got == want and all(frozen[:w]) and not any(frozen[w:]) and trip["ok"] and res["loss_max_abs"] <= tol and rep["loss_max_abs"] <= tol and within
    return {"lrs": got, "encoder_unchanged_after_update": frozen, "state_roundtrip": trip, "resume": res, "repeat": rep, "parameter_bound": bound,
            "within_bound": within, "ok": bool(ok)}


def smoke(ctx, inp: tuple, readout: str, device) -> dict:
    sc = ctx.tp["checks"]["smoke"]; state, _ = ctx.source_state(SEED); m = tr.make_model(state, readout, device); m.set_encoder_trainable(False)
    opt = torch.optim.AdamW(m.new_parameters(), lr=float(sc["lr_new"]), betas=tuple(float(x) for x in ctx.o["betas"]), eps=float(ctx.o["eps"]),
                            weight_decay=float(ctx.o["weight_decay"]))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); enc0 = params_digest(m.encoder.named_parameters()); new0 = [p.detach().clone() for p in m.new_parameters()]
    rgb, mk, y = batches(inp, device); losses, grads, t0 = [], [], time.time()
    for i in range(int(sc["updates"])):
        sl = slice(0, 8) if i % 2 == 0 else slice(8, 16); m.train()
        loss = F.cross_entropy(m(rgb[sl], mk[sl]), y[sl]); opt.zero_grad(set_to_none=True); loss.backward()
        grads.append(float(torch.nn.utils.clip_grad_norm_(m.new_parameters(), float(ctx.o["grad_clip"])))); opt.step(); losses.append(float(loss.detach()))
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(rgb[i:i + 8], mk[i:i + 8]).argmax(1) for i in (0, 8)])
    enc_same = params_digest(m.encoder.named_parameters()) == enc0; changed = all(not torch.equal(p.detach(), q) for p, q in zip(m.new_parameters(), new0) if p.ndim > 1)
    finite = bool(np.isfinite(losses).all() and np.isfinite(grads).all())
    return {"readout": readout, "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])), "grad_norm_first10": float(np.mean(grads[:10])),
            "grad_norm_last10": float(np.mean(grads[-10:])), "accuracy_16": float((pred == y).float().mean()), "encoder_unchanged": enc_same,
            "marker_and_head_changed": changed, "seconds": time.time() - t0, "ok": finite and enc_same and changed}


def timing(ctx, m: dict, fp, readout: str, device, workers: int) -> dict:
    tc = ctx.tp["checks"]["timing"]; n_w, n_t = int(tc["warmup_updates"]), int(tc["timed_updates"]); state, _ = ctx.source_state(SEED)
    model = tr.make_model(state, readout, device); opt = rvc.make_optimizer(model, ctx.o); model.set_encoder_trainable(True)
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])
    torch.manual_seed(SEED); torch.cuda.reset_peak_memory_stats(device); stamps = []; torch.cuda.synchronize(); prev = time.time()
    for batch in rvc.loader(m, GROUP, fp, list(range(n_w + n_t)), workers):
        rvc.check_digests(m, GROUP, int(batch[0]), batch[4]); rvc.train_step(model, opt, batch, (float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])), ctx.o, device)
        torch.cuda.synchronize(); now = time.time(); stamps.append(now - prev); prev = now
    timed = stamps[n_w:]
    return {"readout": readout, "seconds_per_update_mean": float(np.mean(timed)), "seconds_per_update_max": float(np.max(timed)),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20, "hours_per_run": float(np.mean(timed)) * ctx.total / 3600}


def token_figure(inp: tuple, k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames, slot = inp[0][k].numpy(), int(inp[1][k]); w = tp.time_weights(tp.marker_vector(tm.marker_planes(torch.tensor([slot]), size=4)))[0].numpy()
    fig = plt.figure(figsize=(17, 4.4)); gs = fig.add_gridspec(2, 16, height_ratios=[1.25, 1])
    for i in range(16):
        a = fig.add_subplot(gs[0, i]); a.imshow(frames[i][::4, ::4]); a.set_xticks([]); a.set_yticks([]); a.set_title(f"slot {i}", fontsize=7)
        for sp in a.spines.values():
            sp.set_edgecolor("#d62728" if i == slot else "#bbbbbb"); sp.set_linewidth(3 if i == slot else 0.5)
    b = fig.add_subplot(gs[1, :]); b.bar(np.arange(8), w, color="#4c72b0"); b.set_xticks(np.arange(8)); b.set_ylim(0, 1.05)
    b.set_xticklabels([f"time slot {t}\n(frames {max(0, 2 * t - 1)}-{min(15, 2 * t + 1)})" for t in range(8)], fontsize=8)
    b.set_ylabel("Target patches weight")
    fig.suptitle(f"One mixed-view training window in Ordered context (red: marker slot {slot}). The final norm holds 1 + 8 x 7 x 7 tokens of 768 channels: "
                 "Continued CLS reads token 0, Global patches averages all 392 patch tokens, Target patches averages each time slot's 49 tokens and weights the slots as shown.",
                 fontsize=9, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def run_checks(ctx, device, workers: int) -> None:
    """Every attempt, finished or not, adds its GPU hours to the round's check ledger."""
    t0 = time.time(); ledger = ctx.out / "checks" / "ledger.jsonl"; ledger.parent.mkdir(parents=True, exist_ok=True); ok = False
    try:
        ok = checks(ctx, device, workers, t0, ledger)
    finally:
        with ledger.open("a") as fh:
            fh.write(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours": (time.time() - t0) / 3600, "ok": ok,
                                 "code": tr.code_digest()}) + "\n")
    if not ok:
        raise SystemExit("a hard check failed; see check.json")


def checks(ctx, device, workers: int, t0: float, ledger_path: Path) -> bool:
    out = ctx.out / "checks"; res_dir = ctx.root / ctx.tp["outputs"]["results_dir"]; res_dir.mkdir(parents=True, exist_ok=True)
    m, _ = ctx.manifest(SEED); fp = ctx.frame_paths(); rec: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = check_windows(m); pairs = [(c["u"], c["w"]) for c in chosen]; inp = rvc.window_inputs(m, m["order_ordered"], fp, pairs)
        rec["windows"] = [{**c, "kind": int(m["kind"][c["u"], c["w"]]), "marker": int(m["marker"][c["u"], c["w"]]), "target_slots": int(m["n_target_slots"][c["u"], c["w"]])} for c in chosen]
        rec["pairing"] = pairing(ctx, device); rec["inputs"] = input_checks(m, pairs, inp); rec["equivalence"] = equivalence(ctx, inp, device)
        rec["hooks_and_gradients"] = hooks_and_grads(ctx, inp, device); print("pairing, inputs, equivalence, hooks and gradients done", flush=True)
        rec["smoke"] = {r: smoke(ctx, inp, r, device) for r in tp.READOUTS}; print("smokes done", flush=True)
        upd = Updates(m, fp)
        rec["reproducibility"] = {r: reproducibility(ctx, upd, r, device, out / "tmp_resume.pt") for r in tp.READOUTS}
        rec["boundary"] = {r: boundary(ctx, upd, r, device, out / "tmp_resume.pt") for r in tp.READOUTS}; print("reproducibility, resume and boundary done", flush=True)
        rec["timing"] = {r: timing(ctx, m, fp, r, device, workers) for r in tp.READOUTS}
    rec["nondeterministic_operator_warnings"] = sorted({str(w.message).split("\n")[0][:300] for w in caught if "determinis" in str(w.message).lower()})
    earlier = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()] if ledger_path.is_file() else []
    spent = (time.time() - t0) / 3600; hours = max(t["hours_per_run"] for t in rec["timing"].values()); b = ctx.tp["budget"]
    rec["budget"] = {"check_hours_this_run": spent, "check_hours_earlier": float(sum(x["hours"] for x in earlier)), "hours_per_run_max": hours, "runs": len(ctx.tp["queue"]),
                     "evaluation_allowance_hours": float(b["evaluation_allowance_hours"]), "cap_hours": float(b["gpu_hours_max"])}
    rec["budget"]["projected_total_hours"] = rec["budget"]["check_hours_earlier"] + spent + len(ctx.tp["queue"]) * hours + float(b["evaluation_allowance_hours"])
    rec["budget"]["ok"] = rec["budget"]["projected_total_hours"] <= rec["budget"]["cap_hours"]
    rec["ok"] = bool(rec["pairing"]["ok"] and rec["inputs"]["ok"] and rec["equivalence"]["ok"] and rec["hooks_and_gradients"]["ok"] and rec["budget"]["ok"]
                     and all(v["ok"] for v in rec["smoke"].values()) and all(v["ok"] for v in rec["reproducibility"].values()) and all(v["ok"] for v in rec["boundary"].values()))
    rec["identities"] = {tr.run_key(r, s): tr.identity(ctx, r, s, ctx.manifest(s)[1], ctx.tp["source"]["weights"][s]["sha256"]) for s in ctx.seeds for r in tp.READOUTS}
    rec["code"] = tr.code_digest(); rec["seconds"] = time.time() - t0
    k = next(i for i, c in enumerate(chosen) if c["reason"].startswith("mixed") and "slot 8" not in c["reason"]); token_figure(inp, k, res_dir / "input_tokens.png")
    ids = {str(v) for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}; assert_no_ids(rec, ids, "check")
    for p in (out / "check.json", res_dir / "checks.json"):
        p.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v.get("ok") for k, v in rec.items() if isinstance(v, dict) and "ok" in v}, indent=1)); print(json.dumps(rec["budget"], indent=1))
    print(f"check ok: {rec['ok']}")
    return rec["ok"]
