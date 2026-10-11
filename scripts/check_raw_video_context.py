#!/usr/bin/env python
"""Checks before the queue of the raw-video context round (plan Tasks 1-2; run through
scripts/train_raw_video_context.py --check). Training data only; no development data is read.

1. Source encoder: strict load against the checkpoint; parity of the 512-d output against the new model with the zero
   marker branch on the same processed tensor (fp32, max |difference| <= 1e-4).
2. The 16 check windows of the seed-0 manifest, chosen by manifest order alone: Ordered and Shuffled hold the same
   pixel multiset with every target slot in place, Target-only repeats the target pixels in every slot, the marker
   has exactly one plane at the recomputed slot, the three groups share targets, and the decoded pixels equal the
   prepared digests.
3. Isolation: forward takes (rgb, marker) only; replacing labels, recording names and native flags changes no input
   tensor and no output. The marker branch receives a gradient that depends on the marker slot through the frozen
   trunk, and with non-zero weights a moved marker changes the output. The hook never outlives a forward, also after
   an exception, and repeated forwards agree.
4. Reproducibility (two runs of 6 updates) and resume (3 + 3) on the formal schedule for every group; a short
   schedule (warm-up 3, total 8, resume after 2) across the unfreeze boundary: learning rates, a frozen encoder
   before and a changing one after, finite gradients, resume and repeat agreement.
5. 300-update fits per group on the check windows (encoder frozen, constant lr 1e-4 for the new parameters).
6. Timing per group in the unfrozen state on formal batches (5 warm-up, 50 timed updates, decoding included), peak
   memory and the projected round cost; the input-token figure.
Writes runs/raw_video_context_v1/checks/check.json and the id-free copy in the results directory. Fails when a hard
check fails or the projection exceeds the budget.
"""
from __future__ import annotations

import copy
import inspect
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import echo_routing.vit.target_mvit as tm
import scripts.train_raw_video_context as tr
from echo_routing.evaluate.failure_units import session_of
from scripts.prepare_fold_output_correction import assert_no_ids
from scripts.train_joint_visual_temporal import FramePaths, params_digest

SEED = 0


# ----------------------------------------------------------------------------------------------- windows and inputs
def check_windows(m: dict) -> list[dict]:
    """16 windows chosen by manifest order and these coverage rules (kind, marker slot, target slots, repeats)."""
    kind, mk, nt, rep = m["kind"], m["marker"], m["n_target_slots"], m["repeats"]
    rules = [("native, repeated samples", lambda u, w: kind[u, w] == 0 and rep[u, w] > 0),
             ("native, marker at slot 0", lambda u, w: kind[u, w] == 0 and mk[u, w] == 0),
             ("native, marker at slot 15", lambda u, w: kind[u, w] == 0 and mk[u, w] == 15),
             ("mixed, marker at slot 0", lambda u, w: kind[u, w] == 1 and mk[u, w] == 0),
             ("mixed, marker at slot 15", lambda u, w: kind[u, w] == 1 and mk[u, w] == 15),
             ("two or more target slots", lambda u, w: nt[u, w] >= 2),
             ("native, marker at slot 8", lambda u, w: kind[u, w] == 0 and mk[u, w] == 8),
             ("mixed, marker at slot 8", lambda u, w: kind[u, w] == 1 and mk[u, w] == 8)]
    flat = [(u, w) for u in range(kind.shape[0]) for w in range(kind.shape[1])]; chosen: list[dict] = []
    for name, rule in rules:
        hit = next((p for p in flat if rule(*p) and all((c["u"], c["w"]) != p for c in chosen)), None)
        if hit is None:
            raise SystemExit(f"no check window for '{name}'")
        chosen.append({"u": hit[0], "w": hit[1], "reason": name})
    for p in flat:
        if len(chosen) == 16:
            break
        if all((c["u"], c["w"]) != p for c in chosen):
            chosen.append({"u": p[0], "w": p[1], "reason": "next in manifest order"})
    return chosen


def decode_groups(m: dict, fp, pairs: list[tuple[int, int]]) -> dict:
    return {g: tr.window_inputs(m, m[f"order_{tm.KEYS[g]}"], fp, pairs) for g in tm.GROUPS}


def input_checks(m: dict, pairs: list[tuple[int, int]], inp: dict) -> dict:
    o, s_, t_ = (inp[g][0].numpy() for g in tm.GROUPS[::-1])          # Ordered, Shuffled, Target-only frames
    fails = {k: 0 for k in ("multiset", "target_slots_moved", "target_only_slot", "marker_slot", "marker_plane", "targets", "digest")}
    for k, (u, w) in enumerate(pairs):
        ho, hs, ht = ([tm.frame_hash(x[k, i]) for i in range(tm.N_SLOTS)] for x in (o, s_, t_))
        slots = tm.target_slots(m["s"][u, w], int(m["c"][u, w])); mk = tm.marker_slot(m["s"][u, w], int(m["c"][u, w]))
        fails["multiset"] += int(sorted(ho) != sorted(hs))
        fails["target_slots_moved"] += int(any(hs[i] != ho[i] for i in slots))
        fails["target_only_slot"] += int(any(h != ho[mk] for h in ht))
        fails["marker_slot"] += int(any(int(inp[g][1][k]) != mk for g in tm.GROUPS))
        plane = tm.marker_planes(inp[tm.GROUPS[0]][1][k:k + 1])[0, 0]
        fails["marker_plane"] += int(not (plane.sum() == 224 * 224 and bool(plane[mk].eq(1).all()) and float(plane.sum(dim=(1, 2))[mk]) == 224 * 224))
        fails["targets"] += int(len({int(inp[g][2][k]) for g in tm.GROUPS} | {int(m["target"][u, w])}) != 1)
        fails["digest"] += int(any(not np.array_equal(inp[g][3][k].numpy(), m[f"digest_{tm.KEYS[g]}"][u, w]) for g in tm.GROUPS))
    return {"windows": len(pairs), "failures": fails, "ok": not any(fails.values())}


# ----------------------------------------------------------------------------------------------- model checks
def strict_load(ctx) -> dict:
    enc = tm.build_source_encoder(ctx.checkpoint); ref = torch.load(ctx.checkpoint, map_location="cpu", weights_only=True); st = enc.state_dict()
    equal = set(ref) == set(st) and all(torch.equal(ref[k], st[k]) for k in ref)
    return {"tensors": len(ref), "all_equal": bool(equal), "parameters": int(sum(p.numel() for p in enc.parameters())),
            "projection": list(enc.head[-1].weight.shape), "ok": bool(equal and enc.head[-1].out_features == tm.EMBED_DIM)}


@torch.no_grad()
def parity(ctx, inp: dict, device) -> dict:
    src = tm.build_source_encoder(ctx.checkpoint).to(device).eval(); init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device).eval()
    u8, slots = inp["Ordered context"][0][:8], inp["Ordered context"][1][:8]; rgb = tm.to_rgb(u8.to(device))
    a = src(rgb); b = model.embed(rgb, tm.marker_planes(slots.to(device))); c = model.embed(rgb, tm.marker_planes(((slots + 5) % 16).to(device)))
    d = max(float((a - b).abs().max()), float((a - c).abs().max()))
    return {"max_abs": d, "windows": 8, "ok": d <= float(ctx.rv["checks"]["parity_max_abs"])}


def isolation(ctx, m: dict, fp, pairs: list, inp: dict, device) -> dict:
    init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device).eval()
    sig = list(inspect.signature(model.forward).parameters)
    m2 = dict(m); aliases = np.asarray([f"alias_{i}" for i in range(len(m["videos"]))])
    m2.update({"videos": aliases, "target": ((m["target"].astype(int) + 1) % 5).astype(m["target"].dtype), "kind": (1 - m["kind"]).astype(m["kind"].dtype),
               "seq": np.full_like(m["seq"], -7)})
    fp2 = FramePaths({a: fp.dirs[str(v)] for a, v in zip(aliases, m["videos"])})
    same_in, same_out = True, True
    for g in tm.GROUPS:
        u8b, slotb, _, digb = tr.window_inputs(m2, m2[f"order_{tm.KEYS[g]}"], fp2, pairs[:8])
        u8a, slota = inp[g][0][:8], inp[g][1][:8]
        same_in &= bool(torch.equal(u8a, u8b) and torch.equal(slota, slotb) and torch.equal(inp[g][3][:8], digb))
        with torch.no_grad():
            oa = model(tm.to_rgb(u8a.to(device)), tm.marker_planes(slota.to(device))); ob = model(tm.to_rgb(u8b.to(device)), tm.marker_planes(slotb.to(device)))
        same_out &= bool(torch.equal(oa, ob))
    return {"forward_parameters": sig, "inputs_unchanged": same_in, "outputs_unchanged": same_out, "ok": sig == ["rgb", "marker"] and same_in and same_out}


def marker_branch(ctx, inp: dict, device) -> dict:
    init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device).eval(); model.set_encoder_trainable(False)
    u8, slots, target = (x[:8] for x in inp["Ordered context"][:3]); rgb = tm.to_rgb(u8.to(device)); grads = []
    for sl in (slots, (slots + 4) % 16):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(rgb, tm.marker_planes(sl.to(device))), target.to(device)).backward()
        grads.append(model.marker_stem.weight.grad.detach().clone())
    enc_grads = sum(p.grad is not None for p in model.encoder_parameters())
    probe = copy.deepcopy(model)
    with torch.no_grad():
        probe.marker_stem.weight.copy_(torch.randn(probe.marker_stem.weight.shape, generator=torch.Generator().manual_seed(1)).to(device) * 1e-2)
        moved = float((probe(rgb, tm.marker_planes(slots.to(device))) - probe(rgb, tm.marker_planes(((slots + 4) % 16).to(device)))).abs().max())
    nz = [float(g.abs().max()) for g in grads]; diff = float((grads[0] - grads[1]).abs().max())
    return {"marker_grad_abs_max": nz, "marker_grad_difference_between_slots": diff, "encoder_parameters_with_grad_while_frozen": int(enc_grads),
            "output_change_with_nonzero_marker_weights": moved, "ok": min(nz) > 0 and diff > 0 and enc_grads == 0 and moved > 0}


def hook_hygiene(ctx, inp: dict, device) -> dict:
    init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device).eval()
    with torch.no_grad():
        model.marker_stem.weight.copy_(torch.randn(model.marker_stem.weight.shape, generator=torch.Generator().manual_seed(2)).to(device) * 1e-2)
    u8, slots = inp["Ordered context"][0][:4], inp["Ordered context"][1][:4]; rgb = tm.to_rgb(u8.to(device)); mk = tm.marker_planes(slots.to(device))
    conv = model.encoder.conv_proj; src = tm.build_source_encoder(ctx.checkpoint).to(device).eval()
    with torch.no_grad():
        first = model(rgb, mk); after_forward = len(conv._forward_hooks)
        block = model.encoder.blocks[0]; original = block.forward

        def boom(*a, **k):
            raise RuntimeError("check: exception inside the encoder")
        block.forward = boom
        try:
            model(rgb, mk); raised = False
        except RuntimeError:
            raised = True
        finally:
            del block.forward
        restored = block.forward.__func__ is original.__func__
        after_exception = len(conv._forward_hooks); second = model(rgb, mk); third = model(rgb, mk)
        plain = float((model.encoder(rgb) - src(rgb)).abs().max())
    return {"hooks_after_forward": after_forward, "exception_raised": raised, "hooks_after_exception": after_exception, "block_restored": restored,
            "repeat_forwards_equal": bool(torch.equal(first, second) and torch.equal(second, third)), "plain_encoder_vs_source_max_abs": plain,
            "ok": after_forward == 0 and raised and after_exception == 0 and restored and bool(torch.equal(first, second) and torch.equal(second, third)) and plain <= 1e-4}


# ----------------------------------------------------------------------------------------------- short training runs
class Updates:
    """Decoded formal updates of one group, kept in memory for the short runs."""

    def __init__(self, m: dict, group: str, fp) -> None:
        self.m, self.group, self.fp, self.cache = m, group, fp, {}

    def __call__(self, u: int):
        if u not in self.cache:
            out = tr.window_inputs(self.m, self.m[f"order_{tm.KEYS[self.group]}"], self.fp, [(u, w) for w in range(tr.WINDOWS)])
            tr.check_digests(self.m, self.group, u, out[3]); self.cache[u] = (u, *out)
        return self.cache[u]


def run_updates(ctx, upd: Updates, us: list[int], warmup: int, total: int, device, state=None) -> dict:
    """Train on updates `us` with schedule (warmup, total) from the seed-0 initial state or a resume file."""
    init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device); opt = tr.make_optimizer(model, ctx.o)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if state is not None:
        tr.load_resume(state, model, opt, "check")
    enc0 = params_digest(model.encoder.named_parameters()); recs = []
    for u in us:
        lrs = tr.apply_schedule(model, opt, u, warmup, total, ctx.o); s = tr.train_step(model, opt, upd(u), lrs, ctx.o, device)
        enc_grads = [p.grad for p in model.encoder_parameters()]
        s["encoder_grads"] = {"none": sum(g is None for g in enc_grads), "nonfinite": sum(g is not None and not bool(torch.isfinite(g).all()) for g in enc_grads),
                              "all_zero": sum(g is not None and float(g.abs().max()) == 0.0 for g in enc_grads),
                              "projection_abs_max": None if model.encoder.head[-1].weight.grad is None else float(model.encoder.head[-1].weight.grad.abs().max())}
        s["encoder_digest"] = params_digest(model.encoder.named_parameters()); recs.append(s)
    return {"model": model, "opt": opt, "records": recs, "encoder_start": enc0}


def max_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sb = b.state_dict()
    return max(float((v.float() - sb[k].float()).abs().max()) for k, v in a.state_dict().items())


def repro_resume(ctx, upd: Updates, device, tmp: Path) -> dict:
    """Formal schedule, updates 0-5: two identical runs, and 3 + 3 with a resume file against 6 in one go."""
    c = ctx.rv["checks"]; us = list(range(int(c["reproducibility"]["updates"])))
    a = run_updates(ctx, upd, us, ctx.warmup, ctx.total, device); b = run_updates(ctx, upd, us, ctx.warmup, ctx.total, device)
    half = run_updates(ctx, upd, us[:3], ctx.warmup, ctx.total, device); tr.save_resume(tmp, half["model"], half["opt"], 3, "check", [])
    rest = run_updates(ctx, upd, us[3:], ctx.warmup, ctx.total, device, state=tmp); tmp.unlink(missing_ok=True)
    la, lb, lc = ([r["loss"] for r in x["records"]] for x in (a, b, {"records": half["records"] + rest["records"]}))
    init, _ = ctx.init(SEED); fresh = tr.make_model(ctx, init, "cpu")
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lb)), "params_max_abs": max_diff(a["model"], b["model"])}
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lc)), "params_max_abs": max_diff(a["model"], rest["model"])}
    new_changed = any(not torch.equal(p.detach().cpu(), q) for p, q in zip(a["model"].new_parameters(), fresh.new_parameters()))
    marker_changed = not torch.equal(a["model"].marker_stem.weight.detach().cpu(), fresh.marker_stem.weight)
    enc_same = all(r["encoder_digest"] == a["encoder_start"] for r in a["records"])
    ok = (rep["loss_max_abs"] <= float(c["reproducibility"]["loss_abs"]) and rep["params_max_abs"] <= float(c["reproducibility"]["params_abs"])
          and res["loss_max_abs"] <= float(c["resume"]["loss_abs"]) and res["params_max_abs"] <= float(c["resume"]["params_abs"]) and new_changed and marker_changed and enc_same)
    return {"losses": la, "reproducibility": rep, "resume": res, "encoder_unchanged_in_warmup": enc_same, "new_parameters_changed": new_changed,
            "marker_changed": marker_changed, "lrs": [(r["lr_encoder"], r["lr_new"]) for r in a["records"]], "ok": bool(ok)}


def boundary(ctx, upd: Updates, device, tmp: Path) -> dict:
    """Short schedule across the unfreeze boundary: lr per update, encoder frozen before and changing after, gradients
    at the first unfrozen update, a resume before the boundary and a repeat in the unfrozen state."""
    c = ctx.rv["checks"]; b = c["boundary"]; w, t, k = int(b["warmup"]), int(b["total"]), int(b["resume_at"]); us = list(range(t))
    d = run_updates(ctx, upd, us, w, t, device); f = run_updates(ctx, upd, us, w, t, device)
    half = run_updates(ctx, upd, us[:k], w, t, device); tr.save_resume(tmp, half["model"], half["opt"], k, "check", [])
    e = run_updates(ctx, upd, us[k:], w, t, device, state=tmp); tmp.unlink(missing_ok=True)
    want = [tr.schedule(u, w, t, ctx.o) for u in us]; got = [(r["lr_encoder"], r["lr_new"]) for r in d["records"]]
    frozen = [r["encoder_digest"] == d["encoder_start"] for r in d["records"]]
    g = d["records"][w]["encoder_grads"]; le = [r["loss"] for r in half["records"] + e["records"]]; ld = [r["loss"] for r in d["records"]]
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, le)), "params_max_abs": max_diff(d["model"], e["model"])}
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, [r["loss"] for r in f["records"]])), "params_max_abs": max_diff(d["model"], f["model"])}
    ok = (got == want and all(frozen[:w]) and not any(frozen[w:]) and g["none"] == 0 and g["nonfinite"] == 0 and (g["projection_abs_max"] or 0) > 0
          and res["loss_max_abs"] <= float(c["resume"]["loss_abs"]) and res["params_max_abs"] <= float(c["resume"]["params_abs"])
          and rep["loss_max_abs"] <= float(c["reproducibility"]["loss_abs"]) and rep["params_max_abs"] <= float(c["reproducibility"]["params_abs"]))
    return {"schedule": {"warmup": w, "total": t, "resume_after": k}, "lrs": got, "lrs_expected": want, "encoder_unchanged_after_update": frozen,
            "first_unfrozen_update_grads": g, "resume": res, "repeat_unfrozen": rep, "losses": ld, "ok": bool(ok)}


def fit(ctx, inp: dict, group: str, device) -> dict:
    """300 updates on the 16 check windows: encoder frozen, constant lr 1e-4 for the new parameters (no score gate)."""
    fc = ctx.rv["checks"]["fit"]; init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device); model.set_encoder_trainable(False)
    opt = torch.optim.AdamW(model.new_parameters(), lr=float(fc["lr_new"]), betas=tuple(float(x) for x in ctx.o["betas"]), eps=float(ctx.o["eps"]),
                            weight_decay=float(ctx.o["weight_decay"]))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); enc0 = params_digest(model.encoder.named_parameters())
    u8, slots, target = inp[group][:3]; rgb = tm.to_rgb(u8.to(device)); mk = tm.marker_planes(slots.to(device)); y = target.to(device)
    losses, grads, t0 = [], [], time.time()
    for i in range(int(fc["updates"])):
        sl = slice(0, 8) if i % 2 == 0 else slice(8, 16); model.train()
        loss = F.cross_entropy(model(rgb[sl], mk[sl]), y[sl]); opt.zero_grad(set_to_none=True); loss.backward()
        grads.append(float(torch.nn.utils.clip_grad_norm_(model.new_parameters(), float(ctx.o["grad_clip"])))); opt.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(rgb[i:i + 8], mk[i:i + 8]).argmax(1) for i in (0, 8)]).cpu()
    conf = np.zeros((5, 5), int); np.add.at(conf, (target.numpy(), pred.numpy()), 1); tp = np.diag(conf); den = 2 * tp + conf.sum(0) - tp + conf.sum(1) - tp
    present = conf.sum(1) > 0; f1 = float(np.mean((2 * tp / np.maximum(den, 1))[present]))
    fresh = tr.make_model(ctx, init, "cpu")
    enc_same = params_digest(model.encoder.named_parameters()) == enc0
    new_changed = all(not torch.equal(p.detach().cpu(), q) for p, q in zip(model.new_parameters(), fresh.new_parameters()) if p.ndim > 1)
    finite = all(np.isfinite(losses)) and all(np.isfinite(grads))
    return {"group": group, "updates": len(losses), "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])),
            "losses_every_25": losses[::25], "grad_norm_first10": float(np.mean(grads[:10])), "grad_norm_last10": float(np.mean(grads[-10:])),
            "accuracy_16": float((pred == target).float().mean()), "macro_f1_present_classes_16": f1, "classes_present": int(present.sum()),
            "marker_abs_max": float(model.marker_stem.weight.detach().abs().max()), "encoder_unchanged": enc_same, "new_parameters_changed": new_changed,
            "seconds": time.time() - t0, "ok": bool(finite and enc_same and new_changed)}


def timing(ctx, m: dict, fp, group: str, device, workers: int) -> dict:
    """The unfrozen state on formal batches: 5 warm-up and 50 timed updates, decoding and the marker included."""
    tc = ctx.rv["checks"]["timing"]; n_w, n_t = int(tc["warmup_updates"]), int(tc["timed_updates"])
    init, _ = ctx.init(SEED); model = tr.make_model(ctx, init, device); opt = tr.make_optimizer(model, ctx.o); model.set_encoder_trainable(True)
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])
    torch.manual_seed(SEED); torch.cuda.reset_peak_memory_stats(device); stamps = []
    torch.cuda.synchronize(); t_prev = time.time()
    for batch in tr.loader(m, group, fp, list(range(n_w + n_t)), workers):
        tr.check_digests(m, group, int(batch[0]), batch[4])
        tr.train_step(model, opt, batch, (float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])), ctx.o, device)
        torch.cuda.synchronize(); now = time.time(); stamps.append(now - t_prev); t_prev = now
    timed = stamps[n_w:]
    return {"group": group, "seconds_per_update_mean": float(np.mean(timed)), "seconds_per_update_median": float(np.median(timed)),
            "seconds_per_update_max": float(np.max(timed)), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "hours_per_run": float(np.mean(timed)) * ctx.total / 3600}


# ----------------------------------------------------------------------------------------------- figure
def token_figure(m: dict, pairs: list, chosen: list, inp: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = next(i for i, c in enumerate(chosen) if c["reason"].startswith("mixed") and "slot 8" not in c["reason"])
    mk = int(inp["Ordered context"][1][k]); fig, ax = plt.subplots(3, 16, figsize=(17, 4.6))
    for r, g in enumerate(tm.GROUPS[::-1]):
        frames = inp[g][0][k].numpy()
        for i in range(16):
            a = ax[r, i]; a.imshow(frames[i][::4, ::4]); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor("#d62728" if i == mk else "#bbbbbb"); sp.set_linewidth(3 if i == mk else 0.5)
            if r == 0:
                a.set_title(f"slot {i}", fontsize=7)
        ax[r, 0].set_ylabel(g, fontsize=9)
    fig.suptitle("Model input of one mixed-view training window (frames shown at 1/4 size). Red: the marker slot, where the 0/1 marker plane is 1.\n"
                 "Both the RGB video (3 x 16 x 224 x 224) and the marker (1 x 16 x 224 x 224) pass through stems with kernel 3x7x7 and stride 2x4x4; "
                 "their sum gives 8 x 56 x 56 tokens of 96 channels plus a class token.", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def run_checks(ctx, device, workers: int) -> None:
    t0 = time.time(); out = ctx.out / "checks"; out.mkdir(parents=True, exist_ok=True); res_dir = ctx.root / ctx.rv["outputs"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True); m, _ = ctx.manifest(SEED); fp = ctx.frame_paths(); rec: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = check_windows(m); pairs = [(c["u"], c["w"]) for c in chosen]; inp = decode_groups(m, fp, pairs)
        rec["windows"] = [{**c, "kind": int(m["kind"][c["u"], c["w"]]), "marker": int(m["marker"][c["u"], c["w"]]),
                           "target_slots": int(m["n_target_slots"][c["u"], c["w"]]), "repeats": int(m["repeats"][c["u"], c["w"]])} for c in chosen]
        rec["inputs"] = input_checks(m, pairs, inp); rec["strict_load"] = strict_load(ctx); rec["parity"] = parity(ctx, inp, device)
        rec["isolation"] = isolation(ctx, m, fp, pairs, inp, device); rec["marker_branch"] = marker_branch(ctx, inp, device)
        rec["hook_hygiene"] = hook_hygiene(ctx, inp, device); print("input, parity, isolation and hook checks done", flush=True)
        rec["repro_resume"] = {g: repro_resume(ctx, Updates(m, g, fp), device, out / "tmp_resume.pt") for g in tm.GROUPS}
        rec["boundary"] = boundary(ctx, Updates(m, "Ordered context", fp), device, out / "tmp_resume.pt"); print("reproducibility, resume and boundary checks done", flush=True)
        rec["fit"] = {g: fit(ctx, inp, g, device) for g in tm.GROUPS}; print("fits done", flush=True)
        rec["timing"] = {g: timing(ctx, m, fp, g, device, workers) for g in tm.GROUPS}
    rec["nondeterministic_operator_warnings"] = sorted({str(w.message).split("\n")[0][:300] for w in caught if "determinis" in str(w.message).lower()})
    hours = max(t["hours_per_run"] for t in rec["timing"].values()); spent = (time.time() - t0) / 3600; b = ctx.rv["budget"]
    rec["budget"] = {"check_hours": spent, "hours_per_run_max": hours, "runs": len(ctx.rv["queue"]), "evaluation_allowance_hours": float(b["evaluation_allowance_hours"]),
                     "projected_total_hours": spent + len(ctx.rv["queue"]) * hours + float(b["evaluation_allowance_hours"]), "cap_hours": float(b["gpu_hours_max"])}
    rec["budget"]["ok"] = rec["budget"]["projected_total_hours"] <= rec["budget"]["cap_hours"]
    hard = ["inputs", "strict_load", "parity", "isolation", "marker_branch", "hook_hygiene", "boundary", "budget"]
    rec["ok"] = bool(all(rec[k]["ok"] for k in hard) and all(v["ok"] for v in rec["repro_resume"].values()) and all(v["ok"] for v in rec["fit"].values()))
    rec["identities"] = {tr.run_key(g, s): tr.identity(ctx, g, s, ctx.entry(s)["files"]["manifest"]["sha256"], ctx.entry(s)["files"]["init"]["sha256"])
                         for s in ctx.rv["training"]["seeds"] for g in tm.GROUPS}
    rec["code"] = tr.code_digest(); rec["seconds"] = time.time() - t0
    token_figure(m, pairs, chosen, inp, res_dir / "input_tokens.png")
    ids = {str(v) for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}; assert_no_ids(rec, ids, "check")
    for p in (out / "check.json", res_dir / "checks.json"):
        p.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else None) for k, v in rec.items() if isinstance(v, dict)}, indent=1))
    print(json.dumps(rec["budget"], indent=1)); print(f"check ok: {rec['ok']}")
    if not rec["ok"]:
        raise SystemExit("a hard check failed; see check.json")
