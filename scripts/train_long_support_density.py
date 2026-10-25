#!/usr/bin/env python
"""A33 (configs/experiments/video_long_support_density_recognition_v1.yaml): long-support density recognition screen.

  --stage audit    source plans / raw frames of every pool recording; 32-slot plans of every window (CPU)
  --stage check    Base16 reproduction, 32-frame adaptation, token grid, prefix + suffix, objective, gradients, forced
                   equality, resume, frozen weights, timing and disk -> the schedule (900 or the one fallback 600) (GPU)
  --stage cache    --arm R32|I32: the arm's frozen-prefix cache over both pools and its Init logits (GPU)
  --stage train    --arm: the arm's fold-0 and fold-1 models from the same adapted A08_s0 state (GPU)
  --stage readout  --arm: after the arm's weights are committed, held-out logits of both models (GPU)
  --stage clean    --arm: delete the arm's cache once its readouts exist
  --stage report   Base16 / Init / trained tables, I32 - R32, the protocol's decision (CPU)
The original A08 + B3 path only supplies Base and the runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")     # fewer fragmentation retries; same arithmetic

import argparse  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

import echo_routing.temporal.interval_clips as ic  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.long_support as ls  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402

A31_CONFIG = "configs/experiments/video_session_correction_v1.yaml"
SEED, FOLDS, ARMS = 0, (0, 1), ("R32", "I32")
BATCH, MICRO = 16, 8
now, write_json, ledger_append = t31.now, t31.write_json, t31.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["long_support_density"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG)

    def schedule(self) -> dict:
        return json.loads((self.out / "schedule.json").read_text())


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, job, kind, started, time.time() - t0, ok, note)


def private_ids(ctx: Ctx) -> set:
    pools = [t31.load_pool(ctx.a31, f) for f in FOLDS]; rec = {r[0] for p in pools for q in p for r in q["refs"]}
    return rec | {v.rsplit("_", 1)[0] for v in rec} | {q["id"] for p in pools for q in p}


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    flat = json.loads(json.dumps(rec, default=float)); assert_no_ids(flat, private_ids(ctx), name)
    for p in (ctx.out / name, ctx.results / name):
        p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)


# ----------------------------------------------------------------------------------------------- windows
def fold_windows(ctx: Ctx, f: int) -> dict:
    """A31's windows of pool f with their 32-slot plans (both arms), in A31's cache-row order."""
    seqs = t31.load_pool(ctx.a31, f); W = t31.pool_W(seqs); fr = [ls.fragment_ids(q) for q in seqs]; plans = []
    for j in range(len(W["seq"])):
        q = seqs[int(W["seq"][j])]; c = int(W["centre"][j]); a = window_frames(c, len(q["labels"]), 16, 16)
        if not np.array_equal(a, W["s"][j]) or int(W["marker"][j]) != tm.marker_slot(a, c):
            raise SystemExit("A31's windows differ from window_frames / marker_slot")
        p = ls.slot_plan(a, q["refs"], fr[int(W["seq"][j])]); p["marker"] = ls.marker32(p["coord"], c); plans.append(p)
    return {"seqs": seqs, "W": W, "plans": plans}


def audit(ctx: Ctx) -> str:
    paths = load_paths(); rec = {"sources": {}, "windows": {}}; vids = set(); fps_bad = []
    for f in FOLDS:
        seqs = t31.load_pool(ctx.a31, f); vids |= {r[0] for q in seqs for r in q["refs"]}
    data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); cnt = Counter(); bad = []
    for v in sorted(vids):
        n_img = len(list_frame_paths(paths.ev9v_images / v)); fi = np.asarray(data.frame_idx(v)); st = sorted(set(np.diff(fi).tolist())) if len(fi) > 1 else []
        mp4 = paths.ev9v_images.parent / "Videos" / f"{v}.mp4"
        pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate,time_base,nb_frames", "-of", "json", str(mp4)], capture_output=True, text=True).stdout or "{}").get("streams", [{}])[0] if mp4.is_file() else {}
        ok = st in ([3], []) and fi.max() < n_img and (not pr or int(pr.get("nb_frames", n_img)) == n_img)
        cnt[(tuple(st), pr.get("r_frame_rate"), pr.get("avg_frame_rate"), pr.get("time_base"), bool(pr))] += 1
        if not ok:
            bad.append(v)
        if pr and pr.get("r_frame_rate") != "30/1":
            fps_bad.append(v)
    rec["sources"] = {"recordings": len(vids), "plan_step_rate_timebase_original": {str(k): n for k, n in cnt.items()}, "failing": len(bad), "not_30fps": len(fps_bad)}
    for f in FOLDS:
        F = fold_windows(ctx, f); r = Counter(); ins, distinct, even = [], [], 0
        for p in F["plans"]:
            r.update(p["reasons"]); ins.append(sum(x == "inserted" for x in p["reasons"])); distinct.append(len(set(p["I32"]))); even += p["marker"] % 2 == 0
            if p["R32"][0] != p["I32"][0] or p["R32"][31] != p["I32"][31] or any(p["R32"][2 * i] != p["I32"][2 * i] for i in range(16)):
                raise SystemExit("an even slot differs between arms")
        rec["windows"][str(f)] = {"windows": len(F["plans"]), "odd_slot_reasons": dict(r), "inserted_per_window_mean": float(np.mean(ins)), "windows_without_insert": int(sum(x == 0 for x in ins)),
                                  "distinct_frames_I32_mean": float(np.mean(distinct)), "marker_even": int(even), "marker_odd": int(len(F["plans"]) - even)}
    publish(ctx, "audit.json", rec); print(json.dumps(rec, indent=1))
    if bad:
        raise SystemExit(f"{len(bad)} sources fail the plan / frame checks")
    return f"{len(vids)} sources"


# ----------------------------------------------------------------------------------------------- model and tokens
def adapted(ctx: Ctx, device) -> tuple:
    m16 = ctx.a31.a08(SEED, device); m32, ch = ls.adapt(m16); return m16, m32, ch


def branch(ctx: Ctx, device):
    _, m32, _ = adapted(ctx, device); return ic.branch(m32, True)


def opt_for(m):
    tr = ic.trainable(m)
    return t31.opt_of([("representation", [p for n, p in tr if not n.startswith("classifier.")], t31.LR_REP), ("classifier", [p for n, p in tr if n.startswith("classifier.")], t31.LR_HEAD)])


def thw32(m, device) -> tuple:
    with torch.no_grad():
        _, thw = ic.prefix(m, torch.zeros(1, 3, ls.N_SLOTS, 224, 224, device=device), tm.marker_planes(torch.tensor([16], device=device), n_slots=ls.N_SLOTS))
    return tuple(int(x) for x in thw)


def head(m, tok: torch.Tensor, thw: tuple, device) -> torch.Tensor:
    return m.classifier(ic.suffix(m, tok.to(device, non_blocking=True), thw))


def window_inputs(F: dict, idx: np.ndarray, arm: str, frames: np.ndarray, at: dict, device):
    ref = np.asarray([[at[r] for r in F["plans"][j][arm]] for j in idx]); mk = torch.as_tensor([F["plans"][j]["marker"] for j in idx], device=device)
    return tm.to_rgb(torch.from_numpy(frames[ref]).to(device)), tm.marker_planes(mk, n_slots=ls.N_SLOTS)


def chunk_frames(F: dict, idx: np.ndarray, arm: str, fp, workers: int) -> tuple:
    refs = sorted({r for j in idx for r in F["plans"][j][arm]}); return rr.decode_frames(refs, fp, workers), {r: i for i, r in enumerate(refs)}


def online_tokens(m, F: dict, idx: np.ndarray, arm: str, fp, device) -> tuple:
    frames, at = chunk_frames(F, idx, arm, fp, 0); xs = []
    with torch.no_grad():
        for b in range(0, len(idx), BATCH):
            rgb, mk = window_inputs(F, idx[b:b + BATCH], arm, frames, at, device); x, _ = ic.prefix(m, rgb, mk); xs.append(x.float().cpu())
    return torch.cat(xs)


def step(m, opt, tok, inv, w, ix, y, thw, device, u: int, sch: dict, micro: int | None = MICRO) -> dict:
    """One update; interval microbatches keep each interval's windows and sum to the batch mean."""
    t31.sched(opt, u, sch); params = [p for _, p in ic.trainable(m)]; opt.zero_grad(set_to_none=True); inv, w, ix, y = (x.to(device) for x in (inv, w, ix, y)); n = int(y.shape[0]); lv = 0.0
    tok = tok.to(device)                     # one upload per update; microbatches index it on the GPU
    for a in range(0, n, micro or n):
        sel = (ix >= a) & (ix < a + (micro or n)); r = torch.unique(inv[sel]); loc = torch.searchsorted(r, inv[sel])
        part = t29.pooled_ce(head(m, tok[r], thw, device)[loc], w[sel], ix[sel] - a, y[a:a + (micro or n)]).sum() / n; part.backward(); lv += float(part.detach())
    gn = float(torch.nn.utils.clip_grad_norm_(params, t31.CLIP)); opt.step(); return {"u": int(u), "loss": lv, "grad_norm": gn}


class Cache:
    """The arm's prefix cache. Rows are read with positioned reads (os.preadv) from parallel threads straight into one
    buffer: the same bytes as the memory map, without page-fault storms or inter-process copies."""

    def __init__(self, ctx: Ctx, arm: str, threads: int = 8) -> None:
        p = ctx.out / "cache" / f"prefix_{arm}.npy"; self.rec = json.loads((ctx.out / "cache" / f"prefix_{arm}.json").read_text()); self.mm = np.load(p, mmap_mode="r")
        self.fd, self.base, self.row = os.open(p, os.O_RDONLY), self.mm.offset, int(np.prod(self.mm.shape[1:])) * 4; self.pool = ThreadPoolExecutor(threads)

    def _read(self, buf: np.ndarray, row: int) -> None:
        mv, pos, got = memoryview(buf), self.base + row * self.row, 0
        while got < self.row:
            k = os.preadv(self.fd, [mv[got:]], pos + got)
            if k <= 0:
                raise OSError("short read from the prefix cache")
            got += k

    def get(self, rows: np.ndarray) -> torch.Tensor:
        rows = np.asarray(rows, np.int64); out = np.empty((len(rows), *self.mm.shape[1:]), np.float32); flat = out.reshape(len(rows), -1).view(np.uint8)
        list(self.pool.map(lambda i: self._read(flat[i], int(rows[i])), range(len(rows)))); return torch.from_numpy(out)


def prefetched(ctx: Ctx, arm: str, R: dict, us, depth: int = 2):
    """Update batches built by a background thread of this process, `depth` ahead of the GPU."""
    c = Cache(ctx, arm); q: queue.Queue = queue.Queue(maxsize=depth); end = object()

    def work() -> None:
        try:
            for u in us:
                rows, w, ix, y = t29.batch_rows(R, R["bt"][u], R["bu"][u]); uq, inv = np.unique(rows.numpy(), return_inverse=True)
                q.put((u, c.get(uq), torch.from_numpy(inv), w, ix, y))
            q.put(end)
        except BaseException as e:  # noqa: BLE001 - handed to the consumer
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not end:
        if isinstance(item, BaseException):
            raise item
        yield item


class Upd(Dataset):
    def __init__(self, ctx: Ctx, arm: str, R: dict, us) -> None:
        self.ctx, self.arm, self.R, self.us, self.c = ctx, arm, R, list(us), None

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        if self.c is None:
            self.c = Cache(self.ctx, self.arm)
        u = self.us[i]; rows, w, ix, y = t29.batch_rows(self.R, self.R["bt"][u], self.R["bu"][u]); uq, inv = np.unique(rows.numpy(), return_inverse=True)
        return u, self.c.get(uq), torch.from_numpy(inv), w, ix, y


# ----------------------------------------------------------------------------------------------- check (GPU)
def check(ctx: Ctx, device, workers: int) -> str:
    res = {"config_sha256": sha256_file(ROOT / ctx.config)}; a31 = ctx.a31; m16, m32, ch = adapted(ctx, device); params = a31.a27.params(SEED); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(a31.a15.a10.a07p)
    a08_sha0 = tet.state_digest(m16); res["adaptation"] = {"changed_tensors": ch, "strict_load": True, "other_tensors_bit_equal": all(torch.equal(v, m32.state_dict()[k]) for k, v in m16.state_dict().items() if not k.endswith("rel_pos_t"))}
    thw = thw32(m32, device); res["token_grid_block15"] = {"thw": thw, "tokens": 1 + thw[0] * thw[1] * thw[2], "bytes_per_window_fp32": (1 + thw[0] * thw[1] * thw[2]) * 768 * 4}
    out, nw = {}, {}
    for f in FOLDS:
        F = fold_windows(ctx, f); W = F["W"]; O = t31.outputs(a31, SEED, f); R = t31.runs(a31, SEED, f); fp = t28.frame_paths(F["seqs"])
        if not (np.array_equal(O["seq"], W["seq"]) and np.array_equal(O["centre"], W["centre"])):
            raise SystemExit("A31 outputs are not in this window order")
        pick = np.random.default_rng([20261025, 33, 9, f]).choice(len(W["seq"]), 32, replace=False); nw[f] = len(W["seq"])
        refs16 = {int(j): [tuple(F["seqs"][int(W["seq"][j])]["refs"][k]) for k in W["s"][j]] for j in pick}; uniq = sorted({r for v in refs16.values() for r in v}); fr16 = rr.decode_frames(uniq, fp, 0); at16 = {r: i for i, r in enumerate(uniq)}
        with torch.no_grad():
            ref16 = np.asarray([[at16[r] for r in refs16[int(j)]] for j in pick])
            z16 = m16(tm.to_rgb(torch.from_numpy(fr16[ref16]).to(device)), tm.marker_planes(torch.from_numpy(W["marker"][pick]).long().to(device))).float().cpu().numpy()
        pz = rr.per_sample(W, O["logits"], F["seqs"]); B = decode(params, pz); split_at = np.cumsum([len(q["labels"]) for q in F["seqs"]])[:-1]
        base_ok = np.array_equal(np.concatenate([B[q["id"]] for q in F["seqs"]]), R["base_labels"])
        P = {"seqs": F["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(F["seqs"]))), f"heldout_f{f}"); sr = t31.seq_rows(a8.cfg, P, dict(zip([q["id"] for q in F["seqs"]], np.split(R["base_labels"], split_at))), ev)
        old = json.loads((a31.out / f"readout_s{SEED}.json").read_text())["models"][f"fit_f{1 - f}_heldout_f{f}"]["rows"]["Base"]
        o = {"base16_full_forward_vs_A31_logits_max_abs (32 windows)": float(np.abs(z16 - O["logits"][pick]).max()), "base16_argmax_mismatch": int((z16.argmax(1) != O["logits"][pick].argmax(1)).sum()),
             "base_runs_reproduced": bool(base_ok), "base_sequences_equal_A31": bool(sr["row"] == old["sequence_row"] and rsh.counts_of(F["seqs"], sr) == old["counts"])}
        sub = pick[:8]; brm = ic.branch(m32, True)
        for arm in ARMS:
            frames, at = chunk_frames(F, sub, arm, fp, 0); rgb, mk = window_inputs(F, sub, arm, frames, at, device)
            with torch.no_grad():
                full = m32(rgb, mk).float(); x, th = ic.prefix(brm, rgb, mk); ps = brm.classifier(ic.suffix(brm, x, th)).float(); fly = ls.on_the_fly(m16)(rgb, mk).float()
            o[arm] = {"prefix_suffix_vs_full_max_abs": float((ps - full).abs().max()), "argmax_equal": bool(torch.equal(ps.argmax(1), full.argmax(1))), "adapted_vs_runtime_resize_max_abs": float((full - fly).abs().max()), "thw": list(th)}
        out[str(f)] = o
    res["parity"] = out
    # objective, microbatches, binding, frozen weights, forced equality, resume (fit fold 0, plan row 0, online prefix of R32 and I32)
    F0 = fold_windows(ctx, 0); R0 = t31.runs(a31, SEED, 0); fp0 = t28.frame_paths(F0["seqs"]); off = 0; sch = {"updates": 900, "warmup": 90}
    rows, w, ix, y = t29.batch_rows(R0, R0["bt"][0], R0["bu"][0]); uq, inv = np.unique(rows.numpy(), return_inverse=True); inv = torch.from_numpy(inv)
    tok = {arm: online_tokens(m32, F0, uq - off, arm, fp0, device) for arm in ARMS}
    z = torch.randn(9, 5, dtype=torch.float64); ww = torch.tensor([.5, .3, .2, 1., .25, .25, .25, .25, 1.], dtype=torch.float64); ii = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3]); yy = torch.tensor([1, 4, 0, 2]); p_ = torch.softmax(z, -1).numpy()
    res["pooled_ce_numpy_max_abs"] = float(np.abs(t29.pooled_ce(z, ww, ii, yy).numpy() - np.asarray([-np.log(sum(ww[k].item() * p_[k, yy[j]] for k in range(9) if ii[k] == j)) for j in range(4)])).max())
    grads = {}
    for name, micro in (("micro8", 8), ("micro4", 4)):
        mm = branch(ctx, device); mm.zero_grad(set_to_none=True); iv, w_, ix_, y_ = (x.to(device) for x in (inv, w, ix, y)); n = int(y_.shape[0])
        for a in range(0, n, micro):
            sel = (ix_ >= a) & (ix_ < a + micro); r = torch.unique(iv[sel]); loc = torch.searchsorted(r, iv[sel]); (t29.pooled_ce(head(mm, tok["R32"][r.cpu()], thw, device)[loc], w_[sel], ix_[sel] - a, y_[a:a + micro]).sum() / n).backward()
        grads[name] = {k: p.grad.detach().clone() for k, p in ic.trainable(mm) if p.grad is not None}
        if name == "micro8":
            res["zero_grad_tensors"] = sorted(k for k, p in ic.trainable(mm) if p.grad is None or float(p.grad.abs().sum()) == 0)
    res["microbatch_grad_max_abs"] = float(max((grads["micro8"][k] - grads["micro4"][k]).abs().max() for k in grads["micro8"])); res["grad_max"] = float(max(v.abs().max() for v in grads["micro8"].values()))
    ma = branch(ctx, device); oa = opt_for(ma); pd0 = t28.prefix_digest(ma); st0 = {k: p.detach().clone() for k, p in ic.trainable(ma)}
    la = [step(ma, oa, tok["R32"], inv, w, ix, y, thw, device, u, sch) for u in range(3)]
    mf = branch(ctx, device); of = opt_for(mf); forced = [step(mf, of, tok["R32"].clone(), inv, w, ix, y, thw, device, u, sch) for u in range(3)]    # the I32 path fed R32 inputs
    mi = branch(ctx, device); oi = opt_for(mi); li = [step(mi, oi, tok["I32"], inv, w, ix, y, thw, device, u, sch) for u in range(3)]
    pa, pf, pi = dict(ic.trainable(ma)), dict(ic.trainable(mf)), dict(ic.trainable(mi))
    res["forced_equal"] = {"loss_max_abs": float(max(abs(x["loss"] - y_["loss"]) for x, y_ in zip(la, forced))), "params_max_abs": float(max((pa[k] - pf[k]).detach().abs().max() for k in pa)),
                           "I32_vs_R32_tokens_max_abs": float((tok["I32"] - tok["R32"]).abs().max()), "I32_vs_R32_loss_first": [la[0]["loss"], li[0]["loss"]], "I32_vs_R32_params_max_abs": float(max((pa[k] - pi[k]).detach().abs().max() for k in pa))}
    res["updates_real"] = {"params_moved_max_abs": float(max((pa[k] - st0[k]).detach().abs().max() for k in pa)), "frozen_prefix_unchanged": t28.prefix_digest(ma) == pd0, "optimizer_binds_exactly": sorted(id(p) for g in oa.param_groups for p in g["params"]) == sorted(id(p) for _, p in ic.trainable(ma)),
                           "trainable_nominal": int(sum(p.numel() for _, p in ic.trainable(ma)))}
    mb = branch(ctx, device); ob = opt_for(mb); [step(mb, ob, tok["R32"], inv, w, ix, y, thw, device, u, sch) for u in range(2)]; sv = {"t": {k: p.detach().clone() for k, p in ic.trainable(mb)}, "o": ob.state_dict()}
    mc = branch(ctx, device); oc = opt_for(mc)
    with torch.no_grad():
        for k, p in ic.trainable(mc):
            p.copy_(sv["t"][k])
    oc.load_state_dict(sv["o"]); step(mc, oc, tok["R32"], inv, w, ix, y, thw, device, 2, sch); pc = dict(ic.trainable(mc)); res["resume_params_max_abs"] = float(max((pa[k] - pc[k]).detach().abs().max() for k in pa))
    # timing: prefix (decode + forward) per window for both arms; one update on a real batch; cache-read throughput
    tim = {}
    for arm in ARMS:
        idx = np.arange(400); torch.cuda.synchronize(); t0 = time.time(); frames, at = chunk_frames(F0, idx, arm, fp0, workers); td = time.time() - t0; t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(idx), BATCH):
                rgb, mk = window_inputs(F0, idx[b:b + BATCH], arm, frames, at, device); x, _ = ic.prefix(m32, rgb, mk); _ = m32.classifier(ic.suffix(m32, x, thw)); _ = x.float().cpu().numpy()
        torch.cuda.synchronize(); tim[arm] = {"decode_s_per_window": td / len(idx), "prefix_s_per_window": (time.time() - t0) / len(idx)}
    mt = branch(ctx, device); ot = opt_for(mt); torch.cuda.synchronize(); t0 = time.time()
    for u in range(5):
        step(mt, ot, tok["R32"], inv, w, ix, y, thw, device, u, sch)
    torch.cuda.synchronize(); tim["update_s_gpu"] = (time.time() - t0) / 5; tim["update_windows"] = int(len(uq)); tim["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    probe = ctx.out / "cache" / "io_probe.npy"; probe.parent.mkdir(parents=True, exist_ok=True); per = res["token_grid_block15"]["bytes_per_window_fp32"]; nprobe = 2000
    mmp = np.lib.format.open_memmap(probe, mode="w+", dtype=np.float32, shape=(nprobe, per // 3072, 768)); mmp[:] = 1.0; mmp.flush(); del mmp
    subprocess.run(["sync"]); r_ = np.load(probe, mmap_mode="r"); g = np.random.default_rng(1); t0 = time.time()
    for _ in range(3):
        sel = np.sort(g.choice(nprobe, len(uq), replace=False)); _ = np.asarray(r_[sel])
    tim["cache_read_s_per_update_upper"] = (time.time() - t0) / 3; del r_; probe.unlink()
    res["timing"] = tim
    # projection and schedule (decided once, before any formal update)
    nw = int(nw[0] + nw[1])     # decode was timed with the cache stage's workers; the probe file was just written (page cache), so IO enters as a 1.25 margin
    cache_h = {arm: nw * (tim[arm]["decode_s_per_window"] + tim[arm]["prefix_s_per_window"]) / 3600 * 1.15 for arm in ARMS}
    upd_s = tim["update_s_gpu"] * 1.25; readout_h = 2 * nw * tim["R32"]["prefix_s_per_window"] * 0.3 / 3600
    spent = t31.vsh.ledger_hours(ctx.ledger, "gpu")
    proj = {U: spent + sum(cache_h.values()) + 4 * U * upd_s / 3600 + readout_h + 0.15 for U in (900, 600)}
    free = shutil.disk_usage(ctx.out).free; need = nw * per
    U = 900 if proj[900] <= 4.0 else (600 if proj[600] <= 4.0 else None)
    res["budget"] = {"gpu_spent_before_check": spent, "cache_hours": cache_h, "update_seconds": upd_s, "readout_hours": readout_h, "projection": proj, "cap": 4.0, "windows": nw,
                     "disk_free_gb": free / 1e9, "cache_need_gb": need / 1e9, "disk_ok": free > need + 15e9}
    res["a08_unchanged"] = tet.state_digest(m16) == a08_sha0
    po = res["parity"]
    res["ok"] = bool(all(v["base16_full_forward_vs_A31_logits_max_abs (32 windows)"] <= 1e-4 and v["base16_argmax_mismatch"] == 0 and v["base_runs_reproduced"] and v["base_sequences_equal_A31"]
                         and all(v[a]["prefix_suffix_vs_full_max_abs"] <= 1e-4 and v[a]["argmax_equal"] and v[a]["adapted_vs_runtime_resize_max_abs"] <= 1e-4 for a in ARMS) for v in po.values())
                     and len(ch) == 16 and res["adaptation"]["other_tensors_bit_equal"] and res["pooled_ce_numpy_max_abs"] < 1e-10 and res["microbatch_grad_max_abs"] <= 1e-5 * max(1.0, res["grad_max"])
                     and res["forced_equal"]["loss_max_abs"] == 0 and res["forced_equal"]["params_max_abs"] == 0 and res["forced_equal"]["I32_vs_R32_tokens_max_abs"] > 0
                     and res["updates_real"]["params_moved_max_abs"] > 0 and res["updates_real"]["frozen_prefix_unchanged"] and res["updates_real"]["optimizer_binds_exactly"] and res["resume_params_max_abs"] <= 1e-6 and res["a08_unchanged"])
    res["schedule_decision"] = {"updates": U, "warmup": None if U is None else U // 10, "fallback_used": U == 600, "feasible": U is not None and res["budget"]["disk_ok"]}
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps(res, indent=1, default=str)[:7000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    if not res["schedule_decision"]["feasible"]:
        raise SystemExit("neither 900 nor 600 updates fit, or the disk cannot hold one arm's cache")
    write_json(ctx.out / "schedule.json", {"updates": U, "warmup": U // 10, "fixed_at": now(), "reason": "engineering projection in the check stage, before any formal update"}); return f"schedule {U}"


# ----------------------------------------------------------------------------------------------- cache, train, readout (GPU)
def cache(ctx: Ctx, arm: str, device, workers: int) -> str:
    sch = ctx.schedule(); _, m32, ch = adapted(ctx, device); m = ic.branch(m32, True); d0 = tet.state_digest(m); thw = thw32(m, device); Fs = {f: fold_windows(ctx, f) for f in FOLDS}
    n = {f: len(Fs[f]["plans"]) for f in FOLDS}; off = {0: 0, 1: n[0]}; ntok = 1 + thw[0] * thw[1] * thw[2]; need = (n[0] + n[1]) * ntok * 768 * 4; cdir = ctx.out / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cdir).free < need + 15e9:
        raise SystemExit("not enough disk for this arm's cache")
    mm = np.lib.format.open_memmap(cdir / f"prefix_{arm}.npy", mode="w+", dtype=np.float32, shape=(n[0] + n[1], ntok, 768)); init = np.zeros((n[0] + n[1], 5), np.float32); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    for f in FOLDS:
        F = Fs[f]; W = F["W"]; fp = t28.frame_paths(F["seqs"])
        for c0 in range(0, len(F["seqs"]), 150):
            idx = np.flatnonzero((W["seq"] >= c0) & (W["seq"] < c0 + 150)); a = time.time(); frames, at = chunk_frames(F, idx, arm, fp, workers); acc["decode_seconds"] += time.time() - a
            for b in range(0, len(idx), BATCH):
                ii = idx[b:b + BATCH]; torch.cuda.synchronize(); a = time.time()
                with torch.no_grad():
                    rgb, mk = window_inputs(F, ii, arm, frames, at, device); x, _ = ic.prefix(m, rgb, mk); z = m.classifier(ic.suffix(m, x, thw))
                torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a; mm[off[f] + ii] = x.float().cpu().numpy(); init[off[f] + ii] = z.float().cpu().numpy()
            del frames
            print(f"{arm} fold {f}: {int(idx.max()) + 1}/{n[f]} windows; GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s", flush=True)
    mm.flush(); del mm
    if tet.state_digest(m) != d0:
        raise SystemExit("the branch changed during caching")
    np.savez(cdir / f"init_{arm}.npz", logits=init, offsets=np.asarray([off[0], off[1], n[0] + n[1]]))
    write_json(cdir / f"prefix_{arm}.json", {"arm": arm, "rows": n, "offsets": off, "thw": thw, "tokens": ntok, "changed_tensors": [c["name"] for c in ch], "time": now(), **acc, "schedule": sch,
                                             "init_sha256": sha256_file(cdir / f"init_{arm}.npz")})
    return f"{n[0] + n[1]} windows"


def train(ctx: Ctx, arm: str, device, workers: int) -> None:
    sch = ctx.schedule(); thw = tuple(json.loads((ctx.out / "cache" / f"prefix_{arm}.json").read_text())["thw"])
    for f in FOLDS:
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            rd = ctx.out / "train" / f"{arm}_f{f}"; rd.mkdir(parents=True, exist_ok=True); R = t31.runs(ctx.a31, SEED, f); m = branch(ctx, device); o = opt_for(m); pd0 = t28.prefix_digest(m); log = []; gs = 0.0
            for u, tok, inv, w, ix, y in prefetched(ctx, arm, R, range(int(sch["updates"]))):
                torch.cuda.synchronize(); a = time.time(); log.append(step(m, o, tok, inv, w, ix, y, thw, device, int(u), sch)); torch.cuda.synchronize(); gs += time.time() - a
                if (int(u) + 1) % 90 == 0:
                    print(f"[{arm}_f{f}] update {int(u) + 1}/{sch['updates']}: loss {np.mean([x['loss'] for x in log[-90:]]):.4f}  {time.time() - t0:.0f} s", flush=True)
            if t28.prefix_digest(m) != pd0:
                raise SystemExit("a frozen prefix parameter changed")
            torch.save({n_: p.detach().cpu().clone() for n_, p in ic.trainable(m)}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
            man = {"arm": arm, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "loss_first_90": float(np.mean([x["loss"] for x in log[:90]])), "loss_last_90": float(np.mean([x["loss"] for x in log[-90:]])),
                   "seconds": time.time() - t0, "gpu_seconds": gs, "schedule": sch}
            write_json(rd / "run_manifest.json", man); print(json.dumps(man), flush=True); ok = True
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {arm}_f{f} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def timing(ctx: Ctx, arm: str, device, n: int = 30) -> str:
    """Engineering timing of the training loop on the real cache: a fresh branch, plan rows 0..n-1 of fold 0, nothing saved.
    Also checks that the positioned reads return the memory map's bytes."""
    sch = {"updates": 900, "warmup": 90}; thw = tuple(json.loads((ctx.out / "cache" / f"prefix_{arm}.json").read_text())["thw"]); R = t31.runs(ctx.a31, SEED, 0); c = Cache(ctx, arm)
    rows = np.sort(np.random.default_rng(5).choice(c.mm.shape[0], 64, replace=False)); same = bool(np.array_equal(c.get(rows).numpy(), np.asarray(c.mm[rows])))
    m = branch(ctx, device); o = opt_for(m); k = 0
    for u, tok, inv, w, ix, y in prefetched(ctx, arm, R, range(n)):
        step(m, o, tok, inv, w, ix, y, thw, device, int(u), sch); k += 1
        if k == 5:
            torch.cuda.synchronize(); t5 = time.time()
    torch.cuda.synchronize(); s_ = (time.time() - t5) / (k - 5); rec = {"arm": arm, "updates_timed": k - 5, "seconds_per_update": s_, "reads_equal_memmap": same, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "discarded": True}
    write_json(ctx.out / f"timing_{arm}.json", rec); print(json.dumps(rec), flush=True)
    if not same:
        raise SystemExit("positioned reads differ from the memory map")
    return f"{s_:.2f} s/update"


def load_trained(ctx: Ctx, arm: str, f: int, device):
    rd = ctx.out / "train" / f"{arm}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_f{f}: weights not committed or changed")
    st = torch.load(rd / "final.pt", map_location="cpu", weights_only=True); m = branch(ctx, device)
    with torch.no_grad():
        for k, p in ic.trainable(m):
            p.copy_(st[k].to(p.device))
    return m.eval()


def readout(ctx: Ctx, arm: str, device) -> str:
    rec = json.loads((ctx.out / "cache" / f"prefix_{arm}.json").read_text()); thw = tuple(rec["thw"]); C = Cache(ctx, arm); off = {int(k): v for k, v in rec["offsets"].items()}; n = {int(k): v for k, v in rec["rows"].items()}
    for f in FOLDS:
        h = 1 - f; p = ctx.out / f"heldout_{arm}_f{f}.npz"
        if p.exists():
            raise SystemExit("this held-out readout exists; it runs once")
        m = load_trained(ctx, arm, f, device); z = np.zeros((n[h], 5), np.float32)
        with torch.no_grad():
            for b in range(0, n[h], 256):
                r = np.arange(b, min(n[h], b + 256)) + off[h]; z[r - off[h]] = head(m, C.get(r), thw, device).float().cpu().numpy()
        np.savez(p, logits=z, fit_fold=f, heldout_fold=h); print(f"{arm}_f{f} -> fold {h}: {n[h]} windows", flush=True)
    return "held-out logits"


def clean(ctx: Ctx, arm: str) -> str:
    if not all((ctx.out / f"heldout_{arm}_f{f}.npz").exists() for f in FOLDS) or not (ctx.out / "cache" / f"init_{arm}.npz").exists():
        raise SystemExit("readouts are missing; the cache stays")
    p = ctx.out / "cache" / f"prefix_{arm}.npy"; sz = p.stat().st_size; p.unlink(); return f"deleted {sz / 1e9:.1f} GB"


# ----------------------------------------------------------------------------------------------- report (CPU)
MODELS = ("Base16", "Init_R32", "Init_I32", "R32", "I32")


def fold_logits(ctx: Ctx, h: int, n: dict, off: dict) -> dict:
    """Per-window logits of every model on held-out fold h (fold-local window order)."""
    z = {"Base16": t31.outputs(ctx.a31, SEED, h)["logits"]}
    for arm in ARMS:
        with np.load(ctx.out / "cache" / f"init_{arm}.npz") as q:
            z[f"Init_{arm}"] = q["logits"][off[h]:off[h] + n[h]]
        with np.load(ctx.out / f"heldout_{arm}_f{1 - h}.npz") as q:
            if int(q["heldout_fold"]) != h:
                raise SystemExit("held-out file of the wrong fold")
            z[arm] = q["logits"]
    return z


def model_tables(z: np.ndarray, h: int, off: dict, n: dict, D: dict) -> dict:
    """Run table, set coverage, change table and complete sequences of one model on held-out fold h."""
    import scripts.diagnose_a5c_evidence as d32  # noqa: E402
    pk, seqs, W = D["pk"], D["seqs"], D["W"]; full = torch.zeros(off[1] + n[1], 5); full[off[h]:off[h] + n[h]] = torch.from_numpy(z)
    pp = t29.pooled_probs(full, pk, np.arange(len(pk["labels"]))); pred = pp.argmax(1); y, b = pk["labels"], pk["base"]; M = D["sets"]
    ok5 = (y == 3) & (b == 3); s1 = M["S1"]; s2 = M["S2"]
    runs_ = {**t31.run_table(pp, pk, seqs, D["sess"]), "changes": d32.change_table(y, b, pred),
             "correct_A5C_kept": {"runs": int(ok5.sum()), "kept": int((ok5 & (pred == 3)).sum())}, "A32_S1_fixed": {"runs": int(s1.sum()), "fixed": int((s1 & (pred == 3)).sum())},
             "A32_S2_kept": {"runs": int(s2.sum()), "kept": int((s2 & (pred == 3)).sum())}, "A4C_read_as_A5C_fixed": {"runs": int(((y == 2) & (b == 3)).sum()), "fixed": int(((y == 2) & (b == 3) & (pred == 2)).sum())}}
    lab = W["label"]; pure = (W["labels16"] == lab[:, None]).all(1); wp = z.argmax(1)
    win = {k: {"windows": int(m_.sum()), "accuracy": float((wp[m_] == lab[m_]).mean())} for k, m_ in (("pure", pure), ("mixed", ~pure))}
    ps = rr.per_sample(W, z, seqs); preds = D["base_preds"] if D["is_base"] else {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in seqs}
    sr = t31.seq_rows(D["a8cfg"], D["P"], preds, D["ev"]); return {"runs": runs_, "windows": win, "sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr), "pred": pred}


def report(ctx: Ctx) -> str:
    rec = json.loads((ctx.out / "cache" / "prefix_R32.json").read_text()); off = {int(k): v for k, v in rec["offsets"].items()}; n = {int(k): v for k, v in rec["rows"].items()}
    a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); sp = np.load(ROOT / "runs/a5c_source_external_evidence_v1/sets_private.npy", allow_pickle=True).item()[SEED]
    out = {"per_fold": {}, "oof": {}, "comparison": {}, "decision": {}}; preds = {m: [] for m in MODELS}; ys, bs, kinds, sess_all = [], [], [], []
    for h in FOLDS:
        seqs = t31.load_pool(ctx.a31, h); W = t31.pool_W(seqs); R = t31.runs(ctx.a31, SEED, h); pk = R["P"]; key = [(int(i), int(a), int(b)) for i, a, b in zip(pk["seq"], pk["a"], pk["b"])]
        if key != sp[h]["key"]:
            raise SystemExit("A32 set masks are not on these runs")
        split_at = np.cumsum([len(q["labels"]) for q in seqs])[:-1]; P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}
        D = {"pk": pk, "seqs": seqs, "W": W, "sets": sp[h]["masks"], "sess": np.load(ctx.a31.out / "data" / f"run_sessions_s{SEED}_f{h}.npy", allow_pickle=True), "a8cfg": a8.cfg, "P": P,
             "ev": bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"), "base_preds": dict(zip([q["id"] for q in seqs], np.split(R["base_labels"], split_at)))}
        z = fold_logits(ctx, h, n, off); res = {}
        for m in MODELS:
            D["is_base"] = m == "Base16"; t = model_tables(z[m], h, off, n, D); preds[m].append(t.pop("pred")); res[m] = t
        out["per_fold"][f"heldout_f{h}"] = res; ys.append(pk["labels"]); bs.append(pk["base"]); kinds.append(np.asarray([seqs[i]["kind"] for i in pk["seq"]])); sess_all.append(np.char.add(f"f{h}:", D["sess"].astype(str)))
    y, b, kind, sess = (np.concatenate(x) for x in (ys, bs, kinds, sess_all)); wr = b != y
    for m in MODELS:
        p = np.concatenate(preds[m]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); ok5 = (y == 3) & (b == 3)
        out["oof"][m] = {"runs": int(len(y)), "base_wrong": int(wr.sum()), "fixed": fx, "broken": br, "net_fixed": fx - br, "correct_A5C_kept": int((ok5 & (p == 3)).sum()), "correct_A5C": int(ok5.sum()),
                         "A5C_read_as_A4C_fixed": int(((y == 3) & (b == 2) & (p == 3)).sum()), "A4C_read_as_A5C_fixed": int(((y == 2) & (b == 3) & (p == 2)).sum())}
    for h in FOLDS:
        F_ = out["per_fold"][f"heldout_f{h}"]; I, Rr = F_["I32"], F_["R32"]
        out["comparison"][f"heldout_f{h}"] = {"net_fixed": I["runs"]["all"]["net_fixed"] - Rr["runs"]["all"]["net_fixed"], "segment_f1_50_multi": I["sequence_row"]["segment_f1_50_multi"] - Rr["sequence_row"]["segment_f1_50_multi"],
                                              "native_macro_f1": I["sequence_row"]["native_macro_f1"] - Rr["sequence_row"]["native_macro_f1"], "missed_short_true": I["counts"]["fn_short"] - Rr["counts"]["fn_short"],
                                              "short_false_positives": I["counts"]["fp_short"] - Rr["counts"]["fp_short"], "correct_A5C_kept": I["runs"]["correct_A5C_kept"]["kept"] - Rr["runs"]["correct_A5C_kept"]["kept"],
                                              "A5C_read_as_A4C_fixed": I["runs"]["A5C_to_A4C_errors"]["fixed"] - Rr["runs"]["A5C_to_A4C_errors"]["fixed"], "A4C_read_as_A5C_fixed": I["runs"]["A4C_to_A5C_errors"]["fixed"] - Rr["runs"]["A4C_to_A5C_errors"]["fixed"]}
    pI, pR = np.concatenate(preds["I32"]), np.concatenate(preds["R32"]); nat = kind == "native"; d = ((pI == y).astype(float) - (pR == y).astype(float))[nat]
    us = np.unique(sess[nat]); gi = np.searchsorted(us, sess[nat]); g = np.random.default_rng([20261025, 33, 77]); dr = g.integers(0, len(us), (2000, len(us))); wts = np.stack([np.bincount(x, minlength=len(us)) for x in dr]).astype(float)
    per = np.bincount(gi, weights=d, minlength=len(us)); cnt = np.bincount(gi, minlength=len(us)).astype(float)
    out["native_I32_minus_R32_run_accuracy"] = {"unit": "session (native runs: one recording, one session)", "sessions": int(len(us)), "difference": float(d.mean()), "ci95_descriptive": np.percentile((wts @ per) / (wts @ cnt), [2.5, 97.5]).tolist()}
    c = out["comparison"]; both = lambda k, fn: all(fn(c[f"heldout_f{h}"][k]) for h in FOLDS)  # noqa: E731
    rule = {"net_fixed_ge_10": both("net_fixed", lambda v: v >= 10), "segment_f1_not_lower": both("segment_f1_50_multi", lambda v: v >= 0), "missed_short_le_3": both("missed_short_true", lambda v: v <= 3), "correct_A5C_kept_ge_minus5": both("correct_A5C_kept", lambda v: v >= -5)}
    close = (not both("segment_f1_50_multi", lambda v: v >= 0)) or (not both("net_fixed", lambda v: v > 0)) or (both("A4C_read_as_A5C_fixed", lambda v: v > 0) and not both("correct_A5C_kept", lambda v: v >= 0))
    out["decision"] = {"seed2_rule": rule, "seed2_rule_met": all(rule.values()), "close_configuration": bool(close and not all(rule.values()))}
    publish(ctx, "report.json", out); print(json.dumps({"oof": out["oof"], "comparison": out["comparison"], "decision": out["decision"]}, indent=1)); return "report"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("audit", "check", "cache", "timing", "train", "readout", "clean", "report"), required=True)
    ap.add_argument("--arm", choices=ARMS); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("audit", "report"):
        return staged(ctx, f"{a.stage} (CPU)", "cpu", lambda: audit(ctx) if a.stage == "audit" else report(ctx))
    if a.stage == "clean":
        return staged(ctx, f"clean {a.arm} (CPU)", "cpu", lambda: clean(ctx, a.arm))
    t31.vsh.determinism(); device = torch.device("cuda")
    if a.stage == "train":
        return train(ctx, a.arm, device, a.workers)
    fn = {"check": lambda: check(ctx, device, a.workers), "cache": lambda: cache(ctx, a.arm, device, a.workers), "readout": lambda: readout(ctx, a.arm, device), "timing": lambda: timing(ctx, a.arm, device)}[a.stage]
    staged(ctx, f"{a.stage}{' ' + a.arm if a.arm else ''} (GPU job wall time)", "gpu", fn)


if __name__ == "__main__":
    main()
