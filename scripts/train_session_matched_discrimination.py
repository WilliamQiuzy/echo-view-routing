#!/usr/bin/env python
"""A42 (configs/experiments/video_session_matched_discrimination_v1.yaml): same-session A4C / A5C margin ranking (S) against
cyclic cross-session rewiring of the same endpoint windows (C); both continue the whole A08 for 900 updates of 16 windows.

  --stage check    pair bank / plan, pixel path, Init = A08 / A31 cache / Base, rank formula and gradient path, forced C
                   partner = identity, microbatch and skip-pool parity, repeat and 3 + 3 resume, binding / movement,
                   label-free readout pooling, timing / peak memory / disk and the launch projection (GPU)
  --stage train    --seed s: C_f0, S_f0, C_f1, S_f1, resumable (GPU)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the weights are committed, one opposite-fold online readout (+ the source-2 rule) (GPU);
                   --dry-run reads Init (z = q) through the same tables and publishes nothing (CPU)
The frozen A08 + B3 path supplies q, Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.prepare_video_sequence_objective as a07  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_masked_feature as t40  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_predicted_run_context as t38  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.session_pairs as sp  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG, A32_CONFIG = t40.A31_CONFIG, t40.A32_CONFIG
FOLDS, ARMS, U, WARM, PER, SAVE_EVERY, SPLIT_EVERY = (0, 1), ("C", "S"), 900, 90, sp.PER, 150, 50
LR_ENC, LR_NEW, CLIP = 1e-5, 1e-4, 1.0
SCH = {"updates": U, "warmup": WARM}
LIMITS = {"loss": 1e-4, "params": 1e-5}
TOL = {"parity": 1e-4, "grad_rel": 1e-5, "formula": 1e-8}   # float64; torch softplus (threshold 20) deviates <= e^-20 / 4 from log1p(exp)
EXPECTED_BANK = {0: {"native_recordings": 740, "sessions": 35, "A4C_recordings": 55, "A5C_recordings": 48, "pairs": 77},
                 1: {"native_recordings": 747, "sessions": 44, "A4C_recordings": 72, "A5C_recordings": 54, "pairs": 89}}
A4C, A5C = sp.A4C, sp.A5C
now, write_json, ledger_append, tables, a5c_extras, ceil_pct = t31.now, t31.write_json, t31.ledger_append, t40.tables, t40.a5c_extras, t40.ceil_pct


def model_seed(s: int, f: int) -> int:
    return 2026102742 + 10 * int(s) + int(f)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["session_matched"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t40.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t40.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- plan and batches
class Groups(t40.Groups):
    def draw(self, s: int, f: int, u: int) -> np.ndarray:
        """Slots 0-7: A40's native group then constructed group, drawn from A42's stream 1."""
        rng = np.random.default_rng([sp.BASE, sp.ROUND, int(s), int(f), int(u), 1])
        i = int(rng.integers(len(self.nat))); j = int(rng.integers(len(self.nat[i][1]))); k = int(rng.integers(len(self.con))); l_ = int(rng.integers(len(self.con[k][1])))
        (ni, ns), (ci, cs) = self.nat[i], self.con[k]; rn, rc = self.rows[ni], self.rows[ci]
        return np.concatenate([rn[a07.group_indices(len(rn), int(ns[j]))], rc[a07.group_indices(len(rc), int(cs[l_]))]])


class Fold:
    def __init__(self, ctx: Ctx, f: int) -> None:
        self.f, self.seqs = f, t31.load_pool(ctx.a31, f); self.W = t31.pool_W(self.seqs); self.G = Groups(self.seqs, self.W); self.fp = t28.frame_paths(self.seqs)
        self.bank = sp.pair_bank(self.seqs, session_of)

    def plan(self, s: int, u: int) -> tuple[np.ndarray, np.ndarray]:
        """The 16 pool_W rows of update u and the 4 chosen bank sessions."""
        ex, ch = sp.extra_draw(self.bank, self.G.rows, s, self.f, u); return np.concatenate([self.G.draw(s, self.f, u), ex]), ch


class Batches(Dataset):
    """Item = (update, rows, clean uint8 frames (16, 16, 224, 224, 3), marker slots, centre labels) of a fit fold."""

    def __init__(self, F_: Fold, s: int, us) -> None:
        self.F, self.s, self.us = F_, s, list(us)

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        F_ = self.F; u = self.us[i]; rows = F_.plan(self.s, u)[0]; out = np.empty((PER, tm.N_SLOTS, 224, 224, 3), np.uint8); memo = {}
        for w, r in enumerate(rows):
            q = F_.seqs[int(F_.W["seq"][r])]
            for k, p in enumerate(F_.W["s"][r]):
                ref = tuple(q["refs"][int(p)])
                if ref not in memo:
                    v, fi, g = ref; pth = F_.fp(v); memo[ref] = jv.load_frame_u8(pth[min(int(fi), len(pth) - 1)], round(float(g), 6))
                out[w, k] = memo[ref]
        return u, torch.from_numpy(rows), torch.from_numpy(out), torch.from_numpy(F_.W["marker"][rows].astype(np.int64)), torch.from_numpy(F_.W["label"][rows].astype(np.int64))


def loader(F_: Fold, s: int, us, workers: int):
    return DataLoader(Batches(F_, s, us), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def plan_sha(F_: Fold, s: int, us) -> str:
    h = hashlib.sha256()
    for u in us:
        h.update(F_.plan(s, u)[0].astype(np.int64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- model, update
def build(ctx: Ctx, s: int, device, patched: bool = True):
    """The A08 copy, every parameter explicitly trainable, eval mode; the skip pools made deterministic (training copies)."""
    m = ctx.a31.a08(s, device); m.set_encoder_trainable(True)
    for p in m.parameters():
        p.requires_grad_(True)
    if patched:
        sp.deterministic_skip_pools(m.encoder)
    return m.eval()


def make_opt(m):
    return t31.opt_of([("encoder", list(m.encoder.parameters()), LR_ENC), ("new", [m.marker_stem.weight, m.classifier.weight, m.classifier.bias], LR_NEW)])


def trunk_params(m) -> list:
    """Shared trunk for the monitor: every encoder parameter except the 512 projection, plus marker_stem.weight."""
    return [p for n, p in m.encoder.named_parameters() if not n.startswith("head.")] + [m.marker_stem.weight]


def logits16(m, rgb: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
    """Microbatch 1: 16 separate forwards of the unchanged recogniser, differentiable logits kept."""
    return torch.cat([m(rgb[w:w + 1], mk[w:w + 1]) for w in range(rgb.shape[0])])


def sq_norm(gs) -> float:
    return float(sum(0.0 if g is None else float(g.double().pow(2).sum()) for g in gs)) ** 0.5


def monitor_of(z: torch.Tensor) -> dict:
    with torch.no_grad():
        d = sp.margins(z.detach()); d4, d5 = d[sp.EXTRA4], d[sp.EXTRA5]; out = {}
        for arm, par in sp.PARTNER.items():
            p = torch.stack([d4[j] for j in par]); out[f"rank_{arm}"] = float(F.softplus(-(d5 - p)).mean()); out[f"pair_acc_{arm}"] = float((d5 > p).float().mean())
        return out


def update(m, opt, b, partner, device, split: bool = False) -> dict:
    """One real update: 16 microbatch-1 logits, CE + 0.5 x rank under `partner`, one backward, one clip and one AdamW step."""
    u, _, u8, mk, y = t40.on_device(b, device); t31.sched(opt, u, SCH); opt.zero_grad(set_to_none=True)
    z = logits16(m, tm.to_rgb(u8), mk); tot, ce, rk = sp.total_loss(z, y, partner); rec = {"u": u, "ce": float(ce.detach()), "rank": float(rk.detach()), "total": float(tot.detach()), **monitor_of(z)}
    if split:
        ps = list(m.parameters()); tr = {id(p) for p in trunk_params(m)}; gr = torch.autograd.grad(sp.COEF * rk, ps, retain_graph=True, allow_unused=True)
    tot.backward()
    if split:
        gc = [None if p.grad is None else p.grad - (0 if g is None else g) for p, g in zip(ps, gr)]
        rec.update({"trunk_grad_scaled_rank": sq_norm([g for p, g in zip(ps, gr) if id(p) in tr]), "trunk_grad_ce": sq_norm([g for p, g in zip(ps, gc) if id(p) in tr]),
                    "all_grad_scaled_rank": sq_norm(gr), "all_grad_ce": sq_norm(gc)}); del gr, gc
    rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(list(m.parameters()), CLIP)); opt.step(); return rec


def run_updates(m, opt, F_: Fold, s: int, us, partner, device, workers: int, log: list, split_every: int = 0, hook=None) -> float:
    gsec = 0.0
    for b in loader(F_, s, us, workers):
        torch.cuda.synchronize(); a = time.time(); u = int(b[0]); rec = update(m, opt, b, partner, device, bool(split_every) and (u % split_every == 0 or u == U - 1))
        torch.cuda.synchronize(); gsec += time.time() - a; log.append(rec)
        if hook is not None:
            hook(u, rec)
    return gsec


def state_of(m, opt) -> dict:
    return {"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, "opt": opt.state_dict(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def load_state(m, opt, st: dict) -> None:
    m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])


def flat_state(m) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def params_max_diff(a: dict, b: dict) -> float:
    return float(max((a[k].float() - b[k].float()).abs().max() for k in a))


@torch.no_grad()
def fit_monitor(m, F_: Fold, s: int, device) -> dict:
    """Training inputs only: CE, accuracy, rank under both pairings and pair accuracy on updates 0-3's batches (64 windows)."""
    rows = []
    for b in (Batches(F_, s, range(4))[i] for i in range(4)):
        u, _, u8, mk, y = t40.on_device(b, device); z = m(tm.to_rgb(u8), mk); rows.append({"ce": float(F.cross_entropy(z, y)), "accuracy": float((z.argmax(1) == y).float().mean()), **monitor_of(z)})
    return {**{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, "batches": "updates 0-3 (eval, no_grad, batch 16)"}


def movement(m, s: int) -> dict:
    """Per-group drift from the pinned A08 (A40's drift) and the tensors that did not move at all."""
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); cur = flat_state(m)
    still = sorted(k for k in cur if cur[k].is_floating_point() and torch.equal(cur[k], a08[k]))
    return {"drift_from_A08": t40.drift(m, s), "unchanged_tensors": still, "tensors": int(sum(1 for k in cur if cur[k].is_floating_point()))}


# ----------------------------------------------------------------------------------------------- train
def identity(ctx: Ctx, arm: str, s: int, f: int, psha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(sp.__file__), Path(tm.__file__), Path(__file__), Path(t40.__file__)):
        h.update(p.read_bytes())
    spl = json.loads((ctx.a31.out / "split.json").read_text())
    h.update(json.dumps({"arm": arm, "seed": s, "fold": f, "a08": t31.A08[s], "pools": spl["pool_sha256"], "plan_sha256": psha, "config": sha256_file(ROOT / ctx.config)}, sort_keys=True).encode()); return h.hexdigest()


def train_one(ctx: Ctx, arm: str, s: int, F_: Fold, ident: str, psha: str, device, workers: int) -> dict:
    key = f"{arm}_s{s}_f{F_.f}"; rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True); rp = rd / "resume.pt"; partner = sp.PARTNER[arm]
    m = build(ctx, s, device); opt = make_opt(m); init_digest = tet.state_digest(m); before = fit_monitor(m, F_, s, device); torch.manual_seed(model_seed(s, F_.f)); start, log, resumes = 0, [], []
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["identity"] != ident:
            raise SystemExit(f"{key}: the resume file belongs to another identity")
        load_state(m, opt, st); start, log, before, init_digest = st["update"], st["log"], st["fit_before"], st["init_digest"]; resumes.append({"from_update": start, "at": now()})
    prog = ctx.out / f"progress_s{s}.json"; t0 = time.time(); torch.cuda.reset_peak_memory_stats()

    def hook(u: int, rec: dict) -> None:
        if (u + 1) % SPLIT_EVERY == 0:
            last = log[-SPLIT_EVERY:]; print(f"[{key}] update {u + 1}/{U}: CE {np.mean([x['ce'] for x in last]):.4f} rank {np.mean([x['rank'] for x in last]):.4f} pairS {np.mean([x['pair_acc_S'] for x in last]):.3f} grad {np.mean([x['grad_norm'] for x in last]):.3f}  {time.time() - t0:.0f} s", flush=True)
            write_json(prog, {"seed": s, "model": key, "update": u + 1, "total": U, "updated_at": now(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0)})
        if (u + 1) % SAVE_EVERY == 0 and u + 1 < U:
            tmp = rp.with_suffix(".tmp"); torch.save({**state_of(m, opt), "update": u + 1, "log": log, "identity": ident, "fit_before": before, "init_digest": init_digest}, tmp); tmp.replace(rp)

    gsec = run_updates(m, opt, F_, s, range(start, U), partner, device, workers, log, SPLIT_EVERY, hook)
    steps = [int(st_["step"]) for st_ in opt.state_dict()["state"].values()]
    if len(log) != U or [x["u"] for x in log] != list(range(U)) or not steps or set(steps) != {U}:
        raise SystemExit(f"{key}: {len(log)} logged updates, optimiser steps {sorted(set(steps))}")
    after = fit_monitor(m, F_, s, device); mv = movement(m, s)
    torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    spl = [x for x in log if "trunk_grad_ce" in x]
    man = {"arm": arm, "seed": s, "fit_fold": F_.f, "partner": list(partner), "coefficient": sp.COEF, "updates": len(log), "optimizer_steps": U, "identity": ident, "plan_sha256": psha, "a08_sha256": t31.A08[s],
           "init_state_digest": init_digest, "final_sha256": sha256_file(rd / "final.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "fit_before": before, "fit_after": after, **mv, "resumes": resumes,
           "gpu_seconds_this_process": gsec, "seconds_this_process": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
           "loss": {k: {"first_90": float(np.mean([x[k] for x in log[:WARM]])), "last_90": float(np.mean([x[k] for x in log[-WARM:]]))} for k in ("ce", "rank", "rank_S", "rank_C", "pair_acc_S", "pair_acc_C", "grad_norm")},
           "grad_split_every_50": [{k: x[k] for k in ("u", "trunk_grad_ce", "trunk_grad_scaled_rank", "all_grad_ce", "all_grad_scaled_rank")} for x in spl]}
    write_json(rd / "run_manifest.json", man); return man


def train(ctx: Ctx, s: int, device, workers: int) -> None:
    for f in FOLDS:
        F_ = Fold(ctx, f); psha = plan_sha(F_, s, range(U))
        for arm in ARMS:
            key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; ident = identity(ctx, arm, s, f, psha); mp = rd / "run_manifest.json"
            if mp.exists() and json.loads(mp.read_text())["identity"] == ident and sha256_file(rd / "final.pt") == json.loads(mp.read_text())["final_sha256"]:
                print(f"[{key}] finished earlier", flush=True); continue
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                man = train_one(ctx, arm, s, F_, ident, psha, device, workers); torch.cuda.empty_cache(); ok = True; note = f"final {man['final_sha256'][:12]}"
                if sha256_file(rd / "final.pt") == man["final_sha256"] and (rd / "resume.pt").exists():
                    (rd / "resume.pt").unlink()                                  # regenerable; the final weights are verified
                print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256", "fit_before", "fit_after", "loss")}), flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {key} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "models": man, "config_sha256": sha256_file(ROOT / ctx.config), "loaded_code_sha256": t38.loaded_code(), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    p = ctx.results / f"weights_s{s}.json"
    if sha256_file(p) != manifest_sha or sha256_file(ctx.out / f"weights_s{s}.json") != manifest_sha:
        raise SystemExit("the committed weight manifest differs from the server's")
    models = json.loads(p.read_text())["models"]; com = ctx.out / "weights_committed.json"; cur = json.loads(com.read_text()) if com.exists() else {}
    for k, man in models.items():
        if sha256_file(ctx.out / "train" / k / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{k}: weights differ from the manifest")
        cur[k] = man["final_sha256"]
    cur[f"commit_s{s}"] = commit; write_json(com, cur); return f"{len(models)} models pinned at {commit}"


# ----------------------------------------------------------------------------------------------- check
def plan_checks(ctx: Ctx, s: int) -> dict:
    """(1) Bank counts, the 900 plans of each fit fold, S / C edges, identical multisets, exposure and duplicates."""
    out = {}; spl = json.loads((ctx.a31.out / "split.json").read_text())
    for f in FOLDS:
        F_ = Fold(ctx, f); W, seqs = F_.W, F_.seqs; cnt = sp.bank_counts(F_.bank, seqs); grid_ok = True
        for i, q in enumerate(seqs):
            w = vsf.windows(len(q["refs"])); r = F_.G.rows[i]
            grid_ok &= bool(np.array_equal(w["centres"], W["centre"][r]) and np.array_equal(w["samples"], W["s"][r]) and np.array_equal(w["markers"], W["marker"][r]))
        pool_sess = {session_of(str(r[0])) for q in seqs for r in q["refs"]}; other = {session_of(str(r[0])) for q in t31.load_pool(ctx.a31, 1 - f) for r in q["refs"]}
        orig_ok = extra_ok = s_same = c_cross = multiset_eq = distinct4 = True; lab, dup, ex_w, ex_rec, ex_sess, sess_exp = Counter(), 0, set(), set(), set(), Counter()
        for u in range(U):
            rows, ch = F_.plan(s, u); sq = W["seq"][rows]; c = W["centre"][rows]; y = W["label"][rows]; lab.update(y.tolist())
            nat, con = seqs[int(sq[0])], seqs[int(sq[4])]
            orig_ok &= bool(len(set(sq[:4].tolist())) == 1 and len(set(sq[4:8].tolist())) == 1 and nat["kind"] == "native" and con["kind"] == "multi" and len(set(y[4:8].tolist())) > 1
                            and np.all(np.diff(c[4:8]) > 0) and np.all(np.diff(c[:4]) >= 0) and np.array_equal(y, np.asarray([seqs[int(a)]["labels"][int(b)] for a, b in zip(sq, c)])))
            ex = [seqs[int(sq[k])] for k in range(8, 16)]; ss = [session_of(q["id"]) for q in ex]
            extra_ok &= bool(all(q["kind"] == "native" for q in ex) and list(y[8:12]) == [A4C] * 4 and list(y[12:16]) == [A5C] * 4 and ss[:4] == [F_.bank["sessions"][k] for k in ch]
                             and all(int(sq[8 + i]) in F_.bank["a4c"][k] and int(sq[12 + i]) in F_.bank["a5c"][k] for i, k in enumerate(ch)))
            distinct4 &= len(set(ch.tolist())) == 4 and len(set(ss[:4])) == 4
            s_same &= all(ss[4 + i] == ss[sp.PARTNER["S"][i]] for i in range(4)); c_cross &= all(ss[4 + i] != ss[sp.PARTNER["C"][i]] for i in range(4))
            eS = sorted((int(rows[12 + i]), int(rows[8 + sp.PARTNER["S"][i]])) for i in range(4)); eC = sorted((int(rows[12 + i]), int(rows[8 + sp.PARTNER["C"][i]])) for i in range(4))
            multiset_eq &= sorted(r for e in eS for r in e) == sorted(r for e in eC for r in e) and eS != eC
            dup += len(set(rows[8:].tolist()) & set(rows[:8].tolist())); ex_w.update(rows[8:].tolist()); ex_rec.update(int(x) for x in sq[8:]); ex_sess.update(ss[:4]); sess_exp.update(ss[:4])
        e = np.asarray(list(sess_exp.values()))
        out[str(f)] = {"bank": cnt, "bank_equals_expected": cnt == EXPECTED_BANK[f], "bank_sessions_in_other_fold": len(set(F_.bank["sessions"]) & other), "pool_sessions_in_other_fold": len(pool_sess & other),
                       "grid_equals_A07_windows": grid_ok, "original8_valid_900": orig_ok, "extra8_valid_900": extra_ok, "four_distinct_sessions_900": distinct4, "S_edges_same_session": s_same,
                       "C_edges_cross_session": c_cross, "endpoint_window_label_multisets_identical": multiset_eq, "label_counts_900x16": {str(k): int(v) for k, v in sorted(lab.items())},
                       "exposure": {"extra_slots": 8 * U, "distinct_extra_windows": len(ex_w), "distinct_extra_recordings": len(ex_rec), "distinct_bank_sessions_used": len(ex_sess),
                                    "session_draws_min_median_max": [int(e.min()), float(np.median(e)), int(e.max())], "extra_windows_also_in_original8_same_update": dup},
                       "selection_inputs": "kinds, recording ids, clip codes (no Base output, S1 / harm set or held-out result)", "draw_repeat_equal": plan_sha(F_, s, range(U)) == plan_sha(F_, s, range(U)), "plan_sha256": plan_sha(F_, s, range(U))}
    out["isolation"] = spl["isolation"]; return out


def init_checks(ctx: Ctx, s: int, F0: Fold, device) -> dict:
    """(2) pixels by the A31 path, Init state = pinned A08, online fit-window logits = A31 cache, patched = original forward."""
    b = Batches(F0, s, [0])[0]; u, rows, u8, mk_slots, y = b; rows = rows.numpy(); res = {}
    fr, idx = t37.chunk_frames(F0.seqs, {"seq": F0.W["seq"][rows]}, F0.W["s"][rows], F0.fp, 0); res["pixels_equal_A31_path"] = bool(np.array_equal(fr[idx], u8.numpy()))
    res["identical_arm_batches"] = bool(torch.equal(Batches(F0, s, [0])[0][2], u8))
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); mp, mo = build(ctx, s, device), build(ctx, s, device, patched=False); mc = build(ctx, s, device)
    res["init_state_equals_A08"] = all(torch.equal(v.cpu(), a08[k]) for k, v in mp.state_dict().items()) and sorted(mp.state_dict()) == sorted(a08)
    res["identical_arm_start"] = tet.state_digest(mp) == tet.state_digest(mc); res["patched_blocks"] = [i for i, bl in enumerate(mp.encoder.blocks) if isinstance(getattr(bl.pool_skip, "pool", None), sp.GatherMaxPool3d)]
    O = t31.outputs(ctx.a31, s, 0); rgb, mk = tm.to_rgb(u8.to(device)), tm.marker_planes(mk_slots.to(device))
    with torch.no_grad():
        zp = logits16(mp, rgb, mk); zo = logits16(mo, rgb, mk)
    res["init_online_fit_logits_vs_A31_cache_max_abs"] = float(np.abs(zp.cpu().numpy() - O["logits"][rows]).max())
    res["patched_equals_original_forward_bitwise"] = bool(torch.equal(zp, zo)); del mp, mo, mc; torch.cuda.empty_cache()
    res["Base_from_cached_q"] = t40.init_base(ctx, s); return res


def formula_checks(ctx: Ctx, s: int, F0: Fold, device) -> dict:
    """(3) float64 reference of CE / rank / total and their logit gradients for both partner maps; large margins; properties."""
    g = torch.Generator().manual_seed(42); out = {}; y = torch.tensor([0, 1, 4, 2, 3, 3, 2, 1] + [A4C] * 4 + [A5C] * 4)
    for name, z0 in (("random", torch.randn(PER, 5, generator=g, dtype=torch.float64) * 3), ("large_margins", torch.randn(PER, 5, generator=g, dtype=torch.float64) * 1e3)):
        for arm, par in sp.PARTNER.items():
            z = z0.clone().requires_grad_(True); tot, ce, rk = sp.total_loss(z, y, par); tot.backward(); r = sp.reference(z0.numpy(), y.numpy(), par)
            out[f"{name}_{arm}"] = {"ce_abs": abs(float(ce) - r["ce"]), "rank_abs": abs(float(rk) - r["rank"]), "total_abs": abs(float(tot) - r["total"]), "grad_max_abs": float(np.abs(z.grad.numpy() - r["g_total"]).max()),
                                    "finite": bool(torch.isfinite(tot) and torch.isfinite(z.grad).all())}
    z = torch.randn(PER, 5, generator=g, dtype=torch.float64); sw = z[:, [0, 1, 3, 2, 4]]
    out["margin_antisymmetric_under_A4C_A5C_swap"] = bool(torch.allclose(sp.margins(sw), -sp.margins(z))); out["rank_shift_invariant"] = abs(float(sp.rank_loss(z + 7.0, sp.PARTNER["C"]) - sp.rank_loss(z, sp.PARTNER["C"]))) < 1e-12
    zs = z.clone(); zs[12:16, A5C] += 100.0; out["rank_to_zero_when_A5C_ahead"] = float(sp.rank_loss(zs.clone(), sp.PARTNER["S"]))
    out["C_is_permutation_without_fixed_point"] = sorted(sp.PARTNER["C"]) == [0, 1, 2, 3] and all(sp.PARTNER["C"][i] != i for i in range(4))
    zz = torch.zeros(PER, 5, dtype=torch.float64); out["CE_denominator_16_coef_0.5_at_zero_logits"] = {"ce": float(sp.total_loss(zz, y, sp.PARTNER["S"])[1]), "expected_ce": float(np.log(5)), "rank": float(sp.rank_loss(zz, sp.PARTNER["S"])), "expected_rank": float(np.log(2)),
                                                                                  "total": float(sp.total_loss(zz, y, sp.PARTNER["S"])[0]), "expected_total": float(np.log(5) + 0.5 * np.log(2))}
    # real fit logits (Init, microbatch 1) under the formal partner maps, and the rank-only gradient path through the model
    b = Batches(F0, s, [5])[0]; u, _, u8, mk, yy = t40.on_device(b, device); m = build(ctx, s, device); z = logits16(m, tm.to_rgb(u8), mk)
    for arm, par in sp.PARTNER.items():
        r = sp.reference(z.detach().double().cpu().numpy(), yy.cpu().numpy(), par); tot, ce, rk = sp.total_loss(z, yy, par)
        out[f"real_fit_logits_{arm}"] = {"ce": float(ce), "rank": float(rk), "ce_ref_abs": abs(float(ce) - r["ce"]), "rank_ref_abs": abs(float(rk) - r["rank"])}
    sp.rank_loss(z, sp.PARTNER["C"]).backward(); gr = {n: p.grad for n, p in m.named_parameters()}
    nz = lambda pre: any(v is not None and float(v.abs().sum()) > 0 for k, v in gr.items() if k.startswith(pre))  # noqa: E731
    cw = gr["classifier.weight"].abs().sum(1).cpu().numpy()
    out["rank_only_gradient_path"] = {"marker_stem": nz("marker_stem."), "conv_proj": nz("encoder.conv_proj."), "block_0": nz("encoder.blocks.0."), "block_15": nz("encoder.blocks.15."), "encoder_head_512": nz("encoder.head."),
                                      "classifier_rows_A4C_A5C_nonzero": bool(cw[A4C] > 0 and cw[A5C] > 0), "classifier_other_rows_zero": bool(cw[[0, 1, 4]].sum() == 0),
                                      "finite": all(v is None or bool(torch.isfinite(v).all()) for v in gr.values())}
    del m; torch.cuda.empty_cache(); return out


def grads(m) -> dict:
    return {n: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in m.named_parameters()}


def rel(ga: dict, gb: dict) -> dict:
    mx = max(float(v.abs().max()) for v in gb.values()); d = max(float((ga[k] - gb[k]).abs().max()) for k in gb); return {"max_abs": d, "grad_max": mx, "relative": d / mx}


def gradient_checks(ctx: Ctx, s: int, F_: Fold, device) -> dict:
    """(4/5) microbatch 1 vs two batches of 8; patched vs original MaxPool3d backward; zero-gradient tensors (CE + rank)."""
    b = Batches(F_, s, [7])[0]; u, _, u8, mk, y = t40.on_device(b, device); rgb = tm.to_rgb(u8); par = sp.PARTNER["C"]; out = {}
    m = build(ctx, s, device); sp.total_loss(logits16(m, rgb, mk), y, par)[0].backward(); g_mb = grads(m); m.zero_grad(set_to_none=True)
    out["zero_gradient_tensors_CE_plus_rank"] = sorted(k for k, v in g_mb.items() if float(v.abs().sum()) == 0)
    sp.total_loss(torch.cat([m(rgb[:8], mk[:8]), m(rgb[8:], mk[8:])]), y, par)[0].backward(); out["microbatch1_vs_batch_8_plus_8"] = rel(g_mb, grads(m)); del m; torch.cuda.empty_cache()
    mo = build(ctx, s, device, patched=False); torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        sp.total_loss(logits16(mo, rgb, mk), y, par)[0].backward(); out["patched_vs_original_maxpool_backward"] = rel(g_mb, grads(mo))
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
    del mo; torch.cuda.empty_cache(); out["warn_only_restored_false"] = not torch.is_deterministic_algorithms_warn_only_enabled(); return out


def mech_run(ctx: Ctx, s: int, F_: Fold, partner, us, device, workers: int, state: dict | None = None, split_every: int = 0):
    m = build(ctx, s, device); opt = make_opt(m); torch.manual_seed(model_seed(s, F_.f))
    if state is not None:
        load_state(m, opt, state)
    log = []; g = run_updates(m, opt, F_, s, us, partner, device, workers, log, split_every); return m, opt, log, g


def training_checks(ctx: Ctx, s: int, F_: Fold, device, workers: int) -> dict:
    us = range(85, 91); res = {}; m = build(ctx, s, device); opt = make_opt(m); init = flat_state(m); init = {k: v.clone() for k, v in init.items()}
    ids = sorted(id(p) for g in opt.param_groups for p in g["params"]); grp = {g["name"]: [id(p) for p in g["params"]] for g in opt.param_groups}; names = {id(p): n for n, p in m.named_parameters()}
    res["binding"] = {"all_trainable": all(p.requires_grad for p in m.parameters()), "binds_exactly": ids == sorted(id(p) for p in m.parameters()) and len(ids) == len(set(ids)),
                      "encoder_group_is_encoder_incl_projection": set(grp["encoder"]) == {id(p) for p in m.encoder.parameters()} and any(names[i].startswith("encoder.head.") for i in grp["encoder"]),
                      "new_group": [names[i] for i in grp["new"]], "eval_mode": not m.training and not any(x.training for x in m.modules()),
                      "groups": {g["name"]: {"tensors": len(g["params"]), "parameters": int(sum(p.numel() for p in g["params"])), "lr": g["base"], "weight_decay": g["weight_decay"]} for g in opt.param_groups},
                      "parameters": int(sum(p.numel() for p in m.parameters())), "betas_eps": [list(opt.defaults["betas"]), opt.defaults["eps"]]}
    del m, opt; torch.cuda.empty_cache()
    L = lambda a, b: max(max(abs(x["ce"] - y["ce"]), abs(x["rank"] - y["rank"])) for x, y in zip(a[2], b[2]))  # noqa: E731
    P = lambda a, b: params_max_diff(flat_state(a[0]), flat_state(b[0]))  # noqa: E731
    sa = mech_run(ctx, s, F_, sp.PARTNER["S"], us, device, workers); fin = flat_state(sa[0])
    grp_moved = {nm: {"tensors_moved": int(sum(1 for k in fin if k.startswith(pre) and not torch.equal(fin[k], init[k]))), "tensors": int(sum(1 for k in fin if k.startswith(pre))),
                      "max_abs": float(max((fin[k] - init[k]).abs().max() for k in fin if k.startswith(pre)))} for nm, pre in (("encoder_blocks", "encoder.blocks."), ("conv_proj", "encoder.conv_proj."), ("encoder_pos_norm", ("encoder.pos_encoding.", "encoder.norm.")), ("encoder_head_512", "encoder.head."), ("marker_stem", "marker_stem."), ("classifier", "classifier."))}
    res["real_update_S_6"] = {"moved": grp_moved, "unchanged_tensors": sorted(k for k in fin if torch.equal(fin[k], init[k])), "optimizer_steps": sorted({int(v["step"]) for v in sa[1].state_dict()["state"].values()}), "ce": [x["ce"] for x in sa[2]], "rank": [x["rank"] for x in sa[2]]}
    sb = mech_run(ctx, s, F_, sp.PARTNER["S"], us, device, workers); res["repeat_6_S"] = {"loss_max_abs": L(sa, sb), "params_max_abs": P(sa, sb)}; del sb; torch.cuda.empty_cache()
    ci = mech_run(ctx, s, F_, (0, 1, 2, 3), us, device, workers); res["forced_C_partner_identity_vs_S"] = {"loss_max_abs": L(ci, sa), "params_max_abs": P(ci, sa), "formal_C_partner_unchanged": sp.PARTNER["C"] == (1, 2, 3, 0)}; del ci
    ca = mech_run(ctx, s, F_, sp.PARTNER["C"], us, device, workers); res["C_differs_from_S"] = {"params_max_abs": P(ca, sa), "rank_first": [ca[2][0]["rank"], sa[2][0]["rank"]]}; del sa; torch.cuda.empty_cache()
    c3 = mech_run(ctx, s, F_, sp.PARTNER["C"], range(85, 88), device, workers); p = ctx.out / "check_resume.pt"; torch.save(state_of(c3[0], c3[1]), p); log3 = c3[2]; del c3
    st = torch.load(p, map_location="cpu", weights_only=False); cr = mech_run(ctx, s, F_, sp.PARTNER["C"], range(88, 91), device, workers, st); p.unlink()
    res["resume_3_plus_3_C"] = {"loss_max_abs": max(max(abs(x["ce"] - y["ce"]), abs(x["rank"] - y["rank"])) for x, y in zip(ca[2], log3 + cr[2])), "params_max_abs": P(ca, cr)}
    res["timing"] = {"seconds_per_update_gpu_C_6": ca[3] / len(us)}; del ca, cr; torch.cuda.empty_cache()
    s1 = mech_run(ctx, s, F_, sp.PARTNER["S"], [100], device, workers, split_every=50); s0 = mech_run(ctx, s, F_, sp.PARTNER["S"], [100], device, workers)
    res["split_monitor_leaves_update_unchanged"] = {"params_max_abs": P(s1, s0), "split_recorded": "trunk_grad_ce" in s1[2][0], **{k: s1[2][0][k] for k in ("trunk_grad_ce", "trunk_grad_scaled_rank", "all_grad_ce", "all_grad_scaled_rank")}}; del s1, s0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); m, opt, log, gsec = mech_run(ctx, s, F_, sp.PARTNER["C"], range(200, 230), device, workers); wall = time.time() - t0
    res["timing"].update({"seconds_per_update_gpu_C_30": gsec / 30, "seconds_per_update_wall_C_30_incl_build_and_loader": wall / 30, "peak_gpu_gb_training": torch.cuda.max_memory_allocated() / 1e9}); del m, opt
    torch.cuda.empty_cache(); return res


def readout_label_free(ctx: Ctx, s: int) -> dict:
    """(5) Pooling / relabel / RLE inputs: with every reference label replaced by a constant, the cached-q readout is unchanged."""
    X = t40.full_q(ctx, s); D = d32.fold_data(ctx.d, s, 0); pk = D["pk"]; z = X["q"]; off = X["offset"][0]; W = t31.pool_W(D["seqs"])
    pk2 = {**pk, "labels": np.zeros_like(pk["labels"])}; seqs2 = [{**q, "labels": np.zeros_like(np.asarray(q["labels"]))} for q in D["seqs"]]; rows = np.arange(len(pk["labels"]))
    a, b = t29.pooled_probs(z, pk, rows), t29.pooled_probs(z, pk2, rows); zf = z[off:off + len(W["seq"])].numpy()
    pa, pb = rr.per_sample(W, zf, D["seqs"]), rr.per_sample(t31.pool_W(seqs2), zf, seqs2)
    same = all(np.array_equal(ir.relabel(D["base_preds"][q["id"]], pa[q["id"]]), ir.relabel(D["base_preds"][q2["id"]], pb[q2["id"]])) for q, q2 in zip(D["seqs"], seqs2))
    return {"run_probs_equal": bool(np.array_equal(a, b)), "sequence_relabel_equal": bool(same), "window_grid_equal": bool(np.array_equal(t31.pool_W(seqs2)["centre"], W["centre"])), "fold": 0}


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; t_start = time.time(); res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL, "limits": LIMITS, "contract": sp.CONTRACT, "expected_bank": EXPECTED_BANK,
                                         "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                                      "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.are_deterministic_algorithms_enabled(), "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
    spl = json.loads((ctx.a31.out / "split.json").read_text()); res["pool_sha256"] = spl["pool_sha256"]; res["plan"] = plan_checks(ctx, s)
    if not all(res["plan"][str(f)]["bank_equals_expected"] for f in FOLDS):
        publish(ctx, "checks.json", {**res, "ok": False, "stop": "pair bank differs from the asserted counts", "finished": now()}); raise SystemExit("pair bank differs from the asserted counts: engineering stop before pixels")
    F0 = Fold(ctx, 0); res["init"] = init_checks(ctx, s, F0, device); res["formula"] = formula_checks(ctx, s, F0, device); res["gradients"] = gradient_checks(ctx, s, F0, device)
    res["training"] = training_checks(ctx, s, F0, device, workers); res["readout_label_free"] = readout_label_free(ctx, s)
    res["a08_files_unchanged"] = all(sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{k}/final.pt") == t31.A08[k] for k in (0, 2))
    spu = max(res["training"]["timing"]["seconds_per_update_gpu_C_30"], res["training"]["timing"]["seconds_per_update_wall_C_30_incl_build_and_loader"]); tm_ = t40.readout_timing(ctx, s, device, workers)
    nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f))["seq"]) for f in FOLDS); spent = t31.vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t_start) / 3600
    train_h = 4 * U * spu * 1.15 / 3600; read_h = nw * tm_["seconds_per_window_two_models"] * 1.15 / 3600; proj = spent + train_h + read_h + 0.5
    res["projection"] = {"spent_gpu_h_incl_this_check": spent, "train_4x900_h": train_h, "readout_h": read_h, "reserve_h": 0.5, "source0_total_h": proj, "source2_if_triggered_h": train_h + read_h,
                         "cap_h": ctx.v["budget"]["gpu_hours_max"], "windows": nw, "seconds_per_update_used": spu, **tm_}
    free = shutil.disk_usage(ctx.out).free / 1e9; res["disk"] = {"free_gb": free, "per_model_final_gb": 4 * res["training"]["binding"]["parameters"] / 1e9, "resume_gb": 12 * res["training"]["binding"]["parameters"] / 1e9,
                                                                 "min_free_gb": ctx.v["budget"]["min_free_disk_gb"]}
    res["loaded_code_sha256"] = t38.loaded_code(); res["failed"] = check_failures(res); res["ok"] = not res["failed"]; res["finished"] = now(); publish(ctx, "checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loaded_code_sha256", "plan")}, indent=1, default=str)[:12000], flush=True)
    if not res["ok"]:
        raise SystemExit(f"checks failed: {res['failed']}")
    return f"{spu:.2f} s per update; source-0 projection {proj:.2f} GPU h"


def check_failures(r: dict) -> list[str]:
    I, Fm, G, T, Pr, N = r["init"], r["formula"], r["gradients"], r["training"], r["projection"], r["numerics"]; lim = lambda d: d["loss_max_abs"] <= LIMITS["loss"] and d["params_max_abs"] <= LIMITS["params"]  # noqa: E731
    conds = {"isolation": not any(r["plan"]["isolation"].values()),
             "plan": all(all(r["plan"][str(f)][k] for k in ("bank_equals_expected", "grid_equals_A07_windows", "original8_valid_900", "extra8_valid_900", "four_distinct_sessions_900", "S_edges_same_session",
                                                             "C_edges_cross_session", "endpoint_window_label_multisets_identical", "draw_repeat_equal")) and r["plan"][str(f)]["bank_sessions_in_other_fold"] == 0
                         and r["plan"][str(f)]["pool_sessions_in_other_fold"] == 0 for f in FOLDS),
             "numerics": not N["tf32_matmul"] and not N["tf32_cudnn"] and N["cudnn_deterministic"] and not N["cudnn_benchmark"] and N["deterministic"] and not N["deterministic_warn_only"],
             "pixels": I["pixels_equal_A31_path"] and I["identical_arm_batches"], "init_state": I["init_state_equals_A08"] and I["identical_arm_start"] and I["patched_blocks"] == [1, 3, 14],
             "init_online_parity": I["init_online_fit_logits_vs_A31_cache_max_abs"] <= TOL["parity"], "patched_forward": I["patched_equals_original_forward_bitwise"],
             "base_from_cache": all(all(v[k] for k in ("runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in I["Base_from_cached_q"].values()),
             "formula": all(v["ce_abs"] <= TOL["formula"] and v["rank_abs"] <= TOL["formula"] and v["total_abs"] <= TOL["formula"] and v["grad_max_abs"] <= TOL["formula"] and v["finite"] for k, v in Fm.items() if k[:6] in ("random", "large_"))
             and Fm["margin_antisymmetric_under_A4C_A5C_swap"] and Fm["rank_shift_invariant"] and Fm["rank_to_zero_when_A5C_ahead"] < 1e-30 and Fm["C_is_permutation_without_fixed_point"]
             and abs(Fm["CE_denominator_16_coef_0.5_at_zero_logits"]["total"] - Fm["CE_denominator_16_coef_0.5_at_zero_logits"]["expected_total"]) < 1e-12
             and all(Fm[f"real_fit_logits_{a}"]["ce_ref_abs"] <= 1e-5 and Fm[f"real_fit_logits_{a}"]["rank_ref_abs"] <= 1e-5 for a in ARMS),
             "rank_gradient_path": all(Fm["rank_only_gradient_path"].values()),
             "microbatch_parity": G["microbatch1_vs_batch_8_plus_8"]["relative"] <= TOL["grad_rel"], "skip_pool_parity": G["patched_vs_original_maxpool_backward"]["relative"] <= TOL["grad_rel"] and G["warn_only_restored_false"],
             "binding": T["binding"]["all_trainable"] and T["binding"]["binds_exactly"] and T["binding"]["encoder_group_is_encoder_incl_projection"] and T["binding"]["eval_mode"]
             and T["binding"]["new_group"] == ["marker_stem.weight", "classifier.weight", "classifier.bias"],
             "real_update": T["real_update_S_6"]["optimizer_steps"] == [6] and all(v["tensors_moved"] > 0 for v in T["real_update_S_6"]["moved"].values()),
             "repeat": lim(T["repeat_6_S"]), "forced_identity": lim(T["forced_C_partner_identity_vs_S"]) and T["forced_C_partner_identity_vs_S"]["formal_C_partner_unchanged"], "resume": lim(T["resume_3_plus_3_C"]),
             "C_not_S": T["C_differs_from_S"]["params_max_abs"] > 0, "split_monitor": T["split_monitor_leaves_update_unchanged"]["params_max_abs"] <= LIMITS["params"] and T["split_monitor_leaves_update_unchanged"]["split_recorded"],
             "readout_label_free": all(r["readout_label_free"][k] for k in ("run_probs_equal", "sequence_relabel_equal", "window_grid_equal")), "a08_files": r["a08_files_unchanged"],
             "memory": T["timing"]["peak_gpu_gb_training"] <= 40.0, "projection": Pr["source0_total_h"] <= Pr["cap_h"], "disk": r["disk"]["free_gb"] >= r["disk"]["min_free_gb"]}
    return sorted(k for k, v in conds.items() if not v)


# ----------------------------------------------------------------------------------------------- readout
def load_model(ctx: Ctx, arm: str, s: int, f: int, device):
    """The committed final weights in the original A08 module (original nn.MaxPool3d; no backward at readout)."""
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    m = ctx.a31.a08(s, device); m.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True), strict=True); return m.eval()


def readout(ctx: Ctx, s: int, device, workers: int, dry: bool = False) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists() and not dry:
        raise SystemExit("this seed's readout exists; it runs once")
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m_: [] for m_ in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; tim = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    a40 = json.loads((ROOT / "docs/results/2026-10-27_masked_feature_recognition/readout_s0.json").read_text())["cells"] if s == 0 else None; cache = {}
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        if dry:
            Z = {a: X["q"][off:off + len(t31.pool_W(D["seqs"])["seq"])].numpy() for a in ARMS}
        else:
            Z = t40.heldout_logits(ctx, s, h, {a: load_model(ctx, a, s, f, device) for a in ARMS}, device, workers, tim); torch.cuda.empty_cache()
        cell = {}
        for arm in ARMS:
            z = X["q"].clone(); z[off:off + len(Z[arm])] = torch.from_numpy(Z[arm]); t = tables(z, D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
            cell[arm]["coverage"], sets = t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets); cell[arm]["boundaries"] = t38.boundaries(z, D, off)
            ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs); cache[f"logits_{arm}_heldout_f{h}"] = Z[arm]; cache[f"run_pred_{arm}_heldout_f{h}"] = p
        t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0])
        cache[f"run_pred_Base_heldout_f{h}"] = p; cache[f"run_label_heldout_f{h}"] = D["pk"]["labels"]
        s1 = np.flatnonzero(D["M"]["S1"]); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a40 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A40_Base_cell"] = all(js(cell["Base"][k]) == a40[f"heldout_f{h}"]["Base"][k] for k in a40[f"heldout_f{h}"]["Base"] if k in cell["Base"])
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("S", "C"), ("S", "Base"), ("C", "Base"))}
        if not dry:
            mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}
            cell["fit_only"] = {arm: {k: mans[arm][k] for k in ("partner", "coefficient", "fit_before", "fit_after", "drift_from_A08", "loss")} for arm in ARMS}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_])), "broken_correct_A5C": int(((b == y) & (y == A5C) & (p != y)).sum()), "broken_correct_A5C_into_A4C": int(((b == y) & (y == A5C) & (p == A4C)).sum()),
                          "broken_correct_A4C": int(((b == y) & (y == A4C) & (p != y)).sum()), "fixed_true_A5C_read_A4C": int(((y == A5C) & (b == A4C) & (p == A5C)).sum()), "fixed_true_A4C_read_A5C": int(((y == A4C) & (b == A5C) & (p == A4C)).sum())}
    rec["seed2_rule"], rec["values"], rec["summary"] = rule_of(rec, s1recs)
    if dry:
        print(json.dumps({"dry_run_init": True, "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}, "summary": rec["summary"], "oof": rec["oof"]}, indent=1)); return "dry run (Init); nothing published"
    cp = ctx.out / f"readout_cache_s{s}.npz"; np.savez(cp, **cache)
    nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f_))["seq"]) for f_ in FOLDS); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())
    rec["private_cache"] = {"path": f"{ctx.v['outputs']['root']}/readout_cache_s{s}.npz (server only)", "sha256": sha256_file(cp), "arrays": sorted(cache)}
    rec["deployment_cost"] = {"forwards": "Base's one 224 forward per window plus one forward of the new recognition copy (a second full A08-size encoder)", "parameters": "A08 34.6M (Base) + 34.6M (new copy)",
                              "gpu_ms_per_window_new_copy": 1e3 * tim["gpu_seconds"] / (2 * nw), "cpu_decode_ms_per_window_shared": 1e3 * tim["decode_seconds"] / nw, "gpu_ms_per_window_A31_base_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()),
                              "readout_gpu_seconds": tim["gpu_seconds"], "readout_decode_seconds": tim["decode_seconds"], "note": "two models (C, S) per held-out window share one decode; the A31 pass also wrote its prefix cache"}
    rec["written"] = now(); publish(ctx, name, rec)
    print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rec["seed2_rule"], "values": rec["values"], "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}}, indent=1)); return f"rule met: {rec['seed2_rule']['met']}"


def rule_of(rec: dict, s1recs: dict) -> tuple[dict, dict, dict]:
    C = lambda h: rec["cells"][f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C(h)[a]["sequence_row"][k] - C(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    n1, n5, n4 = tot("S", "S1_fixed", "runs"), tot("S", "correct_A5C_kept", "runs"), tot("S", "correct_A4C_kept", "runs"); s1S, s1C = tot("S", "S1_fixed", "fixed"), tot("C", "S1_fixed", "fixed")
    k5S, k5C, k4S, k4C = tot("S", "correct_A5C_kept", "kept"), tot("C", "correct_A5C_kept", "kept"), tot("S", "correct_A4C_kept", "kept"), tot("C", "correct_A4C_kept", "kept")
    recS = [len(r) for r in s1recs["S"]]; recS_all = len(set().union(*s1recs["S"]))
    rule = {"target_each_fold_S_minus_C_S1_ge_1": all(C(h)["S"]["S1_fixed"]["fixed"] - C(h)["C"]["S1_fixed"]["fixed"] >= 1 for h in FOLDS), "target_sum_S_minus_C_S1_ge_5": s1S - s1C >= 5,
            "S1_S": f"{s1S}/{n1}", "S1_C": f"{s1C}/{n1}", "S1_needed": ceil_pct(10, n1), "signal_S1_runs_ok": s1S >= ceil_pct(10, n1),
            "S1_S_true_class_recordings_by_fold": recS, "S1_S_true_class_recordings_total": recS_all, "signal_recordings_ok": recS_all >= 3 and all(r >= 1 for r in recS),
            "correct_A5C_S": f"{k5S}/{n5}", "correct_A5C_C": f"{k5C}/{n5}", "A5C_needed": ceil_pct(98, n5), "retain_A5C_ok": k5S >= ceil_pct(98, n5) and k5S >= k5C - 3,
            "correct_A4C_S": f"{k4S}/{n4}", "correct_A4C_C": f"{k4C}/{n4}", "A4C_needed": ceil_pct(98, n4), "retain_A4C_ok": k4S >= ceil_pct(98, n4) and k4S >= k4C - 3,
            "net_S_minus_Base_each_fold_ge_0": all(net("S", h) >= 0 for h in FOLDS), "net_S_minus_C_each_fold_ge_0": all(net("S", h) - net("C", h) >= 0 for h in FOLDS),
            "dF_S_minus_Base_each_fold_ge_0": all(d_("S", "Base", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "dF_S_minus_C_each_fold_ge_0": all(d_("S", "C", h, "segment_f1_50_multi") >= 0 for h in FOLDS),
            "native_S_minus_Base_each_fold_ge_-0.003": all(d_("S", "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_S_minus_Base_each_fold_ge_-0.005": all(d_("S", "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
            "missed_short_S_le_Base_plus_3": tot("S", "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3}
    rule["met"] = all(v for v in rule.values() if isinstance(v, bool))
    values = {"S_minus_C_S1": [C(h)["S"]["S1_fixed"]["fixed"] - C(h)["C"]["S1_fixed"]["fixed"] for h in FOLDS], "S_minus_C_net": [net("S", h) - net("C", h) for h in FOLDS], "S_minus_Base_net": [net("S", h) for h in FOLDS],
              "C_minus_Base_net": [net("C", h) for h in FOLDS], "S_minus_C_dF": [d_("S", "C", h, "segment_f1_50_multi") for h in FOLDS], "S_minus_Base_dF": [d_("S", "Base", h, "segment_f1_50_multi") for h in FOLDS],
              "C_minus_Base_dF": [d_("C", "Base", h, "segment_f1_50_multi") for h in FOLDS], "S_minus_Base_native": [d_("S", "Base", h, "native_macro_f1") for h in FOLDS],
              "S_minus_Base_boundary_recall": [d_("S", "Base", h, "boundary_recall_multi") for h in FOLDS], "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}}
    summary = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C(h)[m_]["correct_A5C_kept"]["kept"],
                                       "A4C_kept": C(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    return rule, values, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("check", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "freeze":
        return staged(ctx, f"freeze s{a.seed} (CPU)", "cpu", lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha))
    if a.stage == "readout" and a.dry_run:
        return staged(ctx, f"readout dry run s{a.seed} (Init, CPU)", "cpu", lambda: readout(ctx, a.seed, None, a.workers, True))
    t31.vsh.determinism(); torch.use_deterministic_algorithms(True, warn_only=False); device = torch.device("cuda")
    if a.stage == "train":
        return train(ctx, a.seed, device, a.workers)
    fn = {"check": lambda: check(ctx, device, a.workers), "readout": lambda: readout(ctx, a.seed, device, a.workers)}[a.stage]
    staged(ctx, f"{a.stage}{'' if a.stage == 'check' else ' s' + str(a.seed)} (GPU job wall time)", "gpu", fn)


if __name__ == "__main__":
    main()
