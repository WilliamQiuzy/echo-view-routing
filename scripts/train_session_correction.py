#!/usr/bin/env python
"""A31 (configs/experiments/video_session_correction_v1.yaml): session-held-out interval recognition correction.

  --stage split    session folds of A15's large-fit sources, one pool per fold by A15's builder rules, isolation (CPU)
  --stage infer    per seed: A08 over both pools' request windows -> logits, 512-d embeddings, frozen-prefix cache (GPU)
  --stage base     per seed: Base + B3, T / P runs, multiplicities, batch plans T / U / E per fit fold (CPU)
  --stage check    identities, A08 parity, zero update = Base, objective, microbatch, forced equality, resume, timing (GPU)
  --stage train    per seed: F-U, F-E (CPU) and T-U, T-E (GPU) on each fit fold
  --stage readout  per seed, after the weights are committed: every model on its held-out fold (runs and full sequences)
  --stage report   OOF sums, comparisons and the session bootstrap from the readouts (CPU)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # this process only; A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.b3_coverage_evaluate as bce  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_clips as ic  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.compose.recipe_builder import Pool, make_multi_recipes  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A15_CONFIG = "configs/experiments/video_router_data_v1.yaml"
A27_CONFIG = "configs/experiments/video_interval_recognition_v1.yaml"
A08 = {0: "9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc", 2: "f95fed7ba78cc0b5e2474c71a156356507b4588a4b324957fd5b53fe66ee8472"}
SEEDS, FOLDS, ARMS = (0, 2), (0, 1), ("F_U", "F_E", "T_U", "T_E")
PER, LR_HEAD, LR_REP, WD, CLIP = 16, 1e-3, 1e-4, 1e-4, 1.0
TOKENS, DIM = 393, 768
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.v = cfg, config, cfg["video_session_correction"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a15 = trd.Ctx(load_cfg(SimpleNamespace(config=A15_CONFIG, set=[])), A15_CONFIG); self.a27 = t27.Ctx(load_cfg(SimpleNamespace(config=A27_CONFIG, set=[])), A27_CONFIG)
        self.t28 = SimpleNamespace(pin={"A08_weights": A08})

    def a08(self, s: int, device):
        return t28.Ctx.a08(self.t28, s, device)

    def schedule(self) -> dict:
        p = self.out / "schedule.json"
        return json.loads(p.read_text()) if p.exists() else {"updates": 1350, "warmup": 135}


# ----------------------------------------------------------------------------------------------- split and pools (CPU)
def fit_clips(ctx: Ctx):
    """A15's large-fit sources exactly: protocol-train clips whose session is outside A15's check sessions."""
    data = vpr.training_data(ctx.a15.a10.a07p); sess = sorted({session_of(v) for v in data.clips}); perm = np.random.default_rng([20261021, 15, 1]).permutation(len(sess))
    chk = {sess[i] for i in perm[: int(round(0.2 * len(sess)))]}; return data, [v for v in data.clips if session_of(v) not in chk]


def fold_sessions(clips: list[str], label: dict, k: int) -> dict:
    sess = sorted({session_of(v) for v in clips}); perm = np.random.default_rng([20261024, 31, k]).permutation(len(sess)); return {sess[i]: j % 2 for j, i in enumerate(perm)}


def build_pool(ctx: Ctx, data, idx, clips: list[str], fold: int) -> dict:
    a01 = ctx.a15.a10.a07p.a01().cfg; rc, mc = a01["recipes"], a01.get("multi", {}); sub = idx[idx["video_id"].isin(set(clips))]
    rec = make_multi_recipes(Pool.from_index(sub["video_id"], sub["label_index"], sub["n_samples"]), 600, np.random.default_rng([20261024, 31, fold, 2]), n_fragments=tuple(mc.get("n_fragments", (4, 8))),
                             edit_variant=rc["edit_variant"], edit_prob=float(mc.get("edit_prob", 0.5)), min_len=int(rc["min_len"]), max_len=int(rc["max_len"]), reuse=True, p_same_next=float(mc.get("p_same_next", 0.35)))
    seqs = []
    for k, r in enumerate(rec):
        labels, _ = vsf.stream_labels_joins(r.fragments)
        seqs.append({"id": f"a31_f{fold}_{k:04d}", "kind": "multi", "labels": labels.astype(np.int64), "refs": vsf.stream_refs(r.fragments, data.frame_idx), "frags": [trd.frag_key(f) for f in r.fragments]})
    rng = np.random.default_rng([20261024, 31, fold, 4]); nat, short = [], {}
    for c in range(5):
        cl = [v for v in clips if data.label[v] == c]; per, took = {}, 0
        for j in rng.permutation(len(cl)):
            v = cl[j]; s = session_of(v)
            if per.get(s, 0) >= 2:
                continue
            per[s] = per.get(s, 0) + 1; nat.append(v); took += 1
            if took == 200:
                break
        short[str(c)] = 200 - took
    for v in nat:
        f = data.frame_idx(v); seqs.append({"id": v, "kind": "native", "labels": np.full(len(f), data.label[v], np.int64), "refs": vsf.native_refs(v, f), "frags": [v]})
    return {"seqs": seqs, "native_shortfall_by_class": short, "multi_failures": 600 - len(rec)}


def window_ids(seqs: list[dict]) -> tuple[dict, set]:
    refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    ids = {hashlib.sha1(repr((tuple(tuple(refs[int(k)]) for k in W["ref"][j]), int(W["marker"][j]))).encode()).hexdigest() for j in range(len(W["seq"]))}
    return W, ids


def split(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config)}
    try:
        data, clips = fit_clips(ctx); idx = vsf.training_index(data.dir / "index.csv")
        for k in range(10):
            fs = fold_sessions(clips, data.label, k); per = {f: Counter(data.label[v] for v in clips if fs[session_of(v)] == f) for f in FOLDS}
            if all(per[f][c] >= 20 for f in FOLDS for c in range(5)):
                break
        else:
            raise SystemExit("no split met the coverage rule")
        rec["split"] = {"k": k, "sessions": {str(f): int(sum(1 for x in fs.values() if x == f)) for f in FOLDS}, "clips": {str(f): int(sum(1 for v in clips if fs[session_of(v)] == f)) for f in FOLDS},
                        "per_class_clips": {str(f): [int(per[f][c]) for c in range(5)] for f in FOLDS}}
        pools, wids, recs, sess = {}, {}, {}, {}
        for f in FOLDS:
            fc = [v for v in clips if fs[session_of(v)] == f]; pl = build_pool(ctx, data, idx, fc, f); pools[f] = pl; W, wids[f] = window_ids(pl["seqs"])
            recs[f] = {r[0] for q in pl["seqs"] for r in q["refs"]}; sess[f] = {session_of(v) for v in recs[f]}
            if not recs[f] <= set(fc):
                raise SystemExit(f"fold {f}: a sequence uses a recording outside its fold")
            rec[f"pool_{f}"] = {"multi": int(sum(q["kind"] == "multi" for q in pl["seqs"])), "native": int(sum(q["kind"] == "native" for q in pl["seqs"])), "native_shortfall_by_class": pl["native_shortfall_by_class"],
                                "multi_failures": pl["multi_failures"], "windows": int(len(W["seq"])), "samples": int(sum(len(q["labels"]) for q in pl["seqs"])), "recordings": len(recs[f]), "sessions": len(sess[f]),
                                "window_identities": len(wids[f])}
            np.save(ctx.out / f"pool_{f}.npy", np.array([{"id": q["id"], "kind": q["kind"], "labels": q["labels"], "refs": q["refs"], "frags": q["frags"]} for q in pl["seqs"]], dtype=object), allow_pickle=True)
        rec["isolation"] = {"sessions_shared": len(sess[0] & sess[1]), "recordings_shared": len(recs[0] & recs[1]), "window_identities_shared": len(wids[0] & wids[1])}
        if any(rec["isolation"].values()):
            raise SystemExit(f"folds are not isolated: {rec['isolation']}")
        rec["pool_sha256"] = {str(f): sha256_file(ctx.out / f"pool_{f}.npy") for f in FOLDS}; rec["finished"] = now()
        ids = {q["id"] for f in FOLDS for q in pools[f]["seqs"] if q["kind"] == "native"} | recs[0] | recs[1]; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "split")
        for p in (ctx.out / "split.json", ctx.results / "split.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps(flat, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "split and pools (CPU)", "cpu", started, time.time() - t0, ok, note)


def load_pool(ctx: Ctx, f: int) -> list[dict]:
    sp = json.loads((ctx.out / "split.json").read_text())
    if sha256_file(ctx.out / f"pool_{f}.npy") != sp["pool_sha256"][str(f)]:
        raise SystemExit(f"pool {f} differs from split.json")
    return [dict(x) for x in np.load(ctx.out / f"pool_{f}.npy", allow_pickle=True)]


def pool_W(seqs: list[dict]) -> dict:
    refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; return rr.half_windows(seqs, at, 5)


# ----------------------------------------------------------------------------------------------- A08 pass and prefix cache (GPU)
def infer(ctx: Ctx, s: int, device, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    try:
        m = ic.branch(ctx.a08(s, device), True); d0 = tet.state_digest(m); seqs = {f: load_pool(ctx, f) for f in FOLDS}; W = {f: pool_W(seqs[f]) for f in FOLDS}
        n = {f: len(W[f]["seq"]) for f in FOLDS}; off = {0: 0, 1: n[0]}; cdir = ctx.out / "cache"; cdir.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(cdir / f"prefix_s{s}.npy", mode="w+", dtype=np.float32, shape=(n[0] + n[1], TOKENS, DIM)); fp = t28.frame_paths(seqs[0] + seqs[1])
        for f in FOLDS:
            lg, em = np.zeros((n[f], 5), np.float32), np.zeros((n[f], 512), np.float32); base_row = 0
            for c0 in range(0, len(seqs[f]), 150):
                ch = seqs[f][c0:c0 + 150]; Wc = pool_W(ch); refs = sorted({tuple(r) for q in ch for r in q["refs"]}); a = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - a
                for b in range(0, len(Wc["seq"]), 32):
                    ks = np.arange(b, min(b + 32, len(Wc["seq"]))); torch.cuda.synchronize(); a = time.time()
                    with torch.no_grad():
                        x, thw = ic.prefix(m, tm.to_rgb(torch.from_numpy(frames[Wc["ref"][ks]]).to(device)), tm.marker_planes(torch.from_numpy(Wc["marker"][ks]).long().to(device))); e = ic.suffix(m, x, thw); z = m.classifier(e)
                    torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a; g = base_row + ks; mm[off[f] + g] = x.float().cpu().numpy(); lg[g] = z.float().cpu().numpy(); em[g] = e.float().cpu().numpy()
                if not (np.array_equal(Wc["seq"] + c0, W[f]["seq"][base_row:base_row + len(Wc["seq"])]) and np.array_equal(Wc["centre"], W[f]["centre"][base_row:base_row + len(Wc["seq"])])):
                    raise SystemExit("chunk windows differ from the pool's")
                base_row += len(Wc["seq"]); del frames
                print(f"seed {s} fold {f}: {base_row}/{n[f]} windows; GPU {acc['gpu_seconds']:.0f} s", flush=True)
            np.savez(ctx.out / f"outputs_s{s}_f{f}.npz", logits=lg, emb=em, seq=W[f]["seq"], centre=W[f]["centre"])
        mm.flush(); del mm
        if tet.state_digest(m) != d0:
            raise SystemExit("the branch changed during inference")
        rec = {"key": t29.prefix_code_digest(), "seed": s, "rows": n, "offsets": off, "a08": A08[s], "time": now(), **acc, "outputs_sha256": {str(f): sha256_file(ctx.out / f"outputs_s{s}_f{f}.npz") for f in FOLDS}}
        write_json(cdir / f"prefix_s{s}.json", rec); ok = True; note = f"{n[0] + n[1]} windows"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"A08 pass and prefix cache seed {s} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def outputs(ctx: Ctx, s: int, f: int) -> dict:
    rec = json.loads((ctx.out / "cache" / f"prefix_s{s}.json").read_text())
    if sha256_file(ctx.out / f"outputs_s{s}_f{f}.npz") != rec["outputs_sha256"][str(f)]:
        raise SystemExit("outputs differ from their record")
    with np.load(ctx.out / f"outputs_s{s}_f{f}.npz") as z:
        return {k: z[k] for k in z.files} | {"offset": int(rec["offsets"][str(f)])}


class Prefix:
    def __init__(self, ctx: Ctx, s: int) -> None:
        rec = json.loads((ctx.out / "cache" / f"prefix_s{s}.json").read_text())
        if rec["key"] != t29.prefix_code_digest():
            raise SystemExit("the prefix cache belongs to other code")
        self.mm = np.load(ctx.out / "cache" / f"prefix_s{s}.npy", mmap_mode="r")

    def get(self, rows: np.ndarray) -> torch.Tensor:
        order = np.argsort(rows); out = np.empty((len(rows), TOKENS, DIM), np.float32); out[order] = self.mm[np.asarray(rows)[order]]; return torch.from_numpy(out)


# ----------------------------------------------------------------------------------------------- Base, runs and batches (CPU)
def base(ctx: Ctx, s: int) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"seed": s, "folds": {}}
    try:
        params = ctx.a27.params(s); bdir = ctx.out / "data"; bdir.mkdir(parents=True, exist_ok=True); U = 1350     # plans always cover 1,350 updates; a uniform shortening uses their first rows
        for f in FOLDS:
            seqs = load_pool(ctx, f); W = pool_W(seqs); O = outputs(ctx, s, f); off = int(O["offset"]); q = rr.per_sample(W, O["logits"], seqs); B = decode(params, q)
            near = t27.nearest_rows(W, seqs); T, P = [], []
            for i, x in enumerate(seqs):
                lab = np.asarray(x["labels"])
                T += [{"rows": r + off, "weights": w, "label": c, "length": b - a, "purity": 1.0} for a, b, c in ir.runs(lab) for r, w in [ir.interval_rows(near[i], a, b)]]
                P += [{"rows": r + off, "weights": w, "label": ir.majority(lab[a:b])[0], "length": b - a, "purity": ir.majority(lab[a:b])[1], "base": c, "seq": i, "a": a, "b": b} for a, b, c in ir.runs(B[x["id"]]) for r, w in [ir.interval_rows(near[i], a, b)]]
            pt, pp = ir.pack(T), ir.pack(P); pp["base"] = np.asarray([x["base"] for x in P]); pp["seq"] = np.asarray([x["seq"] for x in P]); pp["a"] = np.asarray([x["a"] for x in P]); pp["b"] = np.asarray([x["b"] for x in P])
            wrong = np.flatnonzero(pp["base"] != pp["labels"]); right = np.flatnonzero(pp["base"] == pp["labels"])
            g0, g1, g2 = (np.random.default_rng([20261024, 31, s, f, j]) for j in (0, 1, 2))
            bt = g0.integers(0, len(pt["labels"]), (U, PER)); bu = g1.integers(0, len(pp["labels"]), (U, PER)); be = np.concatenate([wrong[g2.integers(0, len(wrong), (U, PER // 2))], right[g2.integers(0, len(right), (U, PER // 2))]], 1)
            np.savez(bdir / f"runs_s{s}_f{f}.npz", **{f"T_{k}": v for k, v in pt.items()}, **{f"P_{k}": v for k, v in pp.items()}, bt=bt, bu=bu, be=be, base_labels=np.concatenate([B[x["id"]] for x in seqs]))
            ue_err = float((pp["base"][bu] != pp["labels"][bu]).mean()); diff = float(np.mean([not np.array_equal(np.sort(bu[u]), np.sort(be[u])) for u in range(U)]))
            sess_of_run = [session_of(Counter(r[0] for r in seqs[i]["refs"][a:b]).most_common(1)[0][0]) for i, a, b in zip(pp["seq"], pp["a"], pp["b"])]
            rec["folds"][str(f)] = {"T_runs": len(pt["labels"]), "P_runs": len(pp["labels"]), "P_base_wrong": int(len(wrong)), "P_error_share": float(len(wrong) / len(pp["labels"])),
                                    "U_batch_error_share": ue_err, "E_batch_error_share": 0.5, "U_E_batches_differ_fraction": diff, "wrong_pairs": dict(Counter(f"{int(pp['labels'][k])}->{int(pp['base'][k])}" for k in wrong)),
                                    "wrong_sessions": len({sess_of_run[k] for k in wrong}), "E_wrong_unique_drawn": int(len(np.unique(be[:, :PER // 2]))), "runs_sha256": sha256_file(bdir / f"runs_s{s}_f{f}.npz")}
            np.save(bdir / f"run_sessions_s{s}_f{f}.npy", np.asarray(sess_of_run, dtype=object), allow_pickle=True)
        for p in (ctx.out / f"base_s{s}.json", ctx.results / f"base_s{s}.json"):
            write_json(p, rec)
        print(json.dumps(rec, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"Base, runs and batch plans seed {s} (CPU)", "cpu", started, time.time() - t0, ok, note)


def runs(ctx: Ctx, s: int, f: int) -> dict:
    rec = json.loads((ctx.out / f"base_s{s}.json").read_text())["folds"][str(f)]; p = ctx.out / "data" / f"runs_s{s}_f{f}.npz"
    if sha256_file(p) != rec["runs_sha256"]:
        raise SystemExit("run file differs from its record")
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    return {"T": {k[2:]: v for k, v in d.items() if k.startswith("T_")}, "P": {k[2:]: v for k, v in d.items() if k.startswith("P_")}, "bt": d["bt"], "bu": d["bu"], "be": d["be"], "base_labels": d["base_labels"]}


# ----------------------------------------------------------------------------------------------- training
def sched(opt, u: int, sch: dict) -> None:
    for g in opt.param_groups:
        g["lr"] = g["base"] * lr_factor(int(u), int(sch["warmup"]), int(sch["updates"]))


def opt_of(params_groups):
    return torch.optim.AdamW([{"params": p, "lr": 0.0, "base": lr, "name": n} for n, p, lr in params_groups], betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)


def head_and_opt(init):
    h = ir.copied_head(*init); return h, opt_of([("classifier", list(h.parameters()), LR_HEAD)])


def branch_and_opt(ctx: Ctx, s: int, device, rep_lr: float = LR_REP):
    m = ic.branch(ctx.a08(s, device), True); tr = ic.trainable(m)
    return m, opt_of([("representation", [p for n, p in tr if not n.startswith("classifier.")], rep_lr), ("classifier", [p for n, p in tr if n.startswith("classifier.")], LR_HEAD)])


class Upd(Dataset):
    def __init__(self, ctx: Ctx, s: int, R: dict, pkey: str, us) -> None:
        self.ctx, self.s, self.R, self.pkey, self.us, self.px = ctx, s, R, pkey, list(us), None

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        if self.px is None:
            self.px = Prefix(self.ctx, self.s)
        u = self.us[i]; rows, w, ix, y = t29.batch_rows(self.R, self.R["bt"][u], self.R[self.pkey][u]); uq, inv = np.unique(rows.numpy(), return_inverse=True)
        return u, self.px.get(uq), torch.from_numpy(inv), w, ix, y


def f_steps(h, opt, emb: torch.Tensor, R: dict, pkey: str, us, sch: dict, log: list) -> None:
    for u in us:
        sched(opt, u, sch); rows, w, ix, y = t29.batch_rows(R, R["bt"][u], R[pkey][u]); loss = t29.pooled_ce(h(emb[rows]), w, ix, y).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(h.parameters(), CLIP)); opt.step(); log.append({"u": int(u), "loss": float(loss.detach()), "grad_norm": gn})


def t_steps(m, opt, ctx: Ctx, s: int, R: dict, pkey: str, us, sch: dict, device, log: list, workers: int = 4) -> float:
    params = [p for _, p in ic.trainable(m)]; g = 0.0
    for u, tok, inv, w, ix, y in DataLoader(Upd(ctx, s, R, pkey, us), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True):
        u = int(u); sched(opt, u, sch); inv, w, ix, y = (x.to(device) for x in (inv, w, ix, y)); torch.cuda.synchronize(); a = time.time()
        loss = t29.pooled_ce(t29.t_forward(m, tok, device)[inv], w, ix, y).mean(); opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(params, CLIP)); opt.step()
        torch.cuda.synchronize(); g += time.time() - a; log.append({"u": u, "loss": float(loss.detach()), "grad_norm": gn})
        if (u + 1) % 135 == 0:
            print(f"  update {u + 1}/{sch['updates']}: loss {np.mean([x['loss'] for x in log[-135:]]):.4f}", flush=True)
    return g


def fit_readout(arm: str, model, R: dict, emb: np.ndarray, px, device) -> dict:
    """Fit-fold P readout (training data): accuracy, fixed / broken against Base."""
    pk = R["P"]; idx = np.arange(len(pk["labels"])); rows = np.unique(pk["rows"]); z = arm_logits(arm, model, rows, emb, px, device); pp = t29.pooled_probs(z, pk, idx); pred = pp.argmax(1); y = pk["labels"]; wr = pk["base"] != y
    return {"accuracy": float((pred == y).mean()), "base_accuracy": float((~wr).mean()), "fixed": int((wr & (pred == y)).sum()), "broken": int((~wr & (pred != y)).sum()), "base_wrong": int(wr.sum())}


def arm_logits(arm: str, model, rows: np.ndarray, emb_full: np.ndarray, px, device) -> torch.Tensor:
    """Per-window logits at global cache rows (dense up to max row); F reads embeddings (indexed by the same global rows)."""
    full = torch.zeros(int(rows.max()) + 1, 5)
    with torch.no_grad():
        if arm.startswith("F"):
            full[rows] = model(torch.from_numpy(emb_full[rows]))
        else:
            for b in range(0, len(rows), 256):
                r = rows[b:b + 256]; full[r] = t29.t_forward(model, px.get(r), device).float().cpu()
    return full


def seed_emb(ctx: Ctx, s: int) -> np.ndarray:
    """Embeddings of both folds at their global cache rows."""
    return np.concatenate([outputs(ctx, s, f)["emb"] for f in FOLDS])


def train(ctx: Ctx, s: int, device, workers: int) -> None:
    sch = ctx.schedule(); emb = seed_emb(ctx, s); e = torch.from_numpy(emb); init = ctx.a27.init(s); px = Prefix(ctx, s)
    for f in FOLDS:
        R = runs(ctx, s, f)
        for arm in ARMS:
            t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0; pkey = "bu" if arm.endswith("U") else "be"
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); log = []
                if arm.startswith("F"):
                    torch.set_num_threads(1); model, opt = head_and_opt(init); f_steps(model, opt, e, R, pkey, range(int(sch["updates"])), sch, log); state = {f"classifier.{k}": v.detach().clone() for k, v in model.state_dict().items()}; ok_p = True
                else:
                    torch.set_num_threads(8); model, opt = branch_and_opt(ctx, s, device); pd0 = t28.prefix_digest(model); gsec = t_steps(model, opt, ctx, s, R, pkey, range(int(sch["updates"])), sch, device, log, workers)
                    ok_p = t28.prefix_digest(model) == pd0; state = {n: p.detach().cpu().clone() for n, p in ic.trainable(model)}
                    if not ok_p:
                        raise SystemExit("a frozen prefix parameter changed")
                fr = fit_readout(arm, model, R, emb, px, device); torch.save(state, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "frozen_prefix_unchanged": ok_p, "fit_fold_readout": fr,
                       "loss": {"first 135": float(np.mean([x["loss"] for x in log[:135]])), "last 270": float(np.mean([x["loss"] for x in log[-270:]]))}, "seconds": time.time() - t0, "gpu_seconds": gsec, "schedule": sch}
                write_json(rd / "run_manifest.json", man); print(f"[{arm}_s{s}_f{f}] {len(log)} updates in {man['seconds']:.0f} s; fit-fold fixed {fr['fixed']}/{fr['base_wrong']} broken {fr['broken']}; final {man['final_sha256'][:12]}", flush=True); ok = True
            except BaseException as e_:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e_).__name__}: {str(e_)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} ({'CPU' if arm.startswith('F') else 'GPU job wall time'})", "cpu" if arm.startswith("F") else "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- check (GPU)
def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config)}
    try:
        s = 0; sch = {"updates": 1350, "warmup": 135}; emb = seed_emb(ctx, s); px = Prefix(ctx, s); init = ctx.a27.init(s); a08 = ctx.a08(s, device); params = ctx.a27.params(s); out = {}
        for f in FOLDS:
            seqs = load_pool(ctx, f); W = pool_W(seqs); O = outputs(ctx, s, f); off = int(O["offset"]); R = runs(ctx, s, f); rows = np.arange(len(W["seq"])) + off
            pick = np.random.default_rng([20261024, 31, 9, f]).choice(len(W["seq"]), 64, replace=False); fp = t28.frame_paths(seqs); err = 0.0; mis = 0
            for k in pick:
                qi = int(W["seq"][k]); Wq = pool_W([seqs[qi]]); j = int(np.flatnonzero((W["seq"] == qi))[0]); kk = int(k) - j; refs = sorted({tuple(r) for r in seqs[qi]["refs"]}); frames = rr.decode_frames(refs, fp, 0)
                with torch.no_grad():
                    z = a08(tm.to_rgb(torch.from_numpy(frames[Wq["ref"][[kk]]]).to(device)), tm.marker_planes(torch.from_numpy(Wq["marker"][[kk]]).long().to(device))).float().cpu().numpy()[0]
                err = max(err, float(np.abs(z - O["logits"][k]).max())); mis += int(z.argmax() != O["logits"][k].argmax())
            h0 = ir.copied_head(*init); zf = arm_logits("F", h0, rows, emb, None, device)[rows]; zt = arm_logits("T", ic.branch(a08, True), rows[:2000], emb, px, device)[rows[:2000]]
            base_lab = R["base_labels"]; pz = rr.per_sample(W, zf.numpy(), seqs); zero_ok = all(np.array_equal(ir.relabel(b_, pz[q["id"]]), b_) for q, b_ in zip(seqs, np.split(base_lab, np.cumsum([len(q["labels"]) for q in seqs])[:-1])))
            out[str(f)] = {"full_forward_vs_cached_logits_max_abs (64 windows)": err, "argmax_mismatches": mis, "F_zero_update_logits_max_abs": float((zf - torch.from_numpy(O["logits"])).abs().max()),
                           "T_zero_update_logits_max_abs (2,000 windows)": float((zt - torch.from_numpy(O["logits"][:2000])).abs().max()), "F_zero_update_sequences_equal_base": zero_ok}
        res["parity"] = out
        a15w = np.load(ROOT / "runs/video_router_data_v1/features/windows_s0.npz"); P15 = ctx.a27.pool(); fit15 = bc.members(P15, "large")
        ids15 = {}
        for i in fit15:
            q = P15["seqs"][i]
            if q["kind"] == "native":
                Wq = pool_W([q]); rows15 = np.flatnonzero(P15["W"]["seq"] == i)
                for kk, g in enumerate(rows15):
                    ids15[(q["id"], int(Wq["centre"][kk]))] = int(g)
        common, mx = 0, 0.0
        for f in FOLDS:
            seqs = load_pool(ctx, f); W = pool_W(seqs); O = outputs(ctx, s, f)
            for k in range(len(W["seq"])):
                q = seqs[int(W["seq"][k])]
                if q["kind"] == "native" and (q["id"], int(W["centre"][k])) in ids15:
                    common += 1; mx = max(mx, float(np.abs(O["logits"][k] - a15w["logits"][ids15[(q["id"], int(W["centre"][k]))]]).max()))
        res["A15_parity_on_identical_native_windows"] = {"windows": common, "logits_max_abs": mx}
        R = runs(ctx, s, 0); z = torch.randn(9, 5, dtype=torch.float64); ww = torch.tensor([.5, .3, .2, 1., .25, .25, .25, .25, 1.], dtype=torch.float64); ii = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3]); yy = torch.tensor([1, 4, 0, 2])
        p_ = torch.softmax(z, -1).numpy(); ref = [-np.log(sum(ww[k].item() * p_[k, yy[j]] for k in range(9) if ii[k] == j)) for j in range(4)]; res["pooled_ce_numpy_max_abs"] = float(np.abs(t29.pooled_ce(z, ww, ii, yy).numpy() - np.asarray(ref)).max())
        item = Upd(ctx, s, R, "be", [200])[0]; ma, oa = branch_and_opt(ctx, s, device); mb, ob = branch_and_opt(ctx, s, device); ga, gb = {}, {}
        for mm_, micro, gd in ((ma, None, ga), (mb, 7, gb)):
            u_, tok, inv, w_, ix_, y_ = item; inv, w_, ix_, y_ = (x.to(device) for x in (inv, w_, ix_, y_)); mm_.zero_grad(set_to_none=True); n = int(y_.shape[0])
            if micro is None:
                t29.pooled_ce(t29.t_forward(mm_, tok, device)[inv], w_, ix_, y_).mean().backward()
            else:
                for a in range(0, n, micro):
                    sel = (ix_ >= a) & (ix_ < a + micro); r = torch.unique(inv[sel]); loc = torch.searchsorted(r, inv[sel]); (t29.pooled_ce(t29.t_forward(mm_, tok[r.cpu()], device)[loc], w_[sel], ix_[sel] - a, y_[a:a + micro]).sum() / n).backward()
            gd.update({k: p.grad.detach().clone() for k, p in ic.trainable(mm_) if p.grad is not None})
        res["microbatch_grad_max_abs"] = float(max((ga[k] - gb[k]).abs().max() for k in ga)); res["grad_max"] = float(max(v.abs().max() for v in ga.values()))
        res["zero_grad_tensors"] = sorted(k for k, p in ic.trainable(ma) if p.grad is None or float(p.grad.abs().sum()) == 0)
        us = range(135, 145); hf, of = head_and_opt(init); lf = []; f_steps(hf, of, torch.from_numpy(emb), R, "be", us, sch, lf)
        mt, ot = branch_and_opt(ctx, s, device, rep_lr=0.0); rep0 = {k: p.detach().clone() for k, p in ic.trainable(mt) if not k.startswith("classifier.")}; pd0 = t28.prefix_digest(mt); lt = []; t_steps(mt, ot, ctx, s, R, "be", us, sch, device, lt, 2)
        res["forced_equal"] = {"classifier_max_abs": float(max((mt.classifier.weight.detach().cpu() - hf.weight.detach()).abs().max(), (mt.classifier.bias.detach().cpu() - hf.bias.detach()).abs().max())),
                               "classifier_moved": float((hf.weight.detach() - init[0]).abs().max()), "representation_unchanged": all(torch.equal(p.detach(), rep0[k]) for k, p in ic.trainable(mt) if not k.startswith("classifier.")),
                               "prefix_unchanged": t28.prefix_digest(mt) == pd0}
        a_, oa_ = branch_and_opt(ctx, s, device); t_steps(a_, oa_, ctx, s, R, "be", range(132, 138), sch, device, [], 2); b_, ob_ = branch_and_opt(ctx, s, device); t_steps(b_, ob_, ctx, s, R, "be", range(132, 135), sch, device, [], 2)
        st = {"t": {k: p.detach().clone() for k, p in ic.trainable(b_)}, "o": ob_.state_dict()}; c_, oc_ = branch_and_opt(ctx, s, device)
        with torch.no_grad():
            for k, p in ic.trainable(c_):
                p.copy_(st["t"][k])
        oc_.load_state_dict(st["o"]); t_steps(c_, oc_, ctx, s, R, "be", range(135, 138), sch, device, [], 2); pa = dict(ic.trainable(a_))
        res["resume_params_max_abs"] = float(max((pa[k] - p).detach().abs().max() for k, p in ic.trainable(c_)))
        tr = ic.trainable(ma); res["trainable"] = {"nominal": int(sum(p.numel() for _, p in tr)), "binds_exactly": sorted(id(p) for g in oa.param_groups for p in g["params"]) == sorted(id(p) for _, p in tr)}
        tm0 = time.time(); m5, o5 = branch_and_opt(ctx, s, device); t_steps(m5, o5, ctx, s, R, "bu", range(300, 330), sch, device, [], workers); spu = (time.time() - tm0) / 30
        crec = json.loads((ctx.out / "cache" / "prefix_s0.json").read_text()); pass_h = (crec["gpu_seconds"] + crec["decode_seconds"]) / 3600 * 1.1
        spent = vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; rows_total = sum(crec["rows"].values()) if isinstance(crec["rows"], dict) else crec["rows"]
        proj = {"T_training_8_models": 8 * 1350 * spu / 3600, "seed2_pass_and_cache": pass_h, "readouts_T_8_models": 8 * (rows_total / 2) * 0.0012 / 3600 + 0.1, "fit_readouts": 8 * (rows_total / 2) * 0.0012 / 3600}
        res["timing"] = {"T_seconds_per_update_wall": spu, "seed0_pass_hours": crec["gpu_seconds"] / 3600}; res["budget"] = {"spent": spent, "projection": spent + sum(proj.values()), "parts": proj, "cap": 8.0}
        fits = res["budget"]["projection"] <= 8.0
        updates = 1350 if fits else int(1350 * (8.0 - spent - proj["seed2_pass_and_cache"] - proj["readouts_T_8_models"] - proj["fit_readouts"]) / proj["T_training_8_models"])
        res["schedule_decision"] = {"updates": max(1, min(1350, updates)), "warmup": max(1, round(135 * min(1350, updates) / 1350)), "shortened": not fits}
        po = res["parity"]
        res["ok"] = bool(all(v["full_forward_vs_cached_logits_max_abs (64 windows)"] <= 1e-4 and v["argmax_mismatches"] == 0 and v["F_zero_update_logits_max_abs"] <= 1e-4 and v["T_zero_update_logits_max_abs (2,000 windows)"] <= 1e-4 and v["F_zero_update_sequences_equal_base"] for v in po.values())
                         and res["A15_parity_on_identical_native_windows"]["logits_max_abs"] <= 1e-4 and res["pooled_ce_numpy_max_abs"] < 1e-10 and res["microbatch_grad_max_abs"] <= 1e-5 * max(1.0, res["grad_max"])
                         and res["forced_equal"]["classifier_max_abs"] <= 1e-4 and res["forced_equal"]["representation_unchanged"] and res["forced_equal"]["prefix_unchanged"] and res["resume_params_max_abs"] <= 1e-5
                         and res["trainable"]["nominal"] == 7497989 and res["trainable"]["binds_exactly"] and len(res["zero_grad_tensors"]) == 4)
        res["finished"] = now()
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            write_json(p, json.loads(json.dumps(res, default=str)))
        if res["ok"]:
            write_json(ctx.out / "schedule.json", {**res["schedule_decision"], "fixed_at": now(), "reason": "engineering timing in the check stage, before formal training"})
        print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- held-out readout (after the weights are committed)
def load_model(ctx: Ctx, arm: str, s: int, f: int, device):
    man = json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    p = ctx.out / "train" / f"{arm}_s{s}_f{f}" / "final.pt"
    if sha256_file(p) != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    st = torch.load(p, map_location="cpu", weights_only=True)
    if arm.startswith("F"):
        h = torch.nn.Linear(512, 5); h.load_state_dict({k.split(".", 1)[1]: v for k, v in st.items()}); return h.eval()
    m = ic.branch(ctx.a08(s, device), True)
    with torch.no_grad():
        for k, p_ in ic.trainable(m):
            p_.copy_(st[k].to(p_.device))
    return m.eval()


def run_table(pp: np.ndarray, pk: dict, seqs: list[dict], sess: np.ndarray) -> dict:
    """All held-out P runs at their real proportions, with breakdowns."""
    y, b = pk["labels"], pk["base"]; pred = pp.argmax(1); wr = b != y; L = pk["b"] - pk["a"]; kinds = np.asarray([seqs[i]["kind"] for i in pk["seq"]]); pure = pk["purity"] >= 1.0
    def part(mask):
        w_, c_ = wr & mask, (~wr) & mask; fx, br = int((w_ & (pred == y)).sum()), int((c_ & (pred != y)).sum())
        return {"runs": int(mask.sum()), "base_wrong": int(w_.sum()), "fixed": fx, "base_correct": int(c_.sum()), "broken": br, "net_fixed": fx - br, "accuracy_difference": (fx - br) / max(1, int(mask.sum())),
                "fix_rate": fx / max(1, int(w_.sum())), "damage_rate": br / max(1, int(c_.sum())), "sessions_with_fixes": len({sess[k] for k in np.flatnonzero(w_ & (pred == y))}), "sessions_with_breaks": len({sess[k] for k in np.flatnonzero(c_ & (pred != y))})}
    out = {"all": part(np.ones(len(y), bool)), "A4C_to_A5C_errors": part(wr & (y == 2) & (b == 3)), "A5C_to_A4C_errors": part(wr & (y == 3) & (b == 2)), "native": part(kinds == "native"), "multi": part(kinds == "multi"),
           "pure": part(pure), "mixed": part(~pure), "short_le16": part(L <= 16), "long": part(L > 16), **{f"class_{c}": part(y == c) for c in range(5)}}
    out["sessions"] = len(set(sess.tolist())); return out


def seq_rows(a8cfg, P: dict, preds: dict, ev) -> dict:
    m, t = bc.evaluate_pool(a8cfg, preds, ev); return {"row": bce.check_row(m), "tables": t, "preds": preds}


def readout(ctx: Ctx, s: int, device) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"seed": s, "models": {}}
    try:
        if (ctx.out / f"readout_s{s}.json").exists():
            raise SystemExit("this seed's held-out readout exists; it runs once")
        a8 = rsh.a08_ctx(); emb = seed_emb(ctx, s); px = Prefix(ctx, s); data = vpr.training_data(ctx.a15.a10.a07p)
        for f in FOLDS:
            h = 1 - f; seqs = load_pool(ctx, h); W = pool_W(seqs); O = outputs(ctx, s, h); off = int(O["offset"]); Rh = runs(ctx, s, h); pk = Rh["P"]; sess = np.load(ctx.out / "data" / f"run_sessions_s{s}_f{h}.npy", allow_pickle=True)
            P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"); split_at = np.cumsum([len(q["labels"]) for q in seqs])[:-1]
            base_preds = {q["id"]: b_ for q, b_ in zip(seqs, np.split(Rh["base_labels"], split_at))}; rows = np.arange(len(W["seq"])) + off
            res = {"Base": {"sequence": seq_rows(a8.cfg, P, base_preds, ev)}}
            for arm in ARMS:
                m = load_model(ctx, arm, s, f, device); z = arm_logits(arm, m, rows, emb, px, device); pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); zz = z[rows].numpy()
                ps = rr.per_sample(W, zz, seqs); preds = {q["id"]: ir.relabel(base_preds[q["id"]], ps[q["id"]]) for q in seqs}
                res[arm] = {"runs": run_table(pp, pk, seqs, sess), "sequence": seq_rows(a8.cfg, P, preds, ev), "pred_runs": pp.argmax(1)}
            rows_out = {k: {"sequence_row": v["sequence"]["row"], "counts": rsh.counts_of(seqs, v["sequence"]), **({"runs": v["runs"]} if "runs" in v else {})} for k, v in res.items()}
            cmp_, acc_ = {}, {}
            systems = {f"{k} s{s}": {"heldout": v["sequence"]} for k, v in res.items()}
            import scripts.report_hard_replay as rhr
            rhr.compare(seqs, systems, "heldout", s, {"F_E - F_U": ("F_E", "F_U"), "T_E - T_U": ("T_E", "T_U"), "T_U - F_U": ("T_U", "F_U"), "T_E - F_E": ("T_E", "F_E"), **{f"{a} - Base": (a, "Base") for a in ARMS}}, cmp_, acc_)
            np.savez(ctx.out / f"heldout_preds_s{s}_f{f}.npz", **{a: res[a]["pred_runs"] for a in ARMS}, labels=pk["labels"], base=pk["base"], sessions=sess.astype(str))
            rec["models"][f"fit_f{f}_heldout_f{h}"] = {"rows": rows_out, "comparisons": cmp_, "accounting": acc_}
            print(f"seed {s} fit fold {f} -> held-out {h}: " + json.dumps({a: rows_out[a]["runs"]["all"] for a in ARMS}), flush=True)
        ids = {q["id"] for f in FOLDS for q in load_pool(ctx, f) if q["kind"] == "native"} | {r[0] for f in FOLDS for q in load_pool(ctx, f) for r in q["refs"]}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "readout")
        for p in (ctx.out / f"readout_s{s}.json", ctx.results / f"readout_s{s}.json"):
            write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"held-out readout seed {s} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- report (CPU)
def report(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"oof": {}, "bootstrap": {}}
    try:
        for s in SEEDS:
            arrs = {f: dict(np.load(ctx.out / f"heldout_preds_s{s}_f{f}.npz", allow_pickle=True)) for f in FOLDS}
            y = np.concatenate([arrs[f]["labels"] for f in FOLDS]); b = np.concatenate([arrs[f]["base"] for f in FOLDS]); sess = np.concatenate([np.char.add(f"f{f}:", arrs[f]["sessions"].astype(str)) for f in FOLDS])
            pred = {a: np.concatenate([arrs[f][a] for f in FOLDS]) for a in ARMS}; wr = b != y; us = np.unique(sess); gi = np.searchsorted(us, sess)
            def net(p):
                return (wr & (p == y)).astype(float) - ((~wr) & (p != y)).astype(float)
            oof = {a: {"base_wrong": int(wr.sum()), "fixed": int((wr & (pred[a] == y)).sum()), "base_correct": int((~wr).sum()), "broken": int((~wr & (pred[a] != y)).sum())} for a in ARMS}
            for a in ARMS:
                o = oof[a]; o["net_fixed"] = o["fixed"] - o["broken"]; o["fix_rate"] = o["fixed"] / max(1, o["base_wrong"]); o["damage_rate"] = o["broken"] / max(1, o["base_correct"]); o["accuracy_difference"] = o["net_fixed"] / len(y)
            rng = np.random.default_rng([20261024, 31, 77, s]); draws = rng.integers(0, len(us), (2000, len(us))); wts = np.stack([np.bincount(d, minlength=len(us)) for d in draws]).astype(float)
            bs = {}
            for name, (x, z) in {"F_E - F_U": ("F_E", "F_U"), "T_E - T_U": ("T_E", "T_U"), "T_U - F_U": ("T_U", "F_U"), "T_E - F_E": ("T_E", "F_E"), **{f"{a} - Base": (a, None) for a in ARMS}}.items():
                d = net(pred[x]) - (net(pred[z]) if z else 0.0); per = np.bincount(gi, weights=d, minlength=len(us)); cnt = np.bincount(gi, minlength=len(us)).astype(float)
                vals = (wts @ per) / (wts @ cnt); bs[name] = {"net_fixed_difference": float(d.sum()), "accuracy_difference": float(d.sum() / len(y)), "ci95": np.percentile(vals, [2.5, 97.5]).tolist()}
            rec["oof"][str(s)] = oof; rec["bootstrap"][str(s)] = {"unit": "majority-recording session of each run (approximation: runs spanning sessions are assigned to one)", "sessions": int(len(us)), "draws": 2000, "results": bs}
        for p in (ctx.out / "report.json", ctx.results / "report.json"):
            write_json(p, json.loads(json.dumps(rec, default=float)))
        print(json.dumps(rec, indent=1, default=float)[:5000], flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "report (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("split", "infer", "base", "check", "train", "readout", "report"), required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("split", "base", "report"):
        return split(ctx) if a.stage == "split" else (base(ctx, a.seed) if a.stage == "base" else report(ctx))
    vsh.determinism(); device = torch.device("cuda")
    {"infer": lambda: infer(ctx, a.seed, device, a.workers), "check": lambda: check(ctx, device, a.workers), "train": lambda: train(ctx, a.seed, device, a.workers), "readout": lambda: readout(ctx, a.seed, device)}[a.stage]()


if __name__ == "__main__":
    main()
