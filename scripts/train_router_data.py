#!/usr/bin/env python
"""A15 (configs/experiments/video_router_data_v1.yaml): the session-disjoint router-fit/check split, the generated pools
(large fit with a nested small fit, and check), chunked frozen A08 inference on whole sequences, A14's zones, P features and
single-zone utility labels, and the matched small-P/large-P routers (CPU).

  --probe              pools, then decode and frozen inference of the first chunk only; a budget projection
  --prepare            pools, chunked frozen A08 inference (seeds 0 and 2), Base/Alternative, zones, P features and labels
  --check              zone self-test, pool rebuild, isolation, cache reproduction, no-op/accept-all, optimiser, starts, resume
  --train S [S ...]    small-P and large-P routers of the seeds (CPU), with check logging
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.local_router as lr  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.compose.recipe_builder import Pool, make_multi_recipes  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402

A14_CONFIG = "configs/experiments/video_local_router_v1.yaml"
CHUNK = 150
ARMS = {"small-P": "small_fit", "large-P": "large_fit"}
UPDATES, BATCH, WARM = 1350, 128, 135
CAND_KEYS = ("seq", "a", "b", "fp", "pos", "du", "dtp", "dfp", "dfn", "kind")
now, write_json, ledger_append, ledger_hours = tl.now, tl.write_json, tl.ledger_append, tl.ledger_hours


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vr = cfg, config, cfg["video_router_data"]
        self.a14 = tl.Ctx(load_cfg(SimpleNamespace(config=A14_CONFIG, set=[])), A14_CONFIG); self.a10 = self.a14.a10
        o = self.vr["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]; self.ledger = ROOT / self.vr["budget"]["ledger"]
        self.scale = float(self.vr["pools"]["scale"])


# ----------------------------------------------------------------------------------------------- pools
def frag_key(f) -> list:
    return [str(f.video_id), int(f.label), str(f.variant), int(f.start), int(f.end)]


def pools(ctx: Ctx) -> dict:
    """Sessions split once; every stream drawn from its own group's training index only; natives under per-class and
    per-session caps; the small fit pool drawn from the large one. Nothing here reads a prediction."""
    data = vpr.training_data(ctx.a10.a07p); idx = vsf.training_index(data.dir / "index.csv")
    sess = sorted({session_of(v) for v in data.clips}); perm = np.random.default_rng([20261021, 15, 1]).permutation(len(sess))
    chk = {sess[i] for i in perm[: int(round(0.2 * len(sess)))]}
    clips = {"fit": [v for v in data.clips if session_of(v) not in chk], "check": [v for v in data.clips if session_of(v) in chk]}
    a01 = ctx.a10.a07p.a01().cfg; rc, mc = a01["recipes"], a01.get("multi", {}); n = lambda x: int(round(x * ctx.scale))  # noqa: E731

    def gen(group: str, count: int, sub_seed: int) -> list[dict]:
        sub = idx[idx["video_id"].isin(set(clips[group]))]
        rec = make_multi_recipes(Pool.from_index(sub["video_id"], sub["label_index"], sub["n_samples"]), count, np.random.default_rng([20261021, 15, sub_seed]),
                                 n_fragments=tuple(mc.get("n_fragments", (4, 8))), edit_variant=rc["edit_variant"], edit_prob=float(mc.get("edit_prob", 0.5)),
                                 min_len=int(rc["min_len"]), max_len=int(rc["max_len"]), reuse=True, p_same_next=float(mc.get("p_same_next", 0.35)))
        out = []
        for k, r in enumerate(rec):
            labels, _ = vsf.stream_labels_joins(r.fragments)
            out.append({"id": f"a15_{group}_{k:04d}", "kind": "multi", "group": group, "labels": labels.astype(np.int64), "refs": vsf.stream_refs(r.fragments, data.frame_idx),
                        "frags": [frag_key(f) for f in r.fragments]})
        return out

    def nat(cands: list[str], per_class: int, sub_seed: int) -> tuple[list[str], dict]:
        rng = np.random.default_rng([20261021, 15, sub_seed]); out, short = [], {}
        for c in range(5):
            cl = [v for v in cands if data.label[v] == c]; per, took = {}, 0
            for j in rng.permutation(len(cl)):
                v = cl[j]; s = session_of(v)
                if per.get(s, 0) >= 2:
                    continue
                per[s] = per.get(s, 0) + 1; out.append(v); took += 1
                if took == per_class:
                    break
            short[str(c)] = per_class - took
        return out, short

    def native(v: str, group: str) -> dict:
        f = data.frame_idx(v); return {"id": v, "kind": "native", "group": group, "labels": np.full(len(f), data.label[v], np.int64), "refs": vsf.native_refs(v, f), "frags": [v]}

    fit_multi, chk_multi = gen("fit", n(1500), 2), gen("check", n(600), 3)
    fit_nat, fshort = nat(clips["fit"], n(200), 4); chk_nat, cshort = nat(clips["check"], n(40), 5)
    small_multi = set(np.random.default_rng([20261021, 15, 6]).choice(len(fit_multi), size=n(300), replace=False).tolist()); small_nat, sshort = nat(fit_nat, n(40), 7)
    seqs = fit_multi + [native(v, "fit") for v in fit_nat] + chk_multi + [native(v, "check") for v in chk_nat]
    small = np.asarray([i in small_multi for i in range(len(fit_multi))] + [v in set(small_nat) for v in fit_nat] + [False] * (len(chk_multi) + len(chk_nat)), bool)
    group = np.asarray([q["group"] for q in seqs])
    rec_of = {g: {r[0] for q in seqs if q["group"] == g for r in q["refs"]} for g in ("fit", "check")}
    if rec_of["fit"] & rec_of["check"] or {session_of(v) for v in rec_of["fit"]} & {session_of(v) for v in rec_of["check"]}:
        raise SystemExit("router fit and check share a recording or a session")
    allowed = {g: set(clips[g]) for g in clips}
    if not all(r[0] in allowed[q["group"]] for q in seqs for r in q["refs"]):
        raise SystemExit("a sequence uses a recording outside its group")
    manifest = [{"id": q["id"], "kind": q["kind"], "group": q["group"], "small": bool(small[i]), "frags": q["frags"]} for i, q in enumerate(seqs)]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    a01_bank = {json.dumps([frag_key(f) for f in r.fragments]) for r in data.recipes}

    def reuse(sel: list[dict]) -> dict:
        per = {}
        for q in sel:
            for f in q["frags"]:
                per[f[0]] = per.get(f[0], 0) + 1
        v = np.asarray(list(per.values()) or [0])
        return {"streams": len(sel), "fragments_mean": float(np.mean([len(q["frags"]) for q in sel])) if sel else 0.0, "samples_mean": float(np.mean([len(q["labels"]) for q in sel])) if sel else 0.0,
                "distinct_recordings": len(per), "fragments_per_recording_mean": float(v.mean()), "fragments_per_recording_max": int(v.max()),
                "streams_identical_to_an_a01_bank_stream": int(sum(json.dumps(q["frags"]) in a01_bank for q in sel))}
    small_sel = [q for i, q in enumerate(seqs) if small[i] and q["kind"] == "multi"]
    counts = {"sessions": {"fit": len(sess) - len(chk), "check": len(chk)}, "training_clips": {"fit": len(clips["fit"]), "check": len(clips["check"])},
              "large_fit": {"multi": len(fit_multi), "native": len(fit_nat), "native_shortfall_by_class": fshort}, "check": {"multi": len(chk_multi), "native": len(chk_nat), "native_shortfall_by_class": cshort},
              "small_fit": {"multi": len(small_multi), "native": len(small_nat), "native_shortfall_by_class": sshort},
              "recordings_used": {g: len(rec_of[g]) for g in rec_of}, "sessions_used": {g: len({session_of(v) for v in rec_of[g]}) for g in rec_of},
              "recipe_reuse": {"large_fit": reuse(fit_multi), "small_fit": reuse(small_sel), "check": reuse(chk_multi)}, "scale": ctx.scale}
    return {"seqs": seqs, "group": group, "small": small, "manifest": manifest, "digest": digest, "counts": counts}


def arm_masks(P: dict, seq_index: np.ndarray) -> dict:
    g, sm = P["group"][seq_index], P["small"][seq_index]
    return {"large_fit": g == "fit", "small_fit": (g == "fit") & sm, "check": g == "check"}


# ----------------------------------------------------------------------------------------------- inference and candidates
def models(ctx: Ctx, device) -> dict:
    out = {}
    for s in (0, 2):
        m, ssha = tl.tf.frozen(ctx.a14.a13, s, device); head, thr, hsha = ctx.a14.alt(s)
        out[s] = {"model": m, "digest": tl.tf.path_digest(m), "start_sha256": ssha, "head": head, "threshold": thr, "head_sha256": hsha, "bparams": ctx.a14.base_params(s)}
    return out


def candidates_p(seqs: list[dict], probs: dict, base: dict, alt: dict, with_labels: bool = True, offset: int = 0) -> dict:
    """Per zone: sequence index, [a, b), the 52 P features and (with labels) the single-zone utility; lists, see to_arrays."""
    rows = {k: [] for k in CAND_KEYS}
    for qi, q in enumerate(seqs):
        b, a_ = base[q["id"]], alt[q["id"]]; zs = lr.zones(b, a_)
        if not zs:
            continue
        util = lr.zone_utilities(b, a_, q["labels"], zs) if with_labels else [{"positive": False, "f1_base": 0.0, "f1_after": 0.0, "dtp": 0, "dfp": 0, "dfn": 0}] * len(zs)
        for z, u in zip(zs, util):
            for k, v in (("seq", qi + offset), ("a", z[0]), ("b", z[1]), ("kind", q["kind"]), ("fp", lr.features_p(probs[q["id"]], b, a_, z)), ("pos", bool(u["positive"])),
                         ("du", float(u["f1_after"] - u["f1_base"])), ("dtp", int(u["dtp"])), ("dfp", int(u["dfp"])), ("dfn", int(u["dfn"]))):
                rows[k].append(v)
    return rows


def to_arrays(rows: dict) -> dict:
    if not rows["seq"]:
        return {"seq": np.zeros(0, int), "a": np.zeros(0, int), "b": np.zeros(0, int), "fp": np.zeros((0, 52)), "pos": np.zeros(0, bool), "du": np.zeros(0), "dtp": np.zeros(0, int),
                "dfp": np.zeros(0, int), "dfn": np.zeros(0, int), "kind": np.zeros(0, "<U6")}
    return {k: np.asarray(v) for k, v in rows.items()}


def process(chunk: list[dict], fp, M: dict, device, workers: int) -> tuple[dict, dict]:
    refs = rw.half_refs(chunk); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(chunk, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
    t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); dsec = time.time() - t0; out, gsec = {}, 0.0
    for s, mm in M.items():
        lg, em, sec = rsh.infer(mm["model"], W, orders, frames, device); gsec += sec
        probs = rr.per_sample(W, lg, chunk); edges = rvs.edge_scores(mm["head"], W, lg, em, chunk, torch.device("cpu")); base, alt = tl.systems(probs, edges, chunk, mm["bparams"], mm["threshold"])
        out[s] = {"seq": W["seq"], "centre": W["centre"], "logits": lg, "emb": em, "probs": probs, "base": base, "alt": alt}
    del frames
    return out, {"refs": len(refs), "windows": int(len(W["seq"])), "decode_seconds": dsec, "gpu_seconds": gsec}


def frame_paths(ctx: Ctx, seqs: list[dict]):
    vids = {r[0] for q in seqs for r in q["refs"]}
    if vids - set(prv.allowlist(ctx.a10.rv, ROOT)):
        raise SystemExit("a pool recording lies outside the training allowlist")
    return FramePaths(prv.frame_dirs(ctx.a10.paths, ctx.a10.rv, vids)), vids


def probe(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note, st = time.time(), now(), False, "", {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    try:
        P = pools(ctx); seqs = P["seqs"]; fp, vids = frame_paths(ctx, seqs); device = torch.device("cuda"); tl.tf.vsh.determinism(); M = models(ctx, device)
        refs_per_chunk = [len(rw.half_refs(seqs[c0:c0 + CHUNK])) for c0 in range(0, len(seqs), CHUNK)]; windows = int(sum(len(window_centres(len(q["labels"]), 5)) for q in seqs))
        _, st = process(seqs[:CHUNK], fp, M, device, workers)
        proj = {"sequences": len(seqs), "chunks": len(refs_per_chunk), "windows": windows, "decoded_refs_total": int(sum(refs_per_chunk)), "first_chunk": st,
                "projected_gpu_hours": st["gpu_seconds"] / st["windows"] * windows / 3600, "projected_decode_worker_cpu_hours": st["decode_seconds"] * workers / st["refs"] * sum(refs_per_chunk) / 3600,
                "caps": {"gpu": ctx.vr["budget"]["gpu_hours_max"], "cpu": ctx.vr["budget"]["cpu_hours_max"]}, "spent_before": {"gpu": ledger_hours(ctx.ledger, "gpu"), "cpu": ledger_hours(ctx.ledger, "cpu")},
                "pool_digest": P["digest"], "counts": P["counts"], "time": now()}
        flat = json.loads(json.dumps(proj, default=str)); assert_no_ids(flat, {q["id"] for q in seqs if q["kind"] == "native"} | vids | {session_of(v) for v in vids}, "probe")
        ctx.out.mkdir(parents=True, exist_ok=True); write_json(ctx.out / "probe.json", flat); print(json.dumps(flat, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(st["gpu_seconds"], total)
        ledger_append(ctx.ledger, "probe (first-chunk frozen inference, GPU)", "gpu", started, g, ok, note)
        ledger_append(ctx.ledger, f"probe (CPU incl. {workers} decode workers)", "cpu", started, (total - g) + st["decode_seconds"] * (workers - 1), ok, note)


def summary(c: dict, P: dict) -> dict:
    seqs = P["seqs"]; m = arm_masks(P, c["seq"]); out = {}; short = (c["b"] - c["a"]) <= 16
    for arm, mask in m.items():
        pos = c["pos"] & mask; members = sorted(set(c["seq"][mask].tolist())); recs = {r[0] for i in members for r in seqs[i]["refs"]}
        out[arm] = {"zones": int(mask.sum()), "positive": int(pos.sum()), "negative": int((mask & ~c["pos"]).sum()), "neutral": int((mask & (np.abs(c["du"]) < 1e-12)).sum()),
                    "harmful": int((mask & (c["du"] < -1e-12)).sum()), "short_zones": int((mask & short).sum()), "positive_short": int((pos & short).sum()),
                    "positive_recovers_true": int((pos & (c["dtp"] > 0)).sum()), "positive_removes_false_only": int((pos & (c["dtp"] == 0) & (c["dfp"] < 0)).sum()),
                    "positive_trades_match": int((pos & (c["dtp"] < 0)).sum()), "positive_by_kind": {k: int((pos & (c["kind"] == k)).sum()) for k in ("multi", "native")},
                    "zones_by_kind": {k: int((mask & (c["kind"] == k)).sum()) for k in ("multi", "native")}, "dtp_sum_positive": int(c["dtp"][pos].sum()), "dfp_sum_positive": int(c["dfp"][pos].sum()),
                    "sequences_with_zones": len(members), "sequences_with_positive": len(set(c["seq"][pos].tolist())), "recordings_with_zones": len(recs),
                    "sessions_with_zones": len({session_of(v) for v in recs}), "preparation_met": bool(pos.any() and (mask & ~c["pos"]).any()) if arm != "check" else None}
    return out


def prepare(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note, gsec, dsec = time.time(), now(), False, "", 0.0, 0.0
    try:
        P = pools(ctx); seqs = P["seqs"]; pr = ctx.out / "probe.json"
        if pr.exists() and json.loads(pr.read_text())["pool_digest"] != P["digest"]:
            raise SystemExit("the rebuilt pools differ from the probe's")
        fp, vids = frame_paths(ctx, seqs); device = torch.device("cuda"); tl.tf.vsh.determinism(); M = models(ctx, device)
        rows = {s: {k: [] for k in CAND_KEYS} for s in M}; win = {s: {k: [] for k in ("seq", "centre", "logits", "emb")} for s in M}
        preds = {s: {"base": [], "alt": []} for s in M}; nchunk = -(-len(seqs) // CHUNK)
        for k, c0 in enumerate(range(0, len(seqs), CHUNK)):
            chunk = seqs[c0:c0 + CHUNK]; res, st = process(chunk, fp, M, device, workers); gsec += st["gpu_seconds"]; dsec += st["decode_seconds"]
            for s, r in res.items():
                cr = candidates_p(chunk, r["probs"], r["base"], r["alt"], offset=c0)
                for key in CAND_KEYS:
                    rows[s][key] += cr[key]
                for key in ("centre", "logits", "emb"):
                    win[s][key].append(r[key])
                win[s]["seq"].append(r["seq"] + c0); preds[s]["base"] += [r["base"][q["id"]] for q in chunk]; preds[s]["alt"] += [r["alt"][q["id"]] for q in chunk]
            prog = {"stage": "prepare (data preparation + frozen A08 inference; no training)", "chunk": k + 1, "of": nchunk, "sequences_done": c0 + len(chunk), "sequences": len(seqs),
                    "elapsed_s": round(time.time() - t0, 1), "gpu_s": round(gsec, 1), "decode_s": round(dsec, 1),
                    "zones_so_far": {str(s): len(rows[s]["seq"]) for s in rows}, "positive_so_far": {str(s): int(sum(rows[s]["pos"])) for s in rows}}
            write_json(ctx.out / "progress.json", prog); print(json.dumps(prog), flush=True)
        for s, mm in M.items():
            if tl.tf.path_digest(mm["model"]) != mm["digest"] or mm["model"].training:
                raise SystemExit("the frozen path changed")
        fdir = ctx.out / "features"; fdir.mkdir(parents=True, exist_ok=True); lens = np.asarray([len(q["labels"]) for q in seqs]); off = np.concatenate([[0], np.cumsum(lens)])
        rec = {"config_sha256": sha256_file(ROOT / ctx.config), "pool_digest": P["digest"], "counts": P["counts"], "seeds": {}}
        for s, mm in M.items():
            c = to_arrays(rows[s]); c.update(arm_masks(P, c["seq"])); np.savez(fdir / f"candidates_s{s}.npz", **c)
            np.savez(fdir / f"windows_s{s}.npz", **{k: np.concatenate(v) for k, v in win[s].items()}, start_sha256=mm["start_sha256"], path_digest=mm["digest"])
            np.savez(fdir / f"preds_s{s}.npz", base=np.concatenate(preds[s]["base"]).astype(np.int8), alt=np.concatenate(preds[s]["alt"]).astype(np.int8),
                     labels=np.concatenate([q["labels"] for q in seqs]).astype(np.int8), offsets=off)
            rec["seeds"][str(s)] = {"arms": summary(c, P), "start_sha256": mm["start_sha256"], "path_digest": mm["digest"], "alternative_head_sha256": mm["head_sha256"], "alternative_threshold": mm["threshold"]}
            print(f"seed {s}: {json.dumps(rec['seeds'][str(s)]['arms'])}", flush=True)
        rec.update(decode_seconds=dsec, gpu_seconds=gsec, windows=int(sum(len(x) for x in win[0]["seq"])), finished=now())
        ids = {q["id"] for q in seqs if q["kind"] == "native"} | vids | {session_of(v) for v in vids}; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        write_json(ctx.out / "pool_manifest.json", {"digest": P["digest"], "sequences": P["manifest"]})
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        ledger_append(ctx.ledger, "prepare (chunked frozen A08 inference, GPU)", "gpu", started, g, ok, note)
        ledger_append(ctx.ledger, f"prepare (CPU incl. {workers} decode workers, zones, labels)", "cpu", started, (total - g) + dsec * (workers - 1), ok, note)


# ----------------------------------------------------------------------------------------------- stored outputs
def load_cand(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "features" / f"candidates_s{s}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_preds(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "features" / f"preds_s{s}.npz", allow_pickle=False) as z:
        off = z["offsets"]; cut = lambda x: [x[off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]  # noqa: E731
        return {"base": cut(z["base"]), "alt": cut(z["alt"]), "labels": cut(z["labels"])}


def routed_lists(c: dict, pr: dict, accept: np.ndarray, members) -> dict:
    """Per member sequence index: Base with the accepted zones replaced by the Alternative."""
    out = {}
    for i in members:
        sel = np.flatnonzero(c["seq"] == i); zs = [(int(c["a"][k]), int(c["b"][k])) for k in sel]
        out[i] = lr.apply(pr["base"][i], pr["alt"][i], zs, [bool(accept[k]) for k in sel]) if sel.size else pr["base"][i].copy()
    return out


def pooled_f1(preds: dict, refs: list) -> dict:
    tp = npred = nref = 0
    for i, p in preds.items():
        r = lr.seq_f1(p, refs[i]); tp, npred, nref = tp + r["tp"], npred + r["pred"], nref + r["ref"]
    return {"f1": 2 * tp / max(1, npred + nref), "tp": tp, "pred": npred, "ref": nref}


def check_log(c: dict, pr: dict, kinds: dict, sc: np.ndarray, thr) -> dict:
    """Router-check logging only (never a choice): zone classes, balanced accuracy, accepted zones by utility sign and the
    pooled full-sequence segment F1@50 with all accepted zones applied."""
    m = c["check"]; acc = np.zeros(sc.size, bool) if thr == "noop" else sc >= float(thr); pm, nm = m & c["pos"], m & ~c["pos"]
    bal = float(0.5 * (acc[pm].mean() + (~acc[nm]).mean())) if (pm.any() and nm.any()) else "unavailable"
    out = {"threshold": thr, "zones": int(m.sum()), "positives": int(pm.sum()), "negatives": int(nm.sum()), "balanced_accuracy": bal, "accepted": int((acc & m).sum()),
           "accepted_beneficial": int((acc & pm).sum()), "accepted_neutral": int((acc & m & (np.abs(c["du"]) < 1e-12)).sum()), "accepted_harmful": int((acc & m & (c["du"] < -1e-12)).sum())}
    for kind, members in kinds.items():
        out[f"pooled_segment_f1_{kind}"] = pooled_f1(routed_lists(c, pr, acc & m, members), pr["labels"])
    return out


# ----------------------------------------------------------------------------------------------- routers
def fit_router(c: dict, mask: np.ndarray, s: int, upto: int = UPDATES, state: dict | None = None, start: int = 0):
    """A14's P recipe on one arm's fit zones: own normalisation and balanced BCE weights; batches from the shared base
    uniform numbers mapped onto this arm's fit list."""
    X = c["fp"].astype(np.float32); y = c["pos"].astype(np.float32); mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-6; Xn = ((X - mu) / sd).astype(np.float32)
    npos, nneg = float(y[mask].sum()), float((1 - y[mask]).sum()); wpos, wneg = (npos + nneg) / (2 * max(1.0, npos)), (npos + nneg) / (2 * max(1.0, nneg))
    m = tl.build("P", X.shape[1], s); opt = torch.optim.AdamW(m.parameters(), lr=0.0, weight_decay=1e-4)
    if state is not None:
        m.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    idx_fit = np.flatnonzero(mask); u0 = np.random.default_rng([20261021, 15, 8, s]).random((UPDATES, BATCH)); draws = np.minimum((u0 * len(idx_fit)).astype(np.int64), len(idx_fit) - 1)
    losses = []
    for u in range(start, upto):
        opt.param_groups[0]["lr"] = 1e-3 * lr_factor(u, WARM, UPDATES); bi = idx_fit[draws[u]]; xb, yb = torch.from_numpy(Xn[bi]), torch.from_numpy(y[bi])
        w = torch.where(yb > 0.5, torch.full_like(yb, wpos), torch.full_like(yb, wneg)); loss = (w * F.binary_cross_entropy_with_logits(m(xb), yb, reduction="none")).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0); opt.step(); losses.append(float(loss.detach()))
    return m, opt, losses, {"mu": mu, "sd": sd, "wpos": wpos, "wneg": wneg, "n_fit": int(mask.sum()), "n_pos": int(npos), "n_neg": int(nneg)}


def seq_meta(ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
    man = json.loads((ctx.out / "pool_manifest.json").read_text())["sequences"]
    return np.asarray([q["group"] for q in man]), np.asarray([q["kind"] for q in man])


def check_kinds(ctx: Ctx) -> dict:
    grp, kinds = seq_meta(ctx); return {k: [int(i) for i in np.flatnonzero((grp == "check") & (kinds == k))] for k in ("multi", "native")}


def train(ctx: Ctx, seeds: list[int]) -> None:
    ck = check_kinds(ctx)
    for s in seeds:
        c = load_cand(ctx, s); pr = load_preds(ctx, s); rd = ctx.out / "routers"; rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / f"check_reference_s{s}.json", {"Base": check_log(c, pr, ck, np.zeros(c["seq"].size), "noop"), "Alternative (accept all)": check_log(c, pr, ck, np.ones(c["seq"].size), 0.5)})
        for arm, key in ARMS.items():
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                mask = c[key]
                if int((c["pos"] & mask).sum()) == 0 or int((~c["pos"] & mask).sum()) == 0:
                    write_json(rd / f"{arm}_s{s}.json", {"arm": arm, "seed": s, "status": "preparation not met", "positive": int((c["pos"] & mask).sum()), "negative": int((~c["pos"] & mask).sum())})
                    note = "preparation not met"; ok = True; print(f"[{arm}_s{s}] preparation not met", flush=True); continue
                m, _, losses, stats = fit_router(c, mask, s); sc = tl.score(m, c["fp"], stats); p = rd / f"{arm}_s{s}.pt"
                torch.save({"model": m.state_dict(), "mu": torch.from_numpy(stats["mu"]), "sd": torch.from_numpy(stats["sd"]), "d": int(stats["mu"].size)}, p)
                fpos, fneg = mask & c["pos"], mask & ~c["pos"]
                rec = {"arm": arm, "seed": s, "status": "trained", "parameters": int(sum(x.numel() for x in m.parameters())), "fit": {k: stats[k] for k in ("n_fit", "n_pos", "n_neg", "wpos", "wneg")},
                       "loss_first": float(np.mean(losses[:WARM])), "loss_last": float(np.mean(losses[-WARM:])), "updates": len(losses),
                       "fit_balanced_accuracy_at_0.5": float(0.5 * ((sc[fpos] >= 0.5).mean() + (sc[fneg] < 0.5).mean())), "check_at_0.5": check_log(c, pr, ck, sc, 0.5), "sha256": sha256_file(p)}
                write_json(rd / f"{arm}_s{s}.json", rec); ok = True
                note = json.dumps({"loss_last": rec["loss_last"], "fit_bal": rec["fit_balanced_accuracy_at_0.5"], "check_bal": rec["check_at_0.5"]["balanced_accuracy"]})
                print(f"[{arm}_s{s}] {json.dumps(rec, default=str)}", flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train router {arm}_s{s} (CPU)", "cpu", started, time.time() - t0, ok, note)


def check(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"zone_selftest": lr.selftest()}
    try:
        P = pools(ctx); seqs = P["seqs"]; prep = json.loads((ctx.out / "preparation.json").read_text()); out["pool_rebuilt_equal"] = P["digest"] == prep["pool_digest"]
        fit_v, chk_v = ({r[0] for q in seqs if q["group"] == g for r in q["refs"]} for g in ("fit", "check"))
        out["session_disjoint"] = not (fit_v & chk_v) and not ({session_of(v) for v in fit_v} & {session_of(v) for v in chk_v})
        out["small_inside_large"] = bool(np.all(P["group"][P["small"]] == "fit"))
        s = 0; c = load_cand(ctx, s); pr = load_preds(ctx, s); head, thr, _ = ctx.a14.alt(s); bparams = ctx.a14.base_params(s)
        with np.load(ctx.out / "features" / f"windows_s{s}.npz") as z:
            wseq, wcen, wlg, wem = z["seq"], z["centre"], z["logits"], z["emb"]
            out["cache_identity"] = {"start_sha256": str(z["start_sha256"]), "path_digest_recorded": str(z["path_digest"]) == prep["seeds"][str(s)]["path_digest"]}
        is_nat = np.asarray([q["kind"] == "native" for q in seqs])
        sub = list(range(0, 60)) + [int(i) for i in np.flatnonzero(P["group"] == "check")[:60]] + [int(i) for i in np.flatnonzero((P["group"] == "fit") & is_nat)[:30]]
        ss = [seqs[i] for i in sub]; refs = rw.half_refs(ss); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(ss, at, 5)
        rows = np.concatenate([np.flatnonzero(wseq == i) for i in sub]); out["window_plan_equal"] = bool(np.array_equal(W["centre"], wcen[rows]))
        lg, em = wlg[rows], wem[rows]; probs = rr.per_sample(W, lg, ss); edges = rvs.edge_scores(head, W, lg, em, ss, torch.device("cpu"))
        base, alt = tl.systems(probs, edges, ss, bparams, thr)
        out["cache_reproduces_base_alt"] = all(np.array_equal(base[q["id"]], pr["base"][i]) and np.array_equal(alt[q["id"]], pr["alt"][i]) for i, q in zip(sub, ss))
        shuffled = [{**q, "labels": np.random.default_rng(7).permutation(q["labels"])} for q in ss]
        c1, c2 = to_arrays(candidates_p(ss, probs, base, alt, with_labels=False)), to_arrays(candidates_p(shuffled, probs, base, alt, with_labels=False))
        c3 = to_arrays(candidates_p(ss, probs, base, alt)); keep = np.isin(c["seq"], sub); local = {g: k for k, g in enumerate(sub)}
        out["isolation"] = bool(np.array_equal(c1["a"], c2["a"]) and np.array_equal(c1["b"], c2["b"]) and np.array_equal(c1["fp"], c2["fp"]))
        order = np.argsort([local[int(g)] for g in c["seq"][keep]], kind="stable")
        out["stored_candidates_reproduced"] = bool(np.array_equal(np.asarray([local[int(g)] for g in c["seq"][keep]])[order], c3["seq"]) and np.allclose(c["fp"][keep][order], c3["fp"])
                                                   and np.array_equal(c["pos"][keep][order], c3["pos"]))
        zz = [lr.zones(pr["base"][i], pr["alt"][i]) for i in range(len(seqs))]
        out["noop_equals_base"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [False] * len(zz[i])), pr["base"][i]) for i in range(len(seqs)))
        out["all_equals_alternative"] = all(np.array_equal(lr.apply(pr["base"][i], pr["alt"][i], zz[i], [True] * len(zz[i])), pr["alt"][i]) for i in range(len(seqs)))
        out["zones_match_candidates"] = int(sum(len(z) for z in zz)) == int(c["seq"].size)
        r1, r2 = tl.build("P", 52, s), tl.build("P", 52, s); out["identical_start_both_arms"] = all(torch.equal(a, b) for a, b in zip(r1.state_dict().values(), r2.state_dict().values()))
        opt = torch.optim.AdamW(r1.parameters()); out["optimiser_only_router"] = [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in r1.parameters()]
        out["parameters"] = int(sum(p.numel() for p in r1.parameters()))
        m6, _, l6, _ = fit_router(c, c["large_fit"], s, upto=6); m3, o3, _, _ = fit_router(c, c["large_fit"], s, upto=3)
        m33, _, l33, _ = fit_router(c, c["large_fit"], s, upto=6, state={"model": m3.state_dict(), "opt": o3.state_dict()}, start=3)
        out["resume"] = {"loss_equal": l6[3:] == l33, "params_equal": all(torch.equal(a, b) for a, b in zip(m6.state_dict().values(), m33.state_dict().values()))}
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cpu_spent": ledger_hours(ctx.ledger, "cpu"), "caps": {"gpu": ctx.vr["budget"]["gpu_hours_max"], "cpu": ctx.vr["budget"]["cpu_hours_max"]}}
        keys = ("pool_rebuilt_equal", "session_disjoint", "small_inside_large", "window_plan_equal", "cache_reproduces_base_alt", "isolation", "stored_candidates_reproduced", "noop_equals_base",
                "all_equals_alternative", "zones_match_candidates", "identical_start_both_arms", "optimiser_only_router")
        out["ok"] = bool(out["zone_selftest"]["ok"] and all(out[k] for k in keys) and out["resume"]["loss_equal"] and out["resume"]["params_equal"] and out["parameters"] == 1729
                         and out["cache_identity"]["path_digest_recorded"]); ok = out["ok"]
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true"); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--train", nargs="+", type=int)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.probe:
        return probe(ctx, a.workers)
    if a.prepare:
        return prepare(ctx, a.workers)
    if a.check:
        return check(ctx)
    train(ctx, a.train)


if __name__ == "__main__":
    main()
