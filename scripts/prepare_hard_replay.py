#!/usr/bin/env python
"""A20 preparation (configs/experiments/video_hard_replay_v1.yaml), CPU only: 4-window replay candidates from the 1,500
multi streams of A15's large-fit pool, each seed's A08 difficulty, the paired U/H replay lists, and the 16-window training
manifests (A10's 12 pinned windows + 4 replay windows) with pixel digests and cached own-teacher logits.

  --prepare S [S ...]   seeds to prepare (seed 1's lists are used only after a met promotion rule)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A10_CONFIG, A15_CONFIG, B3C_CONFIG = "configs/experiments/video_switch_head_v1.yaml", "configs/experiments/video_router_data_v1.yaml", "configs/experiments/video_b3_coverage_v1.yaml"
U_TOTAL, BASE, REPLAY, SHORT = 1350, 12, 4, 16
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vh = cfg, config, cfg["video_hard_replay"]; p = self.vh["pinned"]
        for k in ("A10_config", "A10_manifest_index", "A08_choices", "A15_pool_manifest"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG)
        self.bc = bc.Ctx(load_cfg(SimpleNamespace(config=B3C_CONFIG, set=[])), B3C_CONFIG)
        o = self.vh["outputs"]; self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.vh["budget"]["ledger"]


def run_lengths(labels: np.ndarray) -> np.ndarray:
    """Per sample, the length of its label run over the whole stream (neighbouring same-label fragments form one run)."""
    lab = np.asarray(labels); out = np.empty(lab.size, np.int64); i = 0
    while i < lab.size:
        j = i
        while j < lab.size and lab[j] == lab[i]:
            j += 1
        out[i:j] = j - i; i = j
    return out


def candidates(P: dict, W: dict) -> dict:
    """Every 4 adjacent windows of each large-fit multi stream: rows, mixed/short flags, target tuple."""
    seqs = P["seqs"]; streams = [i for i in range(len(seqs)) if P["group"][i] == "fit" and seqs[i]["kind"] == "multi"]
    order = np.argsort(W["seq"], kind="stable"); bounds = np.searchsorted(W["seq"][order], np.arange(len(seqs) + 1)); rows_all, info = [], []
    for i in streams:
        rows = order[bounds[i]:bounds[i + 1]]; lab = np.asarray(seqs[i]["labels"]); rl = run_lengths(lab)
        for k in range(len(rows) - 3):
            r = rows[k:k + 4]; c = W["centre"][r]
            mixed = bool(any(len(set(np.asarray(W["labels16"][x]).tolist())) >= 2 for x in r)); short = bool(any(rl[int(cc)] <= SHORT for cc in c))
            rows_all.append(r); info.append((i, k, mixed, short, tuple(int(lab[int(cc)]) for cc in c)))
    R = np.asarray(rows_all, np.int64)
    return {"rows": R, "stream": np.asarray([x[0] for x in info]), "k": np.asarray([x[1] for x in info]), "mixed": np.asarray([x[2] for x in info]),
            "short": np.asarray([x[3] for x in info]), "tuple": [x[4] for x in info], "streams": streams}


def difficulty(C: dict, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    lp = torch.log_softmax(torch.from_numpy(logits), -1).numpy(); ce = -lp[np.arange(len(target)), target]
    return ce[C["rows"]].mean(1)


def sample(C: dict, ce: np.ndarray, seed: int) -> dict:
    """The paired U/H lists: per update a stream, U's group and H's pick from one fixed numpy stream."""
    rng = np.random.default_rng([20261021, 20, seed]); cand = np.flatnonzero(C["mixed"]); bucket = {b: {} for b in (True, False)}
    for g in cand:
        bucket[bool(C["short"][g])].setdefault(int(C["stream"][g]), []).append(int(g))
    for b in (True, False):
        if not bucket[b]:
            raise SystemExit(f"the {'short' if b else 'non-short'} bucket is empty: preparation not met")
    streams = {b: sorted(bucket[b]) for b in (True, False)}; U, H, n_strat, n_top = [], [], [], []
    for u in range(U_TOTAL):
        b = (u % 2 == 0); a, x, y = rng.random(3); s = streams[b][min(int(a * len(streams[b])), len(streams[b]) - 1)]
        cs = sorted(bucket[b][s], key=lambda g: int(C["k"][g])); gu = cs[min(int(x * len(cs)), len(cs) - 1)]
        strat = [g for g in cs if C["tuple"][g] == C["tuple"][gu]]; strat = sorted(strat, key=lambda g: (-float(ce[g]), int(C["k"][g])))
        top = strat[:max(1, math.ceil(len(strat) / 4))]; gh = top[min(int(y * len(top)), len(top) - 1)] if len(strat) > 1 else gu
        U.append(gu); H.append(gh); n_strat.append(len(strat)); n_top.append(len(top))
    return {"U": np.asarray(U), "H": np.asarray(H), "n_stratum": np.asarray(n_strat), "n_top": np.asarray(n_top), "streams_per_bucket": {"short": len(streams[True]), "non_short": len(streams[False])}}


class Digests(Dataset):
    def __init__(self, m: dict, fp, updates: list[int]) -> None:
        self.m, self.fp, self.updates = m, fp, updates

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; return u, rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(BASE, BASE + REPLAY)])[3]


def manifest(ctx: Ctx, P: dict, W: dict, C: dict, groups: np.ndarray, seed: int, teacher_pool: np.ndarray, videos: list, vidx: dict) -> dict:
    a10, msha = ctx.a10.manifest(seed); seqs = P["seqs"]; m = {}
    for k in ("vid", "frame", "gamma", "order_ordered", "marker", "target", "digest_ordered"):
        src = a10[k][:U_TOTAL]; shape = (U_TOTAL, BASE + REPLAY) + src.shape[2:]; m[k] = np.zeros(shape, src.dtype); m[k][:, :BASE] = src
    remap = np.asarray([vidx[str(v)] for v in a10["videos"]], np.int32); m["vid"][:, :BASE] = remap[a10["vid"][:U_TOTAL]]
    rows = C["rows"][groups]                                                      # (U, 4) pool window rows
    for u in range(U_TOTAL):
        for w in range(REPLAY):
            r = int(rows[u, w]); q = seqs[int(W["seq"][r])]; s = W["s"][r]
            refs = [q["refs"][int(j)] for j in s]; m["vid"][u, BASE + w] = [vidx[str(x[0])] for x in refs]; m["frame"][u, BASE + w] = [int(x[1]) for x in refs]
            m["gamma"][u, BASE + w] = [float(x[2]) for x in refs]; m["order_ordered"][u, BASE + w] = np.arange(16); m["marker"][u, BASE + w] = int(W["marker"][r])
            m["target"][u, BASE + w] = int(q["labels"][int(W["centre"][r])])
    m["videos"] = np.asarray(videos); m["pool_rows"] = rows.astype(np.int64)
    cache = np.load(ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz")
    if sha256_file(ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz") != ctx.vh["pinned"]["A18_teacher_cache"][seed] or str(cache["manifest_sha256"]) != msha:
        raise SystemExit(f"seed {seed}: A18's teacher cache differs from its pin or from A10's manifest")
    if not np.array_equal(cache["digests"][:U_TOTAL], a10["digest_ordered"][:U_TOTAL]):
        raise SystemExit(f"seed {seed}: A18's cached requests differ from A10's base windows")
    m["teacher"] = np.concatenate([cache["logits"][seed, :U_TOTAL], teacher_pool[rows]], 1).astype(np.float32)
    return m


def prepare(ctx: Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, dsec = time.time(), now(), False, "", 0.0
    try:
        P = bc.pool(ctx.bc); seqs = P["seqs"]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; Wf = rr.half_windows(seqs, at, 5)
        W = {"seq": Wf["seq"], "centre": Wf["centre"], "s": Wf["s"], "marker": Wf["marker"], "labels16": Wf["labels16"]}
        if not (np.array_equal(W["seq"], P["W"]["seq"]) and np.array_equal(W["centre"], P["W"]["centre"])):
            raise SystemExit("the rebuilt window plan differs from the pool's")
        C = candidates(P, W); target = np.asarray([int(seqs[int(W["seq"][r])]["labels"][int(W["centre"][r])]) for r in range(len(W["seq"]))])
        vids_pool = sorted({str(r[0]) for i in C["streams"] for r in seqs[i]["refs"]}); allow = set(prv.allowlist(ctx.a10.rv, ROOT))
        if set(vids_pool) - allow or any(P["group"][i] != "fit" for i in C["streams"]):
            raise SystemExit("a replay recording is outside the training allowlist or outside large-fit")
        a10v = [str(v) for s in (0, 1, 2) for v in ctx.a10.manifest(s)[0]["videos"].tolist()]; videos = list(dict.fromkeys(a10v + vids_pool)); vidx = {v: i for i, v in enumerate(videos)}
        fp = FramePaths(prv.frame_dirs(ctx.a10.paths, ctx.a10.rv, set(videos))); md = ctx.out / "manifests"; md.mkdir(parents=True, exist_ok=True)
        rec = {"config_sha256": sha256_file(ROOT / ctx.config), "pool_digest": P["digest"], "candidates": {"groups": int(len(C["rows"])), "mixed": int(C["mixed"].sum()),
               "mixed_short": int((C["mixed"] & C["short"]).sum()), "mixed_non_short": int((C["mixed"] & ~C["short"]).sum()), "streams": len(C["streams"])}, "seeds": {}, "index": {}}
        for s in seeds:
            pl = bc.pool_logits(ctx.bc, P, f"A08_s{s}"); ce = difficulty(C, pl, target); S = sample(C, ce, s); rs = {}
            wrong = (pl.argmax(1) != target)
            for arm in ("U", "H"):
                g = S[arm]; m = manifest(ctx, P, W, C, g, s, pl, videos, vidx); a = time.time(); dg = np.zeros((U_TOTAL, REPLAY, 32), np.uint8)
                for u, d in DataLoader(Digests(m, fp, list(range(U_TOTAL))), batch_size=None, num_workers=workers):
                    dg[int(u)] = d.numpy()
                dsec += time.time() - a; m["digest_ordered"][:, BASE:] = dg; path = md / f"{arm}_s{s}.npz"; np.savez(path, **m)
                rows = C["rows"][g]; recs = {seqs[int(W["seq"][r])]["refs"][int(j)][0] for r in rows.ravel() for j in W["s"][r]}
                streams_used = C["stream"][g]; uniq, cnt = np.unique(streams_used, return_counts=True)
                rs[arm] = {"mean_A08_CE": float(ce[g].mean()), "median_A08_CE": float(np.median(ce[g])), "A08_wrong_rate_replay": float(wrong[rows].mean()),
                           "short_share": float(C["short"][g].mean()), "mixed_share": float(C["mixed"][g].mean()), "unique_requests": int(len(np.unique(rows))),
                           "unique_groups": int(len(np.unique(g))), "source_recordings": len(recs), "source_sessions": len({session_of(v) for v in recs}), "streams_used": int(len(uniq)),
                           "max_stream_reuse": int(cnt.max()), "tuple_classes": {str(k): int(v) for k, v in zip(*np.unique([str(C["tuple"][x]) for x in g], return_counts=True))}}
                rec["index"][f"{arm}_s{s}"] = {"file": path.name, "sha256": sha256_file(path)}
            rs["H_equals_U"] = float((S["H"] == S["U"]).mean()); rs["n_stratum_1"] = float((S["n_stratum"] == 1).mean()); rs["n_stratum_median"] = float(np.median(S["n_stratum"]))
            rs["streams_per_bucket"] = S["streams_per_bucket"]; rs["A08_CE_all_mixed_candidates"] = {"mean": float(ce[C["mixed"]].mean()), "short": float(ce[C["mixed"] & C["short"]].mean()),
                                                                                                   "non_short": float(ce[C["mixed"] & ~C["short"]].mean())}
            rec["seeds"][str(s)] = rs; print(f"seed {s}: {json.dumps(rs)[:900]}", flush=True)
        rec["decode_seconds"], rec["finished"] = dsec, now(); ids = set(vids_pool) | {session_of(v) for v in vids_pool}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        write_json(md / "index.json", rec["index"]); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; ledger_append(ctx.ledger, f"prepare replay lists and manifests (CPU incl. {workers} decode workers)", "cpu", started, total + dsec * (workers - 1), ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--prepare", nargs="+", type=int, required=True)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); prepare(ctx, a.prepare, a.workers)


if __name__ == "__main__":
    main()
