#!/usr/bin/env python
"""Adapters of the frozen EV9V test benchmark: each turns one registered frozen system into per-item labels on the
benchmark's 10 Hz axis. Adapters receive items with (recording, frame, gamma) references and lengths only; reference
labels, joins, fragments and sources never reach them (placeholder zeros carry the length).

  cached_probs  historical per-item probabilities (cache_v2), argmax or B3 at the saved numeric setting
  a08_b3        frozen A08 on A08's request windows (16 samples, stride 5, target marker), nearest-centre probabilities,
                B3 at A08's committed numeric setting
  a14_router    the same A08 outputs, A13's frozen H edges and threshold, A14's frozen P router and its committed choice

Online A08 outputs are stored under a key of benchmark digest + weights + adapter + code + precision, per chunk of
items, so an interrupted run resumes and a mismatched store is refused.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_local_router as rlr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402

A14_CONFIG = "configs/experiments/video_local_router_v1.yaml"
A14_CHOICES = {"path": "docs/results/2026-10-21_local_router/choices.json", "sha256": "3302adc121e58d8d540879a63dcd3a93f57d5b058739c4baf9372e5f276873e2"}
ADAPTER_VERSION = "ev9v-benchmark-adapters-v1"
CHUNK = 150
RV_MIN = {"source": {"frame_manifest_columns": ["video_id", "frames_dir"]}}


def registry(cfg: dict) -> dict:
    reg = yaml.safe_load((ROOT / cfg["frozen_external_benchmark"]["registry"]["path"]).read_text())
    if reg["benchmark"] != cfg["frozen_external_benchmark"]["benchmark"]["id"]:
        raise SystemExit("the registry belongs to another benchmark")
    return reg["systems"]


def verify_weights(name: str, e: dict) -> dict:
    out = {}
    for k, w in e["weights"].items():
        if (got := sha256_file(ROOT / w["path"])) != w["sha256"]:
            raise SystemExit(f"{name}: {k} weights differ from the registry")
        out[k] = got
    return out


def code_digest(files) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def adapter_seqs(items: list[dict]) -> list[dict]:
    """What a model adapter may see: id, length (placeholder labels) and sample references."""
    return [{"id": x["id"], "kind": "item", "labels": np.zeros(x["n"], int), "refs": [tuple(r) for r in x["refs"]]} for x in items]


# ----------------------------------------------------------------------------------------------- historical caches
def cached_probs(items: list[dict], name: str, e: dict) -> dict:
    verify_weights(name, e); d = ROOT / e["cache"]; paths = load_paths()
    if (paths.cache_root / "FRAME_ORDER").read_text().strip() != "numeric-2026-09-30":
        raise SystemExit("cache_v2 frame order stamp differs")
    labels, probs, mt = {}, {}, []
    for x in items:
        f = d / f"{x['id']}.npz"
        with np.load(f) as z:
            p = z["prob"].astype(float)
        if p.shape != (x["n"], 5) or not np.isfinite(p).all():
            raise SystemExit(f"{name}: a cached item has the wrong shape or non-finite values")
        if "b3" in e["decoder"]:
            b = e["decoder"]["b3"]; lab = decode_with_prob("b3a", p, {"decode": {"b3a": {k: b[k] for k in ("half_window", "threshold", "nms_radius")}}})[0]
        else:
            lab = decode_with_prob("b0", p, {})[0]
        labels[x["id"]], probs[x["id"]] = lab.astype(int), p; mt.append(f.stat().st_mtime)
    return {"labels": labels, "probs": probs, "meta": {"cache": e["cache"], "files": len(items), "written_utc": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(mt))), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(mt)))]}}


# ----------------------------------------------------------------------------------------------- frozen A08 outputs
def a08_model(path: Path, device):
    m = vsh.make_model(torch.load(path, map_location="cpu", weights_only=True), device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def windows(seqs: list[dict]) -> tuple[dict, list, dict]:
    refs = sorted({r for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; return rr.half_windows(seqs, at, 5), refs, at


def a08_outputs(items: list[dict], bdigest: str, wspec: dict, store: Path, device, workers: int, acc: dict) -> dict:
    """Logits and embeddings of every request window of every item, by chunks of CHUNK items (resumable)."""
    seqs = adapter_seqs(items); W, _, _ = windows(seqs)
    key = hashlib.sha256(json.dumps({"benchmark": bdigest, "weights": wspec["sha256"], "adapter": ADAPTER_VERSION, "precision": "fp32", "windows": "16 samples, stride 5, marker, nearest centre",
                                     "code": code_digest([Path(__file__), Path(rr.__file__), Path(rsh.__file__), Path(vsh.__file__)])}, sort_keys=True).encode()).hexdigest()
    sdir = store / f"a08_{wspec['sha256'][:12]}_{key[:12]}"; sdir.mkdir(parents=True, exist_ok=True); model, d0 = None, None; lg, em, sq, ce = [], [], [], []
    fp = FramePaths(prv.frame_dirs(load_paths(), RV_MIN, {r[0] for q in seqs for r in q["refs"]}))
    for c0 in range(0, len(seqs), CHUNK):
        chunk = seqs[c0:c0 + CHUNK]; f = sdir / f"chunk_{c0 // CHUNK:03d}.npz"; Wc, refs, _ = windows(chunk)
        if f.is_file():
            with np.load(f) as z:
                ok = str(z["key"]) == key and z["ids"].tolist() == [q["id"] for q in chunk] and np.array_equal(z["seq"], Wc["seq"]) and np.array_equal(z["centre"], Wc["centre"])
                if not ok:
                    raise SystemExit(f"{f}: a stored chunk does not match its key or windows")
                lg.append(z["logits"]); em.append(z["emb"])
        else:
            if model is None:
                model = a08_model(ROOT / wspec["path"], device); d0 = tet.state_digest(model)
            t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - t0
            orders = np.broadcast_to(np.arange(16, dtype=np.int64), Wc["s"].shape).copy(); a, e, sec = rsh.infer(model, Wc, orders, frames, device)
            acc["gpu_seconds"] += sec; acc["windows"] += len(Wc["seq"]); del frames
            np.savez(f, key=key, ids=np.asarray([q["id"] for q in chunk]), seq=Wc["seq"], centre=Wc["centre"], logits=a.astype(np.float32), emb=e.astype(np.float32), seconds=sec)
            lg.append(a.astype(np.float32)); em.append(e.astype(np.float32)); print(f"  {sdir.name} chunk {c0 // CHUNK + 1}: {len(Wc['seq'])} windows, GPU {sec:.0f} s", flush=True)
        sq.append(Wc["seq"] + c0); ce.append(Wc["centre"])
    if not (np.array_equal(np.concatenate(sq), W["seq"]) and np.array_equal(np.concatenate(ce), W["centre"])):
        raise SystemExit("the chunked windows differ from the item window plan")
    unchanged = None if model is None else tet.state_digest(model) == d0
    if unchanged is False:
        raise SystemExit("A08's state changed during inference")
    return {"W": W, "seqs": seqs, "logits": np.concatenate(lg), "emb": np.concatenate(em), "key": key, "store": str(sdir.relative_to(ROOT)), "state_unchanged": unchanged}


def a08_b3_labels(out: dict, params: dict) -> dict:
    probs = rr.per_sample(out["W"], out["logits"], out["seqs"]); return {k: v.astype(int) for k, v in decode(params, probs).items()}


# ----------------------------------------------------------------------------------------------- A14 router
class A14:
    def __init__(self) -> None:
        self.ctx = tl.Ctx(load_cfg(SimpleNamespace(config=A14_CONFIG, set=[])), A14_CONFIG)
        if sha256_file(ROOT / A14_CHOICES["path"]) != A14_CHOICES["sha256"]:
            raise SystemExit("A14's committed choices differ from their pin")
        self.choices = json.loads((ROOT / A14_CHOICES["path"]).read_text())["choices"]

    def labels(self, out: dict, seed: int, e: dict) -> dict:
        W, seqs, lg, em = out["W"], out["seqs"], out["logits"], out["emb"]; head, thr, hsha = self.ctx.alt(seed); m, stats, rsha = rlr.router(self.ctx, "P", seed)
        if hsha != e["weights"]["a13_head"]["sha256"] or rsha != e["weights"]["router"]["sha256"]:
            raise SystemExit("A13 head or A14 router differs from the registry")
        hd0, rd0 = tet.state_digest(head), tet.state_digest(m)
        probs = rr.per_sample(W, lg, seqs); embs = tl.per_sample_emb(W, em, seqs); edges = rvs.edge_scores(head, W, lg, em, seqs, torch.device("cpu"))
        base, alt = tl.systems(probs, edges, seqs, self.ctx.base_params(seed), thr); cand = tl.candidates(seqs, probs, embs, base, alt, with_labels=False)
        sc = tl.score(m, cand["fp"], stats); t = self.choices[f"P_s{seed}"]["threshold"]; accept = np.zeros(sc.size, bool) if t == "noop" else sc >= float(t)
        P = {"seqs": seqs, "seed": {seed: {"base": base, "alt": alt, "cand": cand}}}; routed = rlr.routed(P, seed, accept)
        noop, allin = rlr.routed(P, seed, np.zeros(sc.size, bool)), rlr.routed(P, seed, np.ones(sc.size, bool))
        diag = {"zones": int(sc.size), "accepted": int(accept.sum()), "threshold": t, "head_threshold": thr,
                "noop_equals_base": all(np.array_equal(noop[k], base[k]) for k in base), "accept_all_equals_alternative": all(np.array_equal(allin[k], alt[k]) for k in alt),
                "states_unchanged": tet.state_digest(head) == hd0 and tet.state_digest(m) == rd0}
        if not (diag["noop_equals_base"] and diag["accept_all_equals_alternative"] and diag["states_unchanged"]):
            raise SystemExit(f"A14 router checks failed: {diag}")
        cast = lambda d: {k: np.asarray(v, int) for k, v in d.items()}  # noqa: E731
        return {"labels": cast(routed), "base": cast(base), "alternative": cast(alt), "diag": diag}
