#!/usr/bin/env python
"""Prepare (pool, mining, manifests, weights), check, train and queue A11 (configs/experiments/video_sustained_switch_v1.yaml).

Every run strictly loads its seed's frozen A08 all_s{k}; the same weights in eval mode are the teacher. Update u reads row u
of this round's manifest: A10's rows (A07's native and change groups, A10's same-label join group) with the native group of
odd rows replaced by a hard-negative pool group. One central-edge head per run reads the 4 windows of each group. U and H
share everything except the negative weights of the BCE. Only training data is read.

  --prepare        pool, frozen A10 V scores (GPU), manifests, central-edge labels, U/H weights, digests, counts
  --check          GPU: lambda rule, pairing, edges, weights, gradients, teacher, optimiser, resume, timing, projection
  --queue K [...]  runs in order (U_s0, H_s0, ...)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = "video_sustained_switch_v1"
GROUP, NG, WINDOWS = 4, 3, 12
CENTRAL = np.array([1, 4, 7])                    # the central edge of each group in A10's 9-edge order
ARMS = ("U", "H")
SOURCES = ("random_native", "hard_pool", "change_group", "same_label_join")
LADDER = vsh.LADDER
D_IN, HIDDEN, EPS = 30 + 2 * tm.EMBED_DIM + 4, 32, 1e-6
TRAINING_KEYS = ("predecessors", "source", "arms", "model", "data", "loss", "training")
now, write_json, ledger_hours, ledger_append, determinism, make_model, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.make_model, vsh.max_diff


# ----------------------------------------------------------------------------------------------- head and features
class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(D_IN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_head(seed: int) -> Head:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3000 + int(seed)); return Head()


def central_features(z: torch.Tensor, e: torch.Tensor, idx4) -> torch.Tensor:
    """idx4: (G, 4) window indices w0..w3; the central edge is w1 | w2."""
    idx4 = torch.as_tensor(idx4, device=z.device, dtype=torch.long); p = torch.softmax(z, -1)[idx4]; u = (e / (e.norm(dim=-1, keepdim=True) + EPS))[idx4]
    p1, p2, pl, pr = p[:, 1], p[:, 2], (p[:, 0] + p[:, 1]) / 2, (p[:, 2] + p[:, 3]) / 2
    ml, mr = (u[:, 0] + u[:, 1]) / 2, (u[:, 2] + u[:, 3]) / 2; dl, ds = u[:, 2] - u[:, 1], mr - ml
    cos = lambda a, b: (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + EPS)  # noqa: E731
    scal = torch.stack([dl.norm(dim=-1), ds.norm(dim=-1), 1 - cos(u[:, 1], u[:, 2]), 1 - cos(ml, mr)], -1)
    return torch.cat([p1, p2, (p2 - p1).abs(), pl, pr, (pr - pl).abs(), dl, ds, scal], -1)


TRAIN_IDX = np.arange(WINDOWS).reshape(NG, GROUP)


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vs = cfg, config, cfg["video_sustained_switch"]; self.src = self.vs["source"]; p10 = self.src["a10_config"]; vsh.pinned(p10)
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=p10["path"], set=[])), p10["path"]); self.rv, self.paths = self.a10.rv, self.a10.paths
        tr = self.vs["training"]; self.seeds = [int(s) for s in tr["seeds"]]; self.total, self.warmup, self.o = int(tr["total_updates"]), int(tr["warmup_updates"]), tr["optimiser"]
        o = self.vs["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.mdir = self.out / "manifests"; self.ledger = ROOT / self.vs["budget"]["ledger"]; self._m, self._fp = {}, None
        if self.total != self.a10.total or self.warmup != self.a10.warmup or self.o != self.a10.o:
            raise SystemExit("the schedule or optimiser differs from A10's")

    def manifest(self, seed: int) -> tuple[dict, str]:
        if seed not in self._m:
            e = json.loads((self.mdir / "index.json").read_text())["seeds"][str(seed)]; path = self.mdir / e["file"]
            if (got := sha256_file(path)) != e["sha256"]:
                raise SystemExit(f"{path.name}: sha256 differs from index.json")
            with np.load(path) as z:
                m = {k: z[k] for k in z.files}
            if vpr.arrays_digest(m) != e["arrays_digest"]:
                raise SystemExit(f"{path.name}: arrays differ from index.json")
            self._m[seed] = (m, got)
        return self._m[seed]

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            vids = {str(v) for s in (0, 1, 2) for v in self.manifest(s)[0]["videos"].tolist()}
            if vids - set(prv.allowlist(self.rv, ROOT)):
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(prv.frame_dirs(self.paths, self.rv, vids))
        return self._fp

    def lam(self) -> float:
        rec = json.loads((self.out / "lambda.json").read_text())
        if rec["config_sha256"] != sha256_file(ROOT / self.config):
            raise SystemExit("lambda.json was fixed under another configuration")
        return float(rec["lambda"])


# ----------------------------------------------------------------------------------------------- pool and scoring
def pool(seqs: list[dict], info: list[dict]) -> list[tuple[int, int, str]]:
    """2,700 groups without a label change: 1,350 native, 1,350 same-label stream intervals; at most 4 per session."""
    nat, strm = [], []
    for i, (q, f) in enumerate(zip(seqs, info)):
        c, lab = np.asarray(f["windows"]["centres"]), np.asarray(q["labels"])
        for k in range(len(c) - GROUP + 1):
            if q["kind"] == "native":
                nat.append((i, k))
            elif vsh.changes_in(lab, int(c[k]), int(c[k + GROUP - 1])) == 0:
                strm.append((i, k))
    rng = np.random.default_rng([20261020, 0]); out, per = [], Counter()
    for kind, cand in (("native", nat), ("stream", strm)):
        took = 0
        for j in rng.permutation(len(cand)):
            i, k = cand[j]; q = seqs[i]; c1 = int(info[i]["windows"]["centres"][k + 1]); sess = session_of(q["id"] if q["kind"] == "native" else q["refs"][c1][0])
            if per[sess] >= 4:
                continue
            per[sess] += 1; out.append((i, k, kind)); took += 1
            if took == 1350:
                break
    return out


def pair_arrays(seqs, info, pairs: list[tuple[int, int, int]]) -> dict:
    """A manifest-like dict holding the two central windows of each listed (sequence, centre index a, centre index b)."""
    wins = [[(qi, ca), (qi, cb)] for qi, ca, cb in pairs]; U = len(wins); refs_of = lambda qi, ci: [seqs[qi]["refs"][int(x)] for x in info[qi]["windows"]["samples"][int(ci)]]  # noqa: E731
    videos = sorted({r[0] for w in wins for qi, ci in w for r in refs_of(qi, ci)}); vix = {v: i for i, v in enumerate(videos)}
    m = {"vid": np.empty((U, 2, 16), np.int32), "frame": np.empty((U, 2, 16), np.int32), "gamma": np.empty((U, 2, 16), np.float32), "marker": np.empty((U, 2), np.int8),
         "target": np.zeros((U, 2), np.int8)}
    for u, w in enumerate(wins):
        for j, (qi, ci) in enumerate(w):
            refs = refs_of(qi, ci); m["vid"][u, j] = [vix[r[0]] for r in refs]; m["frame"][u, j] = [int(r[1]) for r in refs]; m["gamma"][u, j] = [float(r[2]) for r in refs]
            m["marker"][u, j] = int(info[qi]["windows"]["markers"][int(ci)])
    m["order_ordered"] = np.broadcast_to(np.arange(16, dtype=np.int8), (U, 2, 16)).copy(); m["videos"] = np.asarray(videos)
    return m


@torch.no_grad()
def score_pairs(models: list, m: dict, fp, device, workers: int, batch: int = 32) -> np.ndarray:
    """Mean sigmoid score of the frozen A10 V heads on each pair's edge (A10's V feature definition)."""
    U = m["vid"].shape[0]; out = np.zeros(U); order = m["order_ordered"]
    for b0 in range(0, U, batch):
        us = list(range(b0, min(U, b0 + batch))); u8, slots, _, _ = rvc.window_inputs(m, order, fp, [(u, j) for u in us for j in (0, 1)])
        rgb, mk = tm.to_rgb(u8.to(device)), tm.marker_planes(slots.to(device)); a = np.arange(0, 2 * len(us), 2); b = a + 1; acc = np.zeros(len(us))
        for model, heads in models:
            e = model.embed(rgb, mk); z = model.classifier(e); acc += torch.sigmoid(heads["V"](vsh.edge_features(z, e, a, b, "V"))).cpu().numpy()
        out[b0:b0 + len(us)] = acc / len(models)
    return out


def a10_v_models(ctx: Ctx, device) -> list:
    out = []
    for key in ("V_s0", "V_s2"):
        man = json.loads((ctx.a10.out / "train" / key / "run_manifest.json").read_text()); w = ROOT / "models_frozen" / vsh.NS / key / "final.pt"
        if sha256_file(w) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen weights differ from the run record")
        st_ = torch.load(w, map_location="cpu", weights_only=True); model = make_model(st_["model"], device).eval(); heads = vsh.build_heads(int(man["seed"])).to(device)
        heads.load_state_dict(st_["heads"]); heads.eval(); out.append((model, heads))
    return out


def weights(edge: np.ndarray, source: np.ndarray, score: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """U: balanced; H: the same positives, negatives with raw 3 in each source's top quartile of frozen scores, 1 otherwise,
    rescaled to U's negative mass."""
    pos, neg = edge == 1, edge == 0; N = int(pos.sum() + neg.sum()); wp, wn = N / (2 * max(1, int(pos.sum()))), N / (2 * max(1, int(neg.sum())))
    wU = np.where(pos, wp, np.where(neg, wn, 0.0)); raw = np.zeros_like(wU); q = {}
    for k in range(len(SOURCES)):
        sel = neg & (source == k)
        if sel.any():
            q[SOURCES[k]] = float(np.quantile(score[sel], 0.75)); raw[sel] = np.where(score[sel] >= q[SOURCES[k]], 3.0, 1.0)
    c = float(neg.sum()) / float(raw[neg].sum()); wH = np.where(pos, wp, np.where(neg, wn * c * raw, 0.0))
    return {"w_pos": wp, "w_neg_U": wn, "quartile_thresholds": q, "rescale": c, "mass_neg_U": float(wU[neg].sum()), "mass_neg_H": float(wH[neg].sum()),
            "mass_pos_U": float(wU[pos].sum()), "mass_pos_H": float(wH[pos].sum())}, wU, wH


def prepare(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0
    try:
        rec = {"config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config), "started": started}
        data = vpr.training_data(ctx.a10.a07p); seqs = data.seqs; info = vpr.sequence_info(seqs); P = pool(seqs, info)
        rec["pool"] = {"groups": len(P), "native": sum(k == "native" for *_, k in P), "stream": sum(k == "stream" for *_, k in P),
                       "sessions": len({session_of(seqs[i]["id"] if seqs[i]["kind"] == "native" else seqs[i]["refs"][int(info[i]["windows"]["centres"][k + 1])][0]) for i, k, _ in P})}
        a10m = {s: ctx.a10.manifest(s)[0] for s in (0, 1, 2)}
        for s, m in a10m.items():
            if ctx.a10.manifest(s)[1] != json.loads((ctx.a10.mdir / "index.json").read_text())["seeds"][str(s)]["sha256"]:
                raise SystemExit("an A10 manifest differs from its index")
        # pairs to score: every pool group's central pair and every central pair of A10's rows 0-1,349 (all seeds)
        pool_pairs = [(i, k + 1, k + 2) for i, k, _ in P]
        man_pairs = {s: [(int(m["seq"][u, 4 * g + 1]), int(m["centre_index"][u, 4 * g + 1]), int(m["centre_index"][u, 4 * g + 2])) for u in range(ctx.total) for g in range(NG)] for s, m in a10m.items()}
        allp = sorted(set(pool_pairs) | {p for v in man_pairs.values() for p in v}); pm = pair_arrays(seqs, info, allp)
        if set(map(str, pm["videos"].tolist())) - set(prv.allowlist(ctx.rv, ROOT)):
            raise SystemExit("a scored recording lies outside the training allowlist")
        device = torch.device("cuda"); determinism(); fp = FramePaths(prv.frame_dirs(ctx.paths, ctx.rv, set(map(str, pm["videos"].tolist()))))
        t1 = time.time(); sc = score_pairs(a10_v_models(ctx, device), pm, fp, device, workers); gsec = time.time() - t1; smap = dict(zip(allp, sc.tolist()))
        rec["scoring"] = {"pairs": len(allp), "gpu_seconds": gsec}
        hard = sorted(range(len(P)), key=lambda j: -smap[pool_pairs[j]])[:1350]; rec["hard_set"] = {"groups": len(hard), "score_min": float(min(smap[pool_pairs[j]] for j in hard)),
                                                                                                 "native": sum(P[j][2] == "native" for j in hard)}
        ctx.mdir.mkdir(parents=True, exist_ok=True); index = {"config": ctx.config, "seeds": {}}; rec["seeds"] = {}
        for s, m10 in a10m.items():
            rng = np.random.default_rng([20261020, 1, s]); pick = [hard[j] for j in rng.choice(len(hard), size=ctx.total // 2 + ctx.total % 2, replace=False)]
            wins, src, k = [], np.zeros((ctx.total, NG), np.int8), 0
            for u in range(ctx.total):
                ws = [(int(m10["seq"][u, w]), int(m10["centre_index"][u, w])) for w in range(WINDOWS)]
                if u % 2 == 1:
                    i, kk, _ = P[pick[k]]; k += 1; ws[:GROUP] = [(i, kk + t) for t in range(GROUP)]; src[u, 0] = 1
                src[u, 1], src[u, 2] = 2, 3; wins.append(ws)
            m = vsh.build_manifest(seqs, info, wins, m10)
            keep = np.ones((ctx.total, WINDOWS), bool); keep[1::2, :GROUP] = False
            eq = np.array_equal(m["videos"][m["vid"]][keep], m10["videos"][m10["vid"][:ctx.total]][keep])
            for kk in ("frame", "gamma", "c", "target", "marker", "seq", "centre_index"):
                eq = eq and np.array_equal(m[kk][keep], m10[kk][:ctx.total][keep])
            if not eq:
                raise SystemExit(f"seed {s}: the kept windows differ from A10's rows")
            refs = {(str(m["videos"][v]), int(f), round(float(g), 6)) for v, f, g in zip(m["vid"][1::2, :GROUP].ravel(), m["frame"][1::2, :GROUP].ravel(), m["gamma"][1::2, :GROUP].ravel())}
            hashes = prv.pixel_hashes(refs, prv.frame_dirs(ctx.paths, ctx.rv, {r[0] for r in refs}), workers)
            dg = m10["digest_ordered"][:ctx.total].copy(); new = td.window_digests({**m, "vid": m["vid"][1::2], "frame": m["frame"][1::2], "gamma": m["gamma"][1::2],
                                                                                     "order_ordered": m["order_ordered"][1::2]}, hashes, range(GROUP))
            dg[1::2, :GROUP] = new; m["digest_ordered"] = dg
            edge = m["edge"][:, CENTRAL].copy()
            score = np.array([[smap[(int(m["seq"][u, 4 * g + 1]), int(m["centre_index"][u, 4 * g + 1]), int(m["centre_index"][u, 4 * g + 2]))] for g in range(NG)] for u in range(ctx.total)])
            info_w, wU, wH = weights(edge, src, score); m.update({"c_edge": edge, "c_join": m["edge_join"][:, CENTRAL].copy(), "c_source": src, "c_score": score, "w_U": wU, "w_H": wH})
            path = ctx.mdir / f"VSS_s{s}.npz"
            if path.is_file():
                raise SystemExit(f"{path.name} exists; manifests are never overwritten")
            np.savez(path, **m); sha = sha256_file(path); index["seeds"][str(s)] = {"file": path.name, "sha256": sha, "arrays_digest": vpr.arrays_digest(m)}
            cnt = {SOURCES[k]: {"pos": int(((edge == 1) & (src == k)).sum()), "neg": int(((edge == 0) & (src == k)).sum()), "masked": int(((edge == -1) & (src == k)).sum())} for k in range(len(SOURCES))}
            rec["seeds"][str(s)] = {"sha256": sha, "central_edges": cnt, "weights": info_w, "hard_groups_used": len(pick)}
        write_json(ctx.mdir / "index.json", index)
        ids = {str(v) for s in (0, 1, 2) for v in np.load(ctx.mdir / f"VSS_s{s}.npz")["videos"].tolist()}; ids |= {session_of(v) for v in ids}
        rec["finished"] = now(); rec["seconds"] = time.time() - t0; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps({k: rec[k] for k in ("pool", "scoring", "hard_set")}), json.dumps(rec["seeds"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare (pool scoring on GPU)", "gpu", started, gsec, ok, note)
        ledger_append(ctx.ledger, "prepare (CPU)", "cpu", started, time.time() - t0 - gsec, ok, note)


# ----------------------------------------------------------------------------------------------- training
def make_optimizer(model, head, o: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW([{"params": model.encoder_parameters(), "lr": 0.0}, {"params": model.new_parameters(), "lr": 0.0}, {"params": list(head.parameters()), "lr": 0.0}],
                             betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))


def bound(opt, model, head) -> bool:
    ids = lambda ps: [id(p) for p in ps]  # noqa: E731
    return (len(opt.param_groups) == 3 and ids(opt.param_groups[0]["params"]) == ids(model.encoder_parameters())
            and ids(opt.param_groups[1]["params"]) == ids(model.new_parameters()) and ids(opt.param_groups[2]["params"]) == ids(head.parameters()))


def models(ctx: Ctx, seed: int, device):
    state, ssha = ctx.a10.start_state(seed); student = make_model(state, device); teacher = make_model(state, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher, build_head(seed).to(device), ssha


def train_step(model, teacher, head, opt, batch, y: np.ndarray, w: np.ndarray, lam: float, lrs, o: dict, device, keep: bool = False) -> dict:
    u8, slots, target = batch[1], batch[2], batch[3]; t = target.to(device); model.train(); head.train()
    rgb, mk = tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)); e = model.embed(rgb, mk); z = model.classifier(e)
    with torch.no_grad():
        zt = teacher(rgb, mk)
    ce = F.cross_entropy(z, t); lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean()
    s = head(central_features(z, e, TRAIN_IDX)); yy = torch.as_tensor(y, device=device).float(); ww = torch.as_tensor(w, device=device).float(); km = yy >= 0
    bce = (ww[km] * F.binary_cross_entropy_with_logits(s[km], yy[km], reduction="none")).mean() if bool(km.any()) else s.sum() * 0.0
    loss = ce + kl + float(lam) * bce
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); pred = z.detach().argmax(1).cpu().numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "bce": float(bce.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
           "lr_head": lrs[2], "correct": int((pred == target.numpy()).sum()), "edges_used": int(km.sum()),
           "edge_acc": float(((s.detach()[km] > 0).float() == yy[km]).float().mean()) if bool(km.any()) else float("nan")}
    if keep:
        rec["logits"] = z.detach().cpu()
    return rec


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, lam: float) -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vs[k] for k in TRAINING_KEYS}, "arm": arm, "seed": int(seed), "lambda": float(lam),
               "manifest_sha256": msha, "start_sha256": ssha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": h.hexdigest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def save_state(path: Path, model, head, opt, done: int, ident: str, log: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": done, "model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(), "log": log}, tmp)
    tmp.replace(path)


def load_state(path: Path, model, head, opt, ident: str) -> tuple[int, list]:
    st_ = torch.load(path, map_location="cpu", weights_only=False)
    if st_["identity"] != ident:
        raise SystemExit(f"{path} belongs to another identity")
    model.load_state_dict(st_["model"]); head.load_state_dict(st_["head"]); opt.load_state_dict(st_["opt"]); torch.set_rng_state(st_["rng_cpu"]); torch.cuda.set_rng_state_all(st_["rng_cuda"])
    return int(st_["updates"]), list(st_["log"])


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    lam = ctx.lam(); m, msha = ctx.manifest(seed); W = m[f"w_{arm}"]; model, teacher, head, ssha = models(ctx, seed, device)
    ident = identity(ctx, arm, seed, msha, ssha, lam); key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        return {"key": key, "status": "finished"}
    opt = make_optimizer(model, head, ctx.o)
    if not bound(opt, model, head):
        raise SystemExit("the optimiser is not bound to the model and head")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = load_state(ck / "resume.pt", model, head, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []; ul.write_text("".join(x + "\n" for x in lines))
    else:
        for p in (ul, rd / "train_log.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]} lambda {lam}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []
    torch.cuda.reset_peak_memory_stats(device)
    for batch in vsh.loader(m, ctx.frame_paths(), list(range(done, ctx.total)), workers):
        u = int(batch[0]); rvc.check_digests(m, vsh.vso.GROUP_NAME, u, batch[4])
        if not np.array_equal(batch[3].numpy(), m["target"][u]) or not np.array_equal(batch[2].numpy(), m["marker"][u]):
            raise SystemExit(f"update {u}: the loader's targets or markers differ from the manifest")
        lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, teacher, head, opt, batch, m["c_edge"][u], W[u], lam, lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.nanmean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "bce", "grad_norm", "edge_acc")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)),
                   "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); save_state(ck / "resume.pt", model, head, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} BCE {rec['bce']:.4f} edge {rec['edge_acc']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "head": head.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "lambda": lam, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "start_sha256": ssha,
           "updates": done, "epochs": log, "final_checkpoint": str(final), "final_sha256": sha256_file(final), "head_parameters": int(sum(p.numel() for p in head.parameters()))}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


# ----------------------------------------------------------------------------------------------- checks and queue
def batch_of(ctx: Ctx, seed: int, u: int):
    m, _ = ctx.manifest(seed); b = vsh.Inputs(m, ctx.frame_paths(), [u])[0]; rvc.check_digests(m, vsh.vso.GROUP_NAME, u, b[4])
    return b, m


def run_updates(ctx: Ctx, arm: str, lam: float, us: list[int], device, state_file: Path | None = None, keep: bool = False) -> dict:
    model, teacher, head, _ = models(ctx, 0, device); opt = make_optimizer(model, head, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    if state_file is not None:
        load_state(state_file, model, head, opt, "check")
    recs = []
    for u in us:
        b, m = batch_of(ctx, 0, u); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        recs.append(train_step(model, teacher, head, opt, b, m["c_edge"][u], m[f"w_{arm}"][u], lam, lrs, ctx.o, device, keep=keep))
    return {"model": model, "head": head, "opt": opt, "teacher": teacher, "records": recs}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism(); data = vpr.training_data(ctx.a10.a07p); seqs = data.seqs; bad = 0
        for s in (0, 1, 2):
            m, _ = ctx.manifest(s)
            for u in range(50):
                for g in range(NG):
                    q = seqs[int(m["seq"][u, 4 * g + 1])]; lab = np.asarray(q["labels"]).tolist(); a, b = int(m["c"][u, 4 * g + 1]), int(m["c"][u, 4 * g + 2]); seg = lab[a:b + 1]
                    n = sum(1 for i in range(1, len(seg)) if seg[i] != seg[i - 1]); want = -1 if b <= a else (0 if n == 0 else (1 if seg[0] != seg[-1] else -1))
                    bad += int(want != int(m["c_edge"][u, g])) + int(int(m["seq"][u, 4 * g + 1]) != int(m["seq"][u, 4 * g + 2]))
            neg, pos = m["c_edge"] == 0, m["c_edge"] == 1
            if abs(m["w_U"][neg].sum() - m["w_H"][neg].sum()) > 1e-6 * m["w_U"][neg].sum() or not np.array_equal(m["w_U"][pos], m["w_H"][pos]):
                raise SystemExit(f"seed {s}: H's masses differ from U's")
        out["edge_recheck"] = {"updates": 50, "seeds": 3, "mismatches": bad}
        if bad:
            raise SystemExit(f"{bad} central-edge mismatches")
        rows = []
        for s in (0, 2):
            model, teacher, head, _ = models(ctx, s, device); torch.manual_seed(s); torch.cuda.manual_seed_all(s); enc = model.encoder_parameters(); model.set_encoder_trainable(True)
            for u in range(4):
                b, m = batch_of(ctx, s, u); model.train(); rgb, mk = tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)); t = b[3].to(device)
                e = model.embed(rgb, mk); z = model.classifier(e)
                with torch.no_grad():
                    zt = teacher(rgb, mk)
                lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); base = F.cross_entropy(z, t) + (lpt.exp() * (lpt - lp)).sum(-1).mean()
                sc = head(central_features(z, e, TRAIN_IDX)); y = torch.as_tensor(m["c_edge"][u], device=device).float(); w = torch.as_tensor(m["w_U"][u], device=device).float(); km = y >= 0
                bce = (w[km] * F.binary_cross_entropy_with_logits(sc[km], y[km], reduction="none")).mean()
                norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None)))  # noqa: E731
                g0 = norm(torch.autograd.grad(base, enc, retain_graph=True, allow_unused=True)); g1 = norm(torch.autograd.grad(bce, enc, allow_unused=True))
                rows.append({"seed": s, "u": u, "ratio": g1 / max(g0, 1e-30), "bce": float(bce)})
            del model, teacher, head; torch.cuda.empty_cache()
        r = float(np.median([x["ratio"] for x in rows])); fit = [x for x in LADDER if x * r <= 1.0]; lam = fit[0] if fit else LADDER[-1]
        out["lambda_rule"] = {"batches": rows, "ratio": r, "lambda": lam}; print(f"lambda rule: ratio {r:.4g} -> lambda {lam}", flush=True)
        rec = {}
        for arm in ARMS:
            x = run_updates(ctx, arm, lam, [0], device, keep=True); rec[arm] = (x["records"][0]["logits"], torch.cuda.get_rng_state_all()); del x; torch.cuda.empty_cache()
        out["pairing"] = {"logits_identical": bool(torch.equal(rec["U"][0], rec["H"][0])), "cuda_rng_identical": all(torch.equal(p, q) for p, q in zip(rec["U"][1], rec["H"][1]))}
        x = run_updates(ctx, "H", lam, [ctx.warmup], device); mx = lambda p: None if p.grad is None else float(p.grad.abs().max())  # noqa: E731
        out["gradients"] = {"projection": mx(x["model"].encoder.head[-1].weight), "conv_proj": mx(x["model"].encoder.conv_proj.weight), "head": mx(x["head"].net[0].weight),
                            "teacher_with_grad": int(sum(p.grad is not None for p in x["teacher"].parameters()))}; del x; torch.cuda.empty_cache()
        model, teacher, head, _ = models(ctx, 0, device); b, _ = batch_of(ctx, 0, 0); model.eval()
        with torch.no_grad():
            out["teacher_equals_start"] = bool(torch.equal(model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))), teacher(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))))
        out["optimiser_bound"] = bool(bound(make_optimizer(model, head, ctx.o), model, head)); out["head_parameters"] = int(sum(p.numel() for p in head.parameters())); del model, teacher, head
        us = list(range(ctx.warmup, ctx.warmup + 6)); full = run_updates(ctx, "H", lam, us, device); a3 = run_updates(ctx, "H", lam, us[:3], device); tmp = ctx.out / "check_resume.pt"
        save_state(tmp, a3["model"], a3["head"], a3["opt"], 3, "check", []); del a3; b3 = run_updates(ctx, "H", lam, us[3:], device, state_file=tmp); tmp.unlink()
        rerun = run_updates(ctx, "H", lam, us, device); rr_ = max(max_diff(full["model"], rerun["model"]), max_diff(full["head"], rerun["head"])); del rerun
        out["resume"] = {"loss_max_abs": float(max(abs(x["loss"] - y["loss"]) for x, y in zip(full["records"][3:], b3["records"]))),
                         "params_max_abs": max(max_diff(full["model"], b3["model"]), max_diff(full["head"], b3["head"])), "rerun_params_max_abs": rr_}
        out["resume"]["ok"] = bool(out["resume"]["loss_max_abs"] <= 1e-4 and out["resume"]["params_max_abs"] <= max(1e-5, 2 * rr_)); del full, b3; torch.cuda.empty_cache()
        model, teacher, head, _ = models(ctx, 0, device); opt = make_optimizer(model, head, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.cuda.reset_peak_memory_stats(device)
        m, _ = ctx.manifest(0); ts = []
        for i, batch in enumerate(vsh.loader(m, ctx.frame_paths(), list(range(ctx.warmup, ctx.warmup + 23)), workers)):
            torch.cuda.synchronize(); a = time.time(); u = int(batch[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
            train_step(model, teacher, head, opt, batch, m["c_edge"][u], m["w_H"][u], lam, lrs, ctx.o, device); torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)) * 1.05; out["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}; del model, teacher, head, opt
        spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; runs = 4 * ctx.total * spu / 3600
        out["budget"] = {"spent": spent, "four_runs": runs, "evaluation": 0.35, "projection": spent + runs + 0.35, "target": float(ctx.vs["budget"]["screen_target_gpu_hours"]), "cap": float(ctx.vs["budget"]["gpu_hours_max"])}
        g = out["gradients"]
        out["ok"] = bool(out["pairing"]["logits_identical"] and out["pairing"]["cuda_rng_identical"] and g["projection"] and g["conv_proj"] and g["head"] and g["teacher_with_grad"] == 0
                         and out["teacher_equals_start"] and out["optimiser_bound"] and out["resume"]["ok"] and out["budget"]["projection"] <= out["budget"]["cap"])
        if out["ok"]:
            write_json(ctx.out / "lambda.json", {"config_sha256": out["config_sha256"], "lambda": lam, "ratio": r, "time": now()})
        out["finished"] = now(); ok = out["ok"]; note = f"lambda {lam}, ratio {r:.4g}, {spu:.3f} s/update, projection {out['budget']['projection']:.3f}"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps({k: out[k] for k in ("edge_recheck", "pairing", "gradients", "teacher_equals_start", "optimiser_bound", "head_parameters", "resume", "timing", "budget", "ok")}, indent=1), flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check", "gpu", started, time.time() - t0, ok, note)


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"])
    for i, key in enumerate(keys):
        arm, s = key.rsplit("_s", 1); spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * ctx.total * spu / 3600 + 0.35
        print(f"[queue] before {key}: spent {spent:.3f}, projection {proj:.3f} of {ctx.vs['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vs["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, int(s), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.workers)
    determinism(); device = torch.device("cuda")
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
