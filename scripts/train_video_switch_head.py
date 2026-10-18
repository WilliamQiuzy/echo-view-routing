#!/usr/bin/env python
"""Prepare, check, train and queue A10 (configs/experiments/video_switch_head_v1.yaml).

Every run strictly loads its seed's frozen A08 all_s{k} into the unchanged TargetMViT; the same weights, frozen in eval
mode, are the teacher. Update u reads row u of this round's manifest through V01's loader: A07's native and
change-containing groups of row u plus a same-label join group, 12 windows in one forward. R trains CE + KL; P and V add
lambda_b times balanced BCE of their switch head on the 9 adjacent-window edges. Both heads exist in every arm, built
under fork_rng, so the backbone's random sequence is the same in all arms. Only training data is read.

  --prepare            CPU: pins, manifests (groups 0-1 equal to A07's rows), join groups, edge labels, digests, counts
  --check              GPU: the lambda rule, pairing, edges, gradients, teacher, optimiser, resume, timing, projection
  --queue K [K ...]    runs in order (R_s0, P_s0, ...)
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
import scripts.prepare_video_feature_temporal as pvft  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
import scripts.train_video_sequence_objective as vso  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

NS = "video_switch_head_v1"
GROUP, NGROUPS = vpr.GROUP, 3
WINDOWS, SLOTS = GROUP * NGROUPS, vpr.SLOTS
EDGE_A = np.array([g * GROUP + t - 1 for g in range(NGROUPS) for t in range(1, GROUP)])
EDGE_B = EDGE_A + 1
RNG_BASE = 20261019
LADDER = (1.0, 0.5, 0.2, 0.1, 0.05)
ARMS = ("R", "P", "V")
HEAD_DIMS = {"P": 15, "V": 15 + tm.EMBED_DIM}
HIDDEN = 32
TRAINING_KEYS = ("predecessors", "source", "arms", "model", "data", "loss", "training")
CODE_FILES = (Path(__file__), Path(rvc.__file__), Path(vso.__file__), Path(vpr.__file__), Path(tm.__file__), Path(prv.__file__), Path(pvft.__file__),
              Path(vsf.__file__), Path(td.__file__))
make_model, optimiser_bound, max_diff, now, write_json, ledger_hours, determinism = vso.make_model, vso.optimiser_bound, vso.max_diff, vso.now, vso.write_json, vso.ledger_hours, rvc.determinism
ledger_append = vpr.ledger_append


def pinned(pin: dict) -> Path:
    return vpr.pinned(ROOT, pin)


# ----------------------------------------------------------------------------------------------- heads
class SwitchHead(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_heads(seed: int) -> nn.ModuleDict:
    """Both heads, each initialised under fork_rng with its own fixed seed, so the global CPU RNG is unchanged."""
    out = {}
    for name, base in (("P", 1000), ("V", 2000)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base + int(seed)); out[name] = SwitchHead(HEAD_DIMS[name])
    return nn.ModuleDict(out)


def edge_features(z: torch.Tensor, e: torch.Tensor, a, b, kind: str) -> torch.Tensor:
    a, b = torch.as_tensor(a, device=z.device, dtype=torch.long), torch.as_tensor(b, device=z.device, dtype=torch.long)
    p = torch.softmax(z, -1); x = torch.cat([p[a], p[b], (p[b] - p[a]).abs()], -1)
    if kind == "V":
        x = torch.cat([x, F.layer_norm(e[b] - e[a], (e.shape[-1],))], -1)
    return x


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vs = cfg, config, cfg["video_switch_head"]; self.src = self.vs["source"]
        for k in ("a07_config", "a08_config", "a08_choices"):
            pinned(self.src[k])
        a07 = self.src["a07_config"]; a07cfg = load_cfg(SimpleNamespace(config=a07["path"], set=[]))
        self.a07 = vso.Ctx(a07cfg, a07["path"]); self.a07p = vpr.Ctx(a07cfg, a07["path"]); self.rv, self.paths = self.a07.rv, self.a07.paths
        a08 = load_cfg(SimpleNamespace(config=self.src["a08_config"]["path"], set=[]))["video_within_view"]["source"]["manifests"]; self.a07pins = a08
        tr = self.vs["training"]; self.seeds = [int(s) for s in tr["seeds"]]; self.total, self.warmup, self.o = int(tr["total_updates"]), int(tr["warmup_updates"]), tr["optimiser"]
        o = self.vs["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.mdir = self.out / "manifests"; self.ledger = ROOT / self.vs["budget"]["ledger"]; self._m, self._fp = {}, None

    def a07_manifest(self, seed: int) -> dict:
        m, sha = self.a07.manifest(seed)
        if sha != self.a07pins[seed]["sha256"]:
            raise SystemExit(f"seed {seed}: A07's manifest differs from A08's pin")
        return m

    def start_state(self, seed: int) -> tuple[dict, str]:
        p = pinned(self.src["a08_weights"][seed])
        return torch.load(p, map_location="cpu", weights_only=True), self.src["a08_weights"][seed]["sha256"]

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
        return float(rec["lambda_b"])


# ----------------------------------------------------------------------------------------------- manifests (CPU)
def changes_in(lab: np.ndarray, a: int, b: int) -> int:
    return int((lab[a + 1:b + 1] != lab[a:b]).sum()) if b > a else 0


def edge_label(lab: np.ndarray, a: int, b: int) -> int:
    """1 change, 0 none, -1 masked (a repeat, or equal ends with changes inside)."""
    if b <= a:
        return -1
    n = changes_in(lab, a, b)
    return 0 if n == 0 else 1 if lab[a] != lab[b] else -1


def join_groups(seqs: list[dict], info: list[dict]) -> list[tuple[int, int]]:
    """(sequence, start) of every bank group whose 4 centres span a same-label join and no label change."""
    out = []
    for i, (q, f) in enumerate(zip(seqs, info)):
        if q["kind"] == "native":
            continue
        lab, c = np.asarray(q["labels"]), np.asarray(f["windows"]["centres"]); joins = np.asarray(q["joins"], int)
        same = joins[(joins > 0) & (lab[np.clip(joins - 1, 0, None)] == lab[np.clip(joins, 0, lab.size - 1)])]
        for k in range(len(c) - GROUP + 1):
            lo, hi = int(c[k]), int(c[k + GROUP - 1])
            if changes_in(lab, lo, hi) == 0 and bool(((same > lo) & (same <= hi)).any()):
                out.append((i, k))
    return out


def plan(seqs, info, a07m: dict, eligible: list, seed: int, total: int) -> list[list[tuple[int, int]]]:
    out = []
    for u in range(int(total)):
        ws = []
        for g in range(2):
            qi = int(a07m["group_seq"][u, g]); idx = vpr.group_indices(len(info[qi]["windows"]["centres"]), int(a07m["group_start"][u, g])); ws += [(qi, int(ci)) for ci in idx]
        rng = np.random.default_rng([RNG_BASE, int(seed), int(u)]); qi, k = eligible[int(rng.integers(len(eligible)))]; ws += [(qi, k + t) for t in range(GROUP)]
        out.append(ws)
    return out


def build_manifest(seqs, info, windows, a07m: dict) -> dict:
    U = len(windows); refs_of = lambda qi, ci: [seqs[qi]["refs"][int(x)] for x in info[qi]["windows"]["samples"][int(ci)]]  # noqa: E731
    videos = sorted({r[0] for ws in windows for qi, ci in ws for r in refs_of(qi, ci)}); vix = {v: i for i, v in enumerate(videos)}
    m = {"vid": np.empty((U, WINDOWS, SLOTS), np.int32), "frame": np.empty((U, WINDOWS, SLOTS), np.int32), "gamma": np.empty((U, WINDOWS, SLOTS), np.float32),
         "s": np.empty((U, WINDOWS, SLOTS), np.int32), "c": np.empty((U, WINDOWS), np.int32), "target": np.empty((U, WINDOWS), np.int8),
         "kind": np.empty((U, WINDOWS), np.int8), "seq": np.empty((U, WINDOWS), np.int32), "centre_index": np.empty((U, WINDOWS), np.int32),
         "marker": np.empty((U, WINDOWS), np.int8), "repeats": np.zeros((U, WINDOWS), bool), "edge": np.empty((U, len(EDGE_A)), np.int8),
         "edge_join": np.zeros((U, len(EDGE_A)), bool)}
    for u, ws in enumerate(windows):
        for w, (qi, ci) in enumerate(ws):
            W = info[qi]["windows"]; s = W["samples"][int(ci)]; c = int(W["centres"][int(ci)]); refs = refs_of(qi, ci)
            m["vid"][u, w] = [vix[r[0]] for r in refs]; m["frame"][u, w] = [int(r[1]) for r in refs]; m["gamma"][u, w] = [float(r[2]) for r in refs]
            m["s"][u, w], m["c"][u, w], m["target"][u, w], m["kind"][u, w] = s, c, int(np.asarray(seqs[qi]["labels"])[c]), w // GROUP
            m["seq"][u, w], m["centre_index"][u, w], m["marker"][u, w] = qi, int(ci), int(W["markers"][int(ci)])
            m["repeats"][u, w] = bool(w % GROUP > 0 and ws[w - 1] == (qi, ci))
        for k, (wa, wb) in enumerate(zip(EDGE_A, EDGE_B)):
            qi = ws[wa][0]; lab = np.asarray(seqs[qi]["labels"]); a, b = int(m["c"][u, wa]), int(m["c"][u, wb]); joins = np.asarray(seqs[qi]["joins"], int)
            m["edge"][u, k] = edge_label(lab, a, b); m["edge_join"][u, k] = bool(((joins > a) & (joins <= b)).any())
    m["order_ordered"] = np.broadcast_to(np.arange(SLOTS, dtype=np.int8), (U, WINDOWS, SLOTS)).copy(); m["videos"] = np.asarray(videos)
    return m


def groups01_equal(m: dict, a07m: dict, U: int) -> bool:
    same = np.array_equal(m["videos"][m["vid"][:, :2 * GROUP]], a07m["videos"][a07m["vid"][:U, :2 * GROUP]])
    for k in ("frame", "gamma", "s", "c", "target", "marker", "centre_index", "seq"):
        same = same and np.array_equal(m[k][:, :2 * GROUP], a07m[k][:U, :2 * GROUP])
    return bool(same)


def edge_counts(m: dict, seqs: list[dict]) -> dict:
    e, j = m["edge"], m["edge_join"]; g = np.repeat(np.arange(NGROUPS), GROUP - 1)[None, :].repeat(len(e), 0)
    out = {"by_group": {}, "pos": int((e == 1).sum()), "neg": int((e == 0).sum()), "masked": int((e == -1).sum())}
    for gi, name in enumerate(("native", "constructed_change", "same_label_join")):
        sel = g == gi; out["by_group"][name] = {"pos": int(((e == 1) & sel).sum()), "neg": int(((e == 0) & sel).sum()), "masked": int(((e == -1) & sel).sum()),
                                                "join_neg": int(((e == 0) & j & sel).sum()), "join_pos": int(((e == 1) & j & sel).sum())}
    pairs, short = Counter(), 0
    for u in range(len(e)):
        for k, (wa, wb) in enumerate(zip(EDGE_A, EDGE_B)):
            if e[u, k] == 1:
                pairs[f"{vpr.CLASSES[int(m['target'][u, wa])]}->{vpr.CLASSES[int(m['target'][u, wb])]}"] += 1
                lab = np.asarray(seqs[int(m["seq"][u, wb])]["labels"]); b = int(m["c"][u, wb]); y = lab[b]; lo = b
                while lo > 0 and lab[lo - 1] == y:
                    lo -= 1
                hi = b
                while hi + 1 < lab.size and lab[hi + 1] == y:
                    hi += 1
                short += int(hi - lo + 1 <= 16)
    out["positive_class_pairs"] = dict(sorted(pairs.items())); out["positives_into_segments_of_at_most_16"] = short
    w = {"pos": float((out["pos"] + out["neg"]) / (2 * max(1, out["pos"]))), "neg": float((out["pos"] + out["neg"]) / (2 * max(1, out["neg"])))}
    out["balanced_weights"] = w
    return out


def prepare(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        rec = {"config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config), "started": started,
               "pins": {k: sha256_file(pinned(ctx.src[k])) for k in ("a07_config", "a08_config", "a08_choices")} | {"a07_inherited": vpr.verify_sources(ctx.a07p)},
               "a08_weights": {str(s): sha256_file(pinned(p)) for s, p in ctx.src["a08_weights"].items()}}
        data = vpr.training_data(ctx.a07p); seqs = data.seqs; info = vpr.sequence_info(seqs); eligible = join_groups(seqs, info)
        rec["join_groups"] = {"eligible": len(eligible), "streams": len({q for q, _ in eligible})}
        seeds = sorted({0, 1, 2}); ms = {}
        for s in seeds:
            a07m = ctx.a07_manifest(s); ws = plan(seqs, info, a07m, eligible, s, ctx.total)
            if plan(seqs, info, a07m, eligible, s, ctx.total) != ws:
                raise SystemExit(f"seed {s}: a second draw differs")
            m = build_manifest(seqs, info, ws, a07m)
            if not groups01_equal(m, a07m, ctx.total):
                raise SystemExit(f"seed {s}: groups 0-1 differ from A07's rows")
            ms[s] = (m, a07m)
        refs = {(str(m["videos"][v]), int(f), round(float(g), 6)) for m, _ in ms.values() for v, f, g in
                zip(m["vid"][:, 2 * GROUP:].ravel(), m["frame"][:, 2 * GROUP:].ravel(), m["gamma"][:, 2 * GROUP:].ravel())}
        vids = {str(v) for m, _ in ms.values() for v in m["videos"].tolist()}
        if vids - set(prv.allowlist(ctx.rv, ROOT)):
            raise SystemExit("a recording lies outside the training allowlist")
        dirs = prv.frame_dirs(ctx.paths, ctx.rv, {r[0] for r in refs}); t1 = time.time(); hashes = prv.pixel_hashes(refs, dirs, workers)
        rec["pixels"] = {"distinct_join_group_frames": len(hashes), "seconds": time.time() - t1}
        ctx.mdir.mkdir(parents=True, exist_ok=True); index = {"config": ctx.config, "seeds": {}}; rec["seeds"] = {}
        for s, (m, a07m) in ms.items():
            m["digest_ordered"] = np.concatenate([a07m["digest_ordered"][:ctx.total, :2 * GROUP], td.window_digests(m, hashes, range(2 * GROUP, WINDOWS))], axis=1)
            path = ctx.mdir / f"VSH_s{s}.npz"
            if path.is_file():
                raise SystemExit(f"{path.name} exists; manifests are never overwritten")
            np.savez(path, **m); sha = sha256_file(path)
            index["seeds"][str(s)] = {"file": path.name, "sha256": sha, "arrays_digest": vpr.arrays_digest(m)}
            g2 = m["seq"][:, 2 * GROUP:]; srcs = {str(m["videos"][v]) for v in m["vid"][:, 2 * GROUP:].ravel()}
            rec["seeds"][str(s)] = {"sha256": sha, "edges": edge_counts(m, seqs), "join_group": {"distinct_streams": int(len(set(g2.ravel().tolist()))),
                                    "recordings": len(srcs), "sessions": len({session_of(v) for v in srcs})}, "repeated_targets": int(m["repeats"].sum())}
        write_json(ctx.mdir / "index.json", index)
        ids = {str(v) for m, _ in ms.values() for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}
        rec["finished"] = now(); rec["seconds"] = time.time() - t0; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps({k: rec[k] for k in ("join_groups", "pixels")}), json.dumps(rec["seeds"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- inputs, optimiser, identity
class Inputs(Dataset):
    def __init__(self, m: dict, fp, updates: list[int]) -> None:
        self.m, self.order, self.fp, self.updates = m, m["order_ordered"], fp, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]
        return (u, *rvc.window_inputs(self.m, self.order, self.fp, [(u, w) for w in range(WINDOWS)]))


def loader(m: dict, fp, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(Inputs(m, fp, updates), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def make_optimizer(model, heads, o: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW([{"params": model.encoder_parameters(), "lr": 0.0, "name": "encoder"}, {"params": model.new_parameters(), "lr": 0.0, "name": "new"},
                              {"params": list(heads.parameters()), "lr": 0.0, "name": "heads"}],
                             betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))


def bound(opt, model, heads) -> bool:
    ids = lambda ps: [id(p) for p in ps]  # noqa: E731
    return (len(opt.param_groups) == 3 and ids(opt.param_groups[0]["params"]) == ids(model.encoder_parameters())
            and ids(opt.param_groups[1]["params"]) == ids(model.new_parameters()) and ids(opt.param_groups[2]["params"]) == ids(heads.parameters()))


def apply_schedule(model, opt, u: int, warmup: int, total: int, o: dict) -> tuple[float, float, float]:
    f = lr_factor(int(u), int(warmup), int(total)); enc = 0.0 if u < warmup else float(o["lr_encoder"]) * f
    model.set_encoder_trainable(u >= warmup); new, head = float(o["lr_new"]) * f, float(o["lr_head"]) * f
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"], opt.param_groups[2]["lr"] = enc, new, head
    return enc, new, head


def code_digest() -> str:
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, lam: float) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "video_switch_head": {k: ctx.vs[k] for k in TRAINING_KEYS}, "arm": arm, "seed": int(seed),
               "lambda_b": float(lam) if arm != "R" else 0.0, "manifest_sha256": msha, "start_sha256": ssha, "environment": prv.environment(),
               "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def models(ctx: Ctx, seed: int, device):
    state, ssha = ctx.start_state(seed); student = make_model(state, device); teacher = make_model(state, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher, build_heads(seed).to(device), ssha


# ----------------------------------------------------------------------------------------------- one update
def train_step(model, teacher, heads, opt, batch, edges: np.ndarray, arm: str, lam: float, weights: dict, lrs, o: dict, device, keep: bool = False) -> dict:
    u8, slots, target = batch[1], batch[2], batch[3]; t = target.to(device); model.train(); heads.train()
    rgb, mk = tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device))
    e = model.embed(rgb, mk); z = model.classifier(e)
    with torch.no_grad():
        zt = teacher(rgb, mk)
    ce = F.cross_entropy(z, t); lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean()
    loss = ce + kl; y = torch.as_tensor(edges, device=device).long(); keepm = y >= 0; rec = {"bce": float("nan")}
    if arm in ("P", "V"):
        s = heads[arm](edge_features(z, e, EDGE_A, EDGE_B, arm)); yy = y[keepm].float(); ss = s[keepm]
        w = torch.where(yy > 0.5, torch.full_like(yy, weights["pos"]), torch.full_like(yy, weights["neg"]))
        bce = (w * F.binary_cross_entropy_with_logits(ss, yy, reduction="none")).mean() if yy.numel() else s.sum() * 0.0
        loss = loss + float(lam) * bce; rec["bce"] = float(bce.detach())
        rec["edge_acc"] = float(((ss.detach() > 0).float() == yy).float().mean()) if yy.numel() else float("nan")
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); pred = z.detach().argmax(1).cpu().numpy(); tt = target.numpy()
    rec.update({"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1], "lr_head": lrs[2],
                "correct": int((pred == tt).sum()), "edges_used": int(keepm.sum())})
    if keep:
        rec["logits"] = z.detach().cpu()
    return rec


# ----------------------------------------------------------------------------------------------- formal run
def save_state(path: Path, model, heads, opt, done: int, ident: str, log: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": done, "model": model.state_dict(), "heads": heads.state_dict(), "opt": opt.state_dict(), "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(), "log": log}, tmp)
    tmp.replace(path)


def load_state(path: Path, model, heads, opt, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another identity")
    model.load_state_dict(st["model"]); heads.load_state_dict(st["heads"]); opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["updates"]), list(st["log"])


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    lam = ctx.lam() if arm != "R" else 0.0; m, msha = ctx.manifest(seed); prep = json.loads((ctx.out / "preparation.json").read_text())
    weights = prep["seeds"][str(seed)]["edges"]["balanced_weights"]; model, teacher, heads, ssha = models(ctx, seed, device)
    ident = identity(ctx, arm, seed, msha, ssha, lam); key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key
    rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        print(f"[{key}] already finished", flush=True); return {"key": key, "status": "finished"}
    opt = make_optimizer(model, heads, ctx.o)
    if not bound(opt, model, heads):
        raise SystemExit("the optimiser is not bound to the model and heads")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = load_state(ck / "resume.pt", model, heads, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []
        ul.write_text("".join(x + "\n" for x in lines))
        with (rd / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (ul, rd / "resumes.jsonl", rd / "train_log.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]} lambda_b {lam}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, ctx.frame_paths(), list(range(done, ctx.total)), workers):
        u = int(batch[0]); rvc.check_digests(m, vso.GROUP_NAME, u, batch[4])
        if not np.array_equal(batch[3].numpy(), m["target"][u]) or not np.array_equal(batch[2].numpy(), m["marker"][u]):
            raise SystemExit(f"update {u}: the loader's targets or markers differ from the manifest")
        lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, teacher, heads, opt, batch, m["edge"][u], arm, lam, weights, lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.nanmean([a.get(k, np.nan) for a in acc]))  # noqa: E731
            rec = {"updates": done, "n": len(acc), **{k: mean(k) for k in ("loss", "ce", "kl", "bce", "grad_norm")}, "edge_acc": mean("edge_acc") if arm != "R" else None,
                   "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)), "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} BCE {rec['bce']:.4f} acc {rec['accuracy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "lambda_b": lam, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha,
           "start_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log,
           "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "head_parameters": {k: int(sum(p.numel() for p in h.parameters())) for k, h in heads.items()},
           "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True)
    print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, seed: int, u: int):
    m, _ = ctx.manifest(seed); b = Inputs(m, ctx.frame_paths(), [u])[0]; rvc.check_digests(m, vso.GROUP_NAME, u, b[4])
    return b, m["edge"][u]


def grad_ratio(ctx: Ctx, seed: int, device, weights: dict) -> list[dict]:
    model, teacher, heads, _ = models(ctx, seed, device); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rows = []
    enc = model.encoder_parameters(); model.set_encoder_trainable(True)
    for u in range(4):
        b, edges = batch_of(ctx, seed, u); model.train(); rgb, mk = tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)); t = b[3].to(device)
        e = model.embed(rgb, mk); z = model.classifier(e)
        with torch.no_grad():
            zt = teacher(rgb, mk)
        lp, lpt = F.log_softmax(z, -1), F.log_softmax(zt, -1); base = F.cross_entropy(z, t) + (lpt.exp() * (lpt - lp)).sum(-1).mean()
        g0 = torch.autograd.grad(base, enc, retain_graph=True, allow_unused=True); y = torch.as_tensor(edges, device=device).long(); km = y >= 0; row = {"seed": seed, "u": u}
        norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None)))  # noqa: E731
        row["grad_base"] = norm(g0)
        for k in ("P", "V"):
            s = heads[k](edge_features(z, e, EDGE_A, EDGE_B, k)); yy = y[km].float()
            w = torch.where(yy > 0.5, torch.full_like(yy, weights["pos"]), torch.full_like(yy, weights["neg"]))
            bce = (w * F.binary_cross_entropy_with_logits(s[km], yy, reduction="none")).mean(); gk = torch.autograd.grad(bce, enc, retain_graph=True, allow_unused=True)
            row[f"grad_{k}"] = norm(gk); row[f"ratio_{k}"] = row[f"grad_{k}"] / max(row["grad_base"], 1e-30); row[f"bce_{k}"] = float(bce)
        rows.append(row)
    del model, teacher, heads; torch.cuda.empty_cache()
    return rows


def run_updates(ctx: Ctx, arm: str, lam: float, us: list[int], device, weights: dict, state_file: Path | None = None, keep: bool = False) -> dict:
    model, teacher, heads, _ = models(ctx, 0, device); opt = make_optimizer(model, heads, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    if state_file is not None:
        load_state(state_file, model, heads, opt, "check")
    recs = []
    for u in us:
        b, edges = batch_of(ctx, 0, u); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        recs.append(train_step(model, teacher, heads, opt, b, edges, arm, lam, weights, lrs, ctx.o, device, keep=keep))
    return {"model": model, "heads": heads, "opt": opt, "teacher": teacher, "records": recs}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism(); prep = json.loads((ctx.out / "preparation.json").read_text()); W = {s: prep["seeds"][str(s)]["edges"]["balanced_weights"] for s in (0, 1, 2)}
        # edge labels recomputed independently on the first 50 updates of each seed
        data = vpr.training_data(ctx.a07p); seqs = data.seqs; bad = 0
        for s in (0, 1, 2):
            m, _ = ctx.manifest(s)
            for u in range(50):
                for k, (wa, wb) in enumerate(zip(EDGE_A, EDGE_B)):
                    q = seqs[int(m["seq"][u, wa])]; lab = np.asarray(q["labels"]).tolist(); a, b = int(m["c"][u, wa]), int(m["c"][u, wb])
                    seg = lab[a:b + 1]; nchg = sum(1 for i in range(1, len(seg)) if seg[i] != seg[i - 1])
                    want = -1 if b <= a else (0 if nchg == 0 else (1 if seg[0] != seg[-1] else -1))
                    bad += int(want != int(m["edge"][u, k])) + int(int(m["seq"][u, wa]) != int(m["seq"][u, wb]))
        out["edge_recheck"] = {"updates": 50, "seeds": 3, "mismatches": bad}
        if bad:
            raise SystemExit(f"{bad} edge label or pairing mismatches")
        # the lambda rule (training batches only)
        rows = grad_ratio(ctx, 0, device, W[0]) + grad_ratio(ctx, 2, device, W[2])
        r = max(float(np.median([x["ratio_P"] for x in rows])), float(np.median([x["ratio_V"] for x in rows]))); fit = [x for x in LADDER if x * r <= 1.0]; lam = fit[0] if fit else LADDER[-1]
        out["lambda_rule"] = {"batches": rows, "ratio": r, "lambda_b": lam, "flag_none_fits": not fit}; print(f"lambda rule: ratio {r:.4g} -> lambda_b {lam}", flush=True)
        # pairing: one update of R, P and V from identical states give identical logits before the loss and the same RNG after
        rec = {}
        for arm in ARMS:
            x = run_updates(ctx, arm, lam, [0], device, W[0], keep=True); rec[arm] = (x["records"][0]["logits"], torch.cuda.get_rng_state_all()); del x; torch.cuda.empty_cache()
        out["pairing"] = {"logits_identical": all(torch.equal(rec["R"][0], rec[a][0]) for a in ("P", "V")),
                          "cuda_rng_identical": all(all(torch.equal(p, q) for p, q in zip(rec["R"][1], rec[a][1])) for a in ("P", "V"))}
        # gradients after unfreezing: the trained head, the encoder, the marker and classifier; none for the teacher or the other head
        grads = {}
        for arm in ("P", "V"):
            x = run_updates(ctx, arm, lam, [ctx.warmup], device, W[0]); mdl, hd, tch = x["model"], x["heads"], x["teacher"]
            mx = lambda p: None if p.grad is None else float(p.grad.abs().max())  # noqa: E731
            grads[arm] = {"projection": mx(mdl.encoder.head[-1].weight), "conv_proj": mx(mdl.encoder.conv_proj.weight), "marker": mx(mdl.marker_stem.weight),
                          "classifier": mx(mdl.classifier.weight), "own_head": mx(hd[arm].net[0].weight), "other_head": mx(hd["V" if arm == "P" else "P"].net[0].weight),
                          "teacher_with_grad": int(sum(p.grad is not None for p in tch.parameters())), "teacher_requires_grad": int(sum(p.requires_grad for p in tch.parameters()))}
            del x; torch.cuda.empty_cache()
        out["gradients"] = grads
        g_ok = all(g["projection"] and g["conv_proj"] and g["own_head"] and g["other_head"] is None and g["teacher_with_grad"] == 0 and g["teacher_requires_grad"] == 0 for g in grads.values())
        # teacher equals the student at the start (eval) and the optimiser binding
        model, teacher, heads, _ = models(ctx, 0, device); b, _ = batch_of(ctx, 0, 0); model.eval()
        with torch.no_grad():
            zs = model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))); zt = teacher(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))
        out["teacher_equals_start"] = bool(torch.equal(zs, zt)); out["optimiser_bound"] = bool(bound(make_optimizer(model, heads, ctx.o), model, heads))
        out["head_parameters"] = {k: int(sum(p.numel() for p in h.parameters())) for k, h in heads.items()}; out["inference_inputs"] = "rgb, marker (edge features from the model's own outputs only)"
        del model, teacher, heads; torch.cuda.empty_cache()
        # resume: 3 + 3 against 6 (V, after unfreezing)
        us = list(range(ctx.warmup, ctx.warmup + 6)); full = run_updates(ctx, "V", lam, us, device, W[0]); a3 = run_updates(ctx, "V", lam, us[:3], device, W[0]); tmp = ctx.out / "check_resume.pt"
        save_state(tmp, a3["model"], a3["heads"], a3["opt"], 3, "check", []); del a3; b3 = run_updates(ctx, "V", lam, us[3:], device, W[0], state_file=tmp); tmp.unlink()
        rerun = run_updates(ctx, "V", lam, us, device, W[0]); rr = max(max_diff(full["model"], rerun["model"]), max_diff(full["heads"], rerun["heads"])); del rerun
        out["resume"] = {"loss_max_abs": float(max(abs(x["loss"] - y["loss"]) for x, y in zip(full["records"][3:], b3["records"]))),
                         "params_max_abs": max(max_diff(full["model"], b3["model"]), max_diff(full["heads"], b3["heads"])), "rerun_params_max_abs": rr,
                         "by_part": {"encoder": max_diff(full["model"].encoder, b3["model"].encoder), "heads": max_diff(full["heads"], b3["heads"])}}
        out["resume"]["ok"] = bool(out["resume"]["loss_max_abs"] <= 1e-4 and out["resume"]["params_max_abs"] <= max(1e-5, 2 * rr)); del full, b3; torch.cuda.empty_cache()
        # timing: 20 formal V updates after 3 warm-up updates
        model, teacher, heads, _ = models(ctx, 0, device); opt = make_optimizer(model, heads, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device); m, _ = ctx.manifest(0); ts = []
        for i, batch in enumerate(loader(m, ctx.frame_paths(), list(range(ctx.warmup, ctx.warmup + 23)), workers)):
            torch.cuda.synchronize(); a = time.time(); u = int(batch[0]); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
            train_step(model, teacher, heads, opt, batch, m["edge"][u], "V", lam, W[0], lrs, ctx.o, device); torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)) * 1.10; out["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        del model, teacher, heads, opt; torch.cuda.empty_cache()
        spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; runs = 6 * ctx.total * spu / 3600
        out["budget"] = {"spent": spent, "six_runs": runs, "evaluation": 0.35, "screen_cap": float(ctx.vs["budget"]["screen_gpu_hours"]), "within": bool(spent + runs + 0.35 <= float(ctx.vs["budget"]["screen_gpu_hours"]))}
        out["budget"]["hard_cap"] = float(ctx.vs["budget"]["gpu_hours_max"]); out["budget"]["within_hard_cap"] = bool(spent + runs + 0.35 <= out["budget"]["hard_cap"])
        out["ok"] = bool(out["pairing"]["logits_identical"] and out["pairing"]["cuda_rng_identical"] and g_ok and out["teacher_equals_start"] and out["optimiser_bound"] and out["resume"]["ok"]
                         and out["budget"]["within_hard_cap"])
        if out["ok"]:
            write_json(ctx.out / "lambda.json", {"config_sha256": out["config_sha256"], "lambda_b": lam, "ratio": r, "time": now()})
        out["finished"] = now(); out["seconds"] = time.time() - t0; ok = out["ok"]; note = f"lambda_b {lam}, ratio {r:.4g}, {spu:.3f} s/update, projection {spent + runs + 0.5:.3f}"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps({k: out[k] for k in ("edge_recheck", "pairing", "gradients", "teacher_equals_start", "optimiser_bound", "head_parameters", "resume", "timing", "budget", "ok")}, indent=1, default=str), flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- queue
def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"])
    for i, key in enumerate(keys):
        arm, s = key.rsplit("_s", 1); s = int(s); spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * ctx.total * spu / 3600 + 0.5
        print(f"[queue] before {key}: spent {spent:.3f}, projection {proj:.3f} of {ctx.vs['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vs["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, s, device, workers); ok = True; note = json.dumps(r)
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
