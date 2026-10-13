#!/usr/bin/env python
"""Prepare the video feature temporal round (docs/superpowers/plans/2026-10-12-video-feature-temporal-training.md, Task 1;
configs/experiments/video_feature_temporal_v1.yaml). Training split only; no development data is read here.

1. Sources: every pin, the frozen Ordered models' identities, the environment and the two official commits.
2. Training data: the E1 index read as video_id, split and n_samples, then label_index of the training rows only; the
   3,364 protocol clips; the 1,500-stream bank (recipe digest checked); every sequence's 10 Hz references from the numeric
   frame indices (streams assembled from their recipes first, checked against windowed.stream_plan).
3. Plans: per seed, 50 epochs x 375 updates of the float32 batch rule, every update's structure digest compared with the
   paired ASRF plan; ASRF class weights and pos_weight from the epoch-0 labels; the initial states of the three heads.
4. Timing: windows of the first training sequences in list order through the three source models; the projection of all
   training windows x three models against the budget. Above the cap nothing is extracted.
5. Features: every training sequence encoded whole (stride-5 windows, model.embed, nearest centre), float32, with the
   parity of the 16 check sequences (source classifier on the stored features against the source forward) and the source
   states' digests before and after. Writes the features, the sequence file (labels and joins) and preparation.json.

  python scripts/prepare_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.train_joint_visual_temporal as tjv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_l1_asrf import batch_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.features.cache import variant_dir  # noqa: E402
from echo_routing.temporal.dev_heads import ASRFHead, MSTCNHead, asrf_weights, check_pinned  # noqa: E402
from echo_routing.temporal.dev_run import digest  # noqa: E402
from echo_routing.temporal.dev_train import seed_everything  # noqa: E402
from echo_routing.temporal.windowed import stream_plan  # noqa: E402

HEADS = ("linear", "mstcn", "asrf")
CHUNK_SAMPLES = 20000            # sequences are decoded and encoded in chunks of about this many samples
FEATURE_FILES = (Path(vsf.__file__), Path(tm.__file__), Path(rr.__file__), Path(prv.__file__), Path(tjv.__file__))   # whole files behind a feature


# ----------------------------------------------------------------------------------------------- helpers
def pinned(root: Path, pin: dict) -> Path:
    path = Path(root) / pin["path"]
    if not path.is_file():
        raise SystemExit(f"{pin['path']}: missing")
    if (got := sha256_file(path)) != pin["sha256"]:
        raise SystemExit(f"{pin['path']}: sha256 {got[:12]} differs from the pin {pin['sha256'][:12]}")
    return path


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_digest(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode()); h.update(Path(p).read_bytes())
    return h.hexdigest()


def write_atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str)); tmp.replace(path)


def setup_gpu() -> torch.device:
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True, warn_only=True)
    return device


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """The round's configuration and the source round's (read only); outputs only under this round's root."""

    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vt = cfg, config, cfg["video_feature_temporal"]
        self.paths = load_paths(); self.root = self.paths.repo_root; src = self.vt["source"]
        pinned(self.root, self.vt["task_sheet"])
        self.rv = load_config(pinned(self.root, src["config"]))["raw_video_context"]
        if (src["protocol"] != self.rv["source"]["protocol"] or src["echoprime_checkpoint"] != self.rv["source"]["echoprime"]["checkpoint"]
                or src["e1_feature_manifest"] != self.rv["evaluation"]["reference_inputs"]["e1_feature_manifest"]):
            raise SystemExit("a source pin differs from the source configuration's")
        self.seeds = [int(s) for s in self.vt["data"]["seeds"]]
        self.out = self.root / self.vt["outputs"]["root"]; self.results = self.root / self.vt["outputs"]["results_dir"]
        d = self.vt["data"]
        if int(d["epochs"]) * int(d["updates_per_epoch"]) != int(d["total_updates"]):
            raise SystemExit("epochs x updates_per_epoch differs from total_updates")

    def verify_sources(self) -> dict:
        src = self.vt["source"]
        for pin in (*self.vt["stop_record"].values(), *src["report"].values(), *(src[k] for k in ("protocol", "e1_feature_manifest", "frame_manifest",
                                                                                                    "window_plans", "echoprime_checkpoint"))):
            if isinstance(pin, dict) and "path" in pin:
                pinned(self.root, pin)
        for s in self.seeds:
            m = src["models"][s]
            for k in ("final", "init", "run_manifest"):
                pinned(self.root, m[k])
            man = json.loads((self.root / m["run_manifest"]["path"]).read_text())
            if man["identity"] != m["identity"] or man["final_sha256"] != m["final"]["sha256"]:
                raise SystemExit(f"ordered_s{s}: the frozen run manifest does not carry the pinned identity and weights")
            for k in ("plans", "inits"):
                pinned(self.root, src["paired_asrf"][k][s])
        for k in ("config", "summary"):
            pinned(self.root, src["paired_asrf"][k])
        for h in ("mstcn", "asrf"):
            check_pinned(self.root / self.vt["heads"][h]["repo"], self.vt["heads"][h]["commit"])
        fman = json.loads((self.root / src["e1_feature_manifest"]["path"]).read_text())
        if fman["encoder"]["cache_key"] != src["e1_cache_key"]:
            raise SystemExit("the E1 cache key differs from the config")
        env = prv.environment(); want = src["environment"]
        if any(str(env[k]) != str(want[k]) for k in want):
            raise SystemExit(f"environment {env} differs from the pinned {want}")
        return env


def model_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_source(ctx: Ctx, seed: int, device) -> tm.TargetMViT:
    """The frozen Ordered model as the source evaluation built it, in eval mode with no trainable parameter."""
    m = ctx.vt["source"]["models"][seed]
    init = torch.load(pinned(ctx.root, m["init"]), map_location="cpu", weights_only=True)
    model = tm.build_model(pinned(ctx.root, ctx.vt["source"]["echoprime_checkpoint"]), init)
    model.load_state_dict(torch.load(pinned(ctx.root, m["final"]), map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------------------------- training data
class TrainingData:
    """The training split only: clips, labels of training rows, the bank and every sequence's references."""

    def __init__(self, ctx: Ctx) -> None:
        src = ctx.vt["source"]; self.proto = json.loads(pinned(ctx.root, src["protocol"]).read_text())
        self.dir = variant_dir(ctx.paths.cache_root, src["e1_cache_key"], "orig")
        idx = vsf.training_index(self.dir / "index.csv")
        self.clips = list(self.proto["train"]["clips"])
        if set(idx["video_id"]) != set(self.clips) or len(idx) != len(self.clips):
            raise SystemExit("the index's training rows differ from the protocol's training clips")
        self.label = dict(zip(idx["video_id"], idx["label_index"].astype(int))); self.n = dict(zip(idx["video_id"], idx["n_samples"].astype(int)))
        bank = multi_bank(idx, "train", ctx.cfg); self.bank_digest = digest([recipe_to_dict(r) for r in bank.recipes])
        if self.bank_digest != self.proto["train"]["multi_bank"]["digest"] or len(bank.recipes) != 1500:
            raise SystemExit("the rebuilt training bank differs from the protocol's")
        self.recipes = bank.recipes; self._fidx: dict[str, np.ndarray] = {}
        if not {f.video_id for r in self.recipes for f in r.fragments} <= set(self.clips):
            raise SystemExit("a bank fragment is not a protocol training clip")
        self.seqs = []
        for v in self.clips:
            f = self.frame_idx(v)
            self.seqs.append({"id": v, "kind": "native", "refs": vsf.native_refs(v, f), "labels": np.full(len(f), self.label[v], np.int64),
                              "joins": np.zeros(0, np.int64)})
        for r in self.recipes:
            labels, joins = vsf.stream_labels_joins(r.fragments)
            refs = vsf.stream_refs(r.fragments, self.frame_idx)
            plan = [(x.video_id, int(x.frame_idx), float(x.gamma)) for x in stream_plan(r, self._loader)]
            if plan != refs or len(refs) != labels.size:
                raise SystemExit(f"{r.recipe_id}: the assembled references differ from windowed.stream_plan")
            self.seqs.append({"id": r.recipe_id, "kind": "bank", "refs": refs, "labels": labels, "joins": joins})

    def frame_idx(self, vid: str) -> np.ndarray:
        if vid not in self._fidx:
            with np.load(self.dir / f"{vid}.npz") as z:
                f = z["frame_idx"].astype(np.int64)
            if f.size != self.n[vid]:
                raise SystemExit(f"{vid}: {f.size} frame indices, the index says {self.n[vid]}")
            self._fidx[vid] = f
        return self._fidx[vid]

    def _loader(self, vid: str, variant: str) -> SimpleNamespace:
        return SimpleNamespace(frame_idx=self.frame_idx(vid))

    def check_ids(self) -> list[int]:
        """The 16 check sequences: the first 8 training clips and the first 8 bank streams, in list order."""
        nat = [i for i, q in enumerate(self.seqs) if q["kind"] == "native"][:8]; bank = [i for i, q in enumerate(self.seqs) if q["kind"] == "bank"][:8]
        return nat + bank

    def ids(self) -> set[str]:
        return set(self.clips) | {session_of(v) for v in self.clips} | {r.recipe_id for r in self.recipes}


def placeholder_batches(data: TrainingData) -> vsf.Batches:
    """The batch structure without features (zero rows): ids, lengths, labels and joins only, for plans."""
    streams = [vsf.Seq(q["id"], False, np.zeros((q["labels"].size, 1), np.float32), q["labels"], q["joins"]) for q in data.seqs if q["kind"] == "bank"]
    native = {v: np.zeros((data.n[v], 1), np.float32) for v in data.clips}
    return vsf.Batches(streams, [(v, data.label[v], data.n[v]) for v in data.clips], native)


def make_plans(ctx: Ctx, data: TrainingData) -> dict:
    d = ctx.vt["data"]; b = placeholder_batches(data); out = {}
    for s in ctx.seeds:
        epochs, first = [], None
        for e in range(int(d["epochs"])):
            bs = b.epoch(s, e, int(d["native_per_epoch"]), int(d["native_window"]), int(d["streams_per_batch"]), int(d["natives_per_batch"]))
            if len(bs) != int(d["updates_per_epoch"]) or any(len(x) != int(d["streams_per_batch"]) + int(d["natives_per_batch"]) for x in bs):
                raise SystemExit(f"seed {s} epoch {e}: not {d['updates_per_epoch']} full batches")
            epochs.append([batch_digest(x) for x in bs]); first = first if e else [q for x in bs for q in x]
        cw, pw = asrf_weights(first, num_classes=5)
        old = json.loads(pinned(ctx.root, ctx.vt["source"]["paired_asrf"]["plans"][s]).read_text())
        same = sum(a == b for ea, eb in zip(epochs, old["epochs"]) for a, b in zip(ea, eb))
        plan = {"seed": s, "epochs": epochs, "class_weight": cw.tolist(), "pos_weight": float(pw),
                "paired_plan": {"updates_compared": int(sum(len(x) for x in epochs)), "updates_equal": int(same),
                                "class_weight_equal": old["asrf_class_weight"] == cw.tolist(), "pos_weight_equal": float(old["asrf_pos_weight"]) == float(pw)}}
        if plan["paired_plan"]["updates_equal"] != plan["paired_plan"]["updates_compared"]:
            raise SystemExit(f"seed {s}: {plan['paired_plan']} - the batches differ from the paired ASRF plan")
        out[s] = plan
    return out


# ----------------------------------------------------------------------------------------------- initial states
def source_classifier(ctx: Ctx, seed: int) -> dict:
    st = torch.load(pinned(ctx.root, ctx.vt["source"]["models"][seed]["final"]), map_location="cpu", weights_only=True)
    return {"weight": st["classifier.weight"].clone(), "bias": st["classifier.bias"].clone()}


def make_heads(ctx: Ctx, plan: dict, device: str = "cpu") -> dict:
    h = ctx.vt["heads"]; m, a = h["mstcn"]["model"], h["asrf"]["model"]
    mstcn = MSTCNHead(ctx.root / h["mstcn"]["repo"], int(m["in_dim"]), num_classes=int(m["num_classes"]), num_stages=int(m["num_stages"]),
                      num_layers=int(m["num_layers"]), num_f_maps=int(m["num_f_maps"]))
    asrf = ASRFHead(ctx.root / h["asrf"]["repo"], int(a["in_channel"]), torch.tensor(plan["class_weight"], dtype=torch.float32),
                    torch.tensor(plan["pos_weight"]), num_classes=int(a["n_classes"]), lambda_b=float(h["asrf"]["lambda_b"]),
                    boundary_th=float(h["asrf"]["boundary_th"]), device=device)
    return {"mstcn": mstcn, "asrf": asrf}


def make_inits(ctx: Ctx, plans: dict) -> dict:
    out = {}
    for s in ctx.seeds:
        heads = make_heads(ctx, plans[s]); states = {"linear": source_classifier(ctx, s)}
        for k in ("mstcn", "asrf"):
            seed_everything(s); states[k] = {n: v.detach().cpu().clone() for n, v in heads[k].build().state_dict().items()}
        old = torch.load(pinned(ctx.root, ctx.vt["source"]["paired_asrf"]["inits"][s]), map_location="cpu", weights_only=True)
        asrf_equal = set(old) == set(states["asrf"]) and all(torch.equal(old[n], states["asrf"][n]) for n in old)
        if asrf_equal:
            states["asrf"] = old
        for k in HEADS:
            path = ctx.out / "inits" / f"{k}_s{s}.pt"
            if path.is_file():
                prev = torch.load(path, map_location="cpu", weights_only=True)
                if set(prev) != set(states[k]) or any(not torch.equal(prev[n], states[k][n]) for n in prev):
                    raise SystemExit(f"{path} exists with other content")
            else:
                path.parent.mkdir(parents=True, exist_ok=True); torch.save(states[k], path)
            out[f"{k}_s{s}"] = {"path": str(path.relative_to(ctx.root)), "sha256": sha256_file(path),
                                "parameters": int(sum(v.numel() for v in states[k].values()))}
        out[f"asrf_s{s}"]["equals_paired_asrf_init"] = bool(asrf_equal)
    return out


# ----------------------------------------------------------------------------------------------- features
@torch.inference_mode()
def encode(model, idx: np.ndarray, markers: np.ndarray, frames: np.ndarray, device, batch: int = 32, logits: bool = False) -> tuple:
    """Window features (and, for the parity check, the source forward's logits of the same tensors)."""
    feats, lg = [], []
    for b in range(0, len(idx), batch):
        u8 = torch.from_numpy(frames[idx[b:b + batch]]).to(device, non_blocking=True)
        rgb, mk = tm.to_rgb(u8), tm.marker_planes(torch.from_numpy(markers[b:b + batch]).long().to(device))
        feats.append(model.embed(rgb, mk).float().cpu().numpy())
        if logits:
            lg.append(model(rgb, mk).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, tm.EMBED_DIM), np.float32), (np.concatenate(lg) if lg else None)


def chunks(seqs: list[dict], limit: int = CHUNK_SAMPLES) -> list[list[int]]:
    out, cur, size = [], [], 0
    for i, q in enumerate(seqs):
        cur.append(i); size += len(q["refs"])
        if size >= limit:
            out.append(cur); cur, size = [], 0
    return out + ([cur] if cur else [])


def frame_paths(ctx: Ctx, vids: set[str]):
    return tjv.FramePaths(prv.frame_dirs(ctx.paths, ctx.rv, vids))


def feature_identity(ctx: Ctx, seed: int, seqs: list[dict]) -> str:
    h = hashlib.sha256()
    for q in seqs:
        h.update(q["id"].encode()); h.update(vsf.refs_digest(q["refs"]).encode())
    code = file_digest(FEATURE_FILES) + "".join(inspect.getsource(fn) for fn in (build_source, encode, extract, chunks, frame_paths, save_features))
    payload = {"model": ctx.vt["source"]["models"][seed]["final"]["sha256"], "references": h.hexdigest(), "features": ctx.vt["features"],
               "environment": prv.environment(), "code": hashlib.sha256(code.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def time_extraction(ctx: Ctx, data: TrainingData, models: dict, device, workers: int, n_windows: int = 320) -> dict:
    """The first training sequences in list order: decode, then one warm-up batch and the timed windows per model."""
    take, total = [], 0
    for i, q in enumerate(data.seqs):
        take.append(i); total += len(vsf.windows(len(q["refs"]))["centres"])
        if total >= n_windows + 32:
            break
    refs = sorted({r for i in take for r in data.seqs[i]["refs"]}); at = {r: j for j, r in enumerate(refs)}
    fp = frame_paths(ctx, {r[0] for r in refs}); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); t_dec = time.time() - t0
    idx, mk = [], []
    for i in take:
        q = data.seqs[i]; w = vsf.windows(len(q["refs"]))
        idx += [[at[q["refs"][j]] for j in s] for s in w["samples"]]; mk += w["markers"].tolist()
    idx, mk = np.asarray(idx), np.asarray(mk); per = {}
    for s, model in models.items():
        encode(model, idx[:32], mk[:32], frames, device); torch.cuda.synchronize(); t1 = time.time()
        encode(model, idx[32:32 + n_windows], mk[32:32 + n_windows], frames, device); torch.cuda.synchronize()
        per[s] = (time.time() - t1) / n_windows
    return {"windows_timed": n_windows, "seconds_per_window": per, "decode_seconds_per_frame": t_dec / max(1, len(refs)), "frames_timed": len(refs)}


def extract(ctx: Ctx, data: TrainingData, models: dict, device, workers: int) -> dict:
    """Every training sequence whole through the three source models; float32 window and sample features per seed."""
    seqs = data.seqs; ids = [q["id"] for q in seqs]; n = np.array([len(q["refs"]) for q in seqs])
    wins = [vsf.windows(int(k)) for k in n]; nwin = np.array([len(w["centres"]) for w in wins])
    s_off = np.concatenate([[0], np.cumsum(n)]); w_off = np.concatenate([[0], np.cumsum(nwin)])
    wfeat = {s: np.zeros((int(w_off[-1]), tm.EMBED_DIM), np.float32) for s in models}
    check = set(data.check_ids()); parity = {s: {"windows": 0, "max_abs": 0.0, "argmax_equal": True} for s in models}
    check_logits = {s: {} for s in models}; t_dec = t_enc = 0.0
    for ci, ch in enumerate(chunks(seqs)):
        refs = sorted({r for i in ch for r in seqs[i]["refs"]}); at = {r: j for j, r in enumerate(refs)}
        t0 = time.time(); frames = rr.decode_frames(refs, frame_paths(ctx, {r[0] for r in refs}), workers); t_dec += time.time() - t0
        idx = np.asarray([[at[seqs[i]["refs"][j]] for j in s] for i in ch for s in wins[i]["samples"]], np.int64).reshape(-1, vsf.FRAMES)
        mk = np.concatenate([wins[i]["markers"] for i in ch]); rows = np.concatenate([np.arange(w_off[i], w_off[i + 1]) for i in ch])
        t1 = time.time()
        for s, model in models.items():
            f, _ = encode(model, idx, mk, frames, device); wfeat[s][rows] = f
            for i in ch:
                if i in check:
                    r = np.arange(w_off[i], w_off[i + 1]); sel = np.searchsorted(rows, r)
                    _, lg = encode(model, idx[sel], mk[sel], frames, device, logits=True); check_logits[s][i] = lg
        t_enc += time.time() - t1; del frames
        print(f"chunk {ci}: {len(ch)} sequences, {len(idx)} windows ({time.time() - t0:.0f} s)", flush=True)
    out = {}
    for s, model in models.items():
        W, b = model.classifier.weight.detach().cpu().double().numpy(), model.classifier.bias.detach().cpu().double().numpy()
        for i, lg in check_logits[s].items():
            mine = wfeat[s][w_off[i]:w_off[i + 1]].astype(np.float64) @ W.T + b
            parity[s]["windows"] += len(lg); parity[s]["max_abs"] = max(parity[s]["max_abs"], float(np.abs(mine - lg).max()))
            parity[s]["argmax_equal"] &= bool(np.array_equal(mine.argmax(1), lg.argmax(1)))
        samples = np.concatenate([vsf.map_to_samples(int(n[i]), wins[i]["centres"], wfeat[s][w_off[i]:w_off[i + 1]]) for i in range(len(seqs))])
        norms = np.linalg.norm(wfeat[s], axis=1)
        out[s] = {"window": wfeat[s], "samples": samples, "norms": {"mean": float(norms.mean()), "std": float(norms.std()),
                                                                    "p01": float(np.percentile(norms, 1)), "p50": float(np.median(norms)), "p99": float(np.percentile(norms, 99))}}
    meta = {"ids": ids, "n": n, "s_off": s_off, "w_off": w_off, "centres": np.concatenate([w["centres"] for w in wins]),
            "markers": np.concatenate([w["markers"] for w in wins])}
    return {"features": out, "meta": meta, "parity": parity, "seconds": {"decode": t_dec, "encode": t_enc}}


def save_features(ctx: Ctx, split: str, seed: int, ident: str, meta: dict, feats: dict) -> Path:
    path = ctx.out / "features" / split / f"ordered_s{seed}.npz"; path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, identity=ident, ids=np.asarray(meta["ids"]), n=meta["n"], sample_offsets=meta["s_off"], window_offsets=meta["w_off"],
                 centres=meta["centres"], markers=meta["markers"], window=feats["window"], samples=feats["samples"])
    tmp.replace(path)
    return path


def save_sequences(ctx: Ctx, split: str, data: TrainingData) -> Path:
    """Labels, joins and kinds in their own file; the feature files hold no label."""
    path = ctx.out / "features" / split / "sequences.npz"; path.parent.mkdir(parents=True, exist_ok=True); seqs = data.seqs
    joins = [q["joins"] for q in seqs]; tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, ids=np.asarray([q["id"] for q in seqs]), kinds=np.asarray([q["kind"] for q in seqs]), n=np.array([len(q["refs"]) for q in seqs]),
                 labels=np.concatenate([q["labels"] for q in seqs]), join_offsets=np.concatenate([[0], np.cumsum([j.size for j in joins])]),
                 joins=np.concatenate(joins) if any(j.size for j in joins) else np.zeros(0, np.int64),
                 refs_digest=np.asarray([vsf.refs_digest(q["refs"]) for q in seqs]))
    tmp.replace(path)
    return path


def ledger(ctx: Ctx, job: str, started: str, seconds: float, ok: bool, note: str = "") -> None:
    p = ctx.out / "ledger.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"job": job, "started": started, "finished": now(), "hours": seconds / 3600, "ok": bool(ok), "note": note}) + "\n")


def ledger_hours(ctx: Ctx) -> float:
    p = ctx.out / "ledger.jsonl"
    return sum(float(json.loads(x)["hours"]) for x in p.read_text().splitlines() if x.strip()) if p.is_file() else 0.0


# ----------------------------------------------------------------------------------------------- main
def run(ctx: Ctx, workers: int) -> None:
    t0, started, ok, rec = time.time(), now(), False, {}
    try:
        env = ctx.verify_sources(); data = TrainingData(ctx); ids = data.ids()
        plans = make_plans(ctx, data)
        for s, p in plans.items():
            path = ctx.out / "plans" / f"plan_s{s}.json"
            if path.is_file() and json.loads(path.read_text()) != json.loads(json.dumps(p)):
                raise SystemExit(f"{path} exists with other content")
            write_atomic_json(path, p)
        inits = make_inits(ctx, plans); print("plans and initial states written", flush=True)
        device = setup_gpu(); models = {s: build_source(ctx, s, device) for s in ctx.seeds}; before = {s: model_digest(m) for s, m in models.items()}
        timing = time_extraction(ctx, data, models, device, workers)
        n_windows = int(sum(len(vsf.windows(len(q["refs"]))["centres"]) for q in data.seqs)); n_frames = len({r for q in data.seqs for r in q["refs"]})
        b = ctx.vt["budget"]
        proj = {"spent": ledger_hours(ctx) + (time.time() - t0) / 3600, "training_features": (n_windows * sum(timing["seconds_per_window"].values())
                + n_frames * timing["decode_seconds_per_frame"]) / 3600, "reserve": float(b["reserve_hours"])}
        proj["total_before_training"] = sum(proj.values()); proj["cap"] = float(b["gpu_hours_max"]); proj["left_for_checks_and_training"] = proj["cap"] - proj["total_before_training"]
        rec.update({"started": started, "environment": env, "bank_digest": data.bank_digest, "sequences": {"natives": len(data.clips), "bank_streams": len(data.recipes)},
                    "samples": int(sum(len(q["refs"]) for q in data.seqs)), "windows_per_model": n_windows, "unique_frames": n_frames,
                    "plans": {s: {"sha256": sha256_file(ctx.out / "plans" / f"plan_s{s}.json"), **p["paired_plan"], "class_weight": p["class_weight"],
                                  "pos_weight": p["pos_weight"]} for s, p in plans.items()}, "inits": inits, "timing": timing, "projection": proj})
        print(json.dumps({"timing": timing, "projection": proj}, indent=1), flush=True)
        if proj["left_for_checks_and_training"] <= 0:
            raise SystemExit("the projected feature extraction leaves no budget for the checks and the nine trainings")
        idents = {s: feature_identity(ctx, s, data.seqs) for s in ctx.seeds}
        done = all((ctx.out / "features" / "train" / f"ordered_s{s}.npz").is_file() for s in ctx.seeds) and all(
            str(np.load(ctx.out / "features" / "train" / f"ordered_s{s}.npz")["identity"]) == idents[s] for s in ctx.seeds)
        if done:
            rec["features"] = "reused: the saved training features carry this identity"
        else:
            ex = extract(ctx, data, models, device, workers)
            bad = {s: p for s, p in ex["parity"].items() if p["max_abs"] > 1e-4 or not p["argmax_equal"]}
            if bad:   # before anything is saved, so a failed parity never leaves a cache that a rerun would reuse
                raise SystemExit(f"source parity failed: {bad}")
            for s in ctx.seeds:
                save_features(ctx, "train", s, idents[s], ex["meta"], ex["features"][s])
            rec.update({"parity": ex["parity"], "norms": {s: ex["features"][s]["norms"] for s in ctx.seeds}, "extraction_seconds": ex["seconds"],
                        "features": {s: {"path": f"{ctx.vt['outputs']['root']}/features/train/ordered_s{s}.npz", "identity": idents[s]} for s in ctx.seeds}})
        seq_path = save_sequences(ctx, "train", data); rec["sequence_file"] = {"path": str(seq_path.relative_to(ctx.root)), "sha256": sha256_file(seq_path)}
        after = {s: model_digest(m) for s, m in models.items()}
        rec["source_state_unchanged"] = {s: before[s] == after[s] for s in ctx.seeds}
        if not all(rec["source_state_unchanged"].values()):
            raise SystemExit("a source model's state changed")
        rec["finished"] = now(); ok = True
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for d in (ctx.out, ctx.results):
            write_atomic_json(d / "preparation.json", flat)
        print(json.dumps({k: rec[k] for k in ("parity", "source_state_unchanged") if k in rec}, indent=1, default=str), flush=True)
    finally:
        ledger(ctx, "prepare", started, time.time() - t0, ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; run(Ctx(load_cfg(a), a.config), a.workers)


if __name__ == "__main__":
    main()
