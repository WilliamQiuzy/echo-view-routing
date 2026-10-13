#!/usr/bin/env python
"""Train one head and seed of the video feature temporal round, or run the checks before the queue
(docs/superpowers/plans/2026-10-12-video-feature-temporal-training.md, Task 2; configs/experiments/video_feature_temporal_v1.yaml).

The heads read only the seed's frozen Ordered features (float32, (T, 512)) and the valid-length mask. Linear is a copy of
the source classifier trained with masked weighted cross-entropy; MS-TCN and ASRF are the official models through their
adapters (dev_heads). Every update of a seed is the same batch for the three heads (the float32 batch rule; every epoch's
structure digests equal the seed's plan). AdamW lr 3e-4, weight decay 1e-4, clip 1; 750 warm-up updates, then cosine to
0; 50 epochs x 375 updates; the final weights. No development data is read.

  --check              the batch, label, padding and inference checks, the 100-update smoke of every head, the 6-update
                       reproducibility and the 3 + 3 resume, the timing of every head and the budget projection
  --head H --seed S    one formal run with resume state and a training-only monitor after every epoch

  python scripts/train_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml --check
  python scripts/train_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml --head asrf --seed 0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_l1_asrf as tla  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.temporal.dev_data as dd  # noqa: E402
import echo_routing.temporal.dev_heads as dh  # noqa: E402
import echo_routing.temporal.dev_train as dtr  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HEADS = ("linear", "mstcn", "asrf")
TRAINING_KEYS = ("data", "heads", "training", "features")
CODE_FILES = (Path(__file__), Path(dh.__file__), Path(dd.__file__), Path(dtr.__file__), Path(vsf.__file__), Path(tla.__file__))


# ----------------------------------------------------------------------------------------------- the linear head
class LinearHead:
    """Linear(512, 5) at every position of (B, 512, T); masked cross-entropy with the seed's ASRF class weights."""

    name = "linear"

    def __init__(self, class_weight: torch.Tensor, in_dim: int = 512, num_classes: int = 5, device: str | torch.device = "cpu") -> None:
        self.class_weight, self.in_dim, self.num_classes = class_weight.to(device), int(in_dim), int(num_classes)

    def build(self) -> torch.nn.Module:
        return torch.nn.Linear(self.in_dim, self.num_classes)

    def loss(self, model: torch.nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        lg = model(batch["x"].transpose(1, 2))                                   # (B, T, C)
        ce = F.cross_entropy(lg.reshape(-1, self.num_classes), batch["labels"].reshape(-1), weight=self.class_weight, ignore_index=dd.PAD_LABEL)
        return ce, {"ce": float(ce.detach())}

    @torch.no_grad()
    def decode(self, model: torch.nn.Module, seqs: list, device, batch_size: int = 32) -> dict[str, dict[str, np.ndarray]]:
        model.eval(); out = {}
        for s in seqs:
            lg = model(torch.from_numpy(np.ascontiguousarray(s.feat, np.float32)).to(device))
            prob = torch.softmax(lg, -1).cpu().numpy()
            out[s.seq_id] = {"labels": prob.argmax(1), "prob": prob}
        return out


# ----------------------------------------------------------------------------------------------- context and data
class Ctx(pv.Ctx):
    """The round's configuration with the preparation record; outputs only under this round's root."""

    def __init__(self, cfg: dict, config: str) -> None:
        super().__init__(cfg, config)
        self.prep = json.loads((self.out / "preparation.json").read_text())
        self.ckpt = self.root / self.vt["outputs"]["checkpoints"]; d = self.vt["data"]
        self.total, self.per_epoch = int(d["total_updates"]), int(d["updates_per_epoch"])

    def plan(self, seed: int) -> tuple[dict, str]:
        path = self.out / "plans" / f"plan_s{seed}.json"; sha = sha256_file(path)
        if sha != self.prep["plans"][str(seed)]["sha256"]:
            raise SystemExit(f"{path} differs from the preparation record")
        return json.loads(path.read_text()), sha

    def init_state(self, head: str, seed: int) -> tuple[dict, str]:
        rec = self.prep["inits"][f"{head}_s{seed}"]; path = self.root / rec["path"]; sha = sha256_file(path)
        if sha != rec["sha256"]:
            raise SystemExit(f"{path} differs from the preparation record")
        return torch.load(path, map_location="cpu", weights_only=True), sha


class TrainSet:
    """One seed's training sequences: float32 features of the seed's encoder, labels and joins from their own file."""

    def __init__(self, ctx: Ctx, seed: int) -> None:
        sq = np.load(ctx.out / "features" / "train" / "sequences.npz"); fz = np.load(ctx.out / "features" / "train" / f"ordered_s{seed}.npz")
        if str(fz["identity"]) != ctx.prep["features"][str(seed)]["identity"]:
            raise SystemExit(f"seed {seed}: the training features differ from the preparation record")
        if sha256_file(ctx.out / "features" / "train" / "sequences.npz") != ctx.prep["sequence_file"]["sha256"]:
            raise SystemExit("the sequence file differs from the preparation record")
        ids, kinds, n = sq["ids"].tolist(), sq["kinds"].tolist(), sq["n"]
        if fz["ids"].tolist() != ids or not np.array_equal(fz["n"], n):
            raise SystemExit("the feature file and the sequence file list different sequences")
        self.feature_identity = str(fz["identity"]); samples = fz["samples"]
        if samples.dtype != np.float32:
            raise SystemExit("the training features are not float32")
        off, labels, joff, joins = np.concatenate([[0], np.cumsum(n)]), sq["labels"], sq["join_offsets"], sq["joins"]
        self.streams, self.clips, native = [], [], {}
        for i, (sid, kind) in enumerate(zip(ids, kinds)):
            f = samples[off[i]:off[i + 1]]; lab = labels[off[i]:off[i + 1]]
            if kind == "bank":
                self.streams.append(vsf.Seq(sid, False, f, lab, joins[joff[i]:joff[i + 1]]))
            else:
                native[sid] = f; self.clips.append((sid, int(lab[0]), int(n[i])))
        self.batches = vsf.Batches(self.streams, self.clips, native); self.seed = int(seed)
        check = [i for i, k in enumerate(kinds) if k == "native"][:8] + [i for i, k in enumerate(kinds) if k == "bank"][:8]
        self.check = [self.batches.crop(ids[i], int(labels[off[i]]), 0, min(256, int(n[i]))) if kinds[i] == "native" else self.streams[[s.seq_id for s in self.streams].index(ids[i])]
                      for i in check]

    def epoch(self, ctx: Ctx, epoch: int) -> list[list]:
        d = ctx.vt["data"]
        return self.batches.epoch(self.seed, epoch, int(d["native_per_epoch"]), int(d["native_window"]), int(d["streams_per_batch"]), int(d["natives_per_batch"]))


# ----------------------------------------------------------------------------------------------- heads, optimiser, state
def make_head(ctx: Ctx, head: str, plan: dict, device):
    if head == "linear":
        return LinearHead(torch.tensor(plan["class_weight"], dtype=torch.float32), device=device)
    h = pv.make_heads(ctx, plan, device)[head]
    if h.commit != ctx.vt["heads"][head]["commit"]:
        raise SystemExit(f"{head} commit {h.commit} != {ctx.vt['heads'][head]['commit']}")
    return h


def new_model(head, state: dict, device) -> torch.nn.Module:
    model = head.build(); model.load_state_dict(state, strict=True)
    return model.to(device)


def make_optimiser(ctx: Ctx, model: torch.nn.Module) -> tuple:
    o, s = ctx.vt["training"]["optimiser"], ctx.vt["training"]["schedule"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]),
                            weight_decay=float(o["weight_decay"]))
    warm = int(s["warmup_updates"])
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda c: dtr.lr_factor(c, warm, ctx.total)), float(o["grad_clip"])


def determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)


def normalise(batch: dict) -> dict:
    """By the valid mask, before any loss: padding labels become the ignore value and padding features 0."""
    out = dict(batch)
    out["labels"] = batch["labels"].masked_fill(batch["valid"] <= 0, dd.PAD_LABEL); out["x"] = batch["x"] * batch["mask"]
    return out


def update(head, model, seqs: list, opt, sched, clip: float, device) -> dict:
    lr = float(opt.param_groups[0]["lr"]); model.train(); loss, parts = head.loss(model, normalise(dd.collate(seqs, device)))
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {parts}")
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step(); sched.step()
    return {"total": float(loss.detach()), **parts, "grad_norm": gn, "lr": lr}


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if st["cuda"]:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_resume(path: Path, model, opt, sched, completed: int, ident: str, records: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "completed_updates": completed, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": rng_state(), "records": records}, tmp)
    tmp.replace(path)


def load_resume(path: Path, model, opt, sched, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); set_rng_state(st["rng"])
    return int(st["completed_updates"]), list(st["records"])


def identity(ctx: Ctx, head: str, seed: int, feature_identity: str, plan_sha: str, init_sha: str, commit: str | None) -> str:
    code = pv.file_digest(CODE_FILES) + "".join(inspect.getsource(fn) for fn in (pv.make_heads, pv.source_classifier))
    payload = {"experiment": ctx.cfg.get("experiment"), "head": head, "seed": int(seed), "config": {k: ctx.vt[k] for k in TRAINING_KEYS},
               "features": feature_identity, "sequence_file": ctx.prep["sequence_file"]["sha256"], "plan": plan_sha, "init": init_sha,
               "official_commit": commit, "total_updates": ctx.total, "code": hashlib.sha256(code.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_key(head: str, seed: int) -> str:
    return f"{head}_s{seed}"


def finished(man_path: Path, ident: str) -> bool:
    if not man_path.is_file():
        return False
    man = json.loads(man_path.read_text())
    return man.get("identity") == ident and int(man.get("updates", 0)) == int(man.get("total_updates", -1)) and Path(man["final_checkpoint"]).is_file() \
        and sha256_file(Path(man["final_checkpoint"])) == man["final_sha256"]


@torch.no_grad()
def monitor(head, model, seqs: list, device) -> dict:
    """Training fit on the 16 check sequences (accuracy of the head's own output); never development data."""
    out = head.decode(model, seqs, device); acc = [float(np.mean(out[s.seq_id]["labels"] == s.labels)) for s in seqs]
    return {"check_accuracy_mean": float(np.mean(acc)), "check_accuracy_min": float(np.min(acc))}


# ----------------------------------------------------------------------------------------------- one formal run
def train(ctx: Ctx, head_name: str, seed: int, device) -> None:
    data = TrainSet(ctx, seed); plan, plan_sha = ctx.plan(seed); state, init_sha = ctx.init_state(head_name, seed)
    head = make_head(ctx, head_name, plan, device); ident = identity(ctx, head_name, seed, data.feature_identity, plan_sha, init_sha, getattr(head, "commit", None))
    key = run_key(head_name, seed); run_dir = ctx.out / "train" / key; ck = ctx.ckpt / key; man_path = run_dir / "run_manifest.json"
    if finished(man_path, ident):
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    run_dir.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    model = new_model(head, state, device); opt, sched, clip = make_optimiser(ctx, model); completed, records, resumed = 0, [], []
    seed_first = True
    if (ck / "resume.pt").is_file():
        completed, records = load_resume(ck / "resume.pt", model, opt, sched, ident); resumed.append(completed); seed_first = False
        print(f"{key}: resumed after {completed} updates", flush=True)
    if seed_first:
        dtr.seed_everything(seed)
    log = run_dir / "train_log.jsonl"
    if completed == 0:
        log.write_text("")
    for epoch in range(completed // ctx.per_epoch, int(ctx.vt["data"]["epochs"])):
        t0 = time.time(); bs = data.epoch(ctx, epoch); sums = defaultdict(float)
        if [tla.batch_digest(x) for x in bs] != plan["epochs"][epoch]:
            raise SystemExit(f"{key} epoch {epoch}: the batches differ from the seed's plan")
        for x in bs:
            for k, v in update(head, model, x, opt, sched, clip, device).items():
                sums[k] += v
        completed += len(bs); t_train = time.time() - t0
        rec = {"epoch": epoch, "updates": completed, "loss": {k: v / len(bs) for k, v in sums.items() if k not in ("lr", "grad_norm")},
               "grad_norm_mean": sums["grad_norm"] / len(bs), "lr_last": float(opt.param_groups[0]["lr"]), "seconds_train": t_train,
               **monitor(head, model, data.check, device)}
        model.train(); records.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        save_resume(ck / "resume.pt", model, opt, sched, completed, ident, records)
        print(f"{key} epoch {epoch} upd {completed} loss {rec['loss']['total']:.4f} fit {rec['check_accuracy_mean']:.3f} ({t_train:.0f} s)", flush=True)
    if completed != ctx.total:
        raise SystemExit(f"{key}: {completed} updates, expected {ctx.total}")
    torch.save(model.state_dict(), ck / "final.pt"); (ck / "resume.pt").unlink(missing_ok=True)
    man = {"identity": ident, "head": head_name, "seed": seed, "updates": completed, "total_updates": ctx.total, "final_checkpoint": str(ck / "final.pt"),
           "final_sha256": sha256_file(ck / "final.pt"), "init_sha256": init_sha, "plan_sha256": plan_sha, "feature_identity": data.feature_identity,
           "official_commit": getattr(head, "commit", None), "parameters": dtr.n_params(model), "class_weight": plan["class_weight"],
           "pos_weight": plan["pos_weight"], "interrupted": bool(resumed), "resumed_after_updates": resumed, "records": records}
    man_path.write_text(json.dumps(man, indent=1, default=str)); print(json.dumps({"run": key, "updates": completed}), flush=True)


# ----------------------------------------------------------------------------------------------- checks
def max_diff(a: dict, b: dict) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def run_updates(ctx: Ctx, head, state: dict, batches: list, device, seed: int, resume: dict | None = None) -> tuple:
    model = new_model(head, state, device); opt, sched, clip = make_optimiser(ctx, model)
    if resume:
        model.load_state_dict(resume["model"]); opt.load_state_dict(resume["opt"]); sched.load_state_dict(resume["sched"]); set_rng_state(resume["rng"])
    else:
        dtr.seed_everything(seed)
    losses = [update(head, model, b, opt, sched, clip, device)["total"] for b in batches]
    return model, opt, sched, losses


def check_head(ctx: Ctx, head_name: str, seed: int, data: TrainSet, plan: dict, device) -> dict:
    state, _ = ctx.init_state(head_name, seed); head = make_head(ctx, head_name, plan, device); rep = {}
    b0 = data.epoch(ctx, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = run_updates(ctx, head, state, b0[:6], device, seed); b = run_updates(ctx, head, state, b0[:6], device, seed)
    rep["nondeterministic_operators"] = sorted({str(w.message).split(" does not have")[0][:120] for w in caught if "deterministic" in str(w.message)})
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3], b[3])), "params_max_abs": max_diff(a[0].state_dict(), b[0].state_dict())}
    c = run_updates(ctx, head, state, b0[:3], device, seed)
    st = {"model": {k: v.clone() for k, v in c[0].state_dict().items()}, "opt": c[1].state_dict(), "sched": c[2].state_dict(), "rng": rng_state()}
    d = run_updates(ctx, head, state, b0[3:6], device, seed, resume=st)
    rep["resume"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3][3:], d[3])), "params_max_abs": max_diff(a[0].state_dict(), d[0].state_dict())}
    # padding: on the CPU, where every operator is deterministic, a changed padding label must leave the normalised loss
    # bit-identical; on the GPU the official 2-d cross-entropy has no deterministic kernel, so the same input is also
    # evaluated twice there and both GPU differences are recorded beside the exact CPU check
    cpu_head = make_head(ctx, head_name, plan, "cpu"); cm = new_model(cpu_head, state, "cpu"); cm.train(); cb = normalise(dd.collate(b0[0], "cpu"))
    alt = dict(cb); lab = cb["labels"].clone(); lab[cb["valid"] == 0] = -7; alt["labels"] = lab; na = normalise(alt)
    rep["normalised_inputs_equal"] = all(torch.equal(cb[k], na[k]) for k in cb)
    torch.manual_seed(1); c1, _ = cpu_head.loss(cm, cb); torch.manual_seed(1); c2, _ = cpu_head.loss(cm, na)
    rep["padding_label_loss_diff"] = abs(float(c1) - float(c2)); rep["padded_positions"] = int((cb["valid"] == 0).sum())
    model = new_model(head, state, device); model.train(); batch = normalise(dd.collate(b0[0], device))
    torch.manual_seed(1); l1, _ = head.loss(model, batch); torch.manual_seed(1); l1b, _ = head.loss(model, batch)
    g = dict(batch); gl = batch["labels"].clone(); gl[batch["valid"] == 0] = -7; g["labels"] = gl; torch.manual_seed(1); l2, _ = head.loss(model, normalise(g))
    rep["gpu_same_input_loss_diff"] = abs(float(l1) - float(l1b)); rep["gpu_padding_label_loss_diff"] = abs(float(l1) - float(l2))
    model.zero_grad(); l1.backward()
    grads = {n: (float(p.grad.abs().max()) if p.grad is not None else None) for n, p in model.named_parameters()}
    rep["zero_or_missing_gradients"] = [n for n, g in grads.items() if g is None or g == 0.0 or not np.isfinite(g)]
    # inference isolation: other labels, joins and ids of the same lengths change no output
    seqs = data.check; out1 = head.decode(model, seqs, device)
    other = [vsf.Seq(f"other_{i}", not s.is_native, s.feat, (s.labels + 1) % 5, np.array([max(1, s.labels.size // 3)])) for i, s in enumerate(seqs)]
    out2 = head.decode(model, other, device)
    rep["inference_isolation"] = all(np.array_equal(out1[s.seq_id]["labels"], out2[o.seq_id]["labels"]) and np.array_equal(out1[s.seq_id]["prob"], out2[o.seq_id]["prob"])
                                     for s, o in zip(seqs, other))
    # smoke: 100 updates on the 16 check sequences (0-7 and 8-15 alternately), constant learning rate
    sm = new_model(head, state, device); w0 = {k: v.clone() for k, v in sm.state_dict().items()}
    opt = torch.optim.AdamW(sm.parameters(), lr=float(ctx.vt["training"]["optimiser"]["lr"]), weight_decay=float(ctx.vt["training"]["optimiser"]["weight_decay"]))
    dtr.seed_everything(seed); hist = []
    for u in range(int(ctx.vt["checks"]["smoke"]["updates"])):
        sm.train(); loss, parts = head.loss(sm, normalise(dd.collate(seqs[:8] if u % 2 == 0 else seqs[8:], device)))
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)); opt.step()
        if u % 25 == 0 or u == 99:
            hist.append({"update": u, "loss": float(loss.detach()), "grad_norm": gn})
    rep["smoke"] = {"history": hist, "finite": all(np.isfinite(h["loss"]) and np.isfinite(h["grad_norm"]) for h in hist),
                    "weights_changed": max_diff(sm.state_dict(), w0) > 0, **monitor(head, sm, seqs, device)}
    # timing: the formal batches of epoch 0, 5 warm-up and 50 timed updates, plus one monitor and one resume save
    tm_ = new_model(head, state, device); opt, sched, clip = make_optimiser(ctx, tm_); dtr.seed_everything(seed)
    for x in b0[:5]:
        update(head, tm_, x, opt, sched, clip, device)
    torch.cuda.synchronize(); t0 = time.time()
    for x in b0[5:55]:
        update(head, tm_, x, opt, sched, clip, device)
    torch.cuda.synchronize(); per = (time.time() - t0) / 50; t1 = time.time(); monitor(head, tm_, seqs, device)
    tmp = ctx.out / "checks" / f"timing_{head_name}.pt"; tmp.parent.mkdir(parents=True, exist_ok=True)
    save_resume(tmp, tm_, opt, sched, 55, "timing", []); tmp.unlink(); per_epoch = time.time() - t1
    rep["timing"] = {"seconds_per_update": per, "seconds_per_epoch_overhead": per_epoch,
                     "hours_per_run": (per * ctx.total + per_epoch * int(ctx.vt["data"]["epochs"])) / 3600}
    tol = float(ctx.vt["checks"]["reproducibility"]["tolerance"])
    rep["ok"] = bool(rep["reproducibility"]["loss_max_abs"] <= tol and rep["reproducibility"]["params_max_abs"] <= tol and rep["resume"]["loss_max_abs"] <= tol
                     and rep["resume"]["params_max_abs"] <= tol and rep["padding_label_loss_diff"] == 0.0 and rep["normalised_inputs_equal"] and rep["padded_positions"] > 0
                     and not rep["zero_or_missing_gradients"] and rep["inference_isolation"] and rep["smoke"]["finite"] and rep["smoke"]["weights_changed"])
    return rep


def check_pairing(ctx: Ctx, data: TrainSet, plan: dict) -> dict:
    """The three heads of a seed read the same batches: rebuilt three times, collated tensors bit-identical; every epoch's
    digests equal the plan; features float32."""
    out = {"epochs_checked": 0, "updates_checked": 0}
    for e in range(int(ctx.vt["data"]["epochs"])):
        bs = [data.epoch(ctx, e) for _ in HEADS]
        if any([tla.batch_digest(x) for x in b] != plan["epochs"][e] for b in bs):
            raise SystemExit(f"epoch {e}: a rebuilt batch differs from the plan")
        if e in (0, int(ctx.vt["data"]["epochs"]) - 1):
            for x, y, z in zip(*bs):
                cx, cy, cz = (dd.collate(v) for v in (x, y, z))
                if any(not torch.equal(cx[k], c[k]) for c in (cy, cz) for k in cx):
                    raise SystemExit(f"epoch {e}: the three heads' collated batches differ")
                if cx["x"].dtype != torch.float32:
                    raise SystemExit("a batch is not float32")
                out["updates_checked"] += 1
        out["epochs_checked"] += 1
    return out


def run_check(ctx: Ctx) -> None:
    t0, started, rec = time.time(), pv.now(), {"passed": False}
    try:
        determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
        rec.update({"started": started, "frozen_sources_unchanged": all(sha256_file(ctx.root / ctx.vt["source"]["models"][s]["final"]["path"]) ==
                                                                          ctx.vt["source"]["models"][s]["final"]["sha256"] for s in ctx.seeds)})
        idents, heads = {}, {}
        for s in ctx.seeds:
            data = TrainSet(ctx, s); plan, plan_sha = ctx.plan(s); rec.setdefault("pairing", {})[f"s{s}"] = check_pairing(ctx, data, plan)
            for h in HEADS:
                _, init_sha = ctx.init_state(h, s); hd = make_head(ctx, h, plan, device)
                idents[run_key(h, s)] = identity(ctx, h, s, data.feature_identity, plan_sha, init_sha, getattr(hd, "commit", None))
                if s == ctx.seeds[0]:
                    heads[h] = check_head(ctx, h, s, data, plan, device); print(json.dumps({h: heads[h]}, indent=1, default=str), flush=True)
            del data
        per_run = {h: heads[h]["timing"]["hours_per_run"] for h in HEADS}; b = ctx.vt["budget"]
        proj = {"spent": pv.ledger_hours(ctx) + (time.time() - t0) / 3600, "nine_runs": sum(per_run.values()) * len(ctx.seeds), "reserve": float(b["reserve_hours"])}
        proj["total"] = sum(proj.values()); proj["cap"] = float(b["gpu_hours_max"]); proj["within_cap"] = proj["total"] <= proj["cap"]
        rec.update({"identities": idents, "heads": heads, "hours_per_run": per_run, "projection": proj, "finished": pv.now()})
        rec["passed"] = bool(all(heads[h]["ok"] for h in HEADS) and proj["within_cap"] and rec["frozen_sources_unchanged"])
        flat = json.loads(json.dumps(rec, default=str))
        for d in (ctx.out / "checks", ctx.results):
            pv.write_atomic_json(d / "checks.json", flat)
        print(json.dumps({k: rec[k] for k in ("passed", "hours_per_run", "projection")}, indent=1, default=str), flush=True)
        if not rec["passed"]:
            raise SystemExit("a check failed or the projection exceeds the cap; see checks.json")
    finally:
        pv.ledger(ctx, "check", started, time.time() - t0, rec["passed"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--check", action="store_true")
    ap.add_argument("--head", choices=HEADS); ap.add_argument("--seed", type=int)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.check:
        run_check(ctx); return
    if a.head is None or a.seed not in ctx.seeds:
        raise SystemExit("pass --check, or --head and one of the round's seeds")
    determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
    train(ctx, a.head, a.seed, device)


if __name__ == "__main__":
    main()
