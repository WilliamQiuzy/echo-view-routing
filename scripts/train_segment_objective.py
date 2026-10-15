#!/usr/bin/env python
"""Prepare, check, queue and run the segment objective round (docs/superpowers/plans/2026-10-14-segment-objective-training.md,
Tasks 1-3; configs/experiments/segment_objective_v1.yaml).

Three arms per seed train the same Linear(512, 5) head (2,565 parameters) on the seed's frozen V01 Ordered training
features from A01's caches: F (frame-weighted CE, A01's linear condition), B (segment-balanced CE) and O (B plus the
transcript-forward term). Every arm starts from a bit-identical copy of A01's pinned linear initial state (the seed's V01
source classifier), reads A01's batches (every update's digest equals A01's plan) and runs A01's optimiser and schedule
for 18,750 updates; the final weights are kept. No development array, label or probability is read here, and no video
encoder is built.

  --prepare            pins, environment, A01's records (plans, initial states, feature identities, sequence file,
                       frozen heads), full cache hashes (development caches hashed only), the three arm copies of each
                       initial state and the training-only descriptive counts (CPU)
  --check              batch pairing over all epochs, identical initial predictions, reproducibility, 3 + 3 resume,
                       padding, inference isolation, 100-update smoke, F against A01's fp32 cross-entropy, B == O on an
                       all-native batch, the bounded reader on the training cache, timing and the projection (GPU)
  --queue              the nine runs in the fixed order with a projection before each, then the freeze (GPU)
  --arm A --seed S     one formal run with resume state and training-only monitors after every epoch

  python scripts/train_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --prepare
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_l1_asrf as tla  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.temporal.dev_data as dd  # noqa: E402
import echo_routing.temporal.dev_train as dtr  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.losses import segment_objective as so  # noqa: E402

NS = "segment_objective_v1"
ARMS = so.ARMS
A01_HEADS = ("linear", "mstcn", "asrf")
DATA_KEYS = ("epochs", "updates_per_epoch", "total_updates", "streams_per_batch", "natives_per_batch", "native_window", "native_per_epoch")
TRAINING_KEYS = ("arms", "data", "model", "objectives", "training")
CODE_FILES = (Path(__file__), Path(so.__file__), Path(tv.__file__), Path(pv.__file__), Path(dd.__file__), Path(dtr.__file__), Path(vsf.__file__),
              Path(tla.__file__))
PARAMETERS = 5 * 512 + 5


# ----------------------------------------------------------------------------------------------- small helpers
def now() -> str:
    return pv.now()


def code_digest() -> str:
    return pv.file_digest(CODE_FILES)


def run_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def pinned(root: Path, pin: dict) -> Path:
    return pv.pinned(root, pin)


def environment() -> dict:
    import pandas
    import scipy
    import torchvision
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def ledger_append(path: Path, job: str, kind: str, started: str, seconds: float, ok: bool, note: str = "") -> None:
    if kind not in ("gpu", "cpu"):
        raise ValueError("a ledger line is a GPU or a CPU stage")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"job": job, "kind": kind, "started": started, "finished": now(), "hours": float(seconds) / 3600, "ok": bool(ok), "note": note}) + "\n")


def ledger_hours(path: Path, kind: str) -> float:
    path = Path(path)
    if not path.is_file():
        return 0.0
    return float(sum(x["hours"] for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if x.get("kind") == kind))


def run_lines(path: Path, key: str, since: str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [x for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if x["job"] == f"train {key}" and x["started"] >= since]


def ensure_run_line(path: Path, key: str, spawned: str, seconds: float, note: str) -> bool:
    """Books a run's GPU time when the run process wrote no line of its own since it was spawned (it was killed, for
    example together with the queue); returns whether a line was added."""
    if run_lines(path, key, spawned):
        return False
    ledger_append(path, f"train {key}", "gpu", spawned, seconds, False, note)
    return True


def terminate(signum, frame) -> None:  # noqa: ARG001 - signal handler signature
    """SIGTERM ends a stage like an exception, so its ledger lines are still written."""
    raise SystemExit(f"terminated by signal {signum}")


def projection(spent: float, per_arm: dict, remaining: list[str], reserve: float, cap: float) -> dict:
    """Spent GPU hours + the remaining runs at each arm's hours + the reserve, against the cap."""
    runs = float(sum(per_arm[k.rsplit("_s", 1)[0]] for k in remaining)); total = float(spent) + runs + float(reserve)
    return {"spent": float(spent), "remaining_runs": runs, "remaining": list(remaining), "reserve": float(reserve), "total": total, "cap": float(cap),
            "within_cap": bool(total <= float(cap))}


# ----------------------------------------------------------------------------------------------- the round's context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.so = cfg, config, cfg["segment_objective"]
        self.paths = load_paths(); self.root = self.paths.repo_root; self.src = self.so["source"]
        self.out = self.root / self.so["outputs"]["root"]; self.results = self.root / self.so["outputs"]["results_dir"]
        self.ckpt = self.root / self.so["outputs"]["checkpoints"]; self.ledger = self.root / self.so["budget"]["ledger"]
        self.seeds = [int(s) for s in self.so["seeds"]]; d = self.so["data"]
        self.total, self.per_epoch, self.epochs = int(d["total_updates"]), int(d["updates_per_epoch"]), int(d["epochs"])
        if self.epochs * self.per_epoch != self.total or tuple(self.so["arms"]["order"]) != ARMS:
            raise SystemExit("the configured epochs, updates or arms are inconsistent")
        self._a01 = None

    def config_sha(self) -> str:
        return sha256_file(self.root / self.config)

    def a01(self) -> tv.Ctx:
        """A01's context (read only): its pinned configuration, its server preparation equal to the pinned results copy
        and the same data, seeds, optimiser and schedule as this round."""
        if self._a01 is None:
            pinned(self.root, self.src["a01_config"])
            if sha256_file(self.root / self.src["a01_server_preparation"]) != self.src["a01_preparation"]["sha256"]:
                raise SystemExit("A01's server preparation record differs from the pinned results copy")
            a = tv.Ctx(load_cfg(SimpleNamespace(config=self.src["a01_config"]["path"], set=[])), self.src["a01_config"]["path"])
            if any(int(a.vt["data"][k]) != int(self.so["data"][k]) for k in DATA_KEYS) or [int(s) for s in a.vt["data"]["seeds"]] != self.seeds:
                raise SystemExit("the data or seeds differ from A01's")
            t = self.so["training"]
            if a.vt["training"]["optimiser"] != t["optimiser"] or a.vt["training"]["schedule"] != t["schedule"]:
                raise SystemExit("the optimiser or schedule differs from A01's")
            self._a01 = a
        return self._a01

    def alpha(self, plan: dict) -> np.ndarray:
        return so.check_class_weights(plan["class_weight"])

    def init_state(self, arm: str, seed: int, prep: dict) -> tuple[dict, str]:
        rec = prep["inits"][run_key(arm, seed)]; path = self.root / rec["path"]; sha = sha256_file(path)
        if sha != rec["sha256"]:
            raise SystemExit(f"{path} differs from the preparation record")
        return torch.load(path, map_location="cpu", weights_only=True), sha


def read_record(ctx: Ctx, name: str) -> dict:
    f = ctx.out / name
    return json.loads(f.read_text()) if f.is_file() else {}


def preconditions(ctx: Ctx, stage: str) -> None:
    env = environment(); want = ctx.src["environment"]
    if any(str(env[k]) != str(v) for k, v in want.items()):
        raise SystemExit(f"environment {env} differs from the pinned {want}")
    for name in {"check": ("preparation.json",), "queue": ("preparation.json", "checks.json"), "run": ("preparation.json", "checks.json")}.get(stage, ()):
        rec = read_record(ctx, name)
        if rec.get("code") != code_digest() or rec.get("config_sha256") != ctx.config_sha():
            raise SystemExit(f"{name} is missing or was made by other code or another configuration; run that stage again first")
        if name == "checks.json" and rec.get("passed") is not True:
            raise SystemExit("the check did not pass")


def run_stage(ctx: Ctx, name: str, kind: str, fn) -> None:
    """Every stage, finished or failed, appends its wall-clock hours to the ledger with its kind."""
    cap = float(ctx.so["budget"]["cpu_hours_max" if kind == "cpu" else "gpu_hours_max"]); spent = ledger_hours(ctx.ledger, kind)
    if spent >= cap:
        raise SystemExit(f"the {kind.upper()} budget is spent ({spent:.3f} of {cap} h)")
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        preconditions(ctx, name); fn(ctx); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, name, kind, started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- initial states and data
def make_arm_inits(state: dict, inits_dir: Path, seed: int, root: Path) -> dict:
    """Three bit-identical copies of one initial state, one per arm; an existing file must hold the same tensors."""
    want = {k: v.detach().cpu().clone() for k, v in state.items()}; out = {}
    for arm in ARMS:
        path = Path(inits_dir) / f"{run_key(arm, seed)}.pt"
        if path.is_file():
            prev = torch.load(path, map_location="cpu", weights_only=True)
            if set(prev) != set(want) or any(not torch.equal(prev[k], want[k]) for k in want):
                raise SystemExit(f"{path} exists with other content")
        else:
            path.parent.mkdir(parents=True, exist_ok=True); torch.save(want, path)
        out[run_key(arm, seed)] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "parameters": int(sum(v.numel() for v in want.values()))}
    return out


def library_of(ctx: Ctx) -> tuple[list[dict], dict]:
    """The training sequence file (labels and joins, no feature) as a list, with its sha256 checked against A01's record."""
    a01 = ctx.a01(); path = a01.out / "features" / "train" / "sequences.npz"
    if sha256_file(path) != a01.prep["sequence_file"]["sha256"]:
        raise SystemExit("the training sequence file differs from A01's preparation record")
    with np.load(path) as z:
        ids, kinds, n, labels, joff, joins = z["ids"].tolist(), z["kinds"].tolist(), z["n"], z["labels"], z["join_offsets"], z["joins"]
    off = np.concatenate([[0], np.cumsum(n)]); lib = []
    for i, (sid, kind) in enumerate(zip(ids, kinds)):
        lib.append({"id": sid, "kind": kind, "labels": labels[off[i]:off[i + 1]], "joins": joins[joff[i]:joff[i + 1]]})
    return lib, {"sequences": len(ids), "natives": kinds.count("native"), "bank_streams": kinds.count("bank"), "samples": int(off[-1])}


def label_batches(lib: list[dict]) -> vsf.Batches:
    """The batch rule on labels alone (one zero feature column), as A01's plans were made."""
    streams = [vsf.Seq(q["id"], False, np.zeros((q["labels"].size, 1), np.float32), q["labels"], q["joins"]) for q in lib if q["kind"] == "bank"]
    native = {q["id"]: np.zeros((q["labels"].size, 1), np.float32) for q in lib if q["kind"] == "native"}
    clips = [(q["id"], int(q["labels"][0]), int(q["labels"].size)) for q in lib if q["kind"] == "native"]
    return vsf.Batches(streams, clips, native)


def _stats(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())} if x.size else {"n": 0}


def training_counts(lib: list[dict], epochs: list[list[list]], alpha) -> dict:
    """Training-only descriptive counts: runs, transcripts, lengths, classes, same-view joins and X-Y-X middles of the
    library, and the mean share of each update's F and B loss weight by population, run length, class and X-Y-X middle.
    They never set a weight or filter a sample."""
    alpha = so.check_class_weights(alpha); nat = [q for q in lib if q["kind"] == "native"]; bank = [q for q in lib if q["kind"] == "bank"]
    runs = [so.reference_runs(q["labels"]) for q in bank]; lens = {c: [] for c in range(5)}; k_count = defaultdict(int); xyx = short = 0
    for r in runs:
        k_count[len(r.classes)] += 1; segs = [SimpleNamespace(label=int(c)) for c in r.classes]
        for j, (s, e, c) in enumerate(zip(r.starts, r.ends, r.classes)):
            lens[int(c)].append(int(e - s)); short += int(e - s <= 16)
            xyx += int(0 < j < len(segs) - 1 and segs[j - 1].label == segs[j + 1].label and {segs[j - 1].label, segs[j].label} == {2, 3})
    same = sum(int(q["labels"][j - 1] == q["labels"][j]) for q in bank for j in q["joins"])
    library = {"natives": len(nat), "streams": len(bank), "native_length": _stats([q["labels"].size for q in nat]),
               "stream_length": _stats([q["labels"].size for q in bank]), "runs_per_stream": {str(k): v for k, v in sorted(k_count.items())},
               "stream_runs": int(sum(len(r.classes) for r in runs)), "stream_run_length_by_class": {so_cls(c): _stats(v) for c, v in lens.items()},
               "stream_runs_at_most_16": int(short), "same_view_joins": int(same), "joins": int(sum(q["joins"].size for q in bank)),
               "xyx_a4c_a5c_middles": int(xyx), "distinct_stream_transcripts": len({r.transcript for r in runs})}
    keys = ("native", "stream", "short", "xyx_a4c_a5c"); acc = {w: {**dict.fromkeys(keys, 0.0), "by_class": dict.fromkeys(range(5), 0.0)} for w in ("frame", "segment")}
    n_upd = 0
    for ep in epochs:
        for batch in ep:
            items = []
            for s in batch:
                r = so.reference_runs(s.labels)
                for j, (a, b, c) in enumerate(zip(r.starts, r.ends, r.classes)):
                    mid = 0 < j < len(r.classes) - 1 and r.classes[j - 1] == r.classes[j + 1] and {int(r.classes[j - 1]), int(c)} == {2, 3}
                    items.append((bool(s.is_native), int(b - a), int(c), bool(mid)))
            for w in ("frame", "segment"):
                mass = np.array([alpha[c] * (n if w == "frame" else 1) for _, n, c, _ in items]); mass = mass / mass.sum()
                for (is_nat, n, c, mid), m in zip(items, mass):
                    acc[w]["native" if is_nat else "stream"] += m; acc[w]["short"] += m * (n <= 16); acc[w]["xyx_a4c_a5c"] += m * mid; acc[w]["by_class"][c] += m
            n_upd += 1
    mass = {w: {**{k: v / n_upd for k, v in a.items() if k != "by_class"}, "by_class": {so_cls(c): v / n_upd for c, v in a["by_class"].items()}}
            for w, a in acc.items()}
    return {"library": library, "batches": {"updates": n_upd, "mass": mass}}


def so_cls(c: int) -> str:
    return ("PLAX", "PSAX", "A4C", "A5C", "SC4C")[int(c)]


def check_pairing(data, plan: dict, a01ctx, arms=ARMS) -> dict:
    """The arms of a seed read the same batches: rebuilt once per arm, every epoch's digests equal the plan, and the
    collated tensors of the first and last epoch are bit-identical and float32."""
    epochs = int(a01ctx.vt["data"]["epochs"]); out = {"epochs_checked": 0, "updates_checked": 0}
    for e in range(epochs):
        bs = [data.epoch(a01ctx, e) for _ in arms]
        if any([tla.batch_digest(x) for x in b] != plan["epochs"][e] for b in bs):
            raise SystemExit(f"epoch {e}: a rebuilt batch differs from the plan")
        if e in (0, epochs - 1):
            for xs in zip(*bs):
                cs = [dd.collate(v) for v in xs]
                if any(not torch.equal(cs[0][k], c[k]) for c in cs[1:] for k in cs[0]):
                    raise SystemExit(f"epoch {e}: the arms' collated batches differ")
                if cs[0]["x"].dtype != torch.float32:
                    raise SystemExit("a batch is not float32")
                out["updates_checked"] += 1
        out["epochs_checked"] += 1
    return out


# ----------------------------------------------------------------------------------------------- model, loss, updates
def new_model(state: dict, device) -> torch.nn.Module:
    model = torch.nn.Linear(512, 5); model.load_state_dict(state, strict=True)
    return model.to(device)


def make_optimiser(training: dict, total: int, model: torch.nn.Module) -> tuple:
    """A01's AdamW, warm-up and cosine schedule on the head's parameters only."""
    o, s = training["optimiser"], training["schedule"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]),
                            weight_decay=float(o["weight_decay"]))
    warm = int(s["warmup_updates"])
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda c: dtr.lr_factor(c, warm, int(total))), float(o["grad_clip"])


def batch_runs(seqs: list) -> list[so.Runs]:
    """Reference runs from each sequence's own labels (a native crop's runs are recomputed inside the crop)."""
    return [so.reference_runs(s.labels) for s in seqs]


def loss_from_batch(arm: str, model: torch.nn.Module, batch: dict, runs: list[so.Runs], alpha) -> tuple[torch.Tensor, dict]:
    """batch: a normalised collate (x is zero at padding); the model sees the features only."""
    return so.objective(arm, model(batch["x"].transpose(1, 2)), runs, alpha)


def update(arm: str, model, seqs: list, opt, sched, clip: float, alpha, device) -> dict:
    lr = float(opt.param_groups[0]["lr"]); model.train()
    loss, parts = loss_from_batch(arm, model, tv.normalise(dd.collate(seqs, device)), batch_runs(seqs), alpha)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {parts}")
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step(); sched.step()
    return {"total": float(loss.detach()), **parts, "grad_norm": gn, "lr": lr}


def run_updates(arm: str, state: dict, batches: list, alpha, training: dict, total: int, device, seed: int, resume: dict | None = None) -> tuple:
    model = new_model(state, device); opt, sched, clip = make_optimiser(training, total, model)
    if resume:
        model.load_state_dict(resume["model"]); opt.load_state_dict(resume["opt"]); sched.load_state_dict(resume["sched"]); tv.set_rng_state(resume["rng"])
    else:
        dtr.seed_everything(seed)
    losses = [update(arm, model, b, opt, sched, clip, alpha, device)["total"] for b in batches]
    return model, opt, sched, losses


@torch.no_grad()
def predict(model, feats, device) -> dict:
    """Per-sample softmax of the head on each sequence's features (float32 (T, 512)); nothing else is an input."""
    model.eval(); out = {}
    for k, f in feats.items():
        out[k] = torch.softmax(model(torch.from_numpy(np.ascontiguousarray(f, np.float32)).to(device)), -1).cpu().numpy()
    return out


def _edit(pred: np.ndarray, labels: np.ndarray) -> float:
    ref = so.reference_runs(labels).transcript
    return so.edit_distance(so.reference_runs(pred).transcript, ref) / len(ref)


@torch.no_grad()
def monitor(model, seqs: list, alpha, device) -> dict:
    """Training fit on the 16 check sequences (never development data): every arm's terms, argmax accuracy of the native
    crops and the bank streams, and the normalised transcript edit distance of the argmax output."""
    was = model.training; model.eval()
    b = tv.normalise(dd.collate(seqs, device)); logits = model(b["x"].transpose(1, 2)); terms = so.all_terms(logits, batch_runs(seqs), alpha)
    pred = logits.argmax(-1).cpu().numpy(); out = dict(terms)
    for name, flag in (("native", True), ("multi", False)):
        idx = [i for i, s in enumerate(seqs) if bool(s.is_native) == flag]
        out[f"{name}_accuracy"] = float(np.mean([np.mean(pred[i, :seqs[i].labels.size] == seqs[i].labels) for i in idx])) if idx else float("nan")
        out[f"edit_distance_{name}"] = float(np.mean([_edit(pred[i, :seqs[i].labels.size], seqs[i].labels) for i in idx])) if idx else float("nan")
    model.train(was)
    return out


def identity(ctx: Ctx, arm: str, seed: int, feature_identity: str, plan_sha: str, init_sha: str, alpha) -> str:
    payload = {"experiment": NS, "arm": arm, "seed": int(seed), "config": {k: ctx.so[k] for k in TRAINING_KEYS}, "features": feature_identity,
               "sequence_file": ctx.a01().prep["sequence_file"]["sha256"], "plan": plan_sha, "init": init_sha,
               "class_weight": [float(x) for x in alpha], "total_updates": ctx.total, "code": code_digest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def identities(ctx: Ctx, prep: dict) -> dict:
    a01 = ctx.a01(); out = {}
    for s in ctx.seeds:
        plan, plan_sha = a01.plan(s); alpha = ctx.alpha(plan)
        for arm in ARMS:
            _, init_sha = ctx.init_state(arm, s, prep)
            out[run_key(arm, s)] = identity(ctx, arm, s, a01.prep["features"][str(s)]["identity"], plan_sha, init_sha, alpha)
    return out


# ----------------------------------------------------------------------------------------------- one formal run
def train(ctx: Ctx, arm: str, seed: int, device) -> None:
    a01 = ctx.a01(); prep = read_record(ctx, "preparation.json"); cache = ctx.root / prep["caches"][str(seed)]["train"]["path"]
    if sha256_file(cache) != prep["caches"][str(seed)]["train"]["sha256"]:
        raise SystemExit(f"seed {seed}: the training cache differs from the preparation record")
    data = tv.TrainSet(a01, seed); plan, plan_sha = a01.plan(seed)
    alpha = ctx.alpha(plan); state, init_sha = ctx.init_state(arm, seed, prep)
    ident = identity(ctx, arm, seed, data.feature_identity, plan_sha, init_sha, alpha)
    key = run_key(arm, seed); run_dir = ctx.out / "train" / key; ck = ctx.ckpt / key; man_path = run_dir / "run_manifest.json"
    if tv.finished(man_path, ident):
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    run_dir.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    model = new_model(state, device); opt, sched, clip = make_optimiser(ctx.so["training"], ctx.total, model); completed, records, resumed = 0, [], []
    if (ck / "resume.pt").is_file():
        completed, records = tv.load_resume(ck / "resume.pt", model, opt, sched, ident); resumed.append(completed)
        print(f"{key}: resumed after {completed} updates", flush=True)
    else:
        dtr.seed_everything(seed)
    log = run_dir / "train_log.jsonl"
    if completed == 0:
        log.write_text("")
    for epoch in range(completed // ctx.per_epoch, ctx.epochs):
        t0 = time.time(); bs = data.epoch(a01, epoch); sums = defaultdict(float)
        if [tla.batch_digest(x) for x in bs] != plan["epochs"][epoch]:
            raise SystemExit(f"{key} epoch {epoch}: the batches differ from the seed's plan")
        for x in bs:
            for k, v in update(arm, model, x, opt, sched, clip, alpha, device).items():
                sums[k] += v
        completed += len(bs); t_train = time.time() - t0
        rec = {"epoch": epoch, "updates": completed, "loss": {k: v / len(bs) for k, v in sums.items() if k not in ("lr", "grad_norm")},
               "grad_norm_mean": sums["grad_norm"] / len(bs), "lr_last": float(opt.param_groups[0]["lr"]), "seconds_train": t_train,
               "monitor": monitor(model, data.check, alpha, device)}
        records.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        tv.save_resume(ck / "resume.pt", model, opt, sched, completed, ident, records)
        print(f"{key} epoch {epoch} upd {completed} loss {rec['loss']['total']:.4f} fit {rec['monitor']['multi_accuracy']:.3f} ({t_train:.0f} s)", flush=True)
    if completed != ctx.total:
        raise SystemExit(f"{key}: {completed} updates, expected {ctx.total}")
    torch.save(model.state_dict(), ck / "final.pt"); (ck / "resume.pt").unlink(missing_ok=True)
    man = {"identity": ident, "arm": arm, "seed": seed, "updates": completed, "total_updates": ctx.total, "final_checkpoint": str(ck / "final.pt"),
           "final_sha256": sha256_file(ck / "final.pt"), "init_sha256": init_sha, "plan_sha256": plan_sha, "feature_identity": data.feature_identity,
           "parameters": dtr.n_params(model), "class_weight": [float(x) for x in alpha], "interrupted": bool(resumed), "resumed_after_updates": resumed,
           "records": records}
    man_path.write_text(json.dumps(man, indent=1, default=str)); print(json.dumps({"run": key, "updates": completed}), flush=True)


# ----------------------------------------------------------------------------------------------- stages
def stage_prepare(ctx: Ctx) -> None:
    root, src = ctx.root, ctx.src; checked = {"task_sheet": sha256_file(pinned(root, ctx.so["task_sheet"]))}
    for k in ("a01_config", "a01_preparation", "a01_summary", "a04_readme", "a04_summary", "a03_config"):
        checked[k] = sha256_file(pinned(root, src[k]))
    a01 = ctx.a01(); a01_env = a01.verify_sources(); summary = json.loads(pinned(root, src["a01_summary"]).read_text())
    lib, lib_counts = library_of(ctx); ids = {q["id"] for q in lib} | {session_of(q["id"]) for q in lib if q["kind"] == "native"}
    rec = {"config": ctx.config, "config_sha256": ctx.config_sha(), "code": code_digest(), "environment": environment(), "a01_environment": a01_env,
           "checked": checked, "library": lib_counts, "sequence_file": a01.prep["sequence_file"], "seeds": {}, "inits": {}, "caches": {}, "a01_frozen": {},
           "v01_outputs": {}, "counts": {}}
    lb = label_batches(lib); d = ctx.so["data"]
    for s in ctx.seeds:
        plan, plan_sha = a01.plan(s); alpha = ctx.alpha(plan)
        if plan["class_weight"] != a01.prep["plans"][str(s)]["class_weight"]:
            raise SystemExit(f"seed {s}: the plan's class weights differ from A01's preparation record")
        state, init_sha = a01.init_state("linear", s); source = pv.source_classifier(a01, s)
        if set(state) != {"weight", "bias"} or tuple(state["weight"].shape) != (5, 512) or tuple(state["bias"].shape) != (5,) \
                or sum(v.numel() for v in state.values()) != PARAMETERS:
            raise SystemExit(f"seed {s}: A01's linear initial state is not a Linear(512, 5)")
        if not (torch.equal(state["weight"], source["weight"]) and torch.equal(state["bias"], source["bias"])):
            raise SystemExit(f"seed {s}: A01's linear initial state is not the V01 source classifier")
        rec["inits"].update(make_arm_inits(state, ctx.out / "inits", s, root))
        epochs = [lb.epoch(s, e, int(d["native_per_epoch"]), int(d["native_window"]), int(d["streams_per_batch"]), int(d["natives_per_batch"])) for e in range(ctx.epochs)]
        if [[tla.batch_digest(x) for x in ep] for ep in epochs] != plan["epochs"]:
            raise SystemExit(f"seed {s}: the label-only batches differ from A01's plan")
        rec["counts"][str(s)] = training_counts(lib, epochs, alpha)
        train_cache = a01.out / "features" / "train" / f"ordered_s{s}.npz"; dev_cache = a01.out / "features" / "development" / f"ordered_s{s}.npz"
        with np.load(train_cache) as z:
            ident = str(z["identity"])
        if ident != a01.prep["features"][str(s)]["identity"]:
            raise SystemExit(f"seed {s}: the training cache's identity differs from A01's preparation record")
        rec["caches"][str(s)] = {"train": {"path": str(train_cache.relative_to(root)), "sha256": sha256_file(train_cache), "identity": ident},
                                 "development": {"path": str(dev_cache.relative_to(root)), "sha256": sha256_file(dev_cache), "note": "hashed only; not opened before the freeze"}}
        for h in A01_HEADS:
            f = root / "models_frozen" / "video_feature_temporal_v1" / f"{h}_s{s}" / "final.pt"; want = summary["runs"][f"{h}_s{s}"]["final_sha256"]
            if sha256_file(f) != want:
                raise SystemExit(f"{h}_s{s}: A01's frozen final weights differ from its summary")
            rec["a01_frozen"][f"{h}_s{s}"] = want
        for h in ("selection", "calibration"):
            rec["v01_outputs"][f"{h}_s{s}"] = sha256_file(pinned(root, a01.vt["source"]["outputs"][h][s]))
        rec["seeds"][str(s)] = {"plan": {"path": f"runs/video_feature_temporal_v1/plans/plan_s{s}.json", "sha256": plan_sha}, "class_weight": [float(x) for x in alpha],
                                "a01_init": {"sha256": init_sha, "parameters": PARAMETERS, "equals_v01_source_classifier": True},
                                "feature_identity": a01.prep["features"][str(s)]["identity"]}
        print(f"seed {s}: sources, plan, initial states, caches and counts done", flush=True)
    flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
    for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
        pv.write_atomic_json(p, flat)


def check_arm(ctx: Ctx, arm: str, seed: int, data, plan: dict, prep: dict, device) -> dict:
    a01 = ctx.a01(); state, _ = ctx.init_state(arm, seed, prep); alpha = ctx.alpha(plan); tr = ctx.so["training"]; b0 = data.epoch(a01, 0); rep = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = run_updates(arm, state, b0[:6], alpha, tr, ctx.total, device, seed); b = run_updates(arm, state, b0[:6], alpha, tr, ctx.total, device, seed)
    rep["nondeterministic_operators"] = sorted({str(w.message).split(" does not have")[0][:120] for w in caught if "deterministic" in str(w.message)})
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3], b[3])), "params_max_abs": tv.max_diff(a[0].state_dict(), b[0].state_dict())}
    c = run_updates(arm, state, b0[:3], alpha, tr, ctx.total, device, seed)
    st_ = {"model": {k: v.clone() for k, v in c[0].state_dict().items()}, "opt": c[1].state_dict(), "sched": c[2].state_dict(), "rng": tv.rng_state()}
    d = run_updates(arm, state, b0[3:6], alpha, tr, ctx.total, device, seed, resume=st_)
    rep["resume"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3][3:], d[3])), "params_max_abs": tv.max_diff(a[0].state_dict(), d[0].state_dict())}
    # padding: padded features and labels that reach the head unnormalised (other logits at padding) change no loss or
    # gradient, because the loss reads the valid positions of the reference runs only (GPU)
    m = new_model(state, device); raw = dd.collate(b0[0], device); pad = raw["valid"] == 0; alt = dict(raw)
    x = raw["x"].clone(); x[pad[:, None, :].expand_as(x)] = 99.0; lab = raw["labels"].clone(); lab[pad] = 3; alt["x"], alt["labels"] = x, lab
    runs = batch_runs(b0[0]); l1, _ = loss_from_batch(arm, m, tv.normalise(raw), runs, alpha); g1 = torch.autograd.grad(l1, list(m.parameters()))
    l2, _ = loss_from_batch(arm, m, alt, runs, alpha); g2 = torch.autograd.grad(l2, list(m.parameters()))
    rep["padding"] = {"padded_positions": int(pad.sum()), "loss_diff": abs(float(l1) - float(l2)), "grad_max_abs": max(float((p - q).abs().max()) for p, q in zip(g1, g2))}
    rep["zero_or_missing_gradients"] = [i for i, g in enumerate(g1) if not torch.isfinite(g).all() or float(g.abs().max()) == 0.0]
    # inference isolation: the head's output depends on the features alone (other ids, no label input)
    feats = {s.seq_id: s.feat for s in data.check}; p1 = predict(m, feats, device); p2 = predict(m, {f"other{i}": f for i, f in enumerate(feats.values())}, device)
    rep["inference_isolation"] = all(np.array_equal(p1[k], p2[f"other{i}"]) for i, k in enumerate(feats))
    # smoke: 100 updates on the 16 check sequences (0-7 and 8-15 alternately), constant learning rate, no fit threshold
    sm = new_model(state, device); w0 = {k: v.clone() for k, v in sm.state_dict().items()}; o = tr["optimiser"]
    opt = torch.optim.AdamW(sm.parameters(), lr=float(o["lr"]), weight_decay=float(o["weight_decay"])); dtr.seed_everything(seed); hist = []
    for u in range(int(ctx.so["checks"]["smoke"]["updates"])):
        seqs = data.check[:8] if u % 2 == 0 else data.check[8:]; sm.train()
        loss, _ = loss_from_batch(arm, sm, tv.normalise(dd.collate(seqs, device)), batch_runs(seqs), alpha)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)); opt.step()
        if u % 25 == 0 or u == 99:
            hist.append({"update": u, "loss": float(loss.detach()), "grad_norm": gn})
    rep["smoke"] = {"history": hist, "finite": all(np.isfinite(h["loss"]) and np.isfinite(h["grad_norm"]) for h in hist),
                    "weights_changed": tv.max_diff(sm.state_dict(), w0) > 0, "monitor": monitor(sm, data.check, alpha, device)}
    # timing: formal batches of epoch 0 (5 warm-up, 50 timed), then one epoch's batch building, monitor and resume save
    tm_ = new_model(state, device); opt, sched, clip = make_optimiser(tr, ctx.total, tm_); dtr.seed_everything(seed)
    for xb in b0[:5]:
        update(arm, tm_, xb, opt, sched, clip, alpha, device)
    torch.cuda.synchronize(); t0 = time.time()
    for xb in b0[5:55]:
        update(arm, tm_, xb, opt, sched, clip, alpha, device)
    torch.cuda.synchronize(); per = (time.time() - t0) / 50; t1 = time.time()
    [tla.batch_digest(xb) for xb in data.epoch(a01, 1)]; monitor(tm_, data.check, alpha, device)
    tmp = ctx.out / "checks" / f"timing_{arm}.pt"; tmp.parent.mkdir(parents=True, exist_ok=True); tv.save_resume(tmp, tm_, opt, sched, 55, "timing", []); tmp.unlink()
    per_epoch = time.time() - t1
    rep["timing"] = {"seconds_per_update": per, "seconds_per_epoch_overhead": per_epoch, "hours_per_run": (per * ctx.total + per_epoch * ctx.epochs) / 3600}
    tol = float(ctx.so["checks"]["reproducibility"]["tolerance"])
    rep["ok"] = bool(max(rep["reproducibility"].values()) <= tol and max(rep["resume"].values()) <= tol and rep["padding"]["loss_diff"] == 0.0
                     and rep["padding"]["grad_max_abs"] == 0.0 and rep["padding"]["padded_positions"] > 0 and not rep["zero_or_missing_gradients"]
                     and rep["inference_isolation"] and rep["smoke"]["finite"] and rep["smoke"]["weights_changed"])
    return rep


def numerics(ctx: Ctx, data, plan: dict, prep: dict, device) -> dict:
    """F against A01's fp32 masked weighted cross-entropy, and B against O on the all-native part of a formal batch."""
    a01 = ctx.a01(); state, _ = ctx.init_state("frame", ctx.seeds[0], prep); alpha = ctx.alpha(plan); m = new_model(state, device)
    seqs = data.epoch(a01, 0)[0]; b = tv.normalise(dd.collate(seqs, device)); runs = batch_runs(seqs)
    lf, _ = loss_from_batch("frame", m, b, runs, alpha); gf = torch.autograd.grad(lf, list(m.parameters()))
    lg = m(b["x"].transpose(1, 2)); ce = F.cross_entropy(lg.reshape(-1, 5), b["labels"].reshape(-1), weight=torch.tensor(plan["class_weight"], dtype=torch.float32, device=device),
                                                        ignore_index=dd.PAD_LABEL)
    gc = torch.autograd.grad(ce, list(m.parameters()))
    nat = [s for s in seqs if s.is_native]; bn = tv.normalise(dd.collate(nat, device)); rn = batch_runs(nat)
    lb, _ = loss_from_batch("segment", m, bn, rn, alpha); gb = torch.autograd.grad(lb, list(m.parameters()))
    lo, _ = loss_from_batch("order", m, bn, rn, alpha); go = torch.autograd.grad(lo, list(m.parameters()))
    return {"frame_vs_a01_fp32_ce": {"loss_abs": abs(float(lf) - float(ce)), "grad_max_abs": max(float((p.double() - q.double()).abs().max()) for p, q in zip(gf, gc))},
            "segment_vs_order_all_native": {"natives": len(nat), "loss_abs": abs(float(lb) - float(lo)), "grad_max_abs": max(float((p - q).abs().max()) for p, q in zip(gb, go))}}


def reader_check(ctx: Ctx, data) -> dict:
    """The bounded development-cache reader on the training cache of the first seed (training data; real layout)."""
    import scripts.report_segment_objective as rs   # the reader lives with the report; imported here only for this check
    path = ctx.a01().out / "features" / "train" / f"ordered_s{ctx.seeds[0]}.npz"; lay = rs.member_layout(path, "samples")
    full = np.concatenate([s.feat for s in data.streams[:3]]); rows = sum(s.feat.shape[0] for s in data.streams[:3])
    with np.load(path) as z:
        off = np.concatenate([[0], np.cumsum(z["n"])]); ids = z["ids"].tolist()
    i0 = ids.index(data.streams[0].seq_id); got = rs.read_rows(path, lay, [(int(off[i0]), int(off[i0]) + rows)])
    return {"layout": {"offset": lay.offset, "dtype": str(lay.dtype), "shape": list(lay.shape)}, "rows_compared": int(rows), "equal": bool(np.array_equal(got, full))}


def stage_check(ctx: Ctx) -> None:
    tv.determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction()); t0 = time.time()
    prep = read_record(ctx, "preparation.json"); a01 = ctx.a01(); rec = {"started": now(), "code": code_digest(), "config_sha256": ctx.config_sha(), "pairing": {},
                                                                         "initial_predictions_equal": {}, "arms": {}, "passed": False}
    for s in ctx.seeds:
        data = tv.TrainSet(a01, s); plan, _ = a01.plan(s); rec["pairing"][f"s{s}"] = check_pairing(data, plan, a01)
        feats = {q.seq_id: q.feat for q in data.check}; outs = [predict(new_model(ctx.init_state(arm, s, prep)[0], device), feats, device) for arm in ARMS]
        rec["initial_predictions_equal"][f"s{s}"] = all(np.array_equal(outs[0][k], o[k]) for o in outs[1:] for k in feats)
        if s == ctx.seeds[0]:
            for arm in ARMS:
                rec["arms"][arm] = check_arm(ctx, arm, s, data, plan, prep, device); print(json.dumps({arm: rec["arms"][arm]}, default=str), flush=True)
            rec["numerics"] = numerics(ctx, data, plan, prep, device); rec["reader"] = reader_check(ctx, data)
        del data
    rec["identities"] = identities(ctx, prep); per_arm = {a: rec["arms"][a]["timing"]["hours_per_run"] for a in ARMS}; b = ctx.so["budget"]
    rec["hours_per_run"] = per_arm
    rec["projection"] = projection(ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600, per_arm, [run_key(a, s) for s in ctx.seeds for a in ARMS],
                                   float(b["gpu_reserve_hours"]), float(b["gpu_hours_max"]))
    rec["passed"] = bool(all(rec["arms"][a]["ok"] for a in ARMS) and all(rec["initial_predictions_equal"].values()) and rec["reader"]["equal"]
                         and rec["projection"]["within_cap"] and rec["numerics"]["segment_vs_order_all_native"]["loss_abs"] <= 1e-9)
    rec["finished"] = now(); flat = json.loads(json.dumps(rec, default=str))
    for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
        pv.write_atomic_json(p, flat)
    print(json.dumps({k: rec[k] for k in ("passed", "hours_per_run", "projection")}, indent=1, default=str), flush=True)
    if not rec["passed"]:
        raise SystemExit("a check failed or the projection exceeds the cap; see checks.json")


def freeze(ctx: Ctx, idents: dict, prep: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; dest = ctx.root / "models_frozen" / NS / key
        if not tv.finished(man_path, ident):
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(man_path.read_text())
        if not dest.exists():
            arm, s = key.rsplit("_s", 1); init = ctx.root / prep["inits"][key]["path"]
            files = [man["final_checkpoint"], str(man_path), str(init), str(ctx.root / ctx.config)]
            r = subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                                "--meta", f"arm={arm}", "--meta", f"seed={s}", "--meta", f"updates={man['updates']}",
                                "--notes", "Segment objective round (2026-10-14): Linear(512, 5) final weights after 18,750 updates on frozen V01 Ordered features."],
                               cwd=ctx.root, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"{key}: freeze_model.py failed ({(r.stderr or r.stdout)[-500:]}); a partly created frozen directory must be inspected by hand")
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ctx.root)), "final_sha256": man["final_sha256"], "updates": man["updates"], "interrupted": man["interrupted"]}
    return out


def stage_queue(ctx: Ctx, config: str, logs: str = "logs") -> None:
    """The nine runs in the configured order; before each, the projection from the ledger's GPU hours, the remaining runs
    at the larger of the checked timing and the slowest finished run of that arm, and the reserve. Then the freeze."""
    prep = read_record(ctx, "preparation.json"); check = read_record(ctx, "checks.json"); idents = identities(ctx, prep); b = ctx.so["budget"]
    if check.get("identities") != idents:
        raise SystemExit("the passed check record carries other run identities; run the check again")
    queue = list(ctx.so["training"]["queue"]); status_path = ctx.out / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]:
        raise SystemExit("the status file belongs to other identities")
    status.setdefault("jobs", {})
    for k in queue:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    pv.write_atomic_json(status_path, status)
    for key in queue:
        arm, s = key.rsplit("_s", 1); man = ctx.out / "train" / key / "run_manifest.json"
        if tv.finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; pv.write_atomic_json(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done = {k: v for k, v in status["jobs"].items() if v.get("state") == "done" and "hours" in v}
        per_arm = {a: max([float(check["hours_per_run"][a]), *[v["hours"] for k, v in done.items() if k.startswith(a + "_")]]) for a in ARMS}
        remaining = [k for k in queue if not tv.finished(ctx.out / "train" / k / "run_manifest.json", idents[k])]
        proj = projection(ledger_hours(ctx.ledger, "gpu"), per_arm, remaining, float(b["gpu_reserve_hours"]), float(b["gpu_hours_max"]))
        status["projection"] = {"at": now(), "before": key, **proj}
        if not proj["within_cap"]:
            status["stopped"] = {"at": now(), "reason": f"projected {proj['total']:.2f} GPU hours exceed the cap {proj['cap']}", "before": key}
            pv.write_atomic_json(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / logs / f"{NS}_{key}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/train_segment_objective.py", "--config", config, "--arm", arm, "--seed", s]
        status["jobs"][key] = {"state": "running", "identity": idents[key], "arm": arm, "seed": int(s), "log": str(log.relative_to(ROOT)), "started": now()}
        pv.write_atomic_json(status_path, status); started, t0 = now(), time.time(); ctx.inflight = (key, started, t0)
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        ctx.inflight = None; hours = (time.time() - t0) / 3600; ok = rc == 0 and tv.finished(man, idents[key])
        ensure_run_line(ctx.ledger, key, started, hours * 3600, f"the run wrote no ledger line (return code {rc}); booked by the queue")
        status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else "failed", "returncode": rc, "finished": now(), "hours": hours}
        if ok:
            m = json.loads(man.read_text()); status["jobs"][key].update({"updates": m["updates"], "final_sha256": m["final_sha256"], "interrupted": m["interrupted"]})
        else:
            status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        pv.write_atomic_json(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.3f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; locate the cause and run the queue again (finished runs are skipped, this one resumes)")
    status["frozen"] = freeze(ctx, idents, prep); status["finished"] = now(); pv.write_atomic_json(status_path, status); print("queue finished and frozen", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); g = ap.add_mutually_exclusive_group()
    g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", action="store_true")
    ap.add_argument("--arm", choices=ARMS); ap.add_argument("--seed", type=int); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        run_stage(ctx, "prepare", "cpu", stage_prepare)
    elif a.check:
        run_stage(ctx, "check", "gpu", stage_check)
    elif a.queue:
        signal.signal(signal.SIGTERM, terminate); preconditions(ctx, "queue"); t0, started, ok, note = time.time(), now(), False, ""; ctx.inflight = None
        try:
            stage_queue(ctx, a.config, a.logs); ok = True
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:   # a run killed with the queue is booked here; the queue's own time is its wall time minus its runs
            if ctx.inflight is not None:
                key, spawned, ts0 = ctx.inflight; ensure_run_line(ctx.ledger, key, spawned, time.time() - ts0, f"interrupted with the queue ({note}); booked by the queue")
            run_hours = sum(float(x["hours"]) for x in (json.loads(line) for line in ctx.ledger.read_text().splitlines() if line.strip())
                            if x["job"].startswith("train ") and x["started"] >= started)
            ledger_append(ctx.ledger, "queue overhead and freeze", "cpu", started, max(0.0, time.time() - t0 - run_hours * 3600), ok, note)
    elif a.arm is not None and a.seed in ctx.seeds:
        signal.signal(signal.SIGTERM, terminate); t0, started, ok, note = time.time(), now(), False, ""
        try:
            preconditions(ctx, "run"); tv.determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
            train(ctx, a.arm, a.seed, device); ok = True
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:   # every run, direct or from the queue, finished, failed or interrupted, books its own GPU time
            ledger_append(ctx.ledger, f"train {run_key(a.arm, a.seed)}", "gpu", started, time.time() - t0, ok, note)
    else:
        raise SystemExit("pass --prepare, --check, --queue, or --arm and one of the round's seeds")


if __name__ == "__main__":
    main()
