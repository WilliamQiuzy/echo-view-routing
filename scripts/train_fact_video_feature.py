#!/usr/bin/env python
"""Prepare, check, queue, run, freeze and decode the training streams of the FACT video feature round
(docs/superpowers/plans/2026-10-15-fact-video-feature-training.md, Tasks 1-2; configs/experiments/fact_video_feature_v1.yaml).

Exactly three formal trainings: the official GTEA FACT (pinned checkout, existing FACTHead adapter) on each seed's frozen
V01 Ordered 512-d video features from A01's caches, with A01's batches (every update's digest equals A01's plan) and
F02's untrained FACT initial states, F02's Adam recipe and 18,750 updates; the final weights are kept. The video encoder
is never built or run, no feature is extracted, and no development array, label or probability is read here.

  --prepare      pins and environment, A01's and A05's records (plans, caches, sequence file, frozen controls), the
                 official checkout (HEAD, no tracked change, source digest), the 12-token rule from the training labels,
                 every epoch's label-only batch digests, every batch of every epoch unpadded against its sequences,
                 float32 spot checks against the source cache, the bit-identical copies of F02's initial states and
                 their strict load (CPU); development caches are only hashed
  --check        timing first, then strict initial states, 6-update reproducibility and 3 + 3 resume, padding,
                 isolation under five label sequences at the initial and the overfit state, the overfit smoke on the
                 16 check sequences (at most 2,000 updates), FACT and control decoding times and the budget
                 projection (GPU)
  --queue        the three runs with a projection before each, then the freeze (GPU)
  --seed S       one formal run with resume state and a training-only monitor after every epoch
The freeze and the decoding of the training streams after the runs live in scripts/fact_video_feature_post.py, outside
the training identity, so that a repair there never invalidates a finished run.

  python scripts/train_fact_video_feature.py --config configs/experiments/fact_video_feature_v1.yaml --prepare
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.fact_integration as fi  # noqa: E402
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_fact_asrf as tfa  # noqa: E402
import scripts.train_l1_asrf as tla  # noqa: E402
import scripts.train_segment_objective as ts  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.dev_data as dd  # noqa: E402
import echo_routing.temporal.dev_train as dtr  # noqa: E402
import echo_routing.temporal.fact_head as fh  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = "fact_video_feature_v1"
ARM = "FACT"
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
DATA_KEYS = ("epochs", "updates_per_epoch", "total_updates", "streams_per_batch", "natives_per_batch", "native_window", "native_per_epoch")
TRAINING_KEYS = ("seeds", "data", "fact", "monitor")
CODE_FILES = tuple(Path(m.__file__) for m in (fh, tfa, fi, tla, ts, tv, pv, dd, dtr, vsf)) + (Path(__file__),)
STAGE_OVERHEAD_HOURS = 0.05       # report-stage overhead after A05's recorded evaluation stages (0.006 to 0.019 GPU h each)

ledger_append, ledger_hours, ensure_run_line, terminate, environment, now = (ts.ledger_append, ts.ledger_hours, ts.ensure_run_line, ts.terminate,
                                                                              ts.environment, ts.now)


def run_key(seed: int) -> str:
    return f"{ARM}_s{seed}"


def code_digest() -> str:
    return pv.file_digest(CODE_FILES)


def pinned(root: Path, pin: dict) -> Path:
    return pv.pinned(root, pin)


def max_diff(a: dict, b: dict) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


# ----------------------------------------------------------------------------------------------- the round's context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.fv = cfg, config, cfg["fact_video_feature"]
        self.paths = load_paths(); self.root = self.paths.repo_root; self.src = self.fv["source"]; self.f = self.fv["fact"]
        o = self.fv["outputs"]
        self.out, self.results, self.ckpt = self.root / o["root"], self.root / o["results_dir"], self.root / o["checkpoints"]
        self.ledger = self.root / self.fv["budget"]["ledger"]; self.seeds = [int(s) for s in self.fv["seeds"]]; d = self.fv["data"]
        self.total, self.per_epoch, self.epochs = int(d["total_updates"]), int(d["updates_per_epoch"]), int(d["epochs"])
        if self.epochs * self.per_epoch != self.total or self.fv["queue"] != [run_key(s) for s in self.seeds]:
            raise SystemExit("the configured epochs, updates or queue are inconsistent")
        self._a01 = None

    def config_sha(self) -> str:
        return sha256_file(self.root / self.config)

    def a01(self) -> tv.Ctx:
        if self._a01 is None:
            pinned(self.root, self.src["a01_config"])
            if sha256_file(self.root / self.src["a01_server_preparation"]) != self.src["a01_preparation"]["sha256"]:
                raise SystemExit("A01's server preparation record differs from the pinned results copy")
            a = tv.Ctx(load_cfg(SimpleNamespace(config=self.src["a01_config"]["path"], set=[])), self.src["a01_config"]["path"])
            if any(int(a.vt["data"][k]) != int(self.fv["data"][k]) for k in DATA_KEYS) or [int(s) for s in a.vt["data"]["seeds"]] != self.seeds:
                raise SystemExit("the data or seeds differ from A01's")
            self._a01 = a
        return self._a01

    def json_pin(self, key: str) -> dict:
        return json.loads(pinned(self.root, self.src[key]).read_text())


def read_record(ctx: Ctx, name: str) -> dict:
    f = ctx.out / name
    return json.loads(f.read_text()) if f.is_file() else {}


# ----------------------------------------------------------------------------------------------- official model
def official_state(repo: Path, commit: str) -> dict:
    """HEAD at the pinned commit, no tracked modification (untracked files are not official sources) and the digest
    of every tracked file. A difference stops the round; nothing is reset."""
    repo = Path(repo)
    try:
        head = fh.check_pinned(repo, commit)
        changes = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                                 check=True).stdout.splitlines()
        files = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, text=True, check=True).stdout.split("\0")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"{repo}: git failed ({e.returncode}); the official checkout cannot be verified") from None
    if changes:
        raise SystemExit(f"{repo}: {len(changes)} tracked file(s) differ from the pinned commit; located, not reset")
    h = hashlib.sha256(); names = sorted(f for f in files if f)
    for name in names:
        h.update(name.encode() + b"\0" + (repo / name).read_bytes())
    return {"head": head, "tracked_changes": [], "files": len(names), "digest": h.hexdigest()}


def check_rule(rule: fh.TokenRule, f: dict) -> None:
    if (rule.max_transcript, rule.ntoken) != (int(f["expected_longest_transcript"]), int(f["expected_ntoken"])) or \
            abs(rule.average_transcript - float(f["expected_average_transcript"])) > 1e-12 or rule.token_ratio != float(f["token_ratio"]):
        raise SystemExit(f"the token rule {rule} differs from the fixed configuration; the token count is never changed to fit")


def rule_of(lib: list[dict], f: dict, native_per_epoch: int) -> fh.TokenRule:
    rule = fh.token_rule([q["labels"] for q in lib if q["kind"] == "bank"], int(native_per_epoch), float(f["token_ratio"]))
    check_rule(rule, f)
    return rule


def make_head(ctx: Ctx, rule: fh.TokenRule) -> fh.FACTHead:
    f = ctx.f; head = fh.FACTHead(ctx.root / f["repo"], int(f["in_dim"]), rule, num_classes=int(f["n_classes"]), base_config=f["base_config"])
    c = head.cfg
    if c.FACT.trans is not bool(f["trans"]) or c.Loss.match != f["match"] or list(c.bg_class) != list(f["bg_class"]) or int(c.FACT.ntoken) != int(f["expected_ntoken"]):
        raise SystemExit("the expanded FACT configuration differs from the fixed configuration")
    if head.commit != f["commit"]:
        raise SystemExit(f"FACT commit {head.commit} != {f['commit']}")
    return head


def copy_init(src: Path, dst: Path, root: Path) -> dict:
    """A bit-identical copy of an untrained initial state; an existing copy must hold the same bytes."""
    src, dst = Path(src), Path(dst)
    if dst.is_file():
        if dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst} exists with other content")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        if dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst}: the copy differs from its source")
    return {"path": str(dst.relative_to(root)), "sha256": sha256_file(dst)}


def strict_init(head, path: Path) -> dict:
    """The state loads strictly into a fresh build with the same tensor names, shapes and dtypes."""
    model = head.build(); fresh = model.state_dict(); st = torch.load(path, map_location="cpu", weights_only=True)
    if set(st) != set(fresh):
        raise SystemExit(f"{Path(path).name}: {len(set(fresh) - set(st))} tensor(s) missing and {len(set(st) - set(fresh))} unexpected against a fresh build")
    bad = [k for k in fresh if tuple(st[k].shape) != tuple(fresh[k].shape) or st[k].dtype != fresh[k].dtype]
    if bad:
        raise SystemExit(f"{Path(path).name}: {len(bad)} tensor(s) differ in shape or dtype from a fresh build")
    model.load_state_dict(st, strict=True)
    return {"tensors": len(st), "parameters": int(sum(p.numel() for p in model.parameters()))}


def new_model(head, init_path: Path, device) -> torch.nn.Module:
    model = head.build(); model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


# ----------------------------------------------------------------------------------------------- identity
def identity_payload(fv: dict, seed: int, official: dict, expanded: str, rule: dict, inputs: dict, code: str) -> dict:
    return {"experiment": NS, "seed": int(seed), "config": {k: fv[k] for k in TRAINING_KEYS}, "official": official, "expanded_config": expanded,
            "token_rule": rule, "inputs": inputs, "total_updates": int(fv["data"]["total_updates"]), "code": code}


def identity_of(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def identity(ctx: Ctx, seed: int, head, prep: dict) -> str:
    a01 = ctx.a01(); key = run_key(seed); _, plan_sha = a01.plan(seed)
    inputs = {"features": a01.prep["features"][str(seed)]["identity"], "train_cache_sha256": prep["caches"][str(seed)]["train"]["sha256"],
              "sequence_file": a01.prep["sequence_file"]["sha256"], "plan": plan_sha, "init": prep["inits"][key]["sha256"]}
    official = {"head": head.commit, "digest": prep["official"]["digest"]}
    return identity_of(identity_payload(ctx.fv, seed, official, head.cfg.dump(), head.rule.__dict__, inputs, code_digest()))


def identities(ctx: Ctx, head, prep: dict) -> dict:
    return {run_key(s): identity(ctx, s, head, prep) for s in ctx.seeds}


def run_state(manifest: Path, ident: str, total: int = 18750):
    """'done': this identity, every update and an intact final checkpoint; 'other': any other finished record; None: none."""
    manifest = Path(manifest)
    if not manifest.is_file():
        return None
    m = json.loads(manifest.read_text()); ck = Path(m.get("final_checkpoint", ""))
    ok = m.get("identity") == ident and int(m.get("completed_updates", -1)) == int(total) and ck.is_file() and sha256_file(ck) == m.get("final_sha256")
    return "done" if ok else "other"


# ----------------------------------------------------------------------------------------------- batches and inputs
def check_batches(data, plan: dict, a01ctx) -> dict:
    """Every epoch's structure digests against the plan, and every batch unpadded and compared element by element with
    its sequences (features and labels, as cut from the source cache)."""
    epochs = int(a01ctx.vt["data"]["epochs"]); out = {"epochs_checked": 0, "batches_unpadded": 0, "sequences_compared": 0, "samples": 0}
    for e in range(epochs):
        bs = data.epoch(a01ctx, e)
        if [tla.batch_digest(x) for x in bs] != plan["epochs"][e]:
            raise SystemExit(f"epoch {e}: a batch differs from the plan")
        for b in bs:
            xs, ys = fh.FACTHead.unpad(dd.collate(b))
            for x, y, s in zip(xs, ys, b):
                if x.dtype != torch.float32 or not np.array_equal(x.numpy(), s.feat) or not np.array_equal(y.numpy(), s.labels):
                    raise SystemExit(f"epoch {e}: an unpadded input differs from its sequence")
                out["sequences_compared"] += 1; out["samples"] += int(s.labels.size)
            out["batches_unpadded"] += 1
        out["epochs_checked"] += 1
    return out


def spot_check(npz_path: Path, data, n_sequences: int = 8) -> dict:
    """The TrainSet's float32 rows of the first constructed streams and native recordings against a separate bounded read
    of the source cache (training data only)."""
    import scripts.report_segment_objective as rs   # the bounded reader
    with np.load(npz_path) as z:
        ids, n = z["ids"].tolist(), z["n"]
    off = np.concatenate([[0], np.cumsum(n)]); lay = rs.member_layout(npz_path, "samples"); equal, rows = True, 0
    pairs = [(s.seq_id, s.feat) for s in data.streams[:n_sequences]] + [(v, data.batches.native_feat[v]) for v, _, _ in data.clips[:n_sequences]]
    for sid, feat in pairs:
        i = ids.index(sid); got = rs.read_rows(npz_path, lay, [(int(off[i]), int(off[i + 1]))])
        equal &= bool(got.dtype == np.float32 and np.array_equal(got, feat)); rows += int(got.shape[0])
    return {"equal": bool(equal), "rows_compared": rows, "sequences": len(pairs)}


def library(ctx: Ctx) -> tuple[list[dict], dict]:
    return ts.library_of(ctx)            # A01's sequence file (labels and joins), its sha256 checked against A01's record


def batch_loss(head, model, batch: dict, seed: int) -> float:
    dtr.seed_everything(seed); model.train(); loss, _ = head.loss(model, batch)
    return float(loss.detach())


def padding_check(head, model, batch: dict, seqs: list, device, seed: int) -> dict:
    """Padded labels and features changed with shapes and valid samples kept: the official loss in training mode under
    the same RNG state, and the eval-mode predictions of the unpadded inputs, must not change."""
    pad = batch["valid"] == 0; alt = dict(batch)
    x = batch["x"].clone(); x[pad[:, None, :].expand_as(x)] = 77.0; lab = batch["labels"].clone(); lab[pad] = 3; alt["x"], alt["labels"] = x, lab
    l1, l2 = batch_loss(head, model, batch, seed), batch_loss(head, model, alt, seed)
    model.eval(); xs, _ = fh.FACTHead.unpad(alt)
    equal = all(np.array_equal(head.predict(model, xi)["labels"], head.predict(model, torch.from_numpy(s.feat).to(xi.device))["labels"]) for xi, s in zip(xs, seqs))
    return {"padded_positions": int(pad.sum()), "loss_diff": abs(l1 - l2), "predictions_equal": bool(equal)}


def smoke_due(step: int, start: int = 500, every: int = 250, last: int = 2000) -> bool:
    """The overfit smoke's check points: every `every` updates from `start`, and the last update."""
    return step == last or (step >= start and (step - start) % every == 0)


# ----------------------------------------------------------------------------------------------- updates and resume
def run_updates(head, init_path: Path, batches: list, fcfg: dict, total: int, device, seed: int, resume: dict | None = None) -> tuple:
    model = new_model(head, init_path, device); opt, sched, clip = tfa.make_optimiser(fcfg, model, total)
    if resume:
        model.load_state_dict(resume["model"]); opt.load_state_dict(resume["opt"]); sched.load_state_dict(resume["sched"]); tfa.set_rng_state(resume["rng"])
    else:
        dtr.seed_everything(seed)
    losses = [tfa.update(head, model, b, opt, sched, clip, device)["total"] for b in batches]
    return model, opt, sched, losses


def resume_check(head, init_path: Path, batches: list, fcfg: dict, total: int, device, seed: int, path: Path, reference: tuple) -> dict:
    """The first half of the updates, a resume file on disk, a fresh model, optimiser and scheduler loaded from it, and
    the second half; compared with an uninterrupted reference run."""
    half = len(batches) // 2; first = run_updates(head, init_path, batches[:half], fcfg, total, device, seed)
    first_diff = max(abs(a - b) for a, b in zip(reference[3][:half], first[3]))
    tfa.save_resume(Path(path), first[0], first[1], first[2], half, "check", [])
    model = new_model(head, init_path, device); opt, sched, clip = tfa.make_optimiser(fcfg, model, total)
    done, _ = tfa.load_resume(Path(path), model, opt, sched, "check"); bound = tfa.optimiser_bound(opt, model)
    losses = [tfa.update(head, model, b, opt, sched, clip, device)["total"] for b in batches[half:]]
    ref_model, ref_opt, _, ref_losses = reference
    return {"completed_before": int(done), "params_max_abs": max_diff(ref_model.state_dict(), model.state_dict()), "first_half_losses_max_abs": first_diff,
            "losses_max_abs": max(abs(a - b) for a, b in zip(ref_losses[half:], losses)), "bound": bool(bound),
            "lr_equal": float(opt.param_groups[0]["lr"]) == float(ref_opt.param_groups[0]["lr"])}


@torch.no_grad()
def term_values(head, model, seqs: list, device) -> dict:
    model.eval()
    return {f"loss_{k}": float(v) for k, v in fi.term_losses(head, model, list(seqs), device).items()}


def monitor(head, model, seqs: list, device) -> dict:
    """Training fit on the 16 check sequences: frame accuracy, segment F1@50, exact transcripts and each loss term."""
    chosen = [("native" if s.is_native else "stream", s) for s in seqs]
    return {**tfa.monitor(head, model, chosen, device), **term_values(head, model, seqs, device)}


# ----------------------------------------------------------------------------------------------- budget helpers
def projection(spent: float, per_run: dict, remaining: list[str], post: float, reserve: float, cap: float) -> dict:
    runs = float(sum(per_run[k] for k in remaining)); total = float(spent) + runs + float(post) + float(reserve)
    return {"spent": float(spent), "remaining_runs": runs, "remaining": list(remaining), "post_training": float(post), "reserve": float(reserve),
            "total": total, "cap": float(cap), "within_cap": bool(total <= float(cap))}


def fit_time(times, sizes) -> tuple[float, float]:
    """Seconds per update as a + b x valid samples (least squares); a degenerate or negative slope falls back to the mean."""
    times, sizes = np.asarray(times, float), np.asarray(sizes, float)
    if sizes.size < 2 or np.ptp(sizes) == 0:
        return float(times.mean()), 0.0
    b, a = np.polyfit(sizes, times, 1)
    return (float(times.mean()), 0.0) if b < 0 else (float(a), float(b))


def run_hours(a: float, b: float, n_batches: int, samples: int, epochs: int, per_epoch_s: float) -> float:
    return (a * n_batches + b * samples + epochs * per_epoch_s) / 3600


def smoke_passed(out: dict, iso: list[dict]) -> dict:
    n = int(out["sequences"])
    req = {"sixteen_sequences": n == 16, "transcripts_exact": out["transcript_exact"] == n, "segment_f1_all_one": out["segment_f1_50_min"] == 1.0,
           "one_to_one": out["match_one_to_one"] == n, "matched_classes_right": out["matched_token_class_correct_min"] == 1.0,
           "isolation": all(i["changed_predictions"] == 0 and i["max_abs_logit_difference"] == 0 and i["adapter_decode_differs"] == 0 for i in iso)}
    return {**req, "all": bool(all(req.values()))}


# ----------------------------------------------------------------------------------------------- checks of the model
def overfit(head, init_path: Path, chosen: list, fcfg: dict, sm: dict, device) -> tuple:
    """The 16 check sequences, two fixed batches of 8 alternating, Adam at the constant smoke rate with the formal betas,
    eps, weight decay and clip; official augmentation on; from the seed-0 initial state (F02's procedure). The fit is
    checked at fixed points and the smoke stops at the first passing check, after at most the configured updates."""
    o = fcfg["optimizer"]; dtr.seed_everything(0); model = new_model(head, init_path, device); seqs = [s for _, s in chosen]
    opt = torch.optim.Adam(model.parameters(), lr=float(sm["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    batches, clip, log, sums, checks = [seqs[:8], seqs[8:16]], float(o["grad_clip"]), [], defaultdict(float), []
    last, out = int(sm["updates"]), None
    for step in range(last):
        model.train(); loss, _ = head.loss(model, dd.collate(batches[step % 2], device))
        opt.zero_grad(set_to_none=True); loss.backward(); sums["grad_norm"] += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step()
        sums["train_loss"] += float(loss); n = step + 1
        if n % 50 == 0:
            terms = term_values(head, model, seqs, device)
            log.append({"update": n, **{k: v / 50 for k, v in sums.items()}, **terms}); sums.clear()
        if smoke_due(n, int(sm["check_from"]), int(sm["check_every"]), last):
            out = fi.outcome(head, model, chosen, device); req = smoke_passed(out, [])
            checks.append({"update": n, **{k: v for k, v in req.items() if k != "isolation"}})
            print(f"overfit {n}: {json.dumps({k: v for k, v in req.items() if k != 'isolation'})}", flush=True)
            if req["all"]:
                break
    return model, log, out, checks


def control_seconds_per_sample(ctx: Ctx, seed: int, seqs: list, device) -> float:
    """A01's frozen ASRF and MS-TCN with their official decoding on training streams (for the budget estimate only)."""
    a01 = ctx.a01(); plan, _ = a01.plan(seed); summary = ctx.json_pin("a01_summary"); spent = 0.0
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        for name in ("asrf", "mstcn"):
            h = tv.make_head(a01, name, plan, device); f = ctx.root / "models_frozen" / "video_feature_temporal_v1" / f"{name}_s{seed}" / "final.pt"
            if sha256_file(f) != summary["runs"][f"{name}_s{seed}"]["final_sha256"]:
                raise SystemExit(f"A01 {name}_s{seed}: the frozen weights differ from A01's summary")
            model = tv.new_model(h, torch.load(f, map_location="cpu", weights_only=True), device).eval()
            torch.cuda.synchronize(); t0 = time.time(); h.decode(model, seqs, device); torch.cuda.synchronize(); spent += time.time() - t0; del model
    finally:
        torch.use_deterministic_algorithms(True)
    return spent / sum(s.labels.size for s in seqs)


def timing(head, init_path: Path, fcfg: dict, batches: list, warm: int, total: int, device) -> dict:
    model = new_model(head, init_path, device); opt, sched, clip = tfa.make_optimiser(fcfg, model, total); dtr.seed_everything(0)
    times, sizes = [], []
    for i, b in enumerate(batches):
        torch.cuda.synchronize(); t0 = time.time(); tfa.update(head, model, b, opt, sched, clip, device); torch.cuda.synchronize()
        if i >= warm:
            times.append(time.time() - t0); sizes.append(int(sum(s.labels.size for s in b)))
    return {"times": times, "sizes": sizes, "model": model, "opt": opt, "sched": sched}


def check_fact(ctx: Ctx, head, data, init_path: Path, device) -> dict:
    ch, f = ctx.fv["checks"], ctx.f; a01 = ctx.a01(); tol = float(ch["reproducibility"]["tolerance"]); seed = ctx.seeds[0]
    n = int(ch["reproducibility"]["updates"]); b0 = data.epoch(a01, 0); rep = {}
    # timing first, from a fresh process state: 5 warm-up and 50 timed formal updates of seed 0 epoch 0, then one
    # epoch's monitor and resume save
    tm = ch["timing"]; warm, timed = int(tm["warmup_updates"]), int(tm["timed_updates"])
    t = timing(head, init_path, f, b0[:warm + timed], warm, ctx.total, device); t1 = time.time()
    monitor(head, t["model"], data.check, device); torch.cuda.synchronize(); t2 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tfa.save_resume(Path(tmp) / "r.pt", t["model"], t["opt"], t["sched"], 0, "timing", [])
    t3 = time.time(); a, b = fit_time(t["times"], t["sizes"])
    rep["timing"] = {"timed_updates": len(t["times"]), "mean_s": float(np.mean(t["times"])), "median_s": float(np.median(t["times"])), "fit_a_s": a,
                     "fit_b_s_per_sample": b, "mean_batch_samples": float(np.mean(t["sizes"])), "monitor_s": t2 - t1, "save_s": t3 - t2, "per_epoch_s": t3 - t1}
    del t
    torch.cuda.empty_cache()
    # reproducibility and resume (GPU; the six losses and every parameter, within the tolerance)
    ra_, rb_ = run_updates(head, init_path, b0[:n], f, ctx.total, device, seed), run_updates(head, init_path, b0[:n], f, ctx.total, device, seed)
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(ra_[3], rb_[3])), "params_max_abs": max_diff(ra_[0].state_dict(), rb_[0].state_dict())}
    with tempfile.TemporaryDirectory() as tmp:
        rep["resume"] = resume_check(head, init_path, b0[:n], f, ctx.total, device, seed, Path(tmp) / "resume.pt", reference=ra_)
    o = ra_[1].defaults; rep["optimizer"] = {"class": type(ra_[1]).__name__, "lr": o["lr"], "betas": list(o["betas"]), "eps": o["eps"], "weight_decay": o["weight_decay"],
                                             "grad_clip": float(f["optimizer"]["grad_clip"]), "bound": tfa.optimiser_bound(ra_[1], ra_[0])}
    rep["optimizer_matches_config"] = (rep["optimizer"]["class"] == f["optimizer"]["name"] and rep["optimizer"]["lr"] == float(f["optimizer"]["lr"])
                                       and rep["optimizer"]["betas"] == [float(x) for x in f["optimizer"]["betas"]] and rep["optimizer"]["eps"] == float(f["optimizer"]["eps"])
                                       and rep["optimizer"]["weight_decay"] == float(f["optimizer"]["weight_decay"]) and rep["optimizer"]["bound"])
    del ra_, rb_
    # padding (training-mode loss under the same RNG state, eval-mode predictions) and the gradient paths at the initial state
    model = new_model(head, init_path, device); batch = dd.collate(b0[0], device); rep["padding"] = padding_check(head, model, batch, b0[0], device, 1)
    dtr.seed_everything(seed); model.train(); model.zero_grad(set_to_none=True); loss, _ = head.loss(model, batch); loss.backward()
    rep["zero_or_missing_gradients"] = sorted(k for k, p in model.named_parameters() if p.grad is None or not float(p.grad.abs().max()) > 0 or not torch.isfinite(p.grad).all())
    rep["parameters"] = int(sum(p.numel() for p in model.parameters())); del model
    # isolation under five label sequences at the initial state, and the decoding times of FACT and the controls
    chosen = [("native" if s.is_native else "stream", s) for s in data.check]; iso_seqs = list(data.check) + list(data.streams[:200])
    init = new_model(head, init_path, device); rep["isolation_initial"] = fi.isolation(head, init, iso_seqs, device, seed=1); init.eval()
    torch.cuda.synchronize(); t0 = time.time(); [head.predict(init, torch.from_numpy(s.feat).to(device)) for s in data.streams[:200]]; torch.cuda.synchronize()
    rep["decode_seconds_per_sample"] = (time.time() - t0) / sum(s.labels.size for s in data.streams[:200]); del init
    rep["control_seconds_per_sample"] = control_seconds_per_sample(ctx, seed, list(data.streams[:200]), device)
    # the overfit smoke with its check points, and isolation at the overfit state
    model, log, out, points = overfit(head, init_path, chosen, f, ch["smoke"] | {"lr": f["optimizer"]["lr"]}, device)
    rep["smoke"] = {"log": log, "check_points": points, "updates": points[-1]["update"], "outcome": {k: v for k, v in out.items() if k != "rows"}, "rows": out["rows"]}
    rep["isolation_overfit"] = fi.isolation(head, model, iso_seqs, device, seed=2); del model
    rep["smoke"]["requirements"] = smoke_passed(out, [rep["isolation_initial"], rep["isolation_overfit"]])
    rep["ok"] = bool(rep["reproducibility"]["loss_max_abs"] <= tol and rep["reproducibility"]["params_max_abs"] <= tol
                     and rep["resume"]["params_max_abs"] <= tol and rep["resume"]["losses_max_abs"] <= tol and rep["resume"]["first_half_losses_max_abs"] <= tol
                     and rep["resume"]["bound"] and rep["resume"]["lr_equal"] and rep["optimizer_matches_config"] and rep["padding"]["loss_diff"] == 0.0
                     and rep["padding"]["padded_positions"] > 0 and rep["padding"]["predictions_equal"] and rep["smoke"]["requirements"]["all"])
    return rep


def plot_smoke(log: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); x = [r["update"] for r in log]
    ax[0].plot(x, [r["train_loss"] for r in log], label="training batches (augmented)"); ax[0].plot(x, [r["loss_total"] for r in log], label="eval, 16 sequences")
    for k in ("loss_frame", "loss_token", "loss_f2a", "loss_a2f", "loss_seg", "loss_smooth"):
        if k in log[0]:
            ax[1].plot(x, [r[k] for r in log], label=k[5:])
    for a_, t in zip(ax, ("official loss on the 16 check sequences", "loss terms (weighted contributions, eval)")):
        a_.set_title(t, fontsize=10); a_.set_xlabel("updates"); a_.set_yscale("log"); a_.legend(fontsize=7); a_.grid(alpha=0.3)
    fig.suptitle("Overfit smoke check on training data only; the smoke weights never start a formal run", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------------------------- stages
def preconditions(ctx: Ctx, stage: str) -> None:
    env = environment(); want = ctx.src["environment"]
    if any(str(env[k]) != str(v) for k, v in want.items()):
        raise SystemExit(f"environment {env} differs from the pinned {want}")
    for name in {"check": ("preparation.json",), "queue": ("preparation.json", "checks.json"), "run": ("preparation.json", "checks.json"),
                 "train_fit": ("preparation.json", "checks.json")}.get(stage, ()):
        rec = read_record(ctx, name)
        if rec.get("code") != code_digest() or rec.get("config_sha256") != ctx.config_sha():
            raise SystemExit(f"{name} is missing or was made by other code or another configuration; run that stage again first")
        if name == "checks.json" and rec.get("passed") is not True:
            raise SystemExit("the check did not pass")


def run_stage(ctx: Ctx, name: str, kind: str, fn) -> None:
    cap = float(ctx.fv["budget"]["cpu_hours_max" if kind == "cpu" else "gpu_hours_max"]); spent = ledger_hours(ctx.ledger, kind)
    if spent >= cap:
        raise SystemExit(f"the {kind.upper()} budget is spent ({spent:.3f} of {cap} h)")
    signal.signal(signal.SIGTERM, terminate); t0, started, ok, note = time.time(), now(), False, ""
    try:
        preconditions(ctx, name); fn(ctx); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, name, kind, started, time.time() - t0, ok, note)


def stage_prepare(ctx: Ctx) -> None:
    root, src, f = ctx.root, ctx.src, ctx.f; checked = {"task_sheet": sha256_file(pinned(root, ctx.fv["task_sheet"]))}
    for k in ("a01_config", "a01_preparation", "a01_summary", "f02_config", "f02_summary", "a05_config", "a05_preparation", "a03_config"):
        checked[k] = sha256_file(pinned(root, src[k]))
    a01 = ctx.a01(); a01_env = a01.verify_sources(); a01sum, f02sum, a05prep = ctx.json_pin("a01_summary"), ctx.json_pin("f02_summary"), ctx.json_pin("a05_preparation")
    official = official_state(root / f["repo"], f["commit"])
    lib, lib_counts = library(ctx)
    if a01.prep["sequence_file"]["sha256"] != a05prep["sequence_file"]["sha256"]:
        raise SystemExit("A01's and A05's sequence file records differ")
    rule = rule_of(lib, f, int(ctx.fv["data"]["native_per_epoch"])); head = make_head(ctx, rule)
    ids = {q["id"] for q in lib} | {session_of(q["id"]) for q in lib if q["kind"] == "native"}
    rec = {"config": ctx.config, "config_sha256": ctx.config_sha(), "code": code_digest(), "environment": environment(), "a01_environment": a01_env,
           "checked": checked, "official": official, "expanded_config_sha256": hashlib.sha256(head.cfg.dump().encode()).hexdigest(),
           "token_rule": rule.__dict__, "library": lib_counts, "sequence_file": a01.prep["sequence_file"], "seeds": {}, "inits": {}, "plans": {}, "caches": {},
           "a01_frozen": {}, "v01_outputs": {}, "batches": {}, "spot_check": {}, "training_stream_samples": {},
           "development_counts": a01sum["counts"]}
    lb = ts.label_batches(lib); d = ctx.fv["data"]
    for s in ctx.seeds:
        plan, plan_sha = a01.plan(s); key = run_key(s)
        epochs = [lb.epoch(s, e, int(d["native_per_epoch"]), int(d["native_window"]), int(d["streams_per_batch"]), int(d["natives_per_batch"])) for e in range(ctx.epochs)]
        if [[tla.batch_digest(x) for x in ep] for ep in epochs] != plan["epochs"]:
            raise SystemExit(f"seed {s}: the label-only batches differ from A01's plan")
        n_batches = sum(len(ep) for ep in epochs); samples = int(sum(q.labels.size for ep in epochs for b in ep for q in b))
        dst = ctx.out / "plans" / f"plan_s{s}.json"; rec["plans"][str(s)] = {"a01_sha256": plan_sha, **copy_init(a01.out / "plans" / f"plan_s{s}.json", dst, root),
                                                                              "updates": n_batches, "samples": samples}
        f02_init = root / "runs" / "fact_asrf_paired_v1" / "inits" / f"{key}.pt"; want = f02sum["runs"][key]["init_sha256"]
        if sha256_file(f02_init) != want:
            raise SystemExit(f"{key}: F02's initial state differs from F02's summary")
        rec["inits"][key] = {**copy_init(f02_init, ctx.out / "inits" / f"{key}.pt", root), "f02_init_sha256": want, **strict_init(head, f02_init)}
        train = a01.out / "features" / "train" / f"ordered_s{s}.npz"; dev = a01.out / "features" / "development" / f"ordered_s{s}.npz"
        tsha, dsha = sha256_file(train), sha256_file(dev); old = a05prep["caches"][str(s)]
        if tsha != old["train"]["sha256"] or dsha != old["development"]["sha256"]:
            raise SystemExit(f"seed {s}: a cache's full sha256 differs from A05's preparation record")
        with np.load(train) as z:
            ident = str(z["identity"])
        if ident != a01.prep["features"][str(s)]["identity"]:
            raise SystemExit(f"seed {s}: the training cache's identity differs from A01's record")
        rec["caches"][str(s)] = {"train": {"path": str(train.relative_to(root)), "sha256": tsha, "identity": ident},
                                 "development": {"path": str(dev.relative_to(root)), "sha256": dsha, "note": "hashed only; not opened before the freeze"}}
        for h in ("linear", "mstcn", "asrf"):
            fz = root / "models_frozen" / "video_feature_temporal_v1" / f"{h}_s{s}" / "final.pt"; w = a01sum["runs"][f"{h}_s{s}"]["final_sha256"]
            if sha256_file(fz) != w:
                raise SystemExit(f"{h}_s{s}: A01's frozen final weights differ from its summary")
            rec["a01_frozen"][f"{h}_s{s}"] = w
        for h in ("selection", "calibration"):
            rec["v01_outputs"][f"{h}_s{s}"] = sha256_file(pinned(root, a01.vt["source"]["outputs"][h][s]))
        data = tv.TrainSet(a01, s); rec["batches"][str(s)] = check_batches(data, plan, a01); rec["spot_check"][str(s)] = spot_check(train, data)
        if not rec["spot_check"][str(s)]["equal"]:
            raise SystemExit(f"seed {s}: the float32 spot check differs from the source cache")
        rec["training_stream_samples"][str(s)] = int(sum(q.labels.size for q in data.streams)); del data
        rec["seeds"][str(s)] = {"plan_sha256": plan_sha, "feature_identity": a01.prep["features"][str(s)]["identity"]}
        print(f"seed {s}: plan, batches, initial state, caches and controls verified", flush=True)
    flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
    for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
        pv.write_atomic_json(p, flat)


def stage_check(ctx: Ctx) -> None:
    tfa.determinism(ARM); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction()); t0 = time.time()
    prep = read_record(ctx, "preparation.json"); a01 = ctx.a01(); rule = fh.TokenRule(**prep["token_rule"]); check_rule(rule, ctx.f); head = make_head(ctx, rule)
    if official_state(ctx.root / ctx.f["repo"], ctx.f["commit"])["digest"] != prep["official"]["digest"]:
        raise SystemExit("the official checkout changed since the preparation")
    rec = {"started": now(), "code": code_digest(), "config_sha256": ctx.config_sha(), "init": {}, "passed": False}
    for s in ctx.seeds:
        key = run_key(s); init = ctx.root / prep["inits"][key]["path"]
        if sha256_file(init) != prep["inits"][key]["sha256"]:
            raise SystemExit(f"{key}: the initial state differs from the preparation record")
        rec["init"][key] = strict_init(head, init)
    s = ctx.seeds[0]; data = tv.TrainSet(a01, s)
    rec["fact"] = check_fact(ctx, head, data, ctx.root / prep["inits"][run_key(s)]["path"], device); smoke = rec["fact"]["smoke"]; del data
    print(json.dumps({k: v for k, v in rec["fact"].items() if k not in ("smoke", "isolation_initial", "isolation_overfit")}, default=str), flush=True)
    print(json.dumps({"smoke": smoke["requirements"], "updates": smoke["updates"], "outcome": smoke["outcome"]}, default=str), flush=True)
    ctx.results.mkdir(parents=True, exist_ok=True); plot_smoke(smoke["log"], ctx.results / "smoke_fit.png")
    tm, b = rec["fact"]["timing"], ctx.fv["budget"]
    per_run = {run_key(s): run_hours(tm["fit_a_s"], tm["fit_b_s_per_sample"], prep["plans"][str(s)]["updates"], prep["plans"][str(s)]["samples"], ctx.epochs, tm["per_epoch_s"])
               for s in ctx.seeds}
    cal, sel = (int(prep["development_counts"][h]["samples"]) for h in ("calibration", "selection")); seeds = len(ctx.seeds)
    fact_samples = sum(prep["training_stream_samples"].values()) + 2 * (cal + sel) * seeds      # train-fit, each half once, one fact-check pass
    control_samples = (3 * cal + sel) * seeds                                                     # calibration, dry run, selection (both halves)
    post = (rec["fact"]["decode_seconds_per_sample"] * fact_samples + rec["fact"]["control_seconds_per_sample"] * control_samples) / 3600 + STAGE_OVERHEAD_HOURS
    rec["hours_per_run"], rec["post_training_hours"] = per_run, post
    rec["post_training_basis"] = {"fact_samples": fact_samples, "control_samples": control_samples, "stage_overhead_hours": STAGE_OVERHEAD_HOURS}
    rec["projection"] = projection(ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600, per_run, list(per_run), post, float(b["gpu_reserve_hours"]),
                                   float(b["gpu_hours_max"]))
    rec["identities"] = identities(ctx, head, prep)
    rec["passed"] = bool(rec["fact"]["ok"] and rec["projection"]["within_cap"])
    rec["finished"] = now(); flat = json.loads(json.dumps(rec, default=str)); lib, _ = library(ctx)
    assert_no_ids(flat, {q["id"] for q in lib} | {session_of(q["id"]) for q in lib if q["kind"] == "native"}, "checks")
    for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
        pv.write_atomic_json(p, flat)
    print(json.dumps({k: rec[k] for k in ("passed", "hours_per_run", "post_training_hours", "projection")}, indent=1, default=str), flush=True)
    if not rec["fact"]["smoke"]["requirements"]["all"]:
        raise SystemExit("the overfit smoke check failed; its curve is saved and this round stops (no extra updates, no other optimiser)")
    if not rec["passed"]:
        raise SystemExit("a check failed or the projection exceeds the cap; see checks.json")


def train(ctx: Ctx, seed: int, device) -> None:
    prep = read_record(ctx, "preparation.json"); a01 = ctx.a01(); key = run_key(seed); c = prep["caches"][str(seed)]["train"]
    if sha256_file(ctx.root / c["path"]) != c["sha256"]:
        raise SystemExit(f"seed {seed}: the training cache differs from the preparation record")
    rule = fh.TokenRule(**prep["token_rule"]); check_rule(rule, ctx.f); head = make_head(ctx, rule)
    if official_state(ctx.root / ctx.f["repo"], ctx.f["commit"])["digest"] != prep["official"]["digest"]:
        raise SystemExit("the official checkout changed since the preparation")
    init = ctx.root / prep["inits"][key]["path"]
    if sha256_file(init) != prep["inits"][key]["sha256"]:
        raise SystemExit(f"{key}: the initial state differs from the preparation record")
    data = tv.TrainSet(a01, seed); plan, plan_sha = a01.plan(seed); ident = identity(ctx, seed, head, prep)
    run_dir, ck = ctx.out / "train" / key, ctx.ckpt / key; man_path = run_dir / "run_manifest.json"; state = run_state(man_path, ident, ctx.total)
    if state == "done":
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    if state == "other":
        raise SystemExit(f"{key}: a finished record of another identity exists in {run_dir}; it is never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    model = new_model(head, init, device); opt, sched, clip = tfa.make_optimiser(ctx.f, model, ctx.total); completed, records, resumed = 0, [], []
    dtr.seed_everything(seed)            # dropout and masking draw from this state; a resume restores the saved one
    if (ck / "resume.pt").is_file():
        completed, records = tfa.load_resume(ck / "resume.pt", model, opt, sched, ident); resumed.append(completed)
        print(f"{key}: resumed after {completed} updates", flush=True)
    if completed % ctx.per_epoch:
        raise SystemExit(f"{key}: a resume after {completed} updates is not at an epoch boundary")
    log = run_dir / "train_log.jsonl"
    if completed == 0:
        log.write_text("")
    print(json.dumps({"run": key, "identity": ident[:16], "completed": completed}), flush=True)
    for epoch in range(completed // ctx.per_epoch, ctx.epochs):
        torch.cuda.reset_peak_memory_stats(device); t0 = time.time(); sums, gmax, lrs = defaultdict(float), 0.0, []; bs = data.epoch(a01, epoch)
        if [tla.batch_digest(x) for x in bs] != plan["epochs"][epoch]:
            raise SystemExit(f"{key} epoch {epoch}: the batches differ from the seed's plan")
        for b in bs:
            r = tfa.update(head, model, b, opt, sched, clip, device); completed += 1; lrs.append(r.pop("lr")); gmax = max(gmax, r["grad_norm"])
            for k, v in r.items():
                sums[k] += v
        t1 = time.time(); mon = monitor(head, model, data.check, device)
        rec = {"epoch": epoch, "updates": completed, "lr_first": lrs[0], "lr_last": lrs[-1], "loss": {k: v / len(bs) for k, v in sums.items() if k != "grad_norm"},
               "grad_norm_mean": sums["grad_norm"] / len(bs), "grad_norm_max": gmax, "seconds_train": t1 - t0, "seconds_monitor": time.time() - t1,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "monitor": mon}
        records.append(rec)
        with log.open("a") as fh_:
            fh_.write(json.dumps(rec) + "\n")
        tfa.save_resume(ck / "resume.pt", model, opt, sched, completed, ident, records)
        print(f"{key} epoch {epoch} updates {completed} lr {lrs[-1]:.2e} loss {rec['loss']['total']:.4f} | monitor F1 {mon['segment_f1_50']:.3f} "
              f"exact {mon['transcripts_exact']}/16 ({rec['seconds_train']:.0f}s)", flush=True)
    if completed != ctx.total:
        raise SystemExit(f"{key}: {completed} updates, expected {ctx.total}")
    torch.save(model.state_dict(), ck / "final.pt")
    man = {"identity": ident, "arm": ARM, "seed": seed, "completed_updates": completed, "total_updates": ctx.total, "checkpoint": {"rule": "final weights"},
           "final_checkpoint": str(ck / "final.pt"), "final_sha256": sha256_file(ck / "final.pt"), "plan_sha256": plan_sha, "init_sha256": prep["inits"][key]["sha256"],
           "official_commit": head.commit, "official_digest": prep["official"]["digest"], "expanded_config": head.cfg.dump(), "token_rule": rule.__dict__,
           "parameters": int(sum(p.numel() for p in model.parameters())), "interrupted": bool(resumed), "resumed_after_updates": resumed,
           "minutes": sum(r["seconds_train"] + r["seconds_monitor"] for r in records) / 60, "records": records}
    tmp = man_path.with_suffix(".tmp"); tmp.write_text(json.dumps(man, indent=1, default=str)); tmp.replace(man_path); (ck / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"run": key, "updates": completed, "minutes": round(man["minutes"], 1)}), flush=True)


def stage_queue(ctx: Ctx, config: str, logs: str = "logs") -> None:
    prep, check = read_record(ctx, "preparation.json"), read_record(ctx, "checks.json"); head = make_head(ctx, fh.TokenRule(**prep["token_rule"]))
    idents = identities(ctx, head, prep); b = ctx.fv["budget"]
    if check.get("identities") != idents:
        raise SystemExit("the passed check record carries other run identities; run the check again")
    status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]:
        raise SystemExit("the status file belongs to other identities")
    status.setdefault("jobs", {})
    for k in ctx.fv["queue"]:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    pv.write_atomic_json(status_path, status)
    for key in ctx.fv["queue"]:
        s = key.rsplit("_s", 1)[1]; man = ctx.out / "train" / key / "run_manifest.json"; st = run_state(man, idents[key], ctx.total)
        if st == "done":
            status["jobs"][key]["state"] = "done"; pv.write_atomic_json(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        if st == "other":
            raise SystemExit(f"{key}: a finished record of another identity exists; it is never overwritten")
        done_hours = [v["hours"] for v in status["jobs"].values() if v.get("state") == "done" and "hours" in v]
        per_run = {k: max([float(check["hours_per_run"][k]), *done_hours]) for k in ctx.fv["queue"]}
        remaining = [k for k in ctx.fv["queue"] if run_state(ctx.out / "train" / k / "run_manifest.json", idents[k], ctx.total) != "done"]
        proj = projection(ledger_hours(ctx.ledger, "gpu"), per_run, remaining, float(check["post_training_hours"]), float(b["gpu_reserve_hours"]), float(b["gpu_hours_max"]))
        status["projection"] = {"at": now(), "before": key, **proj}
        if not proj["within_cap"]:
            status["stopped"] = {"at": now(), "reason": f"projected {proj['total']:.2f} GPU hours exceed the cap {proj['cap']}", "before": key}
            pv.write_atomic_json(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / logs / f"{NS}_{key}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/train_fact_video_feature.py", "--config", config, "--seed", s]
        status["jobs"][key] = {"state": "running", "identity": idents[key], "seed": int(s), "log": str(log.relative_to(ROOT)), "started": now()}
        pv.write_atomic_json(status_path, status); started, t0 = now(), time.time(); ctx.inflight = (key, started, t0)
        with log.open("a") as fh_:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh_, stderr=subprocess.STDOUT).returncode
        ctx.inflight = None; hours = (time.time() - t0) / 3600; ok = rc == 0 and run_state(man, idents[key], ctx.total) == "done"
        ensure_run_line(ctx.ledger, key, started, hours * 3600, f"the run wrote no ledger line (return code {rc}); booked by the queue")
        status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else "failed", "returncode": rc, "finished": now(), "hours": hours}
        if ok:
            m = json.loads(man.read_text()); status["jobs"][key].update({"updates": m["completed_updates"], "final_sha256": m["final_sha256"], "interrupted": m["interrupted"]})
        else:
            status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        pv.write_atomic_json(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.3f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; locate the cause and run the queue again (finished runs are skipped, this one resumes under its identity)")
    import scripts.fact_video_feature_post as post   # outside the training identity
    status["frozen"] = post.freeze(ctx, idents, prep); status["finished"] = now(); pv.write_atomic_json(status_path, status); print("queue finished and frozen", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); g = ap.add_mutually_exclusive_group()
    for n in ("prepare", "check", "queue"):
        g.add_argument(f"--{n.replace('_', '-')}", dest=n, action="store_true")
    ap.add_argument("--seed", type=int); ap.add_argument("--logs", default="logs")
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
            run_h = sum(float(x["hours"]) for x in (json.loads(line) for line in ctx.ledger.read_text().splitlines() if line.strip())
                        if x["job"].startswith("train ") and x["started"] >= started)
            ledger_append(ctx.ledger, "queue overhead and freeze", "cpu", started, max(0.0, time.time() - t0 - run_h * 3600), ok, note)
    elif a.seed in ctx.seeds:
        signal.signal(signal.SIGTERM, terminate); t0, started, ok, note = time.time(), now(), False, ""
        try:
            preconditions(ctx, "run"); tfa.determinism(ARM); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
            train(ctx, a.seed, device); ok = True
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:   # every run books its own GPU time, also when it fails or is interrupted
            ledger_append(ctx.ledger, f"train {run_key(a.seed)}", "gpu", started, time.time() - t0, ok, note)
    else:
        raise SystemExit("pass --prepare, --check, --queue or --seed with one of the round's seeds")


if __name__ == "__main__":
    main()
