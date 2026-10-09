#!/usr/bin/env python
"""Train one arm and seed of the paired FACT-ASRF round (plan docs/superpowers/plans/2026-10-08-fact-asrf-training.md;
configs/experiments/fact_asrf_paired_v1.yaml) on the frozen E1 512-d features. Only the training split is built here;
the development halves belong to scripts/report_fact_asrf.py.

  --prepare                  the three seeds' batch plans (with ASRF's class weights and pos_weight from each seed's
                             epoch-0 labels) and the six initial states; every plan batch unpadded and compared with
                             its sequences; the integration stage's alignment reused when every input and the bank
                             digest are unchanged. A rerun must reproduce the same files. Writes preparation.json.
  --smoke --arm A --seed 0   same-state reproducibility, resume (3 + 3 against 6 updates), padding labels without effect,
                             the optimiser's type, settings and binding, zero gradients, and timing; for FACT also the
                             GTEA overfit on the 16 fixed training sequences, inference label isolation and the
                             gradient paths with their zero-gradient sizes.
  --arm A --seed N           one formal run: 50 epochs x 375 updates, the final weights, a monitor on the 16 fixed
                             training sequences after every epoch; resume state after every epoch.

  python scripts/train_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # FACT trains with deterministic algorithms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.fact_integration import TrainOnly, gradient_paths, gradient_sizes, isolation, outcome, term_losses  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_l1_asrf import batch_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.temporal.dev_data import Seq, collate, native_seq  # noqa: E402
from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights  # noqa: E402
from echo_routing.temporal.dev_run import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor, seed_everything  # noqa: E402
from echo_routing.temporal.fact_head import FACTHead, token_rule  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

NS = "fact_asrf_paired_v1"
ARMS = ("FACT", "ASRF")
N_CLASSES = 5
ASRF_BUILD = {"in_channel": 512, "n_classes": 5, "n_features": 64, "n_layers": 10, "n_stages": 4, "n_stages_asb": 4, "n_stages_brb": 4}   # ASRFHead.build
CODE = ("scripts/train_fact_asrf.py", "scripts/fact_integration.py", "scripts/train_l1_asrf.py", "echo_routing/temporal/fact_head.py",
        "echo_routing/temporal/dev_heads.py", "echo_routing/temporal/dev_data.py", "echo_routing/temporal/dev_train.py",
        "echo_routing/compose/stream_assembler.py", "echo_routing/compose/banks.py")
TRAINING_KEYS = ("inputs", "seeds", "data", "init", "rng", "checkpoint", "monitor")   # the config sections a run's identity covers


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """The round's configuration and the training split (TrainOnly builds no development half)."""

    def __init__(self, cfg: dict, paths) -> None:
        self.cfg, self.fa, self.paths, self.root = cfg, cfg["fact_asrf_paired"], paths, paths.repo_root
        self.tr = self.fa["data"]; self.data = TrainOnly(cfg, self.fa, paths); self.server = self.root / self.fa["outputs"]["server_dir"]
        if int(self.tr["epochs"]) * int(self.tr["updates_per_epoch"]) != int(self.tr["total_updates"]):
            raise SystemExit("epochs x updates_per_epoch differs from total_updates")
        ov = self.fa["arms"]["FACT"]["dataset_overrides"]
        self.rule = token_rule([s.labels for s in self.data.streams], int(self.tr["native_per_epoch"]), float(ov["token_ratio"]))
        if self.rule.ntoken != int(ov["expected_ntoken"]) or abs(self.rule.average_transcript - float(ov["expected_average_transcript"])) > 1e-12:
            raise SystemExit(f"the token rule {self.rule} differs from the integration stage")

    def batches(self, seed: int, epoch: int) -> list[list[Seq]]:
        return self.data.batches(self.tr, seed, epoch)

    def plan_path(self, seed: int) -> Path:
        return self.server / "plans" / f"plan_s{seed}.json"

    def init_path(self, arm: str, seed: int) -> Path:
        return self.server / "inits" / f"{arm}_s{seed}.pt"


def monitor_set(ctx: Ctx) -> list[tuple[str, Seq]]:
    """The integration stage's 16 fixed training sequences (five native windows and eleven bank streams)."""
    rows = json.loads((ctx.root / ctx.fa["inputs"]["integration_alignment"]["monitor_set"]).read_text())["overfit_set"]
    by_id, train, out = {s.seq_id: s for s in ctx.data.streams}, set(ctx.data.train_clips), []
    for row in rows:
        if row["kind"] == "single_view_native":
            vid = row["seq_id"].rsplit("@", 1)[0]
            if vid not in train:
                raise SystemExit("a monitor sequence is not a training clip")
            vf = ctx.data.store.get(vid, "orig"); out.append((row["kind"], native_seq(vid, ctx.data.labels[vid], vf, 0, min(256, vf.n), row["seq_id"])))
        else:
            out.append((row["kind"], by_id[row["seq_id"]]))
    if len(out) != 16:
        raise SystemExit(f"{len(out)} monitor sequences, expected 16")
    return out


# ----------------------------------------------------------------------------------------------- plans and initial states
def make_plan(ctx: Ctx, seed: int, check_unpad: bool = False) -> tuple[dict, int]:
    """One seed's batch plan for both arms; optionally every batch unpadded and compared with its sequences."""
    epochs, first, checked = [], None, 0
    for e in range(int(ctx.tr["epochs"])):
        bs = ctx.batches(seed, e)
        if len(bs) != int(ctx.tr["updates_per_epoch"]):
            raise SystemExit(f"seed {seed} epoch {e}: {len(bs)} updates, expected {ctx.tr['updates_per_epoch']}")
        epochs.append([batch_digest(b) for b in bs]); first = first if e else [s for b in bs for s in b]
        if check_unpad:
            for b in bs:
                xs, labs = FACTHead.unpad(collate(b))
                if any(not np.array_equal(x.numpy(), s.feat.astype(np.float32)) or not np.array_equal(y.numpy(), s.labels) for x, y, s in zip(xs, labs, b)):
                    raise SystemExit(f"seed {seed} epoch {e}: an unpadded batch differs from its sequences")
                checked += 1
    cw, pw = asrf_weights(first, num_classes=N_CLASSES)
    return {"seed": seed, "epochs": epochs, "asrf_class_weight": cw.tolist(), "asrf_pos_weight": float(pw)}, checked


def save_or_check_json(path: Path, obj: dict) -> str:
    if path.is_file():
        if json.loads(path.read_text()) != json.loads(json.dumps(obj)):
            raise SystemExit(f"{path} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj))
    return sha256_file(path)


def load_plan(ctx: Ctx, seed: int) -> tuple[dict, str]:
    path = ctx.plan_path(seed)
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run --prepare first")
    return json.loads(path.read_text()), sha256_file(path)


def initial_state(head, seed: int) -> dict:
    seed_everything(seed)
    return {k: v.detach().cpu().clone() for k, v in head.build().state_dict().items()}


def save_or_check_init(path: Path, state: dict) -> str:
    if path.is_file():
        old = torch.load(path, map_location="cpu", weights_only=True)
        if set(old) != set(state) or any(not torch.equal(old[k], state[k]) for k in state):
            raise SystemExit(f"{path} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)
    return sha256_file(path)


# ----------------------------------------------------------------------------------------------- heads and optimisers
def make_head(ctx: Ctx, arm: str, plan: dict, device):
    a = ctx.fa["arms"][arm]; repo = ctx.root / a["repo"]
    if arm == "FACT":
        ov = a["dataset_overrides"]; head = FACTHead(repo, int(ctx.fa["inputs"]["feature_dim"]), ctx.rule, num_classes=int(ov["n_classes"]), base_config=a["base_config"])
        c = head.cfg
        if c.FACT.trans is not bool(ov["trans"]) or c.Loss.match != ov["match"] or list(c.bg_class) != list(ov["bg_class"]) or int(c.FACT.ntoken) != int(ov["expected_ntoken"]):
            raise SystemExit("the expanded FACT configuration differs from the dataset overrides")
    elif arm == "ASRF":
        if {k: int(v) for k, v in a["model"].items()} != ASRF_BUILD:
            raise SystemExit(f"the config's ASRF architecture {a['model']} differs from what ASRFHead.build constructs")
        head = ASRFHead(repo, int(a["model"]["in_channel"]), torch.tensor(plan["asrf_class_weight"], dtype=torch.float32), torch.tensor(plan["asrf_pos_weight"]),
                        num_classes=int(a["model"]["n_classes"]), lambda_b=float(a["lambda_b"]), boundary_th=float(a["boundary_th"]), device=device)
    else:
        raise SystemExit(f"unknown arm {arm}")
    if head.commit != a["commit"]:
        raise SystemExit(f"{arm} commit {head.commit} != {a['commit']}")
    return head


def lr_lambda(schedule: dict, total: int):
    """Multiplier of the base learning rate for the update after `completed` updates."""
    if schedule["name"] == "step":
        at, factor = int(schedule["decay_at_completed_updates"]), float(schedule["factor"])
        return lambda completed: 1.0 if completed < at else factor
    if schedule["name"] == "warmup_cosine":
        warm = int(schedule["warmup_updates"])
        return lambda completed: lr_factor(completed, warm, total)
    raise SystemExit(f"unknown schedule {schedule['name']}")


def make_optimiser(arm_cfg: dict, model: torch.nn.Module, total: int) -> tuple:
    """The arm's own optimiser and schedule; every key is required."""
    o = arm_cfg["optimizer"]; cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[o["name"]]
    opt = cls(model.parameters(), lr=float(o["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(arm_cfg["schedule"], total)), float(o["grad_clip"])


def determinism(arm: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(arm == "FACT")   # ASRF's CE (nll_loss2d) has no deterministic implementation


def optimiser_bound(opt, model) -> bool:
    ids = {id(p) for p in model.parameters()}
    return all(id(p) in ids for g in opt.param_groups for p in g["params"]) and sum(len(g["params"]) for g in opt.param_groups) == len(ids)


def update(head, model, seqs: list[Seq], opt, sched, clip: float, device) -> dict:
    lr = float(opt.param_groups[0]["lr"]); model.train(); loss, parts = head.loss(model, collate(seqs, device))
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {parts}")
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step(); sched.step()
    return {"total": float(loss.detach()), **parts, "grad_norm": gn, "lr": lr}


# ----------------------------------------------------------------------------------------------- resume
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
    """Into the model being trained and the optimiser and scheduler built for it (the binding fixed in 1122a68)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); set_rng_state(st["rng"])
    return int(st["completed_updates"]), list(st["records"])


def final_checkpoint(records: list[dict], total: int) -> dict:
    """The checkpoint rule: the weights after the last update. No other record field is read."""
    if not records or int(records[-1]["updates"]) != int(total):
        raise SystemExit(f"the run stopped after {records[-1]['updates'] if records else 0} of {total} updates")
    return {"updates": int(total), "rule": "final weights"}


# ----------------------------------------------------------------------------------------------- identity
def identity(ctx: Ctx, arm: str, seed: int, head, plan_sha: str, init_sha: str) -> str:
    d = ctx.data
    payload = {"experiment": ctx.cfg.get("experiment"), "arm": arm, "seed": int(seed), "config": {k: ctx.fa[k] for k in TRAINING_KEYS},
               "arm_config": ctx.fa["arms"][arm], "official_commit": head.commit, "expanded_config": head.cfg.dump() if arm == "FACT" else None,
               "token_rule": ctx.rule.__dict__, "inputs": {"protocol": d.proto_sha, "manifest": d.fman_sha, "cache_key": d.key, "bank_digest": d.bank_digest,
               "index_sha256": {v: sha256_file(Path(d.fman["variants"][v]["dir"]) / "index.csv") for v in d.variants}},
               "plan_sha256": plan_sha, "init_sha256": init_sha, "total_updates": int(ctx.tr["total_updates"]),
               "code": {c: sha256_file(ctx.root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def smoke_stamp(fa: dict, root: Path, arm: str, seed: int) -> str:
    """What a smoke report was produced from: the arm's training configuration and smoke settings, the code, the seed's
    plan and the arm's initial state. The queue recomputes it before trusting a report's verdict and timing."""
    server = root / fa["outputs"]["server_dir"]
    payload = {"config": {k: fa[k] for k in TRAINING_KEYS}, "arm_config": fa["arms"][arm], "smoke": fa["smoke"], "arm": arm, "seed": int(seed),
               "code": {c: sha256_file(root / c) for c in CODE}, "plan_sha256": sha256_file(server / "plans" / f"plan_s{seed}.json"),
               "init_sha256": sha256_file(server / "inits" / f"{arm}_s{seed}.pt")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def finished_run(manifest: Path, ident: str):
    """True: finished with this identity and an intact final checkpoint; False: finished otherwise; None: not finished."""
    if not manifest.is_file():
        return None
    man = json.loads(manifest.read_text())
    return bool(man.get("identity") == ident and Path(man["final_checkpoint"]).is_file() and sha256_file(Path(man["final_checkpoint"])) == man["final_sha256"])


# ----------------------------------------------------------------------------------------------- monitor
@torch.no_grad()
def monitor(head, model, chosen: list[tuple[str, Seq]], device) -> dict:
    """Training-set fit diagnostic on the 16 fixed sequences; never used to choose anything."""
    dec = head.decode(model, [s for _, s in chosen], device); fa, f1, exact = [], [], 0
    for _, s in chosen:
        p = np.asarray(dec[s.seq_id]["labels"], int); ref = labels_to_segments(s.labels); pr = labels_to_segments(p)
        fa.append(float((p == s.labels).mean())); f1.append(segment_f1(pr, ref, 0.5)["f1"]); exact += int([g.label for g in pr] == [g.label for g in ref])
    return {"frame_accuracy": float(np.mean(fa)), "segment_f1_50": float(np.mean(f1)), "transcripts_exact": exact}


# ----------------------------------------------------------------------------------------------- smoke
def new_model(head, init_path: Path, device):
    model = head.build().to(device); model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True)); return model


def fact_overfit(ctx: Ctx, head, init_path: Path, chosen: list, device) -> tuple:
    """The fixed 16 training sequences, 1,000 alternating updates of Adam at the constant smoke rate, official
    augmentation on."""
    a, sm = ctx.fa["arms"]["FACT"], ctx.fa["smoke"]["fact_overfit"]; o = a["optimizer"]; seed = int(ctx.fa["smoke"]["seed"])
    seed_everything(seed); model = new_model(head, init_path, device); seqs = [s for _, s in chosen]
    opt = torch.optim.Adam(model.parameters(), lr=float(sm["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    batches, clip, log, sums = [seqs[:8], seqs[8:16]], float(o["grad_clip"]), [], defaultdict(float)
    for step in range(int(sm["updates"])):
        model.train(); loss, _ = head.loss(model, collate(batches[step % 2], device))
        opt.zero_grad(set_to_none=True); loss.backward(); sums["grad_norm"] += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step()
        sums["train_loss"] += float(loss)
        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                terms = {f"eval_{k}": float(v) for k, v in term_losses(head, model, seqs, device).items()}
            log.append({"update": step + 1, **{k: v / 50 for k, v in sums.items()}, **terms}); sums.clear()
            print(f"FACT overfit {step + 1}: train {log[-1]['train_loss']:.4f} eval total {terms['eval_total']:.4f}", flush=True)
    return model, log, outcome(head, model, chosen, device)


def timing(head, init_path: Path, arm_cfg: dict, batches: list, warm: int, device, total: int) -> dict:
    model = new_model(head, init_path, device); opt, sched, clip = make_optimiser(arm_cfg, model, total); seed_everything(0); times = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    for i, b in enumerate(batches):
        torch.cuda.synchronize(); t0 = time.time(); update(head, model, b, opt, sched, clip, device); torch.cuda.synchronize()
        if i >= warm:
            times.append(time.time() - t0)
    t = np.asarray(times)
    return {"timed_updates": int(t.size), "mean_s": float(t.mean()), "median_s": float(np.median(t)), "p90_s": float(np.percentile(t, 90)),
            "max_s": float(t.max()), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}, model, opt, sched


def smoke(ctx: Ctx, arm: str, seed: int, device) -> dict:
    sm, a = ctx.fa["smoke"], ctx.fa["arms"][arm]; tol, total = float(sm["tolerance"]), int(ctx.tr["total_updates"])
    plan, _ = load_plan(ctx, seed); head = make_head(ctx, arm, plan, device); init_path = ctx.init_path(arm, seed); determinism(arm)
    n = int(sm["reproducibility_updates"]); b6 = ctx.batches(seed, 0)[:n]; rep = {"arm": arm, "seed": seed}
    if [batch_digest(x) for x in b6] != plan["epochs"][0][:n]:
        raise SystemExit("smoke batches differ from the plan")

    def run(batches, state=None):
        model = new_model(head, init_path, device); opt, sched, clip = make_optimiser(a, model, total)
        if state:
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); set_rng_state(state["rng"])
        else:
            seed_everything(seed)
        return model, opt, sched, [update(head, model, b, opt, sched, clip, device)["total"] for b in batches]

    def max_diff(x, y):
        return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)
    ra, rb = run(b6), run(b6)
    rep["reproducibility"] = {"loss_max_abs": max(abs(p - q) for p, q in zip(ra[3], rb[3])), "params_max_abs": max_diff(ra[0].state_dict(), rb[0].state_dict())}
    rc = run(b6[: n // 2])
    with tempfile.TemporaryDirectory() as tmp:   # through disk, as a real resume
        p = Path(tmp) / "resume.pt"; save_resume(p, rc[0], rc[1], rc[2], n // 2, "smoke", [])
        model = new_model(head, init_path, device); opt, sched, clip = make_optimiser(a, model, total)
        done, _ = load_resume(p, model, opt, sched, "smoke")
    rep["resume_bound"] = optimiser_bound(opt, model)
    for b in b6[n // 2:]:
        update(head, model, b, opt, sched, clip, device)
    rep["resume"] = {"completed_before": done, "params_max_abs": max_diff(ra[0].state_dict(), model.state_dict()),
                     "lr_after": float(opt.param_groups[0]["lr"]), "lr_after_continuous": float(ra[1].param_groups[0]["lr"])}
    rep["optimizer"] = {"class": type(ra[1]).__name__, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in ra[1].defaults.items()
                                                        if k in ("lr", "betas", "eps", "weight_decay")}, "grad_clip": float(a["optimizer"]["grad_clip"])}
    rep["optimizer_matches_config"] = (rep["optimizer"]["class"] == a["optimizer"]["name"] and rep["optimizer"]["lr"] == float(a["optimizer"]["lr"])
                                       and rep["optimizer"]["weight_decay"] == float(a["optimizer"]["weight_decay"]) and rep["optimizer"]["eps"] == float(a["optimizer"]["eps"])
                                       and rep["optimizer"]["betas"] == [float(b) for b in a["optimizer"]["betas"]])
    model = new_model(head, init_path, device); batch = collate(b6[0], device)   # padding: same shapes and valid samples, other labels outside the mask
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    seed_everything(1); l1, _ = head.loss(model, batch); seed_everything(1); l2, _ = head.loss(model, alt)
    rep["padding"] = {"loss_diff": abs(float(l1) - float(l2)), "padded_positions": int((batch["valid"] == 0).sum())}
    if arm == "FACT":
        xs, labs = FACTHead.unpad(batch)
        rep["padding"]["unpadded_inputs_equal"] = all(np.array_equal(x.cpu().numpy(), s.feat.astype(np.float32)) and np.array_equal(y.cpu().numpy(), s.labels)
                                                       for x, y, s in zip(xs, labs, b6[0]))
    model.zero_grad(); l1.backward()
    rep["zero_or_missing_gradients"] = [k for k, p in model.named_parameters() if p.grad is None or not float(p.grad.abs().max()) > 0 or not torch.isfinite(p.grad).all()]
    rep["parameters"] = int(sum(p.numel() for p in model.parameters()))
    warm, timed = int(sm["timing"]["warmup_updates"]), int(sm["timing"]["timed_updates"])
    rep["timing"], tmodel, topt, tsched = timing(head, init_path, a, ctx.batches(seed, 0)[:warm + timed], warm, device, total)
    chosen = monitor_set(ctx); torch.cuda.synchronize(); t0 = time.time(); monitor(head, tmodel, chosen, device); torch.cuda.synchronize(); t1 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        save_resume(Path(tmp) / "r.pt", tmodel, topt, tsched, 0, "smoke", [])
    t2 = time.time(); epochs = int(ctx.tr["epochs"])
    rep["timing"].update({"monitor_s": t1 - t0, "save_s": t2 - t1,
                          "estimated_hours_per_run": (rep["timing"]["mean_s"] * total + (t2 - t0) * epochs) / 3600})
    if arm == "FACT":
        iso_seqs = [s for _, s in chosen] + list(ctx.data.streams[:200])
        seed_everything(seed); init = new_model(head, init_path, device)
        rep["isolation_initial"] = isolation(head, init, iso_seqs, device, seed=1)
        rep["gradient_paths_initial"] = gradient_paths(head, init, [s for _, s in chosen[:8]], device, seed)
        watch = rep["gradient_paths_initial"]["official_total"]["zero_gradient_parameters"]
        rep["zero_gradient_sizes_initial"] = gradient_sizes(init, [s for _, s in chosen[:8]], device, seed, watch); del init
        model, log, out = fact_overfit(ctx, head, init_path, chosen, device)
        rep["overfit"] = {"log": log, "outcome": {k: v for k, v in out.items() if k != "rows"}, "rows": out["rows"]}
        rep["isolation_overfit"] = isolation(head, model, iso_seqs, device, seed=2)
        rep["zero_gradient_sizes_overfit"] = gradient_sizes(model, [s for _, s in chosen[:8]], device, seed, watch)
        o = out; iso_ok = all(rep[k]["changed_predictions"] == 0 and rep[k]["max_abs_logit_difference"] == 0 and rep[k]["adapter_decode_differs"] == 0
                              for k in ("isolation_initial", "isolation_overfit"))
        rep["fact_requirements"] = {"transcripts_exact": o["transcript_exact"] == 16, "segment_f1_all_one": o["segment_f1_50_min"] == 1.0,
                                    "one_to_one": o["match_one_to_one"] == 16, "matched_classes_right": o["matched_token_class_correct_min"] == 1.0,
                                    "isolation": iso_ok}
    rep["ok"] = bool(rep["reproducibility"]["loss_max_abs"] <= tol and rep["reproducibility"]["params_max_abs"] <= tol and rep["resume"]["params_max_abs"] <= tol
                     and rep["resume_bound"] and rep["resume"]["lr_after"] == rep["resume"]["lr_after_continuous"] and rep["optimizer_matches_config"]
                     and rep["padding"]["loss_diff"] <= tol and rep["padding"]["padded_positions"] > 0 and rep["padding"].get("unpadded_inputs_equal", True)
                     and all(rep.get("fact_requirements", {}).values()))
    return rep


# ----------------------------------------------------------------------------------------------- prepare
def run_prepare(ctx: Ctx, device) -> None:
    t0 = time.time(); d = ctx.data; ia = ctx.fa["inputs"]["integration_alignment"]
    dc_path = ctx.root / ia["data_checks"]["path"]
    if sha256_file(dc_path) != ia["data_checks"]["sha256"]:
        raise SystemExit("the integration stage's data_checks.json differs from the config")
    dc = json.loads(dc_path.read_text())
    reuse = {"protocol": dc["inputs_sha256"]["protocol"] == d.proto_sha, "manifest": dc["inputs_sha256"]["e1_feature_manifest"] == d.fman_sha,
             "cache_key": dc["cache_key"] == d.key, "bank_digest": dc["bank_digest"] == d.bank_digest}
    if not all(reuse.values()):
        raise SystemExit(f"the integration stage's alignment cannot be reused: {reuse}")
    plans, inits, unpadded = {}, {}, 0
    for seed in [int(s) for s in ctx.fa["seeds"]]:
        plan, checked = make_plan(ctx, seed, check_unpad=True); unpadded += checked
        plans[seed] = {"sha256": save_or_check_json(ctx.plan_path(seed), plan), "epochs": len(plan["epochs"]), "updates": sum(len(e) for e in plan["epochs"]),
                       "asrf_class_weight": plan["asrf_class_weight"], "asrf_pos_weight": plan["asrf_pos_weight"]}
        for arm in ARMS:
            head = make_head(ctx, arm, plan, device); state = initial_state(head, seed)
            inits[f"{arm}_s{seed}"] = {"sha256": save_or_check_init(ctx.init_path(arm, seed), state),
                                       "parameters": int(sum(v.numel() for k, v in state.items() if k in dict(head.build().named_parameters())))}
    chosen = monitor_set(ctx)
    res = {"inputs_sha256": {"protocol": d.proto_sha, "e1_feature_manifest": d.fman_sha}, "cache_key": d.key, "bank_digest": d.bank_digest,
           "integration_alignment_reused": reuse, "token_rule": ctx.rule.__dict__, "plans": plans, "unpadded_batches_checked": unpadded,
           "initial_states": inits, "monitor_set": {"kinds": {k: sum(1 for kk, _ in chosen if kk == k) for k in dict.fromkeys(k for k, _ in chosen)},
                                                    "samples": [int(s.labels.size) for _, s in chosen]},
           "training_clips": len(d.train_clips), "training_sessions": len({session_of(v) for v in d.train_clips}), "bank_streams": len(d.streams),
           "seconds": time.time() - t0}
    assert_no_ids(res, d.ids(), "preparation"); out = ctx.root / ctx.fa["outputs"]["results_dir"] / "preparation.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ("integration_alignment_reused", "plans", "initial_states")}, indent=1))


# ----------------------------------------------------------------------------------------------- one run
def train_one(ctx: Ctx, arm: str, seed: int, device) -> None:
    plan, plan_sha = load_plan(ctx, seed); head = make_head(ctx, arm, plan, device); init_path = ctx.init_path(arm, seed)
    if not init_path.is_file():
        raise SystemExit(f"{init_path} is missing; run --prepare first")
    init_sha = sha256_file(init_path); ident = identity(ctx, arm, seed, head, plan_sha, init_sha); key = f"{arm}_s{seed}"; a = ctx.fa["arms"][arm]
    run_dir, ckpt = ctx.server / "train" / key, ctx.root / "checkpoints" / NS / key
    done = finished_run(run_dir / "run_manifest.json", ident)
    if done is True:
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    if done is False:
        raise SystemExit(f"{key}: a finished run of another identity exists in {run_dir}; move it before training again")
    determinism(arm); run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    total, per_epoch = int(ctx.tr["total_updates"]), int(ctx.tr["updates_per_epoch"]); chosen = monitor_set(ctx)
    model = new_model(head, init_path, device); opt, sched, clip = make_optimiser(a, model, total); completed, records = 0, []
    seed_everything(seed)   # dropout and masking draw from this state; resume restores the saved one
    if (ckpt / "resume.pt").is_file():
        completed, records = load_resume(ckpt / "resume.pt", model, opt, sched, ident); print(f"{key}: resumed after {completed} updates", flush=True)
    if completed % per_epoch:
        raise SystemExit(f"{key}: resume after {completed} updates is not at an epoch boundary")
    log = run_dir / "train_log.jsonl"
    if completed == 0:
        log.write_text("")
    print(json.dumps({"run": key, "identity": ident[:16], "completed": completed, "plan": plan_sha[:12], "init": init_sha[:12]}), flush=True)
    for epoch in range(completed // per_epoch, int(ctx.tr["epochs"])):
        torch.cuda.reset_peak_memory_stats(device); t0 = time.time(); sums, gmax, lrs = defaultdict(float), 0.0, []
        bs = ctx.batches(seed, epoch)
        if [batch_digest(x) for x in bs] != plan["epochs"][epoch]:
            raise SystemExit(f"{key} epoch {epoch}: batches differ from the seed's plan")
        for b in bs:
            r = update(head, model, b, opt, sched, clip, device); completed += 1; lrs.append(r.pop("lr")); gmax = max(gmax, r["grad_norm"])
            for k, v in r.items():
                sums[k] += v
        t1 = time.time(); model.eval(); mon = monitor(head, model, chosen, device)
        rec = {"epoch": epoch, "updates": completed, "lr_first": lrs[0], "lr_last": lrs[-1], "loss": {k: v / len(bs) for k, v in sums.items() if k != "grad_norm"},
               "grad_norm_mean": sums["grad_norm"] / len(bs), "grad_norm_max": gmax, "seconds_train": t1 - t0, "seconds_monitor": time.time() - t1,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "monitor": mon}
        records.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{key} epoch {epoch} updates {completed} lr {lrs[-1]:.2e} loss {rec['loss']['total']:.4f} grad {rec['grad_norm_mean']:.3f} | "
              f"monitor F1 {mon['segment_f1_50']:.3f} exact {mon['transcripts_exact']}/16 ({rec['seconds_train']:.0f}s)", flush=True)
        save_resume(ckpt / "resume.pt", model, opt, sched, completed, ident, records)
    chosen_ckpt = final_checkpoint(records, total); torch.save(model.state_dict(), ckpt / "final.pt")
    man = {"identity": ident, "arm": arm, "seed": seed, "completed_updates": completed, "checkpoint": chosen_ckpt, "final_checkpoint": str(ckpt / "final.pt"),
           "final_sha256": sha256_file(ckpt / "final.pt"), "plan": str(ctx.plan_path(seed)), "plan_sha256": plan_sha, "init": str(init_path), "init_sha256": init_sha,
           "official_commit": head.commit, "expanded_config": head.cfg.dump() if arm == "FACT" else None, "token_rule": ctx.rule.__dict__,
           "parameters": int(sum(p.numel() for p in model.parameters())), "minutes": sum(r["seconds_train"] + r["seconds_monitor"] for r in records) / 60, "records": records}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str)); (ckpt / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"run": key, "final_sha256": man["final_sha256"][:16], "minutes": round(man["minutes"], 1)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--prepare", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=ARMS); ap.add_argument("--seed", type=int)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths()
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); ctx = Ctx(cfg, paths)
    if a.prepare:
        run_prepare(ctx, device); return
    if a.arm is None or a.seed not in [int(s) for s in ctx.fa["seeds"]]:
        raise SystemExit("--arm and one of the round's seeds are required (or --prepare)")
    if a.smoke:
        rep = smoke(ctx, a.arm, a.seed, device); rep["stamp"] = smoke_stamp(ctx.fa, ctx.root, a.arm, a.seed)
        out = ctx.server / "smoke" / f"{a.arm}_s{a.seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
        assert_no_ids(rep, ctx.data.ids(), "smoke"); text = json.dumps(rep, indent=1, default=str); out.write_text(text)
        (ctx.root / ctx.fa["outputs"]["results_dir"] / f"smoke_{a.arm}.json").write_text(text)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("overfit", "gradient_paths_initial")}, indent=1, default=str))
        if not rep["ok"]:
            raise SystemExit("smoke checks failed")
        return
    train_one(ctx, a.arm, a.seed, device)


if __name__ == "__main__":
    main()
