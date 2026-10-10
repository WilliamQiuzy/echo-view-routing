#!/usr/bin/env python
"""Predicted-partition FACT: preparation, boundary-head training, smoke checks and one formal run
(docs/superpowers/plans/2026-10-09-fact-predicted-partition.md; configs/experiments/fact_predicted_partition_v1.yaml).
Training split only; the development halves belong to scripts/report_fact_predicted_partition.py.

  --prepare                    input digests; every batch of the three source plans rebuilt and compared (50 x 375);
                               each head's pos_weight from its seed's epoch-0 batches; the heads' initial states.
  --train-head --seed N        7,500 head updates on the source plan's epochs 0-19, the final weights, the training-set
                               report and the launch check (at least one cut on the seed's epoch-49 sequences).
  --freeze-heads               the three heads frozen with scripts/freeze_model.py once every launch check passed.
  --smoke --group G --seed 0   with the frozen head: reproducibility, 3 + 3 resume, padding, optimiser binding, frozen
                               head, label isolation (initial and fitted), zero gradients against the official model,
                               gradient paths, the 2,000-update fit on the fixed 16 training sequences, timing.
  --group G --seed N           one formal run: 50 epochs x 375 updates from the source initial state, final weights.

  python scripts/train_fact_predicted_partition.py --config configs/experiments/fact_predicted_partition_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # FACT and the head train under deterministic algorithms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.fact_integration as fi  # noqa: E402
import scripts.train_fact_asrf as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.diagnose_fact_localisation import TrainRows  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import write_atomic  # noqa: E402
from scripts.train_l1_asrf import batch_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.predicted_partition as pp  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.temporal.dev_data import Seq, collate  # noqa: E402
from echo_routing.temporal.dev_run import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import seed_everything  # noqa: E402
from echo_routing.temporal.fact_head import token_rule  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments  # noqa: E402

NS, SRC = "fact_predicted_partition_v1", "fact_asrf_paired_v1"
GROUPS = ("Uniform", "Predicted")
A4C, A5C = 2, 3
CODE = ("scripts/train_fact_predicted_partition.py", "echo_routing/temporal/predicted_partition.py", "scripts/train_fact_asrf.py",
        "scripts/fact_integration.py", "scripts/diagnose_fact_localisation.py", "scripts/train_l1_asrf.py", "echo_routing/temporal/fact_head.py",
        "echo_routing/temporal/dev_data.py", "echo_routing/temporal/dev_train.py", "echo_routing/temporal/segments.py",
        "echo_routing/compose/stream_assembler.py", "echo_routing/compose/banks.py", "echo_routing/compose/recipe_builder.py",
        "echo_routing/features/cache.py")
RUN_KEYS = ("source", "data", "groups", "partition", "head", "fact")   # the config sections a run's identity covers


# ----------------------------------------------------------------------------------------------- inputs and context
def load_round(config: str) -> tuple:
    """This round's config and the paired round's; the source config, summary, plans, initial states and frozen weights
    must match their pins, else stop."""
    cfg = load_cfg(SimpleNamespace(config=config, set=[])); fp = cfg["fact_predicted_partition"]; paths = load_paths(); src = fp["source"]; root = paths.repo_root
    for k in ("config", "summary", "monitor_set"):
        if sha256_file(root / src[k]["path"]) != src[k]["sha256"]:
            raise SystemExit(f"the source {k} differs from its pinned sha256; stopping")
    runs_ = json.loads((root / src["summary"]["path"]).read_text())["runs"]
    if (any(runs_[k]["final_sha256"] != v for k, v in src["frozen"].items()) or any(runs_[f"FACT_s{s}"]["init_sha256"] != v for s, v in src["fact_inits"].items())
            or any(runs_[f"FACT_s{s}"]["plan_sha256"] != v for s, v in src["plans"].items())):
        raise SystemExit("the pinned weights, initial states or plans differ from the paired round's summary")
    scfg = load_cfg(SimpleNamespace(config=src["config"]["path"], set=[]))
    if (cfg["protocol"]["hz"], cfg["selection"]["boundary_tolerance_s"]) != (scfg["protocol"]["hz"], scfg["selection"]["boundary_tolerance_s"]):
        raise SystemExit("the sampling rate or boundary tolerance differs from the paired round")
    return cfg, fp, scfg, paths


class PCtx:
    """What scripts.train_fact_asrf.monitor_set, load_plan and make_head read (fa, root, data, rule, plan_path), on the
    allowlist-filtered training rows; the paired round's Ctx is not instantiated."""

    def __init__(self, cfg: dict, scfg: dict, fp: dict, paths) -> None:
        self.cfg, self.fp, self.fa, self.paths, self.root = cfg, fp, scfg["fact_asrf_paired"], paths, paths.repo_root
        self.tr = self.fa["data"]; self.seeds = [int(s) for s in fp["seeds"]]
        self.out, self.results = self.root / fp["outputs"]["server_dir"], self.root / fp["outputs"]["results_dir"]
        self.ckpt, self.src_server = self.root / fp["outputs"]["checkpoint_dir"], self.root / self.fa["outputs"]["server_dir"]
        self.tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
        f, h = fp["fact"]["data"], fp["head"]["data"]
        if (int(f["epochs"]), int(f["updates_per_epoch"]), int(f["total_updates"])) != (int(self.tr["epochs"]), int(self.tr["updates_per_epoch"]), int(self.tr["total_updates"])) \
                or int(h["updates_per_epoch"]) != int(self.tr["updates_per_epoch"]) or int(h["epochs"]) * int(h["updates_per_epoch"]) != int(h["total_updates"]) \
                or int(h["epochs"]) > int(self.tr["epochs"]) or self.seeds != [int(s) for s in self.fa["seeds"]]:
            raise SystemExit("the FACT or head data plan differs from the source plan")
        if self.fa["inputs"]["integration_alignment"]["monitor_set"] != fp["source"]["monitor_set"]["path"]:
            raise SystemExit("the 16 fixed training sequences differ from the pinned file")
        self.data = TrainRows(scfg, self.fa, paths)
        ov = self.fa["arms"]["FACT"]["dataset_overrides"]
        self.rule = token_rule([s.labels for s in self.data.streams], int(self.tr["native_per_epoch"]), float(ov["token_ratio"]))
        if self.rule.ntoken != int(ov["expected_ntoken"]) or abs(self.rule.average_transcript - float(ov["expected_average_transcript"])) > 1e-12:
            raise SystemExit(f"the token rule {self.rule} differs from the paired round")

    def batches(self, seed: int, epoch: int) -> list[list[Seq]]:
        return self.data.batches(self.tr, seed, epoch)

    def plan_path(self, seed: int) -> Path:
        return self.src_server / "plans" / f"plan_s{seed}.json"

    def fact_init(self, seed: int) -> Path:
        path = self.src_server / "inits" / f"FACT_s{seed}.pt"
        if sha256_file(path) != self.fp["source"]["fact_inits"][seed]:
            raise SystemExit(f"{path} differs from its pinned sha256")
        return path

    def head_file(self, seed: int, kind: str) -> Path:
        return self.out / "heads" / f"head_s{seed}_{kind}"

    def frozen_head_dir(self, seed: int) -> Path:
        return self.root / "models_frozen" / NS / f"head_s{seed}"


def plan_batches(ctx: PCtx, seed: int, epoch: int, plan: dict) -> list[list[Seq]]:
    bs = ctx.batches(seed, epoch)
    if len(bs) != int(ctx.tr["updates_per_epoch"]) or [batch_digest(b) for b in bs] != plan["epochs"][epoch]:
        raise SystemExit(f"seed {seed} epoch {epoch}: the batches differ from the source plan")
    return bs


def checked_plan(ctx: PCtx, seed: int) -> tuple[dict, str]:
    plan, sha = tr.load_plan(ctx, seed)
    if sha != ctx.fp["source"]["plans"][seed]:
        raise SystemExit(f"seed {seed}: the source plan differs from its pinned sha256")
    return plan, sha


def as_tensors(seqs: list[Seq], device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return ([torch.from_numpy(s.feat.astype(np.float32)).to(device) for s in seqs], [torch.from_numpy(s.labels).to(device) for s in seqs])


def write_json(path: Path, obj: dict, ctx: PCtx, copy_to_results: bool = False) -> str:
    """Both id guards, then an atomic write; optionally the same text in the results directory (counts only)."""
    text = json.dumps(obj, indent=1, default=str); assert_no_ids(json.loads(text), ctx.data.ids(), path.name)
    if any(x in text for x in ctx.data.sessions()):
        raise SystemExit(f"{path.name}: a session id appears in the output")
    for p in ([path] + ([ctx.results / path.name] if copy_to_results else [])):
        p.parent.mkdir(parents=True, exist_ok=True); write_atomic(p, text)
    return text


def deterministic() -> None:
    tr.determinism("FACT")   # fp32, TF32 off, cuDNN deterministic, deterministic algorithms (also for the head)


# ----------------------------------------------------------------------------------------------- the boundary head
def new_head(ctx: PCtx, device) -> pp.BoundaryHead:
    return pp.BoundaryHead(int(ctx.fa["inputs"]["feature_dim"]), int(ctx.fp["head"]["hidden"])).to(device)


def epoch0_pos_weight(ctx: PCtx, seed: int, plan: dict) -> tuple[float, int, int]:
    labels = [s.labels for b in plan_batches(ctx, seed, 0, plan) for s in b]
    pos, neg = pp.boundary_counts(labels)
    return pp.head_pos_weight(labels), pos, neg


def head_identity(ctx: PCtx, seed: int, plan_sha: str, init_sha: str, pos_weight: float) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "seed": int(seed), "head": ctx.fp["head"], "data": ctx.fp["data"], "plan_sha256": plan_sha,
               "init_sha256": init_sha, "pos_weight": pos_weight, "inputs": {"protocol": ctx.data.proto_sha, "manifest": ctx.data.fman_sha, "bank": ctx.data.bank_digest},
               "code": {c: sha256_file(ctx.root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@torch.no_grad()
def head_report(ctx: PCtx, seed: int, plan: dict, head: pp.BoundaryHead, device) -> dict:
    """The trained head on the seed's epoch-49 training sequences only: cuts against reference changes (one to one
    within 0.5 s), cuts per native window, cuts near same-view joins, A4C/A5C changes found, and every valid cut."""
    hc = ctx.fp["head"]["decode"]; head.eval(); st = Counter(); natives = []
    for s in [s for b in plan_batches(ctx, seed, int(ctx.tr["epochs"]) - 1, plan) for s in b]:
        cuts = pp.head_cuts(head, torch.from_numpy(s.feat.astype(np.float32)).to(device), float(hc["threshold"]), int(hc["nms_radius"]))
        st["valid_cuts"] += len(cuts)
        if s.is_native:
            natives.append(len(cuts)); continue
        st["streams"] += 1; ref = boundaries_from_labels(s.labels); pairs = match_boundaries(np.asarray(cuts, float), ref, ctx.tol); hit = {int(ref[j]) for _, j in pairs}
        same = [int(j) for j in s.joins if s.labels[j] == s.labels[j - 1]]; ac = [int(c) for c in ref if {int(s.labels[c - 1]), int(s.labels[c])} == {A4C, A5C}]
        st.update({"stream_cuts": len(cuts), "reference_changes": int(ref.size), "matched": len(pairs), "same_view_joins": len(same),
                   "same_view_joins_with_a_cut": sum(any(abs(c - j) <= ctx.tol for c in cuts) for j in same), "a4c_a5c_changes": len(ac),
                   "a4c_a5c_changes_found": sum(c in hit for c in ac)})
    n = np.asarray(natives)
    return {"streams": int(st["streams"]), "valid_cuts": int(st["valid_cuts"]),
            "stream_boundary_precision": st["matched"] / st["stream_cuts"] if st["stream_cuts"] else None,
            "stream_boundary_recall": st["matched"] / st["reference_changes"] if st["reference_changes"] else None,
            "stream_counts": {k: int(st[k]) for k in ("stream_cuts", "reference_changes", "matched")},
            "native_windows": int(n.size), "cuts_per_native_window": float(n.mean()) if n.size else None, "native_windows_with_a_cut": int((n > 0).sum()),
            "same_view_join_cut_rate": st["same_view_joins_with_a_cut"] / st["same_view_joins"] if st["same_view_joins"] else None,
            "same_view_joins": int(st["same_view_joins"]), "a4c_a5c_change_recall": st["a4c_a5c_changes_found"] / st["a4c_a5c_changes"] if st["a4c_a5c_changes"] else None,
            "a4c_a5c_changes": int(st["a4c_a5c_changes"])}


def train_head(ctx: PCtx, seed: int, device) -> None:
    """The boundary head of one seed: 7,500 updates on the source plan's epochs 0-19, then the training-set report and
    the launch check. Rerunning must reproduce the same final weights."""
    t0 = time.time(); hc = ctx.fp["head"]; d = hc["data"]; total, epochs = int(d["total_updates"]), int(d["epochs"]); deterministic()
    prep = json.loads((ctx.out / "preparation.json").read_text())["seeds"][str(seed)]
    plan, plan_sha = checked_plan(ctx, seed); pw, _, _ = epoch0_pos_weight(ctx, seed, plan)
    init = ctx.head_file(seed, "init.pt"); init_sha = sha256_file(init)
    if pw != prep["pos_weight"] or init_sha != prep["head_init_sha256"]:
        raise SystemExit(f"seed {seed}: the pos_weight or initial head differs from the preparation")
    head = new_head(ctx, device); head.load_state_dict(torch.load(init, map_location=device, weights_only=True)); o = hc["optimizer"]
    opt = torch.optim.AdamW(head.parameters(), lr=float(o["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, tr.lr_lambda(hc["schedule"], total)); clip = float(o["grad_clip"]); pwt = torch.tensor(pw, device=device)
    seed_everything(seed); records, done = [], 0
    for epoch in range(epochs):
        sums, gmax, lrs = defaultdict(float), 0.0, []
        for b in plan_batches(ctx, seed, epoch, plan):
            xs, ys = as_tensors(b, device); head.train(); lrs.append(float(opt.param_groups[0]["lr"])); loss = pp.head_loss(head, xs, ys, pwt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"seed {seed} update {done}: non-finite head loss")
            opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), clip)); opt.step(); sched.step()
            done += 1; sums["loss"] += float(loss); sums["grad_norm"] += gn; gmax = max(gmax, gn)
        n = int(d["updates_per_epoch"])
        records.append({"epoch": epoch, "updates": done, "lr_first": lrs[0], "lr_last": lrs[-1], "loss": sums["loss"] / n, "grad_norm_mean": sums["grad_norm"] / n, "grad_norm_max": gmax})
        print(f"head s{seed} epoch {epoch} updates {done} loss {records[-1]['loss']:.4f} lr {lrs[-1]:.2e}", flush=True)
    tr.final_checkpoint(records, total)
    final = ctx.head_file(seed, "final.pt"); final.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, final)
    rep = head_report(ctx, seed, plan, head, device)
    man = {"identity": head_identity(ctx, seed, plan_sha, init_sha, pw), "seed": seed, "completed_updates": done, "final": str(final), "final_sha256": sha256_file(final),
           "init_sha256": init_sha, "plan_sha256": plan_sha, "pos_weight": pw, "parameters": int(sum(p.numel() for p in head.parameters())), "records": records,
           "training_report": rep, "launch_check_passed": rep["valid_cuts"] > 0, "gpu_seconds": time.time() - t0}
    write_json(ctx.out / "heads" / f"head_s{seed}.json", man, ctx); write_json(ctx.results / f"head_s{seed}.json", {k: v for k, v in man.items() if k != "final"}, ctx)
    print(json.dumps({"seed": seed, "final_sha256": man["final_sha256"][:16], **{k: rep[k] for k in ("valid_cuts", "stream_boundary_precision", "stream_boundary_recall")}}), flush=True)
    if not man["launch_check_passed"]:
        raise SystemExit(f"seed {seed}: the head proposed no cut on its epoch-49 training sequences; the launch stops here")


def run_freeze_heads(ctx: PCtx) -> None:
    """Freeze the three trained heads after every launch check passed; the identity is recomputed first."""
    mans = {}
    for seed in ctx.seeds:
        man = json.loads((ctx.out / "heads" / f"head_s{seed}.json").read_text()); plan, plan_sha = checked_plan(ctx, seed)
        pw, _, _ = epoch0_pos_weight(ctx, seed, plan)
        if (man["identity"] != head_identity(ctx, seed, plan_sha, man["init_sha256"], pw) or int(man["completed_updates"]) != int(ctx.fp["head"]["data"]["total_updates"])
                or sha256_file(Path(man["final"])) != man["final_sha256"] or not man["launch_check_passed"]):
            raise SystemExit(f"seed {seed}: the head's identity, updates, weights or launch check do not verify")
        mans[seed] = man
    for seed, man in mans.items():
        dest = ctx.frozen_head_dir(seed)
        if not dest.exists():
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, f"head_s{seed}", "--file", man["final"], "--file", str(ctx.out / "heads" / f"head_s{seed}.json"),
                            "--meta", f"identity={man['identity']}", "--meta", f"seed={seed}", "--notes",
                            "Boundary head of the predicted-partition round (2026-10-09); final weights after 7,500 updates; frozen before any FACT training."],
                           cwd=ctx.root, check=True, capture_output=True)
        if sha256_file(dest / Path(man["final"]).name) != man["final_sha256"]:
            raise SystemExit(f"seed {seed}: the frozen head differs from the trained one")
        print(json.dumps({"seed": seed, "frozen": str(dest), "sha256": man["final_sha256"][:16]}), flush=True)


def frozen_head(ctx: PCtx, seed: int, device) -> tuple:
    """The frozen head of a seed (eval mode, no gradient), its cut function and its weights' sha256."""
    d = ctx.frozen_head_dir(seed); files = {f["file"]: f["sha256"] for f in json.loads((d / "MANIFEST.json").read_text())["files"]}
    path = d / f"head_s{seed}_final.pt"; sha = sha256_file(path)
    if files.get(path.name) != sha:
        raise SystemExit(f"seed {seed}: the frozen head differs from its manifest")
    head = new_head(ctx, device); head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    dc = ctx.fp["head"]["decode"]
    return head, pp.head_cut_fn(head, float(dc["threshold"]), int(dc["nms_radius"])), sha


# ----------------------------------------------------------------------------------------------- FACT with a partition policy
def new_fact(ctx: PCtx, head, group: str, cut_fn, seed: int, device):
    """The official FACT at the source initial state, adapted to the group's policy (the uniform group never receives
    the head's cut function)."""
    g = ctx.fp["groups"][group]; model = head.build()
    model.load_state_dict(torch.load(ctx.fact_init(seed), map_location="cpu", weights_only=True))
    pp.partitioned(model, head.mods, g["policy"], cut_fn if g["policy"] == "predicted" else None, int(g.get("grid_samples", pp.GRID)))
    return model.to(device)


def identity(ctx: PCtx, group: str, seed: int, head, plan_sha: str, init_sha: str, head_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "group": group, "seed": int(seed), "config": {k: ctx.fp[k] for k in RUN_KEYS},
               "fact_arm": ctx.fa["arms"]["FACT"], "source_training": {k: ctx.fa[k] for k in tr.TRAINING_KEYS}, "official_commit": head.commit,
               "expanded_config": head.cfg.dump(), "token_rule": ctx.rule.__dict__, "plan_sha256": plan_sha, "init_sha256": init_sha, "head_sha256": head_sha,
               "inputs": {"protocol": ctx.data.proto_sha, "manifest": ctx.data.fman_sha, "bank": ctx.data.bank_digest},
               "code": {c: sha256_file(ctx.root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@torch.no_grad()
def monitor(head, model, chosen: list, device) -> dict:
    """Training-set fit on the 16 fixed sequences (eval mode), with the mean number of intervals entering the U block's
    averaging; never used to choose anything."""
    model.eval(); fa, f1, exact, parts = [], [], 0, []
    for _, s in chosen:
        p = head.predict(model, torch.from_numpy(s.feat.astype(np.float32)).to(device))["labels"]; parts.append(len(pp.last_partition(model)))
        ref, pr = labels_to_segments(s.labels), labels_to_segments(p)
        fa.append(float((p == s.labels).mean())); f1.append(segment_f1(pr, ref, 0.5)["f1"]); exact += int([g.label for g in pr] == [g.label for g in ref])
    return {"frame_accuracy": float(np.mean(fa)), "segment_f1_50": float(np.mean(f1)), "transcripts_exact": exact, "U_intervals_mean": float(np.mean(parts))}


@torch.no_grad()
def isolation(head, model, seqs: list[Seq], device, seed: int) -> dict:
    """Predictions, last-block logits and the U partition under five label sequences passed to the official forward."""
    model.eval(); rng = np.random.default_rng(seed); st = Counter(); worst = 0.0
    for s in seqs:
        x, ref = torch.from_numpy(s.feat.astype(np.float32)).to(device), torch.from_numpy(s.labels).to(device); outs = {}
        for k, lab in {"reference": ref, "zeros": torch.zeros_like(ref), "random": torch.from_numpy(rng.integers(0, 5, ref.numel())).to(device),
                       "time_reversed": ref.flip(0), "single_sample": ref[:1]}.items():
            pred = np.asarray(model([x], [lab])[0]["pred"]); last = model.block_list[-1]
            outs[k] = (pred, last.frame_clogit.clone(), last.action_clogit.clone(), pp.last_partition(model))
        base = outs["reference"]
        for k, (pred, fl, al, part) in outs.items():
            if k != "reference":
                st["comparisons"] += 1; st["changed_predictions"] += int(not np.array_equal(pred, base[0])); st["changed_partitions"] += int(part != base[3])
                worst = max(worst, float((fl - base[1]).abs().max()), float((al - base[2]).abs().max()))
        st["adapter_decode_differs"] += int(not np.array_equal(head.predict(model, x)["labels"], base[0])); st["sequences"] += 1
    return {**st, "max_abs_logit_difference": worst}


def zero_gradient_parameters(head, model, batch: list[Seq], device, seed: int) -> list[str]:
    seed_everything(seed); model.train(); model.zero_grad(set_to_none=True); loss, _ = head.loss(model, collate(batch, device)); loss.backward()
    out = [k for k, p in model.named_parameters() if p.grad is None or not float(p.grad.abs().max()) > 0 or not torch.isfinite(p.grad).all()]
    model.zero_grad(set_to_none=True)
    return out


def fit16(ctx: PCtx, head, group: str, cut_fn, seed: int, chosen: list, device) -> tuple:
    """The fixed 16 training sequences: 2,000 alternating updates of Adam at the constant rate, official augmentation on."""
    sm, o = ctx.fp["smoke"]["fit"], ctx.fa["arms"]["FACT"]["optimizer"]; seed_everything(int(ctx.fp["smoke"]["seed"]))
    model = new_fact(ctx, head, group, cut_fn, seed, device); seqs = [s for _, s in chosen]
    opt = torch.optim.Adam(model.parameters(), lr=float(sm["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    batches, clip, log, sums, every = [seqs[:8], seqs[8:16]], float(o["grad_clip"]), [], defaultdict(float), int(sm["log_every"])
    for step in range(int(sm["updates"])):
        model.train(); loss, _ = head.loss(model, collate(batches[step % 2], device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{group} fit update {step}: non-finite loss")
        opt.zero_grad(set_to_none=True); loss.backward(); sums["grad_norm"] += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step()
        sums["train_loss"] += float(loss)
        if (step + 1) % every == 0:
            model.eval()
            with torch.no_grad():
                terms = {f"eval_{k}": float(v) for k, v in fi.term_losses(head, model, seqs, device).items()}
            log.append({"update": step + 1, **{k: v / every for k, v in sums.items()}, **terms}); sums.clear()
            print(f"{group} fit {step + 1}: train {log[-1]['train_loss']:.4f} eval total {terms['eval_total']:.4f}", flush=True)
    out = fi.outcome(head, model, chosen, device); parts = []
    with torch.no_grad():
        for kind, s in chosen:
            x = torch.from_numpy(s.feat.astype(np.float32)).to(device); head.predict(model, x)
            parts.append({"kind": kind, "samples": int(s.labels.size), "U_intervals": len(pp.last_partition(model)),
                          "head_cuts": len(cut_fn(x)) if cut_fn is not None else None})
    return model, log, out, parts


def timing(ctx: PCtx, head, group: str, cut_fn, seed: int, batches: list, warm: int, device, total: int) -> tuple:
    model = new_fact(ctx, head, group, cut_fn, seed, device); opt, sched, clip = tr.make_optimiser(ctx.fa["arms"]["FACT"], model, total); seed_everything(0); times = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    for i, b in enumerate(batches):
        torch.cuda.synchronize(); t0 = time.time(); tr.update(head, model, b, opt, sched, clip, device); torch.cuda.synchronize()
        if i >= warm:
            times.append(time.time() - t0)
    t = np.asarray(times)
    return {"timed_updates": int(t.size), "mean_s": float(t.mean()), "median_s": float(np.median(t)), "p90_s": float(np.percentile(t, 90)), "max_s": float(t.max()),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}, model, opt, sched


def smoke_stamp(ctx: PCtx, group: str, seed: int, head_sha: str) -> str:
    """What a smoke report was produced from; the queue recomputes it before trusting the verdict and timing."""
    payload = {"config": {k: ctx.fp[k] for k in (*RUN_KEYS, "smoke")}, "fact_arm": ctx.fa["arms"]["FACT"], "group": group, "seed": int(seed), "head_sha256": head_sha,
               "plan_sha256": sha256_file(ctx.plan_path(seed)), "init_sha256": sha256_file(ctx.fact_init(seed)), "code": {c: sha256_file(ctx.root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def smoke(ctx: PCtx, group: str, seed: int, device) -> dict:
    t_start = time.time(); sm, arm = ctx.fp["smoke"], ctx.fa["arms"]["FACT"]; tol, total = float(sm["tolerance"]), int(ctx.tr["total_updates"])
    plan, _ = checked_plan(ctx, seed); head = tr.make_head(ctx, "FACT", plan, device); bh, cut_fn, head_sha = frozen_head(ctx, seed, device); deterministic()
    head_before = {k: v.clone() for k, v in bh.state_dict().items()}; fn = cut_fn if ctx.fp["groups"][group]["policy"] == "predicted" else None
    n = int(sm["reproducibility_updates"]); b6 = plan_batches(ctx, seed, 0, plan)[:n]; rep = {"group": group, "seed": seed, "head_sha256": head_sha}

    def run(batches, state=None):
        model = new_fact(ctx, head, group, fn, seed, device); opt, sched, clip = tr.make_optimiser(arm, model, total)
        if state:
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); tr.set_rng_state(state["rng"])
        else:
            seed_everything(seed)
        return model, opt, sched, [tr.update(head, model, b, opt, sched, clip, device)["total"] for b in batches]

    def max_diff(x, y):
        return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)
    ra, rb = run(b6), run(b6)
    rep["reproducibility"] = {"loss_max_abs": max(abs(p - q) for p, q in zip(ra[3], rb[3])), "params_max_abs": max_diff(ra[0].state_dict(), rb[0].state_dict())}
    rc = run(b6[: n // 2])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "resume.pt"; tr.save_resume(p, rc[0], rc[1], rc[2], n // 2, "smoke", [])
        model = new_fact(ctx, head, group, fn, seed, device); opt, sched, clip = tr.make_optimiser(arm, model, total); done, _ = tr.load_resume(p, model, opt, sched, "smoke")
    rep["resume_bound"] = tr.optimiser_bound(opt, model)
    for b in b6[n // 2:]:
        tr.update(head, model, b, opt, sched, clip, device)
    rep["resume"] = {"completed_before": done, "params_max_abs": max_diff(ra[0].state_dict(), model.state_dict()),
                     "lr_after": float(opt.param_groups[0]["lr"]), "lr_after_continuous": float(ra[1].param_groups[0]["lr"])}
    o = ra[1].defaults; c = arm["optimizer"]
    rep["optimizer_matches_config"] = (type(ra[1]).__name__ == c["name"] and o["lr"] == float(c["lr"]) and o["weight_decay"] == float(c["weight_decay"])
                                       and o["eps"] == float(c["eps"]) and list(o["betas"]) == [float(b) for b in c["betas"]])
    rep["head_not_in_optimizer"] = not ({id(p) for p in bh.parameters()} & {id(p) for g in ra[1].param_groups for p in g["params"]})
    model = new_fact(ctx, head, group, fn, seed, device); batch = collate(b6[0], device)
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    seed_everything(1); l1, _ = head.loss(model, batch); seed_everything(1); l2, _ = head.loss(model, alt)
    rep["padding"] = {"loss_diff": abs(float(l1) - float(l2)), "padded_positions": int((batch["valid"] == 0).sum())}
    official = head.build().to(device); official.load_state_dict(torch.load(ctx.fact_init(seed), map_location=device, weights_only=True))
    zero, zero_official = zero_gradient_parameters(head, model, b6[0], device, seed), zero_gradient_parameters(head, official, b6[0], device, seed)
    rep["zero_gradient_parameters"] = {"this_group": zero, "official": zero_official, "same_as_official": zero == zero_official}; del official
    chosen = tr.monitor_set(ctx); iso_seqs = [s for _, s in chosen] + list(ctx.data.streams[:200])
    rep["isolation_initial"] = isolation(head, new_fact(ctx, head, group, fn, seed, device), iso_seqs, device, seed=1)
    rep["gradient_paths_initial"] = fi.gradient_paths(head, new_fact(ctx, head, group, fn, seed, device), [s for _, s in chosen[:8]], device, seed)
    fitted, log, out, parts = fit16(ctx, head, group, fn, seed, chosen, device)
    rep["fit"] = {"log": log, "outcome": {k: v for k, v in out.items() if k != "rows"}, "rows": out["rows"], "partitions": parts}
    rep["isolation_fit"] = isolation(head, fitted, iso_seqs, device, seed=2); del fitted
    rep["head_unchanged"] = all(torch.equal(v, head_before[k]) for k, v in bh.state_dict().items()) and all(p.grad is None and not p.requires_grad for p in bh.parameters())
    warm, timed = int(sm["timing"]["warmup_updates"]), int(sm["timing"]["timed_updates"])
    rep["timing"], tmodel, topt, tsched = timing(ctx, head, group, fn, seed, plan_batches(ctx, seed, 0, plan)[:warm + timed], warm, device, total)
    torch.cuda.synchronize(); t0 = time.time(); monitor(head, tmodel, chosen, device); torch.cuda.synchronize(); t1 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tr.save_resume(Path(tmp) / "r.pt", tmodel, topt, tsched, 0, "smoke", [])
    t2 = time.time(); epochs = int(ctx.tr["epochs"])
    rep["timing"].update({"monitor_s": t1 - t0, "save_s": t2 - t1, "estimated_hours_per_run": (rep["timing"]["mean_s"] * total + (t2 - t0) * epochs) / 3600})
    iso_ok = all(rep[k][x] == 0 for k in ("isolation_initial", "isolation_fit") for x in ("changed_predictions", "changed_partitions", "adapter_decode_differs")) \
        and all(rep[k]["max_abs_logit_difference"] == 0 for k in ("isolation_initial", "isolation_fit"))
    fit_finite = all(np.isfinite(v) for r in log for v in r.values())
    rep["checks"] = {"reproducibility": rep["reproducibility"]["loss_max_abs"] <= tol and rep["reproducibility"]["params_max_abs"] <= tol,
                     "resume": rep["resume"]["params_max_abs"] <= tol and rep["resume_bound"] and rep["resume"]["lr_after"] == rep["resume"]["lr_after_continuous"],
                     "optimizer": rep["optimizer_matches_config"] and rep["head_not_in_optimizer"], "padding": rep["padding"]["loss_diff"] <= tol and rep["padding"]["padded_positions"] > 0,
                     "frozen_head": rep["head_unchanged"], "label_isolation": iso_ok, "fit_finite": fit_finite,
                     "zero_gradients_as_official": rep["zero_gradient_parameters"]["same_as_official"]}
    rep["ok"] = all(rep["checks"].values()); rep["gpu_seconds"] = time.time() - t_start
    return rep


# ----------------------------------------------------------------------------------------------- prepare
def run_prepare(ctx: PCtx) -> None:
    """Every batch of the three source plans rebuilt and compared; the heads' pos_weight and initial states."""
    t0 = time.time(); seeds = {}
    for seed in ctx.seeds:
        plan, plan_sha = checked_plan(ctx, seed)
        for e in range(int(ctx.tr["epochs"])):
            plan_batches(ctx, seed, e, plan)
        pw, pos, neg = epoch0_pos_weight(ctx, seed, plan); seed_everything(seed); init = {k: v.detach().cpu().clone() for k, v in new_head(ctx, "cpu").state_dict().items()}
        seeds[seed] = {"plan_sha256": plan_sha, "batches_equal_plan": int(ctx.tr["epochs"]) * int(ctx.tr["updates_per_epoch"]), "epoch0_positives": pos,
                       "epoch0_negatives": neg, "pos_weight": pw, "head_init_sha256": tr.save_or_check_init(ctx.head_file(seed, "init.pt"), init),
                       "fact_init_sha256": sha256_file(ctx.fact_init(seed))}
    chosen = tr.monitor_set(ctx)
    res = {"rows": ctx.data.row_check, "inputs_sha256": {"protocol": ctx.data.proto_sha, "e1_feature_manifest": ctx.data.fman_sha}, "bank_digest": ctx.data.bank_digest,
           "token_rule": ctx.rule.__dict__, "seeds": seeds, "head_parameters": int(sum(p.numel() for p in new_head(ctx, "cpu").parameters())),
           "monitor_set": {"kinds": dict(Counter(k for k, _ in chosen)), "samples": [int(s.labels.size) for _, s in chosen]},
           "training_clips": len(ctx.data.train_clips), "training_sessions": len({session_of(v) for v in ctx.data.train_clips}), "seconds": time.time() - t0}
    write_json(ctx.out / "preparation.json", res, ctx, copy_to_results=True)
    print(json.dumps({k: res[k] for k in ("rows", "seeds", "head_parameters")}, indent=1))


# ----------------------------------------------------------------------------------------------- one formal run
def train_one(ctx: PCtx, group: str, seed: int, device) -> None:
    plan, plan_sha = checked_plan(ctx, seed); head = tr.make_head(ctx, "FACT", plan, device); bh, cut_fn, head_sha = frozen_head(ctx, seed, device)
    init_path = ctx.fact_init(seed); init_sha = sha256_file(init_path); ident = identity(ctx, group, seed, head, plan_sha, init_sha, head_sha); key = f"{group}_s{seed}"
    run_dir, ckpt = ctx.out / "train" / key, ctx.ckpt / key; arm = ctx.fa["arms"]["FACT"]; fn = cut_fn if ctx.fp["groups"][group]["policy"] == "predicted" else None
    state = tr.finished_run(run_dir / "run_manifest.json", ident)
    if state is True:
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    if state is False:
        raise SystemExit(f"{key}: a finished run of another identity exists in {run_dir}; move it before training again")
    deterministic(); run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    total, per_epoch = int(ctx.tr["total_updates"]), int(ctx.tr["updates_per_epoch"]); chosen = tr.monitor_set(ctx)
    model = new_fact(ctx, head, group, fn, seed, device); opt, sched, clip = tr.make_optimiser(arm, model, total); completed, records = 0, []
    head_before = {k: v.clone() for k, v in bh.state_dict().items()}
    seed_everything(seed)   # dropout and masking draw from this state; resume restores the saved one
    if (ckpt / "resume.pt").is_file():
        completed, records = tr.load_resume(ckpt / "resume.pt", model, opt, sched, ident); print(f"{key}: resumed after {completed} updates", flush=True)
    if completed % per_epoch:
        raise SystemExit(f"{key}: resume after {completed} updates is not at an epoch boundary")
    log = run_dir / "train_log.jsonl"
    if completed == 0:
        log.write_text("")
    print(json.dumps({"run": key, "identity": ident[:16], "completed": completed, "plan": plan_sha[:12], "init": init_sha[:12], "head": head_sha[:12]}), flush=True)
    for epoch in range(completed // per_epoch, int(ctx.tr["epochs"])):
        torch.cuda.reset_peak_memory_stats(device); t0 = time.time(); sums, gmax, lrs = defaultdict(float), 0.0, []
        for b in plan_batches(ctx, seed, epoch, plan):
            r = tr.update(head, model, b, opt, sched, clip, device); completed += 1; lrs.append(r.pop("lr")); gmax = max(gmax, r["grad_norm"])
            for k, v in r.items():
                sums[k] += v
        t1 = time.time(); mon = monitor(head, model, chosen, device)
        rec = {"epoch": epoch, "updates": completed, "lr_first": lrs[0], "lr_last": lrs[-1], "loss": {k: v / per_epoch for k, v in sums.items() if k != "grad_norm"},
               "grad_norm_mean": sums["grad_norm"] / per_epoch, "grad_norm_max": gmax, "seconds_train": t1 - t0, "seconds_monitor": time.time() - t1,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "monitor": mon}
        records.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{key} epoch {epoch} updates {completed} lr {lrs[-1]:.2e} loss {rec['loss']['fact']:.4f} grad {rec['grad_norm_mean']:.3f} | monitor F1 "
              f"{mon['segment_f1_50']:.3f} exact {mon['transcripts_exact']}/16 U {mon['U_intervals_mean']:.1f} ({rec['seconds_train']:.0f}s)", flush=True)
        tr.save_resume(ckpt / "resume.pt", model, opt, sched, completed, ident, records)
    if not all(torch.equal(v, head_before[k]) for k, v in bh.state_dict().items()):
        raise SystemExit(f"{key}: the frozen head changed during training")
    chosen_ckpt = tr.final_checkpoint(records, total); torch.save(model.state_dict(), ckpt / "final.pt")
    man = {"identity": ident, "group": group, "seed": seed, "policy": ctx.fp["groups"][group]["policy"], "completed_updates": completed, "checkpoint": chosen_ckpt,
           "final_checkpoint": str(ckpt / "final.pt"), "final_sha256": sha256_file(ckpt / "final.pt"), "plan": str(ctx.plan_path(seed)), "plan_sha256": plan_sha,
           "init": str(init_path), "init_sha256": init_sha, "head_sha256": head_sha, "official_commit": head.commit, "expanded_config": head.cfg.dump(),
           "token_rule": ctx.rule.__dict__, "parameters": int(sum(p.numel() for p in model.parameters())), "head_parameters": int(sum(p.numel() for p in bh.parameters())),
           "minutes": sum(r["seconds_train"] + r["seconds_monitor"] for r in records) / 60, "records": records}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str)); (ckpt / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"run": key, "final_sha256": man["final_sha256"][:16], "minutes": round(man["minutes"], 1)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); step = ap.add_mutually_exclusive_group()
    for flag in ("--prepare", "--train-head", "--freeze-heads", "--smoke"):
        step.add_argument(flag, action="store_true")
    ap.add_argument("--group", choices=GROUPS); ap.add_argument("--seed", type=int)
    a = ap.parse_args(); cfg, fp, scfg, paths = load_round(a.config); ctx = PCtx(cfg, scfg, fp, paths)
    if a.prepare:
        run_prepare(ctx); return
    if a.freeze_heads:
        run_freeze_heads(ctx); return
    if a.seed not in ctx.seeds:
        raise SystemExit("one of the round's seeds is required")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.train_head:
        train_head(ctx, a.seed, device); return
    if a.group is None:
        raise SystemExit("--group is required for --smoke and for a formal run")
    if a.smoke:
        rep = smoke(ctx, a.group, a.seed, device); rep["stamp"] = smoke_stamp(ctx, a.group, a.seed, rep["head_sha256"])
        write_json(ctx.out / "smoke" / f"{a.group}_s{a.seed}.json", rep, ctx); write_json(ctx.results / f"smoke_{a.group}.json", rep, ctx)
        print(json.dumps({k: rep[k] for k in ("checks", "timing", "ok")} | {"fit": rep["fit"]["outcome"]}, indent=1, default=str))
        if not rep["ok"]:
            raise SystemExit("smoke checks failed")
        return
    train_one(ctx, a.group, a.seed, device)


if __name__ == "__main__":
    main()
