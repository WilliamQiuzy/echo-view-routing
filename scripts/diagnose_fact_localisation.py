#!/usr/bin/env python
"""FACT short-segment localisation (plan docs/superpowers/plans/2026-10-09-fact-localisation.md;
configs/experiments/fact_localisation_v1.yaml). A post hoc, inference-only diagnostic of the frozen FACT, ASRF and E1 of
the paired FACT-ASRF round on that round's own training sequences: no optimiser, no backward, no development half, no
test split, no EchoNet-Pediatric. Conditions B and C of the intervention read the reference partition; they are
diagnostics, never model results.

  --prepare  the cache index filtered by the protocol's training allowlist before any label is parsed; the training bank
             rebuilt from the kept rows (digest checked); each seed's epoch-49 batches against its saved plan; sequence,
             recording and session counts. Writes preparation.json (counts only).
  --check    the 16 fixed training sequences: the last monitor values of all six runs reproduced; for FACT the ordinary
             pass against the official forward, then A, B, C, A at the last U block with every equality check (native
             windows included); weights and inputs unchanged; timing and the projected cost of --run against the cap.
  --run      every sequence of each seed's epoch 49 (1,500 bank streams, 1,500 native windows): stage readouts, tokens,
             the partition entering U, the final assignment and the intervention, saved per seed on the server; then
             summary.json and report_tables.md (counts only; scripts/fact_localisation_tables.py). Finished seeds of
             the same identity are reused.

  python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # FACT decodes under deterministic algorithms, as it trained

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train_fact_asrf as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.fact_localisation_tables import (BLOCKS, N_CLASSES, WHAT, across_seeds, seed_summary, tables_md, unit_groups,  # noqa: E402
                                              units_of)
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import write_atomic  # noqa: E402
from scripts.train_l1_asrf import batch_digest  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.features.cache import variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore, Seq, TrainData, stream_seq  # noqa: E402
from echo_routing.temporal.dev_run import digest, load_json, sha256_file  # noqa: E402
from echo_routing.temporal.fact_head import token_rule  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments  # noqa: E402

NS, SRC = "fact_localisation_v1", "fact_asrf_paired_v1"
ARMS = ("FACT", "ASRF")
BLOCK_TYPES = ("InputBlock", "UpdateBlock", "UpdateBlockTDU")   # gtea.yaml: the blocks i, u, U
CHECKS = ("official_partition_is_u_runs", "ordinary_equals_official", "A_equals_ordinary", "B_inputs_equal_A", "C_inputs_equal_A",
          "B_equals_A_without_cuts", "C_equals_A_without_cuts", "B_partition_used", "C_partition_used", "A_repeats", "input_unchanged")
ATTRS = ("frame_clogit", "action_clogit", "seg_clogit", "f2a_attn", "a2f_attn", "f2a_attn_logit", "a2f_attn_logit")
CODE = ("scripts/train_fact_asrf.py", "scripts/fact_integration.py", "scripts/train_l1_asrf.py", "echo_routing/temporal/fact_head.py",
        "echo_routing/temporal/dev_heads.py", "echo_routing/temporal/dev_data.py", "echo_routing/compose/stream_assembler.py",
        "echo_routing/compose/banks.py", "echo_routing/compose/recipe_builder.py", "echo_routing/features/cache.py",
        "echo_routing/temporal/segments.py", "echo_routing/temporal/dev_run.py")


# ----------------------------------------------------------------------------------------------- training rows and context
def allowlist_rows(index_csv: Path, clips: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """The index rows of the allowlisted clips in the file's own row order, checked before any label is parsed: no clip
    repeated in the allowlist or the index, none missing, none outside the training split or without samples. Then the
    label column is parsed for the kept rows only (their ids must line up) and must lie in 0..4."""
    allow = set(clips)
    if len(allow) != len(clips):
        raise SystemExit("the training allowlist repeats a clip")
    meta = pd.read_csv(index_csv, dtype={"video_id": str}, usecols=lambda c: c != "label_index")
    keep = meta["video_id"].isin(allow).to_numpy(); rows = meta.loc[keep]
    if rows["video_id"].duplicated().any():
        raise SystemExit("a training clip has more than one index row")
    if len(rows) != len(allow):
        raise SystemExit(f"{len(allow) - len(rows)} training clips have no index row")
    if (rows["split"] != "train").any() or (rows["n_samples"] <= 0).any():
        raise SystemExit("an allowlisted clip is not a training row with samples")
    lines = set((np.flatnonzero(keep) + 1).tolist())   # file lines of the kept rows; line 0 is the header
    sub = pd.read_csv(index_csv, dtype={"video_id": str}, usecols=["video_id", "label_index"], skiprows=lambda i: i > 0 and i not in lines)
    lab = sub["label_index"]
    if sub["video_id"].tolist() != rows["video_id"].tolist():
        raise SystemExit("the parsed label rows do not line up with the kept rows")
    if not pd.api.types.is_integer_dtype(lab) or lab.min() < 0 or lab.max() >= N_CLASSES:
        raise SystemExit("an allowlisted clip has no integer label in 0..4")
    return rows.assign(label_index=lab.to_numpy()), {"index_rows": int(len(meta)), "training_rows_kept": int(len(rows)),
                                                     "rows_dropped_before_label_parsing": int((~keep).sum())}


class TrainRows:
    """The training split from the protocol's allowlist. The cache index's metadata is read and filtered to
    protocol['train']['clips'] in the index's row order; label_index is parsed afterwards, for the kept rows only. Labels,
    the bank and the clip list come from those rows, and no feature of another split is loaded. The index itself also
    lists the rows of other splits (their metadata is read; their labels are not parsed)."""

    def __init__(self, cfg: dict, fa: dict, paths) -> None:
        root, inp = paths.repo_root, fa["inputs"]
        self.proto, self.proto_sha = load_json(root / inp["protocol"]["path"])
        self.fman, self.fman_sha = load_json(root / inp["e1_feature_manifest"]["path"])
        if self.proto_sha != inp["protocol"]["sha256"] or self.fman_sha != inp["e1_feature_manifest"]["sha256"]:
            raise SystemExit("an input sha256 differs from the paired round's config")
        self.key = self.fman["encoder"]["cache_key"]; self.variants = tuple(inp["variants"])
        if self.key != inp["e1_cache_key"] or self.fman["frame_order"] != inp["frame_order"] or int(self.fman["feature_dim"]) != int(inp["feature_dim"]):
            raise SystemExit("the E1 feature manifest differs from the paired round's config")
        for v in self.variants:
            if sha256_file(variant_dir(paths.cache_root, self.key, v) / "index.csv") != self.fman["variants"][v]["index_sha256"]:
                raise SystemExit(f"{v}: index.csv differs from the feature manifest")
        self.train_clips = tuple(self.proto["train"]["clips"])
        rows, self.row_check = allowlist_rows(variant_dir(paths.cache_root, self.key, "orig") / "index.csv", self.train_clips)
        self.labels = dict(zip(rows["video_id"], rows["label_index"].astype(int)))
        bank = multi_bank(rows, "train", cfg); self.bank_digest = digest([recipe_to_dict(r) for r in bank.recipes])
        if self.bank_digest != self.proto["train"]["multi_bank"]["digest"]:
            raise SystemExit(f"training bank digest {self.bank_digest[:16]} != protocol")
        self.recipes = bank.recipes; self.store = FeatureStore(paths.cache_root, self.key)
        self.streams = tuple(stream_seq(render(r, self.store.get)) for r in self.recipes)
        self.train = TrainData(self.streams, tuple((v, self.labels[v]) for v in self.train_clips), self.store)

    def batches(self, tr_cfg: dict, seed: int, epoch: int) -> list[list[Seq]]:
        return self.train.epoch_batches(seed, epoch, int(tr_cfg["native_per_epoch"]), int(tr_cfg["native_window"]),
                                        int(tr_cfg["streams_per_batch"]), int(tr_cfg["natives_per_batch"]))

    def ids(self) -> set[str]:
        return set(self.labels) | self.sessions()

    def sessions(self) -> set[str]:
        return {session_of(v) for v in self.labels}


class Diag:
    """What scripts.train_fact_asrf.monitor_set, load_plan and make_head read (fa, root, data, rule, plan_path), built on
    TrainRows. The paired round's Ctx is not instantiated: its TrainOnly parses every label of the index."""

    def __init__(self, cfg: dict, scfg: dict, fl: dict, paths) -> None:
        self.cfg, self.fl, self.fa, self.paths, self.root = cfg, fl, scfg["fact_asrf_paired"], paths, paths.repo_root
        self.tr = self.fa["data"]; self.server = self.root / self.fa["outputs"]["server_dir"]
        self.out, self.results = self.root / fl["outputs"]["server_dir"], self.root / fl["outputs"]["results_dir"]
        self.tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
        self.short_max = int(fl["units"]["groups"]["short_max_samples"]); self.seeds = [int(s) for s in fl["data"]["seeds"]]
        self.epoch = int(fl["data"]["epoch"])
        if self.epoch != int(self.tr["epochs"]) - 1 or self.seeds != [int(s) for s in self.fa["seeds"]]:
            raise SystemExit("the epoch or seeds differ from the paired round's last epoch and seeds")
        ms = fl["source"]["monitor_set"]
        if self.fa["inputs"]["integration_alignment"]["monitor_set"] != ms["path"] or sha256_file(self.root / ms["path"]) != ms["sha256"]:
            raise SystemExit("the 16 fixed training sequences differ from the pinned file")
        self.data = TrainRows(scfg, self.fa, paths); self.recipe = {r.recipe_id: r for r in self.data.recipes}
        ov = self.fa["arms"]["FACT"]["dataset_overrides"]
        self.rule = token_rule([s.labels for s in self.data.streams], int(self.tr["native_per_epoch"]), float(ov["token_ratio"]))
        if self.rule.ntoken != int(ov["expected_ntoken"]) or abs(self.rule.average_transcript - float(ov["expected_average_transcript"])) > 1e-12:
            raise SystemExit(f"the token rule {self.rule} differs from the paired round")

    def batches(self, seed: int, epoch: int) -> list[list[Seq]]:
        return self.data.batches(self.tr, seed, epoch)

    def plan_path(self, seed: int) -> Path:
        return self.server / "plans" / f"plan_s{seed}.json"


def load_round(config: str) -> tuple:
    """This round's config and the paired round's; the paired config and summary must match their pins, else stop."""
    cfg = load_cfg(SimpleNamespace(config=config, set=[])); fl = cfg["fact_localisation"]; paths = load_paths(); src = fl["source"]
    if sha256_file(paths.repo_root / src["config"]["path"]) != src["config"]["sha256"]:
        raise SystemExit("the paired round's config differs from its pinned sha256; stopping")
    summ = paths.repo_root / src["summary"]["path"]
    if sha256_file(summ) != src["summary"]["sha256"]:
        raise SystemExit("the paired round's summary differs from its pinned sha256; stopping")
    runs_ = json.loads(summ.read_text())["runs"]
    if any(runs_[k]["final_sha256"] != v for k, v in src["frozen"].items()) or any(runs_[f"{a}_s{s}"]["plan_sha256"] != v for a in ARMS for s, v in src["plans"].items()):
        raise SystemExit("the pinned weights or plans differ from the paired round's summary")
    scfg = load_cfg(SimpleNamespace(config=src["config"]["path"], set=[]))
    if (cfg["protocol"]["hz"], cfg["selection"]["boundary_tolerance_s"]) != (scfg["protocol"]["hz"], scfg["selection"]["boundary_tolerance_s"]):
        raise SystemExit("the sampling rate or boundary tolerance differs from the paired round")
    return cfg, fl, scfg, paths


def frozen(ctx: Diag, arm: str, seed: int, device) -> tuple:
    """One frozen final model of the paired round in eval mode; weights, frozen manifest, run manifest and plan checked."""
    key = f"{arm}_s{seed}"; d = ctx.root / "models_frozen" / SRC / key; want = ctx.fl["source"]["frozen"][key]
    files = {f["file"]: f["sha256"] for f in json.loads((d / "MANIFEST.json").read_text())["files"]}
    rm = json.loads((d / "run_manifest.json").read_text()); plan, plan_sha = tr.load_plan(ctx, seed)
    if (sha256_file(d / "final.pt") != want or files.get("final.pt") != want or sha256_file(d / "run_manifest.json") != files.get("run_manifest.json")
            or rm["final_sha256"] != want or rm["plan_sha256"] != plan_sha or plan_sha != ctx.fl["source"]["plans"][seed]):
        raise SystemExit(f"{key}: the frozen weights, manifests or plan differ from the pins")
    head = tr.make_head(ctx, arm, plan, device); model = head.build().to(device)
    model.load_state_dict(torch.load(d / "final.pt", map_location=device, weights_only=True)); model.eval()
    if arm == "FACT":
        check_blocks(model)
    return head, model, rm


def check_blocks(model) -> None:
    if tuple(type(b).__name__ for b in model.block_list) != BLOCK_TYPES:
        raise SystemExit("the FACT blocks are not i, u, U (gtea.yaml)")


def epoch_sequences(ctx: Diag, seed: int) -> list[Seq]:
    """The seed's last training epoch in batch order; every batch digest must equal the saved plan's."""
    plan, sha = tr.load_plan(ctx, seed)
    if sha != ctx.fl["source"]["plans"][seed]:
        raise SystemExit(f"seed {seed}: the plan differs from its pinned sha256")
    bs = ctx.batches(seed, ctx.epoch)
    if len(bs) != int(ctx.tr["updates_per_epoch"]) or [batch_digest(b) for b in bs] != plan["epochs"][ctx.epoch]:
        raise SystemExit(f"seed {seed}: the epoch-{ctx.epoch} batches differ from the saved plan")
    seqs = [s for b in bs for s in b]
    if len({s.seq_id for s in seqs}) != len(seqs):
        raise SystemExit(f"seed {seed}: a sequence repeats within the epoch")
    return seqs


def fragments(ctx: Diag, s: Seq) -> list[list]:
    """[recording, start, end) of each source fragment in the sequence's samples; a native window has one."""
    if s.is_native:
        return [[s.seq_id.rsplit("@", 1)[0], 0, int(s.labels.size)]]
    out, pos = [], 0
    for fr in ctx.recipe[s.seq_id].fragments:
        out.append([fr.video_id, pos, pos + fr.end - fr.start]); pos += fr.end - fr.start
    if pos != s.labels.size:
        raise SystemExit("a stream's fragments do not add up to its length")
    return out


# ----------------------------------------------------------------------------------------------- partitions (half-open [s, e))
def runs(labels) -> list[tuple[int, int]]:
    return [(g.start, g.end) for g in labels_to_segments(np.asarray(labels))]


def check_tiling(intervals: list, n: int) -> None:
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != n or any(s >= e for s, e in intervals) \
            or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("the intervals do not tile the sequence")


def reference_cuts(intervals: list, ref) -> list[list[int]]:
    """B's new cuts: per interval [s, e), the reference class changes c with s < c < e. A change that already lies on a
    boundary of the partition is not added again."""
    changes = boundaries_from_labels(np.asarray(ref)).tolist()
    return [[c for c in changes if s < c < e] for s, e in intervals]


def uniform_cuts(intervals: list, counts: list[int]) -> list[list[int]]:
    """C's new cuts: per interval [s, e) of length L with k cuts (B's count there), s + floor(j L / (k + 1)), j = 1..k."""
    out = []
    for (s, e), k in zip(intervals, counts):
        n = e - s
        if not 0 <= k <= n - 1:
            raise ValueError(f"{k} cuts cannot lie strictly inside an interval of length {n}")
        pts = [s + (j * n) // (k + 1) for j in range(1, k + 1)]
        if len(set(pts)) != k or not all(s < p < e for p in pts):
            raise AssertionError("uniform cuts are not distinct interior points")
        out.append(pts)
    return out


def subdivide(intervals: list, cuts: list[list[int]]) -> list[tuple[int, int]]:
    out = []
    for (s, e), c in zip(intervals, cuts):
        pts = [s, *c, e]; out += list(zip(pts[:-1], pts[1:]))
    return out


def conditions(intervals_a: list, ref) -> dict:
    """A (the given partition), B (A plus the reference changes inside its intervals) and C (as many cuts per interval
    as B, placed uniformly). Only positions and lengths are used; no reference class enters a partition."""
    n = len(ref); check_tiling(intervals_a, n); kb = reference_cuts(intervals_a, ref); kc = uniform_cuts(intervals_a, [len(k) for k in kb])
    b, c = subdivide(intervals_a, kb), subdivide(intervals_a, kc); check_tiling(b, n); check_tiling(c, n)
    return {"A": [tuple(i) for i in intervals_a], "B": b, "C": c, "B_cuts": [x for k in kb for x in k], "C_cuts": [x for k in kc for x in k]}


def closed_segments(intervals: list, action, segment) -> list:
    """Half-open [s, e) as the official closed Segment(action, s, e - 1), whose len is e - s. The action is the previous
    block's argmax at s, never a reference label. Built directly, so adjacent intervals of one class stay separate."""
    check_tiling(intervals, len(action))
    return [segment(int(action[s]), int(s), int(e) - 1) for s, e in intervals]


def fixed_downsample(block, frame_feature, intervals: list, mods: dict):
    """UpdateBlockTDU.temporal_downsample of the pinned checkout with its segment list built from `intervals` instead of
    utils.parse_label of the argmax; every other line is the official one."""
    utils, basic = mods["blocks"].utils, mods["basic"]
    cprob = frame_feature[:, :, -block.nclass:]
    _, pred = cprob[:, 0].max(dim=-1)
    segs = closed_segments(intervals, utils.to_numpy(pred), utils.Segment)
    tdu = basic.TemporalDownsampleUpsample(segs)
    tdu.to(cprob.device)
    seg_feature = tdu.feature_frame2seg(frame_feature)
    seg_feature, _ = block.seg_update(seg_feature)
    seg_feature = torch.relu(seg_feature)
    seg_feature = block.seg_combine(seg_feature)
    seg_feature, seg_clogit = block.process_feature(seg_feature, block.nclass)
    return tdu, seg_feature, seg_clogit


@contextlib.contextmanager
def partition(block, intervals, mods: dict):
    """The block's temporal_downsample replaced on this instance only (None: the official method). The class and the
    official checkout are untouched; the override is removed on exit, also after an error."""
    if intervals is None:
        yield
        return
    if "temporal_downsample" in vars(block):
        raise RuntimeError("a partition override is already installed")
    block.temporal_downsample = lambda ff: fixed_downsample(block, ff, intervals, mods)
    try:
        yield
    finally:
        del block.temporal_downsample


def attributes(model) -> list[dict]:
    """What each block keeps from its last forward (the decode and the loss read these), cloned, and U's partition."""
    out = []
    for b in model.block_list:
        d = {k: getattr(b, k).detach().clone() for k in ATTRS if hasattr(b, k)}
        if hasattr(b, "tdu"):
            d["segs"] = [(int(g.start), int(g.end) + 1) for g in b.tdu.segs]
        out.append(d)
    return out


@torch.no_grad()
def fact_pass(model, x: torch.Tensor, mods: dict, intervals=None) -> dict:
    """One eval-mode pass: FACT._forward_one_video, then the last block's Block.eval (the official decode with trans
    false). FACT.forward runs the same two calls after parsing a label it does not use. intervals replace U's partition."""
    last = model.block_list[-1]
    with partition(last, intervals, mods):
        outs = model._forward_one_video(x.unsqueeze(1))
    return {"outs": outs, "pred": last.eval().clone(), "attrs": attributes(model)}


def block_readout(p: dict, k: int) -> list[int]:
    """Argmax of the class probabilities block k passes on (the softmax of its frame branch, the last channels)."""
    return p["outs"][k][0][:, 0, -N_CLASSES:].argmax(-1).cpu().tolist()


def logit_mismatch(p: dict) -> int:
    return sum(int((p["outs"][k][0][:, 0, -N_CLASSES:].argmax(-1) != a["frame_clogit"][:, 0].argmax(-1)).sum()) for k, a in enumerate(p["attrs"]))


def assignment(last: dict) -> list[int] | None:
    """The official _eval's frame-to-token choice among the non-null tokens; None when every token is null and the
    decode falls back to the frame branch."""
    cls = last["action_clogit"][:, 0].argmax(-1); active = torch.where(cls != N_CLASSES)[0]
    return active[last["a2f_attn"][0][:, active].argmax(-1)].cpu().tolist() if active.numel() else None


def same_attrs(x: list[dict], y: list[dict]) -> bool:
    return len(x) == len(y) and all(set(p) == set(q) and all(p[k] == q[k] if k == "segs" else torch.equal(p[k], q[k]) for k in p) for p, q in zip(x, y))


def same_outs(x: dict, y: dict, blocks: int | None = None) -> bool:
    n = len(x["outs"]) if blocks is None else blocks
    return all(torch.equal(a, b) for i in range(n) for a, b in zip(x["outs"][i], y["outs"][i]))


def same_pass(x: dict, y: dict) -> bool:
    return same_outs(x, y) and torch.equal(x["pred"], y["pred"]) and same_attrs(x["attrs"], y["attrs"])


def weights(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same_weights(model, w0: dict) -> bool:
    sd = model.state_dict()
    return set(sd) == set(w0) and all(torch.equal(sd[k], w0[k]) for k in sd)


@torch.no_grad()
def fact_sequence(head, model, s: Seq, device, run_bc: bool) -> tuple[dict, dict]:
    """One sequence: the official forward (head.predict), the ordinary pass, then A, B, C, A at the last U block, with
    every equality check. Constructed streams always execute B and C. A native window has one reference label, so its B
    and C partitions equal A's; they are executed only with run_bc (the fixed 16 and the tests), else A's outputs are
    reused once the partitions are checked identical."""
    mods = head.mods; x = torch.from_numpy(s.feat.astype(np.float32)).to(device); x0 = x.clone()
    off = head.predict(model, x); s_off = attributes(model)
    p0 = fact_pass(model, x, mods); u = block_readout(p0, 1); cond = conditions(runs(u), s.labels)
    a1 = fact_pass(model, x, mods, cond["A"]); execute = run_bc or not s.is_native
    if not execute and not cond["B"] == cond["A"] == cond["C"]:
        raise SystemExit("a native window has a reference change")
    b = fact_pass(model, x, mods, cond["B"]) if execute else a1
    c = fact_pass(model, x, mods, cond["C"]) if execute else a1
    a2 = fact_pass(model, x, mods, cond["A"]); before_u = len(model.block_list) - 1
    chk = {"official_partition_is_u_runs": s_off[-1]["segs"] == cond["A"],
           "ordinary_equals_official": p0["pred"].cpu().tolist() == off["labels"].tolist() and same_attrs(p0["attrs"], s_off),
           "A_equals_ordinary": same_pass(a1, p0), "B_inputs_equal_A": same_outs(b, a1, before_u), "C_inputs_equal_A": same_outs(c, a1, before_u),
           "B_equals_A_without_cuts": bool(cond["B_cuts"]) or same_pass(b, a1), "C_equals_A_without_cuts": bool(cond["C_cuts"]) or same_pass(c, a1),
           "B_partition_used": b["attrs"][-1]["segs"] == cond["B"], "C_partition_used": c["attrs"][-1]["segs"] == cond["C"],
           "A_repeats": same_pass(a2, a1), "input_unchanged": torch.equal(x, x0)}
    rec = {"stages": {"FACT_i": block_readout(p0, 0), "FACT_u": u, "FACT_U": block_readout(p0, 2), "FACT_final": off["labels"].tolist()},
           "tokens": {name: p0["attrs"][k]["action_clogit"][:, 0].argmax(-1).cpu().tolist() for k, name in enumerate(BLOCKS)},
           "partition_A": [list(i) for i in cond["A"]], "assign": assignment(p0["attrs"][-1]), "logit_argmax_mismatches": logit_mismatch(p0),
           **{k: {"cuts": cond[f"{k}_cuts"], "final": q["pred"].cpu().tolist(), "U": block_readout(q, 2), "executed": q is not a1} for k, q in (("B", b), ("C", c))}}
    return rec, chk


@torch.no_grad()
def asrf_readouts(head, model, seqs: list[Seq], device) -> list[dict]:
    """ASRF's last-stage argmax and its official refinement output (batch 1, as in the round)."""
    dec = head.decode(model, seqs, device)
    return [{"ASRF_frame": np.asarray(dec[s.seq_id]["frame_labels"]).tolist(), "ASRF_final": np.asarray(dec[s.seq_id]["labels"]).tolist()} for s in seqs]


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- one seed (server records)
def seed_paths(ctx: Diag, seed: int) -> tuple[Path, Path]:
    return ctx.out / f"seed{seed}_sequences.jsonl.gz", ctx.out / f"seed{seed}_done.json"


def save_seed(ctx: Diag, seed: int, recs: list[dict], meta: dict) -> None:
    data, done = seed_paths(ctx, seed); ctx.out.mkdir(parents=True, exist_ok=True); tmp = data.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(data); write_atomic(done, json.dumps({**meta, "records_sha256": sha256_file(data)}, indent=1))


def load_seed(ctx: Diag, seed: int, ident: str):
    data, done = seed_paths(ctx, seed)
    if not done.is_file():
        return None
    meta = json.loads(done.read_text())
    if meta["identity"] != ident or sha256_file(data) != meta["records_sha256"]:
        raise SystemExit(f"seed {seed}: the saved records belong to another identity or changed; move them before running again")
    with gzip.open(data, "rt") as fh:
        return [json.loads(line) for line in fh], meta


def run_seed(ctx: Diag, seed: int, device, ident: str) -> None:
    """Every sequence of the seed's epoch 49 through the frozen FACT (with the intervention) and ASRF; E1 from the cache.
    A failed check stops the run before anything is concluded."""
    t0 = time.time(); seqs = epoch_sequences(ctx, seed); gpu = {}
    recs = [{"seq_id": s.seq_id, "native": bool(s.is_native), "ref": s.labels.tolist(), "frags": fragments(ctx, s),
             "stages": {"E1": np.asarray(s.prob).argmax(1).tolist()}} for s in seqs]
    head, model, _ = frozen(ctx, "FACT", seed, device); tr.determinism("FACT"); w0 = weights(model); sync(device); t = time.time()
    for i, (s, r) in enumerate(zip(seqs, recs)):
        rec, chk = fact_sequence(head, model, s, device, run_bc=False); bad = [k for k in CHECKS if not chk[k]]
        if bad:
            raise SystemExit(f"seed {seed}, sequence {i}: checks failed {bad}; stopping")
        r["stages"].update(rec.pop("stages")); r.update(rec)
        if (i + 1) % 500 == 0:
            print(f"seed {seed}: FACT {i + 1}/{len(seqs)} ({time.time() - t:.0f} s)", flush=True)
    sync(device); gpu["FACT"] = time.time() - t; unchanged = same_weights(model, w0); del model; torch.cuda.empty_cache()
    head, model, _ = frozen(ctx, "ASRF", seed, device); tr.determinism("ASRF"); w0 = weights(model); sync(device); t = time.time()
    for r, a in zip(recs, asrf_readouts(head, model, seqs, device)):
        r["stages"].update(a)
    sync(device); gpu["ASRF"] = time.time() - t; unchanged = unchanged and same_weights(model, w0); del model; torch.cuda.empty_cache()
    if not unchanged:
        raise SystemExit(f"seed {seed}: frozen weights changed during inference")
    meta = {"identity": ident, "seed": seed, "sequences": len(recs), "streams": sum(not r["native"] for r in recs), "natives": sum(r["native"] for r in recs),
            "checks_passed": {k: len(recs) for k in CHECKS}, "weights_unchanged": True, "gpu_seconds": gpu, "wall_seconds": time.time() - t0,
            "B_C_executed": {"streams": sum(r["B"]["executed"] and r["C"]["executed"] for r in recs if not r["native"]),
                             "natives": sum(r["B"]["executed"] for r in recs if r["native"])}}
    save_seed(ctx, seed, recs, meta); print(json.dumps({k: meta[k] for k in ("seed", "sequences", "gpu_seconds", "wall_seconds")}), flush=True)


INFERENCE = (allowlist_rows, TrainRows, Diag, load_round, frozen, check_blocks, epoch_sequences, fragments, runs, check_tiling, reference_cuts,
             uniform_cuts, subdivide, conditions, closed_segments, fixed_downsample, partition, attributes, fact_pass, block_readout, logit_mismatch,
             assignment, same_attrs, same_outs, same_pass, weights, same_weights, fact_sequence, asrf_readouts, seed_paths, save_seed, run_seed)


def identity(ctx: Diag) -> str:
    """The run identity: this round's config, the inputs, the imported code, the source of every function that shapes
    the saved records and the constants they use (the summary code is left out, so a report fix does not orphan finished
    inference)."""
    src = hashlib.sha256("".join(inspect.getsource(f) for f in INFERENCE).encode()).hexdigest()
    payload = {"config": ctx.fl, "hz_tol": [ctx.cfg["protocol"]["hz"], ctx.cfg["selection"]["boundary_tolerance_s"]],
               "inputs": {"protocol": ctx.data.proto_sha, "manifest": ctx.data.fman_sha, "bank": ctx.data.bank_digest},
               "code": {c: sha256_file(ctx.root / c) for c in CODE}, "inference_source": src,
               "inference_constants": {"N_CLASSES": N_CLASSES, "BLOCKS": BLOCKS, "BLOCK_TYPES": BLOCK_TYPES, "CHECKS": CHECKS, "ATTRS": ATTRS}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- outputs
def write_json(path: Path, obj: dict, data: TrainRows) -> None:
    """Counts only: both id guards before an atomic write."""
    text = json.dumps(obj, indent=1, default=str); assert_no_ids(json.loads(text), data.ids(), path.name)
    if any(x in text for x in data.sessions()):
        raise SystemExit(f"{path.name}: a session id appears in the output")
    path.parent.mkdir(parents=True, exist_ok=True); write_atomic(path, text)


# ----------------------------------------------------------------------------------------------- steps
def run_prepare(ctx: Diag, ident: str, built: float) -> None:
    """Allowlist filtering, the bank digest, each seed's epoch-49 batches against its plan, and counts with sources."""
    d = ctx.data; bank = sorted(s.seq_id for s in d.streams); bank_src = {fr.video_id for r in d.recipes for fr in r.fragments}; seeds, nat_src = {}, Counter()
    for seed in ctx.seeds:
        t = time.time(); seqs = epoch_sequences(ctx, seed); el = time.time() - t
        st, nat = [s for s in seqs if not s.is_native], [s for s in seqs if s.is_native]; vids = [s.seq_id.rsplit("@", 1)[0] for s in nat]; nat_src.update(set(vids))
        g = Counter(x for s in st for k in range(len(labels_to_segments(s.labels))) for x in unit_groups(labels_to_segments(s.labels), k, ctx.short_max))
        lens = [int(s.labels.size) for s in nat]
        seeds[seed] = {"batches": int(ctx.tr["updates_per_epoch"]), "batch_digests_equal_plan": True, "streams": len(st),
                       "streams_are_the_bank_once_each": sorted(s.seq_id for s in st) == bank, "native_windows": len(nat), "native_recordings": len(set(vids)),
                       "native_sessions": len({session_of(v) for v in vids}), "stream_samples": sum(int(s.labels.size) for s in st), "native_samples": sum(lens),
                       "native_window_length": {"min": min(lens), "median": float(np.median(lens)), "max": max(lens)},
                       "native_windows_short": sum(n <= ctx.short_max for n in lens), "stream_reference_runs": g["all"], "stream_short_runs": g["short"],
                       "stream_xyx_middles": g["xyx"], "stream_a4c_a5c_xyx_middles": g["a4c_a5c_xyx"], "seconds_epoch_build": el}
    every = set(nat_src) | bank_src
    res = {"what": WHAT, "identity": ident, "rows": d.row_check, "inputs_sha256": {"protocol": d.proto_sha, "e1_feature_manifest": d.fman_sha},
           "bank_digest_equals_protocol": True, "training_clips": len(d.train_clips), "training_sessions": len({session_of(v) for v in d.train_clips}),
           "bank_streams": len(d.streams), "bank_recordings": len(bank_src), "bank_sessions": len({session_of(v) for v in bank_src}),
           "token_rule": ctx.rule.__dict__, "seeds": seeds,
           "across_seeds": {"native_recordings": len(nat_src), "native_sessions": len({session_of(v) for v in nat_src}),
                            "native_recordings_in_two_or_more_seeds": sum(c > 1 for c in nat_src.values()), "all_recordings": len(every),
                            "all_sessions": len({session_of(v) for v in every})},
           "build_seconds": built}
    write_json(ctx.results / "preparation.json", res, d); print(json.dumps({k: res[k] for k in ("rows", "seeds", "across_seeds")}, indent=1))


def check_fact(head, model, chosen: list, device, times: dict) -> dict:
    """Each fixed sequence twice: with B and C executed everywhere (natives included) and in the --run mode (timed);
    both must pass every check and give the same record."""
    out = Counter()
    strip = lambda r: json.dumps({**r, **{c: {k: v for k, v in r[c].items() if k != "executed"} for c in ("B", "C")}})  # noqa: E731
    for _, s in chosen:
        full, chk = fact_sequence(head, model, s, device, run_bc=True)
        sync(device); t = time.time(); run, chk2 = fact_sequence(head, model, s, device, run_bc=False); sync(device)
        times["native" if s.is_native else "stream"].append((time.time() - t, int(s.labels.size)))
        out["sequences"] += 1; out["run_mode_record_equal"] += int(strip(full) == strip(run))
        for k in CHECKS:
            out[k] += int(chk[k] and chk2[k])
        out["natives_B_C_executed"] += int(s.is_native and full["B"]["executed"] and full["C"]["executed"])
        out["with_added_cuts"] += int(bool(full["B"]["cuts"])); out["logit_argmax_mismatches"] += full["logit_argmax_mismatches"]
    return dict(out)


def projection(ctx: Diag, prep: dict, times: dict, asrf: dict, spent: float, built: float) -> dict:
    """Projected GPU-held wall time of --run: per seed the FACT time per sequence and per sample of the timed fixed
    sequences (the larger), ASRF's decode time per sequence, and the epoch build measured by --prepare; plus the run's
    own context build and the time this check has already held the GPU."""
    rate = {k: (float(np.mean([t for t, _ in v])), sum(t for t, _ in v) / sum(n for _, n in v)) for k, v in times.items()}
    a = max(asrf.values()); per_seed = {}
    for seed in ctx.seeds:
        p = prep["seeds"][str(seed)]
        per_seed[seed] = (max(rate["stream"][0] * p["streams"], rate["stream"][1] * p["stream_samples"])
                          + max(rate["native"][0] * p["native_windows"], rate["native"][1] * p["native_samples"])
                          + a * (p["streams"] + p["native_windows"]) + p["seconds_epoch_build"])
    total = spent + built + sum(per_seed.values())
    return {"per_seed_seconds": per_seed, "check_seconds": spent, "run_build_seconds": built, "projected_total_hours": total / 3600,
            "fact_seconds_per_sequence": {k: v[0] for k, v in rate.items()}, "asrf_seconds_per_sequence": a}


def run_check(ctx: Diag, ident: str, device, t_cuda: float, built: float) -> None:
    """Task 4 on the 16 fixed training sequences before the full run; writes check.json on the server."""
    path = ctx.results / "preparation.json"; prep = json.loads(path.read_text()) if path.is_file() else {}
    if prep.get("identity") != ident:
        raise SystemExit("run --prepare first (no preparation of the current identity)")
    chosen = tr.monitor_set(ctx); res = {"identity": ident, "monitor": {}, "fact": {}, "asrf_seconds_per_sequence": {}, "weights_unchanged": {}}
    times = {"stream": [], "native": []}
    for seed in ctx.seeds:
        for arm in ARMS:
            key = f"{arm}_s{seed}"; head, model, rm = frozen(ctx, arm, seed, device); tr.determinism(arm); w0 = weights(model); last = rm["records"][-1]
            if (int(last["epoch"]), int(last["updates"])) != (ctx.epoch, int(ctx.tr["total_updates"])):
                raise SystemExit(f"{key}: the last record is not epoch {ctx.epoch} after every update")
            mon = tr.monitor(head, model, chosen, device); res["monitor"][key] = {"reproduced": mon == last["monitor"], "decoded": mon, "logged": last["monitor"]}
            if arm == "FACT":
                res["fact"][key] = check_fact(head, model, chosen, device, times)
            else:
                sync(device); t = time.time(); asrf_readouts(head, model, [s for _, s in chosen], device); sync(device)
                res["asrf_seconds_per_sequence"][key] = (time.time() - t) / len(chosen)
            res["weights_unchanged"][key] = same_weights(model, w0); del model; torch.cuda.empty_cache()
    natives = sum(s.is_native for _, s in chosen)
    res["passed"] = {"monitor_reproduced": all(v["reproduced"] for v in res["monitor"].values()),
                     "fact_checks": all(f[k] == len(chosen) for f in res["fact"].values() for k in (*CHECKS, "run_mode_record_equal")),
                     "natives_B_C_executed": all(f["natives_B_C_executed"] == natives for f in res["fact"].values()),
                     "weights_unchanged": all(res["weights_unchanged"].values())}
    res["gpu_seconds"] = time.time() - t_cuda; res["build_seconds"] = built
    res["projection"] = projection(ctx, prep, times, res["asrf_seconds_per_sequence"], res["gpu_seconds"], built)
    res["passed"]["budget"] = res["projection"]["projected_total_hours"] <= float(ctx.fl["budget"]["gpu_hours_max"]); res["ok"] = all(res["passed"].values())
    write_json(ctx.out / "check.json", res, ctx.data)
    print(json.dumps({"passed": res["passed"], "projection": res["projection"], "monitor": {k: v["reproduced"] for k, v in res["monitor"].items()},
                      "fact": res["fact"]}, indent=1))
    if not res["ok"]:
        raise SystemExit("the checks did not all pass; no full run")


def summarise(ctx: Diag, ident: str, done: dict, check: dict, spent: float) -> dict:
    t0 = time.time(); res = {"what": WHAT, "identity": ident, "seeds": {}}; units_by_seed = {}
    for seed in sorted(done):
        recs, meta = done[seed]; units = [u for r in recs for u in units_of(r, ctx.tol, ctx.short_max)]; units_by_seed[seed] = units
        res["seeds"][seed] = {"run": {k: meta[k] for k in ("sequences", "streams", "natives", "checks_passed", "weights_unchanged", "B_C_executed", "gpu_seconds", "wall_seconds")},
                              **seed_summary(recs, units, ctx.tol, ctx.short_max)}
        print(f"seed {seed} summarised ({time.time() - t0:.0f} s)", flush=True)
    res["across_seeds"] = across_seeds(units_by_seed)
    res["check"] = {"monitor_reproduced": {k: v["reproduced"] for k, v in check["monitor"].items()}, "fact": check["fact"], "passed": check["passed"],
                    "weights_unchanged": check["weights_unchanged"], "projection": check["projection"]}
    res["cost"] = {"cap_hours": float(ctx.fl["budget"]["gpu_hours_max"]), "check_gpu_held_seconds": check["gpu_seconds"],
                   "run_wall_seconds_per_seed": {s: done[s][1]["wall_seconds"] for s in sorted(done)},
                   "inference_seconds_per_seed": {s: done[s][1]["gpu_seconds"] for s in sorted(done)},
                   "gpu_held_hours_before_this_summary": spent / 3600, "summary_seconds": time.time() - t0,
                   "gpu_held_hours_total": (spent + time.time() - t0) / 3600}
    return res


def run_all(ctx: Diag, ident: str, device, t_cuda: float, built: float) -> None:
    """Every seed's epoch 49 (finished seeds of this identity reused), then the summary and the tables."""
    path = ctx.out / "check.json"; check = json.loads(path.read_text()) if path.is_file() else {}
    if not check.get("ok") or check.get("identity") != ident:
        raise SystemExit("--check has not passed for the current identity; no full run")
    cap = float(ctx.fl["budget"]["gpu_hours_max"]) * 3600; spent = float(check["gpu_seconds"]) + built; done, stopped = {}, None
    for seed in ctx.seeds:
        got = load_seed(ctx, seed, ident)
        if got is None:
            need = float(check["projection"]["per_seed_seconds"][str(seed)])
            if spent + need > cap:
                stopped = f"budget: {spent / 3600:.2f} h held and {need / 3600:.2f} h projected for seed {seed} exceed the cap; finished seeds kept"
                break
            run_seed(ctx, seed, device, ident); got = load_seed(ctx, seed, ident)
        done[seed] = got; spent += float(got[1]["wall_seconds"])
    if done:   # a stop for the budget still reports the finished seeds, marked as partial
        res = {**summarise(ctx, ident, done, check, spent), "stopped": stopped}
        write_json(ctx.results / "summary.json", res, ctx.data); tables = tables_md(res)
        if any(x in tables for x in ctx.data.ids()):
            raise SystemExit("an id appears in the report tables")
        write_atomic(ctx.results / "report_tables.md", (f"**Partial: {stopped}**\n\n" if stopped else "") + tables); print(json.dumps(res["cost"], indent=1))
    if stopped:
        raise SystemExit(stopped)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); step = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--prepare", "--check", "--run"):
        step.add_argument(flag, action="store_true")
    a = ap.parse_args(); t0 = time.time(); cfg, fl, scfg, paths = load_round(a.config); ctx = Diag(cfg, scfg, fl, paths)
    ident, built = identity(ctx), time.time() - t0
    print(json.dumps({"identity": ident[:16], "rows": ctx.data.row_check, "build_seconds": round(built, 1)}), flush=True)
    if a.prepare:
        run_prepare(ctx, ident, built); return
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); t_cuda = time.time()
    (run_check if a.check else run_all)(ctx, ident, device, t_cuda, built)


if __name__ == "__main__":
    main()
