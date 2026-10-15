#!/usr/bin/env python
"""A06-R1: the revalidation of the FACT video feature round under the post-hoc single-segment fallback gate
(docs/superpowers/plans/2026-10-15-fact-video-feature-gate-amendment.md; configs/experiments/fact_video_feature_v2.yaml).

The v1 stop record is never rewritten; its check keeps passed = false. This script writes the revision's own launch
record from the immutable v1 evidence (verified by sha256 and field by field), the old training code (exported from git
and compared function by function with the code that will run) and a small new GPU check of the new path.

  --export-old-code  local only (the server has no git history): the old training files at the pinned commit into the
                     revision's old_code_dir; their digest must equal the old records' code digest
  --prepare          (CPU) the original evidence and its server copies, the science against v1, the old code and its
                     comparison with the current code, the revised gate on the old final update-2,000 rows and the
                     inherited checks; then a new preparation of every source under the v2 namespace (the round's own
                     stage_prepare), compared with the old one
  --check            (GPU, at most 0.05 h) timing first (5 + 50 formal updates), 6-update reproducibility and 3 + 3
                     resume, padding and the optimiser on formal mixed batches, the gradient record at the initial
                     state, the FACT and control decoding times and the cumulative projection; writes revalidation.json,
                     the launch record of the v2 queue. No smoke is repeated and no weight of it starts a formal run.

  python scripts/revalidate_fact_video_feature.py --config configs/experiments/fact_video_feature_v2.yaml --prepare
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_fact_asrf as tfa  # noqa: E402
import scripts.train_fact_video_feature as tf  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import echo_routing.temporal.dev_data as dd  # noqa: E402
import echo_routing.temporal.dev_train as dtr  # noqa: E402
import echo_routing.temporal.fact_head as fh  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

SCIENCE = ("seeds", "data", "fact", "monitor", "train_fit", "evaluation", "goal", "readings")
INPUT_SECTIONS = ("source", "queue", "checks")          # the inputs and the definitions of the original checks: equal as well
GATE_FIELDS = ("segments", "tokens_used", "matched_token_class_correct", "transcript_exact", "segment_f1_50", "match_one_to_one")
PREP_EQUAL = ("a01_environment", "a01_frozen", "batches", "caches", "development_counts", "environment", "expanded_config_sha256", "library",
              "official", "seeds", "sequence_file", "spot_check", "token_rule", "training_stream_samples", "v01_outputs")
ISOLATION = {"sequences": 216, "comparisons": 864}       # the 16 check sequences and the first 200 streams, four other label sequences each
INIT = {"tensors": 326, "parameters": 10215424}         # F02's FACT build, as the v1 preparation recorded it


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------------------------- the configuration
def check_scientific_equality(old: dict, new: dict) -> dict:
    """Every scientific section and every input section of the revision equals v1's, field for field."""
    if list(new.get("revision", {}).get("scientific_sections", [])) != list(SCIENCE):
        raise SystemExit("the revision's scientific sections differ from the fixed list")
    keys = [*SCIENCE, *INPUT_SECTIONS]
    diff = [k for k in keys if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]
    if diff:
        raise SystemExit(f"the revision changes {diff}; the science and the inputs must equal v1's field for field")
    return {"equal": keys, "changed": sorted(k for k in set(old) | set(new) if k not in keys and old.get(k) != new.get(k))}


def verify_original_evidence(root: Path, original: dict, server: bool) -> dict:
    """The five pinned v1 files, the old records' own code and configuration fields, and on the server the runs copies of
    the preparation and the check, byte for byte."""
    out = {k: {"path": original[k]["path"], "sha256": sha256_file(tf.pinned(root, original[k]))} for k in ("task_sheet", "config", "preparation", "check", "readme")}
    for k in ("preparation", "check"):
        rec = read_json(Path(root) / original[k]["path"])
        if rec.get("code") != original["code_digest"] or rec.get("config_sha256") != original["config"]["sha256"]:
            raise SystemExit(f"the old {k} record was not made by the pinned code and configuration")
    if server:
        for k, rel in original["server_copies"].items():
            p = Path(root) / rel
            if not p.is_file() or p.read_bytes() != (Path(root) / original[k]["path"]).read_bytes():
                raise SystemExit(f"{rel}: the server copy is missing or differs from the pinned results copy")
            out[k]["server_copy"] = rel
    return out


# ----------------------------------------------------------------------------------------------- the gates
def revised_gate(rows: list[dict]) -> dict:
    """The task sheet's section 3 predicate, verbatim, on the final update-2,000 rows of the old smoke, with every row's
    result and the single- and multi-segment token counts reported in full."""
    common = (len(rows) == 16 and all(
        r['transcript_exact'] and r['segment_f1_50'] == 1.0
        and r['match_one_to_one'] for r in rows))

    def token_ok(r):
        if r['tokens_used'] == 0:
            return (r['segments'] == 1 and r['matched_token_class_correct'] == 0.0
                    and r['transcript_exact'] and r['segment_f1_50'] == 1.0)
        return r['tokens_used'] > 0 and r['matched_token_class_correct'] == 1.0
    gate_fit = common and all(token_ok(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        ok = bool(token_ok(r) and r['transcript_exact'] and r['segment_f1_50'] == 1.0 and r['match_one_to_one'])
        out.append({"index": i, "kind": r.get("kind"), **{k: r[k] for k in GATE_FIELDS}, "unmatched_tokens_null": r.get("unmatched_tokens_null"),
                    "result": "rejected" if not ok else "official fallback" if r['tokens_used'] == 0 else "tokens right"})
    groups = {}
    for name, sel in (("single", [r for r in rows if r["segments"] == 1]), ("multi", [r for r in rows if r["segments"] > 1])):
        groups[name] = {"rows": len(sel), "all_null": sum(r["tokens_used"] == 0 for r in sel), "tokens_used": [r["tokens_used"] for r in sel],
                        "segments": [r["segments"] for r in sel], "matched_token_class_correct": [r["matched_token_class_correct"] for r in sel],
                        "unmatched_tokens_null": [r.get("unmatched_tokens_null") for r in sel]}
    return {"passed": bool(gate_fit), "common": bool(common), "rows": out, "groups": groups,
            "official_fallback": [r["index"] for r in out if r["result"] == "official fallback"], "rejected": [r["index"] for r in out if r["result"] == "rejected"]}


def inherited_checks(check: dict, known_zero: list[str], tol: float = 1e-6) -> dict:
    """Every non-smoke check of the old record field by field, the smoke rows against its outcome and its requirements
    recomputed; nothing is inferred from the record's summary."""
    f = check["fact"]; sm = f["smoke"]; out = {}

    def put(name: str, value, ok) -> None:
        out[name] = {"value": value, "ok": bool(ok)}
    for k in ("loss_max_abs", "params_max_abs"):
        put(f"reproducibility.{k}", f["reproducibility"][k], f["reproducibility"][k] <= tol)
    for k in ("params_max_abs", "losses_max_abs", "first_half_losses_max_abs"):
        put(f"resume.{k}", f["resume"][k], f["resume"][k] <= tol)
    put("resume.completed_before", f["resume"]["completed_before"], f["resume"]["completed_before"] == 3)
    for k in ("bound", "lr_equal"):
        put(f"resume.{k}", f["resume"][k], f["resume"][k] is True)
    put("optimizer_matches_config", f["optimizer_matches_config"], f["optimizer_matches_config"] is True)
    put("optimizer.bound", f["optimizer"]["bound"], f["optimizer"]["bound"] is True)
    pad = f["padding"]
    put("padding.padded_positions", pad["padded_positions"], pad["padded_positions"] > 0)
    put("padding.loss_diff", pad["loss_diff"], pad["loss_diff"] == 0.0)
    put("padding.predictions_equal", pad["predictions_equal"], pad["predictions_equal"] is True)
    put("zero_or_missing_gradients", f["zero_or_missing_gradients"], sorted(f["zero_or_missing_gradients"]) == sorted(known_zero))
    for state in ("isolation_initial", "isolation_overfit"):
        iso = f[state]
        for k in ("changed_predictions", "max_abs_logit_difference", "adapter_decode_differs"):
            put(f"{state}.{k}", iso[k], iso[k] == 0)
        put(f"{state}.coverage", {k: iso[k] for k in ISOLATION}, all(iso[k] == v for k, v in ISOLATION.items()))
    put("init", check["init"], sorted(check["init"]) == ["FACT_s0", "FACT_s1", "FACT_s2"] and all(v == INIT for v in check["init"].values()))
    put("smoke.final_update", sm["updates"], sm["updates"] == 2000 and sm["check_points"][-1]["update"] == 2000)
    rows, oc = sm["rows"], sm["outcome"]
    agg = {"sequences": len(rows), "transcript_exact": sum(bool(r["transcript_exact"]) for r in rows), "match_one_to_one": sum(bool(r["match_one_to_one"]) for r in rows),
           "segment_f1_50_min": min(r["segment_f1_50"] for r in rows), "matched_token_class_correct_min": min(r["matched_token_class_correct"] for r in rows),
           "tokens_used_equals_segments": sum(r["tokens_used"] == r["segments"] for r in rows)}
    put("smoke.rows_match_outcome", agg, all(oc.get(k) == v for k, v in agg.items()))
    req = tf.smoke_passed(oc, [f["isolation_initial"], f["isolation_overfit"]])
    put("smoke.requirements_recomputed", req, req == sm["requirements"])
    failed = [k for k, v in req.items() if not v and k != "all"]
    put("smoke.failed_only_matched_classes", failed, failed == ["matched_classes_right"])
    put("original_gate_failed", {"passed": check["passed"], "ok": f["ok"], "all": req["all"]}, check["passed"] is False and f["ok"] is False and req["all"] is False)
    return {"passed": all(v["ok"] for v in out.values()), "failed_only": failed, "zero_gradients": sorted(f["zero_or_missing_gradients"]), "fields": out}


# ----------------------------------------------------------------------------------------------- the old code
def code_paths() -> list[str]:
    return [Path(p).resolve().relative_to(ROOT).as_posix() for p in tf.CODE_FILES]


def old_code_digest(d: Path) -> str:
    return pv.file_digest([Path(d) / rel for rel in code_paths()])


def verify_old_code(d: Path, digest: str) -> str:
    missing = [rel for rel in code_paths() if not (Path(d) / rel).is_file()]
    if missing:
        raise SystemExit(f"{d}: {len(missing)} old training file(s) missing; run --export-old-code where the git history is and copy the directory here")
    got = old_code_digest(d)
    if got != digest:
        raise SystemExit(f"the exported old code digests to {got[:12]}, not the old records' {digest[:12]}")
    return got


def export_old_code(root: Path, commit: str, dst: Path) -> dict:
    """The old training files at the pinned commit, read from git objects (locally; the server has no history)."""
    files = {}
    for rel in code_paths():
        try:
            blob = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"], capture_output=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise SystemExit(f"git show {commit[:12]}:{rel} failed ({type(e).__name__}); export the old code where the history is") from None
        p = Path(dst) / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(blob); files[rel] = hashlib.sha256(blob).hexdigest()
    rec = {"commit": commit, "files": files, "digest": old_code_digest(dst), "made": tf.now()}
    (Path(dst) / "export.json").write_text(json.dumps(rec, indent=1))
    return rec


def definitions(src: str) -> dict[str, str]:
    """Top-level functions and class methods by name with their AST dumps (no positions), each class's other statements,
    and the rest of the module."""
    out, rest = {}, []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            other = [ast.dump(x) for x in node.bases + node.keywords + node.decorator_list]
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.dump(sub)
                else:
                    other.append(ast.dump(sub))
            out[f"{node.name}.<class>"] = json.dumps(other)
        else:
            rest.append(ast.dump(node))
    out["<module>"] = json.dumps(rest)
    return out


def compare_sources(old: str, new: str, allowed: dict, added_allowed: list) -> dict:
    a, b = definitions(old), definitions(new)
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k]); added = sorted(b.keys() - a.keys()); removed = sorted(a.keys() - b.keys())
    disallowed = sorted({k for k in changed if k not in allowed} | {k for k in added if k not in added_allowed} | set(removed))
    return {"changed": changed, "added": added, "removed": removed, "identical": sorted(k for k in a.keys() & b.keys() if a[k] == b[k]), "disallowed": disallowed}


def compare_code(old_dir: Path, allowed: dict, added: list) -> dict:
    """Every training file but this round's script byte-identical to the old code; the script's functions AST-identical
    except the allowed ones; nothing removed."""
    me = Path(tf.__file__).resolve().relative_to(ROOT).as_posix(); mods, train = {}, None
    for rel in code_paths():
        old, new = (Path(old_dir) / rel).read_bytes(), (ROOT / rel).read_bytes()
        mods[rel] = {"identical": old == new, "old_sha256": hashlib.sha256(old).hexdigest(), "new_sha256": hashlib.sha256(new).hexdigest()}
        if rel == me:
            train = compare_sources(old.decode(), new.decode(), allowed, added)
    disallowed = [rel for rel, m in mods.items() if rel != me and not m["identical"]] + train["disallowed"]
    return {"ok": not disallowed, "modules": mods, "train_script": train, "disallowed": disallowed}


# ----------------------------------------------------------------------------------------------- the preparation
def compare_preparations(old: dict, new: dict) -> dict:
    """The new preparation re-verifies every source; all but the namespace paths, the task sheet, the configuration and
    the code must equal the old preparation's record."""
    diff = [k for k in PREP_EQUAL if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]
    for part in ("inits", "plans"):
        for k in sorted(set(old[part]) | set(new[part])):
            o, n = ({x: v for x, v in d.get(k, {}).items() if x != "path"} for d in (old[part], new[part]))
            if not o or o != n:
                diff.append(f"{part}.{k}")
    if {k: v for k, v in old["checked"].items() if k != "task_sheet"} != {k: v for k, v in new["checked"].items() if k != "task_sheet"}:
        diff.append("checked")
    if any(v != INIT for v in ({x: d[x] for x in INIT} for d in new["inits"].values())):
        diff.append("inits.build")
    if diff:
        raise SystemExit(f"the new preparation differs from the old one in {diff}")
    return {"equal_to_original": [*PREP_EQUAL, "inits (all but the path)", "plans (all but the path)", "checked (all but the task sheet)"],
            "differ_by_design": ["config", "config_sha256", "code", "checked.task_sheet", "inits.*.path", "plans.*.path"]}


def original_records(ctx) -> tuple[dict, dict, dict]:
    org = ctx.revision["original"]
    return (read_json(tf.pinned(ctx.root, org["preparation"])), read_json(tf.pinned(ctx.root, org["check"])),
            yaml.safe_load(tf.pinned(ctx.root, org["config"]).read_text())["fact_video_feature"])


def cpu_evidence(ctx) -> dict:
    """The original evidence, the science, the old code, the revised gate and the inherited checks (CPU, read only)."""
    r = ctx.revision; org = r["original"]; ev = verify_original_evidence(ctx.root, org, server=True)
    _, old_check, old_fv = original_records(ctx); sci = check_scientific_equality(old_fv, ctx.fv)
    old_dir = ctx.root / org["old_code_dir"]; verify_old_code(old_dir, org["code_digest"]); exp = read_json(old_dir / "export.json")
    if exp.get("commit") != org["code_commit"] or exp.get("digest") != org["code_digest"]:
        raise SystemExit("the old code's export record names another commit or digest")
    cmp = compare_code(old_dir, r["allowed_code_changes"], r["allowed_added_functions"])
    gate = revised_gate(old_check["fact"]["smoke"]["rows"])
    inh = inherited_checks(old_check, r["known_zero_gradients"]["names"], float(ctx.fv["checks"]["reproducibility"]["tolerance"]))
    return {"evidence": ev, "science": sci, "old_code": {"commit": org["code_commit"], "digest": org["code_digest"], "export_made": exp.get("made"), "comparison": cmp},
            "gate": gate, "inherited": {"source": {**org["check"], "started": old_check["started"], "finished": old_check["finished"]}, **inh,
                                        "known_zero_gradients": r["known_zero_gradients"]}}


def failures(cpu: dict) -> list[str]:
    return [m for ok, m in ((cpu["old_code"]["comparison"]["ok"], f"the code differs beyond the allowed plumbing: {cpu['old_code']['comparison']['disallowed']}"),
                            (cpu["gate"]["passed"], f"the revised gate rejects rows {cpu['gate']['rejected']}"),
                            (cpu["inherited"]["passed"], "an inherited check fails: " + ", ".join(k for k, v in cpu["inherited"]["fields"].items() if not v["ok"])))
            if not ok]


def write_record(ctx, name: str, rec: dict) -> dict:
    lib, _ = tf.library(ctx); flat = json.loads(json.dumps(rec, default=str))
    assert_no_ids(flat, {q["id"] for q in lib} | {session_of(q["id"]) for q in lib if q["kind"] == "native"}, name)
    for p in (ctx.out / name, ctx.results / name):
        pv.write_atomic_json(p, flat)
    return flat


def stage_prepare(ctx) -> None:
    cpu = cpu_evidence(ctx); bad = failures(cpu); old_prep, _, _ = original_records(ctx)
    rec = {"revision": ctx.revision["name"], "made": tf.now(), "code": tf.code_digest(), "config_sha256": ctx.config_sha(),
           "revalidation_code_sha256": sha256_file(Path(__file__)), **cpu}
    if bad:
        write_record(ctx, "revalidation_prepare.json", {**rec, "passed": False, "failed": bad}); raise SystemExit("; ".join(bad))
    tf.stage_prepare(ctx)                                                       # every source again, under this namespace
    rec["preparation"] = {"rechecked_now": "every field of preparation.json: the pins and the task sheet, A01's and A05's records, the full cache hashes, the "
                                           "official checkout, the token rule, every batch of every epoch, the spot checks and the bit-identical strict "
                                           "initial states (scripts/train_fact_video_feature.py::stage_prepare under this configuration)",
                          "inherited": "nothing of the preparation; the check's inherited items are listed under inherited", **compare_preparations(old_prep, tf.read_record(ctx, "preparation.json"))}
    write_record(ctx, "revalidation_prepare.json", {**rec, "passed": True})
    print(json.dumps({"gate": cpu["gate"]["passed"], "fallback_rows": cpu["gate"]["official_fallback"], "inherited": cpu["inherited"]["passed"],
                      "code_changed": cpu["old_code"]["comparison"]["train_script"]["changed"]}), flush=True)


# ----------------------------------------------------------------------------------------------- the new GPU checks
def post_hours(prep: dict, decode: float, control: float, seeds: int) -> tuple[float, dict]:
    """The post-training estimate of the original check (train-fit, each half once, the dry run and one fact-check pass)."""
    cal, sel = (int(prep["development_counts"][h]["samples"]) for h in ("calibration", "selection"))
    fact_samples = sum(prep["training_stream_samples"].values()) + 2 * (cal + sel) * seeds
    control_samples = (3 * cal + sel) * seeds
    return (decode * fact_samples + control * control_samples) / 3600 + tf.STAGE_OVERHEAD_HOURS, \
        {"fact_samples": fact_samples, "control_samples": control_samples, "stage_overhead_hours": tf.STAGE_OVERHEAD_HOURS}


def new_checks(ctx, head, data, init_path: Path, device) -> dict:
    """The new path's checks with the round's own helpers: timing first, reproducibility and resume, the optimiser, padding
    on the first formal mixed batch, the gradient record at the initial state and the decoding times."""
    r, f, a01 = ctx.fv["revalidation"]["reproducibility"], ctx.f, ctx.a01(); n, seed = int(r["updates"]), ctx.seeds[0]
    tm = ctx.fv["checks"]["timing"]; warm, timed = int(tm["warmup_updates"]), int(tm["timed_updates"]); b0 = data.epoch(a01, 0); rep = {}
    t = tf.timing(head, init_path, f, b0[:warm + timed], warm, ctx.total, device); t1 = time.time()
    tf.monitor(head, t["model"], data.check, device); torch.cuda.synchronize(); t2 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tfa.save_resume(Path(tmp) / "r.pt", t["model"], t["opt"], t["sched"], 0, "timing", [])
    t3 = time.time(); a, b = tf.fit_time(t["times"], t["sizes"])
    rep["timing"] = {"timed_updates": len(t["times"]), "mean_s": float(np.mean(t["times"])), "median_s": float(np.median(t["times"])), "fit_a_s": a,
                     "fit_b_s_per_sample": b, "mean_batch_samples": float(np.mean(t["sizes"])), "monitor_s": t2 - t1, "save_s": t3 - t2, "per_epoch_s": t3 - t1}
    del t
    torch.cuda.empty_cache()
    ra_, rb_ = tf.run_updates(head, init_path, b0[:n], f, ctx.total, device, seed), tf.run_updates(head, init_path, b0[:n], f, ctx.total, device, seed)
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(ra_[3], rb_[3])), "params_max_abs": tf.max_diff(ra_[0].state_dict(), rb_[0].state_dict())}
    with tempfile.TemporaryDirectory() as tmp:
        rep["resume"] = tf.resume_check(head, init_path, b0[:n], f, ctx.total, device, seed, Path(tmp) / "resume.pt", reference=ra_)
    o, oc = ra_[1].defaults, f["optimizer"]
    rep["optimizer"] = {"class": type(ra_[1]).__name__, "lr": o["lr"], "betas": list(o["betas"]), "eps": o["eps"], "weight_decay": o["weight_decay"],
                        "grad_clip": float(oc["grad_clip"]), "bound": tfa.optimiser_bound(ra_[1], ra_[0])}
    rep["optimizer_matches_config"] = (rep["optimizer"]["class"] == oc["name"] and rep["optimizer"]["lr"] == float(oc["lr"]) and rep["optimizer"]["betas"] == [float(x) for x in oc["betas"]]
                                       and rep["optimizer"]["eps"] == float(oc["eps"]) and rep["optimizer"]["weight_decay"] == float(oc["weight_decay"]) and rep["optimizer"]["bound"])
    del ra_, rb_
    model = tf.new_model(head, init_path, device); batch = dd.collate(b0[0], device)
    rep["padding"] = {**tf.padding_check(head, model, batch, b0[0], device, 1), "streams": sum(not s.is_native for s in b0[0]), "natives": sum(bool(s.is_native) for s in b0[0])}
    dtr.seed_everything(seed); model.train(); model.zero_grad(set_to_none=True); loss, _ = head.loss(model, batch); loss.backward()
    rep["zero_or_missing_gradients"] = sorted(k for k, p in model.named_parameters() if p.grad is None or not float(p.grad.abs().max()) > 0 or not torch.isfinite(p.grad).all())
    rep["parameters"] = int(sum(p.numel() for p in model.parameters())); del model
    init = tf.new_model(head, init_path, device); init.eval(); streams = list(data.streams[:200])
    torch.cuda.synchronize(); t0 = time.time(); [head.predict(init, torch.from_numpy(s.feat).to(device)) for s in streams]; torch.cuda.synchronize()
    rep["decode_seconds_per_sample"] = (time.time() - t0) / sum(s.labels.size for s in streams); del init
    rep["control_seconds_per_sample"] = tf.control_seconds_per_sample(ctx, seed, streams, device)
    return rep


def new_failures(ctx, rep: dict, old_check: dict) -> list[str]:
    tol, d = float(ctx.fv["revalidation"]["reproducibility"]["tolerance"]), ctx.fv["data"]
    tests = (("reproducibility", rep["reproducibility"]["loss_max_abs"] <= tol and rep["reproducibility"]["params_max_abs"] <= tol),
             ("resume", all(rep["resume"][k] <= tol for k in ("params_max_abs", "losses_max_abs", "first_half_losses_max_abs")) and rep["resume"]["bound"]
              and rep["resume"]["lr_equal"] and rep["resume"]["completed_before"] == 3),
             ("optimiser", rep["optimizer_matches_config"]),
             ("padding", rep["padding"]["loss_diff"] == 0.0 and rep["padding"]["predictions_equal"] and rep["padding"]["padded_positions"] == old_check["fact"]["padding"]["padded_positions"]
              and (rep["padding"]["streams"], rep["padding"]["natives"]) == (int(d["streams_per_batch"]), int(d["natives_per_batch"]))),
             ("gradients", rep["zero_or_missing_gradients"] == sorted(ctx.revision["known_zero_gradients"]["names"]) and rep["parameters"] == INIT["parameters"]),
             ("timing", rep["timing"]["timed_updates"] == int(ctx.fv["checks"]["timing"]["timed_updates"])))
    return [f"the new {name} check failed" for name, ok in tests if not ok]


def stage_check(ctx) -> None:
    b = ctx.fv["budget"]; cap = float(b["revalidation_gpu_hours_max"]); before = tf.ledger_hours(ctx.ledger, "gpu"); t0, started = time.time(), tf.now()
    if before >= cap:
        raise SystemExit(f"the revalidation's GPU allowance is spent ({before:.4f} of {cap} h)")
    pre, prep = tf.read_record(ctx, "revalidation_prepare.json"), tf.read_record(ctx, "preparation.json")
    for name, rec in (("revalidation_prepare.json", pre), ("preparation.json", prep)):
        if rec.get("code") != tf.code_digest() or rec.get("config_sha256") != ctx.config_sha():
            raise SystemExit(f"{name} is missing or was made by other code or another configuration; run --prepare first")
    if pre.get("passed") is not True:
        raise SystemExit("the revalidation preparation did not pass")
    cpu = cpu_evidence(ctx); _, old_check, _ = original_records(ctx)
    if json.loads(json.dumps({k: cpu[k] for k in ("science", "gate", "inherited")}, default=str)) != {k: pre[k] for k in ("science", "gate", "inherited")}:
        raise SystemExit("the CPU evidence differs from the revalidation preparation's")
    tfa.determinism(tf.ARM); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
    a01 = ctx.a01(); rule = fh.TokenRule(**prep["token_rule"]); tf.check_rule(rule, ctx.f); head = tf.make_head(ctx, rule)
    if tf.official_state(ctx.root / ctx.f["repo"], ctx.f["commit"])["digest"] != prep["official"]["digest"]:
        raise SystemExit("the official checkout changed since the preparation")
    init = {}
    for s in ctx.seeds:
        key = tf.run_key(s); p = ctx.root / prep["inits"][key]["path"]
        if sha256_file(p) != prep["inits"][key]["sha256"]:
            raise SystemExit(f"{key}: the initial state differs from the preparation record")
        init[key] = tf.strict_init(head, p)
    s = ctx.seeds[0]; data = tv.TrainSet(a01, s); new = new_checks(ctx, head, data, ctx.root / prep["inits"][tf.run_key(s)]["path"], device); del data
    tm = new["timing"]
    per_new = {tf.run_key(s): tf.run_hours(tm["fit_a_s"], tm["fit_b_s_per_sample"], prep["plans"][str(s)]["updates"], prep["plans"][str(s)]["samples"], ctx.epochs, tm["per_epoch_s"])
               for s in ctx.seeds}
    per_used = {k: max(float(old_check["hours_per_run"][k]), v) for k, v in per_new.items()}
    post_new, basis = post_hours(prep, new["decode_seconds_per_sample"], new["control_seconds_per_sample"], len(ctx.seeds))
    post_used = max(float(old_check["post_training_hours"]), post_new); idents = tf.identities(ctx, head, prep)
    hours = (time.time() - t0) / 3600
    proj = tf.projection(tf.spent_hours(ctx, "gpu") + hours, per_used, list(per_used), post_used, float(b["gpu_reserve_hours"]), float(b["gpu_hours_max"]))
    bad = failures(cpu) + new_failures(ctx, new, old_check)
    if not proj["within_cap"]:
        bad.append(f"the cumulative projection {proj['total']:.4f} exceeds {proj['cap']} GPU hours")
    if before + hours > cap:
        bad.append(f"the new GPU checks took {before + hours:.4f} h, more than the allowed {cap}")
    org = ctx.revision["original"]
    rec = {"revision": ctx.revision["name"], "kind": ctx.revision["kind"], "original": org, "original_gate_passed": False,
           "original_check": {**org["check"], "passed": old_check["passed"], "failed_requirements": cpu["inherited"]["failed_only"], "started": old_check["started"],
                              "finished": old_check["finished"]},
           "policy": {"gate": ctx.revision["gate"], "task_sheet": ctx.fv["task_sheet"]}, **{k: cpu[k] for k in ("evidence", "science", "old_code", "gate", "inherited")},
           "new": {**new, "init": init, "note": "new checks of the new path on GPU; their weights never start a formal run"},
           "hours_per_run_original": old_check["hours_per_run"], "hours_per_run_new": per_new, "hours_per_run": per_used,
           "post_training_hours_original": old_check["post_training_hours"], "post_training_hours_new": post_new, "post_training_hours": post_used,
           "post_training_basis": basis, "projection": proj, "revalidation_gpu_hours": {"before": before, "this_check": hours, "cap": cap},
           "identities": idents, "code": tf.code_digest(), "config_sha256": ctx.config_sha(), "revalidation_code_sha256": sha256_file(Path(__file__)),
           "started": started, "finished": tf.now(), "passed": not bad, "failed": bad}
    write_record(ctx, "revalidation.json", rec)
    print(json.dumps({"passed": not bad, "failed": bad, "timing_mean_s": tm["mean_s"], "hours_per_run": per_used, "post_training_hours": post_used,
                      "projection": proj, "revalidation_gpu_hours": before + hours}, indent=1, default=str), flush=True)
    if bad:
        raise SystemExit("; ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); g = ap.add_mutually_exclusive_group(required=True)
    for n in ("export_old_code", "prepare", "check"):
        g.add_argument(f"--{n.replace('_', '-')}", dest=n, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = tf.Ctx(load_cfg(a), a.config)
    if ctx.revision is None:
        raise SystemExit("this configuration carries no revision")
    if a.export_old_code:
        org = ctx.revision["original"]; rec = export_old_code(ctx.root, org["code_commit"], ctx.root / org["old_code_dir"]); verify_old_code(ctx.root / org["old_code_dir"], org["code_digest"])
        print(json.dumps({"commit": rec["commit"], "digest": rec["digest"], "files": len(rec["files"])}), flush=True)
    elif a.prepare:
        tf.run_stage(ctx, "revalidation prepare", "cpu", stage_prepare)
    else:
        tf.run_stage(ctx, "revalidation check", "gpu", stage_check)


if __name__ == "__main__":
    main()
