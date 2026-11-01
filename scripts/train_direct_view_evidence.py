#!/usr/bin/env python3
"""A51-P paired direct-view source training, final-only frozen once-only readout."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as TF
import echo_routing.vit.target_mvit as tm
from echo_routing.temporal import direct_view_training as dt
from echo_routing.temporal.direct_view_evidence import source_signal_gate
from echo_routing.temporal.pw_transfer import state_sha
import scripts.train_risk_constrained_a5c as t44
from scripts.bind_direct_view_subset import OUT, PUB
from scripts.prepare_direct_view_evidence import sha, dump

BASE = ROOT / "models_frozen/video_within_view_v1/all_s0/final.pt"
ARMS = ("L", "W")


def ctx():
    p = "configs/experiments/video_risk_constrained_a5c_v1.yaml"
    return t44.Ctx(t44.load_cfg(SimpleNamespace(config=p, set=[])), p)


def build(device="cuda"):
    return t44.build(ctx(), 0, 0, device)


class Inputs:
    def __init__(self):
        self.path = OUT / "bound.private.json"
        self.bound = json.loads(self.path.read_text())
        self.fit = [r for r in self.bound["requests"] if r["role"] == "fit"]
        self.diagnostic = [r for r in self.bound["requests"] if r["role"] == "diagnostic"]
        assert len(self.fit) == 569 and len(self.diagnostic) == 133
        self.cache = {}
        for v in self.bound["videos"]:
            p = ROOT / v["cache"]; assert sha(p) == v["cache_sha256"]
            z = np.load(p)
            self.cache[(v["folder"], v["video"])] = {int(i): f for i, f in zip(z["ordinals"], z["frames"])}
        directory = np.asarray([r["directory_label"] == "A5C" for r in self.fit], np.int64)
        self.plan = dt.source_plan(directory)
        self.plan_sha = hashlib.sha256(self.plan.astype(np.int64).tobytes()).hexdigest()

    def window(self, r):
        lookup = self.cache[(r["directory_label"], r["video"])]
        fr = np.stack([lookup[i] for i in r["frames"]])
        assert hashlib.sha256(fr.tobytes()).hexdigest() == r["pixel_sha256"]
        return fr


def forward(m, h, data, r, device="cuda", base=False):
    x = tm.to_rgb(torch.from_numpy(data.window(r)[None]).to(device))
    marker = tm.marker_planes(torch.tensor([8], device=device))
    e = m.embed(x, marker)
    if base:
        logits = m.classifier(e)
        return logits[:, 3] - logits[:, 2]
    return h(e).reshape(-1)


def flat(m, h):
    return {**{"m." + k: v.detach().cpu() for k, v in m.state_dict().items()},
            **{"h." + k: v.detach().cpu() for k, v in h.state_dict().items()}}


def max_diff(a, b):
    return max(float((a[n].double() - b[n].double()).abs().max()) for n in a)


def step(m, h, opt, data, u, arm, device="cuda", force_weak=False):
    t44.t31.sched(opt, u, {"updates": 600, "warmup": 60})
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.
    for index in data.plan[u]:
        r = data.fit[int(index)]
        label = r["true_label"] if arm == "L" and not force_weak else r["directory_label"]
        y = torch.tensor([float(label == "A5C")], device=device)
        s = forward(m, h, data, r, device)
        loss = TF.binary_cross_entropy_with_logits(s, y) / 8
        loss.backward(); loss_sum += float(loss.detach())
    g = {k: t44.sq_norm([p.grad for p in ps]) for k, ps in t44.groups_of(m, h).items()}
    norm = float(torch.nn.utils.clip_grad_norm_(t44.trainable(m, h), 1.))
    opt.step()
    return {"u": int(u), "loss": loss_sum, "gradient_norm": norm, "groups": g,
            "physical_forwards": 8, "physical_backwards": 8, "pid": os.getpid()}


def project_files():
    files = {}
    for module in list(sys.modules.values()):
        p = getattr(module, "__file__", None)
        if p:
            p = Path(p).resolve()
            if p.is_relative_to(ROOT) and ".venv" not in p.parts and p.suffix == ".py" and p.exists():
                files[str(p.relative_to(ROOT))] = sha(p)
    for name in ["docs/superpowers/plans/2026-11-01-direct-view-supervision.md", "docs/superpowers/plans/2026-11-01-direct-view-marker-amendment.md",
                 "docs/superpowers/plans/2026-11-01-direct-view-axis-check.md", "docs/superpowers/plans/2026-11-01-direct-view-verified-pilot.md"]:
        files[name] = sha(ROOT / name)
    return files


def prepare():
    audit = json.loads((PUB / "independent_source.json").read_text()); assert audit["passed"]
    data = Inputs()
    assert audit["bound_sha256"] == sha(data.path)
    p = OUT / "source_plan.npy"
    if p.exists():
        assert np.array_equal(np.load(p), data.plan), "frozen exposure plan changed"
        if list((OUT / "source_train").glob("*/claim.json")):
            raise RuntimeError("formal source model exists; cannot revise prelaunch provenance")
    else:
        np.save(p, data.plan)
    previous = PUB / "input_manifest.json"
    if previous.exists():
        history = PUB / "prelaunch_history"; history.mkdir(exist_ok=True)
        (history / (sha(previous) + ".input.json")).write_bytes(previous.read_bytes())
        old_checks = PUB / "checks.json"
        if old_checks.exists():
            (history / (sha(old_checks) + ".checks.json")).write_bytes(old_checks.read_bytes())
    rec = {"run_id": "A51-P", "source_arms": list(ARMS), "source_updates_each": 600, "windows_per_update": 8,
           "fit_requests": len(data.fit), "diagnostic_requests": len(data.diagnostic), "bound_sha256": sha(data.path),
           "source_plan_sha256": data.plan_sha, "source_plan_file_sha256": sha(p), "Base_A08_sha256": sha(BASE),
           "geometry_coefficients_sha256": hashlib.sha256(json.dumps(data.bound["geometry_control"], sort_keys=True).encode()).hexdigest(),
           "project_files": project_files(), "versions": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
           "source_train_physical_forwards": 9600, "source_train_physical_backwards": 9600,
           "labels": {"L": "manual marked-frame view", "W": "directory view", "A5C": 1, "A4C": 0, "positive_threshold": "strict >0"},
           "source_models_at_preparation": 0, "heldout_model_forwards_at_preparation": 0}
    rec["config_files"] = {"configs/experiments/video_risk_constrained_a5c_v1.yaml": sha(ROOT / "configs/experiments/video_risk_constrained_a5c_v1.yaml")}
    dump(PUB / "input_manifest.json", rec); print(json.dumps({k: v for k, v in rec.items() if k != "project_files"}))


def verify_input():
    rec = json.loads((PUB / "input_manifest.json").read_text())
    assert sha(BASE) == rec["Base_A08_sha256"]
    assert sha(OUT / "bound.private.json") == rec["bound_sha256"]
    for name, expected in rec["project_files"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in rec["config_files"].items():
        assert sha(ROOT / name) == expected, name
    return rec


def check():
    manifest = verify_input(); data = Inputs(); start = time.time()
    m, h = build(); opt = t44.make_opt(m, h)
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in t44.trainable(m, h)}
    initial = {n: v.clone() for n, v in flat(m, h).items()}
    with torch.no_grad():
        s0 = [float(forward(m, h, data, data.fit[int(i)])) for i in data.plan[0][:4]]
    assert s0 == [-2.] * 4
    # Independent float64 BCE values; retained-graph vs microbatch gradients.
    rows = [data.fit[int(i)] for i in data.plan[0][:4]]
    y = np.asarray([r["true_label"] == "A5C" for r in rows], float)
    predicted = np.asarray(s0)
    reference = float(np.mean(np.logaddexp(0., predicted) - y * predicted))
    opt.zero_grad(set_to_none=True)
    loss = torch.stack([TF.binary_cross_entropy_with_logits(forward(m, h, data, r), torch.tensor([float(r["true_label"] == "A5C")], device="cuda")) for r in rows]).mean()
    assert abs(float(loss.detach()) - reference) < 1e-6
    loss.backward()
    bias_reference = float(np.mean(1. / (1. + np.exp(-predicted)) - y))
    bias_gradient_error = abs(float(h.fc.bias.grad[0]) - bias_reference)
    assert bias_gradient_error < 1e-6
    full = {n: p.grad.detach().cpu().clone() for n, p in list(m.named_parameters()) + [("head." + n, p) for n, p in h.named_parameters()] if p.grad is not None}
    opt.zero_grad(set_to_none=True)
    for r in rows:
        (TF.binary_cross_entropy_with_logits(forward(m, h, data, r), torch.tensor([float(r["true_label"] == "A5C")], device="cuda")) / 4).backward()
    micro = {n: p.grad.detach().cpu().clone() for n, p in list(m.named_parameters()) + [("head." + n, p) for n, p in h.named_parameters()] if p.grad is not None}
    grad_error = max_diff(full, micro); assert grad_error < 1e-5
    logs = [step(m, h, opt, data, u, "L") for u in range(6)]
    assert logs[0]["groups"]["encoder_trunk"] == 0.
    assert all(logs[1]["groups"][k] > 0 for k in ("encoder_trunk", "marker_stem", "projection", "scalar_head"))
    state6 = {n: v.clone() for n, v in flat(m, h).items()}
    m2, h2 = build(); op2 = t44.make_opt(m2, h2)
    assert state_sha(flat(m2, h2)) == state_sha(initial)
    for u in range(3): step(m2, h2, op2, data, u, "L")
    rp = OUT / "check_resume.pt"
    torch.save(t44.state_of(m2, h2, op2, 1.), rp)
    m3, h3 = build(); op3 = t44.make_opt(m3, h3)
    t44.load_state(m3, h3, op3, torch.load(rp, weights_only=False, map_location="cpu"))
    for u in range(3, 6): step(m3, h3, op3, data, u, "L")
    resume_error = max_diff(state6, flat(m3, h3)); assert resume_error < 1e-5
    for u in range(3, 6): step(m2, h2, op2, data, u, "L")
    repeat_error = max_diff(state6, flat(m2, h2)); assert repeat_error < 1e-5
    del m2, h2, op2, m3, h3, op3; rp.unlink(); torch.cuda.empty_cache()
    mw, hw = build(); ow = t44.make_opt(mw, hw)
    mf, hf = build(); of = t44.make_opt(mf, hf)
    for u in range(2):
        step(mw, hw, ow, data, u, "W"); step(mf, hf, of, data, u, "L", force_weak=True)
    forced_error = max_diff(flat(mw, hw), flat(mf, hf)); assert forced_error < 1e-5
    assert all(torch.equal(v, initial[n]) for n, v in flat(m, h).items() if n.startswith("m.classifier."))
    # Timing excludes full checks; exactly six known fit-only production steps.
    torch.cuda.synchronize(); t = time.time()
    for u in range(6, 12): step(m, h, opt, data, u, "L")
    torch.cuda.synchronize(); sec = (time.time() - t) / 6
    result = {"passed": True, "input_manifest_sha256": sha(PUB / "input_manifest.json"),
              "source_formula_error": abs(float(loss.detach()) - reference), "microbatch_gradient_max_abs_error": grad_error,
              "independent_float64_bias_gradient_error": bias_gradient_error, "optimizer_binding_complete": True,
              "repeat_max_abs_error": repeat_error, "resume_3plus3_max_abs_error": resume_error, "forced_W_max_abs_error": forced_error,
              "Init_scores": s0, "first_step_trunk_zero_expected": True, "subsequent_all_four_groups_nonzero": True,
              "old_5class_head_frozen": True, "plan_covers_all569fit_requests": set(data.plan.ravel()) == set(range(569)),
              "seconds_per_source_update": sec, "source_2x600_forecast_GPU_job_seconds": sec * 1200 * 1.3 + 180,
              "peak_GPU_bytes": torch.cuda.max_memory_allocated(), "wall_seconds": time.time() - start,
              "all_checks_fit_only": True, "diagnostic_model_forwards": 0, "Base_A08_unchanged": sha(BASE) == manifest["Base_A08_sha256"]}
    dump(PUB / "checks.json", result); print(json.dumps(result))


@torch.no_grad()
def fit_scores(m, h, data):
    return np.asarray([float(forward(m, h, data, r)) for r in data.fit], np.float32)


def make_bundle(m, h, metadata):
    state = flat(m, h)
    live = {"m." + n for n, p in m.named_parameters() if p.requires_grad} | {"h." + n for n, p in h.named_parameters()}
    return {"base_sha256": sha(BASE), "live": {n: state[n].clone() for n in sorted(live)},
            "full_state_sha256": state_sha(state), "frozen_state_sha256": state_sha({n: v for n, v in state.items() if n not in live}),
            "metadata": metadata}


def restore(bundle, device="cuda"):
    m, h = build(device)
    assert bundle["base_sha256"] == sha(BASE)
    state = flat(m, h); live = set(bundle["live"])
    expected = {"m." + n for n, p in m.named_parameters() if p.requires_grad} | {"h." + n for n, p in h.named_parameters()}
    assert live == expected
    assert state_sha({n: v for n, v in state.items() if n not in live}) == bundle["frozen_state_sha256"]
    ms, hs = m.state_dict(), h.state_dict()
    for n, value in bundle["live"].items():
        (ms if n.startswith("m.") else hs)[n[2:]] = value
    m.load_state_dict(ms, strict=True); h.load_state_dict(hs, strict=True)
    assert state_sha(flat(m, h)) == bundle["full_state_sha256"]
    return m.eval(), h.eval()


def train():
    manifest = verify_input(); checks = json.loads((PUB / "checks.json").read_text()); assert checks["passed"]
    assert checks["input_manifest_sha256"] == sha(PUB / "input_manifest.json"), "stale check receipt"
    pre = json.loads((OUT / "prelaunch_freeze.json").read_text())
    assert pre["input_manifest_sha256"] == sha(PUB / "input_manifest.json") and pre["actual_git_objects_verified"]
    data = Inputs(); entries = {}
    for arm in ARMS:
        rd = OUT / "source_train" / arm; rd.mkdir(parents=True, exist_ok=True)
        dt.sync_directory(rd.parent); dt.sync_directory(OUT)
        ident = {"arm": arm, "input_manifest_sha256": sha(PUB / "input_manifest.json"), "plan_sha256": data.plan_sha}
        if (rd / "manifest.json").exists():
            entry = json.loads((rd / "manifest.json").read_text())
            assert entry["identity"] == ident and entry["updates"] == 600 and sha(rd / "final.pt") == entry["bundle_sha256"]
            assert sha(rd / "optimizer_final.pt") == entry["optimizer_final_sha256"]
            entries[arm] = entry; continue
        claim, rp, journal = rd / "claim.json", rd / "resume.pt", rd / "updates.private.jsonl"
        m, h = build(); opt = t44.make_opt(m, h); initial = {n: v.clone() for n, v in flat(m, h).items()}
        if claim.exists():
            assert json.loads(claim.read_text()) == ident
            if not rp.exists() and (rd / "optimizer_final.pt").exists(): rp = rd / "optimizer_final.pt"
            if not rp.exists(): raise RuntimeError("claimed source model has no safe resumable optimizer state")
            st = torch.load(rp, weights_only=False, map_location="cpu"); assert st["identity"] == ident
            t44.load_state(m, h, opt, st); start_u = int(st["update"])
            logs = [json.loads(line) for line in journal.read_text().splitlines()]
            dt.validate_recovery(rd, start_u, len(logs))
        else:
            with claim.open("x") as f: json.dump(ident, f); f.flush(); os.fsync(f.fileno())
            dt.sync_directory(rd)
            start_u = 0; logs = []
            np.save(rd / "fit_init.npy", fit_scores(m, h, data))
        start_wall = time.time()
        with journal.open("a") as jf:
            def persist(rec):
                jf.write(json.dumps(rec) + "\n"); jf.flush(); os.fsync(jf.fileno()); logs.append(rec)
            for u in range(start_u, 600):
                def compute():
                    rec = step(m, h, opt, data, u, arm)
                    rec["device"] = torch.cuda.get_device_name(0)
                    return rec
                rec = dt.claimed_update(rd, u, compute, persist)
                if (u + 1) % 25 == 0:
                    dt.write_json(OUT / "progress.json", {"run_id": "A51-P", "stage": "source_training", "model": arm,
                        "update": u + 1, "total": 600, "pid": os.getpid(), "device": rec["device"], "updated_epoch": time.time()})
                    print(json.dumps({"arm": arm, "update": u + 1, "loss": rec["loss"], "pid": os.getpid()}), flush=True)
                if (u + 1) % 150 == 0:
                    temp = rp.with_suffix(".tmp")
                    with temp.open("wb") as f:
                        torch.save({**t44.state_of(m, h, opt, 1.), "identity": ident, "update": u + 1}, f)
                        f.flush(); os.fsync(f.fileno())
                    temp.replace(rp); dt.sync_directory(rd)
        assert len(logs) == 600 and [r["u"] for r in logs] == list(range(600))
        steps = [int(v["step"]) for v in opt.state.values()]
        assert steps and set(steps) == {600}
        fs = fit_scores(m, h, data); np.save(rd / "fit_final.npy", fs)
        final = flat(m, h)
        assert all(torch.equal(final[n], value) for n, value in initial.items() if n.startswith("m.classifier."))
        bundle = make_bundle(m, h, ident)
        with (rd / "final.pt").open("wb") as f:
            torch.save(bundle, f); f.flush(); os.fsync(f.fileno())
        dt.sync_directory(rd)
        # CPU complete-state reconstruction, no heldout forward.
        cm, ch = restore(torch.load(rd / "final.pt", weights_only=True, map_location="cpu"), "cpu")
        assert state_sha(flat(cm, ch)) == bundle["full_state_sha256"]
        drift = {n: float((final[n].double() - initial[n].double()).norm()) for n in final}
        groups = {k: max(r["groups"][k] for r in logs[1:]) for k in logs[0]["groups"]}
        assert all(groups[k] > 0 for k in groups)
        optimizer_final = rd / "optimizer_final.pt"
        if rp != optimizer_final: rp.replace(optimizer_final); dt.sync_directory(rd)
        entry = {"identity": ident, "updates": 600, "optimizer_step_values": sorted(set(steps)), "optimizer_state_tensors": len(steps),
                 "optimizer_final_sha256": sha(optimizer_final), "optimizer_RNG_state_retained_for_independent_audit": True,
                 "physical_training_forwards": sum(r["physical_forwards"] for r in logs), "physical_training_backwards": sum(r["physical_backwards"] for r in logs),
                 "bundle_sha256": sha(rd / "final.pt"), "complete_full_state_sha256": bundle["full_state_sha256"],
                 "frozen_state_sha256": bundle["frozen_state_sha256"], "initial_full_state_sha256": state_sha(initial),
                 "journal_sha256": sha(journal), "fit_init_sha256": sha(rd / "fit_init.npy"), "fit_final_sha256": sha(rd / "fit_final.npy"),
                 "group_gradient_max_after_first_step": groups, "drift_by_tensor": drift, "old_5class_head_frozen": True,
                 "model_wall_seconds": time.time() - start_wall, "source_A08_unchanged": sha(BASE) == manifest["Base_A08_sha256"]}
        dump(rd / "manifest.json", entry); entries[arm] = entry
        del m, h, opt, cm, ch; torch.cuda.empty_cache()
    record = {"run_id": "A51-P", "input_manifest_sha256": sha(PUB / "input_manifest.json"), "models": entries,
              "formal_source_updates": 1200, "source_training_forwards": 9600, "source_training_backwards": 9600,
              "diagnostic_model_forwards": 0, "Base_A08_sha256": sha(BASE)}
    dump(PUB / "source_models_manifest.json", record); print(json.dumps({"formal_source_updates": 1200, "all_source_models_complete": True}))


def readout():
    current = verify_input(); manifest = json.loads((PUB / "source_models_manifest.json").read_text())
    assert manifest["input_manifest_sha256"] == sha(PUB / "input_manifest.json"), "stale source-model manifest"
    assert manifest["Base_A08_sha256"] == current["Base_A08_sha256"]
    assert all(manifest["models"][a]["identity"]["input_manifest_sha256"] == sha(PUB / "input_manifest.json") for a in ARMS)
    freeze = json.loads((OUT / "source_models_freeze.json").read_text())
    assert freeze["manifest_sha256"] == sha(PUB / "source_models_manifest.json") and freeze["actual_git_objects_verified"]
    data = Inputs(); values = {}
    for arm in ("G", "L", "W"):
        model_sha = manifest["Base_A08_sha256"] if arm == "G" else manifest["models"][arm]["bundle_sha256"]
        identity = {"arm": arm, "model_sha256": model_sha, "bound_sha256": sha(data.path), "rows": 133}
        cell = dt.open_cell(OUT / "source_cells", arm, identity, 133)
        if arm == "G": m, h = build()
        else:
            p = OUT / "source_train" / arm / "final.pt"; assert sha(p) == model_sha
            m, h = restore(torch.load(p, weights_only=True, map_location="cpu"))
        with torch.no_grad():
            while (i := dt.claim_row(cell)) is not None:
                s = float(forward(m, h, data, data.diagnostic[i], base=arm == "G"))
                dt.finish_row(cell, i, s)
        dt.seal_cell(cell); values[arm] = np.asarray(cell["scores"]).copy(); dt.close_cell(cell)
        del m, h; torch.cuda.empty_cache()
    ctrl = data.bound["geometry_control"]
    x = np.asarray([r["geometry"] for r in data.diagnostic], float)
    values["geometry"] = ((x - ctrl["fit_mean"]) / ctrl["fit_scale"]) @ np.asarray(ctrl["coef"])[0] + ctrl["intercept"][0]
    p = ctrl["constant_prior"]; values["constant"] = np.full(133, np.log(p / (1 - p)))
    dest = OUT / "source_scores.private.npz"
    if dest.exists(): raise RuntimeError("source score aggregate already exists; use cache-only audit")
    with dest.open("xb") as f: np.savez_compressed(f, **values); f.flush(); os.fsync(f.fileno())
    y = np.asarray([r["true_label"] == "A5C" for r in data.diagnostic])
    discord = np.asarray([r["true_label"] != r["directory_label"] for r in data.diagnostic])
    metrics = {arm: dt.source_metrics(y, s, discord) for arm, s in values.items()}
    gate = source_signal_gate(metrics["L"], metrics["W"], metrics["geometry"]["balanced_accuracy"], metrics["constant"]["balanced_accuracy"])
    result = {"run_id": "A51-P", "source_diagnostic_requests": 133, "source_diagnostic_patients": len({r["patient"] for r in data.diagnostic}),
              "manual_A4C_requests": int((~y).sum()), "manual_A5C_requests": int(y.sum()), "directory_discordant_requests": int(discord.sum()),
              "metrics": metrics, "source_advancement_gate": gate, "raw_score_cache_sha256": sha(dest),
              "source_model_freeze": freeze, "source_model_forwards": 399,
              "target_models_or_A5C_recovery_results": 0, "original_full_source_passed": False,
              "scope": "Verified-subset heldout source view classification, not EV9V A5C recovery or anatomy/clinical evidence."}
    dump(PUB / "source_readout.json", result); print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["prepare", "check", "train", "readout"]); args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    started = time.time(); ok = False
    stage_lock = dt.acquire_lock(OUT / "exclusive_worker.lock")
    try:
        {"prepare": prepare, "check": check, "train": train, "readout": readout}[args.stage](); ok = True
    finally:
        stage_lock.close()
        u = resource.getrusage(resource.RUSAGE_SELF)
        receipt = OUT / (args.stage + "_model_process.json")
        if receipt.exists():
            history = OUT / "stage_history"; history.mkdir(exist_ok=True)
            (history / (sha(receipt) + ".json")).write_bytes(receipt.read_bytes())
        dump(receipt, {"stage": args.stage, "ok": ok, "pid": os.getpid(), "start_epoch": started,
             "end_epoch": time.time(), "wall_seconds": time.time() - started, "CPU_seconds": u.ru_utime + u.ru_stime,
             "GPU_job_wall_seconds": time.time() - started if args.stage != "prepare" else 0})


if __name__ == "__main__":
    main()
