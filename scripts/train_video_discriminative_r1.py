#!/usr/bin/env python
"""A09-R1: complete A09's plain arm C along its unchanged 3,750-update schedule (configs/experiments/video_discriminative_r1.yaml).

A09's own trainer (scripts/train_video_discriminative.py, unchanged and pinned) does every update. This wrapper only
verifies the parent, copies each seed's update-1,250 resume state (read only at its source) into this round's
checkpoint directory, points the trainer's run and checkpoint directories at this round, and books a separate ledger.
The trainer's resume check recomputes the parent identity from the parent's configuration and code, so a changed file
or a renamed configuration cannot resume.

  --check            parent pins and identities, full state restore, optimiser binding, input/output parity, GPU, budget
  --queue S [S ...]  continuations of the parent seeds, in order
  --seed1            the conditional seed-1 run from V01 Ordered s1 (only when the R1 decision record says so)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

ARM, PILOT = "plain", 1250
R1_KEYS = ("parent", "arm", "continuation", "seed1_rule")


class R1:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.r = cfg, config, cfg["video_discriminative_r1"]; par = self.r["parent"]
        for k in ("config", "trainer", "evaluator", "manifests_index", "pilot_record", "pilot_choices"):
            td.pinned(par[k])
        if sha256_file(Path(td.__file__)) != par["trainer"]["sha256"]:
            raise SystemExit("the imported trainer differs from the pinned parent trainer")
        pc = load_cfg(SimpleNamespace(config=par["config"]["path"], set=[])); self.a09 = td.Ctx(pc, par["config"]["path"])
        self.t = td.Ctx(pc, par["config"]["path"])           # the parent's manifests and frame paths, this round's directories
        o = self.r["outputs"]; self.out, self.ckpt, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.t.out, self.t.ckpt_root = self.out, self.ckpt; self.ledger = ROOT / self.r["budget"]["ledger"]

    def parent_identity(self, seed: int) -> tuple[str, str, str]:
        _, msha = self.a09.manifest(seed); _, ssha = self.a09.a07.source_state(seed)
        return td.identity(self.a09, ARM, seed, msha, ssha, 0.0), msha, ssha


def code_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes() + td.code_digest().encode()).hexdigest()


def r1_identity(r1: R1, seed: int, parent: str, source_sha: str) -> str:
    payload = {"r1": {k: r1.r[k] for k in R1_KEYS}, "seed": int(seed), "parent_identity": parent, "source_state_sha256": source_sha, "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def verify_parent(r1: R1, seed: int) -> dict:
    """CPU: the recomputed parent identity equals the configured one, the pilot record and the resume state; the resume
    weights equal the evaluated pilot snapshot; the manifest's epoch digests equal index.json's."""
    p = r1.r["parent"]["runs"][seed]; ident, msha, ssha = r1.parent_identity(seed)
    pj = json.loads((r1.a09.out / "train" / f"{ARM}_s{seed}" / f"pilot_u{PILOT}.json").read_text())
    if not (ident == p["identity"] == pj["identity"]) or int(pj["updates"]) != PILOT or pj["sha256"] != p["pilot_snapshot"]["sha256"]:
        raise SystemExit(f"seed {seed}: the parent identity or pilot record differs")
    st = torch.load(td.pinned(p["resume"]), map_location="cpu", weights_only=False); snap = torch.load(td.pinned(p["pilot_snapshot"]), map_location="cpu", weights_only=True)
    if st["identity"] != ident or int(st["updates"]) != PILOT:
        raise SystemExit(f"seed {seed}: the resume state belongs to another identity or update")
    same = set(snap) == set(st["model"]) and all(torch.equal(snap[k], st["model"][k]) for k in snap)
    m, _ = r1.a09.manifest(seed); idx = json.loads((r1.a09.mdir / "index.json").read_text())["seeds"][str(seed)]
    epochs_ok = vpr.epoch_digests(m, r1.a09.upe) == idx["epoch_digests"]
    if not (same and epochs_ok):
        raise SystemExit(f"seed {seed}: resume weights equal snapshot {same}, epoch digests equal {epochs_ok}")
    return {"parent_identity": ident, "manifest_sha256": msha, "source_weights_sha256": ssha, "resume_sha256": p["resume"]["sha256"],
            "pilot_snapshot_sha256": pj["sha256"], "resume_weights_equal_snapshot": same, "epoch_digests_equal": epochs_ok,
            "log_updates": [int(x["updates"]) for x in st["log"]]}


def other_gpu_processes() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
    return [p for p in out if p.strip() and int(p) != os.getpid()]


def check(r1: R1, device, workers: int) -> None:
    t0, started, ok, note = time.time(), td.now(), False, ""; out = {"config_sha256": sha256_file(ROOT / r1.config), "started": started, "seeds": {}}
    try:
        others = other_gpu_processes(); out["other_gpu_processes"] = len(others)
        if others:
            raise SystemExit(f"{len(others)} other GPU process(es) present; nothing starts")
        td.determinism(); a = r1.a09; tmp = r1.out / "check_resume.pt"; r1.out.mkdir(parents=True, exist_ok=True)
        for s in [int(x) for x in r1.r["queue"]]:
            v = verify_parent(r1, s); p = r1.r["parent"]["runs"][s]; state, _ = a.a07.source_state(s)
            shutil.copyfile(td.pinned(p["resume"]), tmp)
            model = td.make_model(state, device); opt = td.make_optimizer(model, a.o); bound = td.optimiser_bound(opt, model)
            done, log = td.load_resume(tmp, model, opt, v["parent_identity"]); tmp.unlink()
            steps_new = {int(opt.state[q]["step"]) for q in model.new_parameters() if q in opt.state}
            steps_enc = {int(opt.state[q]["step"]) for q in model.encoder_parameters() if q in opt.state}
            lrs = td.apply_schedule(model, opt, PILOT, a.warmup, a.total, a.o); f = lr_factor(PILOT, a.warmup, a.total)
            m, _ = a.manifest(s); b = td.Inputs(m, a.frame_paths(), [PILOT])[0]; td.rvc.check_digests(m, td.GROUP_NAME, PILOT, b[4])
            snap = td.make_model(torch.load(td.pinned(p["pilot_snapshot"]), map_location="cpu", weights_only=True), device)
            model.eval(); snap.eval()
            with torch.no_grad():
                z1 = model(td.tm.to_rgb(b[1].to(device)), td.tm.marker_planes(b[2].to(device))); z2 = snap(td.tm.to_rgb(b[1].to(device)), td.tm.marker_planes(b[2].to(device)))
            out["seeds"][str(s)] = {**v, "restored_update": int(done), "optimiser_bound": bool(bound), "adam_steps_new": sorted(steps_new), "adam_steps_encoder": sorted(steps_enc),
                                    "lr_at_1250": list(lrs), "lr_factor_at_1250": f, "input_digests_equal": True, "eval_logits_equal_snapshot": bool(torch.equal(z1, z2)),
                                    "r1_identity": r1_identity(r1, s, v["parent_identity"], v["resume_sha256"])}
            good = (done == PILOT and bound and steps_new == {PILOT} and steps_enc == {PILOT - a.warmup} and bool(torch.equal(z1, z2))
                    and abs(lrs[1] - float(a.o["lr_new"]) * f) < 1e-15 and abs(lrs[0] - float(a.o["lr_encoder"]) * f) < 1e-15)
            out["seeds"][str(s)]["ok"] = bool(good); del model, snap, opt; torch.cuda.empty_cache()
            print(f"seed {s}: {json.dumps({k: out['seeds'][str(s)][k] for k in ('restored_update', 'adam_steps_new', 'adam_steps_encoder', 'lr_at_1250', 'eval_logits_equal_snapshot', 'ok')})}", flush=True)
        spu = float(json.loads((r1.a09.out / "checks.json").read_text())["timing"]["seconds_per_update_compute"]) * 1.10
        rem = sum(a.total - PILOT for _ in r1.r["queue"]) * spu / 3600; seed1 = a.total * spu / 3600; spent = td.ledger_hours(r1.ledger, "gpu") + (time.time() - t0) / 3600
        out["budget"] = {"spent": spent, "continuations": rem, "evaluation_reserve": 0.4, "seed1_with_evaluation": seed1 + 0.3, "cap": float(r1.r["budget"]["gpu_hours_max"])}
        out["budget"]["within_cap"] = bool(spent + rem + 0.4 + seed1 + 0.3 <= out["budget"]["cap"])
        out["ok"] = bool(all(v["ok"] for v in out["seeds"].values()) and out["budget"]["within_cap"]); ok = out["ok"]
        out["finished"] = td.now(); out["seconds"] = time.time() - t0; note = f"budget projection {spent + rem + 0.7 + seed1:.3f} of {out['budget']['cap']}"
        for q in (r1.out / "checks.json", r1.results / "checks.json"):
            q.parent.mkdir(parents=True, exist_ok=True); td.write_json(q, out)
        print(json.dumps(out["budget"]), "ok", out["ok"], flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        td.ledger_append(r1.ledger, "R1 check", "gpu", started, time.time() - t0, ok, note)


def run_one(r1: R1, seed: int, device, workers: int, fresh: bool) -> None:
    """One run under the parent's trainer: a continuation from the copied resume state, or (seed 1) a fresh run."""
    t0, started, ok, note = time.time(), td.now(), False, ""; key = f"{ARM}_s{seed}"; rd, ck = r1.out / "train" / key, r1.ckpt / key
    try:
        others = other_gpu_processes()
        if others:
            raise SystemExit(f"{len(others)} other GPU process(es) present; nothing starts")
        chk = json.loads((r1.out / "checks.json").read_text())
        if not chk["ok"]:
            raise SystemExit("the R1 checks did not pass")
        spent = td.ledger_hours(r1.ledger, "gpu"); spu = float(json.loads((r1.a09.out / "checks.json").read_text())["timing"]["seconds_per_update_compute"]) * 1.10
        need = (r1.a09.total - (0 if fresh else PILOT)) * spu / 3600 + 0.3
        if spent + need > float(r1.r["budget"]["gpu_hours_max"]):
            raise SystemExit(f"projection {spent + need:.3f} exceeds the cap; nothing starts")
        rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
        if fresh:
            _, msha, ssha = r1.parent_identity(seed); parent = td.identity(r1.a09, ARM, seed, msha, ssha, 0.0); src = None; source_sha = ssha
        else:
            v = verify_parent(r1, seed); parent, p = v["parent_identity"], r1.r["parent"]["runs"][seed]; src = td.pinned(p["resume"]); source_sha = p["resume"]["sha256"]
            if not (ck / "resume.pt").is_file() and not (ck / "final.pt").is_file():
                shutil.copyfile(src, ck / "resume.pt")
                if sha256_file(ck / "resume.pt") != source_sha:
                    raise SystemExit("the copied resume state differs from its source")
        launch = {"key": key, "parent_identity": parent, "source": str(src) if src else "V01 Ordered start weights", "source_sha256": source_sha,
                  "r1_identity": r1_identity(r1, seed, parent, source_sha), "fresh": fresh, "pid": os.getpid(), "time": td.now(),
                  "config_sha256": sha256_file(ROOT / r1.config), "r1_code": code_digest()}
        with (rd / "launches.jsonl").open("a") as fh:
            fh.write(json.dumps(launch) + "\n")
        res = td.train(r1.t, ARM, seed, device, workers, None)
        man = json.loads((rd / "run_manifest.json").read_text())
        if man["identity"] != parent or int(man["updates"]) != r1.a09.total:
            raise SystemExit("the finished run's identity or update count differs")
        upd = [json.loads(x)["u"] for x in (rd / "updates.jsonl").read_text().splitlines()]
        rec = {**launch, "final_sha256": man["final_sha256"], "updates_run": [upd[0], upd[-1]] if upd else None, "updates_logged": len(upd), "finished": td.now()}
        td.write_json(rd / "r1_run.json", rec); ok = True; note = json.dumps({"status": res.get("status"), "first_update": rec["updates_run"], "final": man["final_sha256"][:16]})
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        td.ledger_append(r1.ledger, f"R1 train {key}" + (" (fresh)" if fresh else " 1250->3750"), "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+", type=int)
    g.add_argument("--seed1", action="store_true")
    a = ap.parse_args(); a.set = []; r1 = R1(load_cfg(a), a.config); td.determinism(); device = torch.device("cuda")
    if a.check:
        return check(r1, device, a.workers)
    if a.queue:
        for s in a.queue:
            if s not in [int(x) for x in r1.r["queue"]]:
                raise SystemExit(f"seed {s} is not a configured continuation")
            run_one(r1, s, device, a.workers, fresh=False)
        return
    dec = json.loads((r1.out / "evaluation" / "two_seed" / "decision.json").read_text())
    if not dec.get("train_seed1"):
        raise SystemExit("the R1 decision record does not allow seed 1")
    run_one(r1, 1, device, a.workers, fresh=True)


if __name__ == "__main__":
    main()
