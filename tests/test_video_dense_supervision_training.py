"""CPU checks of the video dense supervision training entry (configs/experiments/video_dense_supervision_v1.yaml, Task 3):
the run identity covers the configuration's training sections (supervision and budget included), the arm, the seed, the
manifests, the label file, the source weights and the code; the completion rule; the optimiser stays bound to the loaded
weights after a resume; the three arms of a seed see the same batch and use the RNG identically; the arm labels; the
budget counts failed attempts; and the round's training code never refers to development, test or Pediatric data."""
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.video_dense_supervision as vds
import scripts.check_video_dense_supervision as cv
import scripts.prepare_video_dense_supervision as pd
import scripts.queue_video_dense_supervision as qv
import scripts.train_video_dense_supervision as tv
from echo_routing.evaluate.complementarity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_dense_supervision_v1.yaml").read_text())
RV = yaml.safe_load((ROOT / "configs/experiments/raw_video_context_v1.yaml").read_text())["raw_video_context"]


def fake_ctx(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(cfg=cfg, vd=cfg["video_dense_supervision"], rv=RV, total=3750, epoch_digests=lambda s: [f"e{s}_{i}" for i in range(10)])


def ident(ctx, arm="aligned_dense", seed=0, m="m" * 64, a="a" * 64, s="s" * 64) -> str:
    return tv.identity(ctx, arm, seed, m, a, s)


def test_identity_covers_supervision_budget_inputs_and_code(monkeypatch):
    base = ident(fake_ctx(CFG))
    assert base == ident(fake_ctx(deepcopy(CFG)))
    changed = []
    for path, value in ((("aux", "lambda", "aligned_dense"), 0.25), (("budget", "gpu_hours_max"), 9.0), (("training", "total_updates"), 3000),
                        (("model", "anchors"), [1, 3, 5, 7, 9, 11, 13, 15]), (("aux", "shuffled"), "other rule")):
        c = deepcopy(CFG); node = c["video_dense_supervision"]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value; changed.append(ident(fake_ctx(c)))
    ctx = fake_ctx(CFG)
    changed += [ident(ctx, arm="shuffled_dense"), ident(ctx, seed=1), ident(ctx, m="x" * 64), ident(ctx, a="y" * 64), ident(ctx, s="z" * 64)]
    monkeypatch.setattr(tv, "code_digest", lambda: "other code"); changed.append(ident(ctx))
    assert base not in changed and len(set(changed)) == len(changed)


def test_code_digest_covers_the_model_preparation_and_training_files():
    files = {Path(p).name for p in tv.CODE_FILES}
    assert {"video_dense_supervision.py", "target_mvit.py", "target_patch_readout.py", "train_video_dense_supervision.py",
            "prepare_video_dense_supervision.py", "train_raw_video_context.py"} <= files


def test_a_run_is_finished_only_with_its_identity_and_intact_weights(tmp_path):
    final = tmp_path / "final.pt"; torch.save({"w": torch.ones(2)}, final); man = tmp_path / "run_manifest.json"
    man.write_text(json.dumps({"identity": "abc", "final_checkpoint": str(final), "final_sha256": sha256_file(final)}))
    assert tv.finished(man, "abc") and not tv.finished(man, "abd")
    torch.save({"w": torch.zeros(2)}, final)
    assert not tv.finished(man, "abc") and not tv.finished(tmp_path / "missing.json", "abc")


def test_arm_labels_and_weights():
    aux = {"aligned": np.zeros((2, 8, 8), np.int8), "shuffled": np.ones((2, 8, 8), np.int8)}
    assert tv.arm_labels(aux, "center_only") is aux["aligned"] and tv.arm_labels(aux, "aligned_dense") is aux["aligned"]
    assert tv.arm_labels(aux, "shuffled_dense") is aux["shuffled"]
    with pytest.raises(ValueError):
        tv.arm_labels(aux, "other")
    assert [tv.arm_lambda(CFG["video_dense_supervision"], a) for a in vds.ARMS] == [0.0, 0.5, 0.5]


@pytest.fixture(scope="module")
def state():
    torch.manual_seed(0); base = tm.build_model(None, tm.initial_new_state(0))
    with torch.no_grad():
        base.marker_stem.weight.normal_(0, 0.01)
    return {k: v.clone() for k, v in base.state_dict().items()}


def test_the_optimiser_stays_bound_to_the_loaded_weights_after_a_resume(state, tmp_path):
    o = CFG["video_dense_supervision"]["training"]["optimiser"]
    m = tv.make_model(state, "cpu"); opt = tv.make_optimizer(m, o)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert ids == {id(p) for p in m.parameters()} and [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in m.encoder_parameters()]
    assert all(torch.equal(m.state_dict()[k], v) for k, v in state.items())
    with torch.no_grad():
        m.classifier.weight.add_(1.0)
    tv.save_resume(tmp_path / "r.pt", m, opt, 7, "id", [{"epoch": 0}])
    m2 = tv.make_model(state, "cpu"); opt2 = tv.make_optimizer(m2, o); done, log = tv.load_resume(tmp_path / "r.pt", m2, opt2, "id")
    assert done == 7 and log == [{"epoch": 0}] and torch.equal(m2.classifier.weight, m.classifier.weight)
    assert [id(p) for g in opt2.param_groups for p in g["params"]] == [id(p) for p in m2.parameters()]
    with pytest.raises(SystemExit):
        tv.load_resume(tmp_path / "r.pt", m2, opt2, "another identity")


def test_the_three_arms_see_the_same_batch_and_use_the_rng_identically(state):
    n = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        g = torch.Generator().manual_seed(5); u8 = torch.randint(0, 255, (2, 16, 224, 224, 3), generator=g, dtype=torch.uint8)
        batch = (0, u8, torch.tensor([8, 3]), torch.tensor([2, 3]), torch.zeros(2, 32, dtype=torch.uint8))
        aligned = np.array([[2] * 8, [2, 2, 2, 3, 3, 3, 3, 3]], np.int8); aux = {"aligned": aligned[None], "shuffled": aligned[None][:, :, ::-1].copy()}
        o = CFG["video_dense_supervision"]["training"]["optimiser"]; out = {}
        for arm in vds.ARMS:
            m = tv.make_model(state, "cpu"); opt = tv.make_optimizer(m, o); tv.apply_schedule(m, opt, 0, 375, 3750, o)
            torch.manual_seed(13)
            rec = tv.train_step(m, opt, batch, tv.arm_labels(aux, arm)[0], aux["aligned"][0], tv.arm_lambda(CFG["video_dense_supervision"], arm), (0.0, 1e-4), o, "cpu", keep=True)
            out[arm] = (rec["outputs"][0], rec["outputs"][1], torch.get_rng_state(), rec)
    finally:
        torch.set_num_threads(n)
    ref = out["center_only"]
    assert all(torch.equal(ref[i], out[a][i]) for a in vds.ARMS for i in range(3))
    assert out["center_only"][3]["loss"] == pytest.approx(out["center_only"][3]["center"]) and out["aligned_dense"][3]["time"] != out["shuffled_dense"][3]["time"]
    assert all(np.isfinite(out[a][3]["grad_norm"]) for a in vds.ARMS)


def test_budget_projection_counts_failed_attempts(tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.write_text("\n".join(json.dumps(x) for x in ({"job": "check", "hours": 0.2, "ok": False}, {"job": "check", "hours": 0.3, "ok": True},
                                                       {"job": "train center_only_s0", "hours": 0.5, "ok": True})) + "\n")
    assert tv.ledger_hours(led) == pytest.approx(1.0)
    assert qv.projected(1.0, remaining=8, per_run=0.5, allowance=2.0) == pytest.approx(7.0)
    tv.ledger_append(led, "check", "2026-10-14T00:00:00Z", 0.1, False, "boom")
    assert tv.ledger_hours(led) == pytest.approx(1.1) and json.loads(led.read_text().splitlines()[-1])["ok"] is False


def test_the_queue_and_checks_follow_the_config():
    vd = CFG["video_dense_supervision"]
    assert vd["queue"] == [f"{a}_s{s}" for s in (0, 1, 2) for a in vds.ARMS] and [tv.run_key(a, 0) for a in vds.ARMS] == vd["queue"][:3]
    assert vd["checks"]["cls_parity_max_abs"] == 0.0 and vd["budget"]["gpu_hours_max"] == 8.0 and vd["budget"]["evaluation_allowance_hours"] == 2.0
    assert (vd["training"]["total_updates"], vd["training"]["warmup_updates"], vd["training"]["batch_windows"]) == (3750, 375, 8)


def test_training_code_never_refers_to_development_test_or_pediatric_data():
    banned = ("validation(", "EvalSet", "report_center_supervision", "pediatric", "evaluation/selection", "evaluation/calibration", "split_test", "'test'",
              "ordered_outputs", "continued_cls_outputs", "features/development", '["a01"]', "reports\"]")
    for mod in (tv, cv, qv, pd, vds):
        src = inspect.getsource(mod).lower()
        assert not any(b.lower() in src for b in banned), mod.__name__


def test_a_queue_restart_charges_attempts_left_running(tmp_path):
    """Review M1: an attempt that a killed queue left 'running' is charged at the next start, until its log's last write."""
    import os
    import time as _time
    ctx = SimpleNamespace(out=tmp_path, ledger=tmp_path / "ledger.jsonl"); (tmp_path / "train" / "center_only_s0").mkdir(parents=True)
    log = tmp_path / "job.log"; log.write_text("epoch 0\n"); end = _time.time(); os.utime(log, (end, end))
    started = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(end - 1800))
    status = {"jobs": {"center_only_s0": {"state": "running", "identity": "i0", "pid": 2 ** 22 + 12345, "log": str(log), "started": started},
                       "aligned_dense_s0": {"state": "pending", "identity": "i1"}}}
    qv.reconcile(ctx, status, {"center_only_s0": "i0", "aligned_dense_s0": "i1"})
    line = json.loads(ctx.ledger.read_text().splitlines()[-1])
    assert line["job"] == "train center_only_s0" and line["ok"] is False and line["hours"] == pytest.approx(0.5, abs=0.01)
    assert status["jobs"]["center_only_s0"]["state"] == "interrupted" and status["jobs"]["aligned_dense_s0"]["state"] == "pending"


def test_a_queue_restart_refuses_while_the_trainer_lives(tmp_path):
    import os
    ctx = SimpleNamespace(out=tmp_path, ledger=tmp_path / "ledger.jsonl")
    status = {"jobs": {"center_only_s0": {"state": "running", "identity": "i0", "pid": os.getpid(), "log": "", "started": "2026-10-14T00:00:00Z"}}}
    import scripts.queue_video_dense_supervision as q
    original = q.alive; q.alive = lambda pid, script: True
    try:
        with pytest.raises(SystemExit):
            q.reconcile(ctx, status, {"center_only_s0": "i0"})
    finally:
        q.alive = original
    assert not ctx.ledger.exists()


def test_a_stale_queue_lock_is_replaced(tmp_path):
    import os
    lock = tmp_path / "queue.lock"; lock.write_text(f"{2 ** 22 + 54321} 2026-10-14T00:00:00Z\n")
    qv.acquire_lock(lock)
    assert lock.read_text().split()[0] == str(os.getpid())
