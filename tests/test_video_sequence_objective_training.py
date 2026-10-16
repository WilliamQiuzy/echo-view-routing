"""A07 training side (configs/experiments/video_sequence_objective_v1.yaml, task sheet sections 4-5): the F/B/O equations
at T = 4 through the pinned A05 module (brute-force forward sum, X-Y-X, all-one-class value and logit-gradient identity,
the mixed-batch normalisation), the update step on a stand-in encoder with TargetMViT's interface (identical logits and
RNG use across arms, warm-up freeze with gradients still reaching the marker, order gradients reaching the encoder, the
projection, the marker and the classifier after unfreezing), optimiser binding, resume with RNG, run identity, the budget
rules, the queue order, the source parity rule and the absence of development, test or Pediatric references. CPU only."""
from __future__ import annotations

import inspect
import itertools
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import echo_routing.vit.target_mvit as tm
import scripts.train_video_sequence_objective as tr
from echo_routing.evaluate.complementarity import sha256_file
from echo_routing.temporal.losses import segment_objective as so

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_sequence_objective_v1.yaml").read_text())
VSO = CFG["video_sequence_objective"]
ONES = np.ones(5)


# ----------------------------------------------------------------------------------------------- the equations at T = 4
def logits(seed: int, n: int = 2, scale: float = 2.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 4, 5, generator=g, dtype=torch.float64) * scale).requires_grad_(True)


def enumerated_nll(logp: torch.Tensor, y) -> torch.Tensor:
    """-log of the sum over every positive-duration stay/advance alignment of y to T = 4 samples, divided by 4."""
    T, K = logp.shape[0], len(y); terms = []
    for cuts in itertools.combinations(range(1, T), K - 1):
        bounds = (0, *cuts, T); path = [y[k] for k in range(K) for _ in range(bounds[k + 1] - bounds[k])]
        terms.append(sum(logp[t, path[t]] for t in range(T)))
    return -torch.logsumexp(torch.stack(terms), 0) / T


@pytest.mark.parametrize("targets", [[2, 2, 2, 2], [0, 0, 1, 1], [2, 3, 3, 2], [0, 1, 2, 3], [4, 4, 4, 1]])
def test_the_transcript_term_is_the_brute_force_forward_sum_at_t4(targets):
    z = logits(len(targets) + sum(targets), n=1)[0]; logp = F.log_softmax(z, -1); y = so.reference_runs(np.array(targets)).transcript
    a, b = so.transcript_nll(logp, y), enumerated_nll(logp, y)
    ga, = torch.autograd.grad(a, z, retain_graph=True); gb, = torch.autograd.grad(b, z)
    assert float((a - b).abs()) < 1e-12 and float((ga - gb).abs().max()) < 1e-12


def test_the_three_objectives_match_the_task_sheet_equations():
    z = logits(1); t = np.array([[2, 2, 2, 2], [0, 0, 3, 0]]); runs = tr.group_runs(t.ravel())
    e = -F.log_softmax(z, -1).gather(-1, torch.as_tensor(t).unsqueeze(-1)).squeeze(-1)
    lf = e.mean(); lb = (e[0].mean() + e[1, :2].mean() + e[1, 2:3].mean() + e[1, 3:].mean()) / 4
    q = [so.transcript_nll(F.log_softmax(z[n], -1), r.transcript) for n, r in enumerate(runs)]
    lorder = (1 * q[0] + 3 * q[1]) / 4; lo = (lb + 0.5 * lorder) / 1.5
    for arm, want in (("frame", lf), ("segment", lb), ("order", lo)):
        got, _ = so.objective(arm, z, runs, ONES)
        assert float((got - want).abs()) < 1e-12, arm
    assert [len(r.transcript) for r in runs] == [1, 3]


def test_all_one_class_groups_give_equal_values_and_logit_gradients():
    z = logits(3); runs = tr.group_runs(np.array([1, 1, 1, 1, 4, 4, 4, 4])); vals, grads = [], []
    for arm in so.ARMS:
        loss, _ = so.objective(arm, z, runs, ONES); g, = torch.autograd.grad(loss, z); vals.append(float(loss)); grads.append(g)
    tol = float(VSO["checks"]["single_class_tolerance"])
    assert max(vals) - min(vals) <= tol and all(float((g - grads[0]).abs().max()) <= tol for g in grads)


def test_group_runs_split_the_eight_windows_into_two_groups():
    runs = tr.group_runs(np.array([0, 0, 0, 0, 2, 3, 3, 2]))
    assert [r.transcript for r in runs] == [(0,), (2, 3, 2)] and [r.length for r in runs] == [4, 4]
    with pytest.raises(ValueError):
        tr.group_runs(np.array([0, 1, 2]))


# ----------------------------------------------------------------------------------------------- the update step
class StubEncoder(nn.Module):
    """TargetMViT's encoder interface: a three-channel conv_proj (the marker is added to its output) and head[-1] = the 512-d projection."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_proj = nn.Conv3d(3, 96, (3, 7, 7), (2, 4, 4), (1, 3, 3))
        self.mix = nn.Linear(96, 96)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(96, 512))

    def forward(self, x):
        return self.head(torch.relu(self.mix(self.conv_proj(x).mean(dim=(2, 3, 4)))))


def stub_model(seed: int = 0) -> tm.TargetMViT:
    torch.manual_seed(seed); m = tm.TargetMViT(StubEncoder())
    with torch.no_grad():
        m.marker_stem.weight.normal_(0, 0.01)
    return m


def batch(targets, seed: int = 5):
    g = torch.Generator().manual_seed(seed); u8 = torch.randint(0, 255, (8, 16, 224, 224, 3), generator=g, dtype=torch.uint8)
    return (0, u8, torch.tensor([8, 9, 7, 8, 8, 8, 6, 10]), torch.tensor(targets), torch.zeros(8, 32, dtype=torch.uint8))


O = VSO["training"]["optimiser"]


def step(arm: str, targets, u: int = 0, seed_model: int = 0, rng: int = 13, keep: bool = True):
    m = stub_model(seed_model); opt = tr.make_optimizer(m, O); lrs = tr.apply_schedule(m, opt, u, 375, 3750, O); torch.manual_seed(rng)
    rec = tr.train_step(m, opt, batch(targets), arm, lrs, O, "cpu", keep=keep)
    return m, opt, rec


def test_the_three_arms_see_the_same_logits_and_use_the_rng_identically():
    out = {}
    for arm in so.ARMS:
        m, _, rec = step(arm, [2, 2, 2, 2, 0, 0, 3, 3]); out[arm] = (rec["outputs"], torch.get_rng_state(), rec)
    ref = out["frame"]
    assert all(torch.equal(ref[0], out[a][0]) and torch.equal(ref[1], out[a][1]) for a in so.ARMS)
    assert len({round(out[a][2]["loss"], 12) for a in so.ARMS}) == 3                     # a mixed batch separates the objectives
    assert all(set(out[a][2]["terms"]) >= {"frame", "segment", "order", "objective_order"} for a in so.ARMS)
    assert all(out[a][2]["terms"] == ref[2]["terms"] for a in so.ARMS)                   # the same detached monitor in every arm


def test_pure_groups_give_the_same_update_in_every_arm():
    ws = [step(a, [1, 1, 1, 1, 4, 4, 4, 4])[0] for a in so.ARMS]
    sd = [w.state_dict() for w in ws]
    assert all(float((sd[0][k] - s[k]).abs().max()) <= 1e-6 for s in sd[1:] for k in sd[0])


def test_warmup_freezes_the_encoder_but_the_marker_and_classifier_learn():
    m0 = stub_model(); enc0 = {k: v.clone() for k, v in m0.encoder.state_dict().items()}
    m, _, rec = step("order", [2, 2, 2, 2, 0, 0, 3, 3], u=0)
    assert all(torch.equal(enc0[k], v) for k, v in m.encoder.state_dict().items())
    assert not torch.equal(m.marker_stem.weight, m0.marker_stem.weight) and not torch.equal(m.classifier.weight, m0.classifier.weight)
    assert rec["lr_encoder"] == 0.0 and all(not p.requires_grad for p in m.encoder_parameters())
    m2, _, rec2 = step("order", [2, 2, 2, 2, 0, 0, 3, 3], u=375)
    assert rec2["lr_encoder"] > 0 and any(not torch.equal(enc0[k], v) for k, v in m2.encoder.state_dict().items())


def test_order_gradients_reach_the_encoder_projection_marker_and_classifier_after_unfreezing():
    m = stub_model(); m.train(); m.set_encoder_trainable(True); b = batch([2, 2, 2, 2, 0, 0, 3, 3])
    g = tr.order_gradients(m, b, "cpu")
    assert g["encoder_without_gradient"] == 0 and g["nonfinite"] == 0
    assert g["projection"] > 0 and g["marker"] > 0 and g["classifier"] > 0 and g["conv_proj"] > 0


def test_the_optimiser_binds_exactly_the_encoder_and_the_new_parameters():
    m = stub_model(); opt = tr.make_optimizer(m, O)
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in m.encoder_parameters()]
    assert [id(p) for p in opt.param_groups[1]["params"]] == [id(p) for p in m.new_parameters()]
    assert tr.optimiser_bound(opt, m)
    other = stub_model(1); assert not tr.optimiser_bound(opt, other)


def test_resume_restores_weights_optimiser_and_rng_and_needs_the_identity(tmp_path):
    m, opt, _ = step("segment", [2, 2, 2, 2, 0, 0, 3, 3]); torch.manual_seed(77); cpu = torch.get_rng_state()
    tr.save_resume(tmp_path / "r.pt", m, opt, 3, "id", [{"epoch": 0}])
    m2 = stub_model(9); opt2 = tr.make_optimizer(m2, O); torch.manual_seed(1)
    done, log = tr.load_resume(tmp_path / "r.pt", m2, opt2, "id")
    assert done == 3 and log == [{"epoch": 0}] and torch.equal(torch.get_rng_state(), cpu)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    with pytest.raises(SystemExit):
        tr.load_resume(tmp_path / "r.pt", m2, opt2, "another identity")


def test_the_source_state_loads_strictly_without_new_parameters():
    torch.manual_seed(0); base = tm.build_model(None, tm.initial_new_state(0))
    with torch.no_grad():
        base.marker_stem.weight.normal_(0, 0.01)
    state = {k: v.clone() for k, v in base.state_dict().items()}; m = tr.make_model(state, "cpu")
    assert list(m.state_dict()) == list(state) and all(torch.equal(m.state_dict()[k], v) for k, v in state.items())
    bad = dict(state); bad.pop("classifier.bias")
    with pytest.raises(RuntimeError):
        tr.make_model(bad, "cpu")


def test_the_source_parity_rule():
    a = np.array([[0.1, 2.0, -1.0, 0.0, 0.3], [1.0, 0.0, 0.0, 0.2, 0.1]])
    assert tr.parity(a, a + 5e-5)["ok"] and not tr.parity(a, a + 2e-4)["ok"]
    b = a.copy(); b[1, 0] = -1.0
    assert not tr.parity(a, b)["ok"]


# ----------------------------------------------------------------------------------------------- identity, budget, queue
def fake_ctx(cfg: dict) -> SimpleNamespace:
    rv = yaml.safe_load((ROOT / "configs/experiments/raw_video_context_v1.yaml").read_text())["raw_video_context"]
    return SimpleNamespace(cfg=cfg, vso=cfg["video_sequence_objective"], rv=rv, total=3750, epoch_digests=lambda s: [f"e{s}_{i}" for i in range(10)])


def test_identity_covers_the_rules_inputs_and_code(monkeypatch):
    ctx = fake_ctx(CFG); base = tr.identity(ctx, "order", 0, "m" * 64, "s" * 64)
    assert base == tr.identity(fake_ctx(deepcopy(CFG)), "order", 0, "m" * 64, "s" * 64)
    changed = []
    for path, value in ((("objectives", "class_weights"), [2.0, 1, 1, 1, 1]), (("groups", "rng"), "other"), (("training", "total_updates"), 3000),
                        (("budget", "gpu_hours_max"), 9.0), (("training", "optimiser", "lr_encoder"), 2e-5)):
        c = deepcopy(CFG); node = c["video_sequence_objective"]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value; changed.append(tr.identity(fake_ctx(c), "order", 0, "m" * 64, "s" * 64))
    changed += [tr.identity(ctx, "segment", 0, "m" * 64, "s" * 64), tr.identity(ctx, "order", 1, "m" * 64, "s" * 64),
                tr.identity(ctx, "order", 0, "x" * 64, "s" * 64), tr.identity(ctx, "order", 0, "m" * 64, "y" * 64)]
    monkeypatch.setattr(tr, "code_digest", lambda: "other code"); changed.append(tr.identity(ctx, "order", 0, "m" * 64, "s" * 64))
    assert base not in changed and len(set(changed)) == len(changed)


def test_code_digest_covers_the_round_and_reused_training_files():
    names = {Path(p).name for p in tr.CODE_FILES}
    assert {"train_video_sequence_objective.py", "prepare_video_sequence_objective.py", "segment_objective.py", "target_mvit.py",
            "train_raw_video_context.py", "prepare_raw_video_context.py", "prepare_video_feature_temporal.py", "video_sequence_features.py"} <= names


def test_the_budget_rules():
    b = VSO["budget"]; per = {"frame": 0.48, "segment": 0.49, "order": 0.50}
    launch = tr.launch_projection(0.2, per, b)
    assert launch["total"] == pytest.approx(0.2 + 3 * (0.48 + 0.49 + 0.50) + 1.25 + 0.25) and launch["within_cap"]
    q = tr.queue_projection(3.0, ["order_s1", "frame_s2"], per, [0.47, 0.52], b)
    assert q["total"] == pytest.approx(3.0 + 0.52 + 0.52 + 1.25 + 0.25)
    assert not tr.queue_projection(7.0, ["order_s2"], per, [], b)["within_cap"]


def test_the_ledger_keeps_kinds_and_failed_attempts(tmp_path):
    led = tmp_path / "ledger.jsonl"
    tr.ledger_append(led, "check", "gpu", "2026-10-15T00:00:00Z", 3600 * 0.2, False, "boom"); tr.ledger_append(led, "prepare", "cpu", "2026-10-15T00:00:00Z", 360, True)
    assert tr.ledger_hours(led, "gpu") == pytest.approx(0.2) and tr.ledger_hours(led, "cpu") == pytest.approx(0.1)


def test_the_queue_and_checks_follow_the_config():
    assert VSO["queue"] == [tr.run_key(a, s) for s in (0, 1, 2) for a in so.ARMS] and tr.ARMS == so.ARMS == ("frame", "segment", "order")
    assert (VSO["checks"]["reproducibility"]["params_abs"], VSO["checks"]["boundary"]["adam_step_bound"], VSO["checks"]["source_parity_max_abs"]) == (1e-5, 2.1, 1e-4)


def test_a_run_is_finished_only_with_its_identity_and_intact_weights(tmp_path):
    final = tmp_path / "final.pt"; torch.save({"w": torch.ones(2)}, final); man = tmp_path / "run_manifest.json"
    man.write_text(json.dumps({"identity": "abc", "final_checkpoint": str(final), "final_sha256": sha256_file(final)}))
    assert tr.finished(man, "abc") and not tr.finished(man, "abd")
    torch.save({"w": torch.zeros(2)}, final)
    assert not tr.finished(man, "abc")


def test_a_queue_restart_charges_attempts_left_running(tmp_path):
    import os
    import time as _time
    ctx = SimpleNamespace(out=tmp_path, ledger=tmp_path / "ledger.jsonl"); (tmp_path / "train" / "frame_s0").mkdir(parents=True)
    log = tmp_path / "job.log"; log.write_text("epoch 0\n"); end = _time.time(); os.utime(log, (end, end))
    started = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(end - 1800))
    status = {"jobs": {"frame_s0": {"state": "running", "identity": "i0", "pid": 2 ** 22 + 12345, "log": str(log), "started": started}}}
    tr.reconcile(ctx, status, {"frame_s0": "i0"})
    line = json.loads(ctx.ledger.read_text().splitlines()[-1])
    assert line["job"] == "train frame_s0" and line["kind"] == "gpu" and line["ok"] is False and line["hours"] == pytest.approx(0.5, abs=0.01)


def test_training_code_never_refers_to_development_test_or_pediatric_data():
    banned = ("validation(", "EvalSet", "report_center_supervision", "pediatric", "evaluation/selection", "evaluation/calibration", "'test'",
              "ordered_outputs", "features/development", "window_plans", "half_sets", "clip_index")
    src = inspect.getsource(tr).lower()
    assert not any(b.lower() in src for b in banned)
