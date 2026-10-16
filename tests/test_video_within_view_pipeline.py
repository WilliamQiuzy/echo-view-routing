"""A08 pipeline (configs/experiments/video_within_view_v1.yaml; task sheet sections 2-5): the configuration's pins and
inherited sections, the update step on a stand-in encoder with TargetMViT's interface (coefficient 0 against A07's
original F update, identical logits and RNG use across A, M and F, warm-up freeze, regulariser gradients after
unfreezing, hooks), the edge-mask counts, the read-only manifest check, the run identity, the budget rules, the fixed
readings on synthetic verdict tables, the comparison map, F's stored-output reproduction and the data-access guards.
CPU only; no data."""
from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

import echo_routing.vit.target_mvit as tm
import scripts.report_raw_video_context as rr
import scripts.report_video_within_view as rw
import scripts.train_video_sequence_objective as vso
import scripts.train_video_within_view as tw
from echo_routing.temporal.losses import within_view as wv

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_within_view_v1.yaml").read_text())
WV = CFG["video_within_view"]
A07 = yaml.safe_load((ROOT / "configs/experiments/video_sequence_objective_v1.yaml").read_text())["video_sequence_objective"]
O = WV["training"]["optimiser"]
POS, NEG, NONE = rw.POS, rw.NEG, rw.NONE


# ----------------------------------------------------------------------------------------------- configuration
def test_the_configuration_pins_the_task_sources_and_controls():
    src = WV["source"]
    assert WV["task_sheet"]["sha256"] == "db8bda3696e49068188381514e2c817a0cf75256db4d34824f4a12077a60ec26"
    assert src["a07_config"]["sha256"].startswith("f8d50e07") and src["a07_summary"]["sha256"].startswith("7c25e1b6")
    assert [src["manifests"][s]["sha256"][:8] for s in (0, 1, 2)] == ["0d5653cd", "d92b5a05", "2310dd5e"]
    assert [src["f_control"][s]["weights"]["sha256"][:8] for s in (0, 1, 2)] == ["6fbc987c", "958878ed", "0333fc15"]
    assert all(set(src["f_control"][s]["outputs"]) == {"calibration", "selection"} for s in (0, 1, 2))


def test_the_training_section_is_a07s_and_the_arms_queue_and_budget_follow_the_task():
    assert WV["training"] == A07["training"] and WV["arms"] == ["all", "masked"] and tw.ARMS == wv.ARMS == ("all", "masked")
    assert WV["queue"] == [tw.run_key(a, s) for s in (0, 1, 2) for a in tw.ARMS]
    assert (WV["budget"]["gpu_hours_max"], WV["budget"]["cpu_hours_max"], WV["budget"]["evaluation_reserve_hours"], WV["budget"]["overhead_reserve_hours"]) == (6.0, 3.0, 1.25, 0.25)
    assert (WV["loss"]["coefficient"], WV["loss"]["truncation"]) == (wv.COEFFICIENT, wv.TRUNCATION) == (0.15, 16.0)


# ----------------------------------------------------------------------------------------------- the update step on a stand-in encoder
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


def a08_step(arm, targets, coefficient=wv.COEFFICIENT, u=0, rng=13):
    m = stub_model(); opt = tw.make_optimizer(m, O); lrs = tw.apply_schedule(m, opt, u, 375, 3750, O); torch.manual_seed(rng)
    rec = tw.train_step(m, opt, batch(targets), arm, lrs, O, "cpu", coefficient=coefficient, keep=True)
    return m, opt, rec


def a07_step(targets, u=0, rng=13):
    m = stub_model(); opt = vso.make_optimizer(m, O); lrs = vso.apply_schedule(m, opt, u, 375, 3750, O); torch.manual_seed(rng)
    rec = vso.train_step(m, opt, batch(targets), "frame", lrs, O, "cpu", keep=True)
    return m, opt, rec


@pytest.mark.parametrize("targets", [[1, 1, 1, 1, 3, 3, 3, 3], [2, 2, 2, 2, 0, 0, 3, 3], [4, 4, 4, 4, 0, 1, 1, 0]])
@pytest.mark.parametrize("u", [0, 375])
def test_coefficient_zero_reproduces_a07s_original_f_update(targets, u):
    mf, _, rf = a07_step(targets, u=u)
    for arm in tw.ARMS:
        m, _, rec = a08_step(arm, targets, coefficient=0.0, u=u)
        assert torch.equal(rec["logits"], rf["outputs"]) and rec["loss"] == rf["loss"] and rec["grad_norm"] == rf["grad_norm"]
        assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), mf.state_dict().values()))


def test_a_m_and_f_see_the_same_logits_and_use_the_rng_identically():
    out = {}
    for arm in tw.ARMS:
        _, _, rec = a08_step(arm, [2, 2, 2, 2, 0, 0, 3, 3]); out[arm] = (rec["logits"], torch.get_rng_state(), rec)
    _, _, rf = a07_step([2, 2, 2, 2, 0, 0, 3, 3]); f_rng = torch.get_rng_state()
    assert torch.equal(out["all"][0], out["masked"][0]) and torch.equal(out["all"][0], rf["outputs"])
    assert torch.equal(out["all"][1], out["masked"][1]) and torch.equal(out["all"][1], f_rng)
    assert out["all"][2]["monitor"] == out["masked"][2]["monitor"] and out["all"][2]["loss"] != out["masked"][2]["loss"]
    assert set(out["all"][2]["monitor"]) >= {"S_A", "S_M", "same_sum", "same_edges", "diff_sum", "diff_edges", "zero_edge_groups"}


def test_pure_groups_give_the_same_update_in_both_arms():
    a, _, _ = a08_step("all", [1, 1, 1, 1, 4, 4, 4, 4]); m, _, _ = a08_step("masked", [1, 1, 1, 1, 4, 4, 4, 4])
    assert all(float((x - y).abs().max()) <= 1e-6 for x, y in zip(a.state_dict().values(), m.state_dict().values()))


def test_warmup_freezes_the_encoder_but_the_marker_and_classifier_learn():
    m0 = stub_model(); enc0 = {k: v.clone() for k, v in m0.encoder.state_dict().items()}
    m, _, rec = a08_step("masked", [2, 2, 2, 2, 0, 0, 3, 3], u=0)
    assert all(torch.equal(enc0[k], v) for k, v in m.encoder.state_dict().items()) and rec["lr_encoder"] == 0.0
    assert not torch.equal(m.marker_stem.weight, m0.marker_stem.weight) and not torch.equal(m.classifier.weight, m0.classifier.weight)


def test_the_regulariser_alone_reaches_the_encoder_projection_marker_and_classifier_after_unfreezing():
    for arm in tw.ARMS:
        m = stub_model(); m.train(); m.set_encoder_trainable(True); hooks = tw.hook_counts(m)
        g = tw.regulariser_gradients(m, batch([2, 2, 2, 2, 0, 0, 3, 3]), arm, "cpu")
        assert g["encoder_without_gradient"] == 0 and g["nonfinite"] == 0 and g["regulariser"] > 0
        assert g["projection"] > 0 and g["conv_proj"] > 0 and g["marker"] > 0 and g["classifier"] > 0
        assert tw.hook_counts(m) == hooks


def test_label_changes_never_change_the_eval_forward():
    m = stub_model().eval(); b = batch([2, 2, 2, 2, 0, 0, 3, 3])
    with torch.no_grad():
        z1 = m(tm.to_rgb(b[1]), tm.marker_planes(b[2]))
        for arm in tw.ARMS:
            wv.objective(arm, z1.view(2, 4, 5), np.roll(b[3].numpy(), 3).reshape(2, 4))
        z2 = m(tm.to_rgb(b[1]), tm.marker_planes(b[2]))
    assert torch.equal(z1, z2)


# ----------------------------------------------------------------------------------------------- masks, manifests, identity, budget
def test_the_edge_mask_counts():
    t = np.array([[1, 1, 1, 1, 0, 0, 3, 3], [2, 2, 2, 2, 0, 1, 1, 0], [4, 4, 4, 4, 0, 1, 0, 1]])
    c = tw.mask_counts(t)
    assert c["native"] == {"groups": 3, "selected_edges": 9, "unselected_edges": 0, "zero_edge_groups": 0}
    assert c["constructed"] == {"groups": 3, "selected_edges": 3, "unselected_edges": 6, "zero_edge_groups": 1}
    assert c["constructed_patterns"] == {"101": 1, "010": 1, "000": 1}


def test_the_manifest_check_compares_file_array_and_epoch_digests():
    m = {"vid": np.zeros((750, 8, 16), np.int32), "target": np.ones((750, 8), np.int8)}
    entry = {"sha256": "f" * 64, "arrays_digest": "a", "epoch_digests": ["e0", "e1"]}
    ok = tw.verify_manifest(m, "f" * 64, "f" * 64, entry, entry, arrays=lambda x: "a", epochs=lambda x: ["e0", "e1"])
    assert ok["ok"]
    for kw in ({"file_sha": "0" * 64}, {"arrays": lambda x: "b"}, {"epochs": lambda x: ["e0", "x"]}):
        args = {"file_sha": "f" * 64, "arrays": lambda x: "a", "epochs": lambda x: ["e0", "e1"]} | kw
        with pytest.raises(SystemExit):
            tw.verify_manifest(m, args["file_sha"], "f" * 64, entry, entry, arrays=args["arrays"], epochs=args["epochs"])


def fake_ctx(cfg: dict) -> SimpleNamespace:
    rv = yaml.safe_load((ROOT / "configs/experiments/raw_video_context_v1.yaml").read_text())["raw_video_context"]
    return SimpleNamespace(cfg=cfg, www=cfg["video_within_view"], a07vso=A07, rv=rv, total=3750, epoch_digests=lambda s: [f"e{s}_{i}" for i in range(10)])


def test_identity_covers_the_rules_inputs_and_code(monkeypatch):
    ctx = fake_ctx(CFG); base = tw.identity(ctx, "masked", 0, "m" * 64, "s" * 64)
    assert base == tw.identity(fake_ctx(deepcopy(CFG)), "masked", 0, "m" * 64, "s" * 64)
    changed = []
    for path, value in ((("loss", "coefficient"), 0.2), (("loss", "truncation"), 8.0), (("training", "total_updates"), 3000),
                        (("budget", "gpu_hours_max"), 9.0), (("control", "a07_arm"), "segment")):
        c = deepcopy(CFG); node = c["video_within_view"]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value; changed.append(tw.identity(fake_ctx(c), "masked", 0, "m" * 64, "s" * 64))
    changed += [tw.identity(ctx, "all", 0, "m" * 64, "s" * 64), tw.identity(ctx, "masked", 1, "m" * 64, "s" * 64),
                tw.identity(ctx, "masked", 0, "x" * 64, "s" * 64), tw.identity(ctx, "masked", 0, "m" * 64, "y" * 64)]
    monkeypatch.setattr(tw, "code_digest", lambda: "other code"); changed.append(tw.identity(ctx, "masked", 0, "m" * 64, "s" * 64))
    assert base not in changed and len(set(changed)) == len(changed)


def test_the_code_digest_covers_this_round_and_the_reused_training_files():
    names = {Path(p).name for p in tw.CODE_FILES}
    assert {"train_video_within_view.py", "within_view.py", "train_video_sequence_objective.py", "prepare_video_sequence_objective.py",
            "segment_objective.py", "target_mvit.py", "train_raw_video_context.py", "prepare_raw_video_context.py"} <= names


def test_the_budget_rules():
    b = WV["budget"]; per = {"all": 0.48, "masked": 0.49}
    launch = tw.launch_projection(0.1, per, b)
    assert launch["total"] == pytest.approx(0.1 + 3 * 0.48 + 3 * 0.49 + 1.25 + 0.25) and launch["within_cap"]
    assert not tw.launch_projection(2.0, per, b)["within_cap"]
    q = tw.queue_projection(3.0, ["masked_s1", "all_s2"], per, [0.47, 0.52], b)
    assert q["total"] == pytest.approx(3.0 + 0.52 + 0.52 + 1.25 + 0.25)


# ----------------------------------------------------------------------------------------------- the readings
CMPS = ("M - A", "M - F", "A - F", "M - frozen Ordered", "A - frozen Ordered")


def verdicts() -> dict:
    return {c: {"selection": NONE, "calibration": NONE} for c in CMPS}


def goals(fail_m: bool = False, fail_a: bool = False, met: bool = True) -> dict:
    g = lambda f: {"met": met, "constraint_failures": [{"seed": 0, "half": "selection"}] if f else [], "mean_segment_f1": {"selection": 0.95, "calibration": 0.985}}  # noqa: E731
    return {"A": g(fail_a), "M": g(fail_m), "F": g(False), "frozen Ordered": g(False)}


def both(x):
    return {"selection": x, "calibration": x}


def test_full_success_and_the_mask_specific_signal():
    v = verdicts(); v["M - A"] = v["M - F"] = v["M - frozen Ordered"] = both(POS)
    r = rw.decide(v, goals(), [], [])
    assert r["applied"][:2] == ["full_success", "mask_specific_signal"] and "no_qualifying_gain" not in r["applied"] and r["applied"][-1] == "end"
    assert "full_success" not in rw.decide(v, goals(met=False), [], [])["applied"]
    assert not {"full_success", "mask_specific_signal"} & set(rw.decide(v, goals(fail_m=True), [], [])["applied"])
    v = verdicts(); v["M - A"] = {"selection": POS, "calibration": NONE}; v["M - F"] = {"selection": POS, "calibration": NONE}
    r = rw.decide(v, goals(), [], [])
    assert "mask_specific_signal" in r["applied"] and r["notes"]["mask_specific_signal"]["halves"] == ["selection"]
    assert r["notes"]["mask_specific_signal"]["limitations"]["m_minus_frozen_ordered"] == both(NONE)
    v["M - F"] = {"selection": POS, "calibration": NEG}
    assert "mask_specific_signal" not in rw.decide(v, goals(), [], [])["applied"]


def test_the_smoothing_only_signal_and_the_release_of_harm():
    v = verdicts(); v["A - F"] = {"selection": NONE, "calibration": POS}; v["M - F"] = {"selection": NONE, "calibration": POS}
    r = rw.decide(v, goals(), [], [])
    assert "smoothing_only_signal" in r["applied"] and r["notes"]["smoothing_only_signal"]["halves"] == ["calibration"]
    assert "no_qualifying_gain" not in r["applied"] and r["by_half"]["calibration"] == ["smoothing_only_signal"]
    assert "smoothing_only_signal" not in rw.decide(v, goals(fail_a=True), [], [])["applied"]
    v2 = deepcopy(v); v2["M - A"] = {"selection": NONE, "calibration": POS}
    assert "smoothing_only_signal" not in rw.decide(v2, goals(), [], [])["applied"]          # M - A has an advantage there
    v = verdicts(); v["M - A"] = {"selection": POS, "calibration": NONE}
    r = rw.decide(v, goals(), [], [])
    assert {"release_of_harm_only", "no_qualifying_gain"} <= set(r["applied"]) and r["notes"]["release_of_harm_only"]["halves"] == ["selection"]


def test_trade_offs_and_consistent_negatives_are_listed_and_never_hidden():
    cells = [{"comparison": "M - F", "half": "selection", "gains": ["fewer multi FP segments"], "costs": ["more lost than recovered reference segments"]}]
    neg = [{"decoding": "b3", "comparison": "M - frozen Ordered", "metric": "multi segment F1@50", "half": "calibration", "verdict": NEG, "unfavourable": True}]
    r = rw.decide(verdicts(), goals(), cells, neg)
    assert r["applied"] == ["trade_off", "consistent_negatives", "no_qualifying_gain", "end"]
    assert r["notes"]["trade_off"]["cells"] == cells and r["notes"]["consistent_negatives"] == neg
    assert r["by_half"]["selection"] == ["trade_off"] and r["by_half"]["calibration"] == ["consistent_negatives"]
    assert r["notes"]["no_qualifying_gain"]["primary"] == both(NONE)


def test_unfavourable_directions_by_metric():
    assert rw.unfavourable("multi segment F1@50", NEG) and rw.unfavourable("native macro-F1", NEG) and rw.unfavourable("multi boundary recall", NEG)
    assert rw.unfavourable("multi FP segments", POS) and rw.unfavourable("multi FN segments", POS) and rw.unfavourable("native cuts per clip", POS)
    assert not rw.unfavourable("multi FP segments", NEG) and not rw.unfavourable("multi segment F1@50", POS)


def test_the_comparison_map_follows_the_config():
    ev = WV["evaluation"]["comparisons"]
    assert rw.PRIMARY == "M - A" == ev["primary"] and (ev["primary"], *ev["mandatory"]) == rw.COMPARISONS == CMPS
    assert rw.CMP["b3"]["M - F"] == ("M + B3", "F + B3") and rw.CMP["argmax"]["A - frozen Ordered"] == ("A argmax", "frozen Ordered argmax")
    assert rw.CMP["ordered_setting"]["M - frozen Ordered"] == ("M at Ordered setting", "frozen Ordered + B3")


# ----------------------------------------------------------------------------------------------- F's stored outputs and the guards
def f_npz(tmp_path, ident="fid", tamper=False):
    W = {"seq": np.array([0, 0, 1]), "centre": np.array([2, 7, 2]), "marker": np.array([8, 8, 8])}
    seqs = [{"id": "a", "labels": np.zeros(10)}, {"id": "b", "labels": np.zeros(4)}]
    lg = np.array([[2.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 3.0, 0, 0]], np.float32); orders = np.tile(np.arange(16), (3, 1))
    probs = rr.per_sample(W, lg, seqs)
    if tamper:
        probs["b"] = probs["b"] + 1e-7
    path = tmp_path / "f.npz"; rr.save_outputs(path, ident, W, orders, lg, probs, seqs, 1.0)
    return path, W, orders, seqs, lg


def test_f_outputs_are_reproduced_exactly_or_refused(tmp_path):
    path, W, orders, seqs, lg = f_npz(tmp_path)
    probs = rw.f_probabilities(path, "fid", W, orders, seqs)
    assert all(np.array_equal(probs[k], v) for k, v in rr.per_sample(W, lg, seqs).items())
    with pytest.raises(SystemExit):
        rw.f_probabilities(path, "another identity", W, orders, seqs)
    path2, *_ = f_npz(tmp_path, tamper=True)
    with pytest.raises(SystemExit):
        rw.f_probabilities(path2, "fid", W, orders, seqs)


def test_the_context_keeps_every_reader_method():
    for name in ("a03src", "proto", "plans", "half", "v01_summary", "v01_output", "a07_summary", "a07_choices", "f_output_path"):
        assert inspect.isfunction(getattr(rw.Ctx, name, None)), name


def test_eager_development_loaders_are_forbidden_during_the_report():
    import scripts.report_center_supervision as rcs
    undo = rw.forbid_eager_loaders()
    try:
        with pytest.raises(RuntimeError):
            rcs.validation(None, None, None)
    finally:
        undo()


def test_training_code_never_refers_to_development_test_or_pediatric_data():
    banned = ("validation(", "EvalSet", "report_center_supervision", "pediatric", "evaluation/selection", "evaluation/calibration", "'test'",
              "ordered_outputs", "features/development", "window_plans", "half_sets", "clip_index", "[\"outputs\"][\"", "sample_probs")
    src = (inspect.getsource(tw) + inspect.getsource(wv)).lower()
    assert not any(b.lower() in src for b in banned)


def test_the_report_never_reads_test_or_pediatric_data():
    src = inspect.getsource(rw).lower()
    assert not any(b in src for b in ("pediatric", "'test'", "split_test", "rcs.validation(", "dev_run.inputs("))
    assert "barrier(" in src and "open_selection(" in src and "devaccess({\"calibration\"})" in src
