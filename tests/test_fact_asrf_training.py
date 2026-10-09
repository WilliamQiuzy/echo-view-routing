"""Checks of the paired FACT-ASRF training entry on synthetic data: each arm's optimiser and schedule, FACT's decay and a
resume across it, the final-weights rule, the optimiser binding after a resume, the RNG round trip, the arm-free batch
plan, and the padding checks (FACT where its checkout exists, ASRF on the server)."""
import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import scripts.train_fact_asrf as tr
from echo_routing.temporal.dev_data import Seq, collate
from echo_routing.temporal.dev_train import seed_everything

ROOT = Path(__file__).resolve().parents[1]
FA = yaml.safe_load((ROOT / "configs/experiments/fact_asrf_paired_v1.yaml").read_text())["fact_asrf_paired"]
TOTAL = int(FA["data"]["total_updates"])


def tiny() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2)


def test_each_arm_has_its_own_optimiser():
    opt, _, clip = tr.make_optimiser(FA["arms"]["FACT"], tiny(), TOTAL)
    assert type(opt) is torch.optim.Adam and clip == 10.0
    assert opt.defaults["lr"] == 1e-4 and opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8 and opt.defaults["weight_decay"] == 0.0
    opt, _, clip = tr.make_optimiser(FA["arms"]["ASRF"], tiny(), TOTAL)
    assert type(opt) is torch.optim.AdamW and clip == 1.0 and opt.defaults["lr"] == 3e-4 and opt.defaults["weight_decay"] == 1e-4
    broken = {**FA["arms"]["FACT"], "optimizer": {k: v for k, v in FA["arms"]["FACT"]["optimizer"].items() if k != "eps"}}
    with pytest.raises(KeyError):
        tr.make_optimiser(broken, tiny(), TOTAL)


def lrs(opt, sched, steps: int) -> list[float]:
    out = []
    for _ in range(steps):
        out.append(float(opt.param_groups[0]["lr"])); opt.step(); sched.step()
    return out


def test_fact_rate_drops_after_11719_updates_and_survives_a_resume_there():
    a = FA["arms"]["FACT"]; at = int(a["schedule"]["decay_at_completed_updates"])
    assert at == 11719
    opt, sched, _ = tr.make_optimiser(a, tiny(), TOTAL); seq = lrs(opt, sched, at + 5)
    assert set(seq[:at]) == {1e-4} and seq[at:] == pytest.approx([1e-5] * 5)
    for done in (at - 1, at, at + 1):   # resume after done updates, through the state dicts as in a run
        o1, s1, _ = tr.make_optimiser(a, tiny(), TOTAL); lrs(o1, s1, done)
        o2, s2, _ = tr.make_optimiser(a, tiny(), TOTAL); o2.load_state_dict(o1.state_dict()); s2.load_state_dict(s1.state_dict())
        assert lrs(o2, s2, 3) == pytest.approx(seq[done:done + 3])


def test_asrf_rate_warms_up_then_follows_the_cosine():
    opt, sched, _ = tr.make_optimiser(FA["arms"]["ASRF"], tiny(), TOTAL); seq = lrs(opt, sched, 760)
    assert seq[0] == pytest.approx(3e-4 / 750) and seq[749] == pytest.approx(3e-4) and seq[750] == pytest.approx(3e-4)
    assert seq[759] < 3e-4


def test_final_weights_rule_reads_only_the_update_count():
    recs = [{"updates": 375 * (e + 1), "monitor": {"segment_f1_50": 1.0 - e / 100}, "native_macro_f1": 0.99} for e in range(50)]
    assert tr.final_checkpoint(recs, TOTAL) == {"updates": TOTAL, "rule": "final weights"}
    with pytest.raises(SystemExit):
        tr.final_checkpoint(recs[:-1], TOTAL)
    assert not hasattr(tr, "eval_set") and not hasattr(tr, "EvalSet")   # the training entry cannot build a development half


def test_resume_binds_the_optimiser_to_the_loaded_model(tmp_path):
    a = FA["arms"]["FACT"]; m1 = tiny(); o1, s1, _ = tr.make_optimiser(a, m1, TOTAL)
    m1(torch.ones(4, 3)).sum().backward(); o1.step(); s1.step()
    tr.save_resume(tmp_path / "r.pt", m1, o1, s1, 1, "ident", [{"updates": 1}])
    m2 = torch.nn.Linear(3, 2); o2, s2, _ = tr.make_optimiser(a, m2, TOTAL)
    done, recs = tr.load_resume(tmp_path / "r.pt", m2, o2, s2, "ident")
    assert done == 1 and recs == [{"updates": 1}] and tr.optimiser_bound(o2, m2) and not tr.optimiser_bound(o2, m1)
    before = m2.weight.detach().clone(); m2(torch.ones(4, 3)).sum().backward(); o2.step()
    assert not torch.equal(before, m2.weight.detach())
    with pytest.raises(SystemExit):
        tr.load_resume(tmp_path / "r.pt", m2, o2, s2, "another")


def test_rng_round_trip_covers_python_numpy_and_torch():
    seed_everything(3); st = tr.rng_state(); first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    tr.set_rng_state(st); assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == first


def seq(labels, seed: int, d: int = 4) -> Seq:
    rng = np.random.default_rng(seed); labels = np.asarray(labels, np.int64)
    return Seq(f"s{seed}", False, rng.standard_normal((labels.size, d)).astype(np.float16), labels, np.zeros(0, int))


def test_one_plan_per_seed_for_both_arms():
    assert "arm" not in inspect.signature(tr.make_plan).parameters
    batches = {e: [[seq([0] * 5 + [1] * 3, 10 * e + i), seq([2] * 4, 10 * e + i + 5)] for i in range(2)] for e in range(2)}
    ctx = SimpleNamespace(tr={"epochs": 2, "updates_per_epoch": 2}, batches=lambda seed, e: batches[e])
    p1, checked = tr.make_plan(ctx, 0, check_unpad=True); p2, _ = tr.make_plan(ctx, 0)
    assert p1 == p2 and checked == 4 and len(p1["epochs"]) == 2 and len(p1["asrf_class_weight"]) == 5


@pytest.mark.skipif(not (ROOT / "third_party/fact/.git").exists(), reason="the pinned FACT checkout")
def test_fact_padding_labels_have_no_effect_and_the_unpadded_inputs_match():
    from echo_routing.temporal.fact_head import FACTHead, TokenRule
    head = FACTHead(ROOT / "third_party/fact", 4, TokenRule(8, 2.654333333333333, 1.5, 12), base_config="gtea.yaml")
    seed_everything(0); model = head.build(); b = [seq([0] * 9 + [1] * 6, 1), seq([3] * 5, 2)]; batch = collate(b)
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    seed_everything(1); l1, _ = head.loss(model, batch); seed_everything(1); l2, _ = head.loss(model, alt)
    assert abs(float(l1) - float(l2)) <= 1e-6
    xs, ys = FACTHead.unpad(batch)
    assert all(np.array_equal(x.numpy(), s.feat.astype(np.float32)) and np.array_equal(y.numpy(), s.labels) for x, y, s in zip(xs, ys, b))


@pytest.mark.skipif(not (ROOT / "third_party/asrf/.git").exists(), reason="the pinned ASRF checkout exists on the server only")
def test_asrf_padding_labels_have_no_effect():
    from echo_routing.temporal.dev_heads import ASRFHead
    head = ASRFHead(ROOT / "third_party/asrf", 4, torch.ones(5), torch.tensor(5.0))
    seed_everything(0); model = head.build(); batch = collate([seq([0] * 9 + [1] * 6, 1), seq([3] * 5, 2)])
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    seed_everything(1); l1, _ = head.loss(model, batch); seed_everything(1); l2, _ = head.loss(model, alt)
    assert abs(float(l1) - float(l2)) <= 1e-6


def test_smoke_stamp_tracks_config_plan_and_initial_state(tmp_path):
    fa = {**FA, "outputs": {**FA["outputs"], "server_dir": str(tmp_path)}}
    (tmp_path / "plans").mkdir(); (tmp_path / "inits").mkdir()
    (tmp_path / "plans" / "plan_s0.json").write_text("{}"); (tmp_path / "inits" / "FACT_s0.pt").write_bytes(b"x")
    stamp = tr.smoke_stamp(fa, ROOT, "FACT", 0)
    assert stamp == tr.smoke_stamp(fa, ROOT, "FACT", 0)
    other = {**fa, "arms": {**fa["arms"], "FACT": {**fa["arms"]["FACT"], "optimizer": {**fa["arms"]["FACT"]["optimizer"], "lr": 2e-4}}}}
    assert tr.smoke_stamp(other, ROOT, "FACT", 0) != stamp
    (tmp_path / "plans" / "plan_s0.json").write_text('{"x": 1}')
    assert tr.smoke_stamp(fa, ROOT, "FACT", 0) != stamp
