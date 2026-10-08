"""Checks of the fold-output correction round on synthetic data: the cyclic seen / unseen generators, the rejection of
non-training clips and of streams that cross folds, the native windows (the trainer's rule, read in the fixed variant),
the pairing of the arms, P-out's identity start and per-frame property, the resumed optimizer binding and, where the
pinned checkout exists, the five-channel ASRF."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.prepare_fold_output_correction as pf
import scripts.train_fold_output_correction as tf
from echo_routing.compose.recipe_builder import Fragment, Recipe, recipe_to_dict
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.dev_data import PAD_LABEL, TrainData, collate, native_seq

ROOT = Path(__file__).resolve().parents[1]


def test_generators_follow_the_cyclic_rule():
    assert [pf.generator("T-in", k) for k in range(3)] == [1, 2, 0] and [pf.generator(a, 2) for a in ("T-out", "P-out")] == [2, 2]
    keys = {0: "k0", 1: "k1", 2: "k2"}
    assert pf.arm_cache_key("T-in", 0, keys) == "k1-seen-fold0" and pf.arm_cache_key("T-out", 0, keys) == "k0" and pf.arm_cache_key("P-out", 1, keys) == "k1"
    folds = {"a": 0, "b": 1, "c": 2}; train = {k: {s for s, f in folds.items() if f != k} for k in range(3)}
    for arm in pf.ARMS:
        pf.check_generator(arm, "a", 0, train)                              # T-in reads model 1 (trained on a), the others model 0
    with pytest.raises(SystemExit):
        pf.check_generator("T-in", "a", 0, {0: set(), 1: set(), 2: set()})     # a T-in generator that never saw the session
    with pytest.raises(SystemExit):
        pf.check_generator("T-out", "a", 0, {0: {"a"}, 1: set(), 2: set()})    # a T-out generator that did


def test_non_training_clips_and_cross_fold_streams_are_rejected():
    with pytest.raises(SystemExit):
        pf.check_training_clips({"tr_1", "val_1"}, {"tr_1"})
    r = Recipe(recipe_id="r1", cell="cell", fragments=(Fragment("a_1", 0, "orig", 0, 10), Fragment("b_1", 1, "gamma090", 5, 20)))
    with pytest.raises(SystemExit):
        pf.stream_item(recipe_to_dict(r), 0, {"a_1": 0, "b_1": 1})
    it = pf.stream_item(recipe_to_dict(r), 0, {"a_1": 0, "b_1": 0})
    assert it.sources == (("a_1", "orig", 0, 10), ("b_1", "gamma090", 5, 20)) and it.joins.tolist() == [10] and it.labels.tolist() == [0] * 10 + [1] * 15


def _vf(vid: str, n: int, value: float) -> VideoFeatures:
    prob = np.full((n, 5), value / 5, np.float32); prob[:, 0] += 1 - value
    return VideoFeatures(vid, np.arange(n), np.arange(n) / 10, np.zeros((n, 2), np.float16), prob, np.zeros((n, 5), np.float32))


def test_native_windows_follow_the_trainer_rule_and_read_the_fixed_variant():
    clips = [(f"c{k}_1", k % 5) for k in range(9)]; n = {v: 40 + 37 * k for k, (v, _) in enumerate(clips)}
    recipes = [Recipe(recipe_id=f"r{k}", cell="cell", fragments=(Fragment("c0_1", 0, "orig", 0, 10), Fragment("c1_1", 1, "orig", 0, 12))) for k in range(6)]
    streams = [pf.stream_item(recipe_to_dict(r), 0, {v: 0 for v, _ in clips}) for r in recipes]

    class Store:
        def get(self, vid, variant):
            return _vf(vid, n[vid], 0.5)
    seqs = tuple(native_seq(r.recipe_id, 0, _vf(r.recipe_id, 22, 0.5)) for r in recipes)
    want = TrainData(seqs, tuple(clips), Store()).epoch_batches(7, 3, 5, 64, 2, 2)
    variant = {v: ("gamma090" if k % 2 else "orig") for k, (v, _) in enumerate(clips)}
    got = pf.epoch_items(7, 3, streams, clips, n, variant, {v: 0 for v, _ in clips}, 5, 64, 2, 2)
    assert [[s.seq_id for s in b] for b in want] == [[it.seq_id for it in b] for b in got]
    native = next(it for b in got for it in b if it.kind == "native" and it.sources[0][1] == "gamma090")
    seq = pf.materialize(native, lambda vid, var: _vf(vid, n[vid], 0.9 if var == "gamma090" else 0.1).prob)
    a, b = native.sources[0][2], native.sources[0][3]
    assert np.allclose(seq.feat, _vf(native.sources[0][0], n[native.sources[0][0]], 0.9).prob[a:b])          # the gamma version is read
    assert seq.labels.size == min(64, n[native.sources[0][0]]) and seq.is_native


def test_arms_differ_only_in_the_probabilities():
    r = Recipe(recipe_id="r1", cell="cell", fragments=(Fragment("a_1", 2, "orig", 0, 6), Fragment("a_2", 3, "gamma090", 2, 7)))
    it = pf.stream_item(recipe_to_dict(r), 1, {"a_1": 1, "a_2": 1})
    s_in = pf.materialize(it, lambda v, var: _vf(v, 10, 0.2).prob); s_out = pf.materialize(it, lambda v, var: _vf(v, 10, 0.8).prob)
    assert np.array_equal(s_in.labels, s_out.labels) and np.array_equal(s_in.joins, s_out.joins) and not np.array_equal(s_in.feat, s_out.feat)
    assert pf.batch_digest([it]) == pf.batch_digest([pf.stream_item(recipe_to_dict(r), 1, {"a_1": 1, "a_2": 1})])
    with pytest.raises(SystemExit):
        pf.materialize(it, lambda v, var: np.zeros((10, 4), np.float32))


def test_p_out_starts_at_the_identity_and_reads_no_neighbour():
    head = tf.CalibratorHead(torch.ones(5), 1e-6); model = head.build()
    p = np.random.default_rng(0).dirichlet(np.ones(5), size=12).astype(np.float32); p[3] = [1, 0, 0, 0, 0]
    q = head.probs(model, p, "cpu"); c = np.clip(p, 1e-6, None)
    assert np.abs(q - c / c.sum(1, keepdims=True)).max() < 1e-6 and sum(x.numel() for x in model.parameters()) == 30
    alt = p.copy(); alt[5] = alt[5][::-1]; q2 = head.probs(model, alt, "cpu"); keep = np.arange(12) != 5
    assert np.array_equal(q2[keep], q[keep])
    seqs = [pf.Seq("a", True, p, np.full(12, 2), np.zeros(0, int)), pf.Seq("b", True, p[:7], np.full(7, 3), np.zeros(0, int))]
    batch = collate(seqs); l1, _ = head.loss(model, batch); alt_b = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = PAD_LABEL
    l2, _ = head.loss(model, alt_b)
    assert torch.isfinite(l1) and float(l1) == float(l2)


def test_resumed_temporal_optimizer_is_bound(tmp_path):
    tr = {"lr": 3e-4, "weight_decay": 1e-4, "epochs": 2, "updates_per_epoch": 3, "warmup_epochs": 1}
    head = tf.CalibratorHead(torch.ones(5), 1e-6); model = head.build(); opt, sched = tf.make_optimiser(model, tr)
    p = np.random.default_rng(1).dirichlet(np.ones(5), size=9).astype(np.float32)
    batch_seqs = [pf.Seq("a", True, p, np.array([0, 1, 2, 3, 4, 0, 1, 2, 3]), np.zeros(0, int))]
    tf.update(head, model, batch_seqs, opt, sched, 1.0, "cpu"); tf.save_resume(tmp_path / "resume.pt", model, opt, sched, 0, "id", [])
    model2 = head.build(); opt2, sched2 = tf.make_optimiser(model2, tr)
    start, _ = tf.load_resume(tmp_path / "resume.pt", model2, opt2, sched2, "id")
    assert start == 1 and tf.optimizer_bound(model2, opt2) and torch.equal(model2.W, model.W)
    before = model2.W.detach().clone(); tf.update(head, model2, batch_seqs, opt2, sched2, 1.0, "cpu")
    assert not torch.equal(before, model2.W.detach())
    with pytest.raises(SystemExit):
        tf.load_resume(tmp_path / "resume.pt", head.build(), *tf.make_optimiser(head.build(), tr), "another")


def test_queue_order_and_results_directory():
    fo = {"training": {"seeds": [0, 1, 2]}, "results_dir": "docs/results/2026-10-08_fold_output_correction"}
    assert tf.queue_order(fo)[:4] == [("T-in", 0), ("T-out", 0), ("P-out", 0), ("T-in", 1)] and len(tf.queue_order(fo)) == 9
    pf.check_out("docs/results/2026-10-08_fold_output_correction", fo)
    with pytest.raises(SystemExit):
        pf.check_out("docs/results/2026-10-07_session_fold_library", fo)


@pytest.mark.skipif(not (ROOT / "third_party/asrf/.git").exists(), reason="the pinned ASRF checkout exists on the server only")
def test_asrf_takes_five_channel_probabilities():
    from echo_routing.temporal.dev_heads import ASRFHead
    head = ASRFHead(ROOT / "third_party/asrf", 5, torch.ones(5), torch.tensor(3.0)); model = head.build(); model.train()
    rng = np.random.default_rng(2)
    seqs = [pf.Seq("a", False, rng.dirichlet(np.ones(5), size=30).astype(np.float32), np.array([0] * 12 + [2] * 18), np.array([12])),
            pf.Seq("b", True, rng.dirichlet(np.ones(5), size=17).astype(np.float32), np.full(17, 3), np.zeros(0, int))]
    batch = collate(seqs); assert tuple(batch["x"].shape) == (2, 5, 30)
    loss, _ = head.loss(model, batch); loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for g in grads) and any(float(g.abs().max()) > 0 for g in grads)


def test_prob_store_reads_only_the_probabilities(tmp_path):
    d = tmp_path / "features" / "key" / "gamma090"; d.mkdir(parents=True)
    np.savez_compressed(d / "v_1.npz", frame_idx=np.arange(4), t_sec=np.arange(4) / 10, feat=np.ones((4, 3), np.float16),
                        prob=np.eye(5, dtype=np.float32)[[0, 1, 2, 3]], logit=np.zeros((4, 5), np.float32))
    p = pf.ProbStore(tmp_path, "key").get("v_1", "gamma090")
    assert p.dtype == np.float32 and p.shape == (4, 5) and p[2, 2] == 1.0


def test_t_in_and_t_out_share_one_initial_state(tmp_path):
    class Head:
        def __init__(self, shift=0.0):
            self.shift = shift

        def build(self):
            m = torch.nn.Linear(3, 2)
            with torch.no_grad():
                m.weight += self.shift
            return m
    a = tf.save_or_check_init(Head(), "T-in", 0, tmp_path); b = tf.save_or_check_init(Head(), "T-out", 0, tmp_path)
    assert a == b and a[0].name == "asrf_init_s0.pt"
    with pytest.raises(SystemExit):
        tf.save_or_check_init(Head(1.0), "T-out", 0, tmp_path)                 # another state under the seed's file


def test_architecture_guard_and_finished_runs(tmp_path):
    fo = {"asrf": {"model": {**tf.ASRF_BUILD, "in_channel": 512}, "repo": "third_party/asrf"}}
    with pytest.raises(SystemExit):
        tf.make_head(fo, tmp_path, "T-out", {"class_weight": [1.0] * 5, "pos_weight": 1.0}, "cpu")
    man = tmp_path / "run_manifest.json"; assert tf.finished_run(man, "x") is None
    ck = tmp_path / "final.pt"; torch.save({"w": torch.ones(1)}, ck)
    man.write_text(json.dumps({"identity": "x", "final_checkpoint": str(ck), "final_sha256": tf.sha256_file(ck)}))
    assert tf.finished_run(man, "x") is True and tf.finished_run(man, "y") is False
    torch.save({"w": torch.zeros(1)}, ck); assert tf.finished_run(man, "x") is False                   # the checkpoint changed


def test_a_same_class_join_is_not_a_boundary():
    from echo_routing.temporal.dev_heads import ASRFHead
    labels = np.array([2, 2, 2, 2, 2, 3, 3, 0]); seqs = [pf.Seq("s", False, np.full((8, 5), 0.2, np.float32), labels, np.array([3, 5])),   # a same-class join at 3
                                                          pf.Seq("n", True, np.full((5, 5), 0.2, np.float32), np.full(5, 1), np.zeros(0, int))]
    b = ASRFHead._boundary(collate(seqs))[:, 0].numpy()
    assert b[0].tolist() == [1, 0, 0, 0, 0, 1, 0, 1] and b[1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]          # padding of the second row stays 0


def test_repo_records_refuse_ids_and_cache_digests_follow_content(tmp_path):
    ids = {"sessA_1", "sessA"}
    pf.assert_no_ids({"counts": {"clips": 3}, "rule": "seen / unseen"}, ids)
    for bad in ({"sessA_1": 1}, {"x": ["a", "sessA_1@12"]}, {"y": "session:sessA"}):
        with pytest.raises(SystemExit):
            pf.assert_no_ids(bad, ids)
    for variant in pf.VARIANTS:
        d = tmp_path / "features" / "k" / variant; d.mkdir(parents=True); (d / "index.csv").write_text("video_id\nv_1\n")
        np.savez_compressed(d / "v_1.npz", prob=np.full((2, 5), 0.2, np.float32))
    first = pf.cache_digest(tmp_path, "k", ["v_1"])
    np.savez_compressed(tmp_path / "features" / "k" / "orig" / "v_1.npz", prob=np.full((2, 5), 0.1, np.float32))
    assert pf.cache_digest(tmp_path, "k", ["v_1"]) != first


def _fo():
    from echo_routing.config import load_config
    return load_config(ROOT / "configs/experiments/fold_output_correction_v1.yaml")["fold_output_correction"]


def test_consistency_needs_every_seed():
    import scripts.report_fold_output_correction as rf
    up = [{"ci95": [0.01, 0.05]}] * 3; down = [{"ci95": [-0.05, -0.01]}] * 3
    assert rf.consistency(up) == "consistent positive" and rf.consistency(down) == "consistent negative"
    assert rf.consistency(up[:2] + [{"ci95": [-0.01, 0.03]}]) == "no consistent difference detected"


def test_decision_table_order():
    import scripts.report_fold_output_correction as rf
    fo = _fo(); pos = {"selection": "consistent positive", "calibration": "consistent positive"}
    good = {"segment_f1_50_multi": 0.99, "boundary_recall_multi": 0.97, "native_macro_f1": 0.99}
    rows = [{"selection": good, "calibration": good}] * 3; sec = {"a": pos, "b": pos}
    assert rf.decide({"selection": "consistent positive", "calibration": "no consistent difference detected"}, sec, rows, fo)["action"].startswith("the round ends; no automatic")
    assert "no new-model claim" in rf.decide(pos, {"a": pos, "b": {"selection": "consistent positive", "calibration": "consistent negative"}}, rows, fo)["action"]
    low = {**good, "boundary_recall_multi": 0.90}
    assert "failed constraint" in rf.decide(pos, sec, [{"selection": low, "calibration": good}] + rows[1:], fo)["action"]
    mid = {**good, "segment_f1_50_multi": 0.95}
    assert rf.decide(pos, sec, [{"selection": mid, "calibration": mid}] * 3, fo)["action"].startswith("the lead decides")
    assert rf.decide(pos, sec, rows, fo)["goal_met"] is True


def test_fixed_b3_and_found_changes():
    import scripts.report_fold_output_correction as rf
    p = np.full((40, 5), 0.01, np.float32); p[:20, 2] = 0.96; p[20:, 3] = 0.96
    lab = rf.fixed_b3(p, _fo()["evaluation"]["fixed_b3"])
    assert lab[:18].tolist() == [2] * 18 and lab[-18:].tolist() == [3] * 18 and np.count_nonzero(np.diff(lab)) == 1
    a_only, b_only = rf.change_sets(np.array([10, 30]), np.array([11]), np.array([29]), 2.0)
    assert a_only == {10} and b_only == {30}
