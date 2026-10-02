"""Functional checks of the boundary-controlled context model (2026-10-02 plan, Task 3 "必须先通过的功能检查")."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.compose.stream_assembler import Stream
from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_checkpoint, select_operating_point
from echo_routing.temporal.dev_data import EvalSet, Seq, TrainData, collate, feature_mask_augment
from echo_routing.temporal.dev_heads import BoundaryContextHead, asrf_boundary_gt
from echo_routing.temporal.losses.boundary_context import WEAK_NEGATIVE, boundary_loss, boundary_targets, total_loss
from echo_routing.temporal.models.boundary_context import BoundaryContextConfig, BoundaryContextNet, path_bias, segment_pool

D = 16
CFG = {"in_dim": D, "dim": 32, "num_classes": 5, "heads": 4, "radius": 5}


def _net(seed: int = 0, **kw) -> BoundaryContextNet:
    torch.manual_seed(seed)
    return BoundaryContextNet(BoundaryContextConfig.from_dict({**CFG, **kw})).eval()


def _seq(n: int, seed: int, labels=None, joins=(), native=False, sid=None) -> Seq:
    rng = np.random.default_rng(seed)
    lab = np.zeros(n, np.int64) if labels is None else np.asarray(labels, np.int64)
    return Seq(sid or f"s{seed}", native, rng.normal(size=(n, D)).astype(np.float16), lab, np.asarray(joins, int))


def test_output_shapes_and_t1_is_finite():
    net = _net()
    b = collate([_seq(12, 1), _seq(1, 2)])
    out = net(b["x"], b["mask"])
    assert out["class_logits"].shape == (2, 5, 12) and out["local_logits"].shape == (2, 5, 12)
    assert out["boundary_logits"].shape == (2, 1, 12) and out["gate"].shape == (2, 1, 12)
    assert all(torch.isfinite(v).all() for v in out.values())
    single = collate([_seq(1, 3)])
    assert torch.isfinite(net(single["x"], single["mask"])["class_logits"]).all()


@pytest.mark.parametrize("use_path", [True, False])
def test_padding_does_not_change_valid_positions(use_path):
    net = _net(use_path=use_path)
    short, long_ = _seq(7, 4), _seq(30, 5)
    alone = net(*(lambda b: (b["x"], b["mask"]))(collate([short])))
    padded = net(*(lambda b: (b["x"], b["mask"]))(collate([short, long_])))
    for k in ("class_logits", "local_logits", "boundary_logits", "gate"):
        assert torch.allclose(alone[k][0, :, :7], padded[k][0, :, :7], atol=1e-5), k


def test_boundary_targets_on_a_hand_made_stream():
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    target, weight, counted = boundary_targets(labels, np.array([3, 5, 8]))   # 0|1 change, 1|1 same-class join, 1|2 change
    assert target.tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert weight.tolist() == pytest.approx([0, .1, .1, 1, .1, 1, .1, .1, 1, .1])
    assert counted.tolist() == [False] + [True] * 9
    native_t, native_w, _ = boundary_targets(np.full(6, 3), np.zeros(0, int))
    assert native_t.sum() == 0 and native_w[1:].tolist() == pytest.approx([WEAK_NEGATIVE] * 5)
    b = collate([_seq(10, 1, labels, [3, 5, 8]), _seq(4, 2, [4, 4, 4, 4], native=True)])
    assert b["b_counted"][1].tolist() == [False, True, True, True] + [False] * 6   # padding not counted
    assert b["b_target"][0].tolist() == target.tolist()


def test_weak_negatives_weigh_a_tenth_of_strong_ones():
    n = 12; logits = torch.full((1, n), 0.3)
    labels = np.zeros(n, np.int64)
    t_weak, w_weak, c_weak = boundary_targets(labels, np.zeros(0, int))          # all transitions weak negatives
    t_strong, w_strong, c_strong = boundary_targets(labels, np.arange(1, n))      # every transition a same-class join
    weak = boundary_loss(logits, torch.from_numpy(t_weak)[None], torch.from_numpy(w_weak)[None], torch.from_numpy(c_weak)[None])
    strong = boundary_loss(logits, torch.from_numpy(t_strong)[None], torch.from_numpy(w_strong)[None], torch.from_numpy(c_strong)[None])
    assert float(weak) == pytest.approx(0.1 * float(strong), rel=1e-6)


def test_path_bias_matches_the_explicit_sum():
    torch.manual_seed(0); logits = torch.randn(2, 9)
    got = path_bias(logits)
    b = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4); lg = torch.log1p(-b)
    for t in range(9):
        for j in range(9):
            lo, hi = min(t, j), max(t, j)
            assert torch.allclose(got[:, t, j], lg[:, lo + 1:hi + 1].sum(1), atol=1e-5)
    assert torch.all(torch.diagonal(got, dim1=1, dim2=2) == 0)


def test_low_boundary_probability_is_plain_attention_and_high_blocks_crossing():
    net = _net(use_path=True); attn = net.attn
    h = torch.randn(1, 20, 32); valid = torch.ones(1, 20, dtype=torch.bool)
    with torch.no_grad():
        _, plain = attn(h, valid, None)
        _, low = attn(h, valid, path_bias(torch.full((1, 20), -30.0)))
        assert torch.allclose(plain, low, atol=5e-3)
        z = torch.full((1, 20), -30.0); z[0, 10] = 30.0                    # a confident change between 9 and 10
        _, high = attn(h, valid, path_bias(z))
    cross_plain = plain[0, :, :10, 10:].sum(-1).mean(); cross_high = high[0, :, :10, 10:].sum(-1).mean()
    assert float(cross_high) < 1e-3 * float(cross_plain)


def test_predictions_ignore_recipe_metadata():
    net = _net(); a = _seq(15, 6, labels=[0] * 8 + [1] * 7, joins=[8], sid="multi_x_0001")
    b = Seq("renamed_clip_7", True, a.feat, np.full(15, 4), np.zeros(0, int))   # same features, other ids/labels/joins
    oa = net(*(lambda c: (c["x"], c["mask"]))(collate([a]))); ob = net(*(lambda c: (c["x"], c["mask"]))(collate([b])))
    assert torch.equal(oa["class_logits"], ob["class_logits"])


def test_ablation_variants_share_initial_weights():
    nets = [_net(seed=3, use_path=p, fixed_gate=g) for p, g in ((False, None), (True, None), (True, 0.5))]
    for a in nets[1:]:
        for (ka, va), (kb, vb) in zip(nets[0].state_dict().items(), a.state_dict().items()):
            assert ka == kb and torch.equal(va, vb)


def test_fixed_gate_is_constant_and_total_loss_finite():
    head = BoundaryContextHead({**CFG, "use_path": True, "fixed_gate": 0.5}, 0.2, "full_fixed")
    model = head.build()
    batch = collate([_seq(10, 1, [0] * 5 + [2] * 5, [5]), _seq(6, 2, [1] * 6, native=True)])
    out = model(batch["x"], batch["mask"])
    assert torch.allclose(out["gate"][0], torch.full((1, 10), 0.5))
    loss, parts = total_loss(out, batch, 0.2)
    assert torch.isfinite(loss) and set(parts) == {"stream", "native_frame", "local", "native_clip", "boundary"}
    loss.backward()


def test_epoch_batches_are_deterministic_and_change_by_epoch(tmp_path):
    class Store:
        def get(self, vid, var):
            from echo_routing.features.cache import VideoFeatures
            n = 40 + int(vid[1:])
            return VideoFeatures(vid, np.arange(n), np.arange(n) / 10, np.ones((n, D), np.float32), np.ones((n, 5)) / 5, np.zeros((n, 5)))
    streams = tuple(_seq(20, k, sid=f"st{k}") for k in range(8))
    data = TrainData(streams, tuple((f"v{k}", k % 5) for k in range(10)), Store())
    a = data.epoch_batches(0, 0, 8, 16, 2, 2); b = data.epoch_batches(0, 0, 8, 16, 2, 2); c = data.epoch_batches(0, 1, 8, 16, 2, 2)
    ids = lambda bb: [[s.seq_id for s in x] for x in bb]
    assert ids(a) == ids(b) and ids(a) != ids(c) and len(a) == 4
    assert all(sum(s.is_native for s in x) == 2 and all(s.labels.size <= 16 for s in x if s.is_native) for x in a)


def test_feature_mask_stays_inside_one_fragment_and_away_from_changes():
    aug = feature_mask_augment(1.0, (3, 8), 0.2, 5)
    labels = [0] * 40 + [1] * 40 + [1] * 30
    s = Seq("st", False, np.ones((110, D), np.float16), np.array(labels), np.array([40, 80]))
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = np.flatnonzero((aug(s, rng).feat == 0).all(1))
        assert 3 <= z.size <= 8 and np.all(np.diff(z) == 1)
        frag = np.searchsorted([40, 80], z[0], side="right")
        assert np.all(np.searchsorted([40, 80], z, side="right") == frag)
        assert z.size <= 0.2 * (40 if frag < 2 else 30)
        assert np.all(np.abs(z - 40) >= 5) or np.all(z >= 45)
    native = Seq("n", True, s.feat, s.labels, np.zeros(0, int))
    assert aug(native, rng) is native


def _stream(rid: str, frags) -> Stream:
    fr = tuple(Fragment(f"v{i}", lab, "orig", 0, n) for i, (lab, n) in enumerate(frags))
    labels = np.concatenate([np.full(n, lab) for lab, n in frags])
    joins = np.cumsum([n for _, n in frags])[:-1]
    sem = np.array([j for j in joins if labels[j] != labels[j - 1]], int)
    return Stream(Recipe(rid, "multi", fr), np.zeros((labels.size, 5)), np.zeros((labels.size, D)), labels, joins, sem)


def test_metrics_on_a_hand_made_half():
    multi = _stream("m1", [(0, 20), (1, 20), (1, 20), (2, 20)])          # changes at 20 and 60, same-class join at 40
    pair = _stream("p1", [(3, 15), (4, 15)])
    natives = (Seq("n1", True, np.zeros((30, D)), np.full(30, 2), np.zeros(0, int)),)
    ev = EvalSet("selection", natives, (pair,), (multi,))
    perfect = {"m1": multi.labels, "p1": pair.labels, "n1": np.full(30, 2)}
    m, _ = evaluate_predictions(perfect, ev, hz=10.0, tol_s=0.5)
    assert m["segment_f1_50_multi"] == 1.0 and m["boundary_recall_multi"] == 1.0 and m["native_internal_cuts_per_clip"] == 0
    assert m["same_class_cut_rate_multi"] == 0.0
    cut = dict(perfect); lab = multi.labels.copy(); lab[40:42] = 4; cut["m1"] = lab           # a false cut at the same-class join
    nat = np.full(30, 2); nat[10:13] = 1; cut["n1"] = nat
    m2, _ = evaluate_predictions(cut, ev, hz=10.0, tol_s=0.5)
    assert m2["same_class_cut_rate_multi"] == 1.0 and m2["native_internal_cuts_per_clip"] == 2
    assert m2["segment_f1_50_multi"] < 1.0 and m2["boundary_recall_multi"] == 1.0


def test_selection_rules_floor_ties_and_recall():
    recs = [{"epoch": 0, "native_macro_f1": 0.90, "segment_f1_50_multi": 0.95, "native_internal_cuts_per_clip": 0.1},
            {"epoch": 1, "native_macro_f1": 0.97, "segment_f1_50_multi": 0.900, "native_internal_cuts_per_clip": 0.30},
            {"epoch": 2, "native_macro_f1": 0.97, "segment_f1_50_multi": 0.8995, "native_internal_cuts_per_clip": 0.05}]
    sel = select_checkpoint(recs, floor=0.96, tie=0.002)
    assert sel["chosen"]["epoch"] == 2 and sel["status"] == "ok"       # epoch 0 below the floor; 1 and 2 tie
    assert select_checkpoint(recs, floor=0.99, tie=0.002)["status"] == "native_floor_unmet"
    pts = [dict(r, order=i, boundary_recall_multi=rc) for i, (r, rc) in enumerate(zip(recs, (0.99, 0.94, 0.96)))]
    op = select_operating_point(pts, floor=0.96, recall_min=0.95, tie=0.002)
    assert op["main"]["epoch"] == 2 and op["stability"]["epoch"] == 2 and op["status"] == "ok"
    strict = select_operating_point(pts, floor=0.99, recall_min=0.95, tie=0.002)   # floor unattainable: keep recall
    assert strict["status"] == "native_floor_unmet" and strict["main"]["epoch"] == 0
    none = select_operating_point(pts, floor=0.99, recall_min=0.999, tie=0.002)
    assert none["status"] == "both_unmet" and none["main"]["epoch"] == 0


def test_asrf_boundary_ground_truth():
    assert asrf_boundary_gt(np.array([1, 1, 2, 2, 2, 0])).tolist() == [1, 0, 1, 0, 0, 1]


OFFICIAL = Path(__file__).resolve().parents[1] / "third_party"


@pytest.mark.skipif(not (OFFICIAL / "ms-tcn" / "model.py").exists(), reason="official MS-TCN checkout is server-only")
def test_official_mstcn_head_trains_and_decodes():
    from echo_routing.temporal.dev_heads import MSTCNHead
    head = MSTCNHead(OFFICIAL / "ms-tcn", in_dim=D); model = head.build()
    batch = collate([_seq(12, 1, [0] * 6 + [1] * 6, [6]), _seq(5, 2, [3] * 5, native=True)])
    loss, _ = head.loss(model, batch); assert torch.isfinite(loss); loss.backward()
    out = head.decode(model, [_seq(12, 1), _seq(5, 2)], "cpu")
    assert out["s1"]["labels"].shape == (12,) and out["s2"]["labels"].shape == (5,)


@pytest.mark.skipif(not (OFFICIAL / "asrf" / "libs").exists(), reason="official ASRF checkout is server-only")
def test_official_asrf_head_trains_and_decodes():
    from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights
    seqs = [_seq(12, 1, [0] * 6 + [1] * 6, [6]), _seq(8, 2, [3] * 8, native=True), _seq(9, 3, [2] * 4 + [4] * 5, [4])]
    cw, pw = asrf_weights(seqs, 5)
    head = ASRFHead(OFFICIAL / "asrf", D, cw, pw); model = head.build()
    model.train(); loss, parts = head.loss(model, collate(seqs)); assert torch.isfinite(loss); loss.backward()
    out = head.decode(model, seqs, "cpu")
    assert all(out[s.seq_id]["labels"].shape == (s.labels.size,) for s in seqs)


def test_segment_pool_averages_within_predicted_segments():
    vals = torch.randn(1, 12, 5); valid = torch.ones(1, 12, dtype=torch.bool)
    flat = segment_pool(vals, path_bias(torch.full((1, 12), -30.0)), valid)
    assert torch.allclose(flat[0], vals[0].mean(0, keepdim=True).expand(12, 5), atol=1e-3)
    z = torch.full((1, 12), -30.0); z[0, 5] = 30.0
    split = segment_pool(vals, path_bias(z), valid)
    assert torch.allclose(split[0, :5], vals[0, :5].mean(0, keepdim=True).expand(5, 5), atol=1e-3)
    assert torch.allclose(split[0, 5:], vals[0, 5:].mean(0, keepdim=True).expand(7, 5), atol=1e-3)
    short = segment_pool(vals[:, :7], path_bias(z[:, :7]), valid[:, :7])
    padded = segment_pool(torch.cat([vals[:, :7], torch.zeros(1, 5, 5)], 1), path_bias(z), torch.arange(12)[None] < 7)
    assert torch.allclose(short[0], padded[0, :7], atol=1e-5)


@pytest.mark.parametrize("use_path", [True, False])
def test_segment_pool_model_is_padding_invariant(use_path):
    net = _net(use_path=use_path, segment_pool=True)
    short, long_ = _seq(7, 4), _seq(30, 5)
    alone = net(*(lambda b: (b["x"], b["mask"]))(collate([short])))
    padded = net(*(lambda b: (b["x"], b["mask"]))(collate([short, long_])))
    assert torch.allclose(alone["class_logits"][0, :, :7], padded["class_logits"][0, :, :7], atol=1e-5)
