"""Checks of the predicted-partition FACT (echo_routing/temporal/predicted_partition.py) on synthetic data: the boundary
head's inputs, targets, loss and cuts (position 0 and padding never scored, same-view joins never positive, inputs
aligned with positions, NMS and repeated cuts, length 1); the half-open partitions (grid tail shorter than 8, length 1,
closed-segment conversion, cuts on existing boundaries); and, on the pinned FACT checkout on CPU, that the adapter
equals the official forward, loss and gradients when it adds no cut, equals the localisation round's implementation
for one interval list, keeps gradients, ignores the labels passed to it, and refines native-like sequences by the
same rule."""
from pathlib import Path

import numpy as np
import pytest
import torch

import echo_routing.temporal.predicted_partition as pp
from echo_routing.temporal.dev_data import Seq, collate
from echo_routing.temporal.dev_train import seed_everything

ROOT = Path(__file__).resolve().parents[1]
HAS_FACT = (ROOT / "third_party/fact/.git").exists()


# ----------------------------------------------------------------------------------------------- boundary head
def test_head_inputs_are_aligned_with_positions():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    f = pp.BoundaryHead.features(x)
    assert f.shape == (4, 9)
    assert torch.equal(f[0], torch.cat([x[0], x[0], torch.zeros(3)]))            # x_(-1) = x_0
    assert torch.equal(f[2], torch.cat([x[1], x[2], x[2] - x[1]]))
    torch.manual_seed(0); head = pp.BoundaryHead(3, 8)
    assert head(x).shape == (4,) and head(x[:1]).shape == (1,)


def test_targets_never_score_position_0_or_same_view_joins():
    assert pp.head_targets([2, 2, 3, 3, 3, 2]).tolist() == [0, 0, 1, 0, 0, 1]
    assert pp.head_targets([1, 1, 1, 1]).tolist() == [0, 0, 0, 0]                # a same-view join or a native window
    assert pp.head_targets([4]).tolist() == [0]
    assert pp.boundary_counts([[2, 2, 3, 3], [1, 1, 1]]) == (1, 4)
    assert pp.head_pos_weight([[2, 2, 3, 3], [1, 1, 1]]) == 4.0
    with pytest.raises(ValueError):
        pp.head_pos_weight([[1, 1, 1]])


def test_loss_ignores_position_0_padding_and_one_sample_sequences():
    torch.manual_seed(0); head = pp.BoundaryHead(3, 8); w = torch.tensor(2.0)
    x1, y1 = torch.randn(6, 3), torch.tensor([0, 0, 1, 1, 1, 0]); x2, y2 = torch.randn(4, 3), torch.tensor([2, 2, 2, 2])
    loss = pp.head_loss(head, [x1, x2, torch.randn(1, 3)], [y1, y2, torch.tensor([3])], w)
    per = [torch.nn.functional.binary_cross_entropy_with_logits(head(x)[1:], (y[1:] != y[:-1]).float(), pos_weight=w) for x, y in ((x1, y1), (x2, y2))]
    assert torch.allclose(loss, (per[0] + per[1]) / 2)                           # mean within a sequence, then over sequences
    batch = collate([Seq("a", False, x1.numpy().astype(np.float16), y1.numpy(), np.zeros(0, int)),
                     Seq("b", True, x2.numpy().astype(np.float16), y2.numpy(), np.zeros(0, int))])
    xs = [batch["x"][i, :, :n].T for i, n in enumerate(batch["lengths"].tolist())]   # unpadded: the padding never enters
    ys = [batch["labels"][i, :n] for i, n in enumerate(batch["lengths"].tolist())]
    assert all(x.shape[0] == n for x, n in zip(xs, batch["lengths"].tolist()))
    pp.head_loss(head, xs, ys, w)
    with pytest.raises(ValueError):
        pp.head_loss(head, [torch.randn(1, 3)], [torch.tensor([0])], w)


class Fixed(torch.nn.Module):
    """A stand-in head with given logits."""

    def __init__(self, logits) -> None:
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self.logits


def test_cuts_use_threshold_nms_and_never_position_0():
    assert pp.head_cuts(Fixed([9.0, -1.0, 3.0, 2.5, -2.0, 0.5, -3.0, 4.0]), torch.zeros(8, 1)) == [2, 7]   # 3 and 5 suppressed (radius 2)
    assert pp.head_cuts(Fixed([9.0, 9.0, -1.0]), torch.zeros(3, 1)) == [1]      # position 0 suppresses nothing
    assert pp.head_cuts(Fixed([1.0]), torch.zeros(1, 1)) == []                  # length 1
    assert pp.head_cuts(Fixed([0.0, -0.01, 0.0, 0.0]), torch.zeros(4, 1)) == [2]   # sigmoid 0.5 counts; ties keep the earlier


# ----------------------------------------------------------------------------------------------- partitions
def test_grid_tail_length_one_and_repeated_cuts():
    assert pp.grid_cuts([(0, 20), (20, 28), (28, 37), (37, 38)]) == [8, 16, 36]   # [20, 28) has 8 samples: no cut
    assert pp.refine([(0, 20), (20, 28)], "uniform") == [(0, 8), (8, 16), (16, 20), (20, 28)]
    assert pp.add_cuts([(0, 5), (5, 10)], [5, 7, 7, 0, 10, 12]) == [(0, 5), (5, 7), (7, 10)]   # boundary, repeats, outside ignored
    assert pp.refine([(0, 1), (1, 4)], "predicted", [1, 2]) == [(0, 1), (1, 2), (2, 4)]
    assert pp.refine([(0, 10)], "predicted_even", [3, 4]) == [(0, 3), (3, 6), (6, 10)]          # k = 2 cuts at 0 + floor(j 10 / 3)
    assert pp.refine([(0, 10)], "original") == [(0, 10)]
    with pytest.raises(ValueError):
        pp.refine([(0, 10)], "predicted")


def test_closed_segments_and_runs():
    class Segment:   # the official closed segment: len = end - start + 1
        def __init__(self, action, start, end):
            self.action, self.start, self.end, self.len = action, start, end, end - start + 1
    segs = pp.closed_segments([(0, 3), (3, 4), (4, 9)], np.array([2, 2, 2, 2, 1, 1, 1, 1, 1]), Segment)
    assert [(g.action, g.start, g.end, g.len) for g in segs] == [(2, 0, 2, 3), (2, 3, 3, 1), (1, 4, 8, 5)]   # one class, kept apart
    assert pp.runs([3, 3, 1, 1, 1, 3]) == [(0, 2), (2, 5), (5, 6)]
    with pytest.raises(ValueError):
        pp.closed_segments([(0, 3), (4, 9)], np.zeros(9, int), Segment)


# ----------------------------------------------------------------------------------------------- the adapter on CPU
@pytest.fixture(scope="module")
def fact():
    if not HAS_FACT:
        pytest.skip("the pinned FACT checkout")
    from echo_routing.temporal.fact_head import FACTHead, TokenRule
    return FACTHead(ROOT / "third_party/fact", 8, TokenRule(8, 2.654333333333333, 1.5, 12), base_config="gtea.yaml")


def build(head, policy, cut_fn=None, seed=0):
    torch.manual_seed(seed); m = head.build(); return pp.partitioned(m, head.mods, policy, cut_fn) if policy else m


def video(labels, seed):
    rng = np.random.default_rng(seed); return torch.from_numpy(rng.standard_normal((len(labels), 8)).astype(np.float32)), torch.tensor(labels)


def loss_and_grads(model, xs, ys):
    seed_everything(1); model.train(); model.zero_grad(set_to_none=True)   # FACT's time masking also draws from NumPy
    loss, _ = model(xs, ys, compute_loss=True); loss.backward()
    return loss.detach(), {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}


def test_no_added_cut_equals_the_official_forward_loss_and_gradients(fact):
    xs, ys = zip(video([0] * 6 + [1] * 5 + [0] * 7, 1), video([2] * 7, 2))
    off = build(fact, None)
    for policy, fn in (("original", None), ("predicted", lambda x: []), ("uniform", None)):
        m = build(fact, policy, fn); m.block_list[-1].grid = 10**6                # the grid adds no cut either
        l0, g0 = loss_and_grads(off, list(xs), list(ys)); l1, g1 = loss_and_grads(m, list(xs), list(ys))
        assert torch.equal(l0, l1) and set(g0) == set(g1) and all(torch.equal(g0[k], g1[k]) for k in g0)
        off.eval(); m.eval()
        assert all(np.array_equal(a["pred"], b["pred"]) for a, b in zip(off(list(xs), list(ys)), m(list(xs), list(ys))))


def test_one_interval_list_equals_the_localisation_implementation(fact):
    from scripts.diagnose_fact_localisation import fact_pass
    x, _ = video([0] * 20, 3); off = build(fact, None); off.eval()
    base = pp.runs(fact_pass(off, x, fact.mods)["outs"][1][0][:, 0, -5:].argmax(-1).tolist())
    cuts = [c for s, e in base for c in (s + 2,) if s + 2 < e][:3] or [7]
    ref = fact_pass(off, x, fact.mods, pp.add_cuts(base, cuts))
    m = build(fact, "predicted", lambda _x: cuts); m.eval()
    with torch.no_grad():
        outs = m._forward_one_video(x.unsqueeze(1)); pred = m.block_list[-1].eval()
    assert pp.last_partition(m) == pp.add_cuts(base, cuts)
    assert torch.equal(pred, ref["pred"]) and all(torch.equal(a, b) for o, r in zip(outs, ref["outs"]) for a, b in zip(o, r))


def test_training_path_keeps_gradients_and_the_head_stays_outside(fact):
    torch.manual_seed(5); bh = pp.BoundaryHead(8, 16); fn = pp.head_cut_fn(bh, 0.0, 2)   # threshold 0: the head cuts
    m = build(fact, "predicted", fn); xs, ys = zip(video([0] * 9 + [3] * 9, 4))
    loss, g = loss_and_grads(m, list(xs), list(ys))
    assert torch.isfinite(loss) and any(float(v.abs().max()) > 0 for v in g.values())
    assert not any(k.startswith(("inp", "conv", "out")) for k in dict(m.named_parameters()))   # the head is not a submodule
    assert all(p.grad is None and not p.requires_grad for p in bh.parameters())


def test_labels_passed_in_change_neither_predictions_nor_partitions(fact):
    torch.manual_seed(5); bh = pp.BoundaryHead(8, 16); fn = pp.head_cut_fn(bh, 0.0, 2)
    for policy, cut in (("uniform", None), ("predicted", fn)):
        m = build(fact, policy, cut); m.eval(); x, y = video([1] * 12 + [4] * 11, 6); outs = []
        for lab in (y, torch.zeros_like(y), torch.randint(0, 5, y.shape), y.flip(0), y[:1]):
            pred = m([x], [lab])[0]["pred"]; outs.append((pred.tolist(), pp.last_partition(m)))
        assert all(o == outs[0] for o in outs)


def test_native_like_sequences_are_refined_by_the_same_rule(fact):
    m = build(fact, "uniform"); m.eval(); x, y = video([2] * 30, 7)
    m([x], [y]); part = pp.last_partition(m)
    base = pp.runs(m.block_list[1].frame_clogit[:, 0].argmax(-1).tolist())
    assert part == pp.add_cuts(base, pp.grid_cuts(base)) and len(part) >= 30 // 8   # no flag exempts a one-label sequence
    with pytest.raises(ValueError):
        pp.partitioned(build(fact, None), fact.mods, "predicted")
