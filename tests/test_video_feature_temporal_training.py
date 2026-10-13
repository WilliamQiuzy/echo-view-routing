"""Training pieces of the video feature temporal round: the linear head's masked weighted loss, padding and inference
isolation, the schedule, RNG round trips and identical batches for the three heads. The official MS-TCN and ASRF checks
run where the pinned repositories exist (the server)."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.train_video_feature_temporal as tv  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.temporal.dev_data import PAD_LABEL, collate  # noqa: E402

CW = torch.tensor([0.5, 1.0, 1.0, 2.0, 1.5])


def _seqs(rng, lengths, dim=512):
    return [vsf.Seq(f"s_{i}", i % 2 == 0, rng.normal(size=(n, dim)).astype(np.float32), rng.integers(0, 5, n).astype(np.int64),
                    np.array([n // 2])) for i, n in enumerate(lengths)]


def _linear(seed=0):
    torch.manual_seed(seed); head = tv.LinearHead(CW); model = head.build()
    return head, model


def test_the_linear_loss_is_class_weighted_cross_entropy_over_valid_samples_only():
    rng = np.random.default_rng(0); seqs = _seqs(rng, [7, 12, 3]); head, model = _linear()
    batch = tv.normalise(collate(seqs))
    loss, _ = head.loss(model, batch)
    x = torch.cat([torch.from_numpy(s.feat) for s in seqs]); y = torch.cat([torch.from_numpy(s.labels) for s in seqs])
    want = F.cross_entropy(model(x), y, weight=CW)
    assert torch.allclose(loss, want, atol=1e-6, rtol=0)
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = 3; alt["labels"] = lab   # a real class at padding
    assert float(head.loss(model, tv.normalise(alt))[0]) == float(loss)                              # normalised away before the loss
    alt["x"] = batch["x"] + (1 - batch["mask"]) * 5.0                                                # features at padding as well
    assert torch.equal(tv.normalise(alt)["x"], batch["x"])
    assert int((batch["labels"] == PAD_LABEL).sum()) == int((batch["valid"] == 0).sum()) > 0


def test_the_linear_head_starts_from_the_source_classifier_copy():
    head, _ = _linear(); w = torch.randn(5, 512); b = torch.randn(5)
    model = tv.new_model(head, {"weight": w.clone(), "bias": b.clone()}, "cpu")
    assert torch.equal(model.weight, w) and torch.equal(model.bias, b)
    w.add_(1.0)                                                  # an independent copy: the source tensor is not shared
    assert not torch.equal(model.weight, w)


def test_inference_ignores_labels_joins_and_ids():
    rng = np.random.default_rng(1); seqs = _seqs(rng, [20, 9]); head, model = _linear()
    a = head.decode(model, seqs, "cpu")
    other = [vsf.Seq(f"x_{i}", not s.is_native, s.feat, (s.labels + 2) % 5, np.array([1, 2])) for i, s in enumerate(seqs)]
    b = head.decode(model, other, "cpu")
    for s, o in zip(seqs, other):
        assert np.array_equal(a[s.seq_id]["prob"], b[o.seq_id]["prob"]) and np.array_equal(a[s.seq_id]["labels"], b[o.seq_id]["labels"])
        assert a[s.seq_id]["prob"].dtype == np.float32


def test_the_schedule_warms_up_for_750_updates_then_decays_to_zero():
    ctx = SimpleNamespace(vt={"training": {"optimiser": {"lr": 3e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4, "grad_clip": 1.0},
                                           "schedule": {"warmup_updates": 750}}}, total=18750)
    model = torch.nn.Linear(4, 2); opt, sched, clip = tv.make_optimiser(ctx, model)
    assert clip == 1.0 and opt.param_groups[0]["lr"] == pytest.approx(3e-4 / 750)
    lrs = []
    for _ in range(18750):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    assert lrs[749] == pytest.approx(3e-4) and lrs[750] == pytest.approx(3e-4) and lrs[-1] < 1e-9
    assert all(a >= b for a, b in zip(lrs[750:], lrs[751:]))


def test_the_rng_state_round_trips():
    st = tv.rng_state(); a = (random.random(), np.random.rand(), torch.rand(1).item())
    tv.set_rng_state(st); b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_one_update_changes_the_weights_and_reports_finite_values():
    rng = np.random.default_rng(2); seqs = _seqs(rng, [10, 15]); head, model = _linear()
    ctx = SimpleNamespace(vt={"training": {"optimiser": {"lr": 3e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4, "grad_clip": 1.0},
                                           "schedule": {"warmup_updates": 750}}}, total=18750)
    opt, sched, clip = tv.make_optimiser(ctx, model); w0 = model.weight.detach().clone()
    out = tv.update(head, model, seqs, opt, sched, clip, "cpu")
    assert np.isfinite(out["total"]) and np.isfinite(out["grad_norm"]) and not torch.equal(model.weight, w0)


def test_the_three_heads_read_the_same_batches():
    rng = np.random.default_rng(4)
    streams = [vsf.Seq(f"multi_train_{k:04d}", False, rng.normal(size=(60, 8)).astype(np.float32), rng.integers(0, 5, 60).astype(np.int64), np.array([30]))
               for k in range(8)]
    native = {f"c_{k}": rng.normal(size=(int(rng.integers(9, 300)), 8)).astype(np.float32) for k in range(10)}
    clips = [(v, int(rng.integers(0, 5)), f.shape[0]) for v, f in native.items()]
    builds = [vsf.Batches(streams, clips, native).epoch(0, 3, 8, 256, 4, 4) for _ in range(3)]
    for x, y, z in zip(*builds):
        cx, cy, cz = collate(x), collate(y), collate(z)
        assert all(torch.equal(cx[k], cy[k]) and torch.equal(cx[k], cz[k]) for k in cx) and cx["x"].dtype == torch.float32


# ----------------------------------------------------------------------------------------------- official heads (server)
REPOS = {"mstcn": ROOT / "third_party" / "ms-tcn", "asrf": ROOT / "third_party" / "asrf"}
official = pytest.mark.skipif(not all((p / ".git").exists() for p in REPOS.values()), reason="the pinned official repositories are on the server")


@official
@pytest.mark.parametrize("name", ["mstcn", "asrf"])
def test_official_heads_ignore_padding_labels_and_labels_at_inference(name):
    from echo_routing.temporal.dev_heads import ASRFHead, MSTCNHead
    head = MSTCNHead(REPOS["mstcn"], 512) if name == "mstcn" else ASRFHead(REPOS["asrf"], 512, CW, torch.tensor(20.0))
    torch.manual_seed(0); model = head.build(); model.train()
    rng = np.random.default_rng(5); seqs = _seqs(rng, [40, 25, 33])
    batch = tv.normalise(collate(seqs)); torch.manual_seed(1); l1, _ = head.loss(model, batch)
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = 2; alt["labels"] = lab
    torch.manual_seed(1); l2, _ = head.loss(model, tv.normalise(alt))
    assert float(l1) == float(l2)
    other = [vsf.Seq(f"x_{i}", not s.is_native, s.feat, (s.labels + 1) % 5, np.zeros(0, np.int64)) for i, s in enumerate(seqs)]
    a, b = head.decode(model, seqs, "cpu"), head.decode(model, other, "cpu")
    assert all(np.array_equal(a[s.seq_id]["labels"], b[o.seq_id]["labels"]) for s, o in zip(seqs, other))
