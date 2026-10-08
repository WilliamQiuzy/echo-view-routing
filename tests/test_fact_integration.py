"""Checks of the FACT integration on synthetic data: the official transcript rule and token count, the configuration,
the unpadding of the shared batches, inference label isolation, the loss terms against the official per-block losses,
one-to-one matching with a repeated class, padding and the structural zero gradients, where the pinned checkout exists."""
from pathlib import Path

import numpy as np
import pytest
import torch

from echo_routing.temporal.dev_data import PAD_LABEL, Seq, collate
from echo_routing.temporal.dev_train import seed_everything
from echo_routing.temporal.fact_head import FACTHead, TokenRule, loss_parts, parameter_group, token_rule, transcript_length

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "fact"
needs_fact = pytest.mark.skipif(not (REPO / ".git").exists(), reason="the pinned FACT checkout is cloned on the server (and locally for tests)")
RULE = TokenRule(max_transcript=8, average_transcript=2.65, token_ratio=1.5, ntoken=12)


def seq(labels, seed: int = 0, d: int = 512) -> Seq:
    rng = np.random.default_rng(seed); labels = np.asarray(labels, np.int64)
    return Seq(f"s{seed}", False, rng.standard_normal((labels.size, d)).astype(np.float16), labels, np.zeros(0, int))


def tensors(s: Seq) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(s.feat.astype(np.float32)), torch.from_numpy(s.labels)


def test_transcript_counts_runs_and_merges_a_same_view_join():
    assert transcript_length(np.array([0] * 10 + [0] * 10)) == 1          # a same-view join is one run
    assert transcript_length(np.array([2] * 5 + [3] * 4 + [2] * 6)) == 3   # A4C -> A5C -> A4C keeps both A4C runs
    assert transcript_length(np.array([], int)) == 0


def test_token_rule_follows_the_guide():
    labs = [np.array([0] * 5 + [1] * 5), np.array([2] * 3 + [3] * 3 + [2] * 3 + [4] * 2 + [0] * 2)]   # 2 and 5 runs
    r = token_rule(labs, natives_per_epoch=3, token_ratio=1.5)
    assert (r.max_transcript, r.ntoken) == (5, 7) and r.average_transcript == pytest.approx((2 + 5 + 3) / 5)
    with pytest.raises(ValueError):
        token_rule(labs, 3, 0.5)


def test_unpad_returns_each_video_and_rejects_inner_padding():
    a, b = seq([0] * 7 + [1] * 5, 1, d=8), seq([3] * 4, 2, d=8)
    batch = collate([a, b]); xs, labs = FACTHead.unpad(batch)
    assert [tuple(x.shape) for x in xs] == [(12, 8), (4, 8)]
    assert np.array_equal(xs[1].numpy(), b.feat.astype(np.float32)) and np.array_equal(labs[0].numpy(), a.labels)
    bad = dict(batch); lab = batch["labels"].clone(); lab[0, 3] = PAD_LABEL; bad["labels"] = lab
    with pytest.raises(ValueError):
        FACTHead.unpad(bad)


def test_stream_categories():
    from scripts.fact_integration import categories
    c = categories(np.array([2] * 5 + [2] * 5 + [3] * 4 + [2] * 6), np.array([5, 10, 14]))
    assert c["same_view_join"] and c["A4C_A5C_A4C"] and c["X_Y_X"] and c["repeated_class"]
    assert not c["single_view"] and not c["A5C_A4C_A5C"]
    assert categories(np.array([1] * 8), np.array([4]))["single_view"]


@pytest.fixture(scope="module")
def small() -> FACTHead:
    """The official gtea base keeps CPU tests fast; the breakfast base is checked for its configuration."""
    return FACTHead(REPO, 512, RULE, base_config="gtea.yaml")


@needs_fact
def test_configuration_follows_the_guide():
    c = FACTHead(REPO, 512, RULE).cfg
    assert c.FACT.ntoken == 12 and c.FACT.trans is False and c.Loss.match == "o2o" and list(c.bg_class) == []
    assert c.Loss.nullw == pytest.approx(12 / ((12 - 2.65) * 5))     # utils/train_tools.compute_null_weight
    assert c.FACT.block == "iuUU" and c.Bi.f == "m2" and c.is_frozen()


@needs_fact
def test_inference_ignores_the_labels_passed_to_the_official_forward(small):
    seed_everything(0); model = small.build().eval(); x, ref = tensors(seq([2] * 30 + [3] * 20 + [2] * 25, 3))
    with torch.no_grad():
        outs = [model([x], [lab])[0]["pred"] for lab in (ref, torch.zeros_like(ref), torch.randint(0, 5, ref.shape), ref.flip(0), ref[:1])]
    assert all(np.array_equal(o, outs[0]) for o in outs)
    assert np.array_equal(small.predict(model, x)["labels"], outs[0])


@needs_fact
def test_loss_terms_add_up_to_the_official_loss_and_matching_is_one_to_one(small):
    seed_everything(0); model = small.build().train(); x, lab = tensors(seq([2] * 20 + [3] * 15 + [2] * 20 + [0] * 10, 4))
    seed_everything(1); model._forward_one_video(x.unsqueeze(1)); parts = loss_parts(small, model, lab)   # raises if a block disagrees
    seed_everything(1); official, _ = model([x], [lab], compute_loss=True)
    assert float(parts["total"]) == pytest.approx(float(official), rel=1e-5)
    a, sg = (t.tolist() for t in parts["match"])
    assert parts["n_segments"] == 4 and len(set(a)) == 4 and sorted(sg) == [0, 1, 2, 3]   # the two A4C runs get two tokens


@needs_fact
def test_padding_does_not_change_the_adapter_loss(small):
    seed_everything(0); model = small.build().train(); a, b = seq([0] * 12 + [1] * 9, 5), seq([4] * 6, 6)
    seed_everything(2); in_batch, _ = small.loss(model, collate([a, b]))
    (xa, la), (xb, lb) = tensors(a), tensors(b)
    seed_everything(2); alone, _ = model([xa, xb], [la, lb], compute_loss=True)
    assert float(in_batch) == pytest.approx(float(alone), rel=1e-6)


@needs_fact
def test_parameter_groups_and_the_structural_zero_gradients(small):
    seed_everything(0); model = small.build().train()
    groups = {parameter_group(n, model) for n, _ in model.named_parameters()}
    assert "action_query" in groups and all(g == "action_query" or g.startswith("block") for g in groups)
    x, lab = tensors(seq([2] * 20 + [3] * 15 + [2] * 20, 7)); loss, _ = model([x], [lab], compute_loss=True); loss.backward()
    zero = sorted(n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0)
    # trans=false starts the action tokens from zeros, so the input block's first self-attention sees identical values:
    # its projection weights, output weight and first norm weight receive no gradient at initialisation
    assert zero == sorted(f"block_list.0.action_branch.layers.0.{n}" for n in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "norm1.weight"))


@needs_fact
def test_script_checks_on_cpu(small):
    """The script's own checks on synthetic sequences: isolation, term contributions, the overfit outcome bookkeeping,
    the token flow and the gradient paths."""
    import scripts.fact_integration as fi
    seed_everything(0); model = small.build()
    seqs = [seq([2] * 20 + [3] * 15 + [2] * 20, 8), seq([1] * 30, 9), seq([0] * 10 + [0] * 10 + [4] * 12, 10)]
    iso = fi.isolation(small, model, seqs, "cpu", seed=1)
    assert (iso["sequences"], iso["comparisons"], iso["changed_predictions"], iso["adapter_decode_differs"]) == (3, 12, 0, 0)
    assert iso["max_abs_logit_difference"] == 0.0
    model.eval()
    with torch.no_grad():
        terms = fi.term_losses(small, model, seqs, "cpu")    # raises if the contributions miss the official total
    assert set(terms) >= {"frame", "seg", "token", "f2a", "a2f", "smooth", "total"}
    out = fi.outcome(small, model, [("k", s) for s in seqs], "cpu")
    assert out["sequences"] == 3 and out["match_one_to_one"] == 3 and len(out["rows"]) == 3
    assert [r["segments"] for r in out["rows"]] == [3, 1, 2]
    flow = fi.token_flow(small, model, seqs[0], "cpu")
    assert flow["input"] == [55, 1, 512] and flow["tokens"] == 12 and [b["type"] for b in flow["blocks"]] == ["InputBlock", "UpdateBlock", "UpdateBlockTDU"]
    if flow["token_runs"]:
        assert flow["token_runs"][0]["start"] == 0 and flow["token_runs"][-1]["end"] == 55
    gp = fi.gradient_paths(small, model, seqs, "cpu", seed=0)
    assert "action_query" in gp["token"]["groups_with_gradient"] and all(gp[t]["groups_with_gradient"] for t in gp)
    assert sorted(gp["official_total"]["zero_gradient_parameters"]) == sorted(
        f"block_list.0.action_branch.layers.0.{n}" for n in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "norm1.weight"))
