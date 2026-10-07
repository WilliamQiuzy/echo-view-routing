"""scripts/train_l1_asrf.py: batch identities, the A/B plan, the checkpoint rule, the selection guard, the run-identity
skip rule and (where the pinned ASRF exists) padding invariance and gradients. Synthetic data only."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.train_l1_asrf as tl
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.dev_data import Seq, TrainData

ROOT = Path(__file__).resolve().parents[1]
TR = {"native_per_epoch": 3, "native_window": 8, "streams_per_batch": 1, "natives_per_batch": 1, "epochs": 2, "updates_per_epoch": 3}


class _Store:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def get(self, vid: str, variant: str) -> VideoFeatures:
        n = 10 + int(vid[-1]); rng = np.random.default_rng(int(vid[-1]))
        return VideoFeatures(vid, np.arange(n) * 3, np.arange(n) * 0.1, (self.scale * rng.standard_normal((n, 4))).astype(np.float32),
                             np.full((n, 5), 0.2, np.float32), np.zeros((n, 5), np.float32))


def _data(scale: float) -> SimpleNamespace:
    rng = np.random.default_rng(3)
    streams = tuple(Seq(f"s{i}", False, (scale * rng.standard_normal((6, 4))).astype(np.float16), np.array([0, 0, 0, 2, 2, 2]), np.array([3]))
                    for i in range(3))
    return SimpleNamespace(train=TrainData(streams, (("v1", 1), ("v2", 3), ("v3", 4)), _Store(scale)))


def test_plan_is_identical_for_two_feature_sets():
    a, b = tl.make_plan(_data(1.0), TR, 0), tl.make_plan(_data(5.0), TR, 0)
    assert a == b and len(a["epochs"]) == 2 and all(len(e) == 3 for e in a["epochs"])
    assert tl.make_plan(_data(1.0), TR, 1) != a                     # another seed, other batches
    assert len(a["class_weight"]) == 5 and a["pos_weight"] > 1


def test_plan_refuses_a_wrong_update_count():
    with pytest.raises(SystemExit, match="updates"):
        tl.make_plan(_data(1.0), {**TR, "updates_per_epoch": 4}, 0)


def test_batch_digest_ignores_features_only():
    s = Seq("x", True, np.zeros((4, 4), np.float16), np.array([1, 1, 1, 1]), np.zeros(0, int))
    t = Seq("x", True, np.ones((4, 4), np.float16), np.array([1, 1, 1, 1]), np.zeros(0, int))
    u = Seq("x", True, np.zeros((4, 4), np.float16), np.array([1, 1, 2, 2]), np.zeros(0, int))
    assert tl.batch_digest([s]) == tl.batch_digest([t]) != tl.batch_digest([u])


def test_saved_plan_is_reused_or_refused(tmp_path):
    plan = tl.make_plan(_data(1.0), TR, 0); p, sha = tl.save_or_check_plan(plan, tmp_path)
    assert p.is_file() and tl.save_or_check_plan(plan, tmp_path)[1] == sha
    other = {**plan, "pos_weight": plan["pos_weight"] + 1}
    with pytest.raises(SystemExit, match="differ"):
        tl.save_or_check_plan(other, tmp_path)


def _rec(epoch, native, recall, seg, cuts):
    return {"epoch": epoch, "native_macro_f1": native, "boundary_recall_multi": recall, "segment_f1_50_multi": seg, "native_internal_cuts_per_clip": cuts}


def test_checkpoint_rule():
    rule = {"native_min": 0.98, "recall_min": 0.95}
    recs = [_rec(0, 0.99, 0.96, 0.90, 0.1), _rec(1, 0.97, 0.99, 0.99, 0.0), _rec(2, 0.99, 0.96, 0.95, 0.2), _rec(3, 0.99, 0.96, 0.95, 0.1),
            _rec(4, 0.99, 0.94, 0.99, 0.0)]
    assert tl.choose(recs, rule) == {"epoch": 3, "status": "feasible", "n_feasible": 3}   # tie at 0.95 -> fewer cuts
    recs.append(_rec(5, 0.99, 0.96, 0.95, 0.1))
    assert tl.choose(recs, rule)["epoch"] == 3                                            # full tie -> earlier epoch
    none = [_rec(0, 0.97, 0.99, 0.9, 0.0), _rec(1, 0.99, 0.90, 0.9, 0.0)]
    assert tl.choose(none, rule) == {"epoch": 1, "status": "infeasible (final epoch shown as diagnostic)", "n_feasible": 0}


def test_selection_is_unavailable_and_manifests_by_arm(tmp_path):
    with pytest.raises(RuntimeError, match="selection"):
        tl.selection_forbidden()
    la = {"inputs": {"e1_feature_manifest": {"path": "models_frozen/x.json"}}}
    assert tl.feature_manifest(la, "A", 0, tmp_path) == tmp_path / "models_frozen/x.json"
    assert tl.feature_manifest(la, "B", 2, tmp_path).name == "feature_manifest_L1_s2.json"
    with pytest.raises(ValueError):
        tl.feature_manifest(la, "C", 0, tmp_path)


def test_finished_run_rule(tmp_path):
    from echo_routing.evaluate.complementarity import sha256_file
    assert tl.finished_run(tmp_path / "none.json", "x") is None
    sel, fin = tmp_path / "selected.pt", tmp_path / "final.pt"; sel.write_bytes(b"a"); fin.write_bytes(b"b")
    man = {"identity": "id1", "selected_checkpoint": str(sel), "selected_sha256": sha256_file(sel), "final_checkpoint": str(fin), "final_sha256": sha256_file(fin)}
    p = tmp_path / "run_manifest.json"; p.write_text(json.dumps(man))
    assert tl.finished_run(p, "id1") is True and tl.finished_run(p, "id2") is False
    fin.write_bytes(b"changed")
    assert tl.finished_run(p, "id1") is False


ASRF = ROOT / "third_party/asrf"


@pytest.mark.skipif(not (ASRF / "libs").is_dir(), reason="needs the pinned ASRF repository (server)")
def test_padding_labels_do_not_change_the_loss_and_gradients_flow():
    from echo_routing.temporal.dev_data import collate
    from echo_routing.temporal.dev_heads import ASRFHead
    head = ASRFHead(ASRF, 4, torch.ones(5), torch.tensor(5.0))
    torch.manual_seed(0); model = head.build(); model.train()
    seqs = [Seq("a", True, np.random.default_rng(0).standard_normal((12, 4)).astype(np.float16), np.full(12, 2), np.zeros(0, int)),
            Seq("b", False, np.random.default_rng(1).standard_normal((7, 4)).astype(np.float16), np.array([0, 0, 0, 3, 3, 3, 3]), np.array([3]))]
    batch = collate(seqs)
    torch.manual_seed(1); l1, _ = head.loss(model, batch)
    alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    torch.manual_seed(1); l2, _ = head.loss(model, alt)
    assert float(l1) == float(l2) and int((batch["valid"] == 0).sum()) == 5
    l1.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_config_architecture_must_match_the_built_asrf():
    from echo_routing.config import load_config
    la = load_config(ROOT / "configs/experiments/l1_asrf_integration_v1.yaml")["l1_asrf"]
    assert {k: int(v) for k, v in la["asrf"]["model"].items()} == tl.ASRF_BUILD
    bad = {**la, "asrf": {**la["asrf"], "model": {**la["asrf"]["model"], "n_layers": 12}}}
    with pytest.raises(SystemExit, match="architecture"):
        tl.make_head(bad, SimpleNamespace(repo_root=ROOT), {"class_weight": [1.0] * 5, "pos_weight": 2.0}, torch.device("cpu"))
