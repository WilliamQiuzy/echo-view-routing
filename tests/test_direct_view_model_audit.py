import json
import copy
import math
import numpy as np
import pytest
import torch

from scripts import audit_direct_view_models as audit


def test_independent_metrics_keep_zero_negative_and_count_auc_ties():
    out = audit.metrics(np.array([0, 0, 1, 1]), np.array([0., 1., 1., -1.]), np.array([False, True, True, False]))
    assert out["tp"] == out["tn"] == out["fp"] == out["fn"] == 1
    assert out["balanced_accuracy"] == out["discordant_accuracy"] == .5
    assert out["AUC"] == .375
    json.dumps(out, allow_nan=False)


def example_state():
    return {"m.encoder.body.weight": torch.ones(2, 3), "m.encoder.head.1.weight": torch.ones(2, 2),
            "m.encoder.head.1.bias": torch.zeros(2), "m.marker_stem.weight": torch.ones(1, 1),
            "m.classifier.weight": torch.ones(5, 2), "m.classifier.bias": torch.zeros(5),
            "h.fc.weight": torch.ones(1, 2), "h.fc.bias": torch.zeros(1)}


def example_optimizer():
    initial = example_state()
    groups = [["m.encoder.body.weight"], ["m.marker_stem.weight"],
              ["m.encoder.head.1.weight", "m.encoder.head.1.bias"], ["h.fc.weight", "h.fc.bias"]]
    names = ["encoder_trunk", "marker_stem", "projection", "scalar_head"]
    opt = {"param_groups": [], "state": {}}; cursor = 0
    factor = .5 * (1 + math.cos(math.pi * (599 - 60) / 540))
    for group, name, base in zip(groups, names, [1e-5, 1e-4, 1e-4, 1e-3]):
        ids = []
        for tensor in group:
            ids.append(cursor); opt["state"][cursor] = {"step": torch.tensor(600.), "exp_avg": torch.zeros_like(initial[tensor]), "exp_avg_sq": torch.ones_like(initial[tensor])}; cursor += 1
        opt["param_groups"].append({"name": name, "params": ids, "base": base, "lr": base * factor,
                                     "betas": (.9, .999), "eps": 1e-8, "weight_decay": 1e-4})
    return initial, opt


def test_independent_live_mask_cannot_hide_a_frozen_encoder_tensor():
    initial = example_state(); live = set(initial) - {"m.classifier.weight", "m.classifier.bias"}
    assert audit.verify_live_mask(initial, live)
    with pytest.raises(AssertionError): audit.verify_live_mask(initial, live - {"m.encoder.body.weight"})
    with pytest.raises(AssertionError): audit.verify_live_mask(initial, set(initial))


@pytest.mark.parametrize("corruption", ["duplicate", "missing_state", "moment_shape", "betas", "epsilon", "final_lr"])
def test_optimizer_corruption_is_detected_independently(corruption):
    initial, opt = example_optimizer(); assert audit.verify_optimizer(opt, initial)["verified"]
    bad = copy.deepcopy(opt)
    if corruption == "duplicate": bad["param_groups"][1]["params"] = bad["param_groups"][0]["params"].copy()
    elif corruption == "missing_state": bad["state"].pop(0)
    elif corruption == "moment_shape": bad["state"][0]["exp_avg"] = torch.zeros(1)
    elif corruption == "betas": bad["param_groups"][0]["betas"] = (.8, .999)
    elif corruption == "epsilon": bad["param_groups"][0]["eps"] = 1e-6
    elif corruption == "final_lr": bad["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(AssertionError): audit.verify_optimizer(bad, initial)
