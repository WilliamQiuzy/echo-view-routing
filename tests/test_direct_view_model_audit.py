import json
import numpy as np

from scripts.audit_direct_view_models import metrics


def test_independent_metrics_keep_zero_negative_and_count_auc_ties():
    out = metrics(np.array([0, 0, 1, 1]), np.array([0., 1., 1., -1.]), np.array([False, True, True, False]))
    assert out["tp"] == out["tn"] == out["fp"] == out["fn"] == 1
    assert out["balanced_accuracy"] == out["discordant_accuracy"] == .5
    assert out["AUC"] == .375
    json.dumps(out, allow_nan=False)
