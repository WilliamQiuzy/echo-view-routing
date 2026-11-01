"""Prospective A51-T source-classifier evidence qualification; no model rule."""


def transfer_qualification(metrics, original_label_gate):
    checks = {}
    g = metrics["G"]
    for arm in ("L", "W"):
        a = metrics[arm]
        checks[arm] = {
            "balanced_accuracy": a["balanced_accuracy"] >= .80,
            "A4C_recall": a["a4c_recall"] >= .70,
            "A5C_recall": a["a5c_recall"] >= .70,
            "A4C_false_positive_protection": a["a4c_fpr"] <= g["a4c_fpr"] + .02 + 1e-12,
            "AUC_floor": a["AUC"] >= .85,
            "AUC_increment_over_original_classifier": a["AUC"] - g["AUC"] >= .03 - 1e-12,
            "image_over_geometry": a["balanced_accuracy"] - metrics["geometry"]["balanced_accuracy"] >= .10 - 1e-12,
            "image_over_constant": a["balanced_accuracy"] - metrics["constant"]["balanced_accuracy"] >= .10 - 1e-12,
        }
    absolute = any(all(v.values()) for v in checks.values())
    return {"original_A51P_label_increment_pass": bool(original_label_gate),
            "absolute_source_pass": absolute, "absolute_source_checks": checks,
            "target_transfer_eligible": bool(original_label_gate) or absolute,
            "target_arms_if_eligible": ["L", "W"], "target_folds": [0, 1],
            "source_claim_scope": "Whole classifiers with different trained heads; not an independently identified backbone/source-image effect."}
