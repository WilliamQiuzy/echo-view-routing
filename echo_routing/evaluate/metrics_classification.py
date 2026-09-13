"""Classification endpoints: accuracy, macro-F1, balanced accuracy, confusion (proposal §7.2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    return {
        "n": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
