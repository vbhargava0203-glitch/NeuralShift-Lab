"""Classification metrics computed from aligned probability vectors and labels."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def validate_probabilities(prob, labels) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(prob, dtype=np.float64)
    raw_y = np.asarray(labels)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) == 0:
        raise ValueError("Expected a nonempty NxK probability matrix, K >= 2")
    if raw_y.ndim != 1 or len(raw_y) != len(p):
        raise ValueError("Labels must be a one-dimensional aligned vector")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    if not np.isfinite(raw_y).all() or np.any(raw_y != raw_y.astype(np.int64)):
        raise ValueError("Labels must be integers")
    y = raw_y.astype(np.int64)
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Labels outside probability class range")
    return p, y


def classification_metrics(probabilities, labels) -> dict:
    """Brier sums over classes then averages examples (range [0, 2]).

    NLL clips true-class probabilities at 1e-12 for numerical stability.
    Macro F1 averages over every model class, including absent classes as zero.
    """
    p, y = validate_probabilities(probabilities, labels)
    predicted = p.argmax(axis=1)
    classes = np.arange(p.shape[1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=classes, zero_division=0
    )
    target = np.eye(p.shape[1])[y]
    return {
        "accuracy": float(np.mean(predicted == y)),
        "macro_f1": float(f1.mean()),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12, 1)).mean()),
        "brier": float(np.square(p - target).sum(axis=1).mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": confusion_matrix(y, predicted, labels=classes).tolist(),
        "mean_confidence": float(p.max(axis=1).mean()),
    }
