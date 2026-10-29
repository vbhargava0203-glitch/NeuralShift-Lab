"""Equal-width, top-label expected calibration error."""

import numpy as np

from .classification import validate_probabilities


def calibration_metrics(prob, labels, bins: int = 15) -> dict:
    """Bins are [left,right), with the last bin including probability one.

    ECE weights absolute empirical confidence-accuracy gaps by bin frequency.
    It depends on binning and sample size and is not a proper scoring rule.
    Empty bins have null accuracy/confidence and contribute zero to ECE.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    p, y = validate_probabilities(prob, labels)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    membership = np.minimum((confidence * bins).astype(int), bins - 1)
    records, ece = [], 0.0
    for index in range(bins):
        selected = membership == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        average = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - average)
        records.append({
            "lower": index / bins, "upper": (index + 1) / bins,
            "count": count, "accuracy": accuracy, "confidence": average,
        })
    return {"ece": float(ece), "bins": records}
