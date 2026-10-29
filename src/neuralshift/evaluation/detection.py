"""Binary detection metrics; callers explicitly define the positive event."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def detection_metrics(binary_labels, scores, target_tpr: float = 0.95) -> dict:
    """Higher scores mean positive. Error detection uses incorrect = 1.

    AUPRC denotes sklearn average precision (step-wise PR integration).
    FPR@TPR is the smallest empirical FPR attaining at least target_tpr,
    without interpolation. It is not a deployment threshold selected on test.
    """
    y, score = np.asarray(binary_labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or len(y) == 0:
        raise ValueError("Expected nonempty aligned one-dimensional labels and scores")
    if not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Scores must be finite and labels binary")
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must be in (0, 1]")
    result = {"n_positive": int(y.sum()), "n_negative": int((1 - y).sum())}
    if len(np.unique(y)) < 2:
        result.update(auroc=None, auprc=None, fpr_at_tpr=None,
                      reason="Only one outcome class; detection metrics undefined")
        return result
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    result.update(
        auroc=float(roc_auc_score(y, score)),
        auprc=float(average_precision_score(y, score)),
        fpr_at_tpr=float(fpr[tpr >= target_tpr].min()),
        target_tpr=float(target_tpr), reason=None,
    )
    return result
