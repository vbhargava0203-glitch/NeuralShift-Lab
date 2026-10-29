"""Selective risk among examples accepted from least to most uncertain."""

import numpy as np


def risk_coverage(errors, scores) -> dict:
    """Empirical AURC = mean risk at coverages k/n, k=1,...,n.

    This right-rectangle definition excludes the undefined risk at zero
    coverage. Score ties preserve input order; callers persist sample ids.
    """
    error = np.asarray(errors)
    score = np.asarray(scores, dtype=float)
    if error.ndim != 1 or score.shape != error.shape or len(error) == 0:
        raise ValueError("Expected nonempty aligned one-dimensional arrays")
    if not np.isin(error, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Errors must be binary and uncertainty scores finite")
    order = np.argsort(score, kind="stable")
    accepted = np.arange(1, len(error) + 1)
    risk = np.cumsum(error[order]) / accepted
    return {"coverage": (accepted / len(error)).tolist(), "risk": risk.tolist(),
            "aurc": float(risk.mean())}
