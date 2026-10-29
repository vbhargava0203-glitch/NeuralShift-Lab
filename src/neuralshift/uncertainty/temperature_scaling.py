"""Fit one positive temperature using clean validation labels exclusively."""

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, split: str = "validation"
) -> float:
    """Minimize validation NLL in log-temperature over T in [0.01, 100].

    An explicit provenance guard rejects test/training fitting requests.
    The caller remains responsible for supplying the named held-out split.
    T=1 is retained if optimization does not reduce validation NLL.
    """
    if split != "validation":
        raise ValueError("Temperature must be fitted on the validation split only")
    z = np.asarray(logits.detach().cpu(), dtype=np.float64)
    raw_y = np.asarray(labels.detach().cpu())
    if (z.ndim != 2 or z.shape[1] < 2 or raw_y.ndim != 1
            or len(raw_y) != len(z) or len(raw_y) == 0 or not np.isfinite(z).all()):
        raise ValueError("Expected nonempty finite NxK logits and aligned labels")
    if not np.isfinite(raw_y).all() or np.any(raw_y != raw_y.astype(np.int64)):
        raise ValueError("Labels must contain finite integer class indices")
    y = raw_y.astype(np.int64)
    if np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError("Labels outside logit class range")

    def objective(log_temperature: float) -> float:
        scaled = z / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(objective, bounds=(-np.log(100), np.log(100)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization did not converge")
    return float(np.exp(result.x)) if result.fun <= objective(0.0) else 1.0
