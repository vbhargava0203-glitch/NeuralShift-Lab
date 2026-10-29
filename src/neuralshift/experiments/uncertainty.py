"""Score cached predictions without repeated deterministic model inference."""

import numpy as np
from scipy.special import softmax


def entropy_numpy(p):
    return -(p * np.log(np.clip(p, 1e-15, 1))).sum(axis=-1)


def scored_predictions(logits, mc_probabilities, temperature):
    for calibration, scalar in (("raw", 1.0), ("temperature", temperature)):
        p = softmax(logits / scalar, axis=-1)
        yield "msp", calibration, p, 1 - p.max(axis=1)
        yield "entropy", calibration, p, entropy_numpy(p)
    raw = softmax(logits, axis=-1)
    mean = mc_probabilities.mean(axis=0)
    total = entropy_numpy(mean)
    expected = entropy_numpy(mc_probabilities).mean(axis=0)
    yield "mc_entropy", "raw", raw, total
    yield "mc_mi", "raw", raw, np.maximum(total - expected, 0)
    yield "mc_variance", "raw", raw, mc_probabilities.var(axis=0).mean(axis=-1)
