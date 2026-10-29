"""Reliability diagrams from stored logits, not from illustrative values."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from .plots import save_figure


def reliability_diagram(cache, temperature, output, title, name, bins=15):
    labels = cache["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, (label, scale) in zip(axes, [("Raw", 1.0), ("Temperature-scaled", temperature)], strict=True):
        probabilities = softmax(cache["logits"] / scale, axis=1)
        confidence = probabilities.max(1)
        correct = probabilities.argmax(1) == labels
        indices = np.minimum((confidence * bins).astype(int), bins - 1)
        counts = np.bincount(indices, minlength=bins)
        accuracy = np.divide(np.bincount(indices, weights=correct, minlength=bins), counts,
                             out=np.full(bins, np.nan), where=counts > 0)
        centers = (np.arange(bins) + .5) / bins
        ax.bar(centers, accuracy, width=.9 / bins, color="#0072B2", alpha=.8)
        ax.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted confidence", title=label)
    axes[0].set_ylabel("Accuracy within confidence bin")
    fig.suptitle(title + "\n15 equal-width bins; empty bins omitted")
    fig.tight_layout()
    return save_figure(fig, output, name)
