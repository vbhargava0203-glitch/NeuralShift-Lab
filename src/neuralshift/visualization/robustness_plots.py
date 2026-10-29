"""Within-run robustness curves; no spurious confidence intervals for one seed."""

import matplotlib.pyplot as plt
import pandas as pd

from .plots import save_figure


def robustness_figures(frame, output, title):
    base = frame[(frame.method == "msp") & (frame.calibration == "raw")]
    clean = base[base.corruption == "clean"]
    shifted = base[base.corruption != "clean"]
    families = sorted(shifted.corruption.unique())
    files = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for corruption in families:
        group = shifted[shifted.corruption == corruption]
        curve = pd.concat([clean, group]).sort_values("severity")
        ax.plot(curve.severity, curve.accuracy, "o-", label=corruption)
    if not families:
        ax.scatter(clean.severity, clean.accuracy, label="clean")
    ax.set(xlabel="Corruption severity (0 = clean)", ylabel="Accuracy", ylim=(0, 1), title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    files += save_figure(fig, output, "04_accuracy_degradation")
    count = max(1, len(families))
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 3.6), squeeze=False, sharey=True)
    for ax, corruption in zip(axes[0], families or ["clean"], strict=True):
        curve = pd.concat([clean, shifted[shifted.corruption == corruption]]).sort_values("severity")
        ax.plot(curve.severity, curve.accuracy, "o-", label="Accuracy", color="#0072B2")
        ax.plot(curve.severity, curve.mean_confidence, "s--", label="Mean confidence", color="#D55E00")
        ax.set(xlabel="Severity", ylim=(0, 1), title=corruption)
    axes[0, 0].set_ylabel("Fraction")
    axes[0, -1].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    files += save_figure(fig, output, "05_confidence_reliability")
    return files
