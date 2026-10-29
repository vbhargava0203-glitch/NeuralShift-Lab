"""Consistent, print-friendly plotting utilities."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "msp": "#0072B2",
    "entropy": "#D55E00",
    "mc_entropy": "#009E73",
    "mc_mi": "#CC79A7",
    "mc_variance": "#E69F00",
}


def configure_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 180})


def save_figure(fig, output: Path, name: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        files.append(path)
    plt.close(fig)
    return files


def table_markdown(frame) -> str:
    """Render a table without requiring the optional tabulate dependency."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.5f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
