"""Exploratory paired comparisons using independent training seeds as units."""

import numpy as np
import pandas as pd
from scipy import stats


def summarize_values(values) -> dict:
    """t interval across independent runs; a single run has no estimated CI."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    result = {"n": n, "mean": float(x.mean()) if n else None,
              "std": float(x.std(ddof=1)) if n > 1 else None,
              "ci_low": None, "ci_high": None}
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
        result.update(ci_low=result["mean"] - half, ci_high=result["mean"] + half)
    return result


def paired_comparison(
    frame: pd.DataFrame, metric: str, method_a: str, method_b: str,
    seed: int = 0, bootstrap_samples: int = 2000,
) -> dict:
    """Match methods by seed and all known condition keys, then bootstrap seeds.

    Multiple conditions within a seed are equally weighted before inference.
    Missing pairs are excluded and counted. Duplicate keys fail loudly.
    This estimates variability between training seeds, not test-sample
    uncertainty. P values are exploratory and uncorrected; callers should
    predefine comparisons or apply a family-wise correction externally.
    """
    required = {"seed", "method", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if method_a == method_b:
        raise ValueError("Paired comparison requires two different method names")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    keys = [name for name in ("dataset", "architecture", "model", "corruption",
                             "severity", "calibration", "seed") if name in frame.columns]
    subset = frame[frame.method.isin([method_a, method_b])].copy()
    if subset.duplicated(keys + ["method"]).any():
        raise ValueError("Duplicate method/condition/seed rows prevent unique paired alignment")
    left = subset[subset.method == method_a][keys + [metric]].rename(columns={metric: "a"})
    right = subset[subset.method == method_b][keys + [metric]].rename(columns={metric: "b"})
    joined = left.merge(right, on=keys, how="outer", indicator=True)
    valid = (joined["_merge"] == "both") & np.isfinite(joined.a) & np.isfinite(joined.b)
    paired = joined[valid].copy()
    paired["difference"] = paired.a - paired.b
    differences = paired.groupby("seed", sort=True).difference.mean().to_numpy()
    result = {
        "metric": metric, "method_a": method_a, "method_b": method_b,
        "direction": "method_a minus method_b", "unit": "independent training seed",
        "condition_aggregation": "equal weight within seed",
        "matched_pairs": int(valid.sum()), "excluded_pairs": int((~valid).sum()),
        "n_seeds": len(differences), "mean_difference": None, "std_difference": None,
        "ci_low": None, "ci_high": None, "cohen_dz": None, "p_value": None,
        "inference": "exploratory; no multiple-testing correction",
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
    }
    if len(differences):
        result["mean_difference"] = float(differences.mean())
    if len(differences) < 2:
        result["reason"] = "At least two independent training seeds are needed for intervals"
        return result
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, (bootstrap_samples, len(differences)), replace=True).mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    sd = float(differences.std(ddof=1))
    result.update(std_difference=sd, ci_low=float(low), ci_high=float(high),
                  cohen_dz=float(differences.mean() / sd) if sd > 0 else None,
                  p_value=float(stats.ttest_1samp(differences, 0).pvalue) if sd > 0
                  else (1.0 if np.all(differences == 0) else None), reason=None)
    return result
