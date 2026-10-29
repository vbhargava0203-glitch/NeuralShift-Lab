import pandas as pd
import pytest

from neuralshift.evaluation.statistics import paired_comparison, summarize_values


def comparison_frame():
    rows = []
    for seed, difference in enumerate([0.1, 0.2, 0.3]):
        for severity in [1, 2]:
            rows.extend([
                {"seed": seed, "severity": severity, "method": "a", "auroc": 0.5 + difference},
                {"seed": seed, "severity": severity, "method": "b", "auroc": 0.5},
            ])
    return pd.DataFrame(rows)


def test_paired_alignment_bootstraps_seeds_not_predictions():
    frame = comparison_frame().sample(frac=1, random_state=4)
    a = paired_comparison(frame, "auroc", "a", "b", seed=4)
    b = paired_comparison(frame, "auroc", "a", "b", seed=4)
    assert a == b
    assert a["matched_pairs"] == 6
    assert a["n_seeds"] == 3
    assert a["mean_difference"] == pytest.approx(0.2)
    assert a["ci_low"] <= 0.2 <= a["ci_high"]
    assert a["std_difference"] == pytest.approx(0.1)
    assert a["cohen_dz"] == pytest.approx(2)


def test_missing_duplicate_and_single_seed_handling():
    frame = comparison_frame()
    incomplete = paired_comparison(frame.iloc[1:], "auroc", "a", "b")
    assert incomplete["excluded_pairs"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        paired_comparison(pd.concat([frame, frame.iloc[:1]]), "auroc", "a", "b")
    single = paired_comparison(frame[frame.seed == 0], "auroc", "a", "b")
    assert single["n_seeds"] == 1
    assert single["ci_low"] is None and single["p_value"] is None


def test_run_summary_requires_independent_replicates_for_ci():
    single = summarize_values([0.5])
    assert single["mean"] == 0.5 and single["ci_low"] is None
    multiple = summarize_values([0.2, 0.4, 0.6])
    assert multiple["mean"] == pytest.approx(0.4)
    assert multiple["ci_low"] < multiple["mean"] < multiple["ci_high"]
