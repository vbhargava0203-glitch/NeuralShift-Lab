import numpy as np
import pytest

from neuralshift.evaluation.classification import classification_metrics
from neuralshift.evaluation.detection import detection_metrics
from neuralshift.evaluation.selective_prediction import risk_coverage


def test_classification_hand_calculated():
    p = np.array([[0.8, 0.2], [0.4, 0.6]])
    result = classification_metrics(p, [0, 0])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["nll"] == pytest.approx(-(np.log(0.8) + np.log(0.4)) / 2)
    assert result["brier"] == pytest.approx(0.4)
    assert result["confusion_matrix"] == [[1, 1], [0, 0]]
    assert result["per_class_precision"] == [1.0, 0.0]
    assert result["per_class_recall"] == [0.5, 0.0]


def test_perfect_and_reversed_error_detection():
    good = detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert good["auroc"] == 1
    assert good["auprc"] == 1
    assert good["fpr_at_tpr"] == 0
    bad = detection_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert bad["auroc"] == 0
    assert bad["auprc"] == pytest.approx((1 / 3 + 2 / 4) / 2)
    missing = detection_metrics([0, 0], [0.1, 0.2])
    assert missing["auroc"] is None and missing["reason"]


def test_risk_coverage_order_and_stable_ties():
    result = risk_coverage([0, 1, 0, 1], [0.0, 0.1, 0.2, 0.3])
    assert result["coverage"] == [0.25, 0.5, 0.75, 1]
    assert result["risk"] == pytest.approx([0, 0.5, 1 / 3, 0.5])
    assert result["aurc"] == pytest.approx(1 / 3)
    assert risk_coverage([1, 0], [0.5, 0.5])["risk"] == [1, 0.5]


def test_invalid_metrics_fail_instead_of_fabricating_values():
    with pytest.raises(ValueError):
        classification_metrics([[0.5, 0.8]], [0])
    with pytest.raises(ValueError):
        risk_coverage([], [])
    with pytest.raises(ValueError):
        detection_metrics([0, 1], [0.1, float("nan")])
