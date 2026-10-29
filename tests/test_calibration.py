import numpy as np
import pytest
import torch
from torch.nn.functional import cross_entropy

from neuralshift.evaluation.calibration import calibration_metrics
from neuralshift.uncertainty.temperature_scaling import fit_temperature
from neuralshift.uncertainty.softmax import probabilities


def test_validation_temperature_is_positive_and_improves_objective():
    logits = torch.tensor([[8.0, 0.0], [7.0, 0.0], [0.0, 8.0], [0.0, 7.0]])
    labels = torch.tensor([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert 0 < temperature <= 100
    assert cross_entropy(logits / temperature, labels) <= cross_entropy(logits, labels)
    assert torch.equal(logits.argmax(1), (logits / temperature).argmax(1))
    with pytest.raises(ValueError, match="validation"):
        fit_temperature(logits, labels, split="test")


def test_ece_manually_verifiable_and_endpoint_bins():
    p = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
    labels = np.array([0, 1, 1, 1])
    # All in [0.5,1]: accuracy .75 and average confidence .75.
    result = calibration_metrics(p, labels, bins=2)
    assert result["ece"] == pytest.approx(0)
    assert result["bins"][0]["count"] == 0
    endpoint = calibration_metrics([[1.0, 0.0], [0.0, 1.0]], [0, 0], bins=10)
    assert endpoint["ece"] == pytest.approx(0.5)
    assert endpoint["bins"][-1]["count"] == 2


def test_temperature_input_contract():
    logits = torch.ones(2, 3)
    for labels in (torch.tensor([0.1, 1.0]), torch.tensor([[0], [1]]),
                   torch.tensor([float("nan"), 1]), torch.tensor([0, 3])):
        with pytest.raises(ValueError):
            fit_temperature(logits, labels)


def test_binary_rankings_preserved_but_multiclass_rankings_can_change():
    binary = torch.tensor([[0.2, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert torch.equal(probabilities(binary).max(1).values.argsort(),
                       probabilities(binary, temperature=5).max(1).values.argsort())
    multiclass = torch.tensor([[2.0, 0.0, 0.0], [2.0, 1.5, -10.0]])
    cold = probabilities(multiclass)
    warm = probabilities(multiclass, temperature=5)
    assert torch.equal(cold.argmax(1), warm.argmax(1))
    assert not torch.equal(cold.max(1).values.argsort(), warm.max(1).values.argsort())
