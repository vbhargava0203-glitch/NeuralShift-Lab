import math

import pytest
import torch
from torch import nn

from neuralshift.uncertainty.entropy import entropy
from neuralshift.uncertainty.mc_dropout import mc_predict
from neuralshift.uncertainty.softmax import msp, probabilities


def test_entropy_and_msp_known_cases():
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    assert torch.allclose(entropy(p), torch.tensor([math.log(2), 0.0]))
    assert torch.allclose(entropy(p, normalized=True), torch.tensor([1.0, 0.0]))
    assert torch.allclose(msp(p), torch.tensor([0.5, 0.0]))
    assert torch.allclose(probabilities(torch.zeros(2, 3)).sum(-1), torch.ones(2))
    with pytest.raises(ValueError):
        probabilities(torch.ones(2, 3), temperature=0)


def test_mc_dropout_freezes_bn_and_restores_modes():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 3))
    model.train()
    model[2].eval()  # Restoration must preserve heterogeneous child modes.
    before_mean = model[0].running_mean.clone()
    before_var = model[0].running_var.clone()
    before_batches = model[0].num_batches_tracked.clone()
    states = [module.training for module in model.modules()]
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    torch.manual_seed(9)
    result = mc_predict(model, x, passes=8)
    assert result["probabilities"].shape == (8, 5, 3)
    assert result["mean"].shape == (5, 3)
    assert torch.isfinite(result["mutual_information"]).all()
    assert (result["mutual_information"] >= 0).all()
    assert (result["variance"] > 0).any()
    assert torch.equal(model[0].running_mean, before_mean)
    assert torch.equal(model[0].running_var, before_var)
    assert torch.equal(model[0].num_batches_tracked, before_batches)
    assert states == [module.training for module in model.modules()]
    torch.manual_seed(9)
    assert torch.equal(result["probabilities"], mc_predict(model, x, passes=8)["probabilities"])


def test_mc_rejects_disabled_dropout():
    with pytest.raises(ValueError):
        mc_predict(nn.Linear(4, 3), torch.zeros(2, 4), 3)


def test_mc_restores_model_state_even_if_forward_fails():
    class FailingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            raise RuntimeError("intentional fixture failure")

    model = FailingNetwork().train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match="intentional"):
        mc_predict(model, torch.ones(2, 4), 3)
    assert states == [module.training for module in model.modules()]
