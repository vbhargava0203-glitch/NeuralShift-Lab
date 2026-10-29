"""Software tests use random tensors; these are not research experiments."""

import io

import pytest
import torch
from torch import nn

from neuralshift.models.factory import build_model


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
@pytest.mark.parametrize("channels,size", [(1, 28), (3, 32)])
def test_cpu_shapes_and_probabilities(name, channels, size):
    model = build_model(name, in_channels=channels).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, channels, size, size))
    assert logits.shape == (2, 10)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(2))


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_state_dict_roundtrip(name):
    torch.manual_seed(7)
    model = build_model(name).eval()
    inputs = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        expected = model(inputs)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(name).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        assert torch.equal(expected, restored(inputs))


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_batch_normalization_can_be_disabled(name):
    model = build_model(name, batch_norm=False)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert any(isinstance(module, nn.Dropout) for module in model.modules())


def test_small_image_resnet_stem():
    model = build_model("resnet18", in_channels=3)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert isinstance(model.fc[0], nn.Dropout)


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_invalid_parameters(name):
    with pytest.raises(ValueError):
        build_model(name, dropout=1)
    with pytest.raises(ValueError):
        build_model(name, in_channels=2)
    with pytest.raises(ValueError):
        build_model(name, num_classes=1)


def test_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("unregistered")
