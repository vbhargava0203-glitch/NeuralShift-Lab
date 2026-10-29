import pytest
import torch

from neuralshift.corruptions.registry import SEVERITIES, apply_corruption


@pytest.mark.parametrize("name", list(SEVERITIES))
@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_corruption_contract(name, severity):
    images = torch.linspace(0, 1, 2 * 3 * 12 * 12).reshape(2, 3, 12, 12)
    before = images.clone()
    result = apply_corruption(images, name, severity, seed=7)
    assert result.shape == images.shape
    assert torch.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    assert torch.equal(images, before)


def test_randomness_and_severity_validation():
    images = torch.full((2, 1, 8, 8), 0.5)
    first = apply_corruption(images, "gaussian_noise", 3, seed=4)
    assert torch.equal(first, apply_corruption(images, "gaussian_noise", 3, seed=4))
    assert not torch.equal(first, apply_corruption(images, "gaussian_noise", 3, seed=5))
    for severity in (0, 6, 1.5, True):
        with pytest.raises(ValueError):
            apply_corruption(images, "contrast", severity)
    assert torch.equal(images, apply_corruption(images, "clean", 0))
    with pytest.raises(ValueError):
        apply_corruption(images - 1, "brightness", 1)


def test_defined_brightness_and_contrast():
    images = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    assert torch.allclose(apply_corruption(images, "brightness", 2),
                          torch.tensor([[[[0.1, 1.0], [0.1, 1.0]]]]))
    assert torch.allclose(apply_corruption(images, "contrast", 1),
                          torch.tensor([[[[0.1, 0.9], [0.1, 0.9]]]]))


def test_blur_preserves_constant_images():
    x = torch.full((1, 1, 28, 28), 0.4)
    assert torch.allclose(apply_corruption(x, "blur", 5), x, atol=1e-6)
