"""Precisely defined severity mappings; these are not CIFAR-10-C recipes."""

import torch

from .blur import gaussian_blur
from .brightness import brightness
from .contrast import contrast
from .gaussian_noise import gaussian_noise

SEVERITIES = {
    "gaussian_noise": [0.05, 0.10, 0.15, 0.20, 0.30],
    "blur": [0.5, 0.8, 1.1, 1.5, 2.0],
    "brightness": [0.05, 0.10, 0.20, 0.30, 0.40],
    "contrast": [0.8, 0.6, 0.4, 0.2, 0.1],
}


def apply_corruption(
    images: torch.Tensor, name: str, severity: int, seed: int = 0
) -> torch.Tensor:
    """Return corrupted BCHW pixels; never mutate input or change labels.

    Gaussian noise uses one CPU torch generator per call. To make results
    independent of batching, call on one image with seed + original sample id.
    Apply normalization only after this function. Clean uses severity zero.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("Expected floating-point BCHW images in [0, 1]")
    if images.numel() == 0 or not torch.isfinite(images).all():
        raise ValueError("Images must be nonempty and finite")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Corruption must precede normalization: expected [0, 1]")
    if name == "clean":
        if severity != 0:
            raise ValueError("Clean condition requires severity 0")
        return images.clone()
    if name not in SEVERITIES:
        raise ValueError(f"Unknown corruption {name!r}; choose {list(SEVERITIES)}")
    if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
        raise ValueError("Corruption severity must be an integer from 1 through 5")
    parameter = SEVERITIES[name][severity - 1]
    if name == "gaussian_noise":
        return gaussian_noise(images, parameter, seed)
    if name == "blur":
        return gaussian_blur(images, parameter)
    if name == "brightness":
        return brightness(images, parameter)
    return contrast(images, parameter)
