"""Normalized separable Gaussian kernel with reflection padding."""

import math

import torch
import torch.nn.functional as F


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Sigma is measured in pixels; truncate at 3 sigma and image-size limits."""
    radius = min(math.ceil(3 * sigma), images.shape[-2] - 1, images.shape[-1] - 1)
    if radius < 1:
        return images.clone()
    coordinate = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-coordinate.square() / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    channels = images.shape[1]
    weights = kernel_2d.expand(channels, 1, -1, -1)
    padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, weights, groups=channels).clamp(0, 1)
