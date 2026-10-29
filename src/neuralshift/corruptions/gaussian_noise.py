"""Additive independent Gaussian pixel noise, clipped to [0, 1]."""

import torch


def gaussian_noise(images: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Noise is sampled on CPU for repeatability across available devices."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    return (images + sigma * noise.to(images.device)).clamp(0, 1)
