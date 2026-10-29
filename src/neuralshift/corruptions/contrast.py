"""Reduce contrast around each image's per-channel spatial mean."""

import torch


def contrast(images: torch.Tensor, factor: float) -> torch.Tensor:
    mean = images.mean(dim=(-2, -1), keepdim=True)
    return (mean + factor * (images - mean)).clamp(0, 1)
