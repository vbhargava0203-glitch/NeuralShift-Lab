"""Add a positive offset to image intensity, then clip."""

import torch


def brightness(images: torch.Tensor, offset: float) -> torch.Tensor:
    return (images + offset).clamp(0, 1)
