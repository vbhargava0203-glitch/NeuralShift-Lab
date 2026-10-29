"""Maximum softmax probability uncertainty baseline."""

import math

import torch


def probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and strictly positive")
    if logits.ndim < 2 or logits.shape[-1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits with at least two classes")
    return torch.softmax(logits / temperature, dim=-1)


def msp(prob: torch.Tensor) -> torch.Tensor:
    """Return uncertainty 1 - max probability; higher means less reliable."""
    return 1 - prob.max(dim=-1).values
