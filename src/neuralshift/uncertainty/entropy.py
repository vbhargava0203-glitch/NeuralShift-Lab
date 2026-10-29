"""Shannon predictive entropy, in nats unless normalized."""

import math

import torch


def entropy(prob: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    if prob.shape[-1] < 2:
        raise ValueError("Entropy expects at least two classes")
    # xlogy defines 0 log 0 = 0 without introducing NaNs into the result.
    value = -torch.special.xlogy(prob, prob).sum(dim=-1)
    return value / math.log(prob.shape[-1]) if normalized else value
