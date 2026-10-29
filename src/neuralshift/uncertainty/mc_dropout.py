"""Stochastic dropout inference with frozen batch-normalization statistics."""

import torch
from torch import nn

from .entropy import entropy


def mc_predict(model: nn.Module, x: torch.Tensor, passes: int = 30) -> dict:
    """Return T x B x K predictions and uncertainty summaries.

    Mutual information is H(mean p) - mean H(p), an epistemic proxy under
    the approximate dropout posterior, not a calibrated Bayesian guarantee.
    Variance is population variance across passes, averaged over classes.
    Calling code controls torch's random seed and the batching protocol.
    """
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 2:
        raise ValueError("MC dropout requires at least two stochastic passes")
    modules = list(model.modules())
    previous = [module.training for module in modules]
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    if not any(isinstance(module, dropout_types) and module.p > 0 for module in modules):
        raise ValueError("MC dropout requires a model with active nonzero dropout")
    try:
        model.eval()
        for module in modules:
            if isinstance(module, dropout_types):
                module.train()
        with torch.inference_mode():
            samples = torch.stack([model(x).softmax(dim=-1) for _ in range(passes)])
            mean = samples.mean(dim=0)
            predictive = entropy(mean)
            expected = entropy(samples).mean(dim=0)
            return {
                "probabilities": samples,
                "mean": mean,
                "predictive_entropy": predictive,
                "expected_entropy": expected,
                "mutual_information": (predictive - expected).clamp_min(0),
                "variance": samples.var(dim=0, unbiased=False).mean(dim=-1),
            }
    finally:
        # Direct assignment preserves heterogeneous original child states.
        for module, training in zip(modules, previous, strict=True):
            module.training = training
