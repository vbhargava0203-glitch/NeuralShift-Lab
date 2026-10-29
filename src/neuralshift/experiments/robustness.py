"""A fixed corruption plan and deterministic per-source-example random seeds."""

import torch

from neuralshift.corruptions.registry import apply_corruption


def conditions(config):
    return [("clean", 0)] + [(name, severity) for name in config["evaluation"]["corruptions"]
                             for severity in config["evaluation"]["severities"]]


def corrupt_batch(images, ids, name, severity, seed):
    if name == "clean":
        return images.clone()
    # Source-index seeds make Gaussian noise invariant to batch size and workers.
    return torch.cat([apply_corruption(image[None], name, severity, seed=(seed + int(index) * 1009 + severity * 9176) % 2**32)
                      for image, index in zip(images, ids, strict=True)])
