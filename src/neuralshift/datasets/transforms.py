"""Fixed conventional normalization constants, applied after corruption."""

import torch

STATS = {"fashion_mnist": ((0.2860,), (0.3530,)),
         "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}


def normalize(images, dataset):
    mean, std = STATS[dataset]
    shape = (1, -1, 1, 1) if images.ndim == 4 else (-1, 1, 1)
    return (images - torch.tensor(mean, device=images.device).reshape(shape)) / torch.tensor(std, device=images.device).reshape(shape)
