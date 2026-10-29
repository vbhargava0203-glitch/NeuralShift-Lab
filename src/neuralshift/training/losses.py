"""The fixed training objective, with logits as input."""

from torch.nn import CrossEntropyLoss


def classification_loss():
    return CrossEntropyLoss()
