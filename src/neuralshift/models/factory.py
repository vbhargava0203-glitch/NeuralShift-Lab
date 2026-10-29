"""Single architecture entry point for training and checkpoint restoration."""

from torch import nn

from .cnn import SmallCNN
from .resnet import small_resnet18


def build_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 10,
    dropout: float = 0.25,
    batch_norm: bool = True,
) -> nn.Module:
    """Build an untrained CNN or ResNet-18; reject unknown architecture names."""
    if name == "cnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout, batch_norm=batch_norm)
    if name == "resnet18":
        return small_resnet18(in_channels=in_channels, num_classes=num_classes, dropout=dropout, batch_norm=batch_norm)
    raise ValueError(f"Unknown architecture {name!r}; choose 'cnn' or 'resnet18'")
