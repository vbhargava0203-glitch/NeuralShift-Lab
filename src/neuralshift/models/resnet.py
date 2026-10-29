"""A small-image ResNet-18, initialized from scratch."""

from torch import nn
from torchvision.models import resnet18


def _identity_norm(channels: int) -> nn.Module:
    return nn.Identity()


def small_resnet18(
    in_channels: int = 1,
    num_classes: int = 10,
    dropout: float = 0.25,
    batch_norm: bool = True,
) -> nn.Module:
    """Replace the ImageNet stem by 3x3/stride-1; remove its max pool.

    A dropout layer before the final classifier enables MC dropout evaluation.
    This provides head-only stochasticity, unlike the CNN's multiple dropout
    layers: comparisons must acknowledge this architectural difference.
    No pretrained weights are downloaded or used.
    """
    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")
    if num_classes < 2 or not 0 <= dropout < 1:
        raise ValueError("num_classes must be >=2 and dropout must be in [0,1)")
    norm_layer = nn.BatchNorm2d if batch_norm else _identity_norm
    model = resnet18(weights=None, num_classes=num_classes, norm_layer=norm_layer)
    model.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    return model
