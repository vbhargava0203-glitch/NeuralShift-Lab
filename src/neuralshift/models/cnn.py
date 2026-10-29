"""A small convolutional baseline with explicit stochastic regularization."""

from torch import Tensor, nn


class SmallCNN(nn.Module):
    """Two 3x3 blocks (32, 64 channels), 2x2 pooling, and a 128-unit head.

    Batch normalization is optional. Dropout follows each pooling operation and
    the hidden fully connected layer. Adaptive pooling accepts 28px and 32px
    images without architecture changes. All parameters start untrained.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.25,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if num_classes < 2 or not 0 <= dropout < 1:
            raise ValueError("num_classes must be >=2 and dropout must be in [0,1)")
        blocks: list[nn.Module] = []
        previous = in_channels
        for width in (32, 64):
            blocks.append(nn.Conv2d(previous, width, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(width))
            blocks.extend([nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout)])
            previous = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class logits for a BCHW tensor."""
        return self.classifier(self.features(inputs))
