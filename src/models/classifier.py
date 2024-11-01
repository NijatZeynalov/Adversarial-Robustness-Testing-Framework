import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Classifier(BaseModel):
    """Basic CNN classifier with customizable architecture."""

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            architecture: str = 'simple'
    ):
        super().__init__()
        self.architecture = architecture

        if architecture == 'simple':
            self.features = nn.Sequential(
                # Conv Block 1
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Conv Block 2
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Conv Block 3
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )
        elif architecture == 'resnet':
            self._init_resnet(in_channels, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _init_resnet(
            self,
            in_channels: int,
            num_classes: int
    ) -> None:
        """Initialize ResNet-style architecture."""
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Residual blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def _make_layer(
            self,
            in_channels: int,
            out_channels: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        """Create ResNet layer with specified number of blocks."""
        layers = []

        # First block with possible stride
        layers.append(ResBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out