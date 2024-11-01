import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Detector(BaseModel):
    """Neural network for detecting adversarial examples."""

    def __init__(
            self,
            in_channels: int = 3,
            feature_extractor: Optional[nn.Module] = None
    ):
        super().__init__()

        self.feature_extractor = feature_extractor or self._create_feature_extractor(in_channels)

        self.detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both features and detection probability.

        Returns:
            features: Extracted features
            detection_prob: Probability of being adversarial
        """
        features = self.feature_extractor(x)
        features = features.mean([2, 3])  # Global average pooling
        detection_prob = self.detector(features)
        return features, detection_prob

    def _create_feature_extractor(self, in_channels: int) -> nn.Module:
        """Create a simple feature extractor."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )