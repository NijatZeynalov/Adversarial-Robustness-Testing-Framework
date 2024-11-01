import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdversarialDetector:
    """Detector for adversarial examples using multiple methods."""

    def __init__(
            self,
            methods: list = ['statistical', 'kernel_density', 'feature_squeezing'],
            threshold: float = 0.5
    ):
        self.methods = methods
        self.threshold = threshold
        self.stats = {}

    def detect(
            self,
            x: torch.Tensor,
            model: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Detect adversarial examples using multiple methods."""
        scores = torch.zeros(len(x)).to(x.device)
        method_scores = {}

        for method in self.methods:
            curr_scores = self._apply_method(x, model, method)
            method_scores[method] = curr_scores.mean().item()
            scores += curr_scores

        scores /= len(self.methods)
        predictions = scores > self.threshold

        return predictions, method_scores

    def _apply_method(
            self,
            x: torch.Tensor,
            model: nn.Module,
            method: str
    ) -> torch.Tensor:
        """Apply specific detection method."""
        if method == 'statistical':
            return self._statistical_detection(x)
        elif method == 'kernel_density':
            return self._kernel_density_detection(x, model)
        elif method == 'feature_squeezing':
            return self._feature_squeezing_detection(x, model)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _statistical_detection(self, x: torch.Tensor) -> torch.Tensor:
        """Detect using statistical properties."""
        # Calculate local statistics
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        var = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=1) - mean ** 2

        # Calculate abnormality scores
        scores = torch.abs(x - mean) / (torch.sqrt(var) + 1e-8)
        return scores.mean(dim=[1, 2, 3])

    def _kernel_density_detection(
            self,
            x: torch.Tensor,
            model: nn.Module
    ) -> torch.Tensor:
        """Detect using kernel density estimation."""
        features = self._get_features(model, x)
        if not hasattr(self, 'clean_features'):
            self.clean_features = features
            return torch.zeros(len(x)).to(x.device)

        scores = []
        for feat in features:
            distances = torch.cdist(
                feat.unsqueeze(0),
                self.clean_features
            )
            density = torch.exp(-distances).mean()
            scores.append(density)

        return torch.tensor(scores).to(x.device)

    def _feature_squeezing_detection(
            self,
            x: torch.Tensor,
            model: nn.Module
    ) -> torch.Tensor:
        """Detect using feature squeezing."""
        # Get predictions on original input
        orig_preds = model(x).softmax(dim=1)

        # Get predictions on squeezed input
        x_squeezed = torch.round(x * 255) / 255
        squeezed_preds = model(x_squeezed).softmax(dim=1)

        # Calculate L1 distance between predictions
        scores = torch.abs(orig_preds - squeezed_preds).sum(dim=1)
        return scores

    @staticmethod
    def _get_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from model."""
        features = []
        hooks = []

        def hook(module, input, output):
            features.append(output.flatten(start_dim=1))

        # Register hooks for feature extraction
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook))

        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return torch.cat(features, dim=1)