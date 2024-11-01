import torch
import torch.nn as nn
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FGSM:
    """Fast Gradient Sign Method attack implementation."""

    def __init__(
            self,
            epsilon: float = 0.03,
            clip_min: float = 0.0,
            clip_max: float = 1.0
    ):
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(
            self,
            model: nn.Module,
            data: torch.Tensor,
            labels: torch.Tensor,
            targeted: bool = False
    ) -> torch.Tensor:
        """Generate adversarial examples using FGSM."""
        data.requires_grad = True

        # Forward pass
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Compute gradients
        loss.backward()

        # Create perturbation
        sign = -1 if targeted else 1
        perturbed = data + sign * self.epsilon * data.grad.sign()

        # Clip to valid range
        perturbed = torch.clamp(perturbed, self.clip_min, self.clip_max)

        return perturbed.detach()