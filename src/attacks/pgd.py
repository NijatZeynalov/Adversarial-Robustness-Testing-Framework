import torch
import torch.nn as nn
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PGD:
    """Projected Gradient Descent attack implementation."""

    def __init__(
            self,
            epsilon: float = 0.03,
            steps: int = 40,
            alpha: float = 0.01,
            random_start: bool = True
    ):
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start

    def generate(
            self,
            model: nn.Module,
            data: torch.Tensor,
            labels: torch.Tensor,
            targeted: bool = False
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        x = data.clone()

        if self.random_start:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        for _ in range(self.steps):
            x.requires_grad = True
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            grad = torch.autograd.grad(loss, [x])[0]
            sign = -1 if targeted else 1
            x = x.detach() + sign * self.alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(x - data, -self.epsilon, self.epsilon)
            x = torch.clamp(data + delta, 0, 1)

        return x.detach()