import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CarliniWagner:
    """Carlini-Wagner L2 attack implementation."""

    def __init__(
            self,
            confidence: float = 0.0,
            learning_rate: float = 0.01,
            binary_search_steps: int = 9,
            max_iterations: int = 1000
    ):
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations

    def generate(
            self,
            model: nn.Module,
            data: torch.Tensor,
            labels: torch.Tensor,
            targeted: bool = False
    ) -> torch.Tensor:
        """Generate adversarial examples using C&W attack."""
        batch_size = len(data)
        w = torch.zeros_like(data, requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)

        best_adv = data.clone()
        best_l2 = torch.ones(batch_size).to(data.device) * 1e10

        # Binary search for c
        c = torch.ones(batch_size).to(data.device)
        lower_bound = torch.zeros(batch_size).to(data.device)
        upper_bound = torch.ones(batch_size).to(data.device) * 1e10

        for _ in range(self.binary_search_steps):
            for _ in range(self.max_iterations):
                optimizer.zero_grad()

                adv = torch.tanh(w) * 0.5 + 0.5
                l2_loss = torch.sum((adv - data) ** 2, dim=[1, 2, 3])

                outputs = model(adv)
                target_loss = self._f(outputs, labels, targeted)

                loss = l2_loss + c * target_loss
                loss.sum().backward()
                optimizer.step()

                # Update best results
                mask = l2_loss < best_l2
                best_l2[mask] = l2_loss[mask]
                best_adv[mask] = adv[mask]

            # Update c
            c = self._update_c(c, upper_bound, lower_bound, target_loss)

        return best_adv.detach()

    def _f(
            self,
            outputs: torch.Tensor,
            labels: torch.Tensor,
            targeted: bool
    ) -> torch.Tensor:
        """Calculate f6 loss function."""
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        if targeted:
            return torch.max(
                (1 - one_hot) * outputs - one_hot * outputs,
                -self.confidence
            )
        else:
            return torch.max(
                one_hot * outputs - (1 - one_hot) * outputs,
                -self.confidence
            )

    def _update_c(
            self,
            c: torch.Tensor,
            upper: torch.Tensor,
            lower: torch.Tensor,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """Update constant c using binary search."""
        success = loss <= 0
        upper[success] = torch.min(upper[success], c[success])
        lower[~success] = torch.max(lower[~success], c[~success])

        c[success] = (lower[success] + upper[success]) / 2
        c[~success] = upper[~success]

        return c