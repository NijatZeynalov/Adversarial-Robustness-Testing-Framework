import torch
import torch.nn as nn
from typing import Optional, Dict
from ..attacks import FGSM, PGD
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdversarialTraining:
    """Implements adversarial training defense."""

    def __init__(
            self,
            model: nn.Module,
            attack_method: str = 'pgd',
            eps: float = 0.03,
            ratio: float = 0.5,
            **attack_params
    ):
        self.model = model
        self.ratio = ratio

        # Setup attack
        if attack_method == 'pgd':
            self.attack = PGD(epsilon=eps, **attack_params)
        elif attack_method == 'fgsm':
            self.attack = FGSM(epsilon=eps, **attack_params)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")

    def generate_adv_examples(
            self,
            data: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples for training."""
        batch_size = len(data)
        num_adv = int(batch_size * self.ratio)

        if num_adv == 0:
            return data

        adv_data = data[:num_adv]
        adv_labels = labels[:num_adv]

        with torch.enable_grad():
            adv_examples = self.attack.generate(
                self.model,
                adv_data,
                adv_labels
            )

        return torch.cat([adv_examples, data[num_adv:]])

    def train_step(
            self,
            data: torch.Tensor,
            labels: torch.Tensor,
            optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one adversarial training step."""
        self.model.train()
        mixed_data = self.generate_adv_examples(data, labels)

        optimizer.zero_grad()
        outputs = self.model(mixed_data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'accuracy': (outputs.argmax(1) == labels).float().mean().item()
        }