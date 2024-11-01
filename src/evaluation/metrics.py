import torch
import numpy as np
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RobustnessMetrics:
    """Computes various robustness metrics for model evaluation."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def compute_metrics(
            self,
            clean_data: torch.Tensor,
            adv_data: torch.Tensor,
            labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive robustness metrics."""
        try:
            metrics = {}

            # Accuracy metrics
            clean_acc = self._compute_accuracy(clean_data, labels)
            adv_acc = self._compute_accuracy(adv_data, labels)

            # Perturbation metrics
            l2_dist = self._compute_l2_distance(clean_data, adv_data)
            linf_dist = self._compute_linf_distance(clean_data, adv_data)

            # Success rate
            success_rate = self._compute_attack_success_rate(
                clean_data, adv_data, labels
            )

            metrics.update({
                'clean_accuracy': clean_acc,
                'robust_accuracy': adv_acc,
                'accuracy_drop': clean_acc - adv_acc,
                'attack_success_rate': success_rate,
                'avg_l2_perturbation': l2_dist,
                'avg_linf_perturbation': linf_dist
            })

            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}

    def _compute_accuracy(
            self,
            data: torch.Tensor,
            labels: torch.Tensor
    ) -> float:
        """Compute model accuracy."""
        with torch.no_grad():
            outputs = self.model(data)
            predictions = outputs.argmax(dim=1)
            return (predictions == labels).float().mean().item()

    def _compute_l2_distance(
            self,
            clean_data: torch.Tensor,
            adv_data: torch.Tensor
    ) -> float:
        """Compute average L2 perturbation."""
        with torch.no_grad():
            diff = (clean_data - adv_data).view(len(clean_data), -1)
            l2_dist = torch.norm(diff, p=2, dim=1).mean().item()
            return l2_dist

    def _compute_linf_distance(
            self,
            clean_data: torch.Tensor,
            adv_data: torch.Tensor
    ) -> float:
        """Compute average L∞ perturbation."""
        with torch.no_grad():
            diff = (clean_data - adv_data).view(len(clean_data), -1)
            linf_dist = torch.norm(diff, p=float('inf'), dim=1).mean().item()
            return linf_dist

    def _compute_attack_success_rate(
            self,
            clean_data: torch.Tensor,
            adv_data: torch.Tensor,
            labels: torch.Tensor
    ) -> float:
        """Compute attack success rate."""
        with torch.no_grad():
            clean_pred = self.model(clean_data).argmax(dim=1)
            adv_pred = self.model(adv_data).argmax(dim=1)

            # Only count initially correct predictions
            correct_mask = (clean_pred == labels)
            success_rate = (
                (adv_pred != labels)[correct_mask].float().mean().item()
            )
            return success_rate