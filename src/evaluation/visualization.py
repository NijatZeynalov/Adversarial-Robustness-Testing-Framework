import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Visualization tools for adversarial examples and model behavior."""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir

    def plot_adversarial_examples(
            self,
            clean_data: torch.Tensor,
            adv_data: torch.Tensor,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            indices: Optional[List[int]] = None,
            save_name: Optional[str] = None
    ) -> None:
        """Plot clean and adversarial examples side by side."""
        try:
            if indices is None:
                indices = list(range(min(5, len(clean_data))))

            n_samples = len(indices)
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

            for idx, i in enumerate(indices):
                # Plot clean image
                axes[idx, 0].imshow(
                    clean_data[i].cpu().permute(1, 2, 0)
                )
                axes[idx, 0].set_title(f'Clean\nTrue: {labels[i].item()}')

                # Plot adversarial image
                axes[idx, 1].imshow(
                    adv_data[i].cpu().permute(1, 2, 0)
                )
                axes[idx, 1].set_title(
                    f'Adversarial\nPred: {predictions[i].item()}'
                )

                # Plot perturbation
                perturbation = (adv_data[i] - clean_data[i]).cpu()
                axes[idx, 2].imshow(
                    perturbation.permute(1, 2, 0)
                )
                axes[idx, 2].set_title('Perturbation')

            plt.tight_layout()
            if save_name and self.save_dir:
                plt.savefig(f"{self.save_dir}/{save_name}")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting examples: {str(e)}")

    def plot_metrics(
            self,
            metrics: Dict[str, List[float]],
            save_name: Optional[str] = None
    ) -> None:
        """Plot training/evaluation metrics."""
        try:
            n_metrics = len(metrics)
            fig, axes = plt.subplots(
                (n_metrics + 1) // 2, 2,
                figsize=(12, 4 * ((n_metrics + 1) // 2))
            )
            axes = axes.flatten()

            for idx, (metric_name, values) in enumerate(metrics.items()):
                axes[idx].plot(values)
                axes[idx].set_title(metric_name)
                axes[idx].set_xlabel('Iteration')
                axes[idx].grid(True)

            plt.tight_layout()
            if save_name and self.save_dir:
                plt.savefig(f"{self.save_dir}/{save_name}")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")