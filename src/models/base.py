from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """Base class for all models."""

    def __init__(self):
        super().__init__()
        self.training_history: Dict[str, list] = {
            'loss': [],
            'accuracy': []
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def save_checkpoint(
            self,
            path: str,
            optimizer: Optional[torch.optim.Optimizer] = None,
            epoch: int = 0
    ) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'training_history': self.training_history,
                'epoch': epoch
            }
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(
            self,
            path: str,
            optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint['training_history']

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Checkpoint loaded from {path}")
            return checkpoint

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return {}