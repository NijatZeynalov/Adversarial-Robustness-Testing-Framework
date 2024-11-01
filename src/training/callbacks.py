import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
            self,
            monitor: str = 'val_loss',
            min_delta: float = 0,
            patience: int = 5,
            mode: str = 'min'
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def __call__(
            self,
            model: nn.Module,
            epoch: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict]
    ) -> bool:
        """Check if training should stop."""
        metrics = val_metrics if val_metrics else train_metrics
        current = metrics.get(self.monitor.replace('val_', ''), None)

        if current is None:
            return False

        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True

        return False


class ModelCheckpoint:
    """Model checkpoint callback."""

    def __init__(
            self,
            filepath: str,
            monitor: str = 'val_loss',
            mode: str = 'min',
            save_best_only: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def __call__(
            self,
            model: nn.Module,
            epoch: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict]
    ) -> bool:
        """Save model checkpoint."""
        metrics = val_metrics if val_metrics else train_metrics
        current = metrics.get(self.monitor.replace('val_', ''), None)

        if current is None:
            return False

        if self.mode == 'min':
            improved = current < self.best_value
        else:
            improved = current > self.best_value

        if improved or not self.save_best_only:
            self.best_value = current
            try:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics
                    },
                    self.filepath
                )
                logger.info(f"Saved checkpoint at epoch {epoch}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")

        return False


class LRScheduler:
    """Learning rate scheduler callback."""

    def __init__(
            self,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            monitor: Optional[str] = None
    ):
        self.scheduler = scheduler
        self.monitor = monitor

    def __call__(
            self,
            model: nn.Module,
            epoch: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict]
    ) -> bool:
        """Update learning rate."""
        if self.monitor:
            metrics = val_metrics if val_metrics else train_metrics
            value = metrics.get(self.monitor.replace('val_', ''), None)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()

        return False