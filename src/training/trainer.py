import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from ..models import BaseModel
from ..defenses import AdversarialTraining
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Handles model training with adversarial defense integration."""

    def __init__(
            self,
            model: BaseModel,
            defense: Optional[AdversarialTraining] = None,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.defense = defense
        self.device = device
        self.callbacks = []
        self.history = {'train': [], 'val': []}

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            callbacks: Optional[List] = None
    ) -> Dict:
        """Training loop with optional validation."""
        try:
            self.callbacks = callbacks or []

            for epoch in range(epochs):
                # Training phase
                train_metrics = self._train_epoch(train_loader, optimizer)
                self.history['train'].append(train_metrics)

                # Validation phase
                if val_loader:
                    val_metrics = self._validate(val_loader)
                    self.history['val'].append(val_metrics)

                # Handle callbacks
                stop_training = self._handle_callbacks(epoch, train_metrics, val_metrics)
                if stop_training:
                    break

            return self.history

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return {}

    def _train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer
    ) -> Dict:
        """Run one epoch of training."""
        self.model.train()
        metrics = {'loss': 0.0, 'accuracy': 0.0}

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Generate adversarial examples if defense is active
            if self.defense:
                data = self.defense.generate_adv_examples(data, target)

            optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics
            metrics['loss'] += loss.item()
            metrics['accuracy'] += (output.argmax(dim=1) == target).float().mean().item()

        # Average metrics
        for key in metrics:
            metrics[key] /= len(train_loader)

        return metrics

    def _validate(
            self,
            val_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """Validate the model."""
        self.model.eval()
        metrics = {'loss': 0.0, 'accuracy': 0.0}

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Compute metrics
                loss = nn.CrossEntropyLoss()(output, target)
                metrics['loss'] += loss.item()
                metrics['accuracy'] += (output.argmax(dim=1) == target).float().mean().item()

        # Average metrics
        for key in metrics:
            metrics[key] /= len(val_loader)

        return metrics

    def _handle_callbacks(
            self,
            epoch: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict]
    ) -> bool:
        """Execute callbacks and check for early stopping."""
        stop_training = False

        for callback in self.callbacks:
            stop = callback(self.model, epoch, train_metrics, val_metrics)
            if stop:
                stop_training = True
                break

        return stop_training