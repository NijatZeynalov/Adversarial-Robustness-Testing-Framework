import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .logger import get_logger

logger = get_logger(__name__)


def validate_model(model: nn.Module) -> bool:
    """Validate model architecture and parameters."""
    try:
        # Check if model is on correct device
        device = next(model.parameters()).device
        if not torch.cuda.is_available() and device.type == 'cuda':
            logger.warning("Model on CUDA but CUDA not available")
            return False

        # Check if model has parameters
        if sum(p.numel() for p in model.parameters()) == 0:
            logger.error("Model has no parameters")
            return False

        # Basic forward pass test
        try:
            model.eval()
            with torch.no_grad():
                x = torch.randn(1, 3, 32, 32).to(device)
                _ = model(x)
        except Exception as e:
            logger.error(f"Model forward pass failed: {str(e)}")
            return False

        return True

    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


def validate_data(
        data: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        value_range: Optional[Tuple[float, float]] = (0, 1)
) -> bool:
    """Validate input data."""
    try:
        # Check tensor basics
        if not isinstance(data, torch.Tensor):
            logger.error("Input is not a torch.Tensor")
            return False

        # Check shape
        if expected_shape and data.shape[1:] != expected_shape[1:]:
            logger.error(
                f"Invalid shape: {data.shape}, expected {expected_shape}"
            )
            return False

        # Check value range
        if value_range:
            min_val, max_val = value_range
            if data.min() < min_val or data.max() > max_val:
                logger.error(
                    f"Values outside range [{min_val}, {max_val}]"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False


def validate_attack_params(
        epsilon: float,
        steps: Optional[int] = None,
        step_size: Optional[float] = None
) -> bool:
    """Validate attack parameters."""
    try:
        # Check epsilon
        if epsilon <= 0 or epsilon > 1:
            logger.error("Epsilon must be in (0, 1]")
            return False

        # Check steps if provided
        if steps is not None and steps <= 0:
            logger.error("Steps must be positive")
            return False

        # Check step size if provided
        if step_size is not None:
            if step_size <= 0 or step_size > epsilon:
                logger.error("Invalid step size")
                return False

        return True

    except Exception as e:
        logger.error(f"Attack parameter validation failed: {str(e)}")
        return False


def validate_batch(
        data: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module
) -> bool:
    """Validate a batch of data and labels."""
    try:
        # Check device consistency
        model_device = next(model.parameters()).device
        if data.device != model_device or labels.device != model_device:
            logger.error("Device mismatch between data and model")
            return False

        # Check label validity
        if labels.min() < 0:
            logger.error("Negative label values found")
            return False

        # Check shape consistency
        if len(data) != len(labels):
            logger.error("Batch size mismatch between data and labels")
            return False

        return True

    except Exception as e:
        logger.error(f"Batch validation failed: {str(e)}")
        return False


class ValidationError(Exception):
    """ will add later"""
    pass