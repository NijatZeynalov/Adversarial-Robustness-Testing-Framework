from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Model Settings
    MODEL_NAME: str = Field(
        default="resnet18",
        description="Name of the model architecture"
    )
    MODEL_CHECKPOINT: Optional[Path] = Field(
        default=None,
        description="Path to model checkpoint"
    )
    DEVICE: str = Field(
        default="cuda",
        description="Device to use (cuda/cpu)"
    )

    # Attack Settings
    FGSM_EPSILON: float = Field(
        default=0.03,
        description="FGSM perturbation size"
    )
    PGD_EPSILON: float = Field(
        default=0.03,
        description="PGD perturbation size"
    )
    PGD_STEPS: int = Field(
        default=40,
        description="Number of PGD steps"
    )
    PGD_ALPHA: float = Field(
        default=0.01,
        description="PGD step size"
    )
    CW_CONFIDENCE: float = Field(
        default=1.0,
        description="CW confidence parameter"
    )
    CW_LEARNING_RATE: float = Field(
        default=0.01,
        description="CW optimization learning rate"
    )
    CW_BINARY_STEPS: int = Field(
        default=9,
        description="CW binary search steps"
    )

    # Defense Settings
    ADV_TRAINING_RATIO: float = Field(
        default=0.5,
        description="Ratio of adversarial examples in training"
    )
    INPUT_TRANSFORM_QUALITY: int = Field(
        default=75,
        description="JPEG compression quality"
    )
    DETECTION_THRESHOLD: float = Field(
        default=0.5,
        description="Adversarial detection threshold"
    )

    # Training Settings
    BATCH_SIZE: int = Field(
        default=128,
        description="Training batch size"
    )
    LEARNING_RATE: float = Field(
        default=0.001,
        description="Initial learning rate"
    )
    NUM_EPOCHS: int = Field(
        default=100,
        description="Number of training epochs"
    )
    EARLY_STOPPING_PATIENCE: int = Field(
        default=10,
        description="Early stopping patience"
    )

    # Paths and Logging
    OUTPUT_DIR: Path = Field(
        default=Path("outputs"),
        description="Output directory"
    )
    CHECKPOINT_DIR: Path = Field(
        default=Path("checkpoints"),
        description="Model checkpoint directory"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    def setup_directories(self):
        """Create necessary directories."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Initialize settings
settings = Settings()
settings.setup_directories()