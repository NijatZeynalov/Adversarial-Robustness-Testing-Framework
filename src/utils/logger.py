import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',  # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',  # Red
        'CRITICAL': '\033[0;35m',  # Purple
        'RESET': '\033[0m'
    }

    def format(self, record):
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            record.levelname = (
                f"{self.COLORS.get(record.levelname, '')}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def get_logger(
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO"
) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create log file: {str(e)}")

    return logger