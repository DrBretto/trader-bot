"""Logging utilities for the investment system."""

import logging
from datetime import datetime
from typing import Optional


def setup_logger(name: str = 'investment_system') -> logging.Logger:
    """Set up a logger with standard formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def log_step(step_num: int, total_steps: int, message: str,
             logger: Optional[logging.Logger] = None) -> None:
    """Log a pipeline step with consistent formatting."""
    if logger is None:
        logger = setup_logger()

    logger.info(f"[{step_num}/{total_steps}] {message}")


class StepTimer:
    """Context manager for timing pipeline steps."""

    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None):
        self.step_name = step_name
        self.logger = logger or setup_logger()
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.step_name} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.step_name} ({duration:.2f}s) - {exc_val}")
        return False
