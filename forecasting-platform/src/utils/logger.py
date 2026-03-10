"""Logging utilities."""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    elif not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger
