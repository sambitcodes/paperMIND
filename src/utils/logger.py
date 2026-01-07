"""
Central logging configuration.

Keeps logging consistent across src/ and streamlit_app/.
"""
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create (or fetch) a logger with a consistent formatter.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_root_logging(level: str = "INFO") -> None:
    """
    Configure root logging for scripts/notebooks.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
