"""Structured logging configuration.

Provides a pre-configured module-level logger that respects
the application's log level from settings.
"""

import logging
import sys

from enterprise_kb.config import settings


def setup_logger(name: str = __name__.split(".")[0]) -> logging.Logger:
    """Create and return a structured logger for the given name.

    Args:
        name: Logger name, typically ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.log_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    logger.setLevel(settings.log_level)
    logger.propagate = False

    return logger


logger = setup_logger()
