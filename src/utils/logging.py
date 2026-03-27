"""
Structured logging setup.

Every module calls `get_logger(__name__)` to get a logger scoped to that module.
Output goes to both console and a log file. Format includes timestamps so we can
trace what happened during long evaluation runs.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Get a logger with console output and optional file output.

    Args:
        name: Usually __name__ from the calling module.
        log_file: Optional path to also write logs to a file.
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they already exist (avoids duplicates on repeated calls)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — DEBUG and above (captures everything)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
