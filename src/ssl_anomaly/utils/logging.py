"""Logging helpers — stdlib + rich when available, graceful fallback."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "ssl_anomaly", *, log_file: Path | None = None, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times; configures once."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — use rich if present
    try:
        from rich.logging import RichHandler

        ch: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        ch.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    except ImportError:  # pragma: no cover
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger
