"""Logging helpers for console and run-specific file output."""

import logging
from pathlib import Path


def configure_logging(run_id: str | None = None, data_dir: str = "data") -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("trl_scout")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(console)

    if run_id and not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        run_dir = Path(data_dir) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

    return logger
