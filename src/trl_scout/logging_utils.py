"""Logging helpers for console and run-specific file output."""

import logging
import re
from pathlib import Path

_SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|token|password|secret)\s*[:=]\s*['\"]?([A-Za-z0-9._-]{8,})"
)


class _RedactionFilter(logging.Filter):
    """Redact common credential-shaped values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _SECRET_RE.sub(r"\1=[REDACTED]", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(run_id: str | None = None, data_dir: str = "data") -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("trl_scout")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        console.addFilter(_RedactionFilter())
        logger.addHandler(console)

    if run_id and not any(
        getattr(handler, "trl_run_id", None) == run_id for handler in logger.handlers
    ):
        run_dir = Path(data_dir) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        file_handler.addFilter(_RedactionFilter())
        file_handler.trl_run_id = run_id
        logger.addHandler(file_handler)

    return logger
