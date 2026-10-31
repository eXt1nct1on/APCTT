"""Disk cache for validated LLM JSON responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    """Content-addressed JSON cache stored below the configured data directory."""

    def __init__(self, data_dir: str | Path = "data", enabled: bool = True) -> None:
        """Create a cache rooted at ``DATA_DIR/cache``."""
        self.root = Path(data_dir) / "cache"
        self.enabled = enabled

    @staticmethod
    def key(
        model: str,
        system: str,
        user: str,
        temperature: float,
        schema_name: str,
        sample_idx: int,
    ) -> str:
        """Return the stable cache key for one completion request."""
        payload = json.dumps(
            [model, system, user, temperature, schema_name, sample_idx],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Read a cached JSON object, returning ``None`` for a cache miss."""
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Atomically write a JSON response to the cache."""
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
