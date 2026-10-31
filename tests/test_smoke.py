"""Smoke tests for the project scaffold."""

import importlib


def test_subpackages_import() -> None:
    """All top-level Python packages are importable."""
    for package in (
        "trl_scout",
        "trl_scout.ingest",
        "trl_scout.index",
        "trl_scout.llm",
        "trl_scout.pipeline",
        "trl_scout.report",
    ):
        importlib.import_module(package)
