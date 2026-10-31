"""Tests for LLM retries, caching, fallback, and concurrency."""

import asyncio

import pytest
from pydantic import BaseModel

from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient, gather_limited, load_prompt


class Result(BaseModel):
    """Test response schema."""

    ok: bool


def test_invalid_json_retries_and_validates(tmp_path) -> None:
    """Invalid JSON is repaired on a subsequent attempt."""
    fake = FakeLLM({"Result": ['{"ok": "not-a-bool"}', '{"ok": true}']})
    client = LLMClient(completion=fake, cache=JsonCache(tmp_path), backoff_seconds=0)
    assert client.complete_json("system", "user", Result).ok is True
    assert fake.calls == 2


def test_cache_hit(tmp_path) -> None:
    """A second identical request avoids the backend."""
    fake = FakeLLM({"Result": {"ok": True}})
    client = LLMClient(completion=fake, cache=JsonCache(tmp_path))
    client.complete_json("system", "user", Result)
    client.complete_json("system", "user", Result)
    assert fake.calls == 1
    assert client.stats.cache_hits == 1


def test_fallback_switch() -> None:
    """Transient provider errors eventually use the fallback model."""
    calls: list[str] = []

    def backend(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise RuntimeError("503 service unavailable")
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client = LLMClient(
        completion=backend,
        model="primary",
        fallback_model="fallback",
        cache=JsonCache("data-test-no-cache", enabled=False),
        max_retries=0,
        backoff_seconds=0,
    )
    assert client.complete_json("system", "user", Result).ok
    assert calls == ["primary", "fallback"]


def test_gather_limited() -> None:
    """The semaphore prevents more than the configured concurrent tasks."""
    active = 0
    maximum = 0

    async def task() -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return 1

    assert asyncio.run(gather_limited([task() for _ in range(6)], limit=2)) == [1] * 6
    assert maximum == 2


def test_prompt_loader_requires_variables() -> None:
    """The loader rejects templates with missing substitutions."""
    with pytest.raises(ValueError, match="missing variables"):
        load_prompt("identify")
