"""LiteLLM JSON client with validation, retries, caching, and async helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from trl_scout.config import settings
from trl_scout.llm.cache import JsonCache
from trl_scout.schemas import RunStats

logger = logging.getLogger("trl_scout.llm")
ModelT = TypeVar("ModelT", bound=BaseModel)
Completion = Callable[..., Any]


def load_prompt(name: str, variables: dict[str, str] | None = None) -> tuple[str, str]:
    """Load a prompt template and substitute every required variable."""
    path = Path(__file__).parent / "prompts" / name
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    text = path.read_text(encoding="utf-8")
    if "---USER---" not in text:
        raise ValueError(f"prompt {path.name} must contain a ---USER--- separator")
    system, user = text.split("---USER---", maxsplit=1)
    values = variables or {}
    placeholders = set(re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", text))
    missing = sorted(placeholders - values.keys())
    if missing:
        raise ValueError(f"prompt {path.name} is missing variables: {', '.join(missing)}")
    for key, value in values.items():
        system = system.replace(f"{{{{{key}}}}}", value).replace(f"{{{{ {key} }}}}", value)
        user = user.replace(f"{{{{{key}}}}}", value).replace(f"{{{{ {key} }}}}", value)
    return system.strip(), user.strip()


def _json_text(content: Any) -> str:
    """Extract JSON text and remove optional Markdown fences."""
    if isinstance(content, dict):
        return json.dumps(content)
    if not isinstance(content, str):
        raise ValueError("LLM response content was not text or JSON")
    value = content.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _response_content(response: Any) -> tuple[Any, int, int]:
    """Extract content and available token usage from LiteLLM response shapes."""
    choices = (
        response.get("choices")
        if isinstance(response, dict)
        else getattr(response, "choices", None)
    )
    if not choices:
        raise ValueError("LLM response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else choices[0].message
    content = message.get("content") if isinstance(message, dict) else message.content
    usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    tokens_in = _usage_value(usage, "prompt_tokens")
    tokens_out = _usage_value(usage, "completion_tokens")
    return content, tokens_in, tokens_out


def _usage_value(usage: Any, name: str) -> int:
    """Read a token count from dict-like or object-like usage data."""
    if usage is None:
        return 0
    value = usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)
    return int(value or 0)


class FakeLLM:
    """Deterministic completion backend for tests and ``--mock`` mode."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        default: Any = '{"ok": true}',
        errors: list[Exception] | None = None,
    ) -> None:
        """Create a fake keyed by schema name, with optional queued errors."""
        self.responses = responses or {}
        self.default = default
        self.errors = list(errors or [])
        self.calls = 0

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Return a LiteLLM-shaped response."""
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        schema_name = kwargs.get("schema_name", "")
        value = self.responses.get(schema_name, self.default)
        if isinstance(value, list):
            value = value.pop(0) if value else self.default
        return {
            "choices": [
                {"message": {"content": value if isinstance(value, str) else json.dumps(value)}}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class LLMClient:
    """Validated JSON completion client backed by LiteLLM or an injected backend."""

    def __init__(
        self,
        completion: Completion | None = None,
        *,
        model: str | None = None,
        fallback_model: str | None = None,
        cache: JsonCache | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
    ) -> None:
        """Configure the client and its retry/fallback behavior."""
        self.completion = completion
        self.model = model or settings.llm_model
        self.fallback_model = (
            fallback_model if fallback_model is not None else settings.llm_fallback_model
        )
        self.cache = cache or JsonCache(settings.data_dir, enabled=not settings.no_cache)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.stats = RunStats(llm_calls=0, cache_hits=0, tokens_in=0, tokens_out=0, seconds=0)

    def _backend(self, **kwargs: Any) -> Any:
        """Call the injected backend or LiteLLM."""
        if self.completion is not None:
            return self.completion(**kwargs)
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError("LiteLLM is required for live LLM calls") from exc
        return completion(**kwargs)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """Identify transient provider failures."""
        text = str(exc).lower()
        return isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or any(
            token in text for token in ("429", "timeout", "timed out", "500", "502", "503", "504")
        )

    def _request(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        *,
        schema_name: str,
        temperature: float,
    ) -> tuple[Any, int, int]:
        """Make one provider request, using JSON response mode."""
        response = self._backend(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            schema_name=schema_name,
        )
        return _response_content(response)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: type[ModelT],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: str | None = None,
        sample_idx: int = 0,
    ) -> ModelT:
        """Complete a JSON request and validate it against a Pydantic schema."""
        selected_model = model or self.model
        cache_key = self.cache.key(
            selected_model, system, user, temperature, schema.__name__, sample_idx
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats = self.stats.model_copy(update={"cache_hits": self.stats.cache_hits + 1})
            return schema.model_validate(cached)
        started = time.perf_counter()
        validation_error = ""
        last_error: Exception | None = None
        models = [selected_model]
        if self.fallback_model and self.fallback_model != selected_model:
            models.append(self.fallback_model)
        for current_model in models:
            for attempt in range(self.max_retries + 1):
                request_user = user
                if validation_error:
                    request_user += (
                        "\n\nYour previous JSON failed validation. Correct these errors and "
                        f"return JSON only:\n{validation_error}"
                    )
                try:
                    self.stats = self.stats.model_copy(
                        update={"llm_calls": self.stats.llm_calls + 1}
                    )
                    content, tokens_in, tokens_out = self._request(
                        current_model,
                        system,
                        request_user,
                        max_tokens,
                        schema_name=schema.__name__,
                        temperature=temperature,
                    )
                    self.stats = self.stats.model_copy(
                        update={
                            "tokens_in": self.stats.tokens_in + tokens_in,
                            "tokens_out": self.stats.tokens_out + tokens_out,
                        }
                    )
                    value = schema.model_validate(json.loads(_json_text(content)))
                    self.cache.put(cache_key, value.model_dump(mode="json"))
                    self.stats = self.stats.model_copy(
                        update={"seconds": self.stats.seconds + time.perf_counter() - started}
                    )
                    logger.info("LLM model answered: %s", current_model)
                    return value
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    last_error = exc
                    validation_error = str(exc)
                    if attempt >= self.max_retries:
                        break
                except Exception as exc:
                    last_error = exc
                    if not self._retryable(exc) or attempt >= self.max_retries:
                        break
                time.sleep(self.backoff_seconds * (2**attempt))
        self.stats = self.stats.model_copy(
            update={"seconds": self.stats.seconds + time.perf_counter() - started}
        )
        raise RuntimeError("LLM completion failed after retries and fallback") from last_error

    async def acomplete_json(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Run ``complete_json`` without blocking the event loop."""
        return await asyncio.to_thread(self.complete_json, *args, **kwargs)


async def gather_limited(
    coros: list[Coroutine[Any, Any, ModelT]] | list[Awaitable[ModelT]],
    limit: int | None = None,
) -> list[ModelT]:
    """Await coroutines with a bounded concurrency semaphore."""
    semaphore = asyncio.Semaphore(limit or settings.max_parallel_llm)

    async def guarded(coro: Awaitable[ModelT]) -> ModelT:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(guarded(coro) for coro in coros))
