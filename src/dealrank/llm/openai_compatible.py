"""Any OpenAI-compatible chat endpoint: OpenAI, OpenRouter, vLLM, SGLang, ...

Requires the ``openai`` package (the ``openai`` extra of this package).
"""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from typing import TYPE_CHECKING, Any

from dealrank.llm.base import Completion, ListwiseLLM, now
from dealrank.prompt import Message

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from dealrank.llm.cache import ResponseCache

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Transient HTTP statuses worth retrying.
RETRY_STATUSES = frozenset({500, 502, 503, 520, 524})


class OpenAICompatible(ListwiseLLM):
    """A listwise ranker backed by an OpenAI-compatible chat completions API.

    The defaults are the decoding settings of the paper: temperature 0,
    seed 42 and at most 1,024 output tokens. Failed requests are retried with
    exponential backoff from 2 to 60 seconds, ``attempts`` times in all.

    Args:
        model: The model name the endpoint expects.
        base_url: The API root, e.g. ``http://localhost:8000/v1``. ``None``
            uses OpenAI's.
        api_key: The key. ``None`` reads ``OPENAI_API_KEY`` when ``base_url`` is
            ``None`` (OpenAI's API) and sends a placeholder to any other server.
        temperature: Sampling temperature, or ``None`` to omit it for models
            that reject the parameter.
        seed: Sent with every request; ``None`` omits it.
        max_tokens: The output budget per call.
        extra_body: Extra JSON fields sent with every request, e.g.
            ``{"reasoning": {"enabled": False}}`` for OpenRouter.
        timeout_s: Per-request timeout.
        attempts: Total tries per request.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        seed: int | None = 42,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
        attempts: int = 6,
        max_concurrency: int = 32,
        max_passage_words: int = 300,
        cache: ResponseCache | None = None,
    ) -> None:
        super().__init__(max_passage_words=max_passage_words, max_concurrency=max_concurrency, cache=cache)
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("OpenAICompatible needs the openai package: pip install openai") from exc
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self.attempts = attempts
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI] = weakref.WeakKeyDictionary()

    @property
    def api_key(self) -> str | None:
        """The key given to the constructor, if any (``None`` means read from the environment)."""
        return self._api_key

    @classmethod
    def openrouter(cls, model: str, *, api_key: str | None = None, **kwargs: Any) -> OpenAICompatible:
        """A model served through OpenRouter, with reasoning disabled.

        The key defaults to ``OPENROUTER_API_KEY``. Pass ``extra_body`` to
        override the reasoning setting or to add provider routing.
        """
        kwargs.setdefault("extra_body", {"reasoning": {"enabled": False}})
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        return cls(model, base_url=OPENROUTER_BASE_URL, api_key=key, **kwargs)

    @classmethod
    def vllm(cls, model: str, *, base_url: str = "http://127.0.0.1:8000/v1", **kwargs: Any) -> OpenAICompatible:
        """A model served locally by ``vllm serve``, with thinking disabled.

        Thinking is turned off through the chat template (``enable_thinking``),
        which Qwen3-family templates read.
        """
        kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        kwargs.setdefault("api_key", "EMPTY")
        return cls(model, base_url=base_url, **kwargs)

    def request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": self.max_tokens, "extra_body": self.extra_body}
        if self.seed is not None:
            params["seed"] = self.seed
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def _client(self) -> AsyncOpenAI:
        # One client per event loop: pooled connections cannot cross loops.
        from openai import AsyncOpenAI

        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self._resolved_key(),
                timeout=self.timeout_s,
                max_retries=0,
            )
            self._clients[loop] = client
        return client

    def _resolved_key(self) -> str:
        # Never send an OpenAI key to another provider's server.
        if self._api_key:
            return self._api_key
        if self.base_url is None:
            return os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return "EMPTY"

    async def complete(self, messages: list[Message]) -> Completion:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        params = self.request_params()
        extra_body = params.pop("extra_body")
        started_at = now()
        delay = 2.0
        for attempt in range(self.attempts):
            last = attempt == self.attempts - 1
            t0 = time.monotonic()
            try:
                response = await self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    extra_body=extra_body or None,
                    **params,
                )
                latency = time.monotonic() - t0
                choice = response.choices[0]
                text = choice.message.content or ""
                if not text.strip():
                    raise ValueError("empty completion")
                usage = response.usage
                return Completion(
                    text=text,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_s=latency,
                    started_at=started_at,
                    ended_at=now(),
                    meta=_response_meta(response, attempt),
                )
            except (RateLimitError, APIConnectionError, APITimeoutError, ValueError):
                if last:
                    raise
            except APIStatusError as exc:
                if last or exc.status_code not in RETRY_STATUSES:
                    raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise AssertionError("unreachable")


def _response_meta(response: Any, failed_attempts: int) -> dict[str, Any]:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None) if details is not None else None
    return {
        "response_model": getattr(response, "model", None),
        "provider": getattr(response, "provider", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "cached_tokens": cached_tokens,
        "failed_attempts": failed_attempts,
    }
