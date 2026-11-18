"""Any provider LiteLLM supports.

Requires the ``litellm`` package (the ``litellm`` extra of this package).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from dealrank.llm.base import Completion, ListwiseLLM, now
from dealrank.prompt import Message

if TYPE_CHECKING:
    from dealrank.llm.cache import ResponseCache


class LiteLLM(ListwiseLLM):
    """A listwise ranker that calls ``litellm.acompletion``.

    ``model`` uses LiteLLM's ``provider/model`` naming, e.g.
    ``"openrouter/google/gemini-3-flash-preview"``. Extra keyword arguments in
    ``completion_kwargs`` are passed to every request. Retries are left to
    LiteLLM (``num_retries``).
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = 0.0,
        seed: int | None = 42,
        max_tokens: int = 1024,
        completion_kwargs: dict[str, Any] | None = None,
        max_concurrency: int = 32,
        max_passage_words: int = 300,
        cache: ResponseCache | None = None,
    ) -> None:
        super().__init__(max_passage_words=max_passage_words, max_concurrency=max_concurrency, cache=cache)
        try:
            import litellm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("LiteLLM needs the litellm package: pip install litellm") from exc
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.completion_kwargs = dict(completion_kwargs or {})

    def request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": self.max_tokens, **self.completion_kwargs}
        if self.seed is not None:
            params["seed"] = self.seed
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    async def complete(self, messages: list[Message]) -> Completion:
        import litellm

        started_at = now()
        t0 = time.monotonic()
        response = await litellm.acompletion(model=self.model, messages=messages, **self.request_params())
        latency = time.monotonic() - t0
        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            started_at=started_at,
            ended_at=now(),
            meta={"response_model": getattr(response, "model", None)},
        )
