"""The listwise LLM interface shared by every schedule."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dealrank.prompt import Message, parse_permutation, rankgpt_messages
from dealrank.result import Call

if TYPE_CHECKING:
    from dealrank.llm.cache import ResponseCache


@dataclass
class Completion:
    """The raw answer of a chat completion request."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ListwiseLLM(ABC):
    """An LLM that orders a list of passages for a query.

    Subclasses implement :meth:`complete`, which sends chat messages to a
    model and returns its answer. This class builds the RankGPT prompt, limits
    the number of requests in flight, parses the answer into a permutation and
    optionally caches responses.

    Args:
        max_passage_words: Passages are cut to this many words in the prompt.
        max_concurrency: The most requests in flight at once, across queries.
        cache: An optional :class:`~dealrank.llm.cache.ResponseCache`. A cached
            response is reused only for an identical request.
    """

    model: str = "unknown"

    def __init__(
        self,
        *,
        max_passage_words: int = 300,
        max_concurrency: int = 32,
        cache: ResponseCache | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.max_passage_words = max_passage_words
        self.max_concurrency = max_concurrency
        self.cache = cache
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @abstractmethod
    async def complete(self, messages: list[Message]) -> Completion:
        """Send one chat request and return the model's answer."""

    def request_params(self) -> dict[str, Any]:
        """Parameters that, with the model and messages, identify a request.

        They form the cache key, so two requests with equal parameters must
        be interchangeable.
        """
        return {}

    def messages(self, query: str, passages: Sequence[str]) -> list[Message]:
        """The prompt for one call."""
        return rankgpt_messages(query, passages, self.max_passage_words)

    def request_key(self, query: str, passages: Sequence[str]) -> str:
        """A sha256 digest of the model, the prompt and :meth:`request_params`."""
        payload = json.dumps(
            {"model": self.model, "messages": self.messages(query, passages), **self.request_params()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def rank(self, query: str, passages: Sequence[str]) -> Call:
        """Rank ``passages`` for ``query`` in one call."""
        if len(passages) == 0:
            return Call(permutation=[])
        key = self.request_key(query, passages) if self.cache is not None else None
        if self.cache is not None and key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return self._to_call(hit, len(passages), cached=True)

        messages = self.messages(query, passages)
        async with self._semaphore():
            completion = await self.complete(messages)
        if self.cache is not None and key is not None:
            self.cache.put(key, self.model, completion)
        return self._to_call(completion, len(passages), cached=False)

    async def rank_many(self, jobs: Iterable[tuple[str, Sequence[str]]]) -> list[Call]:
        """Rank several ``(query, passages)`` lists concurrently, as one round."""
        return list(await asyncio.gather(*(self.rank(q, p) for q, p in jobs)))

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self.max_concurrency)
            self._semaphores[loop] = sem
        return sem

    @staticmethod
    def _to_call(completion: Completion, num: int, *, cached: bool) -> Call:
        parsed = parse_permutation(completion.text, num)
        return Call(
            permutation=parsed.permutation,
            text=completion.text,
            repairs=parsed.repairs,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_s=completion.latency_s,
            started_at=completion.started_at,
            ended_at=completion.ended_at,
            cached=cached,
            meta=dict(completion.meta),
        )


def now() -> float:
    """Wall-clock seconds, for :attr:`Completion.started_at` and ``ended_at``."""
    return time.time()
