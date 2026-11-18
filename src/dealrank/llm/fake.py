"""An offline ranker for tests, examples and dry runs.

:class:`FakeLLM` makes no network request. It orders passages by a score
function, so it behaves like the *consistent* ranker of the paper: every call
orders its passages by one fixed ranking.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence

from dealrank.llm.base import Completion, ListwiseLLM, now
from dealrank.prompt import Message
from dealrank.result import Call

ScoreFn = Callable[[str, str], float]


class FakeLLM(ListwiseLLM):
    """Order passages by ``score(query, passage)``, highest first.

    The answer is written as ``[3] > [1] > ...`` and read back by the same
    parser as a real model's. Ties keep the prompt order.

    Args:
        score: A function of ``(query, passage)``, or a mapping from passage
            text to score (missing passages score ``0``). The default scores
            every passage ``0``, which returns the prompt order.
        delay_s: Seconds each call takes, to make concurrency observable.
        tokens_per_word: Prompt tokens reported per prompt word.

    Attributes:
        requests: Every ``(query, passages)`` the model received, in order.
        max_in_flight: The most calls that were running at the same time.
    """

    model = "fake"

    def __init__(
        self,
        score: ScoreFn | Mapping[str, float] | None = None,
        *,
        delay_s: float = 0.0,
        tokens_per_word: float = 1.0,
        max_concurrency: int = 1024,
        max_passage_words: int = 300,
    ) -> None:
        super().__init__(max_passage_words=max_passage_words, max_concurrency=max_concurrency)
        if score is None:
            self._score: ScoreFn = lambda _q, _p: 0.0
        elif isinstance(score, Mapping):
            table = dict(score)
            self._score = lambda _q, p: table.get(p, 0.0)
        else:
            self._score = score
        self.delay_s = delay_s
        self.tokens_per_word = tokens_per_word
        self.requests: list[tuple[str, list[str]]] = []
        self.max_in_flight = 0
        self._in_flight = 0

    async def rank(self, query: str, passages: Sequence[str]) -> Call:
        passages = list(passages)
        self.requests.append((query, passages))
        if not passages:
            return Call(permutation=[])
        messages = self.messages(query, passages)
        started_at = now()
        t0 = time.monotonic()
        async with self._semaphore():
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                await asyncio.sleep(self.delay_s)
            finally:
                self._in_flight -= 1
        order = sorted(range(len(passages)), key=lambda i: -self._score(query, passages[i]))
        words = sum(len(m["content"].split()) for m in messages)
        completion = Completion(
            text=" > ".join(f"[{i + 1}]" for i in order),
            prompt_tokens=int(words * self.tokens_per_word),
            completion_tokens=4 * len(passages),
            latency_s=time.monotonic() - t0,
            started_at=started_at,
            ended_at=now(),
        )
        return self._to_call(completion, len(passages), cached=False)

    async def complete(self, messages: list[Message]) -> Completion:
        raise NotImplementedError("FakeLLM answers in rank(); it never sends chat messages")
