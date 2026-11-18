"""The schedule interface and helpers shared by every schedule."""

from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from typing import Any, Literal, TypeVar

from dealrank.llm.base import ListwiseLLM
from dealrank.partition import dealt_order
from dealrank.result import RerankResult

T = TypeVar("T")

InputOrder = Literal["retriever", "dealt"]
INPUT_ORDERS: tuple[str, ...] = ("retriever", "dealt")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Inside a running event loop (a notebook, say) the coroutine runs on a
    fresh loop in a worker thread; there, prefer ``await schedule.arerank(...)``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class Schedule(ABC):
    """A plan of listwise LLM calls that reranks a list of candidates.

    Candidates are passed in the retriever's order, best first. Subclasses
    implement :meth:`_rerank`, and must group their calls into rounds
    honestly: a call belongs to a later round only if it depends on the
    answer of an earlier call.

    Args:
        llm: The ranker to call. It can also be passed to each
            :meth:`rerank` call instead.
    """

    #: A short identifier of the schedule and its parameters.
    name: str = "schedule"

    def __init__(self, llm: ListwiseLLM | None = None) -> None:
        self.llm = llm

    def rerank(self, query: str, passages: Sequence[str], *, llm: ListwiseLLM | None = None) -> RerankResult:
        """Rerank ``passages`` for ``query`` (blocking)."""
        return run_sync(self.arerank(query, passages, llm=llm))

    async def arerank(self, query: str, passages: Sequence[str], *, llm: ListwiseLLM | None = None) -> RerankResult:
        """Rerank ``passages`` for ``query``; calls in one round run concurrently."""
        ranker = llm or self.llm
        if ranker is None:
            raise ValueError(f"{type(self).__name__} has no llm: pass llm= to the constructor or to rerank()")
        result = await self._rerank(query, list(passages), ranker)
        if sorted(result.ranking) != list(range(len(passages))):
            raise AssertionError(f"{self.name} did not return a permutation of its input")
        return result

    @abstractmethod
    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        """Rerank ``passages``; return positions into ``passages``, best first."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def input_permutation(order: InputOrder | str, n: int, *, window: int = 20) -> list[int]:
    """The order in which a schedule is shown ``n`` retriever-ranked candidates.

    ``"dealt"`` lays the candidates out as the deal into ``ceil(n / window)``
    hands, one hand after another (:func:`dealrank.partition.dealt_order`).
    """
    if order == "retriever":
        return list(range(n))
    if order == "dealt":
        return dealt_order(n, window)
    raise ValueError(f"unknown input order {order!r}; expected one of {INPUT_ORDERS}")


class Reordered(Schedule):
    """Run ``inner`` on the candidates in another order, then map back.

    ``Reordered(BlitzRank(window=20), "dealt")`` is BlitzRank on dealt input:
    with 100 candidates its five opening blocks of twenty become DealRank's
    five hands. With a number of candidates that is not a multiple of the
    window, a block of ``window`` positions holds one hand and the start of
    the next (see :mod:`dealrank.schedules.blitzrank`).
    """

    def __init__(
        self,
        inner: Schedule,
        order: InputOrder | str = "dealt",
        *,
        window: int = 20,
        llm: ListwiseLLM | None = None,
    ) -> None:
        super().__init__(llm or inner.llm)
        if order not in INPUT_ORDERS:
            raise ValueError(f"unknown input order {order!r}; expected one of {INPUT_ORDERS}")
        self.inner = inner
        self.order = order
        self.window = window
        self.name = inner.name + ("-dealt" if order == "dealt" else "")

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        perm = input_permutation(self.order, len(passages), window=self.window)
        result = await self.inner._rerank(query, [passages[i] for i in perm], llm)
        diagnostics = dict(result.diagnostics)
        diagnostics["input_order"] = {"order": self.order, "permutation": perm}
        return RerankResult(ranking=[perm[i] for i in result.ranking], rounds=result.rounds, diagnostics=diagnostics)
