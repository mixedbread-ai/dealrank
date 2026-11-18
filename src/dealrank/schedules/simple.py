"""Reference points: the retriever's order and one call over its top results."""

from __future__ import annotations

from dealrank.llm.base import ListwiseLLM
from dealrank.result import RerankResult, complete_ranking
from dealrank.schedules.base import Schedule


class RetrieverOrder(Schedule):
    """Keep the retriever's order; makes no call."""

    name = "retriever"

    async def arerank(self, query: str, passages: list[str], *, llm: ListwiseLLM | None = None) -> RerankResult:
        return RerankResult(ranking=list(range(len(passages))), rounds=[])

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        return RerankResult(ranking=list(range(len(passages))), rounds=[])


class TopPrefix(Schedule):
    """Rank the retriever's top ``size`` in one call and leave the rest in place."""

    def __init__(self, llm: ListwiseLLM | None = None, *, size: int = 20) -> None:
        super().__init__(llm)
        if size < 1:
            raise ValueError("size must be positive")
        self.size = size
        self.name = f"top{size}"

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        if n == 0:
            return RerankResult(ranking=[], rounds=[])
        call = await llm.rank(query, passages[: self.size])
        return RerankResult(ranking=complete_ranking(call.permutation, n), rounds=[[call]])
