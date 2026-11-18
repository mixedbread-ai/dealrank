"""The sliding window of RankGPT (Sun et al., 2023)."""

from __future__ import annotations

from dealrank.llm.base import ListwiseLLM
from dealrank.result import Call, RerankResult
from dealrank.schedules.base import Schedule


class SlidingWindow(Schedule):
    """Rank windows from the bottom of the list to the top, each after the last.

    A window of ``window`` passages slides up by ``stride``, and each call
    reorders its window in place, so every call is its own round. With 100
    candidates, a window of 20 and a stride of 10 this is nine calls.

    Args:
        llm: The listwise ranker.
        window: Passages per call.
        stride: How far the window moves between calls.
        passes: How many bottom-to-top passes to make.
    """

    def __init__(self, llm: ListwiseLLM | None = None, *, window: int = 20, stride: int = 10, passes: int = 1) -> None:
        super().__init__(llm)
        if not 0 < stride <= window:
            raise ValueError("stride must be in (0, window]")
        if passes < 1:
            raise ValueError("passes must be positive")
        self.window = window
        self.stride = stride
        self.passes = passes
        self.name = f"sliding-window-w{window}-s{stride}" + (f"-x{passes}" if passes > 1 else "")

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        order = list(range(n))
        rounds: list[list[Call]] = []
        for _ in range(self.passes):
            end = n
            while end > 0:
                start = max(0, end - self.window)
                items = order[start:end]
                call = await llm.rank(query, [passages[i] for i in items])
                order[start:end] = [items[i] for i in call.permutation]
                rounds.append([call])
                if start == 0:
                    break
                end -= self.stride
        return RerankResult(ranking=order, rounds=rounds)
