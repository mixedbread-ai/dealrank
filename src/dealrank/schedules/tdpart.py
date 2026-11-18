"""TDPart, top-down partitioning (Parry, MacAvaney and Ganguly, SIGIR 2024).

One level ranks the top ``window`` of the current candidates and takes the
passage at rank ``cutoff`` as the pivot. The rest of the pool is cut into
windows of ``window - 1``, each ranked with the pivot prepended; passages
placed above the pivot are promoted. If nothing is promoted, the level's
order is final. Otherwise the promoted passages, capped at ``budget``, form
the next level. The ranking is the last level's order, then each level's
passages below the cut, level by level.

With ``schedule="parallel"`` all tail windows of a level depend only on the
pivot and form one round; the budget then caps the promoted passages in window
order. With ``schedule="sequential"`` windows run one at a time and stop once
``budget`` passages are promoted, as in the released code's loop.

The budget is the original algorithm's parameter *b*: its Algorithm 1 collects
the passages placed above the pivot into a candidate set of maximum size *b*
and stops searching once it is full, and its main evaluation sets *b* to the
window size, as the default here does. A passage that beats the pivot can
therefore still be left out of the next level when the budget is already
full, and which passages those are depends on the order of the windows. TDPart
has no per-group quota, but it is not free of a cap. With a retriever order
that is the reverse of the ranker's, for example, the budget fills from the
first tail windows and the best passages, which sit last, are left below the
cut (``tests/test_baselines.py`` pins this case).
"""

from __future__ import annotations

from typing import Literal

from dealrank.llm.base import ListwiseLLM
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import Schedule


class TDPart(Schedule):
    """Top-down partitioning around a pivot.

    Args:
        llm: The listwise ranker.
        window: Passages per call.
        cutoff: Rank of the pivot in the first call (default ``window // 2``).
        budget: The most promoted passages carried to the next level
            (default ``window``, as in the original evaluation). Promoted
            passages beyond it are ranked below the cut.
        schedule: ``"parallel"`` or ``"sequential"`` tail windows.
        tail_split: ``"chunks"`` cuts the first level's tail into contiguous
            windows; ``"deal"`` deals it into the same number of windows.
        max_levels: A guard against non-terminating input.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        cutoff: int | None = None,
        budget: int | None = None,
        schedule: Literal["parallel", "sequential"] | str = "parallel",
        tail_split: Literal["chunks", "deal"] | str = "chunks",
        max_levels: int = 100,
    ) -> None:
        super().__init__(llm)
        if window < 2:
            raise ValueError("window must be at least 2")
        cutoff = cutoff if cutoff is not None else window // 2
        if not 1 <= cutoff < window:
            raise ValueError("cutoff must be in [1, window)")
        budget = budget if budget is not None else window
        if budget < 1:
            raise ValueError("budget must be positive")
        if schedule not in ("parallel", "sequential"):
            raise ValueError("schedule must be 'parallel' or 'sequential'")
        if tail_split not in ("chunks", "deal"):
            raise ValueError("tail_split must be 'chunks' or 'deal'")
        self.window = window
        self.cutoff = cutoff
        self.budget = budget
        self.schedule = schedule
        self.tail_split = tail_split
        self.max_levels = max_levels
        self.name = (
            f"tdpart-w{window}-c{cutoff}-b{budget}"
            + ("-sequential" if schedule == "sequential" else "")
            + ("-deal" if tail_split == "deal" else "")
        )

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        if n == 0:
            return RerankResult(ranking=[], rounds=[])
        rounds: list[list[Call]] = []
        below_cut: list[list[int]] = []  # one segment per level
        active = list(range(n))
        head: list[int] = active

        for level in range(self.max_levels):
            if len(active) <= self.window:
                call = await llm.rank(query, [passages[i] for i in active])
                rounds.append([call])
                head = [active[p] for p in call.permutation]
                break

            top, rest = active[: self.window], active[self.window :]
            first = await llm.rank(query, [passages[i] for i in top])
            rounds.append([first])
            ranked_top = [top[p] for p in first.permutation]
            pivot = ranked_top[self.cutoff - 1]
            promoted = ranked_top[: self.cutoff - 1]
            below = ranked_top[self.cutoff :]

            width = self.window - 1  # leave room for the pivot
            windows = [rest[s : s + width] for s in range(0, len(rest), width)]
            if self.tail_split == "deal" and level == 0:
                m = len(windows)
                windows = [rest[i::m] for i in range(m)]
            unprocessed: list[int] = []

            if self.schedule == "parallel":
                calls = await llm.rank_many((query, [passages[i] for i in [pivot, *win]]) for win in windows)
                rounds.append(calls)
                for win, call in zip(windows, calls):
                    _split_at_pivot(win, call, pivot, promoted, below)
            else:
                for idx, win in enumerate(windows):
                    if len(promoted) >= self.budget:
                        unprocessed = [i for later in windows[idx:] for i in later]
                        break
                    call = await llm.rank(query, [passages[i] for i in [pivot, *win]])
                    rounds.append([call])
                    _split_at_pivot(win, call, pivot, promoted, below)

            if len(promoted) == self.cutoff - 1:  # nothing beat the pivot
                head = promoted + [pivot]
                below_cut.append(below + unprocessed)
                break
            kept, overflow = promoted[: self.budget], promoted[self.budget :]
            below_cut.append(overflow + [pivot] + below + unprocessed)
            active = kept
            head = active

        ranking = head + [i for segment in below_cut for i in segment]
        return RerankResult(ranking=complete_ranking(ranking, n), rounds=rounds)


def _split_at_pivot(window: list[int], call: Call, pivot: int, promoted: list[int], below: list[int]) -> None:
    items = [pivot, *window]
    ranked = [items[p] for p in call.permutation]
    at = ranked.index(pivot)
    promoted.extend(ranked[:at])
    below.extend(ranked[at + 1 :])
