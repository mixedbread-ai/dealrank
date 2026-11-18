"""TourRank (Chen et al., WWW 2025), adapted to the permutation prompt.

A tournament runs selection stages over a shrinking pool. Each stage sorts the
pool by points (ties by retriever rank), deals it into ``G`` groups, shuffles
each group with a seeded generator, ranks every group in one call and gives a
point to the top ``m`` of each, who advance. With ``r`` tournaments the final
ranking is by total points, then retriever rank.

TourRank's prompt asks the model to select the top ``m`` of a group. Here the
model returns a full permutation with the RankGPT prompt and the top ``m`` are
its first ``m`` entries. All groups of one stage, across tournaments, are one
round. At 100 candidates the stage plan is TourRank's: groups (5, 5, 1, 1, 1)
advancing (50, 20, 10, 5, 2), so one tournament is 13 calls in 5 rounds.
"""

from __future__ import annotations

import random
from typing import Literal

from dealrank.llm.base import ListwiseLLM
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import Schedule

# TourRank's stage table as (fraction of n advancing, target group size).
_STAGES: tuple[tuple[float, int], ...] = ((0.5, 20), (0.2, 10), (0.1, 20), (0.05, 10), (0.02, 5))
_MIN_GROUP = 5

TourRankSplit = Literal["deal", "chunks_first", "chunks"]


def stage_plan(n: int) -> list[tuple[int, int]]:
    """``(groups, advancing)`` for each stage of a tournament over ``n`` candidates.

    Stage ``s`` advances ``max(2, round(n * fraction_s))`` passages, kept below
    the current pool, in groups near TourRank's sizes with at least five
    passages each and no more groups than advancers.

    >>> stage_plan(100)
    [(5, 50), (5, 20), (1, 10), (1, 5), (1, 2)]
    """
    plan: list[tuple[int, int]] = []
    pool = n
    for fraction, group_size in _STAGES:
        if pool <= 2:
            break
        advancing = min(max(2, round(n * fraction)), pool - 1)
        groups = max(1, min(round(pool / group_size), pool // _MIN_GROUP, advancing))
        plan.append((groups, advancing))
        pool = advancing
    return plan


class TourRank(Schedule):
    """TourRank with ``tournaments`` independent tournaments.

    Args:
        llm: The listwise ranker.
        tournaments: TourRank's ``r``. The paper uses ``r = 1``.
        seed: Seeds the within-group shuffles (tournament ``t`` uses
            ``seed + 7919 * t``).
        split: ``"deal"`` is TourRank's own grouping. ``"chunks_first"`` gives
            the first stage contiguous blocks of the sorted pool, of the same
            sizes, and ``"chunks"`` does so at every stage.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        tournaments: int = 1,
        seed: int = 0,
        split: TourRankSplit | str = "deal",
    ) -> None:
        super().__init__(llm)
        if tournaments < 1:
            raise ValueError("tournaments must be positive")
        if split not in ("deal", "chunks_first", "chunks"):
            raise ValueError("split must be 'deal', 'chunks_first' or 'chunks'")
        self.tournaments = tournaments
        self.seed = seed
        self.split = split
        self.name = f"tourrank-r{tournaments}" + ("" if split == "deal" else f"-{split.replace('_', '-')}")

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        plan = stage_plan(n)
        if not plan:
            return RerankResult(ranking=list(range(n)), rounds=[])

        points = [[0] * n for _ in range(self.tournaments)]
        pools = [list(range(n)) for _ in range(self.tournaments)]
        rngs = [random.Random(self.seed + 7919 * t) for t in range(self.tournaments)]
        rounds: list[list[Call]] = []

        for stage, (group_count, advancing) in enumerate(plan):
            contiguous = self.split == "chunks" or (self.split == "chunks_first" and stage == 0)
            jobs: list[tuple[str, list[str]]] = []
            meta: list[tuple[int, list[int], int]] = []
            for t in range(self.tournaments):
                pool = sorted(pools[t], key=lambda i, t=t: (-points[t][i], i))
                starts = [0]
                for g in range(group_count):
                    starts.append(starts[-1] + len(pool[g::group_count]))
                for g in range(group_count):
                    group = pool[starts[g] : starts[g + 1]] if contiguous else pool[g::group_count]
                    rngs[t].shuffle(group)
                    quota = advancing // group_count + (1 if g < advancing % group_count else 0)
                    jobs.append((query, [passages[i] for i in group]))
                    meta.append((t, group, quota))
            calls = await llm.rank_many(jobs)
            rounds.append(calls)
            pools = [[] for _ in range(self.tournaments)]
            for (t, group, quota), call in zip(meta, calls):
                for local in call.permutation[:quota]:
                    points[t][group[local]] += 1
                    pools[t].append(group[local])

        head = sorted(range(n), key=lambda i: (-sum(p[i] for p in points), i))
        return RerankResult(ranking=complete_ranking(head, n), rounds=rounds, diagnostics={"split": self.split})
