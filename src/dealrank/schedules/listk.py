"""ListK's listwise multi-pivot quickselect (Shin et al., 2026).

LMPQSelect finds an unordered top-``k`` set and LMPQSort orders it. With
``top_k <= window`` the sort is a single call. Each quickselect iteration
ranks ``select_pivots`` pivots in one call, then places every other candidate
into a bucket between consecutive pivots by ranking it with a group of
``physical_pivots`` pivots. Pivot groups are visited from best to worst, and
the search stops after the group that holds the top-``k`` boundary, so the
groups run in sequence while the calls within a group run concurrently.

With ``pivot_policy="prior"`` the pivots are the candidates at retriever ranks
``k`` to ``k + P - 1`` of the current subproblem, the retriever standing in
for ListK's embedding-based choice.
"""

from __future__ import annotations

import random
from typing import Literal

from dealrank.llm.base import ListwiseLLM
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import Schedule


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[s : s + size] for s in range(0, len(items), size)]


def _bucket(call: Call, items: list[int], pivots: set[int], item: int) -> int:
    """The number of pivots the call ranks above ``item``.

    Counting pivots, rather than locating particular ones, gives a consistent
    bucket even when the model reorders the pivots.
    """
    above = 0
    for x in (items[p] for p in call.permutation):
        if x == item:
            return above
        if x in pivots:
            above += 1
    raise ValueError(f"candidate {item} is missing from the call's permutation")


class ListK(Schedule):
    """ListK's LMPQSelect followed by one sorting call.

    The defaults are ListK's empirical setting: 16 pivots inspected in groups
    of 2, with early stopping.

    Args:
        llm: The listwise ranker.
        top_k: The size of the selected set.
        window: Passages per call (ListK's ``L``).
        select_pivots: Pivots per quickselect iteration (``P``).
        physical_pivots: Pivots per bucket call (``P'``).
        sort_pivots: LMPQSort's pivot count; inactive when ``top_k <= window``.
        pivot_policy: ``"prior"`` or ``"random"``.
        seed: Seeds the random pivot policy.
        bucket_split: ``"chunks"`` cuts the non-pivots of the first bucket
            round into contiguous calls; ``"deal"`` deals them into the same
            number of calls.
        max_iterations: A guard on quickselect iterations.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        top_k: int = 10,
        window: int = 20,
        select_pivots: int = 16,
        physical_pivots: int | None = 2,
        sort_pivots: int = 6,
        pivot_policy: Literal["prior", "random"] | str = "prior",
        seed: int = 42,
        bucket_split: Literal["chunks", "deal"] | str = "chunks",
        max_iterations: int = 50,
    ) -> None:
        super().__init__(llm)
        if not 1 <= top_k <= window:
            raise ValueError("ListK here requires 1 <= top_k <= window")
        if not 1 <= select_pivots < window:
            raise ValueError("select_pivots must be in [1, window)")
        if not 1 <= sort_pivots < window:
            raise ValueError("sort_pivots must be in [1, window)")
        physical_pivots = physical_pivots if physical_pivots is not None else select_pivots
        if not 1 <= physical_pivots <= select_pivots:
            raise ValueError("physical_pivots must be in [1, select_pivots]")
        if pivot_policy not in ("prior", "random"):
            raise ValueError("pivot_policy must be 'prior' or 'random'")
        if bucket_split not in ("chunks", "deal"):
            raise ValueError("bucket_split must be 'chunks' or 'deal'")
        self.top_k = top_k
        self.window = window
        self.select_pivots = select_pivots
        self.physical_pivots = physical_pivots
        self.sort_pivots = sort_pivots
        self.pivot_policy = pivot_policy
        self.seed = seed
        self.bucket_split = bucket_split
        self.max_iterations = max_iterations
        self.name = (
            f"listk-k{top_k}-w{window}-p{select_pivots}-pp{physical_pivots}-{pivot_policy}"
            + (f"-s{seed}" if pivot_policy == "random" else "")
            + ("-deal" if bucket_split == "deal" else "")
        )

    def _pivots(self, candidates: list[int], need: int, rng: random.Random) -> list[int]:
        p = min(self.select_pivots, len(candidates) - 1)
        if self.pivot_policy == "random":
            return rng.sample(candidates, p)
        by_prior = sorted(candidates)
        start = min(max(need - 1, 0), len(by_prior) - p)
        return by_prior[start : start + p]

    async def _select(self, query: str, passages: list[str], llm: ListwiseLLM, rounds: list[list[Call]]) -> list[int]:
        candidates = list(range(len(passages)))
        solution: list[int] = []
        need = self.top_k
        rng = random.Random(f"listk:{self.seed}:{query}")
        first_bucket_round = True

        for _ in range(self.max_iterations):
            if need == 0:
                return solution
            if len(candidates) <= self.window:
                call = await llm.rank(query, [passages[i] for i in candidates])
                rounds.append([call])
                solution.extend([candidates[p] for p in call.permutation][:need])
                return solution

            pivots = self._pivots(candidates, need, rng)
            pivot_call = await llm.rank(query, [passages[i] for i in pivots])
            rounds.append([pivot_call])
            sorted_pivots = [pivots[p] for p in pivot_call.permutation]
            pivot_set = set(sorted_pivots)

            active = [i for i in candidates if i not in pivot_set]
            descended = False
            for group in _chunks(sorted_pivots, self.physical_pivots):
                group_set = set(group)
                calls_items = _chunks(active, self.window - len(group))
                if first_bucket_round and calls_items:
                    if self.bucket_split == "deal":
                        m = len(calls_items)
                        calls_items = [active[i::m] for i in range(m)]
                    first_bucket_round = False
                local = [group + items for items in calls_items]
                calls = await llm.rank_many((query, [passages[i] for i in items]) for items in local) if local else []
                if local:
                    rounds.append(calls)

                buckets: list[list[int]] = [[] for _ in range(len(group) + 1)]
                for items, prompt_items, call in zip(calls_items, local, calls):
                    for item in items:
                        buckets[_bucket(call, prompt_items, group_set, item)].append(item)
                for bucket in buckets:
                    bucket.sort()

                for b, pivot in enumerate(group):
                    bucket = buckets[b]
                    if len(bucket) > need:
                        candidates = bucket
                        descended = True
                        break
                    if len(bucket) == need:
                        solution.extend(bucket)
                        return solution
                    solution.extend(bucket)
                    need -= len(bucket)
                    solution.append(pivot)
                    need -= 1
                    if need == 0:
                        return solution
                if descended:
                    break
                active = buckets[-1]

            if descended:
                continue
            if len(active) > need:
                candidates = active
                continue
            if len(active) == need:
                solution.extend(active)
                return solution
            raise RuntimeError("quickselect ran out of candidates before reaching top_k")
        raise RuntimeError("quickselect exceeded max_iterations")

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        if n <= 1:
            return RerankResult(ranking=list(range(n)), rounds=[])
        if n <= self.window:
            call = await llm.rank(query, passages)
            return RerankResult(ranking=list(call.permutation), rounds=[[call]])

        rounds: list[list[Call]] = []
        target = min(self.top_k, n)
        selected = await self._select(query, passages, llm, rounds)
        if len(selected) != target or len(set(selected)) != target:
            raise RuntimeError("quickselect did not return a unique top-k set")
        if target > 1:
            final = await llm.rank(query, [passages[i] for i in selected])
            rounds.append([final])
            head = [selected[p] for p in final.permutation]
        else:
            head = selected
        return RerankResult(ranking=complete_ranking(head, n), rounds=rounds)
