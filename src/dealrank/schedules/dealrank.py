"""DealRank: deal the candidates into hands, keep a quota from each, rank the finalists."""

from __future__ import annotations

import hashlib

from dealrank.llm.base import ListwiseLLM
from dealrank.partition import SPLITS, Split, num_hands, split
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import Schedule


def query_seed(query: str) -> int:
    """A seed derived from the query text, for per-query random splits."""
    return int(hashlib.sha256(query.encode()).hexdigest()[:12], 16)


class DealRank(Schedule):
    """Two rounds: rank dealt hands in parallel, then rank their finalists in one call.

    The ``n`` candidates are dealt by retriever rank into ``H = ceil(n / window)``
    hands, each presented in retriever order. All hands are ranked at once and
    the top ``quota`` of each advance. One final call ranks the finalists,
    listed by their place in their hand and then by retriever rank. The
    eliminated candidates follow the final call's order in the same order.

    With 100 candidates and the defaults this is five hands of twenty, four
    finalists from each and six calls in two rounds.

    Other pool sizes. A pool of at most ``window`` candidates is ranked in a
    single call (one round), and an empty pool makes no call. With an explicit
    ``quota`` and fewer hands than ``window // quota``, the final call holds
    fewer than ``window`` passages: with ``quota=4`` and 21 to 40 candidates
    there are two hands and eight finalists, and the ranks after the eighth
    come from the place in the hand, not from the final call. The paper's
    100-candidate configurations set ``quota=4`` on every pool, including the
    BM25 pools that hold fewer than 100 candidates; ``quota=None`` would fill
    the window instead.

    Deeper pools. This class always makes exactly one final call. If
    ``quota * H`` exceeds ``final_size``, the surplus finalists are dropped by
    place in hand and retriever rank, without being compared by the LLM: with
    ``quota=4`` and 1,000 candidates, 200 passages advance and 180 of them are
    cut that way. Use :class:`HierarchicalDealRank` for such pools, as the
    paper does for 500 and 1,000 candidates; this class does not switch to it.

    Args:
        llm: The listwise ranker.
        window: Passages per hand.
        quota: Finalists kept from each hand. ``None`` keeps
            ``max(1, window // H)``, which fills one window.
        final_size: The most passages in the final call (default ``window``).
            If more advance, the final call takes the best ``final_size`` by
            place in hand, then retriever rank.
        split: ``"deal"`` (DealRank), or ``"chunks"`` or ``"random"`` to run
            the same calls on another split.
        seed: Seed of the random split.
        seed_per_query: Draw a fresh random split for every query, seeded by
            ``seed`` plus a hash of the query text.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        quota: int | None = None,
        final_size: int | None = None,
        split: Split | str = "deal",
        seed: int = 0,
        seed_per_query: bool = False,
    ) -> None:
        super().__init__(llm)
        if window < 2:
            raise ValueError("window must be at least 2")
        if quota is not None and quota < 1:
            raise ValueError("quota must be positive")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self.window = window
        self.quota = quota
        self.final_size = final_size if final_size is not None else window
        self.split = split
        self.seed = seed
        self.seed_per_query = seed_per_query
        name = f"dealrank-w{window}" + (f"-q{quota}" if quota else "")
        if self.final_size != window:
            name += f"-f{self.final_size}"
        if split != "deal":
            name += f"-{split}"
            if split == "random":
                name += f"-s{seed}" + ("-perquery" if seed_per_query else "")
        self.name = name

    def hands(self, n: int, query: str = "") -> list[list[int]]:
        """The first-round hands for ``n`` candidates (positions in retriever order)."""
        seed = self.seed + (query_seed(query) if self.seed_per_query else 0)
        return split(self.split, n, num_hands(n, self.window), seed=seed)

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        if n <= self.window:
            call = await llm.rank(query, passages)
            return RerankResult(ranking=list(call.permutation), rounds=[[call]] if n else [])

        hands = self.hands(n, query)
        quota = self.quota if self.quota is not None else max(1, self.window // num_hands(n, self.window))

        # Round 1: every hand at once.
        calls = await llm.rank_many((query, [passages[i] for i in hand]) for hand in hands)
        place: dict[int, int] = {}
        for hand, call in zip(hands, calls):
            for rank, local in enumerate(call.permutation):
                place[hand[local]] = rank
        by_place = sorted(range(n), key=lambda i: (place[i], i))
        finalists = [i for i in by_place if place[i] < quota][: self.final_size]

        # Round 2: one call over the finalists.
        final = await llm.rank(query, [passages[i] for i in finalists])
        head = [finalists[local] for local in final.permutation]
        return RerankResult(
            ranking=complete_ranking(head, n, rest_order=by_place),
            rounds=[calls, [final]],
            diagnostics={"split": self.split, "quota": quota, "hands": hands, "finalists": finalists},
        )


class HierarchicalDealRank(Schedule):
    """Three rounds for deep pools: deal, keep a quota, deal the survivors again, rank the finalists.

    Round 1 deals the ``n`` candidates into ``ceil(n / window)`` hands and keeps
    ``first_quota`` from each. The survivors, back in retriever order, are
    dealt into ``ceil(survivors / window)`` hands that keep ``second_quota``
    each, so that at most ``final_size`` passages reach round 3, one call over
    every survivor listed in retriever order. Every eliminated candidate is
    eliminated by an LLM call; nothing is truncated without one.

    The paper uses ``first_quota=3`` for 500 candidates (25 hands keep 3, then
    4 hands keep 5) and ``first_quota=2`` for 1,000 (50 hands keep 2, then 5
    hands keep 4), each in three rounds. When the first round already leaves
    at most ``final_size`` survivors, the schedule stops after two rounds.

    The ranking lists the final call's order, then the candidates cut in
    round 2, then those cut in round 1; within each stage by place in hand,
    then retriever rank.

    Args:
        llm: The listwise ranker.
        window: Passages per hand in rounds 1 and 2.
        first_quota: Survivors kept from each first-round hand.
        final_size: The most passages in the final call (default ``window``).
        second_quota: Survivors kept from each second-round hand; ``None``
            keeps ``final_size // (second-round hands)``.
        split: The split used in both filtering rounds.
        seed: Seed of the random split (round 2 uses ``seed + 1``).
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        first_quota: int = 4,
        final_size: int | None = None,
        second_quota: int | None = None,
        split: Split | str = "deal",
        seed: int = 0,
    ) -> None:
        super().__init__(llm)
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 1 <= first_quota < window:
            raise ValueError("first_quota must be in [1, window)")
        final_size = final_size if final_size is not None else window
        if not 1 <= final_size <= window:
            raise ValueError("final_size must be in [1, window]")
        if second_quota is not None and second_quota < 1:
            raise ValueError("second_quota must be positive")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self.window = window
        self.first_quota = first_quota
        self.final_size = final_size
        self.second_quota = second_quota
        self.split = split
        self.seed = seed
        name = f"dealrank-hier-w{window}-q{first_quota}"
        if second_quota is not None:
            name += f"-q{second_quota}"
        if final_size != window:
            name += f"-f{final_size}"
        if split != "deal":
            name += f"-{split}" + (f"-s{seed}" if split == "random" else "")
        self.name = name

    def plan(self, n: int) -> list[tuple[int, int]]:
        """``(hands, quota)`` of each filtering round for ``n`` candidates.

        >>> HierarchicalDealRank(first_quota=2).plan(1000)
        [(50, 2), (5, 4)]
        """
        if n <= self.window:
            return []
        hands1 = num_hands(n, self.window)
        survivors = sum(min(self.first_quota, len(h)) for h in split(self.split, n, hands1, seed=self.seed))
        if survivors <= self.final_size:
            return [(hands1, self.first_quota)]
        hands2 = num_hands(survivors, self.window)
        if hands2 > self.final_size:
            raise ValueError(
                f"{n} candidates leave {survivors} survivors after round 1, too many for one more "
                f"round into a final call of {self.final_size}; lower first_quota"
            )
        quota2 = self.second_quota if self.second_quota is not None else self.final_size // hands2
        if quota2 * hands2 > self.final_size:
            raise ValueError(f"second_quota={quota2} over {hands2} hands overfills a final call of {self.final_size}")
        return [(hands1, self.first_quota), (hands2, quota2)]

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        if n <= self.window:
            call = await llm.rank(query, passages)
            return RerankResult(ranking=list(call.permutation), rounds=[[call]] if n else [])

        plan = self.plan(n)
        rounds: list[list[Call]] = []
        pool = list(range(n))  # candidates still in play, in retriever order
        cut_by_stage: list[list[int]] = []
        stages: list[dict[str, object]] = []
        for stage, (hands_count, quota) in enumerate(plan):
            groups = [[pool[p] for p in g] for g in split(self.split, len(pool), hands_count, seed=self.seed + stage)]
            calls = await llm.rank_many((query, [passages[i] for i in g]) for g in groups)
            rounds.append(calls)
            survivors: list[int] = []
            cut: list[tuple[int, int]] = []
            for group, call in zip(groups, calls):
                ranked = [group[local] for local in call.permutation]
                survivors.extend(ranked[:quota])
                cut.extend((rank, i) for rank, i in enumerate(ranked) if rank >= quota)
            cut_by_stage.append([i for _, i in sorted(cut)])
            stages.append({"hands": groups, "quota": quota})
            pool = sorted(survivors)

        final = await llm.rank(query, [passages[i] for i in pool])
        rounds.append([final])
        head = [pool[local] for local in final.permutation]
        order = head + [i for stage_cut in reversed(cut_by_stage) for i in stage_cut]
        return RerankResult(
            ranking=complete_ranking(order, n),
            rounds=rounds,
            diagnostics={"split": self.split, "stages": stages, "finalists": pool},
        )
