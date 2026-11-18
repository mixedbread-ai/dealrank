"""JointRank (Dedov, ICTIR 2025): overlapping blocks in one round, fused by PageRank.

Every candidate appears in ``replicates`` blocks of ``window``. All blocks are
ranked in one concurrent round, each ranked block yields its pairwise
outcomes, and PageRank on the resulting tournament gives the ranking. With
100 candidates, blocks of 20 and four replicates this is 20 calls.

JointRank needs numpy (the ``jointrank`` extra of this package).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from dealrank.llm.base import ListwiseLLM
from dealrank.partition import deal, num_hands
from dealrank.result import RerankResult, complete_ranking
from dealrank.schedules.base import Schedule

if TYPE_CHECKING:
    import numpy as np


def random_blocks(n: int, window: int, replicates: int, seed: int) -> list[list[int]]:
    """JointRank's randomized equi-replicate design.

    ``replicates`` seeded shuffles of ``0..n-1`` are concatenated and cut into
    consecutive blocks of ``window``. A block that straddles two shuffles can
    repeat an item; the repeat is swapped with an item of another block that
    neither block holds.
    """
    rng = random.Random(seed)
    pool: list[int] = []
    for _ in range(replicates):
        rep = list(range(n))
        rng.shuffle(rep)
        pool.extend(rep)
    blocks = [pool[s : s + window] for s in range(0, len(pool), window)]

    for bi, block in enumerate(blocks):
        seen: set[int] = set()
        for pi, item in enumerate(block):
            if item not in seen:
                seen.add(item)
                continue
            for bj, other in enumerate(blocks):
                if bj != bi and item not in other:
                    for pj, cand in enumerate(other):
                        if cand not in block:
                            block[pi], other[pj] = cand, item
                            seen.add(cand)
                            break
                    else:
                        continue
                    break
    return blocks


def dealt_blocks(n: int, window: int, replicates: int, seed: int) -> list[list[int]]:
    """``replicates`` dealt partitions of ``0..n-1`` into hands of at most ``window``.

    Replicate 0 is the deal. Replicate ``t > 0`` first permutes the items
    within every run of ``H`` consecutive retriever ranks with
    ``random.Random(seed + t)`` and then deals, so each replicate is
    balanced on every prefix and the replicates differ.
    """
    hands = num_hands(n, window)
    blocks: list[list[int]] = []
    for t in range(replicates):
        groups = deal(n, hands)
        if t > 0:
            rng = random.Random(seed + t)
            remap = list(range(n))
            for start in range(0, n, hands):
                run = remap[start : start + hands]
                rng.shuffle(run)
                remap[start : start + hands] = run
            groups = [[remap[i] for i in g] for g in groups]
        blocks.extend(g for g in groups if g)
    return blocks


def pagerank(wins: np.ndarray, damping: float = 0.85, tol: float = 1e-9, max_iter: int = 200) -> np.ndarray:
    """PageRank on a tournament, where ``wins[i, j]`` counts the wins of ``i`` over ``j``.

    Each loser passes its mass to the passages that beat it, in proportion to
    those wins, so a high score means strong. Passages that never lost spread
    their mass uniformly.
    """
    import numpy as np

    n = wins.shape[0]
    losses = wins.sum(axis=0)
    dangling = losses == 0
    transition = np.zeros((n, n))
    transition[:, ~dangling] = wins[:, ~dangling] / losses[~dangling]
    score = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1 - damping) / n + damping * (transition @ score + score[dangling].sum() / n)
        if np.abs(new - score).sum() < tol:
            return new
        score = new
    return score


class JointRank(Schedule):
    """One round of overlapping blocks, fused by PageRank.

    Args:
        llm: The listwise ranker.
        window: Passages per block.
        replicates: How many blocks hold each candidate (JointRank's ``r``).
        seed: Seed of the block design.
        damping: PageRank damping.
        design: ``"random"`` (JointRank's design) or ``"dealt"``.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        replicates: int = 4,
        seed: int = 0,
        damping: float = 0.85,
        design: Literal["random", "dealt"] | str = "random",
    ) -> None:
        super().__init__(llm)
        if design not in ("random", "dealt"):
            raise ValueError("design must be 'random' or 'dealt'")
        self.window = window
        self.replicates = replicates
        self.seed = seed
        self.damping = damping
        self.design = design
        self.name = f"jointrank-w{window}-r{replicates}" + ("-dealt" if design == "dealt" else "")

    def blocks(self, n: int) -> list[list[int]]:
        build = dealt_blocks if self.design == "dealt" else random_blocks
        return build(n, self.window, self.replicates, self.seed)

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("JointRank needs numpy: pip install numpy") from exc

        n = len(passages)
        if n <= self.window:
            call = await llm.rank(query, passages)
            return RerankResult(ranking=list(call.permutation), rounds=[[call]] if n else [])

        blocks = self.blocks(n)
        calls = await llm.rank_many((query, [passages[i] for i in block]) for block in blocks)
        wins: Any = np.zeros((n, n))
        for block, call in zip(blocks, calls):
            ranked = [block[p] for p in call.permutation]
            for a in range(len(ranked)):
                for b in range(a + 1, len(ranked)):
                    wins[ranked[a], ranked[b]] += 1
        score = pagerank(wins, damping=self.damping)
        head = sorted(range(n), key=lambda i: (-score[i], i))
        return RerankResult(ranking=complete_ranking(head, n), rounds=[calls], diagnostics={"design": self.design})
