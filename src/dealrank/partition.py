"""Splitting a ranked candidate list into groups.

Candidates are identified by their 0-based position in the retriever's order,
so ``0`` is the retriever's top result. Every function returns a list of
groups, each a list of positions in ascending (retriever) order.

The paper compares three splits of ``n`` candidates into ``hands`` groups:

* :func:`deal` gives group ``h`` the positions ``h, h + hands, h + 2 * hands, ...``,
  like dealing cards. Every prefix of the retriever's order is split as evenly
  as possible across the groups.
* :func:`chunks` gives group ``h`` the ``h``-th contiguous block, as in
  BlitzRank's released implementation.
* :func:`random_partition` assigns positions to groups uniformly at random.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Literal

Split = Literal["deal", "chunks", "random"]

SPLITS: tuple[str, ...] = ("deal", "chunks", "random")


def num_hands(n: int, window: int) -> int:
    """The fewest groups of at most ``window`` candidates that cover ``n``."""
    if window < 1:
        raise ValueError("window must be positive")
    return math.ceil(n / window)


def deal(n: int, hands: int) -> list[list[int]]:
    """Deal positions ``0..n-1`` round-robin into ``hands`` groups.

    >>> deal(10, 3)
    [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    """
    _check(n, hands)
    groups: list[list[int]] = [[] for _ in range(hands)]
    for i in range(n):
        groups[i % hands].append(i)
    return groups


def chunks(n: int, hands: int) -> list[list[int]]:
    """Split positions ``0..n-1`` into contiguous blocks of ``ceil(n / hands)``.

    Fewer than ``hands`` blocks are returned when the last ones would be empty.

    >>> chunks(10, 3)
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """
    _check(n, hands)
    size = math.ceil(n / hands) if n else 0
    return [list(range(h * size, min((h + 1) * size, n))) for h in range(hands) if h * size < n]


def random_partition(n: int, hands: int, seed: int) -> list[list[int]]:
    """A uniformly random balanced split, reproducible from ``seed``.

    The positions are shuffled with ``random.Random(seed)`` and then dealt, so
    group sizes match :func:`deal`. Each group is returned in retriever order.
    """
    _check(n, hands)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    groups: list[list[int]] = [[] for _ in range(hands)]
    for pos, item in enumerate(order):
        groups[pos % hands].append(item)
    for group in groups:
        group.sort()
    return groups


def split(policy: Split | str, n: int, hands: int, seed: int = 0) -> list[list[int]]:
    """Split ``n`` positions into ``hands`` groups with the named policy.

    Empty groups are dropped, so the result may hold fewer than ``hands``
    groups when ``n < hands`` or when ``policy`` is ``"chunks"``.
    """
    if policy == "deal":
        groups = deal(n, hands)
    elif policy == "chunks":
        groups = chunks(n, hands)
    elif policy == "random":
        groups = random_partition(n, hands, seed)
    else:
        raise ValueError(f"unknown split {policy!r}; expected one of {SPLITS}")
    return [g for g in groups if g]


def dealt_order(n: int, window: int) -> list[int]:
    """The deal into ``ceil(n / window)`` hands, laid out as one list.

    Hand 0's members come first in retriever order, then hand 1's, and so on.
    A schedule whose first calls rank contiguous blocks of ``window`` passages
    of its input (such as BlitzRank) then opens on dealt hands when it is given
    the candidates in this order and ``n`` is a multiple of ``window``.
    Otherwise the hands are shorter than ``window`` and each block of
    ``window`` positions holds one hand and the start of the next.

    >>> dealt_order(10, 5)
    [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    """
    if n == 0:
        return []
    return [i for hand in deal(n, num_hands(n, window)) for i in hand]


def count_cuts(groups: Sequence[Iterable[int]], targets: Iterable[int], quota: int) -> int:
    """How many ``targets`` a split cuts when every group advances ``quota``.

    This assumes a consistent ranker, which orders every call by one fixed
    ranking: a target is cut only when its group holds more than ``quota``
    targets, so the split cuts ``sum_h max(|targets in group h| - quota, 0)``.
    """
    wanted = set(targets)
    return sum(max(sum(1 for i in g if i in wanted) - quota, 0) for g in groups)


def max_load(groups: Sequence[Iterable[int]], top: int) -> int:
    """The largest number of the retriever's top ``top`` positions in one group."""
    return max((sum(1 for i in g if i < top) for g in groups), default=0)


def is_prefix_balanced(groups: Sequence[Sequence[int]], n: int) -> bool:
    """Whether every prefix of the retriever's order is split as evenly as possible."""
    hands = len(groups)
    owner = {i: h for h, g in enumerate(groups) for i in g}
    counts = [0] * hands
    for b in range(n):
        counts[owner[b]] += 1
        lo, hi = (b + 1) // hands, math.ceil((b + 1) / hands)
        if any(c < lo or c > hi for c in counts):
            return False
    return True


def _check(n: int, hands: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
    if hands < 1:
        raise ValueError("hands must be positive")
