"""Proposition 1: expected cuts of dealing, a random partition and contiguous chunks.

    python -m repro.analysis.expected_cuts

The model: the true top ``k`` is a uniformly random ``k``-subset of the
retriever's top ``R``, and a split into ``H`` groups advances the top ``q``
of each group, so it cuts ``sum_h (|T & P_h| - q)_+`` true top-``k``
passages. With ``c_h`` of group ``h``'s ranks within the top ``R``, the
expected cut is ``sum_h g(c_h)``, ``g(c) = E (X - q)_+`` for
``X ~ Hyp(R, k, c)``. Every value here is an exact fraction.

With five groups of twenty and ``k = 10``, ``q = 4`` (the paper's numbers):
chunks cut 6 at ``R = 20``, 2.9 at ``R = 30`` and 1.1 at ``R = 50``, dealing
0, 0.05 and 0.10. A random partition cuts 0.149 on every list, and cuts at
least one passage on 12.7% of lists; ten groups that keep two cut 0.69, and
twenty that keep one 1.68.

The module also gives the call counts of Proposition 2: ``H + 1`` calls for
two rounds, ``H (H + 1)`` for one round in which every two candidates share
a call.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from functools import cache
from math import comb
from typing import Any

N, H, W, Q, K = 100, 5, 20, 4, 10


@cache
def g(R: int, c: int, k: int = K, q: int = Q) -> Fraction:
    """``E (X - q)_+`` for ``X ~ Hyp(R, k, c)``: the expected cut of one group holding ``c`` of the top ``R``."""
    return sum(
        (Fraction((x - q) * comb(c, x) * comb(R - c, k - x), comb(R, k)) for x in range(q + 1, min(c, k) + 1)),
        Fraction(0),
    )


def deal_counts(R: int, hands: int = H) -> list[int]:
    """How many of the top ``R`` ranks each dealt hand holds."""
    return [R // hands + (1 if h < R % hands else 0) for h in range(hands)]


def chunk_counts(R: int, hands: int = H, window: int = W) -> list[int]:
    """How many of the top ``R`` ranks each contiguous chunk of ``window`` holds."""
    return [min(window, max(0, R - h * window)) for h in range(hands)]


def expected_cut(counts: list[int], R: int, k: int = K, q: int = Q) -> Fraction:
    """Expected cut of a split whose groups hold ``counts`` of the top ``R``."""
    return sum((g(R, c, k, q) for c in counts), Fraction(0))


def l_deal(R: int, hands: int = H, k: int = K, q: int = Q) -> Fraction:
    return expected_cut(deal_counts(R, hands), R, k, q)


def l_chunk(R: int, hands: int = H, window: int = W, k: int = K, q: int = Q) -> Fraction:
    return expected_cut(chunk_counts(R, hands, window), R, k, q)


def l_rand(hands: int = H, window: int = W, k: int = K, q: int = Q) -> Fraction:
    """Expected cut of a uniformly random partition into ``hands`` groups of ``window``, for any retriever."""
    return hands * g(hands * window, window, k, q)


def p_cut_random(hands: int = H, window: int = W, k: int = K, q: int = Q) -> Fraction:
    """Probability that a uniformly random partition cuts at least one of the true top ``k``.

    The law of the group counts does not depend on ``T``: the ``k`` passages
    fall into ``hands`` groups of ``window`` slots, and a composition
    ``(c_1, ..., c_H)`` has probability ``prod_h C(window, c_h) / C(hands * window, k)``.
    """

    @cache
    def ways(groups: int, left: int, over: bool) -> int:
        if groups == 0:
            return 1 if left == 0 and over else 0
        return sum(comb(window, c) * ways(groups - 1, left - c, over or c > q) for c in range(min(window, left) + 1))

    return Fraction(ways(hands, k, False), comb(hands * window, k))


def calls_one_round(hands: int = H) -> int:
    """Calls one round needs so that every two of ``hands * window`` candidates share a call (Proposition 2)."""
    return hands * (hands + 1)


def calls_two_rounds(hands: int = H) -> int:
    """Calls of two rounds over ``hands`` groups and one final call (Proposition 2)."""
    return hands + 1


def summary() -> dict[str, Any]:
    by_r = {r: {"chunk": float(l_chunk(r)), "deal": float(l_deal(r))} for r in (20, 30, 50, 100)}
    return {
        "setting": {"N": N, "H": H, "w": W, "q": Q, "k": K},
        "by_R": by_r,
        "random": {"e_cut": float(l_rand()), "p_cut": float(p_cut_random())},
        "random_ten_groups_keep_two": float(l_rand(10, 10, K, 2)),
        "random_twenty_groups_keep_one": float(l_rand(20, 5, K, 1)),
        "calls": {"one_round": calls_one_round(), "two_rounds": calls_two_rounds()},
    }


def render(s: dict[str, Any]) -> str:
    lines = [
        "# Expected cuts of the true top ten (Proposition 1)",
        "",
        "Five groups of twenty, quota four, true top ten uniform over the retriever's top R.",
        "",
        "| R | contiguous chunks | dealing |",
        "|---|---|---|",
    ]
    lines += [f"| {r} | {v['chunk']:.2f} | {v['deal']:.3f} |" for r, v in s["by_R"].items()]
    lines += [
        "",
        f"- Random partition, any retriever: {s['random']['e_cut']:.3f} cut; at least one cut on "
        f"{100 * s['random']['p_cut']:.1f}% of lists.",
        f"- Random partition, ten groups of ten that keep two: {s['random_ten_groups_keep_two']:.2f}; "
        f"twenty groups of five that keep one: {s['random_twenty_groups_keep_one']:.2f}.",
        f"- Proposition 2: one round needs {s['calls']['one_round']} calls, two rounds {s['calls']['two_rounds']}.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Proposition 1's expected cuts.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    s = summary()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
