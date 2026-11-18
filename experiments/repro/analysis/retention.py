"""What a first-round call keeps of the top ten it holds, and dealing under that curve (Section 2).

    python -m repro.analysis.retention

Proposition 1 extends to a noisy ranker whose call advances on average
``a(x)`` of the ``x`` top-ten passages it holds, whenever ``a`` is concave.
This module measures ``a`` on the recorded first rounds of Gemini 3 Flash,
taking the sliding window's top ten as the reference: BlitzRank at ``w = 20``
over BM25 and the dense retriever and on dealt input over BM25, DealRank over
both retrievers, and DealRank on contiguous chunks over both. Each is five
calls of twenty that advance four, on the queries with 100 candidates: 51,955
calls in all.

The measured curve is concave up to ``x = 7`` (its second differences are
negative at ``x = 1..6``). Under the measured curve, with the reference's top
ten uniform over the retriever's top ``R``, dealt hands lose the fewest
passages of every split into five groups, at every ``R`` from 10 to 100: an
exact minimum over all group sizes by dynamic programming.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from math import comb, lcm
from typing import Any

from dealrank.partition import chunks, deal
from repro import paper_rows as P
from repro import runs

N, HANDS, WINDOW, QUOTA, K = P.FULL_POOL, 5, 20, 4, 10

#: (key, label, retriever, run): the first rounds the curve is measured on.
SETTINGS: tuple[tuple[str, str, str, P.Run], ...] = (
    ("blitzrank_bm25", "BlitzRank, BM25", "bm25", P.Run(P.T1, P.package(20))),
    ("blitzrank_dense", "BlitzRank, dense", "dense", P.Run(P.DENSE, P.package(20))),
    ("blitzrank_dealt_bm25", "BlitzRank on dealt input, BM25", "bm25", P.Run(P.T1, P.package(20, dealt=True))),
    ("dealrank_bm25", "DealRank, BM25", "bm25", P.Run(P.T1, P.DEALRANK)),
    ("dealrank_dense", "DealRank, dense", "dense", P.Run(P.DENSE, P.DEALRANK)),
    ("chunks_bm25", "DealRank on chunks, BM25", "bm25", P.Run(P.T1, P.CHUNKS)),
    ("chunks_dense", "DealRank on chunks, dense", "dense", P.Run(P.DENSE, P.CHUNKS)),
)


def first_round(record: Mapping[str, Any]) -> tuple[list[list[int]], set[int]]:
    """The five first-round calls of one record and the passages they advanced, as pool positions."""
    diagnostics = record["diagnostics"]
    if "hands" in diagnostics:  # DealRank: the finalists are what the hands advanced
        return [list(h) for h in diagnostics["hands"]], set(diagnostics["finalists"])
    calls = diagnostics["calls"]  # BlitzRank: five disjoint calls, then one over four passages of each
    return [list(c) for c in calls[:HANDS]], set(calls[HANDS])


def counts() -> dict[str, dict[str, Any]]:
    """Per setting: queries, and per ``x`` the calls holding ``x`` reference passages and how many they advanced."""
    out = {}
    for key, _, retriever, run in SETTINGS:
        experiment = P.T1 if retriever == "bm25" else P.DENSE
        calls: Counter[int] = Counter()
        kept: Counter[int] = Counter()
        queries = 0
        for b in runs.benchmarks_of(retriever):
            pool = runs.pool_docids(retriever, b)
            window = runs.load_arm((P.Run(experiment, P.WINDOW),), retriever, b)[0]
            recs = runs.load_arm((run,), retriever, b)[0]
            for q, rec in sorted(recs.items()):
                if rec["n_candidates"] != N or q not in window:
                    continue
                pos = {d: i for i, d in enumerate(pool[q])}
                top = {pos[d] for d in window[q]["ranking"][:K]}
                hands, advanced = first_round(rec)
                assert sorted(i for h in hands for i in h) == list(range(N)) and len(advanced) == HANDS * QUOTA
                queries += 1
                for hand in hands:
                    held = top & set(hand)
                    calls[len(held)] += 1
                    kept[len(held)] += len(held & advanced)
        out[key] = {"queries": queries, "calls": dict(sorted(calls.items())), "kept": dict(sorted(kept.items()))}
    return out


def curve(by_setting: Mapping[str, Mapping[str, Any]]) -> list[Fraction]:
    """The pooled ``a(0..10)``: mean reference passages advanced by the calls that hold ``x`` of them."""
    calls: Counter[int] = Counter()
    kept: Counter[int] = Counter()
    for s in by_setting.values():
        calls.update(s["calls"])
        kept.update(s["kept"])
    return [Fraction(0)] + [Fraction(kept[x], calls[x]) for x in range(1, K + 1)]


def second_differences(a: list[Fraction]) -> dict[int, Fraction]:
    return {x: a[x - 1] + a[x + 1] - 2 * a[x] for x in range(1, len(a) - 1)}


def losses(a: list[Fraction], depths: range = range(K, N + 1)) -> dict[int, dict[str, Any]]:
    """Expected reference passages lost in the first round, per depth ``R``, under the curve ``a``.

    The reference's top ten is uniform over the retriever's top ``R``, so a
    group with ``c`` of those ranks holds a hypergeometric number of them and
    loses ``x - a(x)`` when it holds ``x``. ``best`` is the least loss over
    every way to share the ``R`` ranks among five groups.
    """
    scale = lcm(*(v.denominator for v in a))
    psi = [int((x - a[x]) * scale) for x in range(K + 1)]
    out = {}
    for depth in depths:
        g = [sum(psi[x] * comb(c, x) * comb(depth - c, K - x) for x in range(min(c, K) + 1)) for c in range(depth + 1)]
        best: list[int | None] = [0] + [None] * depth
        for _ in range(HANDS):
            new: list[int | None] = [None] * (depth + 1)
            for s, v in enumerate(best):
                if v is None:
                    continue
                for c in range(depth - s + 1):
                    if new[s + c] is None or v + g[c] < new[s + c]:
                        new[s + c] = v + g[c]
            best = new
        total = scale * comb(depth, K)
        dealt = sum(g[sum(i < depth for i in hand)] for hand in deal(N, HANDS))
        chunk = sum(g[sum(i < depth for i in hand)] for hand in chunks(N, HANDS))
        out[depth] = {
            "deal": float(Fraction(dealt, total)),
            "chunk": float(Fraction(chunk, total)),
            "deal_is_optimal": dealt == best[depth],
        }
    return out


def retention() -> dict[str, Any]:
    by_setting = counts()
    a = curve(by_setting)
    d2 = second_differences(a)
    by_depth = losses(a)
    return {
        "settings": {key: {"label": label, **by_setting[key]} for key, label, _, _ in SETTINGS},
        "calls": sum(sum(s["calls"].values()) for s in by_setting.values()),
        "curve": [float(v) for v in a],
        "second_differences": {str(x): float(v) for x, v in d2.items()},
        "concave_up_to": next(x for x, v in d2.items() if v > 0),
        "losses": {str(r): by_depth[r] for r in (10, 20, 30, 50, 100)},
        "deal_is_optimal_at_every_depth": all(v["deal_is_optimal"] for v in by_depth.values()),
        "depths": [min(by_depth), max(by_depth)],
    }


def render(r: Mapping[str, Any]) -> str:
    lines = [
        "# What a first-round call keeps of the sliding window's top ten (Gemini 3 Flash)",
        "",
        "| first round | queries | calls |",
        "|---|---|---|",
    ]
    for s in r["settings"].values():
        lines.append(f"| {s['label']} | {s['queries']} | {sum(s['calls'].values())} |")
    lines += [
        f"| all | {sum(s['queries'] for s in r['settings'].values())} | {r['calls']} |",
        "",
        "| top-ten passages held, x | " + " | ".join(str(x) for x in range(1, K + 1)) + " |",
        "|---|" + "---|" * K,
        "| advanced, a(x) | " + " | ".join(f"{v:.3f}" for v in r["curve"][1:]) + " |",
        "| second difference | " + " | ".join(f"{v:+.3f}" for v in r["second_differences"].values()) + " | |",
        "",
        f"- a is concave up to x = {r['concave_up_to']}: its second differences are negative below it.",
        "- Expected passages lost under the measured curve, dealing against chunks: "
        + "; ".join(f"R = {d}: {v['deal']:.2f} against {v['chunk']:.2f}" for d, v in r["losses"].items())
        + ".",
        f"- Dealing loses the fewest of every split into five groups at every depth from {r['depths'][0]} "
        f"to {r['depths'][1]}: {r['deal_is_optimal_at_every_depth']}.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The measured retention curve of a first-round call.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    r = retention()
    print(json.dumps(r, indent=1) if args.json else render(r))


if __name__ == "__main__":
    main()
