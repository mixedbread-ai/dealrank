"""BlitzRank's certification call on chunks and on dealt input.

    python -m repro.analysis.certification

BlitzRank (the authors' package, ``w = 20``, Gemini 3 Flash, BM25 top 100)
opens with calls over disjoint blocks of its input, merges the top four of
each, and then makes a certification call whenever the merge leaves the top
ten unproven. A query makes that call when it issues more calls than the
first calls and the merge. On full pools of 100 candidates, it does so on
59% of queries over contiguous chunks and on 14% when the same candidates are
dealt, which makes its first calls DealRank's hands. Its nDCG@10 changes by
-0.2 [-1.0, +0.8] points (dealt minus chunks).

Every number pools or averages the three runs of each arm
(``table1_gemini`` and ``table1_gemini_repeats``).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import runs
from repro.stats import hier_boot, summary


def certifies(record: Mapping[str, Any]) -> bool:
    """Whether a BlitzRank package record made a call after its cover and merge."""
    k = runs.leading_disjoint(record["diagnostics"]["calls"])
    return record["calls"] > k + 1


def certification(arm: P.Arm, retriever: str = "bm25", n: int = P.FULL_POOL) -> dict[str, Any]:
    """Certification rate, calls and rounds of a package arm over the queries with ``n`` candidates.

    The rate, calls and rounds are means over the records of every run of
    ``arm``, per benchmark, then averaged over benchmarks, as Table 1's cost
    columns are.
    """
    per: dict[str, list[float]] = {"certify_share": [], "calls": [], "rounds_sequential": [], "rounds_concurrent": []}
    queries = 0
    distribution: Counter[int] = Counter()
    for b in runs.benchmarks_of(retriever):
        loaded = runs.load_arm(arm, retriever, b)
        recs = [r for x in loaded for r in runs.full_pool(x, n)]
        queries += len(recs) // len(loaded)
        distribution.update(r["calls"] for r in recs)
        per["certify_share"].append(float(np.mean([certifies(r) for r in recs])))
        per["calls"].append(float(np.mean([r["calls"] for r in recs])))
        per["rounds_sequential"].append(float(np.mean([r["rounds"] for r in recs])))
        per["rounds_concurrent"].append(float(np.mean([runs.first_calls_concurrent(r)[0] for r in recs])))
    return {
        "queries": queries,
        **{k: float(np.mean(v)) for k, v in per.items()},
        "calls_distribution": dict(sorted(distribution.items())),
    }


def ndcg_contrast(retriever: str = "bm25") -> dict[str, Any]:
    """Dealt minus chunks, nDCG@10 points, paired by query."""
    if retriever == "bm25":
        dealt, chunks = P.three_runs(P.package(20, dealt=True)), P.three_runs(P.package(20))
    else:
        dealt, chunks = (P.Run(P.DENSE, P.package(20, dealt=True)),), (P.Run(P.DENSE, P.package(20)),)
    a = runs.scores_by_benchmark(dealt, retriever)
    b = runs.scores_by_benchmark(chunks, retriever)
    diffs = {x: np.array([a[x][q] - b[x][q] for q in sorted(set(a[x]) & set(b[x]))]) for x in a}
    return summary(hier_boot(diffs), 100.0)


def summary_all() -> dict[str, Any]:
    return {
        "chunks": certification(P.three_runs(P.package(20))),
        "dealt": certification(P.three_runs(P.package(20, dealt=True))),
        "ndcg_dealt_minus_chunks": ndcg_contrast("bm25"),
    }


def render(s: Mapping[str, Any]) -> str:
    lines = [
        "# BlitzRank's certification call (authors' package, w = 20, Gemini 3 Flash, BM25, 100 candidates)",
        "",
        "| input | queries | certifies | calls | rounds (sequential) | rounds (first calls concurrent) |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("chunks", "chunks"), ("dealt", "dealt")):
        c = s[key]
        lines.append(
            f"| {label} | {c['queries']} | {100 * c['certify_share']:.1f}% | {c['calls']:.2f} | "
            f"{c['rounds_sequential']:.2f} | {c['rounds_concurrent']:.2f} |"
        )
    d = s["ndcg_dealt_minus_chunks"]
    lines += [
        "",
        f"nDCG@10, dealt minus chunks (three runs each): {d['diff']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}].",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BlitzRank's certification rate on chunks and dealt input.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    s = summary_all()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
