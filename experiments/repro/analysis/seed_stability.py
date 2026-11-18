"""How much a random first round ties each query's result to its seed.

    python -m repro.analysis.seed_stability

Gemini 3 Flash, BM25, queries with 100 candidates. For each query: the
standard deviation of nDCG@10 x 100 across DealRank's three runs (which
differ only by the LLM's nondeterminism), and across three seeds of the
random partition (each drawing a fresh partition per query), averaged over
every three of the ten seeds so that both spreads are taken over three
values. Random minus DealRank, paired by query, with the hierarchical
bootstrap: the standard deviation is 0.72 [0.43, 1.05] points larger. The
mean pairwise overlap of the top-ten sets is reported alongside.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import runs
from repro.stats import hier_boot, summary

ARMS = {"dealrank": P.three_runs(P.DEALRANK), "random": P.ten_seeds()}
#: Both spreads are taken over this many values: DealRank has three runs.
SUBSET = 3


def per_query(arm: P.Arm, benchmark: str) -> dict[str, tuple[float, float]]:
    """``{qid: (sd of nDCG@10 x 100 over three runs, mean pairwise top-10 overlap)}``.

    With more than three runs, the standard deviation is the mean over every
    subset of three.
    """
    recs = runs.load_arm(arm, "bm25", benchmark)
    scores = [runs.run_scores(run, "bm25", benchmark) for run in arm]
    subsets = [list(s) for s in itertools.combinations(range(len(arm)), SUBSET)]
    out = {}
    for q in set.intersection(*(set(s) for s in scores)):
        if any(x[q]["n_candidates"] != P.FULL_POOL for x in recs):
            continue
        values = np.array([100 * s[q] for s in scores])
        tops = [set(x[q]["ranking"][:10]) for x in recs]
        overlap = float(np.mean([len(a & b) for a, b in itertools.combinations(tops, 2)]))
        out[q] = (float(np.mean([np.std(values[s]) for s in subsets])), overlap)
    return out


def stability() -> dict[str, Any]:
    sd, ov = {}, {}
    pooled: dict[str, list[tuple[float, float]]] = {a: [] for a in ARMS}
    for b in runs.benchmarks_of("bm25"):
        a, r = per_query(ARMS["dealrank"], b), per_query(ARMS["random"], b)
        qs = sorted(set(a) & set(r))
        sd[b] = np.array([r[q][0] - a[q][0] for q in qs]) / 100
        ov[b] = np.array([r[q][1] - a[q][1] for q in qs]) / 100
        for arm, x in (("dealrank", a), ("random", r)):
            pooled[arm] += [x[q] for q in qs]
    return {
        "arms": {
            arm: {
                "runs": len(ARMS[arm]),
                "queries": len(v),
                "mean_sd_points": float(np.mean([x[0] for x in v])),
                "mean_top10_overlap": float(np.mean([x[1] for x in v])),
            }
            for arm, v in pooled.items()
        },
        "sd_random_minus_dealrank": summary(hier_boot(sd), 100.0),
        "overlap_random_minus_dealrank": summary(hier_boot(ov), 100.0),
    }


def render(s: Mapping[str, Any]) -> str:
    lines = [
        "# Seed dependence of a random first round (Gemini 3 Flash, BM25, 100 candidates)",
        "",
        "| arm | runs | queries | mean SD of nDCG@10 over three (points) | mean pairwise top-10 overlap |",
        "|---|---|---|---|---|",
    ]
    for arm, v in s["arms"].items():
        lines.append(
            f"| {arm} | {v['runs']} | {v['queries']} | {v['mean_sd_points']:.2f} | {v['mean_top10_overlap']:.2f} |"
        )
    d, o = s["sd_random_minus_dealrank"], s["overlap_random_minus_dealrank"]
    lines += [
        "",
        f"SD, random minus DealRank: {d['diff']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}].",
        f"Top-10 overlap, random minus DealRank: {o['diff']:+.2f} [{o['ci'][0]:+.2f}, {o['ci'][1]:+.2f}].",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-query spread of nDCG@10 across seeds and runs.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    s = stability()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
