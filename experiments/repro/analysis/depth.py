"""Deeper lists: 100, 500 and 1,000 BM25 candidates with Qwen3.8-27B (Table 3).

    python -m repro.analysis.depth

nDCG@10 is the macro over DL19, DL20, DBpedia and SciFact. At 100 candidates
it comes from the Table 2 runs (``table2_models``), at 500 and 1,000 from
``table3_depth``, where hierarchical DealRank deals 500 candidates into 25
hands that keep 3, then 4 hands that keep 5, or 1,000 into 50 hands that keep
2, then 5 hands that keep 4, before one final call.

Seconds are the median client wall-clock per DL19 query with one query in
flight, and rounds the mean over the same timed queries: from
``table1_local_timing`` at 100 candidates and from the timing blocks of
``table3_depth`` at 500 and 1,000. BlitzRank is this library's
implementation on the retriever's order, with its first calls concurrent
and, for the last timing column, with every call in turn.

The ratios are paired by query: each query's seconds are averaged over its
repeats, and the geometric mean of the per-query ratios to DealRank gets a
bootstrap interval over queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import records, runs
from repro.stats import hier_boot, summary

DEPTHS = (100, *P.DEPTHS)
BENCHMARKS = records.DEEP_BENCHMARKS
TIMED = "dl19"
#: Timing columns of Table 3, in order.
COLUMNS = ("dealrank", "blitzrank_concurrent", "blitzrank", "sliding_window")
#: Quality columns of Table 3, in order.
QUALITY = ("dealrank", "blitzrank_concurrent", "sliding_window")
#: Timing repeats of ``table3_depth`` at 500 and 1,000 candidates: the two concurrent
#: schedules are timed three times, the two sequential ones once.
DEEP_REPEATS = {"dealrank": 3, "blitzrank_concurrent": 3, "blitzrank": 1, "sliding_window": 1}


def specs(depth: int) -> dict[str, P.Spec]:
    """The schedule behind each column at ``depth``."""
    return {
        "dealrank": P.DEALRANK if depth == 100 else P.HIERARCHICAL[depth],
        "blitzrank_concurrent": P.blitzrank(20, concurrent=True),
        "blitzrank": P.blitzrank(20),
        "sliding_window": P.WINDOW,
    }


def quality_arm(depth: int, column: str) -> tuple[P.Arm, str, int | None]:
    """``(arm, retriever, depth argument)`` of one nDCG@10 cell.

    BlitzRank returns the same ranking whether or not its first calls run
    concurrently, so at 100 candidates its cell is Table 2's sequential run.
    """
    if depth == 100:
        column = "blitzrank" if column == "blitzrank_concurrent" else column
        return (P.Run(P.T2, specs(100)[column], P.LOCAL),), "bm25", None
    return (P.Run(P.T3, specs(depth)[column], P.LOCAL),), P.DEEP, depth


def timing_arm(depth: int, column: str) -> tuple[P.Arm, str, int | None]:
    """``(arm, retriever, depth argument)`` of one timing cell: every repeat that was run."""
    s = specs(depth)[column]
    if depth == 100:
        return P.timed(s), "bm25", None
    found = records.find_repeats(P.T3, P.LOCAL, P.schedule_name(s), P.DEEP, TIMED, depth)
    repeats = sorted(r for r in found if r is not None)
    if repeats and len(repeats) != DEEP_REPEATS[column]:
        print(
            f"warning: {P.schedule_name(s)} at {depth} candidates has {len(repeats)} timing repeat(s) "
            f"on {TIMED}; the paper's Table 3 uses {DEEP_REPEATS[column]}",
            file=sys.stderr,
        )
    return tuple(P.Run(P.T3, s, P.LOCAL, r) for r in repeats), P.DEEP, depth


def quality(depth: int) -> dict[str, Any]:
    """Macro nDCG@10 per column, and each column minus DealRank."""
    scores = {}
    for column in QUALITY:
        arm, retriever, d = quality_arm(depth, column)
        scores[column] = runs.scores_by_benchmark(arm, retriever, BENCHMARKS, d)
    out: dict[str, Any] = {c: {"ndcg10": 100 * runs.macro(scores[c])} for c in QUALITY}
    for column in QUALITY[1:]:
        a, b = scores[column], scores["dealrank"]
        diffs = {x: np.array([a[x][q] - b[x][q] for q in sorted(set(a[x]) & set(b[x]))]) for x in a}
        out[column]["minus_dealrank"] = summary(hier_boot(diffs), 100.0)
    return out


def timing(depth: int) -> dict[str, Any]:
    """Median seconds and mean rounds per DL19 query per column, and each column's paired ratio to DealRank."""
    out: dict[str, Any] = {}
    walls = {}
    for column in COLUMNS:
        arm, retriever, d = timing_arm(depth, column)
        if not arm:
            raise runs.MissingRuns(
                [str(runs.cell_path(P.Run(P.T3, specs(depth)[column], P.LOCAL, 1), retriever, TIMED, d))]
            )
        recs = runs.timing_records(arm, retriever, (TIMED,), d)
        walls[column] = runs.walls_by_query(arm, retriever, (TIMED,), d)
        out[column] = {
            "seconds": float(np.median([r["wall_s"] for r in recs])),
            "rounds": float(np.mean([r["rounds"] for r in recs])),
            "query_runs": len(recs),
        }
    for column in COLUMNS[1:]:
        out[column]["ratio_to_dealrank"] = runs.paired_ratio(walls[column], walls["dealrank"])
    return out


def summary_all() -> dict[str, Any]:
    return {str(d): {"quality": quality(d), "timing": timing(d)} for d in DEPTHS}


def render(s: Mapping[str, Any]) -> str:
    lines = [
        "# Deeper lists (Qwen3.8-27B; nDCG@10 on DL19, DL20, DBpedia, SciFact; seconds per DL19 query)",
        "",
        "| top | column | nDCG@10 | minus DealRank | seconds | rounds | seconds / DealRank's |",
        "|---|---|---|---|---|---|---|",
    ]
    for d, v in s.items():
        for column in COLUMNS:
            q = v["quality"].get(column)
            t = v["timing"][column]
            md = q.get("minus_dealrank") if q else None
            delta = f"{md['diff']:+.2f} [{md['ci'][0]:+.2f}, {md['ci'][1]:+.2f}]" if md else "--"
            r = t.get("ratio_to_dealrank")
            ratio = f"{r['ratio']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r else "--"
            ndcg = f"{q['ndcg10']:.1f}" if q else "--"
            lines.append(f"| {d} | {column} | {ndcg} | {delta} | {t['seconds']:.1f} | {t['rounds']:.2f} | {ratio} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="nDCG@10, seconds and rounds over 100, 500 and 1,000 candidates.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    s = summary_all()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
