"""DealRank against the sliding window at another cutoff (Gemini 3 Flash, BM25 top 100).

    python -m repro.analysis.ndcg_at_k [--k 20]

The same runs (three each), queries and paired bootstrap as Table 1, scored
at nDCG@k instead of nDCG@10. At nDCG@20 the sliding window minus DealRank is
-0.1 [-1.5, +1.2] points. DealRank's final call ranks twenty passages and the
sliding window orders the whole list, so both are defined at rank twenty;
BlitzRank is run to certify a top ten only and is left out.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import records, runs
from repro.stats import hier_boot, summary

ROWS = ("sliding_window", "dealrank")


def rows_at(k: int) -> dict[str, Any]:
    """Macro nDCG@k x 100 of the sliding window and DealRank, and the sliding window minus DealRank."""
    table = [P.TABLE1_BY_KEY[key] for key in ROWS]
    scores = {row.key: runs.scores_by_benchmark(row.bm25, "bm25", records.BENCHMARKS, k=k) for row in table}
    out: dict[str, Any] = {}
    for row in table:
        a, b = scores[row.key], scores["dealrank"]
        out[row.key] = {"avg": 100 * runs.macro(a)}
        if row.key != "dealrank":
            diffs = {x: np.array([a[x][q] - b[x][q] for q in sorted(set(a[x]) & set(b[x]))]) for x in a}
            out[row.key]["minus_dealrank"] = summary(hier_boot({x: d for x, d in diffs.items() if d.size}), 100.0)
    return out


def render(k: int, rows: Mapping[str, Any]) -> str:
    lines = [
        f"# nDCG@{k} x 100 (Gemini 3 Flash, BM25 top 100)",
        "",
        f"| row | nDCG@{k} | minus DealRank |",
        "|---|---|---|",
    ]
    for key, r in rows.items():
        md = r.get("minus_dealrank")
        delta = f"{md['diff']:+.2f} [{md['ci'][0]:+.2f}, {md['ci'][1]:+.2f}]" if md else "--"
        lines.append(f"| {key} | {r['avg']:.2f} | {delta} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The sliding window minus DealRank at nDCG@k.")
    parser.add_argument("--k", type=int, default=20, help="rank cutoff (default: 20)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    rows = rows_at(args.k)
    print(json.dumps(rows, indent=1) if args.json else render(args.k, rows))


if __name__ == "__main__":
    main()
