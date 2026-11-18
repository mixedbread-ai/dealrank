"""Our BlitzRank and sliding-window runs against the numbers BlitzRank's authors report.

    python -m repro.analysis.published

The BlitzRank paper (arXiv:2602.05448, Table 3) reports nDCG@10 per benchmark
for BlitzRank with twenty passages per call and for the sliding window, with
Gemini 3 Flash over the BM25 top 100 and the RankGPT prompt. On the ten
benchmarks both papers use, Table 1's BlitzRank ``w = 20`` (the authors'
package) and sliding-window rows are within 0.3 points of the macro average
of those published numbers.

The published values below are transcribed from that table; the macro average
over these ten benchmarks is computed here and is not a number the source
prints.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import records, runs

#: nDCG@10 x 100 as published, per benchmark; keys are Table 1 rows.
PUBLISHED: dict[str, dict[str, float]] = {
    "blitzrank_w20_concurrent": {
        "dl19": 74.9,
        "dl20": 72.1,
        "covid": 82.3,
        "nfcorpus": 41.1,
        "signal": 33.0,
        "news": 48.2,
        "robust04": 65.8,
        "touche": 37.9,
        "dbpedia": 45.3,
        "scifact": 80.4,
    },
    "sliding_window": {
        "dl19": 74.0,
        "dl20": 73.1,
        "covid": 82.6,
        "nfcorpus": 40.8,
        "signal": 32.6,
        "news": 49.4,
        "robust04": 66.5,
        "touche": 36.1,
        "dbpedia": 45.7,
        "scifact": 80.6,
    },
}


def comparison() -> dict[str, Any]:
    """Per benchmark and macro: ours, published, and ours minus published, for both rows."""
    out = {}
    for key, published in PUBLISHED.items():
        scores = runs.scores_by_benchmark(P.TABLE1_BY_KEY[key].bm25, "bm25", records.BENCHMARKS)
        ours = {b: 100 * float(np.mean(list(scores[b].values()))) for b in records.BENCHMARKS}
        out[key] = {
            "ours": ours,
            "published": published,
            "ours_macro": float(np.mean(list(ours.values()))),
            "published_macro": float(np.mean([published[b] for b in records.BENCHMARKS])),
        }
        out[key]["ours_minus_published"] = out[key]["ours_macro"] - out[key]["published_macro"]
    return out


def render(c: Mapping[str, Any]) -> str:
    lines = [
        "# Our runs against the published numbers (Gemini 3 Flash, BM25 top 100, nDCG@10 x 100)",
        "",
        "| row | " + " | ".join(records.BENCHMARKS) + " | macro |",
        "|---|" + "---|" * (len(records.BENCHMARKS) + 1),
    ]
    for key, v in c.items():
        for who in ("ours", "published"):
            cells = " | ".join(f"{v[who][b]:.1f}" for b in records.BENCHMARKS)
            lines.append(f"| {key}, {who} | {cells} | {v[who + '_macro']:.2f} |")
    lines.append("")
    for key, v in c.items():
        lines.append(f"{key}: ours minus published, macro over ten benchmarks: {v['ours_minus_published']:+.2f}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Table 1's BlitzRank and sliding-window rows against published numbers."
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    c = comparison()
    print(json.dumps(c, indent=1) if args.json else render(c))


if __name__ == "__main__":
    main()
