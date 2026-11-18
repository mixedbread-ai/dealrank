"""A random partition against dealing as the quota tightens (``quota_sweep``).

    python -m repro.analysis.quota

Qwen3.8-27B, BM25 (ten benchmarks) and dense (eight) top 100. Each setting
keeps 20 finalists for one final call: five hands of 20 that keep 4, ten of
10 that keep 2, twenty of 5 that keep 1. The random arm averages each query
over seeds 0, 1 and 2 (a fresh partition per query). At a quota of one the
random partition's deficit reaches 1.3 [0.4, 2.4] points over the dense
retriever, and is smaller and not significant over BM25.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import runs
from repro.stats import hier_boot, summary

EXPERIMENT = "quota_sweep"
SETTINGS = ((20, 4), (10, 2), (5, 1))


def dealt(window: int, quota: int) -> P.Spec:
    if window == 20:
        return P.spec("dealrank", window=20, quota=quota)
    return P.spec("dealrank", window=window, quota=quota, final_size=20)


def contrast(window: int, quota: int, retriever: str) -> dict[str, Any]:
    """Random (mean of three seeds) minus dealt, nDCG@10 points, paired by query."""
    deal_arm = (P.Run(EXPERIMENT, dealt(window, quota), P.LOCAL),)
    rand_arm = tuple(P.Run(EXPERIMENT, P.random_partition(s, window=window, quota=quota), P.LOCAL) for s in (0, 1, 2))
    diffs, macro = {}, {"dealt": [], "random": []}
    for b in runs.benchmarks_of(retriever):
        d = runs.arm_scores(deal_arm, retriever, b)
        r = runs.arm_scores(rand_arm, retriever, b)
        qs = sorted(set(d) & set(r))
        diffs[b] = np.array([r[q] - d[q] for q in qs])
        macro["dealt"].append(np.mean([d[q] for q in qs]))
        macro["random"].append(np.mean([r[q] for q in qs]))
    return {
        "dealt": 100 * float(np.mean(macro["dealt"])),
        "random": 100 * float(np.mean(macro["random"])),
        "random_minus_dealt": summary(hier_boot(diffs), 100.0),
    }


def summary_all() -> dict[str, Any]:
    return {f"{w}x{q}": {r: contrast(w, q, r) for r in ("bm25", "dense")} for w, q in SETTINGS}


def render(s: Mapping[str, Any]) -> str:
    lines = [
        "# Random partition against dealing as the quota tightens (Qwen3.8-27B)",
        "",
        "| hands of | keep | retriever | dealt | random | random minus dealt |",
        "|---|---|---|---|---|---|",
    ]
    for setting, by in s.items():
        w, q = setting.split("x")
        for r, v in by.items():
            c = v["random_minus_dealt"]
            lines.append(
                f"| {w} | {q} | {r} | {v['dealt']:.1f} | {v['random']:.1f} | "
                f"{c['diff']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random partition minus dealing at quotas 4, 2 and 1.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    s = summary_all()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
