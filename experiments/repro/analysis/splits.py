"""Dealing the split inside other schedules (Figure 3).

    python -m repro.analysis.splits

For each schedule of :data:`repro.paper_rows.FIG3_ROWS`, the dealt arm minus
its usual split (contiguous groups; for JointRank, its random block design),
in nDCG@10 points and in calls per query, paired by query with the
hierarchical bootstrap, over the BM25 (ten benchmarks) and dense (eight)
pools, Gemini 3 Flash. An arm with several runs averages each query over
them first. nDCG@10 uses every query; calls use the queries whose pool holds
a full 100 candidates (1,497 over BM25, 1,468 over the dense retriever), as
Table 1's cost columns do.

Where a quota can bind: DealRank's own split gains 2.7 points over BM25 and
4.4 over the dense retriever; contiguous groups at every stage cost TourRank
2.2 [1.0, 3.6] and 4.5 [2.4, 7.0]; BlitzRank at ``w = 20`` on dealt input
changes by -0.2 [-1.0, +0.8] points over BM25 while making 0.46 and 0.57
fewer calls. Where none can, the split moves TDPart, ListK, BlitzRank at
``w = 10`` and TourRank's first stage by at most 0.6 points, never
significantly; JointRank's dealt design is the only one to score lower.

The module also prints TourRank, which already deals, against DealRank on
the three LLMs it ran with (Gemini 3 Flash, DeepSeek V4-Flash and
Qwen3.5-35B-A3B), BM25: 1.44 [0.49, 2.72] points behind, each query averaged
over the three LLMs.
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

METRICS = ("ndcg10", "calls")


def per_query(arm: P.Arm, retriever: str, benchmark: str) -> dict[str, dict[str, float]]:
    """``{qid: {"ndcg10": points, "calls": calls, "n": candidates}}`` averaged over the arm's runs."""
    recs = runs.load_arm(arm, retriever, benchmark)
    scores = [runs.run_scores(run, retriever, benchmark) for run in arm]
    common = set.intersection(*(set(r) for r in recs)) & set.intersection(*(set(s) for s in scores))
    return {
        q: {
            "ndcg10": float(np.mean([100 * s[q] for s in scores])),
            "calls": float(np.mean([r[q]["calls"] for r in recs])),
            "n": recs[0][q]["n_candidates"],
        }
        for q in common
    }


def contrast(dealt: P.Arm, other: P.Arm, retriever: str) -> dict[str, dict[str, float]]:
    a = {b: per_query(dealt, retriever, b) for b in runs.benchmarks_of(retriever)}
    o = {b: per_query(other, retriever, b) for b in runs.benchmarks_of(retriever)}
    out = {}
    for m in METRICS:
        diffs = {}
        for b in a:
            qs = sorted(set(a[b]) & set(o[b]))
            if m == "calls":  # costs are compared on full pools only
                qs = [q for q in qs if a[b][q]["n"] == o[b][q]["n"] == P.FULL_POOL]
            diffs[b] = np.array([a[b][q][m] - o[b][q][m] for q in qs])
        r = hier_boot({b: d for b, d in diffs.items() if d.size})
        out[m] = {"diff": r["diff"], "lo": r["ci_lo"], "hi": r["ci_hi"], "p": r["p_two_sided"]}
        if m == "calls":
            out[m]["queries"] = r["n_queries"]
    return out


def figure3_data() -> dict[str, Any]:
    """Every value Figure 3 draws: rows grouped by whether a quota can bind, reference first, then by BM25 effect."""
    rows = []
    for row in P.FIG3_ROWS:
        entry: dict[str, Any] = {
            "key": row.key,
            "name": row.name,
            "part": row.part,
            "group": row.group,
            "reference": row.reference,
        }
        for retriever in ("bm25", "dense"):
            c = contrast(row.dealt[retriever], row.other[retriever], retriever)
            entry[retriever] = c["ndcg10"]
            entry[f"calls_{retriever}"] = c["calls"]
        rows.append(entry)
    order = {"binds": 0, "free": 1}
    rows.sort(key=lambda r: (order[r["group"]], not r["reference"], -r["bm25"]["diff"]))
    return {"rows": rows}


def tourrank_across_llms() -> dict[str, Any]:
    """TourRank minus DealRank in nDCG@10 points, per LLM and with each query averaged over the LLMs (BM25)."""
    benchmarks = runs.benchmarks_of("bm25")
    runs.require([(arm(m), "bm25", benchmarks) for m in P.TOURRANK_LLMS for arm in (P.tourrank_arm, _dealrank_arm)])
    diffs = {}
    for m in P.TOURRANK_LLMS:
        t = runs.scores_by_benchmark(P.tourrank_arm(m), "bm25", benchmarks)
        d = runs.scores_by_benchmark(_dealrank_arm(m), "bm25", benchmarks)
        diffs[m] = {b: {q: t[b][q] - d[b][q] for q in sorted(set(t[b]) & set(d[b]))} for b in benchmarks}
    out: dict[str, Any] = {
        m: summary(hier_boot({b: np.array(list(v.values())) for b, v in diffs[m].items()}), 100.0)
        for m in P.TOURRANK_LLMS
    }
    mean = {}
    for b in benchmarks:
        common = sorted(set.intersection(*(set(diffs[m][b]) for m in P.TOURRANK_LLMS)))
        mean[b] = np.array([np.mean([diffs[m][b][q] for m in P.TOURRANK_LLMS]) for q in common])
    out["average"] = summary(hier_boot(mean), 100.0)
    return out


def _dealrank_arm(llm: str) -> P.Arm:
    return P.table2_arm(llm, "dealrank")


def render(d: Mapping[str, Any], tourrank: Mapping[str, Any] | None = None) -> str:
    def ci(c: Mapping[str, float], nd: int = 2) -> str:
        return f"{c['diff']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"

    lines = [
        "# Dealing inside other schedules (Gemini 3 Flash; dealt minus usual split)",
        "",
        "| schedule, part dealt | quota binds | nDCG@10 BM25 | nDCG@10 dense | calls BM25 | calls dense |",
        "|---|---|---|---|---|---|",
    ]
    for r in d["rows"]:
        name = r["name"].replace(r"$\mathit{w}$", "w")
        lines.append(
            f"| {name}, {r['part']} | {'yes' if r['group'] == 'binds' else 'no'} | "
            f"{ci(r['bm25'])} | {ci(r['dense'])} | {ci(r['calls_bm25'])} | {ci(r['calls_dense'])} |"
        )
    if tourrank:
        lines += ["", "## TourRank minus DealRank (nDCG@10 points, BM25)", "", "| LLM | difference |", "|---|---|"]
        for m, r in tourrank.items():
            lines.append(f"| {m} | {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dealt minus usual split inside other schedules.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    d = figure3_data()
    t = tourrank_across_llms()
    print(json.dumps({**d, "tourrank_minus_dealrank": t}, indent=1) if args.json else render(d, t))


if __name__ == "__main__":
    main()
