"""Latency: the API timing of Figure 1(b), local timings and throughput under load.

    python -m repro.analysis.latency

* Figure 1(b) (``fig1_api_timing``: Gemini 3 Flash, one query in flight,
  methods interleaved per query, DL19, DL20, TREC-COVID and Touche, three
  repeats): the median critical path per method, the sum over rounds of the
  slowest uncached call, pooled over every query run. DealRank 3.4 s,
  BlitzRank with its first calls concurrent 4.0 s on chunks and 3.5 s on
  dealt input, the sliding window 13.4 s, each within 0.3 s of the median
  client wall-clock. Each bar is split into rounds in proportion to the mean
  slowest call of each round (a round a query does not make counts 0), so
  the blocks sum to the median. Its rounds column is the mean over
  100-candidate queries of Table 1's runs where the method has a Table 1
  row, and its nDCG@10 column the mean over the six LLMs of Table 2:
  DealRank's mean nDCG@10 x 100 over them, and each other method's paired
  difference from it as in Table 2's average row (with its 95% interval),
  added to that mean.
* Local timing on DL19 (``table1_local_timing``: Qwen3.8-27B, three
  repeats): median and 90th percentile of the client wall-clock per query,
  Table 1's p50 and p90.
* Local timing on four benchmarks (``local_latency``: Qwen3.8-27B, DL19,
  DL20, TREC-COVID and Touche, one query in flight, three repeats). Paired by
  query, BlitzRank with concurrent first calls takes 1.20 [1.14, 1.26] times
  as long as DealRank, or 1.03 [1.01, 1.05] times on dealt input, and the
  sliding window 3.31 [2.96, 3.69] times. DealRank's
  90th percentile is at most 0.36 s above its median on every benchmark,
  that of concurrent BlitzRank up to 1.23 s above.
* Load (``load``: the same four benchmarks with 8 and with 32 queries in
  flight, three repeats of each, each on its own server instance). A
  method's throughput in one repeat is that of the fixed mix of the 196
  queries, ``196 / sum_b (n_b / X_b)`` with ``X_b`` the queries per second of
  its block on benchmark ``b``. With 8 and 32 queries in flight, DealRank
  serves 1.09 [1.06, 1.13] and 1.15 [1.10, 1.20] times the queries per
  second of concurrent BlitzRank, 1.92 [1.86, 1.97] and 1.54 [1.51, 1.58]
  times those of the sliding window, and 1.01 [0.99, 1.02] and
  1.05 [1.01, 1.10] times those of BlitzRank on dealt input: the geometric
  mean over repeats of the per-repeat ratio, with a t-interval on its
  logarithm.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import records, runs
from repro.analysis.certification import certification

LOCAL_LATENCY = "local_latency"
LOAD = "load"
#: The methods timed locally on four benchmarks and under load.
LOCAL_ARMS: dict[str, P.Spec] = {
    "dealrank": P.DEALRANK,
    "blitzrank_concurrent": P.blitzrank(20, concurrent=True),
    "blitzrank_dealt_concurrent": P.blitzrank(20, dealt=True, concurrent=True),
    "sliding_window": P.WINDOW,
}
#: Two-sided 95% Student t quantiles by degrees of freedom.
T_975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004}


def stacked_path(arm: P.Arm, benchmarks: tuple[str, ...] = P.FIG1_BENCHMARKS) -> dict[str, Any]:
    """Median critical path of a timing arm, its mean rounds, and the median split into rounds."""
    recs = runs.timing_records(arm, "bm25", benchmarks)
    lat = [runs.round_latencies(r) for r in recs]
    paths = [sum(x) for x in lat]
    longest = max(len(x) for x in lat)
    mean_round = [float(np.mean([x[i] if i < len(x) else 0.0 for x in lat])) for i in range(longest)]
    median = float(np.median(paths))
    return {
        "query_runs": len(recs),
        "path_s": median,
        "path_p90_s": float(np.percentile(paths, 90)),
        "wall_s": float(np.median([r["wall_s"] for r in recs])),
        "rounds": float(np.mean([r["rounds"] for r in recs])),
        "segments_s": [median * m / sum(mean_round) for m in mean_round],
    }


def figure1_rows(table2: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Every value Figure 1(b) draws, and how many LLMs its nDCG@10 column averages.

    ``table2`` is :func:`repro.tables.table2`'s output, computed if not given.
    """
    from repro import tables

    if table2 is None:
        table2 = tables.table2()
    averages = table2["average"]["comparators"]
    columns = [row.column for row in P.FIG1_ROWS if row.column != "dealrank"]
    llms = sorted(set.intersection(*(set(averages[c]["models"]) for c in columns)))
    if any(sorted(averages[c]["models"]) != llms for c in columns):
        raise ValueError(f"Figure 1's columns {columns} are not averaged over the same LLMs")
    dealrank = table2["average"]["dealrank"]
    out = []
    for row in P.FIG1_ROWS:
        timing = stacked_path(row.timing)
        segs = [{"s": s, "kind": "round"} for s in timing["segments_s"]]
        if row.role == "blitz" and len(segs) > 2:
            for s in segs[2:]:
                s["kind"] = "certify"
        entry: dict[str, Any] = {
            "key": row.key,
            "label": row.label,
            "role": row.role,
            "split": row.split,
            "path_s": timing["path_s"],
            "wall_s": timing["wall_s"],
            "rounds": timing["rounds"],
            "ndcg10": dealrank,
            "segments": segs,
        }
        if row.quiet:
            entry["quiet"] = True
        if row.certify is not None:
            entry["certify_share"] = certification(row.certify)["certify_share"]
        if row.table1 is not None:
            costs = tables.table1_row(P.TABLE1_BY_KEY[row.table1])[0]
            entry["rounds"] = costs["rounds"]
            entry["rounds_max"] = int(costs["rounds_max"])
        if row.column != "dealrank":
            md = averages[row.column]
            entry["ndcg10"] = dealrank + md["diff"]
            entry["ndcg_minus_dealrank"] = [md["diff"], *md["ci"]]
        out.append(entry)
    return out, len(llms)


def figure1_data(table2: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows, n_llms = figure1_rows(table2)
    chunks = next(r for r in rows if r["key"] == "blitzrank_concurrent")
    return {
        "setup": {
            "retriever": "first-stage",
            "hands": 5,
            "window": 20,
            "quota": 4,
            "head": 20,
            "certify_share": chunks["certify_share"],
        },
        "panel_b": {"unit_note": "round", "ndcg_note": f"mean of {n_llms} LLMs", "rows": rows},
    }


def local_timing() -> dict[str, dict[str, Any]]:
    """Median and 90th percentile wall-clock per DL19 query with the local model, per Table 1 row."""
    out = {}
    for row in P.TABLE1:
        if row.timing is None:
            continue
        recs = runs.timing_records(row.timing, "bm25", ("dl19",))
        out[row.key] = {
            **runs.percentiles([r["wall_s"] for r in recs]),
            "rounds_max": int(max(r["rounds"] for r in recs)),
        }
    return out


def local_latency() -> dict[str, Any]:
    """Seconds per query on four benchmarks, one query in flight: quantiles and paired ratios to DealRank."""
    arms = {key: P.runs(LOCAL_LATENCY, s, P.LOCAL, (1, 2, 3)) for key, s in LOCAL_ARMS.items()}
    runs.require([(arm, "bm25", P.FIG1_BENCHMARKS) for arm in arms.values()])
    walls = {key: runs.walls_by_query(arm, "bm25", P.FIG1_BENCHMARKS) for key, arm in arms.items()}
    out: dict[str, Any] = {}
    for key, by_benchmark in walls.items():
        pooled = [w for qs in by_benchmark.values() for ws in qs.values() for w in ws]
        per = {b: runs.percentiles([w for ws in qs.values() for w in ws]) for b, qs in by_benchmark.items()}
        out[key] = {
            **runs.percentiles(pooled),
            "per_benchmark": per,
            "p90_minus_p50_max": max(v["p90"] - v["p50"] for v in per.values()),
        }
        if key != "dealrank":
            out[key]["ratio_to_dealrank"] = runs.paired_ratio(by_benchmark, walls["dealrank"])
    return out


def t_interval(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean of ``values`` and its two-sided 95% Student t interval."""
    v = np.asarray(values, dtype=float)
    half = T_975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def load_throughput() -> dict[str, Any]:
    """Queries per second per method and load, from ``runs/load/blocks.jsonl``, and DealRank's ratios.

    Per repeat, a method's throughput is that of the fixed mix of every
    benchmark's queries, ``sum_b n_b / sum_b (n_b / X_b)``, where ``X_b`` is
    the throughput of its block on benchmark ``b``. A repeat that lacks a
    block of any method is left out.
    """
    path = records.RUNS / LOAD / "blocks.jsonl"
    if not path.exists():
        raise runs.MissingRuns([str(path)])
    blocks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    names = {P.schedule_name(s): key for key, s in LOCAL_ARMS.items()}
    by: dict[int, dict[int, dict[str, dict[str, tuple[int, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for b in blocks:
        key = names.get(b["schedule"])
        if key is not None and b.get("throughput_qps"):
            level = by[int(b["queries_in_flight"])][int(b["repeat"])]
            level[key][b["benchmark"]] = (int(b["queries"]), float(b["throughput_qps"]))
    out = {}
    for level, repeats in sorted(by.items()):
        benchmarks = sorted({b for methods in repeats.values() for cells in methods.values() for b in cells})
        complete = {
            r: methods
            for r, methods in sorted(repeats.items())
            if all(set(methods.get(key, {})) == set(benchmarks) for key in LOCAL_ARMS)
        }
        if not complete:
            continue
        qps = {
            key: [
                sum(n for n, _ in methods[key].values()) / sum(n / x for n, x in methods[key].values())
                for methods in complete.values()
            ]
            for key in LOCAL_ARMS
        }
        entry: dict[str, Any] = {
            "repeats": sorted(complete),
            "benchmarks": benchmarks,
            "qps": {key: float(np.mean(v)) for key, v in qps.items()},
            "dealrank_ratio": {},
        }
        for key in LOCAL_ARMS:
            if key == "dealrank":
                continue
            logs = np.log(np.array(qps["dealrank"]) / np.array(qps[key]))
            if len(logs) > 1:
                mean, lo, hi = t_interval(logs)
                entry["dealrank_ratio"][key] = {
                    "ratio": float(np.exp(mean)),
                    "ci": [float(np.exp(lo)), float(np.exp(hi))],
                }
            else:
                entry["dealrank_ratio"][key] = {"ratio": float(np.exp(logs[0])), "ci": None}
        out[str(level)] = entry
    return out


def _ratio(r: Mapping[str, Any] | None) -> str:
    if r is None:
        return ""
    if r["ci"] is None:
        return f"{r['ratio']:.2f}"
    return f"{r['ratio']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def render(
    fig1: list[dict[str, Any]],
    local: Mapping[str, Any],
    four: Mapping[str, Any] | None,
    load: Mapping[str, Any] | None,
) -> str:
    lines = [
        "# Latency",
        "",
        "## Figure 1(b): Gemini 3 Flash API, one query in flight",
        "",
        "| method | median critical path (s) | median wall-clock (s) | rounds | blocks (s) | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for r in fig1:
        blocks = " + ".join(f"{s['s']:.2f}" for s in r["segments"])
        cells = (
            r["label"],
            f"{r['path_s']:.2f}",
            f"{r['wall_s']:.2f}",
            f"{r['rounds']:.2f}",
            blocks,
            f"{r['ndcg10']:.1f}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Local timing (Qwen3.8-27B, DL19, wall-clock per query)",
        "",
        "| row | p50 (s) | p90 (s) | most rounds |",
        "|---|---|---|---|",
    ]
    for k, v in local.items():
        lines.append(f"| {k} | {v['p50']:.2f} | {v['p90']:.2f} | {v['rounds_max']} |")
    if four is not None:
        lines += [
            "",
            "## Local timing on four benchmarks (Qwen3.8-27B, one query in flight)",
            "",
            "| method | p50 (s) | p90 (s) | largest p90 - p50 on a benchmark (s) | seconds / DealRank's |",
            "|---|---|---|---|---|",
        ]
        for k, v in four.items():
            lines.append(
                f"| {k} | {v['p50']:.2f} | {v['p90']:.2f} | {v['p90_minus_p50_max']:.2f} | "
                f"{_ratio(v.get('ratio_to_dealrank'))} |"
            )
    if load is not None:
        lines += [
            "",
            "## Throughput under load (queries per second, fixed mix of the timed benchmarks)",
            "",
            "| in flight | method | q/s | DealRank's q/s / method's |",
            "|---|---|---|---|",
        ]
        for level, v in load.items():
            for k, q in v["qps"].items():
                lines.append(f"| {level} | {k} | {q:.3f} | {_ratio(v['dealrank_ratio'].get(k))} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="API and local latency, and throughput under load.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--figure1", action="store_true", help="print Figure 1's data as JSON")
    args = parser.parse_args(argv)
    if args.figure1:
        print(json.dumps(figure1_data(), indent=1))
        return
    (fig1, _), local = figure1_rows(), local_timing()
    try:
        four = local_latency()
    except runs.MissingRuns:
        four = None
    try:
        load = load_throughput()
    except runs.MissingRuns:
        load = None
    if args.json:
        print(json.dumps({"figure1": fig1, "local": local, "local_four_benchmarks": four, "load": load}, indent=1))
    else:
        print(render(fig1, local, four, load))


if __name__ == "__main__":
    main()
