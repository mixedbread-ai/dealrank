"""Loading the paper's runs: records of an arm, per-query scores and cost views.

Everything here reads run records only. An arm's records are cached in
memory, so a script that asks for the same cell twice reads it once.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np

from repro import records
from repro.evaluate import per_query_ndcg
from repro.paper_rows import Arm, Run
from repro.stats import hier_boot

Record = dict[str, Any]


class MissingRuns(FileNotFoundError):
    """Raised when record files that a table or figure needs are absent."""

    def __init__(self, paths: Iterable[str]) -> None:
        self.paths = sorted(set(paths))
        listed = "\n  ".join(self.paths[:20]) + ("\n  ..." if len(self.paths) > 20 else "")
        super().__init__(f"{len(self.paths)} record file(s) missing; run their configs first:\n  {listed}")


def benchmarks_of(retriever: str) -> tuple[str, ...]:
    """The benchmarks the paper uses with a first stage."""
    if retriever == "dense":
        return records.DENSE_BENCHMARKS
    if retriever == "bm25-top1000":
        return records.DEEP_BENCHMARKS
    return records.BENCHMARKS


def cell_path(run: Run, retriever: str, benchmark: str, depth: int | None = None) -> Any:
    return records.Cell(run.experiment, run.llm, run.schedule, retriever, benchmark, depth, run.repeat).path


@cache
def pool_size(retriever: str, benchmark: str) -> int | None:
    """The number of queries of one first-stage pool, or ``None`` without its file."""
    path = records.pool_path(retriever, benchmark)
    if not path.exists():
        return None
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


@cache
def load_run(run: Run, retriever: str, benchmark: str, depth: int | None = None) -> dict[str, Record] | None:
    """``{qid: record}`` of one run on one pool, or ``None`` if it was not run.

    A record file that holds fewer queries than its pool (a run cut short with
    ``--max-queries``, or queries that failed and were left unrecorded) is
    reported once on standard error. The analyses still run, on the queries
    shared by the arms they compare.
    """
    path = cell_path(run, retriever, benchmark, depth)
    if not path.exists():
        return None
    loaded = records.records_by_qid(path)
    expected = pool_size(retriever, benchmark)
    if expected is not None and len(loaded) < expected:
        print(
            f"warning: {path} holds {len(loaded)} of the pool's {expected} queries; "
            "analyses use the queries shared by the arms they compare",
            file=sys.stderr,
        )
    return loaded


def load_arm(arm: Arm, retriever: str, benchmark: str, depth: int | None = None) -> list[dict[str, Record]]:
    """The records of every run of ``arm`` on one pool; raises :class:`MissingRuns` if any is absent."""
    loaded = [load_run(run, retriever, benchmark, depth) for run in arm]
    missing = [str(cell_path(run, retriever, benchmark, depth)) for run, x in zip(arm, loaded) if x is None]
    if missing:
        raise MissingRuns(missing)
    return [x for x in loaded if x is not None]


def require(arms: Iterable[tuple[Arm, str, Sequence[str]]], depth: int | None = None) -> None:
    """Raise one :class:`MissingRuns` listing every absent file of ``(arm, retriever, benchmarks)``."""
    missing = [
        str(cell_path(run, retriever, b, depth))
        for arm, retriever, benchmarks in arms
        for b in benchmarks
        for run in arm
        if not cell_path(run, retriever, b, depth).exists()
    ]
    if missing:
        raise MissingRuns(missing)


@cache
def pool_docids(retriever: str, benchmark: str) -> dict[str, list[str]]:
    """``{qid: [docid, ...]}`` of one first-stage pool, in retriever order."""
    return {str(q["qid"]): [c["docid"] for c in q["candidates"]] for q in records.load_pool(retriever, benchmark)}


@cache
def qrels(benchmark: str) -> dict[str, dict[str, int]]:
    return records.load_qrels(benchmark)


@cache
def run_scores(run: Run, retriever: str, benchmark: str, depth: int | None = None, k: int = 10) -> dict[str, float]:
    """Per-query nDCG@k (in [0, 1]) of one run."""
    recs = load_arm((run,), retriever, benchmark, depth)[0]
    return per_query_ndcg(qrels(benchmark), {q: r["ranking"] for q, r in recs.items()}, k)


def arm_scores(arm: Arm, retriever: str, benchmark: str, depth: int | None = None, k: int = 10) -> dict[str, float]:
    """Per-query nDCG@k of ``arm``, averaged over its runs, on the queries every run scored."""
    per_run = [run_scores(run, retriever, benchmark, depth, k) for run in arm]
    common = sorted(set.intersection(*(set(s) for s in per_run)))
    return {q: float(np.mean([s[q] for s in per_run])) for q in common}


def scores_by_benchmark(
    arm: Arm, retriever: str, benchmarks: Sequence[str] | None = None, depth: int | None = None, k: int = 10
) -> dict[str, dict[str, float]]:
    """``{benchmark: {qid: nDCG@k}}`` of ``arm``."""
    return {b: arm_scores(arm, retriever, b, depth, k) for b in benchmarks or benchmarks_of(retriever)}


def macro(per_benchmark: Mapping[str, Mapping[str, float]], keep: Mapping[str, Iterable[str]] | None = None) -> float:
    """Unweighted mean over benchmarks of the query means (optionally over the ``keep`` queries only)."""
    means = []
    for b, scores in per_benchmark.items():
        qs = sorted(keep[b]) if keep is not None else list(scores)
        means.append(np.mean([scores[q] for q in qs]))
    return float(np.mean(means))


# ---- costs from the trace --------------------------------------------------------
def round_latencies(record: Mapping[str, Any]) -> list[float]:
    """The slowest call of each round, ignoring calls answered from a cache (0 for an all-cached round)."""
    out = []
    for calls in record.get("trace") or []:
        lat = [c["latency_s"] for c in calls if not c.get("cached")]
        out.append(max(lat) if lat else 0.0)
    return out


def critical_path(record: Mapping[str, Any]) -> float:
    """Sum over rounds of the slowest uncached call: the latency with spare server capacity."""
    return float(sum(round_latencies(record)))


def tokens_k(record: Mapping[str, Any]) -> float:
    return (record["prompt_tokens"] + record["completion_tokens"]) / 1000


def leading_disjoint(calls: Sequence[Sequence[int]]) -> int:
    """How many calls, from the first, hold pairwise disjoint passages."""
    seen: set[int] = set()
    k = 0
    for members in calls:
        s = set(members)
        if s & seen or len(s) != len(members):
            break
        seen |= s
        k += 1
    return k


def first_calls_concurrent(record: Mapping[str, Any]) -> tuple[int, float]:
    """``(rounds, critical path)`` of a BlitzRank record with its first, mutually disjoint calls issued at once.

    BlitzRank's first calls (the cover) depend on nothing, so they can run
    as one round; every later call keeps its own round and recorded latency.
    A record whose first round already holds several calls is returned as
    recorded. The cover is read from ``diagnostics.calls``, the passages of
    each call.
    """
    widths = record.get("round_widths") or []
    lat = round_latencies(record)
    if record["calls"] == 0:
        return 0, 0.0
    if widths and widths[0] > 1:
        return len(widths), float(sum(lat))
    k = leading_disjoint(record["diagnostics"]["calls"])
    if k == 0:
        return record["calls"], float(sum(lat))
    return 1 + record["calls"] - k, float(max(lat[:k]) + sum(lat[k:]))


def rounds_of(record: Mapping[str, Any], concurrent: bool = False) -> int:
    return first_calls_concurrent(record)[0] if concurrent else int(record["rounds"])


def path_of(record: Mapping[str, Any], concurrent: bool = False) -> float:
    return first_calls_concurrent(record)[1] if concurrent else critical_path(record)


def full_pool(records_by_qid: Mapping[str, Record], n: int = 100) -> list[Record]:
    """The records of the queries whose pool holds exactly ``n`` candidates."""
    return [r for r in records_by_qid.values() if r["n_candidates"] == n]


def timing_records(arm: Arm, retriever: str, benchmarks: Sequence[str], depth: int | None = None) -> list[Record]:
    """Every record of a timing arm, pooled over its repeats and benchmarks."""
    return [r for b in benchmarks for recs in load_arm(arm, retriever, b, depth) for r in recs.values()]


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """Median and 90th percentile (numpy's linear interpolation)."""
    v = np.asarray(values, dtype=float)
    return {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": int(v.size)}


def paired_ratio(
    arm: Mapping[str, Mapping[str, Sequence[float]]], reference: Mapping[str, Mapping[str, Sequence[float]]]
) -> dict[str, Any]:
    """Geometric mean over queries of ``arm / reference``, with the hierarchical bootstrap of the log ratios.

    Both arguments map benchmark to ``{qid: [value per repeat]}``. Each
    query's values are averaged over its repeats before the ratio is taken;
    the mean of the log ratios is the macro mean over benchmarks.
    """
    logs = {}
    for b in sorted(set(arm) & set(reference)):
        qs = sorted(set(arm[b]) & set(reference[b]))
        if qs:
            logs[b] = np.log([np.mean(arm[b][q]) / np.mean(reference[b][q]) for q in qs])
    boot = hier_boot(logs)
    return {
        "ratio": float(np.exp(boot["diff"])),
        "ci": [float(np.exp(boot["ci_lo"])), float(np.exp(boot["ci_hi"]))],
        "benchmarks": boot["n_datasets"],
        "queries": boot["n_queries"],
    }


def walls_by_query(
    arm: Arm, retriever: str, benchmarks: Sequence[str], depth: int | None = None
) -> dict[str, dict[str, list[float]]]:
    """``{benchmark: {qid: [wall seconds per repeat]}}`` of a timing arm."""
    out: dict[str, dict[str, list[float]]] = {}
    for b in benchmarks:
        out[b] = {}
        for recs in load_arm(arm, retriever, b, depth):
            for q, r in recs.items():
                out[b].setdefault(q, []).append(float(r["wall_s"]))
    return out
