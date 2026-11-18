"""Per-query nDCG@10, as the paper computes it.

A ranking (docids, best first) becomes a scored run by giving the document at
rank ``i`` of ``n`` the score ``n - i``, and nDCG@10 is computed by
``ir_measures`` (trec_eval's definition: gain = graded relevance, negative
grades count as 0, discount ``1 / log2(rank + 1)``, ideal DCG over all of the
query's judged documents). Only queries that have both a ranking and qrels
are scored. A query with an empty ranking (an empty first-stage pool) scores
0, as does a query whose qrels hold no positive grade.

:func:`ndcg_at_k` is an independent pure-Python implementation of the same
definition. It is used when ``ir_measures`` is not installed and is checked
against it in the tests.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]


def to_scored_run(run: Mapping[str, Sequence[str]]) -> dict[str, dict[str, float]]:
    """Ranked docids to descending scores, as ``ir_measures`` expects.

    A docid listed twice keeps the score of its last (lower) position.
    """
    return {qid: {d: float(len(docs) - i) for i, d in enumerate(docs)} for qid, docs in run.items()}


def ndcg_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k of one ranking (trec_eval's definition), without dependencies."""
    scores = {d: len(ranking) - i for i, d in enumerate(ranking)}
    ranked = sorted(scores, key=lambda d: -scores[d])[:k]
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    return sum(max(grades.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ranked)) / idcg


def per_query_ndcg(qrels: Qrels, run: Mapping[str, Sequence[str]], k: int = 10) -> dict[str, float]:
    """``{qid: nDCG@k}`` (in [0, 1]) for every query of ``run`` that has qrels."""
    run = {str(q): list(docs) for q, docs in run.items()}
    judged = {q: dict(g) for q, g in qrels.items() if q in run}
    try:
        import ir_measures
    except ImportError:
        _warn_fallback()
        return {q: ndcg_at_k(run[q], judged[q], k) for q in judged}
    measure = ir_measures.nDCG @ k
    return {m.query_id: float(m.value) for m in ir_measures.iter_calc([measure], judged, to_scored_run(run))}


_warned = False


def _warn_fallback() -> None:
    """Say once that the pure-Python nDCG is in use (the paper's numbers use ``ir_measures``)."""
    global _warned
    if not _warned:
        _warned = True
        print("warning: ir_measures is not installed; scoring with the pure-Python nDCG", file=sys.stderr)


def ndcg_x100(qrels: Qrels, records: Mapping[str, Mapping[str, object]], k: int = 10) -> dict[str, float]:
    """Per-query nDCG@k x 100 of run records keyed by qid."""
    run = {q: list(r["ranking"]) for q, r in records.items()}  # type: ignore[call-overload]
    return {q: 100.0 * v for q, v in per_query_ndcg(qrels, run, k).items()}
