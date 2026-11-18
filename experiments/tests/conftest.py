"""Shared fixtures: tiny first-stage pools and qrels for every benchmark and retriever.

``import repro`` works when the tests run from the repository root, since
this file puts ``experiments/`` on the path.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from repro import records  # noqa: E402

#: Queries per fixture pool, and candidates per query (the deep pools are kept short so tests stay fast).
QUERIES = 3
DEPTH = {"bm25": 100, "dense": 100, "bm25-top1000": 60}
WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu")


def write_fixture_data(root: Path) -> None:
    """Write pools and qrels under ``root`` in the layout of :mod:`repro.records`.

    Each benchmark's second query has a short pool (37 candidates) and the
    NFCorpus BM25 pool has a query with no candidate, as the real pools do.
    """
    rng = random.Random(0)
    pools = {"bm25": records.BENCHMARKS, "dense": records.DENSE_BENCHMARKS, "bm25-top1000": records.DEEP_BENCHMARKS}
    qrels: dict[str, list[str]] = {}
    for retriever, benchmarks in pools.items():
        for benchmark in benchmarks:
            lines = []
            for q in range(QUERIES):
                qid = f"{benchmark}-q{q}"
                n = DEPTH[retriever] if q != 1 else 37
                if retriever == "bm25" and benchmark == "nfcorpus" and q == 2:
                    n = 0
                candidates = [
                    {
                        "docid": f"{benchmark}-d{q}-{i}",
                        "score": float(DEPTH[retriever] - i),
                        "text": f"passage {i} for {qid}: " + " ".join(rng.choices(WORDS, k=12)),
                    }
                    for i in range(n)
                ]
                lines.append(json.dumps({"qid": qid, "query": f"query {q} of {benchmark}", "candidates": candidates}))
            path = root / "pools" / retriever / f"{benchmark}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
    for benchmark in records.BENCHMARKS:
        qrels[benchmark] = [
            f"{benchmark}-q{q} 0 {benchmark}-d{q}-{i} {rng.choice([0, 1, 2])}"
            for q in range(QUERIES)
            for i in range(30)
        ]
        path = root / "qrels" / f"{benchmark}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(qrels[benchmark]) + "\n")


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_fixture_data(root)
    return root


@pytest.fixture
def workspace(fixture_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the record layout at the fixture pools and a fresh, empty runs directory."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(records, "DATA", fixture_data)
    monkeypatch.setattr(records, "RUNS", runs)
    yield runs
