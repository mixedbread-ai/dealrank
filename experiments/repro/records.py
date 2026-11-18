"""Benchmarks, file layout and the per-query run record shared by every script.

Layout, relative to the ``experiments/`` directory (override with the
``DEALRANK_DATA``, ``DEALRANK_RUNS`` and ``DEALRANK_OUT`` environment variables):

* ``data/pools/<retriever>/<benchmark>.jsonl``: one first-stage pool per line,
  ``{"qid", "query", "candidates": [{"docid", "score", "text"}]}`` in
  retriever order. ``retriever`` is ``bm25`` (top 100), ``dense`` (top 100)
  or ``bm25-top1000``.
* ``data/qrels/<benchmark>.txt``: TREC qrels, shared by every retriever.
* ``runs/<experiment>/<llm>/<schedule>/<retriever>/<benchmark>[-n<depth>][.r<repeat>].jsonl``:
  one :func:`run_record` per query.
* ``out/``: tables, figures and analysis summaries written by ``repro``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dealrank.result import RerankResult

EXPERIMENTS = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DEALRANK_DATA", EXPERIMENTS / "data"))
RUNS = Path(os.environ.get("DEALRANK_RUNS", EXPERIMENTS / "runs"))
OUT = Path(os.environ.get("DEALRANK_OUT", EXPERIMENTS / "out"))

#: The ten benchmarks of the paper, in table order.
BENCHMARKS: tuple[str, ...] = (
    "dl19",
    "dl20",
    "covid",
    "nfcorpus",
    "signal",
    "news",
    "robust04",
    "touche",
    "dbpedia",
    "scifact",
)

#: Column labels used in the tables.
LABELS: dict[str, str] = {
    "dl19": "DL19",
    "dl20": "DL20",
    "covid": "COVID",
    "nfcorpus": "NFC",
    "signal": "Signal",
    "news": "News",
    "robust04": "Rob04",
    "touche": "Touch\\'{e}",
    "dbpedia": "DBP",
    "scifact": "SciF",
}

#: Benchmarks with a dense pool (the corpora of Signal and News are not public).
DENSE_BENCHMARKS: tuple[str, ...] = tuple(b for b in BENCHMARKS if b not in {"signal", "news"})

#: Benchmarks with a BM25 top-1,000 pool, used for the 500 and 1,000 candidate runs.
DEEP_BENCHMARKS: tuple[str, ...] = ("dl19", "dl20", "dbpedia", "scifact")


@dataclass(frozen=True)
class Cell:
    """One (experiment, LLM, schedule, pool) combination and its record file."""

    experiment: str
    llm: str
    schedule: str
    retriever: str
    benchmark: str
    depth: int | None = None
    repeat: int | None = None

    @property
    def path(self) -> Path:
        stem = self.benchmark + (f"-n{self.depth}" if self.depth else "") + (f".r{self.repeat}" if self.repeat else "")
        return RUNS / self.experiment / self.llm / self.schedule / self.retriever / f"{stem}.jsonl"


def pool_path(retriever: str, benchmark: str) -> Path:
    return DATA / "pools" / retriever / f"{benchmark}.jsonl"


def qrels_path(benchmark: str) -> Path:
    return DATA / "qrels" / f"{benchmark}.txt"


def load_pool(retriever: str, benchmark: str, depth: int | None = None) -> list[dict[str, Any]]:
    """Queries of one pool, candidates in retriever order, cut to ``depth`` if given."""
    with open(pool_path(retriever, benchmark)) as handle:
        queries = [json.loads(line) for line in handle if line.strip()]
    if depth is not None:
        queries = [{**q, "candidates": q["candidates"][:depth]} for q in queries]
    return queries


def load_qrels(benchmark: str) -> dict[str, dict[str, int]]:
    """``{qid: {docid: grade}}`` from a TREC qrels file."""
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path(benchmark)) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, grade = parts
                qrels.setdefault(qid, {})[docid] = int(grade)
    return qrels


def run_record(qid: str, docids: list[str], result: RerankResult, wall_s: float) -> dict[str, Any]:
    """The JSON record of one reranked query.

    ``ranking`` holds docids, best first. ``trace`` lists every call by round,
    with its permutation, tokens, latency and wall-clock stamps, so that
    timing and cut analyses need nothing but the records.
    """
    return {
        "qid": qid,
        "ranking": result.reorder(docids),
        "n_candidates": len(docids),
        "rounds": result.num_rounds,
        "round_widths": result.round_widths,
        "calls": result.num_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "critical_path_s": round(result.critical_path_s(), 4),
        "wall_s": round(wall_s, 4),
        "repair_rate": round(result.repair_rate, 4),
        "trace": [[call.to_dict() for call in calls] for calls in result.rounds],
        "diagnostics": result.diagnostics,
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records of one cell; raises on a repeated qid."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["qid"] in seen:
                raise ValueError(f"{path}: qid {record['qid']} appears twice")
            seen.add(record["qid"])
            records.append(record)
    return records


def records_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    """Records of one cell keyed by qid (as a string)."""
    return {str(r["qid"]): r for r in read_records(path)}


def find_repeats(
    experiment: str, llm: str, schedule: str, retriever: str, benchmark: str, depth: int | None = None
) -> dict[int | None, Path]:
    """Every record file of one cell across repeats, keyed by repeat (``None``: no suffix)."""
    stem = benchmark + (f"-n{depth}" if depth else "")
    root = RUNS / experiment / llm / schedule / retriever
    found: dict[int | None, Path] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"{stem}*.jsonl")):
        rest = path.name[len(stem) : -len(".jsonl")]
        if rest == "":
            found[None] = path
        elif rest.startswith(".r") and rest[2:].isdigit():
            found[int(rest[2:])] = path
    return found


def iter_cells(experiment: str) -> Iterator[tuple[Cell, Path]]:
    """Every record file under ``runs/<experiment>``, parsed back into a :class:`Cell`."""
    root = RUNS / experiment
    for path in sorted(root.glob("*/*/*/*.jsonl")):
        llm, schedule, retriever = path.parts[-4:-1]
        stem = path.stem
        repeat = None
        if ".r" in stem:
            stem, rep = stem.rsplit(".r", 1)
            repeat = int(rep)
        depth = None
        if "-n" in stem:
            stem, d = stem.rsplit("-n", 1)
            depth = int(d)
        yield Cell(experiment, llm, schedule, retriever, stem, depth, repeat), path
