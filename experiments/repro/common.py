"""Pieces shared by the quality runner (:mod:`repro.run`) and the timing harness (:mod:`repro.timing`)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dealrank import FakeLLM, ListwiseLLM, RerankResult
from repro.models import build_llm
from repro.records import load_pool, read_records, run_record

#: Where ``--dry-run`` writes its records, so that they never mix with real runs.
DRY_RUN_DIR = Path(__file__).resolve().parent.parent / "runs-dry-run"


def fake_score(query: str, passage: str) -> float:
    """A fixed pseudo-random relevance score, so that a dry run's FakeLLM is a consistent ranker."""
    digest = hashlib.sha256(f"{query}\x00{passage}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class Rankers:
    """Builds each (model, options) ranker once, so its concurrency limit is shared by every run that uses it.

    With ``dry_run`` every model is a :class:`~dealrank.FakeLLM` that ranks by
    :func:`fake_score` and takes ``fake_delay_s`` per call.
    """

    def __init__(self, *, dry_run: bool = False, base_url: str | None = None, fake_delay_s: float = 0.0) -> None:
        self.dry_run = dry_run
        self.base_url = base_url
        self.fake_delay_s = fake_delay_s
        self._built: dict[str, ListwiseLLM] = {}

    def get(self, name: str, options: dict[str, Any] | None = None) -> ListwiseLLM:
        options = dict(options or {})
        key = json.dumps([name, options], sort_keys=True)
        if key not in self._built:
            if self.dry_run:
                llm: ListwiseLLM = FakeLLM(
                    fake_score,
                    delay_s=self.fake_delay_s,
                    max_concurrency=int(options.get("max_concurrency", 1024)),
                )
            else:
                llm = build_llm(name, base_url=self.base_url, **options)
            self._built[key] = llm
        return self._built[key]


def load_queries(retriever: str, benchmark: str, depth: int | None, max_queries: int | None) -> list[dict[str, Any]]:
    """A pool's queries, cut to ``depth`` candidates and the first ``max_queries`` queries."""
    queries = load_pool(retriever, benchmark, depth)
    counts = Counter(str(q["qid"]) for q in queries)
    repeated = sorted(qid for qid, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"{retriever}/{benchmark}: repeated qids {repeated[:5]}")
    return queries[:max_queries] if max_queries else queries


def done_qids(path: Path) -> set[str]:
    """Qids already recorded in ``path`` (raises if one appears twice)."""
    if not path.exists():
        return set()
    return {str(r["qid"]) for r in read_records(path)}


def make_record(query: dict[str, Any], result: RerankResult, wall_s: float, **extra: Any) -> dict[str, Any]:
    """The :func:`~repro.records.run_record` of one query, with optional extra fields."""
    docids = [str(c["docid"]) for c in query["candidates"]]
    record = run_record(str(query["qid"]), docids, result, wall_s)
    record.update(extra)
    return record


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def texts(query: dict[str, Any]) -> list[str]:
    return [c["text"] for c in query["candidates"]]
