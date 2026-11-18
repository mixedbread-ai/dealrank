"""Normalising the downloaded pools, without downloading anything."""

from __future__ import annotations

import doctest
import json
from pathlib import Path

import pytest

import repro.data
import repro.dense
import repro.models
import repro.timing
from repro import records
from repro.data import QUERY_COUNTS, check, normalise_rank_llm, write_pool


def test_rank_llm_files_are_normalised() -> None:
    line = json.dumps(
        {
            "query": {"qid": 19335, "text": "anthropological definition of environment"},
            "candidates": [
                {"docid": 7, "score": 12.5, "doc": {"contents": "MS MARCO text "}},
                {"docid": "d2", "score": 3, "doc": {"title": "Title", "text": "BEIR text"}},
                {"docid": "d3", "doc": {"title": "", "passage": "passage text"}},
            ],
        }
    )
    (pool,) = normalise_rank_llm([line, ""])
    assert pool["qid"] == "19335"
    assert [c["docid"] for c in pool["candidates"]] == ["7", "d2", "d3"]
    assert [c["text"] for c in pool["candidates"]] == ["MS MARCO text", "Title BEIR text", "passage text"]
    assert pool["candidates"][2]["score"] == 0.0


def test_write_pool_checks_counts_and_repeats(tmp_path: Path) -> None:
    pool = [{"qid": "a", "query": "q", "candidates": []}]
    assert write_pool(tmp_path / "p.jsonl", pool, 1) == 1
    with pytest.raises(ValueError, match="expected 2"):
        write_pool(tmp_path / "p.jsonl", pool, 2)
    with pytest.raises(ValueError, match="twice"):
        write_pool(tmp_path / "p.jsonl", pool * 2)


def test_query_counts_add_up_to_the_paper() -> None:
    assert sum(QUERY_COUNTS.values()) == 1622
    assert set(QUERY_COUNTS) == set(records.BENCHMARKS)
    assert len(records.BENCHMARKS) == 10 and len(records.DENSE_BENCHMARKS) == 8
    assert sum(QUERY_COUNTS[b] for b in records.DENSE_BENCHMARKS) == 1468


def test_check_reports_every_pool(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check() == 1  # the fixture pools hold three queries each
    out = capsys.readouterr().out
    assert "pools/dense/dl19.jsonl" in out and "pools/bm25-top1000/scifact.jsonl" in out


@pytest.mark.parametrize("module", [repro.data, repro.dense, repro.models, repro.timing])
def test_doctests(module: object) -> None:
    result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
