"""Runs resume where they stopped, and failed queries are retried rather than recorded."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repro import records, run
from repro.config import load_config

CONFIG = "table1_gemini"


def dry_run(runs: Path, *benchmarks: str) -> run.Totals:
    """A dry run of the config's quality grids on ``benchmarks``."""
    run.set_runs_dir(runs)
    experiment = load_config(CONFIG)
    return asyncio.run(run.run_grids(experiment, run.Rankers(dry_run=True), benchmarks=benchmarks, log=lambda _s: None))


def test_a_second_run_skips_every_recorded_query(workspace: Path) -> None:
    first = dry_run(workspace, "dl19", "dl20")
    assert first.reranked > 0 and first.failed == 0
    second = dry_run(workspace, "dl19", "dl20")
    assert second.reranked == 0
    assert second.skipped == first.reranked


def test_a_missing_query_is_rerun_alone(workspace: Path) -> None:
    dry_run(workspace, "dl19")
    path = next((workspace / CONFIG).glob("*/dealrank-w20-q4/bm25/dl19.jsonl"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    again = dry_run(workspace, "dl19")
    assert again.reranked == 1
    assert len(records.read_records(path)) == len(lines)


def test_a_failed_query_is_not_recorded_and_is_retried(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from repro import common

    real = common.fake_score

    def flaky(query: str, passage: str) -> float:
        if query == "query 1 of dl19":
            raise RuntimeError("provider error")
        return real(query, passage)

    monkeypatch.setattr(common, "fake_score", flaky)
    failed = dry_run(workspace, "dl19")
    assert failed.failed > 0
    path = next((workspace / CONFIG).glob("*/dealrank-w20-q4/bm25/dl19.jsonl"))
    assert "dl19-q1" not in {r["qid"] for r in records.read_records(path)}

    monkeypatch.setattr(common, "fake_score", real)
    retried = dry_run(workspace, "dl19")
    assert retried.failed == 0 and retried.reranked == failed.failed
    assert "dl19-q1" in {r["qid"] for r in records.read_records(path)}


def test_a_repeated_qid_in_a_record_file_is_refused(workspace: Path) -> None:
    dry_run(workspace, "dl19")
    path = next((workspace / CONFIG).glob("*/dealrank-w20-q4/bm25/dl19.jsonl"))
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="appears twice"):
        dry_run(workspace, "dl19")
