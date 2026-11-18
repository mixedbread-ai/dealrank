"""Analyses on dry-run records: they run end to end, and say when a record file is incomplete."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from repro import run, runs  # noqa: E402
from repro.analysis import quota  # noqa: E402


def dry_run(config: str, runs_dir: Path, *extra: str) -> int:
    args = run.build_parser().parse_args([config, "--dry-run", "--runs-dir", str(runs_dir), *extra])
    return asyncio.run(run.amain(args, log=lambda _s: None))


def _clear_caches() -> None:
    for f in (runs.load_run, runs.run_scores, runs.qrels, runs.pool_docids, runs.pool_size):
        f.cache_clear()


def test_quota_analysis_runs_on_dry_run_records(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _clear_caches()
    try:
        assert dry_run("quota_sweep", workspace) == 0
        quota.main(["--json"])
        out = capsys.readouterr()
        table = json.loads(out.out)
        assert set(table) == {"20x4", "10x2", "5x1"}
        for by_retriever in table.values():
            for cell in by_retriever.values():
                lo, hi = cell["random_minus_dealt"]["ci"]
                assert lo <= cell["random_minus_dealt"]["diff"] <= hi
                assert cell["random_minus_dealt"]["benchmarks"] in (8, 10)
        # Complete record files: no incomplete-run warning (an optional-dependency notice may still appear).
        assert "of the pool's" not in out.err
    finally:
        _clear_caches()


def test_incomplete_record_file_is_reported(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _clear_caches()
    try:
        assert dry_run("quota_sweep", workspace, "--benchmarks", "dl19", "--max-queries", "2") == 0
        capsys.readouterr()
        path = next(workspace.glob("quota_sweep/*/dealrank-w20-q4/bm25/dl19.jsonl"))
        arm = next(a for a in quota_arms() if runs.cell_path(a, "bm25", "dl19") == path)
        loaded = runs.load_run(arm, "bm25", "dl19")
        assert loaded is not None and len(loaded) == 2
        err = capsys.readouterr().err
        assert "holds 2 of the pool's 3 queries" in err
    finally:
        _clear_caches()


def quota_arms() -> list:
    from repro import paper_rows as P

    return [P.Run(quota.EXPERIMENT, quota.dealt(w, q), P.LOCAL) for w, q in quota.SETTINGS]
