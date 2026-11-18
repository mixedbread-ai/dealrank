"""A dry run of every config, on two queries per pool, writes well-formed records."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repro import records, run
from repro.config import CONFIGS, load_config
from repro.schedules import package_available, uses_package

EXPERIMENT_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if p.name != "models.yaml")
RECORD_KEYS = {
    "qid",
    "ranking",
    "n_candidates",
    "rounds",
    "round_widths",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "critical_path_s",
    "wall_s",
    "repair_rate",
    "trace",
    "diagnostics",
}


def dry_run(config: str | Path, runs: Path, *extra: str) -> int:
    args = run.build_parser().parse_args([str(config), "--dry-run", "--runs-dir", str(runs), *extra])
    return asyncio.run(run.amain(args, log=lambda _s: None))


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_dry_run_writes_well_formed_records(path: Path, workspace: Path) -> None:
    assert dry_run(path, workspace, "--max-queries", "2") == 0
    experiment = load_config(path)
    expected = [a for a in experiment.arms() if package_available() or not uses_package(a.spec)]
    assert expected
    timed = {a.cell(experiment.name).path for block in experiment.timing for a in block.arms()}
    for arm in expected:
        cell = arm.cell(experiment.name)
        pool = {q["qid"]: q for q in records.load_pool(arm.retriever, arm.benchmark, arm.depth)[:2]}
        recs = records.read_records(cell.path)
        assert {r["qid"] for r in recs} == set(pool), cell.path
        for rec in recs:
            assert set(rec) >= RECORD_KEYS
            docids = [c["docid"] for c in pool[rec["qid"]]["candidates"]]
            assert sorted(rec["ranking"]) == sorted(docids)
            assert rec["n_candidates"] == len(docids)
            assert rec["round_widths"] == [len(r) for r in rec["trace"]]
            assert rec["calls"] == sum(rec["round_widths"])
            for call in (c for r in rec["trace"] for c in r):
                assert call["started_at"] <= call["ended_at"]
            if cell.path in timed:
                assert rec["timing"]["repeat"] == arm.repeat
    found = {p for p in (workspace / experiment.name).glob("*/*/*/*.jsonl")}
    assert found == {a.cell(experiment.name).path for a in expected}
