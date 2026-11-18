"""The timing harness interleaves methods per query, stamps every call and holds the load."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections import defaultdict
from pathlib import Path

import pytest

from repro import records
from repro.common import Rankers
from repro.config import load_config
from repro.timing import method_order, run_timing

DELAY = 0.02

INTERLEAVED = """\
# A small timing config for the tests.
llms: [qwen3.8-27b]
retriever: bm25
benchmarks: [dl19, dl20]
schedules:
  - {type: dealrank, window: 20, quota: 4}
  - {type: sliding_window, window: 20, stride: 10}
  - {type: blitzrank, window: 20, top_m: 10, input_order: retriever, concurrent_cover: true}
timing: {repeats: 2, seed: 5, warmup_queries: 1}
"""

LOAD = """\
# A small load config for the tests.
llms: [qwen3.8-27b]
retriever: bm25
benchmarks: [dl19]
schedules:
  - {type: dealrank, window: 20, quota: 4}
  - {type: sliding_window, window: 20, stride: 10}
timing: {repeats: [3], seed: 5, queries_in_flight: 4}
"""


def _run(tmp_path: Path, text: str, rankers: Rankers, name: str) -> tuple[object, object]:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    experiment = load_config(path)
    totals = asyncio.run(run_timing(experiment, rankers, log=lambda _s: None))
    return experiment, totals


def test_methods_are_interleaved_per_query_and_every_call_is_stamped(workspace: Path, tmp_path: Path) -> None:
    rankers = Rankers(dry_run=True, fake_delay_s=DELAY)
    experiment, totals = _run(tmp_path, INTERLEAVED, rankers, "interleaved")
    assert totals.failed == 0
    by_query: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for cell in experiment.cells():
        assert cell.repeat in (1, 2)
        for rec in records.read_records(cell.path):
            by_query[(cell.repeat, cell.benchmark, rec["qid"])].append({**rec, "schedule": cell.schedule})

    assert len(by_query) == 2 * 2 * 3
    names = ["dealrank-w20-q4", "sliding-window-w20-s10", "blitzrank-w20-m10-concurrent"]  # config order
    orders = set()
    for (repeat, benchmark, qid), recs in by_query.items():
        assert len(recs) == 3
        recs.sort(key=lambda r: r["timing"]["slot"])
        order = recs[0]["timing"]["order"]
        assert order == method_order(5, repeat, benchmark, qid, names)
        assert [r["schedule"] for r in recs] == order
        orders.add(tuple(order))
        for before, after in itertools.pairwise(recs):
            assert after["timing"]["started_at"] >= before["timing"]["ended_at"]  # one query in flight
        for rec in recs:
            assert rec["timing"]["queries_in_flight"] == 1
            calls = [c for r in rec["trace"] for c in r]
            assert all(c["started_at"] is not None and c["started_at"] <= c["ended_at"] for c in calls)
            assert rec["wall_s"] >= rec["critical_path_s"]
    assert len(orders) > 1

    dealrank = next(r for recs in by_query.values() for r in recs if r["schedule"] == "dealrank-w20-q4")
    hands = dealrank["trace"][0]
    assert len(hands) == 5
    assert max(c["started_at"] for c in hands) < min(c["ended_at"] for c in hands)  # concurrent
    assert 2 * DELAY <= dealrank["critical_path_s"] < 2 * DELAY + 0.2
    window = next(r for recs in by_query.values() for r in recs if r["schedule"].startswith("sliding"))
    assert window["critical_path_s"] >= 9 * DELAY


def test_a_rerun_skips_recorded_queries(workspace: Path, tmp_path: Path) -> None:
    rankers = Rankers(dry_run=True)
    _, first = _run(tmp_path, INTERLEAVED, rankers, "interleaved")
    _, second = _run(tmp_path, INTERLEAVED, rankers, "interleaved")
    assert second.reranked == 0 and second.skipped == first.reranked


def test_load_blocks_keep_the_queries_in_flight(workspace: Path, tmp_path: Path) -> None:
    rankers = Rankers(dry_run=True, fake_delay_s=0.01)
    experiment, totals = _run(tmp_path, LOAD, rankers, "load")
    assert totals.failed == 0
    blocks = [json.loads(line) for line in (workspace / "load" / "blocks.jsonl").read_text().splitlines()]
    assert sorted(b["schedule"] for b in blocks) == ["dealrank-w20-q4", "sliding-window-w20-s10"]
    for block in blocks:
        assert block["queries"] == block["queries_ok"] == 3
        assert block["queries_in_flight"] == 4 and block["repeat"] == 3
        assert block["throughput_qps"] > 0
    for cell in experiment.cells():
        recs = records.read_records(cell.path)
        assert len(recs) == 3 and cell.path.name == "dl19.r3.jsonl"
        assert {r["timing"]["queries_in_flight"] for r in recs} == {4}
    llm = rankers.get("qwen3.8-27b")
    assert llm.max_in_flight > 5  # the hands of several DealRank queries at once


def test_a_partial_load_block_is_refused(workspace: Path, tmp_path: Path) -> None:
    rankers = Rankers(dry_run=True)
    experiment, _ = _run(tmp_path, LOAD, rankers, "load")
    path = experiment.cells()[0].path
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError, match="load block"):
        _run(tmp_path, LOAD, rankers, "load")
