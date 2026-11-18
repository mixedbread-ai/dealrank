"""Every record file the tables, figures and analyses read is written by some config."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from repro import paper_rows as P
from repro import records
from repro.analysis import depth, latency, quota, retention, seed_stability
from repro.config import CONFIGS, load_config
from repro.runs import benchmarks_of

EXPERIMENT_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if p.name != "models.yaml")


def produced() -> set[records.Cell]:
    return {cell for path in EXPERIMENT_CONFIGS for cell in load_config(path).cells()}


def cells(
    arm: P.Arm, retriever: str, benchmarks: tuple[str, ...] | None = None, d: int | None = None
) -> Iterator[records.Cell]:
    for run in arm:
        for benchmark in benchmarks or benchmarks_of(retriever):
            yield records.Cell(run.experiment, run.llm, run.schedule, retriever, benchmark, d, run.repeat)


def table1() -> Iterator[records.Cell]:
    for row in P.TABLE1:
        yield from cells(row.bm25, "bm25")
        if row.timing is not None:
            yield from cells(row.timing, "bm25", ("dl19",))


def table2() -> Iterator[records.Cell]:
    for llm, _ in P.MODELS:
        yield from cells(P.table2_arm(llm, "dealrank"), "bm25")
        for column, _ in P.COMPARATORS + P.UNPRINTED:
            yield from cells(P.table2_arm(llm, column), "bm25")
    for llm, _ in P.EXTRA_MODELS:
        for column in ("dealrank", *P.EXTRA_CELLS[llm]):
            yield from cells(P.table2_arm(llm, column), "bm25")


def tourrank() -> Iterator[records.Cell]:
    for llm in P.TOURRANK_LLMS:
        yield from cells(P.tourrank_arm(llm), "bm25")
        yield from cells(P.table2_arm(llm, "dealrank"), "bm25")


def table3() -> Iterator[records.Cell]:
    for d in depth.DEPTHS:
        for column in depth.QUALITY:
            arm, retriever, arg = depth.quality_arm(d, column)
            yield from cells(arm, retriever, depth.BENCHMARKS, arg)
        for column in depth.COLUMNS:
            s = depth.specs(d)[column]
            if d == 100:
                yield from cells(P.timed(s), "bm25", (depth.TIMED,))
            else:  # seconds are read from whichever timing repeats exist; the first must
                yield from cells((P.Run(P.T3, s, P.LOCAL, 1),), P.DEEP, (depth.TIMED,), d)


def figure1() -> Iterator[records.Cell]:
    for row in P.FIG1_ROWS:
        yield from cells(row.timing, "bm25", P.FIG1_BENCHMARKS)
        if row.certify is not None:
            yield from cells(row.certify, "bm25")


def figure3() -> Iterator[records.Cell]:
    for row in P.FIG3_ROWS:
        for arms in (row.dealt, row.other):
            for retriever, arm in arms.items():
                yield from cells(arm, retriever)


def cuts() -> Iterator[records.Cell]:
    yield from cells(P.three_runs(P.DEALRANK), "bm25")
    yield from cells(P.ten_seeds(), "bm25")
    yield from cells((P.Run(P.T1, P.WINDOW), P.Run(P.T1, P.CHUNKS)), "bm25")
    for s in (P.WINDOW, P.package(20), P.CHUNKS, P.DEALRANK):
        yield from cells((P.Run(P.DENSE, s),), "dense")


def retention_curve() -> Iterator[records.Cell]:
    for _, _, retriever, run in retention.SETTINGS:
        yield from cells((run, P.Run(P.T1 if retriever == "bm25" else P.DENSE, P.WINDOW)), retriever)


def seed_stability_arms() -> Iterator[records.Cell]:
    for arm in seed_stability.ARMS.values():
        yield from cells(arm, "bm25")


def quota_sweep() -> Iterator[records.Cell]:
    for window, q in quota.SETTINGS:
        arm = (
            P.Run(quota.EXPERIMENT, quota.dealt(window, q), P.LOCAL),
            *(P.Run(quota.EXPERIMENT, P.random_partition(s, window=window, quota=q), P.LOCAL) for s in (0, 1, 2)),
        )
        for retriever in ("bm25", "dense"):
            yield from cells(arm, retriever)


def local_latency() -> Iterator[records.Cell]:
    for s in latency.LOCAL_ARMS.values():
        yield from cells(P.runs(latency.LOCAL_LATENCY, s, P.LOCAL, (1, 2, 3)), "bm25", P.FIG1_BENCHMARKS)


READERS = {
    "table1": table1,
    "table2": table2,
    "table3": table3,
    "figure1": figure1,
    "figure3": figure3,
    "tourrank": tourrank,
    "cuts": cuts,
    "retention": retention_curve,
    "seed_stability": seed_stability_arms,
    "quota_sweep": quota_sweep,
    "local_latency": local_latency,
}


@pytest.mark.parametrize("reader", READERS, ids=str)
def test_every_cell_read_is_produced(reader: str) -> None:
    missing = sorted(
        {str(c.path.relative_to(records.RUNS)) for c in READERS[reader]()}
        - {str(c.path.relative_to(records.RUNS)) for c in produced()}
    )
    assert not missing, "\n".join(missing[:20])


def test_load_blocks_cover_every_method_and_benchmark() -> None:
    experiment = load_config(CONFIGS / "load.yaml")
    # The methods repro.analysis.latency.load_throughput reports.
    wanted = {P.schedule_name(s) for s in latency.LOCAL_ARMS.values()}
    for block in experiment.timing:
        assert block.queries_in_flight > 1
        assert {a.schedule for a in block.arms()} == wanted
        assert {a.benchmark for a in block.arms()} == set(P.FIG1_BENCHMARKS)
        assert len(block.repeats) == 3
    assert {block.queries_in_flight for block in experiment.timing} == {8, 32}
    # Repeat numbers name the record files, so the two levels must not share any.
    repeats = [r for block in experiment.timing for r in block.repeats]
    assert len(repeats) == len(set(repeats))


def test_every_random_seed_of_table1_is_configured() -> None:
    seeds = {dict(run.spec)["seed"] for run in P.ten_seeds()}
    assert seeds == set(range(10))
