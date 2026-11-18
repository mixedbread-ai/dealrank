"""Run an experiment config: every (LLM, schedule, pool) cell, resumably.

    python -m repro.run configs/table1_gemini.yaml
    python -m repro.run configs/table1_gemini.yaml --dry-run --max-queries 2

Cells run one after another; within a cell, ``query_concurrency`` queries are
in flight. Each reranked query is appended to its cell's record file
(:class:`repro.records.Cell`) as soon as it finishes, so an interrupted run
resumes where it stopped: queries already recorded are skipped. A query that
fails is logged and left unrecorded, so the next run retries it, and the
command exits with status 1. A config's ``timing`` blocks run after its
quality grids (see :mod:`repro.timing`).

``--dry-run`` replaces every model with an offline :class:`~dealrank.FakeLLM`
and writes to ``runs-dry-run/``, which checks a config end to end for free.
Cells of the released BlitzRank package are skipped in a dry run when the
package is not installed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from repro import records
from repro.common import DRY_RUN_DIR, Rankers, append_record, done_qids, load_queries, make_record, texts
from repro.config import Arm, Experiment, Grid, load_config
from repro.models import cost_usd
from repro.schedules import build_schedule, package_available, uses_package

Log = Callable[[str], None]


@dataclass
class Totals:
    """What a run did, summed over cells."""

    reranked: int = 0
    skipped: int = 0
    failed: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    skipped_cells: list[str] = field(default_factory=list)

    def add(self, other: Totals) -> None:
        for name in ("reranked", "skipped", "failed", "calls", "prompt_tokens", "completion_tokens", "usd"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.skipped_cells += other.skipped_cells

    def __str__(self) -> str:
        text = (
            f"{self.reranked} reranked, {self.skipped} already done, {self.failed} failed; "
            f"{self.calls} calls, {self.prompt_tokens + self.completion_tokens:,} tokens"
        )
        return text + (f", about USD {self.usd:.2f}" if self.usd else "")


def select(
    arms: Sequence[Arm], *, benchmarks: Sequence[str] | None = None, llms: Sequence[str] | None = None
) -> list[Arm]:
    """The arms whose benchmark and LLM pass the command-line filters."""
    return [a for a in arms if (not benchmarks or a.benchmark in benchmarks) and (not llms or a.llm in llms)]


def check_package(arms: Sequence[Arm], *, dry_run: bool, log: Log) -> list[Arm]:
    """Drop package cells in a dry run without the package; refuse a real run without it."""
    needs = [a for a in arms if uses_package(a.spec)]
    if not needs or package_available():
        return list(arms)
    if not dry_run:
        raise SystemExit(
            f"{len(needs)} cells run the released BlitzRank package, which is not installed: "
            "pip install blitzrank==0.1.0"
        )
    log(f"skipping {len(needs)} cells of the BlitzRank package, which is not installed")
    return [a for a in arms if not uses_package(a.spec)]


async def run_cell(
    experiment: str,
    arm: Arm,
    grid: Grid,
    rankers: Rankers,
    *,
    max_queries: int | None = None,
    log: Log = print,
) -> Totals:
    """Rerank every query of one cell that is not recorded yet."""
    cell = arm.cell(experiment)
    path = cell.path
    queries = load_queries(arm.retriever, arm.benchmark, arm.depth, max_queries)
    done = done_qids(path)
    todo = [q for q in queries if str(q["qid"]) not in done]
    totals = Totals(skipped=len(queries) - len(todo))
    if not todo:
        return totals

    schedule = build_schedule(arm.spec)
    llm = rankers.get(arm.llm, grid.llm_options)
    semaphore = asyncio.Semaphore(grid.query_concurrency)

    async def one(query: dict) -> None:
        qid = str(query["qid"])
        async with semaphore:
            t0 = time.monotonic()
            try:
                result = await schedule.arerank(query["query"], texts(query), llm=llm)
            except Exception as exc:  # recorded as a failure; the query is retried on the next run
                totals.failed += 1
                log(f"    FAIL {cell.benchmark} {qid}: {type(exc).__name__}: {exc}")
                return
            wall = time.monotonic() - t0
        append_record(path, make_record(query, result, wall))
        totals.reranked += 1
        totals.calls += result.num_calls
        totals.prompt_tokens += result.prompt_tokens
        totals.completion_tokens += result.completion_tokens

    t0 = time.monotonic()
    await asyncio.gather(*(one(q) for q in todo))
    if not rankers.dry_run:
        totals.usd = cost_usd(arm.llm, totals.prompt_tokens, totals.completion_tokens)
    log(f"  {_label(arm)}: {totals} in {time.monotonic() - t0:.0f} s")
    return totals


async def run_grids(
    experiment: Experiment,
    rankers: Rankers,
    *,
    max_queries: int | None = None,
    benchmarks: Sequence[str] | None = None,
    llms: Sequence[str] | None = None,
    log: Log = print,
) -> Totals:
    """Run every quality cell of ``experiment``."""
    totals = Totals()
    for grid in experiment.grids:
        arms = check_package(
            select(list(grid.arms()), benchmarks=benchmarks, llms=llms), dry_run=rankers.dry_run, log=log
        )
        for arm in arms:
            totals.add(await run_cell(experiment.name, arm, grid, rankers, max_queries=max_queries, log=log))
    return totals


def _label(arm: Arm) -> str:
    depth = f"-n{arm.depth}" if arm.depth else ""
    repeat = f".r{arm.repeat}" if arm.repeat else ""
    return f"{arm.llm} {arm.schedule} {arm.retriever}/{arm.benchmark}{depth}{repeat}"


def set_runs_dir(path: Path) -> None:
    """Write records under ``path`` instead of ``runs/``."""
    records.RUNS = Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m repro.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a config file, or its stem under configs/")
    parser.add_argument("--max-queries", type=int, default=None, help="only the first N queries of each pool")
    parser.add_argument("--dry-run", action="store_true", help="offline FakeLLM; records go to runs-dry-run/")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="only these benchmarks")
    parser.add_argument("--llms", nargs="+", default=None, help="only these models (keys of models.yaml)")
    parser.add_argument("--skip-timing", action="store_true", help="run the quality grids only")
    parser.add_argument("--base-url", default=None, help="URL of the local vLLM server, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--runs-dir", type=Path, default=None, help="write records here instead of runs/")
    return parser


async def amain(args: argparse.Namespace, log: Log = print) -> int:
    from repro.timing import run_timing

    experiment = load_config(args.config)
    if args.runs_dir is not None:
        set_runs_dir(args.runs_dir)
    elif args.dry_run:
        set_runs_dir(DRY_RUN_DIR)
    rankers = Rankers(dry_run=args.dry_run, base_url=args.base_url)
    log(f"{experiment.name}: {len(experiment.cells())} cells -> {records.RUNS / experiment.name}")
    totals = await run_grids(
        experiment, rankers, max_queries=args.max_queries, benchmarks=args.benchmarks, llms=args.llms, log=log
    )
    if experiment.timing and not args.skip_timing:
        totals.add(
            await run_timing(
                experiment,
                Rankers(dry_run=args.dry_run, base_url=args.base_url, fake_delay_s=0.001 if args.dry_run else 0.0),
                max_queries=args.max_queries,
                benchmarks=args.benchmarks,
                log=log,
            )
        )
    log(f"{experiment.name}: {totals}")
    return 1 if totals.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, log=lambda s: print(s, flush=True)))


if __name__ == "__main__":
    sys.exit(main())
