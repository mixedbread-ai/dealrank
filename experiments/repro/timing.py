"""Timing harness: seconds per query with methods interleaved, and throughput under load.

    python -m repro.timing configs/table1_local_timing.yaml
    python -m repro.timing configs/load.yaml --dry-run --max-queries 4

A config's ``timing`` blocks (:class:`repro.config.TimingBlock`) run in one of
two modes.

**One query in flight** (``queries_in_flight: 1``, the default). For each
repeat, benchmark and query, every method of the block reranks that query back
to back, in an order drawn per query from ``seed``, with nothing else in
flight. All methods are thus timed on the same query at the same moment, and
no method is always first or last. A schedule still issues the calls of one
round concurrently, as it would in deployment.

**Load** (``queries_in_flight: Q > 1``). For each repeat and benchmark the
methods run one after another, in an order drawn from ``seed``, each as a
block: Q workers take the benchmark's queries (in a seeded order) one at a
time and start the next as soon as their last one ends. Once every query has
started, the workers keep Q queries in flight with repeated queries, whose
results are discarded, until the last query ends, so the load stays at Q for
the whole block. One line per block in ``runs/<experiment>/blocks.jsonl``
gives its window and throughput (queries completed in the window per second).

Before the first timed query, ``warmup_queries`` queries go through every
method and their results are discarded. Every timed query is recorded with
:func:`~repro.records.run_record` (per-call ``started_at`` / ``ended_at``
stamps, critical path and wall time) plus a ``timing`` field with its repeat,
position in the order, the order itself, ``queries_in_flight`` and its own
start and end stamps. Records of repeat ``r`` go to the cell's ``.r<r>``
file. Recorded queries are skipped on a rerun; in load mode a block is all or
nothing, since a partly rerun block would not see the same load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dealrank import ListwiseLLM, Schedule
from repro import records
from repro.common import DRY_RUN_DIR, Rankers, append_record, done_qids, load_queries, make_record, texts
from repro.config import Experiment, TimingBlock, load_config
from repro.records import Cell
from repro.run import Log, Totals, check_package, select, set_runs_dir
from repro.schedules import build_schedule


def method_order(seed: int, repeat: int, benchmark: str, key: str, names: Sequence[str]) -> list[str]:
    """The seeded order in which the methods run on one query (or one load block).

    >>> method_order(42, 1, "dl19", "q1", ["a", "b", "c"]) == method_order(42, 1, "dl19", "q1", ["a", "b", "c"])
    True
    """
    order = list(names)
    random.Random(f"{seed}|{repeat}|{benchmark}|{key}").shuffle(order)
    return order


class _Context:
    """One (LLM, retriever) slice of a timing block, ready to run."""

    def __init__(
        self,
        experiment: str,
        block: TimingBlock,
        llm_name: str,
        llm: ListwiseLLM,
        retriever: str,
        schedules: dict[str, Schedule],
        benchmarks: list[str],
        queries: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.experiment = experiment
        self.block = block
        self.llm_name = llm_name
        self.llm = llm
        self.retriever = retriever
        self.schedules = schedules
        self.names = list(schedules)
        self.benchmarks = benchmarks
        self.queries = queries

    def cell(self, name: str, benchmark: str, repeat: int) -> Cell:
        return Cell(self.experiment, self.llm_name, name, self.retriever, benchmark, self.block.depth, repeat)

    async def rerank(self, name: str, query: dict[str, Any]) -> tuple[Any, float, float, float]:
        """(result, wall seconds, start stamp, end stamp) of one rerank."""
        started_at = time.time()
        t0 = time.monotonic()
        result = await self.schedules[name].arerank(query["query"], texts(query), llm=self.llm)
        return result, time.monotonic() - t0, started_at, time.time()


async def run_timing(
    experiment: Experiment,
    rankers: Rankers,
    *,
    max_queries: int | None = None,
    benchmarks: Sequence[str] | None = None,
    only_repeat: int | None = None,
    log: Log = print,
) -> Totals:
    """Run every timing block of ``experiment``."""
    totals = Totals()
    for block in experiment.timing:
        arms = check_package(select(list(block.arms()), benchmarks=benchmarks), dry_run=rankers.dry_run, log=log)
        for llm_name in block.llms:
            for retriever in block.retrievers:
                mine = [a for a in arms if a.llm == llm_name and a.retriever == retriever]
                if not mine:
                    continue
                specs = {a.schedule: a.spec for a in mine}
                bench = [b for b in block.benchmarks[retriever] if any(a.benchmark == b for a in mine)]
                ctx = _Context(
                    experiment.name,
                    block,
                    llm_name,
                    rankers.get(llm_name, block.llm_options),
                    retriever,
                    {name: build_schedule(spec) for name, spec in specs.items()},
                    bench,
                    {b: load_queries(retriever, b, block.depth, max_queries) for b in bench},
                )
                repeats = [r for r in block.repeats if only_repeat is None or r == only_repeat]
                if block.queries_in_flight == 1:
                    totals.add(await _interleaved(ctx, repeats, log))
                else:
                    totals.add(await _load(ctx, repeats, log))
    return totals


async def _warm_up(ctx: _Context, log: Log) -> None:
    """Send the first ``warmup_queries`` queries through every method and discard the results."""
    n = ctx.block.warmup_queries
    if n <= 0 or not ctx.benchmarks:
        return
    queries = ctx.queries[ctx.benchmarks[0]][:n]
    t0 = time.monotonic()
    qif = ctx.block.queries_in_flight
    for name in method_order(ctx.block.seed, 0, ctx.benchmarks[0], "warmup", ctx.names):
        semaphore = asyncio.Semaphore(qif)

        async def one(query: dict[str, Any], name: str = name, semaphore: asyncio.Semaphore = semaphore) -> None:
            async with semaphore:
                try:
                    await ctx.rerank(name, query)
                except Exception as exc:  # a failed warm-up call is not a result; report and go on
                    log(f"    warm-up {name} {query['qid']}: {type(exc).__name__}: {exc}")

        await asyncio.gather(*(one(q) for q in queries))
    log(f"  warm-up: {len(queries)} queries x {len(ctx.names)} methods in {time.monotonic() - t0:.0f} s")


async def _interleaved(ctx: _Context, repeats: list[int], log: Log) -> Totals:
    totals = Totals()
    seed = ctx.block.seed
    done = {
        (r, name, b): done_qids(ctx.cell(name, b, r).path)
        for r in repeats
        for name in ctx.names
        for b in ctx.benchmarks
    }
    pending = sum(len(ctx.queries[b]) - len(done[(r, n, b)]) for (r, n, b) in done)
    if pending == 0:
        totals.skipped = sum(len(v) for v in done.values())
        return totals
    await _warm_up(ctx, log)
    for r in repeats:
        for b in ctx.benchmarks:
            t0 = time.monotonic()
            for query in ctx.queries[b]:
                qid = str(query["qid"])
                order = method_order(seed, r, b, qid, ctx.names)
                for slot, name in enumerate(order):
                    if qid in done[(r, name, b)]:
                        totals.skipped += 1
                        continue
                    try:
                        result, wall, started_at, ended_at = await ctx.rerank(name, query)
                    except Exception as exc:  # recorded as a failure; retried on the next run
                        totals.failed += 1
                        log(f"    FAIL r{r} {b} {qid} {name}: {type(exc).__name__}: {exc}")
                        continue
                    timing = {
                        "repeat": r,
                        "slot": slot,
                        "order": order,
                        "seed": seed,
                        "queries_in_flight": 1,
                        "started_at": started_at,
                        "ended_at": ended_at,
                    }
                    append_record(ctx.cell(name, b, r).path, make_record(query, result, wall, timing=timing))
                    done[(r, name, b)].add(qid)
                    _count(totals, result)
            log(f"  r{r} {ctx.llm_name} {b}: {len(ctx.queries[b])} queries in {time.monotonic() - t0:.0f} s")
    return totals


async def _load(ctx: _Context, repeats: list[int], log: Log) -> Totals:
    totals = Totals()
    qif = ctx.block.queries_in_flight
    plan = []
    for r in repeats:
        for b in ctx.benchmarks:
            order = method_order(ctx.block.seed, r, b, f"load{qif}", ctx.names)
            for slot, name in enumerate(order):
                qids = {str(q["qid"]) for q in ctx.queries[b]}
                done = done_qids(ctx.cell(name, b, r).path)
                if qids <= done:
                    totals.skipped += len(qids)
                    continue
                if done:
                    raise RuntimeError(
                        f"{ctx.cell(name, b, r).path} holds {len(done)} of {len(qids)} queries of a load block; "
                        "delete it to rerun the block"
                    )
                plan.append((r, b, slot, order, name))
    if not plan:
        return totals
    await _warm_up(ctx, log)
    for r, b, slot, order, name in plan:
        await _load_block(ctx, r, b, slot, order, name, totals, log)
    return totals


async def _load_block(
    ctx: _Context, r: int, b: str, slot: int, order: list[str], name: str, totals: Totals, log: Log
) -> None:
    """Every query of ``b`` through method ``name`` with ``queries_in_flight`` workers and filler."""
    qif = ctx.block.queries_in_flight
    seed = ctx.block.seed
    primary = list(ctx.queries[b])
    if not primary:
        return
    random.Random(f"{seed}|{r}|{b}|{name}|{qif}|primary").shuffle(primary)
    filler = list(ctx.queries[b])
    random.Random(f"{seed}|{r}|{b}|{name}|{qif}|filler").shuffle(filler)
    block_id = f"r{r}-{b}-q{qif}-{slot}-{name}"
    path = ctx.cell(name, b, r).path
    state = {"next": 0, "left": len(primary)}
    finished = asyncio.Event()
    completions: list[dict[str, Any]] = []

    def next_job() -> tuple[int, dict[str, Any], bool]:
        i = state["next"]
        state["next"] += 1
        if i < len(primary):
            return i, primary[i], False
        return i, filler[(i - len(primary)) % len(filler)], True

    async def worker() -> None:
        while not finished.is_set():
            index, query, is_filler = next_job()
            try:
                result, wall, started_at, ended_at = await ctx.rerank(name, query)
                error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # recorded as a failure of the block's query
                result, error = None, f"{type(exc).__name__}: {exc}"
                started_at = ended_at = time.time()
                wall = 0.0
            completions.append(
                {"filler": is_filler, "ok": error is None, "started_at": started_at, "ended_at": ended_at}
            )
            if is_filler:
                continue
            if error is None:
                timing = {
                    "repeat": r,
                    "slot": slot,
                    "order": order,
                    "seed": seed,
                    "queries_in_flight": qif,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "block": block_id,
                    "start_index": index,
                }
                append_record(path, make_record(query, result, wall, timing=timing))
                _count(totals, result)
            else:
                totals.failed += 1
                log(f"    FAIL r{r} {b} {query['qid']} {name}: {error}")
            state["left"] -= 1
            if state["left"] == 0:
                finished.set()

    tasks = [asyncio.create_task(worker()) for _ in range(qif)]
    await finished.wait()
    last_end = time.time()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    in_window = [c for c in completions if c["ended_at"] <= last_end + 1e-6]
    first_start = min(c["started_at"] for c in completions)
    window = last_end - first_start
    ok = sum(c["ok"] for c in in_window)
    summary = {
        "llm": ctx.llm_name,
        "schedule": name,
        "retriever": ctx.retriever,
        "benchmark": b,
        "depth": ctx.block.depth,
        "repeat": r,
        "slot": slot,
        "order": order,
        "block": block_id,
        "queries_in_flight": qif,
        "queries": len(primary),
        "queries_ok": sum(c["ok"] and not c["filler"] for c in in_window),
        "filler_completed": sum(c["ok"] and c["filler"] for c in in_window),
        "filler_cancelled": state["next"] - len(completions),
        "started_at": first_start,
        "ended_at": last_end,
        "window_s": window,
        "throughput_qps": ok / window if window > 0 else None,
    }
    blocks = records.RUNS / ctx.experiment / "blocks.jsonl"
    blocks.parent.mkdir(parents=True, exist_ok=True)
    with open(blocks, "a") as handle:
        handle.write(json.dumps(summary) + "\n")
    qps = summary["throughput_qps"] or 0.0
    log(f"  r{r} {b} {name}, {qif} in flight: {len(primary)} queries in {window:.1f} s, {qps:.3f} queries/s")


def _count(totals: Totals, result: Any) -> None:
    totals.reranked += 1
    totals.calls += result.num_calls
    totals.prompt_tokens += result.prompt_tokens
    totals.completion_tokens += result.completion_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m repro.timing", description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a config file with a `timing` section, or its stem under configs/")
    parser.add_argument("--max-queries", type=int, default=None, help="only the first N queries of each pool")
    parser.add_argument("--dry-run", action="store_true", help="offline FakeLLM; records go to runs-dry-run/")
    parser.add_argument("--fake-delay", type=float, default=0.001, help="seconds per FakeLLM call in a dry run")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="only these benchmarks")
    parser.add_argument("--repeat", type=int, default=None, help="only this repeat")
    parser.add_argument("--base-url", default=None, help="URL of the local vLLM server, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--runs-dir", type=Path, default=None, help="write records here instead of runs/")
    return parser


async def amain(args: argparse.Namespace, log: Log = print) -> int:
    experiment = load_config(args.config)
    if not experiment.timing:
        raise SystemExit(f"{experiment.name} has no `timing` section")
    if args.runs_dir is not None:
        set_runs_dir(args.runs_dir)
    elif args.dry_run:
        set_runs_dir(DRY_RUN_DIR)
    rankers = Rankers(dry_run=args.dry_run, base_url=args.base_url, fake_delay_s=args.fake_delay if args.dry_run else 0)
    totals = await run_timing(
        experiment, rankers, max_queries=args.max_queries, benchmarks=args.benchmarks, only_repeat=args.repeat, log=log
    )
    log(f"{experiment.name}: {totals}")
    return 1 if totals.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, log=lambda s: print(s, flush=True)))


if __name__ == "__main__":
    sys.exit(main())
