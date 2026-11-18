"""Experiment configs: one YAML file per table or figure, expanded into record cells.

A config names the LLMs, the first-stage pools and the schedules to run::

    llms: [gemini-3-flash]          # keys of configs/models.yaml
    retriever: bm25                 # or `retrievers: [bm25, dense]`
    benchmarks: all                 # or a list; `all` depends on the retriever
    query_concurrency: 12           # queries in flight (quality runs)
    llm_options: {max_concurrency: 32}
    schedules:                      # every schedule of the config
      - {type: dealrank, quota: 4}

These top-level keys, plus ``depth`` (candidates per query) and ``repeats``,
are defaults for two optional sections:

* ``grids``: quality runs, each a mapping that overrides any default and
  lists its own subset of ``schedules`` (usually as YAML aliases);
* ``timing``: one block or a list of blocks for :mod:`repro.timing`, which
  also take ``repeats`` (required), ``seed``, ``warmup_queries`` and
  ``queries_in_flight``.

A config with neither section is one quality grid. A config with a
``timing`` section runs quality grids only if it lists them.

``repeats`` is a count (repeats 1 to n) or a list of repeat numbers. A
quality grid without ``repeats`` writes one record file per cell, with no
repeat suffix. ``all`` resolves to :data:`~repro.records.BENCHMARKS` for
``bm25``, :data:`~repro.records.DENSE_BENCHMARKS` for ``dense`` and
:data:`~repro.records.DEEP_BENCHMARKS` for ``bm25-top1000``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from repro.records import BENCHMARKS, DEEP_BENCHMARKS, DENSE_BENCHMARKS, Cell
from repro.schedules import build_schedule

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

RETRIEVERS: dict[str, tuple[str, ...]] = {
    "bm25": BENCHMARKS,
    "dense": DENSE_BENCHMARKS,
    "bm25-top1000": DEEP_BENCHMARKS,
}

_COMMON_KEYS = {"llms", "retriever", "retrievers", "benchmarks", "depth", "repeats", "llm_options", "schedules"}
_GRID_KEYS = _COMMON_KEYS | {"query_concurrency"}
_TIMING_KEYS = _COMMON_KEYS | {"seed", "warmup_queries", "queries_in_flight"}
_TOP_KEYS = _GRID_KEYS | {"grids", "timing"}


@dataclass(frozen=True)
class Arm:
    """One schedule on one pool, as a grid or timing block runs it."""

    llm: str
    spec: Mapping[str, Any]
    schedule: str
    retriever: str
    benchmark: str
    depth: int | None
    repeat: int | None

    def cell(self, experiment: str) -> Cell:
        return Cell(experiment, self.llm, self.schedule, self.retriever, self.benchmark, self.depth, self.repeat)


@dataclass
class Grid:
    """A set of quality runs: every LLM x schedule x pool x repeat."""

    llms: list[str]
    retrievers: list[str]
    benchmarks: dict[str, list[str]]
    schedules: list[dict[str, Any]]
    depth: int | None = None
    repeats: list[int | None] = field(default_factory=lambda: [None])
    query_concurrency: int = 8
    llm_options: dict[str, Any] = field(default_factory=dict)

    def arms(self) -> Iterator[Arm]:
        for llm in self.llms:
            for spec in self.schedules:
                name = build_schedule(spec).name
                for retriever in self.retrievers:
                    for benchmark in self.benchmarks[retriever]:
                        for repeat in self.repeats:
                            yield Arm(llm, spec, name, retriever, benchmark, self.depth, repeat)


@dataclass
class TimingBlock(Grid):
    """Timed runs: queries one at a time with methods interleaved, or under load."""

    seed: int = 42
    warmup_queries: int = 0
    queries_in_flight: int = 1


@dataclass
class Experiment:
    """A parsed config. ``name`` is the file stem, which names ``runs/<name>/``."""

    name: str
    path: Path
    schedules: list[dict[str, Any]]
    grids: list[Grid]
    timing: list[TimingBlock]

    def arms(self) -> Iterator[Arm]:
        """Every quality and timing arm, quality first."""
        for grid in [*self.grids, *self.timing]:
            yield from grid.arms()

    def cells(self) -> list[Cell]:
        return [arm.cell(self.name) for arm in self.arms()]


def resolve_config_path(path: str | Path) -> Path:
    """A config path as given, or a stem under ``configs/``."""
    path = Path(path)
    if path.exists():
        return path
    candidate = CONFIGS / (path.name if path.suffix else f"{path}.yaml")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no config {path}")


def load_config(path: str | Path) -> Experiment:
    """Parse and check a config, building every schedule once to validate it."""
    path = resolve_config_path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    _check_keys(raw, _TOP_KEYS, path.name)
    schedules = [dict(spec) for spec in raw.get("schedules") or []]
    if not schedules:
        raise ValueError(f"{path.name}: `schedules` is empty")
    names = [build_schedule(spec).name for spec in schedules]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: schedules with the same name: {duplicates}")

    defaults = {k: v for k, v in raw.items() if k not in {"grids", "timing"}}
    grid_specs = raw.get("grids")
    timing_specs = raw.get("timing")
    if isinstance(timing_specs, Mapping):
        timing_specs = [timing_specs]
    if grid_specs is None:
        grid_specs = [] if timing_specs else [{}]

    grids = [_grid(Grid, defaults, spec, schedules, path.name, _GRID_KEYS) for spec in grid_specs]
    timing = [_grid(TimingBlock, defaults, spec, schedules, path.name, _TIMING_KEYS) for spec in timing_specs or []]
    for block in timing:
        if block.repeats == [None]:
            raise ValueError(f"{path.name}: timing blocks need `repeats`")
    experiment = Experiment(path.stem, path, schedules, grids, timing)
    _check_unique_cells(experiment)
    return experiment


def _grid(
    cls: type[Grid],
    defaults: Mapping[str, Any],
    spec: Mapping[str, Any],
    schedules: list[dict[str, Any]],
    where: str,
    allowed: set[str],
) -> Grid:
    _check_keys(spec, allowed, where)
    merged = {k: v for k, v in defaults.items() if k in allowed}
    if "retriever" in spec or "retrievers" in spec:
        merged.pop("retriever", None)
        merged.pop("retrievers", None)
    merged.update(spec)

    llms = list(merged.get("llms") or [])
    if not llms:
        raise ValueError(f"{where}: no `llms`")
    retrievers = merged.get("retrievers") or [merged.get("retriever", "bm25")]
    if isinstance(retrievers, str):
        retrievers = [retrievers]
    benchmarks = {r: _benchmarks(r, merged.get("benchmarks", "all"), where) for r in retrievers}

    chosen = [dict(s) for s in merged.get("schedules") or schedules]
    for spec_ in chosen:
        if spec_ not in schedules:
            raise ValueError(f"{where}: grid schedule {spec_} is not in the top-level `schedules`")

    kwargs: dict[str, Any] = {
        "llms": llms,
        "retrievers": list(retrievers),
        "benchmarks": benchmarks,
        "schedules": chosen,
        "depth": merged.get("depth"),
        "repeats": _repeats(merged.get("repeats")),
        "llm_options": dict(merged.get("llm_options") or {}),
    }
    if cls is Grid:
        kwargs["query_concurrency"] = int(merged.get("query_concurrency", 8))
    else:
        kwargs["seed"] = int(merged.get("seed", 42))
        kwargs["warmup_queries"] = int(merged.get("warmup_queries", 0))
        kwargs["queries_in_flight"] = int(merged.get("queries_in_flight", 1))
        if kwargs["queries_in_flight"] < 1:
            raise ValueError(f"{where}: queries_in_flight must be positive")
    return cls(**kwargs)


def _benchmarks(retriever: str, value: str | Sequence[str], where: str) -> list[str]:
    if retriever not in RETRIEVERS:
        raise ValueError(f"{where}: unknown retriever {retriever!r}; expected one of {sorted(RETRIEVERS)}")
    available = RETRIEVERS[retriever]
    if value == "all":
        return list(available)
    chosen = [value] if isinstance(value, str) else list(value)
    unknown = [b for b in chosen if b not in available]
    if unknown:
        raise ValueError(f"{where}: no {retriever} pool for {unknown}")
    return chosen


def _repeats(value: int | Sequence[int] | None) -> list[int | None]:
    if value is None:
        return [None]
    if isinstance(value, int):
        if value < 1:
            raise ValueError("repeats must be positive")
        return list(range(1, value + 1))
    repeats = [int(r) for r in value]
    if any(r < 1 for r in repeats) or len(set(repeats)) != len(repeats):
        raise ValueError(f"repeats must be distinct positive numbers, not {list(value)}")
    return list(repeats)


def _check_keys(spec: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")


def _check_unique_cells(experiment: Experiment) -> None:
    seen: set[Cell] = set()
    for cell in experiment.cells():
        if cell in seen:
            raise ValueError(f"{experiment.path.name}: two runs write {cell.path}")
        seen.add(cell)
