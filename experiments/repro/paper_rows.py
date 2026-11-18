"""Which runs make up each row of the paper's tables and figures.

Every table and figure script reads its runs from here, so this module is the
one place that ties a printed row to experiment configs and schedule specs.
A schedule's directory name is never written out: it is built from its spec
with :func:`schedule_name`, exactly as the runner names it.

An :class:`Arm` is a tuple of :class:`Run` s. When it holds several runs
(repeats of one schedule, or seeds of a random partition), per-query scores
are averaged over them before any other aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

GEMINI = "gemini-3-flash"
LOCAL = "qwen3.8-27b"

Spec = tuple[tuple[str, Any], ...]


def spec(type: str, **kwargs: Any) -> Spec:
    """A hashable schedule spec, as written in the experiment configs."""
    return tuple(sorted({"type": type, **kwargs}.items()))


def extend(s: Spec, **kwargs: Any) -> Spec:
    """``s`` with more keyword arguments."""
    return spec(**{**dict(s), **kwargs})


@cache
def schedule_name(s: Spec) -> str:
    """The directory name the runner gives the schedule built from ``s``."""
    from repro.schedules import build_schedule

    return build_schedule(dict(s)).name


@dataclass(frozen=True)
class Run:
    """One run of one schedule with one LLM: the record files of one experiment and repeat."""

    experiment: str
    spec: Spec
    llm: str = GEMINI
    repeat: int | None = None

    @property
    def schedule(self) -> str:
        return schedule_name(self.spec)

    def at(self, **changes: Any) -> Run:
        """A copy with some fields changed."""
        return Run(**{**self.__dict__, **changes})


Arm = tuple[Run, ...]


def runs(experiment: str, s: Spec, llm: str = GEMINI, repeats: tuple[int | None, ...] = (None,)) -> Arm:
    return tuple(Run(experiment, s, llm, r) for r in repeats)


# ---- schedule specs ----------------------------------------------------------
DEALRANK = spec("dealrank", window=20, quota=4)
CHUNKS = spec("dealrank", window=20, quota=4, split="chunks")


def random_partition(seed: int, *, window: int = 20, quota: int = 4, per_query: bool = True) -> Spec:
    extra = {"final_size": 20} if window != 20 else {}
    if per_query:
        extra["seed_per_query"] = True
    return spec("dealrank", window=window, quota=quota, split="random", seed=seed, **extra)


RETRIEVER = spec("retriever")
TOP20 = spec("top_prefix", size=20)
WINDOW = spec("sliding_window", window=20, stride=10)
TOURRANK = spec("tourrank", tournaments=1)
TDPART = spec("tdpart", window=20, cutoff=10, budget=20)
JOINTRANK = spec("jointrank", window=20, replicates=4)
LISTK = spec("listk", top_k=10, window=20, select_pivots=16, physical_pivots=2, sort_pivots=6)


def package(window: int, *, dealt: bool = False, concurrent: bool = False) -> Spec:
    """The authors' released BlitzRank package."""
    extra: dict[str, Any] = {}
    if dealt:
        extra["input_order"] = "dealt"
    if concurrent:
        extra["concurrent_cover"] = True
    return spec("blitzrank_package", window=window, top_m=10, **extra)


def blitzrank(window: int, *, dealt: bool = False, concurrent: bool = False) -> Spec:
    """This library's BlitzRank."""
    extra = {"concurrent_cover": True} if concurrent else {}
    return spec("blitzrank", window=window, top_m=10, input_order="dealt" if dealt else "retriever", **extra)


# ---- Table 1 ------------------------------------------------------------------
T1 = "table1_gemini"
T1_REPEATS = "table1_gemini_repeats"
T1_TIMING = "table1_local_timing"
#: Gemini 3 Flash over the dense retriever's top 100 (Figure 3 and the dense-retriever ablation).
DENSE = "dense_gemini"
#: Seeds of the random partition; each draws a fresh partition per query.
RANDOM_SEEDS = tuple(range(10))


def three_runs(s: Spec) -> Arm:
    """Run 1 from table1_gemini, runs 2 and 3 from table1_gemini_repeats."""
    return (Run(T1, s), Run(T1_REPEATS, s, repeat=2), Run(T1_REPEATS, s, repeat=3))


def ten_seeds() -> Arm:
    """The random partition with seeds 0 to 9: seed 0 from table1_gemini, the others from its repeats."""
    return tuple(Run(T1 if seed == 0 else T1_REPEATS, random_partition(seed)) for seed in RANDOM_SEEDS)


def timed(s: Spec) -> Arm:
    """The three timing repeats of one schedule on DL19 with the local model."""
    return runs(T1_TIMING, s, LOCAL, (1, 2, 3))


@dataclass(frozen=True)
class Table1Row:
    """One row of Table 1.

    ``concurrent`` rows reuse the records of a sequential BlitzRank package
    run and issue its leading run of mutually disjoint calls as one round.
    ``timing_from`` names the row whose measured seconds this row shows: a
    row that makes the same calls as DealRank on another split takes
    DealRank's timings.
    """

    key: str
    label: str
    bm25: Arm
    timing: Arm | None = None
    timing_from: str | None = None
    concurrent: bool = False
    shaded: bool = False
    rule_before: bool = False


TABLE1: tuple[Table1Row, ...] = (
    Table1Row("bm25", "BM25", (Run(T1, RETRIEVER),)),
    Table1Row("head20", "Top-20 only (one call)", (Run(T1, TOP20),), timing=timed(TOP20)),
    Table1Row("sliding_window", "Sliding window", three_runs(WINDOW), timed(WINDOW)),
    Table1Row("tourrank_r1", r"TourRank $r{=}1$", (Run(T1, TOURRANK),), timing=timed(TOURRANK)),
    Table1Row("tdpart", "TDPart", (Run(T1, TDPART),), timing=timed(TDPART)),
    Table1Row("jointrank", r"JointRank $r{=}4$ (one round)", (Run(T1, JOINTRANK),), timing=timed(JOINTRANK)),
    Table1Row("blitzrank_w10", r"BlitzRank $w{=}10$", (Run(T1, package(10)),), timing=timed(blitzrank(10))),
    Table1Row("blitzrank_w20", r"BlitzRank $w{=}20$", three_runs(package(20)), timed(package(20))),
    Table1Row(
        "blitzrank_w20_concurrent",
        r"BlitzRank $w{=}20$, concurrent$^\dagger$",
        three_runs(package(20)),
        timed(package(20, concurrent=True)),
        concurrent=True,
    ),
    Table1Row(
        "contiguous",
        "2-round, contiguous chunks",
        (Run(T1, CHUNKS),),
        timing_from="dealrank",
        rule_before=True,
    ),
    Table1Row(
        "blitzrank_w20_dealt_concurrent",
        r"BlitzRank $w{=}20$, dealt, concurrent$^\dagger$",
        three_runs(package(20, dealt=True)),
        timed(blitzrank(20, dealt=True, concurrent=True)),
        concurrent=True,
        shaded=True,
        rule_before=True,
    ),
    Table1Row("random_partition", "Random partition", ten_seeds(), timing_from="dealrank", shaded=True),
    Table1Row("dealrank", "DealRank", three_runs(DEALRANK), timed(DEALRANK), shaded=True),
)
TABLE1_BY_KEY = {row.key: row for row in TABLE1}

#: Rows left out of the bolding of cost columns: no call, or a single call.
NO_COST_BOLD = ("bm25", "head20")

#: Table 1's cost columns use the queries whose pool holds this many candidates.
FULL_POOL = 100


# ---- Table 2 ------------------------------------------------------------------
T2 = "table2_models"

#: The six LLMs of Table 2's average, in table order: (key, label).
MODELS: tuple[tuple[str, str], ...] = (
    ("deepseek-v4-flash", "DeepSeek V4-Flash"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
    ("qwen3.8-27b", "Qwen3.8-27B"),
    ("kimi-k2.6", "Kimi K2.6"),
    ("glm-5.2", "GLM-5.2"),
    ("gemini-3-flash", "Gemini 3 Flash"),
)

#: LLMs Table 2 prints below the six and leaves out of its average, with the comparators each ran:
#: GPT-5.5 is a frontier-model check against BlitzRank at ``w = 10`` only.
EXTRA_MODELS: tuple[tuple[str, str], ...] = (("gpt-5.5", "GPT-5.5"),)
EXTRA_CELLS: dict[str, tuple[str, ...]] = {"gpt-5.5": ("blitzrank_w10",)}

#: Table 2's comparator columns: (key, spec of this library's schedule).
COMPARATORS: tuple[tuple[str, Spec], ...] = (
    ("blitzrank_w20", blitzrank(20)),
    ("blitzrank_w10", blitzrank(10)),
    ("sliding_window", WINDOW),
    ("tdpart", TDPART),
    ("jointrank_r4", JOINTRANK),
)

#: Columns Table 2's runs also give but the table does not print: Figure 1(b) uses them.
UNPRINTED: tuple[tuple[str, Spec], ...] = (("blitzrank_dealt", blitzrank(20, dealt=True)),)

#: Gemini 3 Flash's row is Table 1's: the authors' package for BlitzRank, three runs where Table 1 has them.
GEMINI_TABLE2: dict[str, Arm] = {
    "dealrank": three_runs(DEALRANK),
    "blitzrank_w20": three_runs(package(20)),
    "blitzrank_w10": (Run(T1, package(10)),),
    "sliding_window": three_runs(WINDOW),
    "tdpart": (Run(T1, TDPART),),
    "jointrank_r4": (Run(T1, JOINTRANK),),
    "blitzrank_dealt": (Run(T1, package(20, dealt=True)),),
}


def table2_arm(llm: str, column: str) -> Arm:
    """The runs of one Table 2 cell (``column`` is ``"dealrank"`` or a comparator key)."""
    if llm == GEMINI:
        return GEMINI_TABLE2[column]
    s = DEALRANK if column == "dealrank" else dict(COMPARATORS + UNPRINTED)[column]
    return (Run(T2, s, llm),)


#: TourRank ran with these LLMs: Gemini 3 Flash in Table 1's grid, the other two in Table 2's.
TOURRANK_LLMS = (GEMINI, "deepseek-v4-flash", "qwen3.5-35b-a3b")


def tourrank_arm(llm: str) -> Arm:
    """TourRank's run with one of :data:`TOURRANK_LLMS`."""
    return (Run(T1, TOURRANK),) if llm == GEMINI else (Run(T2, TOURRANK, llm),)


# ---- Table 3 ------------------------------------------------------------------
T3 = "table3_depth"
DEEP = "bm25-top1000"
DEPTHS = (500, 1000)
#: Hierarchical DealRank: 25 hands keep 3 then 4 hands keep 5 (500), 50 hands keep 2 then 5 hands keep 4 (1,000).
HIERARCHICAL: dict[int, Spec] = {
    500: spec("dealrank_hierarchical", window=20, first_quota=3),
    1000: spec("dealrank_hierarchical", window=20, first_quota=2),
}


# ---- Figure 1(b) --------------------------------------------------------------
FIG1 = "fig1_api_timing"
FIG1_BENCHMARKS = ("dl19", "dl20", "covid", "touche")


@dataclass(frozen=True)
class Fig1Row:
    """One bar of Figure 1(b).

    ``timing`` are its timing runs; ``column`` is the Table 2 column its
    nDCG@10 comes from (``"dealrank"`` or a comparator); ``table1`` names the
    Table 1 row that gives its mean and most rounds on 100-candidate queries;
    ``certify`` are the runs its certification rate is measured on.
    """

    key: str
    label: str
    role: str
    split: str
    timing: Arm
    column: str
    quiet: bool = False
    certify: Arm | None = field(default=None)
    table1: str | None = None


FIG1_ROWS: tuple[Fig1Row, ...] = (
    Fig1Row("dealrank", "DealRank", "ours", "dealt", runs(FIG1, DEALRANK, repeats=(1, 2, 3)), "dealrank"),
    Fig1Row(
        "blitzrank_concurrent",
        "BlitzRank on chunks",
        "blitz",
        "chunks",
        runs(FIG1, package(20, concurrent=True), repeats=(1, 2, 3)),
        "blitzrank_w20",
        certify=three_runs(package(20)),
        table1="blitzrank_w20_concurrent",
    ),
    Fig1Row(
        "blitzrank_dealt",
        "BlitzRank on dealt input",
        "blitz",
        "dealt",
        runs(FIG1, package(20, dealt=True, concurrent=True), repeats=(1, 2, 3)),
        "blitzrank_dealt",
        certify=three_runs(package(20, dealt=True)),
        table1="blitzrank_w20_dealt_concurrent",
    ),
    Fig1Row(
        "sliding_window",
        "Sliding window",
        "other",
        "window",
        runs(FIG1, WINDOW, repeats=(1, 2, 3)),
        "sliding_window",
        quiet=True,
    ),
)


# ---- Figure 3 -----------------------------------------------------------------
FIG3 = "fig3_split_ablation"


@dataclass(frozen=True)
class Fig3Row:
    """One row of Figure 3: a dealt arm and the arm it is compared with, per retriever.

    ``group`` is ``"binds"`` when a group's quota can cut a top-ten passage,
    else ``"free"``.
    """

    key: str
    name: str
    part: str
    group: str
    dealt: Mapping[str, Arm]
    other: Mapping[str, Arm]
    reference: bool = False


def _fig3(dealt_bm25: Arm, other_bm25: Arm, dealt_dense: Arm, other_dense: Arm) -> tuple[dict[str, Arm], ...]:
    return {"bm25": dealt_bm25, "dense": dealt_dense}, {"bm25": other_bm25, "dense": other_dense}


def _one(experiment: str, s: Spec) -> Arm:
    return (Run(experiment, s),)


FIG3_ROWS: tuple[Fig3Row, ...] = (
    Fig3Row(
        "dealrank",
        "DealRank",
        "round 1",
        "binds",
        *_fig3(three_runs(DEALRANK), _one(T1, CHUNKS), _one(DENSE, DEALRANK), _one(DENSE, CHUNKS)),
        reference=True,
    ),
    Fig3Row(
        "tourrank_all",
        "TourRank",
        "every stage",
        "binds",
        *_fig3(
            _one(T1, TOURRANK),
            _one(FIG3, spec("tourrank", tournaments=1, split="chunks")),
            _one(FIG3, TOURRANK),
            _one(FIG3, spec("tourrank", tournaments=1, split="chunks")),
        ),
    ),
    Fig3Row(
        "blitz20",
        r"BlitzRank $\mathit{w}$=20",
        "input",
        "binds",
        *_fig3(
            three_runs(package(20, dealt=True)),
            three_runs(package(20)),
            _one(DENSE, package(20, dealt=True)),
            _one(DENSE, package(20)),
        ),
    ),
    Fig3Row(
        "blitz10",
        r"BlitzRank $\mathit{w}$=10",
        "input",
        "free",
        *_fig3(
            _one(FIG3, package(10, dealt=True)),
            _one(T1, package(10)),
            _one(FIG3, package(10, dealt=True)),
            _one(FIG3, package(10)),
        ),
    ),
    Fig3Row(
        "listk",
        "ListK",
        "bins",
        "free",
        *_fig3(
            _one(FIG3, extend(LISTK, bucket_split="deal")),
            _one(FIG3, LISTK),
            _one(FIG3, extend(LISTK, bucket_split="deal")),
            _one(FIG3, LISTK),
        ),
    ),
    Fig3Row(
        "tourrank",
        "TourRank",
        "first stage",
        "free",
        *_fig3(
            _one(T1, TOURRANK),
            _one(FIG3, spec("tourrank", tournaments=1, split="chunks_first")),
            _one(FIG3, TOURRANK),
            _one(FIG3, spec("tourrank", tournaments=1, split="chunks_first")),
        ),
    ),
    Fig3Row(
        "tdpart",
        "TDPart",
        "tail windows",
        "free",
        *_fig3(
            _one(FIG3, extend(TDPART, tail_split="deal")),
            _one(T1, TDPART),
            _one(FIG3, extend(TDPART, tail_split="deal")),
            _one(FIG3, TDPART),
        ),
    ),
    Fig3Row(
        "jointrank",
        "JointRank",
        "block design",
        "free",
        *_fig3(
            _one(FIG3, extend(JOINTRANK, design="dealt")),
            _one(T1, JOINTRANK),
            _one(FIG3, extend(JOINTRANK, design="dealt")),
            _one(FIG3, JOINTRANK),
        ),
    ),
)
