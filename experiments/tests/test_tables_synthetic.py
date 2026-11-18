"""Tables 1, 2 and 3 on a tiny synthetic runs tree: every printed number matches tables.json."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")

from dealrank.partition import split
from repro import paper_rows as P
from repro import records, runs, tables
from repro.analysis import depth

QUERIES = 3
N = 100


def _clear_caches() -> None:
    for f in (runs.load_run, runs.run_scores, runs.qrels, runs.pool_docids, runs.pool_size):
        f.cache_clear()


def _call(rng: random.Random, n: int = 20) -> dict[str, Any]:
    perm = list(range(n))
    rng.shuffle(perm)
    start = 1_700_000_000.0 + rng.random()
    latency = 0.5 + rng.random()
    return {
        "permutation": perm,
        "repairs": [],
        "prompt_tokens": 2000 + rng.randrange(500),
        "completion_tokens": 90,
        "latency_s": latency,
        "started_at": start,
        "ended_at": start + latency,
        "cached": False,
        "meta": {},
    }


def _record(spec: dict[str, Any], qid: str, docs: list[str], rng: random.Random) -> dict[str, Any]:
    kind = spec["type"]
    diagnostics: dict[str, Any] = {}
    if kind == "retriever":
        ranking, widths = list(docs), []
    elif kind == "blitzrank_package":
        w = spec["window"]
        cover = N // w
        calls = [list(range(i * w, (i + 1) * w)) for i in range(cover)]
        calls.append([c[j] for c in calls for j in range(min(4, w))][:w])
        if rng.random() < 0.6:
            calls.append(sorted(rng.sample(range(N), w)))
        diagnostics = {
            "calls": calls,
            "stop": "certified",
            "certified": True,
            "input_order": {"order": spec.get("input_order", "retriever"), "permutation": list(range(N))},
        }
        ranking = rng.sample(docs, len(docs))
        widths = [cover] + [1] * (len(calls) - cover) if spec.get("concurrent_cover") else [1] * len(calls)
    elif kind == "dealrank":
        hands = split(spec.get("split", "deal"), N, N // spec["window"], seed=spec.get("seed", 0))
        fin = [h[j] for h in hands for j in range(spec["quota"])][:20]
        rest = [i for i in range(N) if i not in set(fin)]
        ranking = [docs[i] for i in rng.sample(fin, len(fin)) + rest]
        diagnostics = {"split": spec.get("split", "deal"), "quota": spec["quota"], "hands": hands, "finalists": fin}
        widths = [len(hands), 1]
    else:
        ranking = rng.sample(docs, len(docs))
        widths = [rng.randint(1, 5) for _ in range(rng.randint(1, 9))]
    trace = [[_call(rng) for _ in range(w)] for w in widths]
    path = sum(max(c["latency_s"] for c in r) for r in trace)
    return {
        "qid": qid,
        "ranking": ranking,
        "n_candidates": N,
        "rounds": len(widths),
        "round_widths": widths,
        "calls": sum(widths),
        "prompt_tokens": sum(c["prompt_tokens"] for r in trace for c in r),
        "completion_tokens": sum(c["completion_tokens"] for r in trace for c in r),
        "critical_path_s": path,
        "wall_s": path + 0.05 + rng.random() * 0.1,
        "repair_rate": 0.0,
        "trace": trace,
        "diagnostics": diagnostics,
    }


Needed = tuple[P.Run, str, tuple[str, ...], int | None]


def _needed() -> Iterator[Needed]:
    for row in P.TABLE1:
        yield from ((run, "bm25", records.BENCHMARKS, None) for run in row.bm25)
        if row.timing:
            yield from ((run, "bm25", ("dl19",), None) for run in row.timing)
    for m, _ in P.MODELS:
        for column in ("dealrank", *(c for c, _ in P.COMPARATORS + P.UNPRINTED)):
            yield from ((run, "bm25", records.BENCHMARKS, None) for run in P.table2_arm(m, column))
    for m, _ in P.EXTRA_MODELS:
        for column in ("dealrank", *P.EXTRA_CELLS[m]):
            yield from ((run, "bm25", records.BENCHMARKS, None) for run in P.table2_arm(m, column))
    for fig_row in P.FIG1_ROWS:
        yield from ((run, "bm25", P.FIG1_BENCHMARKS, None) for run in fig_row.timing)
        yield from ((run, "bm25", records.BENCHMARKS, None) for run in fig_row.certify or ())
    for d in depth.DEPTHS:
        for column in depth.QUALITY:
            arm, retriever, arg = depth.quality_arm(d, column)
            yield from ((run, retriever, depth.BENCHMARKS, arg) for run in arm)
        for column in depth.COLUMNS:
            s = depth.specs(d)[column]
            arm = P.timed(s) if d == 100 else (P.Run(P.T3, s, P.LOCAL, 1), P.Run(P.T3, s, P.LOCAL, 2))
            yield from ((run, "bm25" if d == 100 else P.DEEP, (depth.TIMED,), None if d == 100 else d) for run in arm)


@pytest.fixture
def synthetic_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(records, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(records, "DATA", tmp_path / "data")
    _clear_caches()
    rng = random.Random(0)
    pools: dict[tuple[str, str], dict[str, list[str]]] = {}
    for retriever in ("bm25", P.DEEP):
        for b in runs.benchmarks_of(retriever):
            qs = {f"{b}-q{i}": [f"{b}-q{i}-d{j}" for j in range(N)] for i in range(QUERIES)}
            pools[(retriever, b)] = qs
            path = records.pool_path(retriever, b)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "qid": q,
                            "query": q,
                            "candidates": [{"docid": d, "score": float(N - j), "text": ""} for j, d in enumerate(docs)],
                        }
                    )
                    + "\n"
                    for q, docs in qs.items()
                )
            )
    for b in records.BENCHMARKS:
        path = records.qrels_path(b)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{b}-q{i} 0 {b}-q{i}-d{j} {rng.choice([0, 1, 2])}" for i in range(QUERIES) for j in range(0, N, 7)]
        path.write_text("\n".join(lines) + "\n")
    done: set[tuple[P.Run, str, str, int | None]] = set()
    for run, retriever, benchmarks, d in _needed():
        for b in benchmarks:
            if (run, retriever, b, d) in done:
                continue
            done.add((run, retriever, b, d))
            path = runs.cell_path(run, retriever, b, d)
            path.parent.mkdir(parents=True, exist_ok=True)
            spec = dict(run.spec)
            path.write_text(
                "".join(json.dumps(_record(spec, q, docs, rng)) + "\n" for q, docs in pools[(retriever, b)].items())
            )
    yield tmp_path
    _clear_caches()


def test_tables_print_their_json(synthetic_tree: Path) -> None:
    out = synthetic_tree / "out"
    payload = tables.write(out)
    t1 = (out / "main_gemini.tex").read_text()
    t2 = (out / "models.tex").read_text()
    t3 = (out / "depth.tex").read_text()
    assert payload["printed_numbers_checked"] == (
        tables.check_table1(t1, payload["table1"])
        + tables.check_table2(t2, payload["table2"])
        + tables.check_table3(t3, payload["table3"])
    )
    assert payload["printed_numbers_checked"] > 300
    assert json.loads((out / "tables.json").read_text())["table1"]["dealrank"]["avg"] == pytest.approx(
        payload["table1"]["dealrank"]["avg"]
    )


def test_tables_have_the_shape_of_the_paper(synthetic_tree: Path) -> None:
    out = synthetic_tree / "out"
    tables.write(out)
    t1 = (out / "main_gemini.tex").read_text()
    assert t1.count(r" \\") == len(P.TABLE1) + 2  # two header lines
    assert "ten benchmarks" in t1 and "random partition: ten seeds" in t1
    assert f"the {3 * len(records.BENCHMARKS)} queries with 100 candidates" in t1  # three per synthetic pool
    assert "2-round, contiguous chunks & " in t1 and "Random partition & " in t1
    assert r"BlitzRank $w{=}20$ & " in t1 and r"BlitzRank $w{=}20$, concurrent$^\dagger$ & " in t1
    t2 = (out / "models.tex").read_text()
    assert "Six LLMs and GPT-5.5, BM25 top 100, 10 benchmarks" in t2
    assert sum(line.startswith(label + " & ") for line in t2.splitlines() for _, label in P.MODELS) == 6
    extra = next(line for line in t2.splitlines() if line.startswith(r"GPT-5.5$^\dagger$ & "))
    assert extra.count("---") == len(P.COMPARATORS) - 1  # GPT-5.5 ran one comparator only
    assert any(line.startswith(tables.AVERAGE_LABEL + " & ") for line in t2.splitlines())
    t3 = (out / "depth.tex").read_text()
    assert "Seconds per query (rounds)" in t3
    assert [line.split(" & ")[0] for line in t3.splitlines() if line[:1].isdigit()] == ["100", "500", "1{,}000"]


def test_one_table_only(synthetic_tree: Path) -> None:
    payload = tables.write(synthetic_tree / "out1", (1,))
    assert "table1" in payload and "table2" not in payload and "table3" not in payload
    assert not (synthetic_tree / "out1" / "models.tex").exists()


def test_a_wrong_number_is_caught(synthetic_tree: Path) -> None:
    out = synthetic_tree / "out"
    payload = tables.write(out)
    rows = json.loads(json.dumps(payload["table1"]))
    rows["dealrank"]["avg"] += 0.5
    with pytest.raises(AssertionError):
        tables.check_table1((out / "main_gemini.tex").read_text(), rows)
    deep = json.loads(json.dumps(payload["table3"]))
    deep["1000"]["timing"]["dealrank"]["seconds"] += 1.0
    with pytest.raises(AssertionError):
        tables.check_table3((out / "depth.tex").read_text(), deep)


def test_row_conventions(synthetic_tree: Path) -> None:
    rows = tables.table1()
    assert rows["contiguous"]["seconds"]["p50"] == rows["dealrank"]["seconds"]["p50"]
    assert rows["random_partition"]["seconds"]["taken_from"] == "dealrank"
    concurrent, sequential = rows["blitzrank_w20_concurrent"], rows["blitzrank_w20"]
    assert sequential["per_benchmark"] == concurrent["per_benchmark"] and sequential["calls"] == concurrent["calls"]
    assert sequential["seconds"] != concurrent["seconds"]  # timed separately
    assert concurrent["rounds"] == pytest.approx(sequential["rounds"] - 4)  # five first calls become one round
    assert rows["bm25"]["calls"] == 0
    assert "minus_dealrank" not in rows["dealrank"]
    assert len(P.TABLE1_BY_KEY["random_partition"].bm25) == 10


def test_costs_use_full_pools_only(synthetic_tree: Path) -> None:
    before = tables.table1_row(P.TABLE1_BY_KEY["tourrank_r1"])[0]
    path = runs.cell_path(P.Run(P.T1, P.TOURRANK), "bm25", "dl19")
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    recs[0].update(n_candidates=37, calls=1000, rounds=1000)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    _clear_caches()
    after = tables.table1_row(P.TABLE1_BY_KEY["tourrank_r1"])[0]
    assert after["queries_full_pool"]["dl19"] == before["queries_full_pool"]["dl19"] - 1
    assert after["rounds_max"] < 1000 and after["calls"] < 100
    assert after["per_benchmark"] == before["per_benchmark"]  # nDCG@10 still uses every query


def test_figure1_from_runs(synthetic_tree: Path) -> None:
    from repro.analysis.latency import figure1_data

    tab = tables.table2()
    data = figure1_data(tab)
    rows = {r["key"]: r for r in data["panel_b"]["rows"]}
    for r in rows.values():
        assert sum(s["s"] for s in r["segments"]) == pytest.approx(r["path_s"])
    assert 0 <= data["setup"]["certify_share"] <= 1
    assert data["panel_b"]["ndcg_note"] == "mean of 6 LLMs"
    chunks = tab["average"]["comparators"]["blitzrank_w20"]["diff"]
    assert rows["blitzrank_concurrent"]["ndcg10"] - rows["dealrank"]["ndcg10"] == pytest.approx(chunks)
    assert rows["blitzrank_concurrent"]["rounds"] == pytest.approx(
        tables.table1()["blitzrank_w20_concurrent"]["rounds"]
    )
    pytest.importorskip("matplotlib")
    from repro.figures import fig1_hero, style

    style.apply()
    fig1_hero.draw(data)
