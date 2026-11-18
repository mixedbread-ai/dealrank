"""Tables 1, 2 and 3 of the paper, from the run records.

    python -m repro.tables [--table {1,2,3}] [--out DIR]

writes ``main_gemini.tex`` (Table 1), ``models.tex`` (Table 2), ``depth.tex``
(Table 3) and ``tables.json`` (every printed number, unrounded) to
``out/tables/``. ``--table N`` computes and writes one table only, so Table 1
can be reproduced from the Gemini 3 Flash runs and the local timing alone;
``tables.json`` then holds that table only.

Conventions:

* nDCG@10 x 100 per benchmark is the mean over its queries; a row averaged
  over several runs (three repeats, or the ten seeds of the random partition)
  first averages each query over the runs. ``Avg.`` is the unweighted mean over
  benchmarks.
* ``Delta`` is the row minus DealRank, paired by query, with the hierarchical
  bootstrap of :mod:`repro.stats`.
* Calls, rounds and tokens use the queries whose pool holds a full 100
  candidates (1,497 of the 1,622): means over the records of the row's runs,
  per benchmark, then averaged over benchmarks; ``Max`` is the most rounds of
  any such record. The "concurrent" BlitzRank rows count rounds with the
  first, mutually disjoint calls issued at once
  (:func:`repro.runs.first_calls_concurrent`); the row above them is the same
  runs with every call in turn.
* p50 and p90 are the median and 90th percentile of the client wall-clock per
  DL19 query with the local model, pooled over three repeats. Contiguous
  chunks and the random partition make DealRank's calls on another split, so
  they show DealRank's timings.
* Bold marks the best printed value of each column: quality over every
  reranker (BM25 excluded), costs over the rows that make more than one call.
* Table 2 pairs each comparator with the LLM's DealRank run. Its average row averages each query's
  difference over the six LLMs, on the queries all of them answered. The LLMs
  of :data:`repro.paper_rows.EXTRA_MODELS` (GPT-5.5) are printed below the
  six with the comparators they ran (:data:`repro.paper_rows.EXTRA_CELLS`)
  and are left out of the average.
  ``tables.json`` also holds BlitzRank on dealt input
  (:data:`repro.paper_rows.UNPRINTED`), which the table does not print and
  Figure 1(b) uses.
* Table 3 is computed by :mod:`repro.analysis.depth`.

After writing the ``.tex`` files the script reads them back and checks that
every printed number equals the rounded value in ``tables.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from repro import paper_rows as P
from repro import records, runs
from repro.analysis import depth as depth_analysis
from repro.stats import hier_boot, summary

BENCHMARKS = records.BENCHMARKS
COSTS = ("calls", "rounds", "rounds_max", "tokens_k")


def num(v: float, nd: int = 1, sign: bool = False) -> str:
    """A number as printed: fixed decimals, LaTeX minus."""
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def paired_points(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> dict[str, Any]:
    """``a - b`` in nDCG@10 points over the queries both have, per benchmark."""
    diffs = {ds: np.array([a[ds][q] - b[ds][q] for q in sorted(set(a[ds]) & set(b[ds]))]) for ds in a if ds in b}
    return summary(hier_boot({ds: d for ds, d in diffs.items() if d.size}), 100.0)


# ---- Table 1 ---------------------------------------------------------------------
def table1_row(row: P.Table1Row) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Quality and costs of one Table 1 row over BM25, and its per-query scores."""
    per_ds, n_q, n_full, scores = {}, {}, {}, {}
    cost: dict[str, list[float]] = {c: [] for c in COSTS}
    for ds in BENCHMARKS:
        per_run = [runs.run_scores(run, "bm25", ds) for run in row.bm25]
        if len({frozenset(s) for s in per_run}) > 1:
            raise ValueError(f"{row.key}/{ds}: the runs of this row scored different queries")
        scores[ds] = runs.arm_scores(row.bm25, "bm25", ds)
        per_ds[ds] = 100 * float(np.mean(list(scores[ds].values())))
        n_q[ds] = len(scores[ds])
        loaded = runs.load_arm(row.bm25, "bm25", ds)
        recs = [r for x in loaded for r in runs.full_pool(x, P.FULL_POOL)]
        if not recs:
            raise ValueError(f"{row.key}/{ds}: no query with {P.FULL_POOL} candidates")
        n_full[ds] = len(recs) // len(loaded)
        rounds = [runs.rounds_of(r, row.concurrent) for r in recs]
        cost["calls"].append(float(np.mean([r["calls"] for r in recs])))
        cost["tokens_k"].append(float(np.mean([runs.tokens_k(r) for r in recs])))
        cost["rounds"].append(float(np.mean(rounds)))
        cost["rounds_max"].append(float(max(rounds)))
    out = {
        "per_benchmark": per_ds,
        "queries": n_q,
        "queries_full_pool": n_full,
        "avg": float(np.mean(list(per_ds.values()))),
        **{c: float(max(v)) if c == "rounds_max" else float(np.mean(v)) for c, v in cost.items()},
    }
    return out, scores


def local_seconds(rows: dict[str, dict[str, Any]]) -> None:
    """Add ``p50`` / ``p90`` local seconds to the rows that have them, following ``timing_from``."""
    for row in P.TABLE1:
        if row.timing is not None:
            recs = runs.timing_records(row.timing, "bm25", ("dl19",))
            rows[row.key]["seconds"] = runs.percentiles([r["wall_s"] for r in recs])
    for row in P.TABLE1:
        if row.timing_from is not None:
            rows[row.key]["seconds"] = dict(rows[row.timing_from]["seconds"], taken_from=row.timing_from)


def table1() -> dict[str, dict[str, Any]]:
    runs.require(
        [(row.bm25, "bm25", BENCHMARKS) for row in P.TABLE1]
        + [(row.timing, "bm25", ("dl19",)) for row in P.TABLE1 if row.timing is not None]
    )
    rows, scores = {}, {}
    for row in P.TABLE1:
        rows[row.key], scores[row.key] = table1_row(row)
    for key in rows:
        if key != "dealrank":
            rows[key]["minus_dealrank"] = paired_points(scores[key], scores["dealrank"])
    local_seconds(rows)
    return rows


def render_table1(rows: dict[str, dict[str, Any]]) -> str:
    rerankers = [k for k in rows if k != "bm25"]
    best = {ds: round(max(rows[k]["per_benchmark"][ds] for k in rerankers), 1) for ds in BENCHMARKS}
    best_avg = round(max(rows[k]["avg"] for k in rerankers), 1)
    costed = [k for k in rows if k not in P.NO_COST_BOLD]
    best_cost = {c: min(round(rows[k][c], 1) for k in costed) for c in COSTS}
    best_sec = {s: round(min(rows[k]["seconds"][s] for k in costed), 2) for s in ("p50", "p90")}

    def bold(text: str, value: float, target: float, nd: int = 1) -> str:
        return r"\textbf{" + text + "}" if round(value, nd) == target else text

    def q(v: float, b: float) -> str:
        return bold(num(v), v, b)

    dash = r"\hfil--\hfil"
    n_full = sum(rows["dealrank"]["queries_full_pool"].values())
    n_cost = len(COSTS) + 2
    first_cost = len(BENCHMARKS) + 4
    lines = [
        "% Generated by python -m repro.tables; do not edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\caption{Gemini 3 Flash, BM25 top 100, "
        + NUM_WORD[len(BENCHMARKS)]
        + r" benchmarks (nDCG@10$\times$100; DealRank, sliding window, BlitzRank $w{=}20$: three runs; random"
        r" partition: " + NUM_WORD[len(P.RANDOM_SEEDS)] + r" seeds). $\Delta$: vs.\ DealRank, paired 95\% CI."
        r" Shaded: dealt or random split. Chunks and random partition make DealRank's calls. BlitzRank: authors'"
        r" package. Calls, rounds and tokens: the "
        + f"{n_full:,}".replace(",", "{,}")
        + " queries with "
        + str(P.FULL_POOL)
        + r" candidates. Rounds: mean, Max. p50, p90: seconds per DL19 query with Qwen3.8-27B, methods interleaved."
        r" $^\dagger$First calls run concurrently. Bold: best reranker, and for costs best multi-call method.}",
        r"\label{tab:main-gemini}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l>{\columncolor{avgcol}}r>{\columncolor{avgcol}}l@{\hspace{8pt}}"
        + "r" * len(BENCHMARKS)
        + r"@{\hspace{8pt}}"
        + "r" * n_cost
        + "@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\cellcolor{avgcol}} & \multicolumn{"
        + str(len(BENCHMARKS))
        + r"}{c}{nDCG@10 $\times$ 100 per benchmark} & \multicolumn{"
        + str(n_cost)
        + r"}{c}{Cost per query} \\",
        r"\cmidrule(lr){4-" + str(first_cost - 1) + r"}\cmidrule(l){" + f"{first_cost}-{first_cost + n_cost - 1}" + "}",
        r"Method & \textbf{Avg.} & $\Delta$ vs.\ DealRank & "
        + " & ".join(records.LABELS[d] for d in BENCHMARKS)
        + r" & Calls & Rounds & Max & Tok.\ (k) & p50 & p90 \\",
        r"\midrule",
    ]
    for row in P.TABLE1:
        r = rows[row.key]
        if row.rule_before:
            lines.append(r"\midrule")
        sec = r.get("seconds")
        if row.key == "bm25":
            cost = ["0", "0", "0", "0", dash, dash]
        elif row.key in P.NO_COST_BOLD:
            cost = [num(r[c], 0 if c == "rounds_max" else 1) for c in COSTS]
            cost += [num(sec["p50"], 2), num(sec["p90"], 2)]
        else:
            cost = [
                bold(num(r[c], 0 if c == "rounds_max" else 1), r[c], best_cost[c], 0 if c == "rounds_max" else 1)
                for c in COSTS
            ]
            cost += [bold(num(sec[s], 2), sec[s], best_sec[s], 2) for s in ("p50", "p90")]
        if row.key == "bm25":
            quality = [num(r["per_benchmark"][d]) for d in BENCHMARKS]
            avg = num(r["avg"])
        else:
            quality = [q(r["per_benchmark"][d], best[d]) for d in BENCHMARKS]
            avg = q(r["avg"], best_avg)
        if row.key == "dealrank":
            delta = dash
        else:
            md = r["minus_dealrank"]
            delta = (
                num(md["diff"], 1, sign=True)
                + r" {\scriptsize["
                + num(md["ci"][0], 1, sign=True)
                + ", "
                + num(md["ci"][1], 1, sign=True)
                + "]}"
            )
        label = r"\textbf{" + row.label + "}" if row.key == "dealrank" else row.label
        pre = r"\rowcolor{dealrow}" if row.shaded else ""
        lines.append(pre + label + " & " + " & ".join([avg, delta, *quality, *cost]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


# ---- Table 2 ---------------------------------------------------------------------
TWO_DECIMAL_CI = ("blitzrank_w10", "blitzrank_w20")
AVERAGE_LABEL = r"Average (excl.\ " + ", ".join(label for _, label in P.EXTRA_MODELS) + ")"
NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def table2() -> dict[str, Any]:
    llms = [m for m, _ in P.MODELS]
    columns = [c for c, _ in P.COMPARATORS + P.UNPRINTED]
    extra = [(m, c) for m, _ in P.EXTRA_MODELS for c in P.EXTRA_CELLS[m]]
    runs.require(
        [(P.table2_arm(m, "dealrank"), "bm25", BENCHMARKS) for m in [*llms, *(m for m, _ in P.EXTRA_MODELS)]]
        + [(P.table2_arm(m, c), "bm25", BENCHMARKS) for m in llms for c in columns]
        + [(P.table2_arm(m, c), "bm25", BENCHMARKS) for m, c in extra]
    )
    ref = {m: runs.scores_by_benchmark(P.table2_arm(m, "dealrank"), "bm25", BENCHMARKS) for m in llms}
    out: dict[str, Any] = {"models": {}, "average": {"comparators": {}}}
    diffs: dict[str, dict[str, dict[str, dict[str, float]]]] = {c: {} for c in columns}
    for m in llms:
        cells = {}
        for c in columns:
            comp = runs.scores_by_benchmark(P.table2_arm(m, c), "bm25", BENCHMARKS)
            base = ref[m]
            d = {ds: {q: comp[ds][q] - base[ds][q] for q in sorted(set(comp[ds]) & set(base[ds]))} for ds in BENCHMARKS}
            diffs[c][m] = d
            cells[c] = summary(hier_boot({ds: np.array(list(v.values())) for ds, v in d.items()}), 100.0)
        out["models"][m] = {"dealrank": 100 * runs.macro(ref[m]), "comparators": cells}
    shared = {ds: sorted(set.intersection(*(set(ref[m][ds]) for m in llms))) for ds in BENCHMARKS}
    out["average"]["dealrank"] = 100 * float(
        np.mean([np.mean([np.mean([ref[m][ds][q] for m in llms]) for q in shared[ds]]) for ds in BENCHMARKS])
    )
    for c, by_model in diffs.items():
        ms = sorted(by_model)
        avg = {}
        for ds in BENCHMARKS:
            common = sorted(set.intersection(*(set(by_model[m][ds]) for m in ms)))
            avg[ds] = np.array([np.mean([by_model[m][ds][q] for m in ms]) for q in common])
        out["average"]["comparators"][c] = {**summary(hier_boot(avg), 100.0), "models": ms}
    # The LLMs printed below the six: their own cells only, outside the average.
    for m, _ in P.EXTRA_MODELS:
        base = runs.scores_by_benchmark(P.table2_arm(m, "dealrank"), "bm25", BENCHMARKS)
        cells = {}
        for c in P.EXTRA_CELLS[m]:
            comp = runs.scores_by_benchmark(P.table2_arm(m, c), "bm25", BENCHMARKS)
            d = {ds: [comp[ds][q] - base[ds][q] for q in sorted(set(comp[ds]) & set(base[ds]))] for ds in BENCHMARKS}
            cells[c] = summary(hier_boot({ds: np.array(v) for ds, v in d.items()}), 100.0)
        out["models"][m] = {"dealrank": 100 * runs.macro(base), "comparators": cells}
    return out


def render_table2(tab: dict[str, Any]) -> str:
    keys = [c for c, _ in P.COMPARATORS]
    n = len(P.MODELS)
    extra_names = " and ".join(label for _, label in P.EXTRA_MODELS)

    def diff(c: dict[str, Any]) -> str:
        return num(c["diff"], 2, True)

    def ci(c: dict[str, Any], nd: int) -> str:
        lo, hi = c["ci"]
        return r"{\footnotesize[" + num(lo, nd, True) + "," + num(hi, nd, True) + "]}"

    lines = [
        "% Generated by python -m repro.tables; do not edit.",
        r"\begin{table}[!b]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{"
        + NUM_WORD[n].capitalize()
        + " LLMs and "
        + extra_names
        + ", BM25 top 100, "
        + str(len(BENCHMARKS))
        + r" benchmarks. DealRank nDCG@10$\times$100 and each comparator minus it (95\% CI for the average over LLMs,"
        r" resampling benchmarks and queries; ---: not run). BlitzRank: ours (authors' for Gemini). $^\dagger$Run to"
        r" validate DealRank on a frontier model, against BlitzRank at $w{=}10$, and left out of the average.}",
        r"\label{tab:models}",
        r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{@{}lr" + "c" * len(keys) + "@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{BlitzRank} & Sliding & & JointRank \\",
        r"Model & DealRank & $w{=}20$ & $w{=}10$ & window & TDPart & $r{=}4$ \\",
        r"\midrule",
    ]
    for m, label in P.MODELS:
        row = tab["models"][m]
        lines.append(
            label
            + " & "
            + num(row["dealrank"])
            + " & "
            + " & ".join(diff(row["comparators"][k]) for k in keys)
            + r" \\"
        )
    for m, label in P.EXTRA_MODELS:
        row = tab["models"][m]
        cells = [diff(row["comparators"][k]) if k in row["comparators"] else "---" for k in keys]
        lines.append(label + r"$^\dagger$ & " + num(row["dealrank"]) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    avg = tab["average"]
    lines.append(
        AVERAGE_LABEL
        + " & "
        + num(avg["dealrank"])
        + " & "
        + " & ".join(diff(avg["comparators"][k]) for k in keys)
        + r" \\"
    )
    lines.append(
        " & & " + " & ".join(ci(avg["comparators"][k], 2 if k in TWO_DECIMAL_CI else 1) for k in keys) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


# ---- Table 3 ---------------------------------------------------------------------
def table3() -> dict[str, Any]:
    return depth_analysis.summary_all()


def rounds_subscript(x: float) -> str:
    """Mean rounds as Table 3 prints them: whole when they are (or above ten), else one decimal."""
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 or x >= 10 else f"{x:.1f}"


def render_table3(tab: dict[str, Any]) -> str:
    lines = [
        "% Generated by python -m repro.tables; do not edit.",
        r"\begin{table}[t]",
        r"\caption{Deeper lists, Qwen3.8-27B. nDCG@10 on DL19, DL20, DBpedia and SciFact, median seconds per DL19"
        r" query and mean rounds. Blitz\,$\parallel$: first calls concurrent.}",
        r"\label{tab:depth}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3.1pt}",
        r"\begin{tabular}{@{}r rrr rrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{nDCG@10} & \multicolumn{4}{c}{Seconds per query (rounds)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
        r"Top & \textbf{Deal} & Blitz & Window & \textbf{Deal} & Blitz\,$\parallel$ & Blitz & Window \\",
        r"\midrule",
    ]
    for d in depth_analysis.DEPTHS:
        v = tab[str(d)]
        cells = [f"{v['quality'][c]['ndcg10']:.1f}" for c in depth_analysis.QUALITY]
        for c in depth_analysis.COLUMNS:
            t = v["timing"][c]
            sec = f"{t['seconds']:.1f}"
            cells.append(
                (r"\textbf{" + sec + "}" if c == "dealrank" else sec) + "$_{" + rounds_subscript(t["rounds"]) + "}$"
            )
        lines.append(f"{d:,}".replace(",", "{,}") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---- checking the printed numbers -------------------------------------------------------
NUMBER = r"(?:\$-\$|\+)?\d+(?:\.\d+)?"


def numbers(text: str) -> list[float]:
    return [float(x.replace("$-$", "-")) for x in re.findall(NUMBER, text)]


def check_table1(tex: str, rows: dict[str, dict[str, Any]]) -> int:
    """Check every number printed in Table 1 against ``rows``; returns how many were checked."""
    lines = tex.splitlines()
    checked = 0
    for row in P.TABLE1:
        shown = (r"\rowcolor{dealrow}" if row.shaded else "") + (
            r"\textbf{" + row.label + "}" if row.key == "dealrank" else row.label
        )
        line = next(x for x in lines if x.startswith(shown + " & "))
        got = numbers(line[len(shown) :])
        r = rows[row.key]
        want = [r["avg"]]
        if row.key != "dealrank":
            want += [r["minus_dealrank"]["diff"], *r["minus_dealrank"]["ci"]]
        want += [r["per_benchmark"][d] for d in BENCHMARKS]
        want += [0, 0, 0, 0] if row.key == "bm25" else [r[c] for c in COSTS]
        nd = [0 if c == "rounds_max" else 1 for c in COSTS]
        digits = [1] * (len(want) - 4) + ([0] * 4 if row.key == "bm25" else nd)
        if row.key != "bm25":
            want += [r["seconds"]["p50"], r["seconds"]["p90"]]
            digits += [2, 2]
        if len(got) != len(want):
            raise AssertionError(f"Table 1 row {row.key}: {len(got)} numbers printed, {len(want)} expected")
        for g, w, d in zip(got, want, digits):
            if abs(g - round(w, d)) > 1e-9:
                raise AssertionError(f"Table 1 row {row.key}: printed {g}, expected {round(w, d)}")
            checked += 1
    return checked


def check_table2(tex: str, tab: dict[str, Any]) -> int:
    """Check every number printed in Table 2 against ``tab``; returns how many were checked."""
    lines = tex.splitlines()
    pattern = r"(?:\$-\$|\+)?\d+\.\d+"
    checked = 0
    keys = [c for c, _ in P.COMPARATORS]
    shown = [*P.MODELS, *((m, label + r"$^\dagger$") for m, label in P.EXTRA_MODELS), ("average", AVERAGE_LABEL)]
    for m, label in shown:
        i = next(j for j, x in enumerate(lines) if x.startswith(label + " & "))
        got = [float(x.replace("$-$", "-")) for x in re.findall(pattern, lines[i][len(label) :])]
        row = tab["average"] if m == "average" else tab["models"][m]
        want = [round(row["dealrank"], 1)] + [
            round(row["comparators"][k]["diff"], 2) for k in keys if k in row["comparators"]
        ]
        if m == "average":
            cis = [float(x.replace("$-$", "-")) for x in re.findall(pattern, lines[i + 1])]
            want_ci = [round(v, 2 if k in TWO_DECIMAL_CI else 1) for k in keys for v in row["comparators"][k]["ci"]]
            if cis != want_ci:
                raise AssertionError(f"Table 2 average intervals: printed {cis}, expected {want_ci}")
            checked += len(cis)
        if got != want:
            raise AssertionError(f"Table 2 row {m}: printed {got}, expected {want}")
        checked += len(got)
    return checked


def check_table3(tex: str, tab: dict[str, Any]) -> int:
    """Check every number printed in Table 3 against ``tab``; returns how many were checked."""
    lines = tex.splitlines()
    checked = 0
    for d in depth_analysis.DEPTHS:
        label = f"{d:,}".replace(",", "{,}")
        line = next(x for x in lines if x.startswith(label + " & "))
        got = re.findall(r"\d+(?:\.\d+)?", line[len(label) :])
        v = tab[str(d)]
        want = [f"{v['quality'][c]['ndcg10']:.1f}" for c in depth_analysis.QUALITY]
        for c in depth_analysis.COLUMNS:
            want += [f"{v['timing'][c]['seconds']:.1f}", rounds_subscript(v["timing"][c]["rounds"])]
        if got != want:
            raise AssertionError(f"Table 3 row {d}: printed {got}, expected {want}")
        checked += len(got)
    return checked


def write(out: Path, which: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """Compute the tables in ``which``, write their ``.tex`` and JSON files to ``out`` and check them."""
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    checked = 0
    if 1 in which:
        rows = table1()
        (out / "main_gemini.tex").write_text(render_table1(rows))
        checked += check_table1((out / "main_gemini.tex").read_text(), rows)
        payload["table1"] = rows
    if 2 in which:
        tab = table2()
        (out / "models.tex").write_text(render_table2(tab))
        checked += check_table2((out / "models.tex").read_text(), tab)
        payload["table2"] = tab
    if 3 in which:
        deep = table3()
        (out / "depth.tex").write_text(render_table3(deep))
        checked += check_table3((out / "depth.tex").read_text(), deep)
        payload["table3"] = deep
    payload["printed_numbers_checked"] = checked
    (out / "tables.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write Tables 1, 2 and 3 from the run records.")
    parser.add_argument("--table", type=int, choices=(1, 2, 3), help="write one table only (default: all three)")
    parser.add_argument("--out", type=Path, default=records.OUT / "tables", help="output directory")
    args = parser.parse_args(argv)
    payload = write(args.out, (args.table,) if args.table else (1, 2, 3))
    if "table1" in payload:
        print_table1(payload["table1"], args.out)
    if "table2" in payload:
        print_table2(payload["table2"], args.out)
    if "table3" in payload:
        print(f"\nTable 3 ({args.out / 'depth.tex'})")
        print(depth_analysis.render(payload["table3"]))
    print(f"\n{payload['printed_numbers_checked']} printed numbers checked against tables.json.")


def print_table1(t1: dict[str, dict[str, Any]], out: Path) -> None:
    print(f"Table 1 ({out / 'main_gemini.tex'})")
    print("| row | Avg. | Delta vs DealRank | calls | rounds | max | tokens (k) | p50 | p90 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in P.TABLE1:
        r = t1[row.key]
        md = r.get("minus_dealrank")
        delta = f"{md['diff']:+.2f} [{md['ci'][0]:+.2f}, {md['ci'][1]:+.2f}]" if md else "--"
        sec = r.get("seconds") or {}
        print(
            f"| {row.key} | {r['avg']:.2f} | {delta} | {r['calls']:.2f} | {r['rounds']:.2f} | {r['rounds_max']:.0f} | "
            f"{r['tokens_k']:.1f} | {sec.get('p50', float('nan')):.2f} | {sec.get('p90', float('nan')):.2f} |"
        )
    full = sum(t1["dealrank"]["queries_full_pool"].values())
    print(f"\n{sum(t1['dealrank']['queries'].values())} queries, {full} with {P.FULL_POOL} candidates.")


def print_table2(tab: dict[str, Any], out: Path) -> None:
    avg = tab["average"]
    print(f"\nTable 2 ({out / 'models.tex'}): DealRank average {avg['dealrank']:.1f}")
    for c, v in avg["comparators"].items():
        print(
            f"- {c} minus DealRank over {len(v['models'])} LLMs: "
            f"{v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
        )


if __name__ == "__main__":
    main()
