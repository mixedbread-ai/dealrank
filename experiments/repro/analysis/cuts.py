"""Measured cuts: what a split puts over quota, and what reaches the final call.

    python -m repro.analysis.cuts

All numbers use Gemini 3 Flash on the BM25 and dense top 100, on the queries
whose pool holds a full 100 candidates.

* Cuts forced by the split. Taking the sliding window's top ten as the
  reference, how many of them a split of five groups of twenty puts over its
  group's quota of four, ``sum_h (|T & P_h| - 4)_+``: contiguous chunks put
  2.0 over BM25 and 2.8 over the dense retriever, dealt hands about 0.1.
  This needs the sliding window's rankings and the split only.
* Over-full groups of a random partition (ten seeds, each drawing a fresh
  partition per query, against DealRank's three runs). Of the judged-relevant
  passages in the BM25 top 20, 12% land in a group holding five or more of
  them. Such a passage reaches the final call 8.0 [4.8, 11.0] percentage
  points less often than when dealt if its group holds five, and
  16.5 [11.0, 23.1] if it holds six or more. On all those queries, a random
  partition's groups hold 0.07 more of the sliding window's top ten beyond
  their quota than dealt hands do, and it brings 0.07 [-0.01, +0.15] fewer of
  them to the final call.
* Chunks cost more over the better retriever. On the queries of the dense
  pools, DealRank minus DealRank on contiguous chunks is 4.4 nDCG@10 points
  over the dense retriever and, in single runs, 2.3 over BM25, a difference
  of 2.1 [0.2, 4.1].
* Figure 2 (``--figure2``): the cuts forced by chunks and by dealt hands, per
  benchmark and pooled, against the mean retriever rank of the sliding
  window's top ten, next to Proposition 1 and the Mallows model of
  :mod:`repro.analysis.mallows`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dealrank.partition import chunks, deal
from repro import paper_rows as P
from repro import runs
from repro.analysis import expected_cuts, mallows
from repro.stats import bootstrap_mean, hier_boot, summary

QUOTA = 4
HANDS = 5
N = P.FULL_POOL


def _positions(docs: Sequence[str]) -> dict[str, int]:
    return {d: i for i, d in enumerate(docs)}


def finalists(record: Mapping[str, Any], docs: Sequence[str]) -> set[str]:
    """Docids that reached DealRank's final call (all of them when one call ranked the pool)."""
    fin = (record.get("diagnostics") or {}).get("finalists")
    if fin is None:
        return set(docs)
    return {docs[i] for i in fin}


def excess(targets: Sequence[int], owner: Sequence[int], quota: int = QUOTA) -> int:
    """``sum_h (|targets in group h| - quota)_+`` for targets given as positions."""
    count: dict[int, int] = defaultdict(int)
    for p in targets:
        count[owner[p]] += 1
    return sum(max(c - quota, 0) for c in count.values())


def owner_of(hands: Sequence[Sequence[int]], n: int) -> list[int]:
    owner = [0] * n
    for h, hand in enumerate(hands):
        for i in hand:
            owner[i] = h
    return owner


def _boot(by_benchmark: Mapping[str, Sequence[float]], scale: float = 1.0) -> dict[str, Any]:
    return summary(hier_boot({b: np.asarray(v) for b, v in by_benchmark.items() if len(v)}), scale)


# ---- cuts forced by the split ----------------------------------------------------------
def forced_cuts(retriever: str = "bm25") -> dict[str, Any]:
    """How many of the sliding window's top ten each split puts over its group's quota, per query."""
    experiment = P.T1 if retriever == "bm25" else P.DENSE
    owners = {"chunks": owner_of(chunks(N, HANDS), N), "dealt": owner_of(deal(N, HANDS), N)}
    values: dict[str, list[float]] = {split: [] for split in owners}
    for b in runs.benchmarks_of(retriever):
        pool = runs.pool_docids(retriever, b)
        window = runs.load_arm((P.Run(experiment, P.WINDOW),), retriever, b)[0]
        for q, rec in sorted(window.items()):
            if rec["n_candidates"] != N:
                continue
            pos = _positions(pool[q])
            top = [pos[d] for d in rec["ranking"][:10]]
            for split, owner in owners.items():
                values[split].append(float(excess(top, owner)))
    return {split: bootstrap_mean(v) for split, v in values.items()}


# ---- a random partition's over-full groups ----------------------------------------------
def load_row(load: int) -> str:
    return "<=4" if load <= QUOTA else ("5" if load == QUOTA + 1 else ">=6")


def random_overload() -> dict[str, Any]:
    """Judged-relevant passages of the BM25 top 20, by how many of them share their random group.

    Also, on every query with a full pool: how many more of the sliding
    window's top ten a random partition puts over quota than dealt hands do
    (from the partitions alone), and how many fewer of them it brings to the
    final call.
    """
    deal_arm, rand_arm = P.three_runs(P.DEALRANK), P.ten_seeds()
    window = (P.Run(P.T1, P.WINDOW),)
    dealt_owner = owner_of(deal(N, HANDS), N)
    reference_cut: dict[str, list[float]] = defaultdict(list)
    over_quota: dict[str, list[float]] = defaultdict(list)
    share = {"deal": defaultdict(int), "random": defaultdict(int)}
    level: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    gap: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for b in runs.benchmarks_of("bm25"):
        pool, grades = runs.pool_docids("bm25", b), runs.qrels(b)
        deal_runs = runs.load_arm(deal_arm, "bm25", b)
        rand_runs = runs.load_arm(rand_arm, "bm25", b)
        reference = runs.load_arm(window, "bm25", b)[0]
        full = [set(q for q, r in x.items() if r["n_candidates"] == N) for x in deal_runs + rand_runs]
        for q in sorted(set.intersection(*full)):
            docs = pool[q][:N]
            pos = _positions(docs)
            g = grades.get(q, {})
            relevant = [p for p in range(20) if g.get(docs[p], 0) >= 1]
            f_deal = [{pos[d] for d in finalists(x[q], docs)} for x in deal_runs]
            f_rand = [{pos[d] for d in finalists(x[q], docs)} for x in rand_runs]
            owners = [owner_of(x[q]["diagnostics"]["hands"], N) for x in rand_runs]
            if q in reference:
                top = [pos[d] for d in reference[q]["ranking"][:10]]
                cut_deal = np.mean([len(set(top) - f) for f in f_deal])
                reference_cut[b].append(float(np.mean([len(set(top) - f) for f in f_rand]) - cut_deal))
                over_quota[b].append(float(np.mean([excess(top, o) for o in owners]) - excess(top, dealt_owner)))
            if not relevant:
                continue
            dealt_load: dict[int, int] = defaultdict(int)
            for p in relevant:
                dealt_load[dealt_owner[p]] += 1
            survive_dealt = {p: np.mean([p in f for f in f_deal]) for p in relevant}
            for p in relevant:
                share["deal"][load_row(dealt_load[dealt_owner[p]])] += len(rand_runs)
            lv: dict[str, list[float]] = defaultdict(list)
            gp: dict[str, list[float]] = defaultdict(list)
            for f, owner in zip(f_rand, owners):
                load: dict[int, int] = defaultdict(int)
                for p in relevant:
                    load[owner[p]] += 1
                for p in relevant:
                    row = load_row(load[owner[p]])
                    share["random"][row] += 1
                    lv[row].append(100.0 * (p in f))
                    gp[row].append(100.0 * ((p in f) - survive_dealt[p]))
            for row in lv:
                level[row][b].append(float(np.mean(lv[row])))
                gap[row][b].append(float(np.mean(gp[row])))
    total = {a: sum(v.values()) for a, v in share.items()}
    rows = {}
    for row in ("<=4", "5", ">=6"):
        rows[row] = {
            "share_dealt": 100 * share["deal"][row] / total["deal"],
            "share_random": 100 * share["random"][row] / total["random"],
            "random_final": _boot(level[row]) if level[row] else None,
            "minus_dealt": _boot(gap[row]) if gap[row] else None,
        }
    return {
        "by_load": rows,
        "overloaded_share": rows["5"]["share_random"] + rows[">=6"]["share_random"],
        "seeds": len(rand_arm),
        "reference_top10_cut_random_minus_dealt": _boot(reference_cut),
        "reference_top10_over_quota_random_minus_dealt": _boot(over_quota),
    }


# ---- chunks over a better retriever ------------------------------------------------------
def chunk_penalty() -> dict[str, Any]:
    """DealRank minus chunks (nDCG@10 points) over the dense and the BM25 pools, and their difference."""
    dense_arms = [P.WINDOW, P.package(20), P.CHUNKS, P.DEALRANK]
    sc_d = {s: runs.scores_by_benchmark((P.Run(P.DENSE, s),), "dense") for s in dense_arms}
    sc_b = {
        s: runs.scores_by_benchmark((P.Run(P.T1, s),), "bm25", runs.benchmarks_of("dense"))
        for s in (P.CHUNKS, P.DEALRANK)
    }
    pen_d, pen_b = {}, {}
    for b in runs.benchmarks_of("dense"):
        common = sorted(set.intersection(*(set(sc_d[s][b]) for s in dense_arms)))
        pen_d[b] = {q: sc_d[P.DEALRANK][b][q] - sc_d[P.CHUNKS][b][q] for q in common}
        pen_b[b] = {q: sc_b[P.DEALRANK][b][q] - sc_b[P.CHUNKS][b][q] for q in common}
    return {
        "dense": _boot({b: list(v.values()) for b, v in pen_d.items()}, 100.0),
        "bm25": _boot({b: list(v.values()) for b, v in pen_b.items()}, 100.0),
        "dense_minus_bm25": _boot({b: [pen_d[b][q] - pen_b[b][q] for q in pen_d[b]] for b in pen_d}, 100.0),
    }


# ---- Figure 2 ----------------------------------------------------------------------------
def figure2_data() -> dict[str, Any]:
    """Every value Figure 2 draws, from the sliding-window records of ``table1_gemini`` and ``dense_gemini``.

    ``points`` are the pooled means over the queries with a full pool (with a
    plain query bootstrap), ``pools`` the same per benchmark; x is the mean
    retriever rank of the sliding window's top ten. ``mallows_curve`` is the
    model family on a fixed grid of dispersions: no run enters it.
    """
    owners = {"chunk": owner_of(chunks(N, HANDS), N), "deal": owner_of(deal(N, HANDS), N)}
    points, pools = [], []
    for retriever in ("bm25", "dense"):
        experiment = P.T1 if retriever == "bm25" else P.DENSE
        ranks: dict[str, list[float]] = {}
        cut: dict[str, dict[str, list[float]]] = {split: {} for split in owners}
        for b in runs.benchmarks_of(retriever):
            pool = runs.pool_docids(retriever, b)
            window = runs.load_arm((P.Run(experiment, P.WINDOW),), retriever, b)[0]
            ranks[b] = []
            for split in owners:
                cut[split][b] = []
            for q, rec in sorted(window.items()):
                if rec["n_candidates"] != N:
                    continue
                pos = _positions(pool[q])
                top = [pos[d] for d in rec["ranking"][:10]]
                ranks[b].append(float(np.mean(top)) + 1.0)
                for split, owner in owners.items():
                    cut[split][b].append(float(excess(top, owner)))
        mean_rank = float(np.mean([r for b in ranks for r in ranks[b]]))
        for split in owners:
            boot = bootstrap_mean([v for b in cut[split] for v in cut[split][b]])
            points.append(
                {
                    "key": f"{retriever}_{split}",
                    "split": split,
                    "retriever": "BM25" if retriever == "bm25" else "dense",
                    "benchmarks": sum(1 for b in ranks if ranks[b]),
                    "value": boot["mean"],
                    "ci": boot["ci"],
                    "n_queries": boot["n"],
                    "mean_rank_top10": mean_rank,
                }
            )
        for b in ranks:
            if not ranks[b]:
                continue
            for split in owners:
                pools.append(
                    {
                        "retriever": retriever,
                        "benchmark": b,
                        "split": split,
                        "n_queries": len(cut[split][b]),
                        "mean_rank_top10": float(np.mean(ranks[b])),
                        "measured": float(np.mean(cut[split][b])),
                    }
                )
    points.sort(key=lambda p: (p["split"] != "chunk", p["retriever"] != "BM25"))
    return {
        "config": {
            "N": expected_cuts.N,
            "H": expected_cuts.H,
            "w": expected_cuts.W,
            "q": expected_cuts.Q,
            "k": expected_cuts.K,
        },
        "random_partition": {"e_cut": float(expected_cuts.l_rand()), "p_cut": float(expected_cuts.p_cut_random())},
        "points": points,
        "pools": pools,
        "mallows_curve": mallows.curve(),
    }


def summary_all() -> dict[str, Any]:
    return {
        "forced_cuts": {r: forced_cuts(r) for r in ("bm25", "dense")},
        "random_overload": random_overload(),
        "chunk_penalty": chunk_penalty(),
    }


def _ci(c: Mapping[str, Any], nd: int = 1) -> str:
    return f"{c['diff']:+.{nd}f} [{c['ci'][0]:+.{nd}f}, {c['ci'][1]:+.{nd}f}]"


def render(s: Mapping[str, Any]) -> str:
    fc, ov, pen = s["forced_cuts"], s["random_overload"], s["chunk_penalty"]
    lines = [
        "# Measured cuts (Gemini 3 Flash, queries with 100 candidates)",
        "",
        "Of the sliding window's top ten, how many a split puts over its group's quota of four:",
        "",
        "| retriever | queries | contiguous chunks | dealt hands |",
        "|---|---|---|---|",
    ]
    for r, v in fc.items():
        c, d = v["chunks"], v["dealt"]
        lines.append(
            f"| {r} | {c['n']} | {c['mean']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] | "
            f"{d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] |"
        )
    lines += [
        "",
        f"Judged-relevant passages in the BM25 top 20, by how many share their first-round group "
        f"(random partition, {ov['seeds']} seeds):",
        "",
        "| in its group | dealt | random | random: reach final (%) | minus dealt (points) |",
        "|---|---|---|---|---|",
    ]
    for row, v in ov["by_load"].items():
        fin = f"{v['random_final']['diff']:.1f}" if v["random_final"] else "--"
        gap = _ci(v["minus_dealt"]) if v["minus_dealt"] else "--"
        lines.append(f"| {row} | {v['share_dealt']:.1f}% | {v['share_random']:.1f}% | {fin} | {gap} |")
    lines += [
        "",
        f"In a group holding five or more: {ov['overloaded_share']:.1f}% of those passages.",
        "",
        "Of the sliding window's top ten, random partition minus dealt hands (passages per query): "
        f"{_ci(ov['reference_top10_over_quota_random_minus_dealt'], 3)} more over their group's quota, "
        f"{_ci(ov['reference_top10_cut_random_minus_dealt'], 3)} fewer brought to the final call.",
        "",
        f"DealRank minus chunks, nDCG@10 points: dense {_ci(pen['dense'])}, BM25 {_ci(pen['bm25'])}, "
        f"dense minus BM25 {_ci(pen['dense_minus_bm25'])} ({pen['dense']['queries']} queries).",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cuts forced by the split, over-full random groups, chunks' cost.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--figure2", action="store_true", help="print Figure 2's data as JSON")
    args = parser.parse_args(argv)
    if args.figure2:
        print(json.dumps(figure2_data(), indent=1))
        return
    s = summary_all()
    print(json.dumps(s, indent=1) if args.json else render(s))


if __name__ == "__main__":
    main()
