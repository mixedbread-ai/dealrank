"""nDCG@10 on toy qrels, and the pure implementation against ir_measures."""

from __future__ import annotations

import math
import random

import pytest

from repro.evaluate import ndcg_at_k, per_query_ndcg

QRELS = {"q1": {"a": 2, "b": 1, "c": 0}, "q2": {"x": 1}, "q3": {"y": 0}}


def test_known_values() -> None:
    assert ndcg_at_k(["a", "b", "c"], QRELS["q1"]) == pytest.approx(1.0)
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], QRELS["q1"]) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([f"z{i}" for i in range(10)] + ["x"], QRELS["q2"]) == 0.0  # relevant passage at rank 11
    assert ndcg_at_k(["z"] * 10 + ["x"], QRELS["q2"]) == pytest.approx(1 / math.log2(3))  # a repeat counts once


def test_population() -> None:
    run = {"q1": ["a", "b"], "q2": [], "q3": ["y"], "q4": ["a"]}
    scores = per_query_ndcg(QRELS, run)
    assert set(scores) == {"q1", "q2", "q3"}  # no qrels: not scored
    assert scores["q2"] == 0.0  # an empty pool scores 0
    assert scores["q3"] == 0.0  # no positive grade scores 0


def test_negative_grades_count_as_zero() -> None:
    grades = {"a": -2, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], grades) == pytest.approx((2 / math.log2(3) + 1 / 2) / (2 + 1 / math.log2(3)))


def test_pure_matches_ir_measures() -> None:
    ir_measures = pytest.importorskip("ir_measures")
    assert ir_measures  # the implementation used by per_query_ndcg
    rng = random.Random(3)
    qrels, run = {}, {}
    for q in range(60):
        docs = [f"d{q}-{i}" for i in range(rng.randint(0, 40))]
        qrels[f"q{q}"] = {d: rng.choice([0, 0, 1, 2, 3]) for d in docs if rng.random() < 0.5}
        qrels[f"q{q}"][f"unretrieved{q}"] = rng.choice([0, 1, 2])
        rng.shuffle(docs)
        run[f"q{q}"] = docs
    scores = per_query_ndcg(qrels, run)
    for q, v in scores.items():
        assert ndcg_at_k(run[q], qrels[q]) == pytest.approx(v, abs=1e-12)
