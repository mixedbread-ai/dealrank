from __future__ import annotations

import pytest
from support import doc, true_rank

from dealrank import DealRank, FakeLLM, HierarchicalDealRank
from dealrank.partition import deal


def test_dealrank_makes_six_calls_in_two_rounds(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = DealRank(perfect_llm, window=20, quota=4).rerank("q", passages)
    assert result.num_calls == 6
    assert result.num_rounds == 2
    assert result.round_widths == [5, 1]
    assert result.ranking[:20] == list(range(20))
    assert sorted(result.ranking) == list(range(100))


def test_first_round_hands_are_dealt_and_in_retriever_order(passages: list[str], perfect_llm: FakeLLM) -> None:
    DealRank(perfect_llm).rerank("q", passages)
    hands = [[true_rank(p) for p in ps] for _, ps in perfect_llm.requests[:5]]
    assert hands == deal(100, 5)


def test_final_call_lists_finalists_by_place_in_hand_then_retriever_rank() -> None:
    # A ranker that reverses every call: each hand's winners are its last four.
    llm = FakeLLM(lambda _q, p: true_rank(p))
    result = DealRank(llm).rerank("q", [doc(i) for i in range(100)])
    final_prompt = [true_rank(p) for p in llm.requests[-1][1]]
    # place 0 in each hand: the hand's deepest member (95..99), then place 1 (90..94), ...
    assert final_prompt == [95, 96, 97, 98, 99, 90, 91, 92, 93, 94, 85, 86, 87, 88, 89, 80, 81, 82, 83, 84]
    assert result.diagnostics["finalists"] == final_prompt
    # Head: the final call's order; then the eliminated, by place in hand, then retriever rank.
    assert result.ranking[:20] == sorted(final_prompt, reverse=True)
    assert result.ranking[20:25] == [75, 76, 77, 78, 79]


def test_default_quota_fills_one_window() -> None:
    llm = FakeLLM()
    result = DealRank(llm).rerank("q", [doc(i) for i in range(60)])
    assert result.round_widths == [3, 1]
    assert result.diagnostics["quota"] == 6
    assert len(llm.requests[-1][1]) == 18


def test_short_lists_take_one_call() -> None:
    llm = FakeLLM()
    result = DealRank(llm).rerank("q", [doc(i) for i in range(20)])
    assert result.round_widths == [1]
    assert DealRank(llm).rerank("q", []).num_calls == 0


def test_final_size_caps_the_final_call_by_place_then_rank() -> None:
    llm = FakeLLM(lambda _q, p: -true_rank(p))
    result = DealRank(llm, quota=5).rerank("q", [doc(i) for i in range(100)])
    final_prompt = [true_rank(p) for p in llm.requests[-1][1]]
    assert len(final_prompt) == 20
    assert final_prompt == list(range(20))  # the 25 qualifiers are cut to the best 20 by place
    assert result.ranking[:20] == list(range(20))


def test_quota_sweep_settings() -> None:
    llm = FakeLLM()
    small = DealRank(llm, window=5, quota=1, final_size=20).rerank("q", [doc(i) for i in range(100)])
    assert small.round_widths == [20, 1]
    assert len(llm.requests[-1][1]) == 20
    mid = DealRank(llm, window=10, quota=2, final_size=20).rerank("q", [doc(i) for i in range(100)])
    assert mid.round_widths == [10, 1]


def test_chunks_split_cuts_the_true_top_twenty(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = DealRank(perfect_llm, split="chunks").rerank("q", passages)
    assert result.diagnostics["hands"][0] == list(range(20))
    assert result.ranking[:5] == [0, 1, 2, 3, 20]


def test_random_split_is_drawn_per_query_and_reproducible() -> None:
    schedule = DealRank(split="random", seed=0, seed_per_query=True)
    a, b = schedule.hands(100, "query a"), schedule.hands(100, "query b")
    assert a != b
    assert a == DealRank(split="random", seed=0, seed_per_query=True).hands(100, "query a")
    assert a != DealRank(split="random", seed=1, seed_per_query=True).hands(100, "query a")
    fixed = DealRank(split="random", seed=0)
    assert fixed.hands(100, "query a") == fixed.hands(100, "query b")


def test_names_are_distinct_per_setting() -> None:
    names = {
        DealRank(quota=4).name,
        DealRank(quota=4, split="chunks").name,
        DealRank(quota=4, split="random", seed=0, seed_per_query=True).name,
        DealRank(quota=4, split="random", seed=1, seed_per_query=True).name,
        DealRank(window=5, quota=1, final_size=20).name,
    }
    assert len(names) == 5


@pytest.mark.parametrize(
    "n,first_quota,plan,widths",
    [(500, 3, [(25, 3), (4, 5)], [25, 4, 1]), (1000, 2, [(50, 2), (5, 4)], [50, 5, 1])],
)
def test_hierarchical_plans_of_the_paper(n: int, first_quota: int, plan: list, widths: list) -> None:
    llm = FakeLLM(lambda _q, p: -true_rank(p))
    schedule = HierarchicalDealRank(llm, first_quota=first_quota)
    assert schedule.plan(n) == plan
    result = schedule.rerank("q", [doc(i) for i in range(n)])
    assert result.round_widths == widths
    assert result.ranking[:20] == list(range(20))
    assert sorted(result.ranking) == list(range(n))
    # The final call lists the survivors in retriever order.
    assert [true_rank(p) for p in llm.requests[-1][1]] == sorted(result.diagnostics["finalists"])


def test_hierarchical_collapses_to_two_rounds_when_one_filter_suffices() -> None:
    result = HierarchicalDealRank(FakeLLM(), first_quota=4).rerank("q", [doc(i) for i in range(100)])
    assert result.round_widths == [5, 1]


def test_hierarchical_refuses_a_plan_that_overfills_the_final_call() -> None:
    with pytest.raises(ValueError, match="lower first_quota"):
        HierarchicalDealRank(first_quota=19).plan(10_000)


def test_hands_run_concurrently() -> None:
    llm = FakeLLM(delay_s=0.05)
    result = DealRank(llm).rerank("q", [doc(i) for i in range(100)])
    assert llm.max_in_flight == 5
    assert result.critical_path_s() < 0.05 * 2 * 1.9
