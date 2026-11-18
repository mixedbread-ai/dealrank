from __future__ import annotations

import pytest
from support import doc, true_rank

from dealrank import FakeLLM, JointRank, ListK, RetrieverOrder, SlidingWindow, TDPart, TopPrefix, TourRank
from dealrank.schedules import SCHEDULES, Reordered, build_schedule
from dealrank.schedules.tourrank import stage_plan


def test_sliding_window_runs_nine_dependent_calls_bottom_up(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = SlidingWindow(perfect_llm, window=20, stride=10).rerank("q", passages)
    assert result.round_widths == [1] * 9
    first = [true_rank(p) for p in perfect_llm.requests[0][1]]
    assert first == list(range(80, 100))
    assert result.ranking[:10] == list(range(10))


def test_tourrank_stage_plan_and_calls(passages: list[str], perfect_llm: FakeLLM) -> None:
    assert stage_plan(100) == [(5, 50), (5, 20), (1, 10), (1, 5), (1, 2)]
    result = TourRank(perfect_llm, tournaments=1).rerank("q", passages)
    assert result.round_widths == [5, 5, 1, 1, 1]
    assert result.ranking[:2] == [0, 1]


def test_tourrank_first_stage_deals_and_chunk_variants_do_not(passages: list[str]) -> None:
    for split, expected in (("deal", list(range(0, 100, 5))), ("chunks_first", list(range(20)))):
        llm = FakeLLM()
        TourRank(llm, split=split).rerank("q", passages)
        assert sorted(true_rank(p) for p in llm.requests[0][1]) == expected


def test_tdpart_parallel_rounds(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = TDPart(perfect_llm).rerank("q", passages)
    # Rank the top 20, then the 80-passage tail in windows of 19 with the pivot.
    assert result.round_widths == [1, 5]
    assert result.ranking[:10] == list(range(10))


def test_tdpart_budget_caps_promotion_in_window_order(perfect_llm: FakeLLM) -> None:
    # TDPart's budget b (the original algorithm's, default: the window) caps the
    # passages promoted past the pivot. With a retriever order that reverses the
    # true order, every tail window beats the pivot, the budget fills from the
    # first windows, and the best passages, which come last, stay below the cut.
    reversed_pool = [doc(99 - i) for i in range(100)]
    capped = TDPart(perfect_llm).rerank("q", reversed_pool)
    assert [true_rank(reversed_pool[i]) for i in capped.ranking[:10]] == list(range(61, 71))
    # Which passages the cap keeps depends on how the tail is cut into windows.
    dealt = TDPart(perfect_llm, tail_split="deal").rerank("q", reversed_pool)
    assert [true_rank(reversed_pool[i]) for i in dealt.ranking[:10]] == list(range(4, 50, 5))
    # A budget that cannot fill recovers the true top ten.
    uncapped = TDPart(perfect_llm, budget=100).rerank("q", reversed_pool)
    assert [true_rank(reversed_pool[i]) for i in uncapped.ranking[:10]] == list(range(10))


def test_tdpart_dealt_tail(passages: list[str]) -> None:
    llm = FakeLLM()
    TDPart(llm, tail_split="deal").rerank("q", passages)
    tail_windows = [[true_rank(p) for p in ps[1:]] for _, ps in llm.requests[1:6]]
    assert tail_windows[0] == list(range(20, 100, 5))


def test_jointrank_one_round_of_equireplicate_blocks(passages: list[str], perfect_llm: FakeLLM) -> None:
    pytest.importorskip("numpy")
    for design in ("random", "dealt"):
        schedule = JointRank(perfect_llm, design=design)
        blocks = schedule.blocks(100)
        assert len(blocks) == 20
        assert all(len(set(b)) == len(b) == 20 for b in blocks)
        counts = [sum(i in b for b in blocks) for i in range(100)]
        assert counts == [4] * 100
        result = schedule.rerank("q", passages)
        assert result.round_widths == [20]
        # One round of PageRank need not recover the exact order, even from a perfect ranker.
        assert result.ranking[0] == 0
        assert len(set(result.ranking[:10]) & set(range(10))) >= 8


def test_listk_selects_and_sorts_the_top_ten(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = ListK(perfect_llm).rerank("q", passages)
    assert result.ranking[:10] == list(range(10))
    assert result.round_widths[0] == 1  # the pivot-sort call
    assert result.round_widths[-1] == 1  # the final sort


def test_top_prefix_and_retriever_order(passages: list[str]) -> None:
    llm = FakeLLM(lambda _q, p: true_rank(p))
    result = TopPrefix(llm, size=20).rerank("q", passages)
    assert result.ranking[:20] == list(range(19, -1, -1))
    assert result.ranking[20:] == list(range(20, 100))
    assert RetrieverOrder().rerank("q", passages).ranking == list(range(100))


def test_build_schedule_round_trips_every_schedule() -> None:
    for kind in SCHEDULES:
        assert build_schedule({"type": kind}).name


def test_build_schedule_input_order() -> None:
    assert build_schedule({"type": "blitzrank", "input_order": "retriever"}).name == "blitzrank-w20-m10"
    assert build_schedule({"type": "blitzrank"}).name == "blitzrank-w20-m10-dealt"
    wrapped = build_schedule({"type": "sliding_window", "input_order": "dealt"})
    assert isinstance(wrapped, Reordered)
    assert wrapped.name == "sliding-window-w20-s10-dealt"
    with pytest.raises(ValueError, match="unknown schedule"):
        build_schedule({"type": "nope"})


def test_reordered_maps_rankings_back(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = Reordered(SlidingWindow(), "dealt").rerank("q", passages, llm=perfect_llm)
    assert result.ranking[:10] == list(range(10))
    assert sorted(result.ranking) == list(range(100))


def test_missing_llm_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="has no llm"):
        SlidingWindow().rerank("q", [doc(0), doc(1)])
