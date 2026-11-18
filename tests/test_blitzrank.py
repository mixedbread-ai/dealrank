from __future__ import annotations

import random

from support import doc, true_rank

from dealrank import BlitzRank, DealRank, FakeLLM
from dealrank.partition import chunks, deal, dealt_order


def noisy(seed: int):
    """A ranker whose order differs from the retriever's by seeded noise.

    The noise is fixed per query and passage, so the ranker is still consistent
    from one call to the next.
    """

    def score(query: str, passage: str) -> float:
        return -true_rank(passage) + random.Random(f"{seed}|{query}|{passage}").gauss(0, 8)

    return score


def test_dealt_input_opens_on_dealrank_hands(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = BlitzRank(perfect_llm, window=20, top_m=10, input_order="dealt").rerank("q", passages)
    opening = [[true_rank(p) for p in ps] for _, ps in perfect_llm.requests[:5]]
    assert opening == deal(100, 5)
    # The sixth call merges the top four of each hand, as DealRank's final call does.
    merge = sorted(true_rank(p) for p in perfect_llm.requests[5][1])
    assert merge == list(range(20))
    assert result.num_calls == 6
    assert result.diagnostics["certified"]
    assert result.ranking[:10] == list(range(10))


def test_dealt_input_on_a_pool_that_is_not_a_multiple_of_the_window() -> None:
    # 95 candidates deal into five hands of 19, but the opening blocks are still
    # cut every 20 positions of the dealt order: the first block is hand 0 and
    # the first passage of hand 1, so the blocks are not DealRank's hands.
    llm = FakeLLM(lambda _q, p: -true_rank(p))
    BlitzRank(llm, window=20, input_order="dealt").rerank("q", [doc(i) for i in range(95)])
    hands = deal(95, 5)
    assert [len(h) for h in hands] == [19] * 5
    first = [true_rank(p) for p in llm.requests[0][1]]
    assert first == hands[0] + hands[1][:1]
    assert first == dealt_order(95, 20)[:20]
    # With 85 candidates the first block holds seven of the retriever's top twenty; a hand holds four.
    llm = FakeLLM(lambda _q, p: -true_rank(p))
    BlitzRank(llm, window=20, input_order="dealt").rerank("q", [doc(i) for i in range(85)])
    assert sum(true_rank(p) < 20 for p in llm.requests[0][1]) == 7
    assert all(sum(i < 20 for i in hand) == 4 for hand in deal(85, 5))


def test_retriever_order_opens_on_chunks_and_needs_a_certification_call(
    passages: list[str], perfect_llm: FakeLLM
) -> None:
    result = BlitzRank(perfect_llm, input_order="retriever").rerank("q", passages)
    opening = [[true_rank(p) for p in ps] for _, ps in perfect_llm.requests[:5]]
    assert opening == chunks(100, 5)
    assert result.num_calls == 7
    assert result.ranking[:10] == list(range(10))


def test_dealt_ranking_maps_back_to_input_positions(passages: list[str], perfect_llm: FakeLLM) -> None:
    result = BlitzRank(perfect_llm).rerank("q", passages)
    assert result.diagnostics["input_order"]["permutation"] == dealt_order(100, 20)
    assert sorted(result.ranking) == list(range(100))
    assert result.ranking[:10] == list(range(10))


def test_concurrent_cover_changes_rounds_but_not_decisions(passages: list[str]) -> None:
    for seed in range(5):
        seq_llm, par_llm = FakeLLM(noisy(seed)), FakeLLM(noisy(seed))
        seq = BlitzRank(seq_llm).rerank("q", passages)
        par = BlitzRank(par_llm, concurrent_cover=True).rerank("q", passages)
        assert seq.ranking == par.ranking
        assert seq_llm.requests == par_llm.requests
        assert par.round_widths[0] == 5
        assert par.num_rounds == seq.num_rounds - 4
        assert all(w == 1 for w in seq.round_widths)


def test_concurrent_dealt_blitzrank_matches_dealrank_when_it_certifies(passages: list[str]) -> None:
    llm_a, llm_b = FakeLLM(lambda _q, p: -true_rank(p)), FakeLLM(lambda _q, p: -true_rank(p))
    blitz = BlitzRank(llm_a, concurrent_cover=True).rerank("q", passages)
    dealrank = DealRank(llm_b).rerank("q", passages)
    assert blitz.round_widths == dealrank.round_widths == [5, 1]
    assert blitz.ranking[:10] == dealrank.ranking[:10]


def test_window_ten_deals_into_ten_hands(passages: list[str], perfect_llm: FakeLLM) -> None:
    BlitzRank(perfect_llm, window=10).rerank("q", passages)
    opening = [[true_rank(p) for p in ps] for _, ps in perfect_llm.requests[:10]]
    assert opening == deal(100, 10)


def test_small_and_empty_inputs() -> None:
    llm = FakeLLM(lambda _q, p: -true_rank(p))
    one_call = BlitzRank(llm).rerank("q", [doc(i) for i in range(15)])
    assert one_call.num_calls == 1
    assert one_call.ranking[:10] == list(range(10))
    assert BlitzRank(llm).rerank("q", []).num_calls == 0


def test_every_answer_yields_a_permutation(passages: list[str]) -> None:
    for seed in range(5):
        result = BlitzRank(FakeLLM(noisy(seed)), input_order="retriever").rerank("q", passages)
        assert sorted(result.ranking) == list(range(100))
        assert result.diagnostics["certified"]
