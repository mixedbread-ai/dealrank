from __future__ import annotations

from dealrank.prompt import parse_permutation, rankgpt_messages, truncate_words


def test_rankgpt_messages_structure_and_wording() -> None:
    messages = rankgpt_messages("how do cards work", ["first passage", "second\npassage"])
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == (
        "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
    )
    assert messages[1]["content"] == (
        "I will provide you with 2 passages, each indicated by number identifier []. "
        "\nRank the passages based on their relevance to query: how do cards work."
    )
    assert messages[2]["content"] == "Okay, please provide the passages."
    assert messages[3]["content"] == "[1] first passage"
    assert messages[4]["content"] == "Received passage [1]."
    assert messages[5]["content"] == "[2] second passage"
    assert messages[-1]["content"] == (
        "Search Query: how do cards work. \nRank the 2 passages above based on their relevance to the "
        "search query. The passages should be listed in descending order using identifiers. The most "
        "relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. "
        "Only response the ranking results, do not say any word or explain."
    )


def test_passages_are_cut_to_300_words() -> None:
    long = " ".join(f"w{i}" for i in range(500))
    content = rankgpt_messages("q", [long])[3]["content"]
    assert content == "[1] " + " ".join(f"w{i}" for i in range(300))
    assert truncate_words("  a\nb   c ", 2) == "a b"


def test_parse_clean_permutation() -> None:
    parsed = parse_permutation("[3] > [1] > [2]", 3)
    assert parsed.permutation == [2, 0, 1]
    assert not parsed.repaired


def test_parse_bare_digits_and_prose() -> None:
    assert parse_permutation("Ranking: 2 > 1", 2).permutation == [1, 0]


def test_parse_repairs_in_fixed_order() -> None:
    parsed = parse_permutation("[2] > [3] > [2] > [9]", 4)
    assert parsed.permutation == [1, 2, 0, 3]
    assert parsed.repairs == ["dedupe", "out_of_range", "fill_missing"]


def test_parse_empty_answer_returns_input_order() -> None:
    parsed = parse_permutation("I cannot rank these.", 3)
    assert parsed.permutation == [0, 1, 2]
    assert parsed.repairs == ["empty"]


def test_parse_zero_is_out_of_range() -> None:
    assert parse_permutation("[0] > [1]", 2).repairs == ["out_of_range", "fill_missing"]
