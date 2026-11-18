"""The RankGPT permutation prompt and its parser.

Every schedule in this package sends the same prompt, the multi-turn
permutation prompt of RankGPT (Sun et al., 2023), and reads the answer with the
same parser. Passages are cut to their first ``max_words`` whitespace-separated
words, 300 by default, as in the paper's experiments.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
)

PREFIX_TEMPLATE = (
    "I will provide you with {num} passages, each indicated by number identifier []. "
    "\nRank the passages based on their relevance to query: {query}."
)

SUFFIX_TEMPLATE = (
    "Search Query: {query}. \nRank the {num} passages above based on their relevance "
    "to the search query. The passages should be listed in descending order using "
    "identifiers. The most relevant passages should be listed first. The output format "
    "should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say "
    "any word or explain."
)

Message = dict[str, str]


def truncate_words(text: str, max_words: int) -> str:
    """Keep the first ``max_words`` words of ``text``, joined by single spaces."""
    words = text.replace("\n", " ").strip().split()
    return " ".join(words[:max_words])


def rankgpt_messages(query: str, passages: Sequence[str], max_words: int = 300) -> list[Message]:
    """Chat messages asking the model to order ``passages`` by relevance to ``query``.

    Passages are numbered ``[1]`` to ``[n]`` in the order given, one user turn
    each, and each is acknowledged by an assistant turn, as in RankGPT.
    """
    num = len(passages)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PREFIX_TEMPLATE.format(num=num, query=query)},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for i, passage in enumerate(passages, 1):
        messages.append({"role": "user", "content": f"[{i}] {truncate_words(passage, max_words)}"})
        messages.append({"role": "assistant", "content": f"Received passage [{i}]."})
    messages.append({"role": "user", "content": SUFFIX_TEMPLATE.format(num=num, query=query)})
    return messages


@dataclass(frozen=True)
class ParsedPermutation:
    """A permutation read from model output, with the repairs it needed.

    ``permutation`` holds 0-based positions into the prompt's passages, best
    first, and always covers every position exactly once.
    """

    permutation: list[int]
    repairs: list[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return bool(self.repairs)


def parse_permutation(text: str, num: int) -> ParsedPermutation:
    """Read a permutation of ``range(num)`` from a model's answer.

    Every integer in ``text`` is read in order, so ``[3] > [1]`` and ``3 > 1``
    parse alike. Identifiers are 1-based. The repairs are fixed: repeated
    identifiers keep their first occurrence (``dedupe``), identifiers outside
    ``1..num`` are dropped (``out_of_range``), and missing positions are
    appended in input order (``fill_missing``). An answer with no integer at
    all is recorded as ``empty`` and returns the input order.

    >>> parse_permutation("[2] > [3] > [2] > [9]", 4)
    ParsedPermutation(permutation=[1, 2, 0, 3], repairs=['dedupe', 'out_of_range', 'fill_missing'])
    """
    tokens = [int(t) for t in re.findall(r"\d+", text)]
    repairs: list[str] = []
    if not tokens:
        repairs.append("empty")

    seen: set[int] = set()
    permutation: list[int] = []
    duplicate = out_of_range = False
    for token in tokens:
        idx = token - 1
        if idx < 0 or idx >= num:
            out_of_range = True
            continue
        if idx in seen:
            duplicate = True
            continue
        seen.add(idx)
        permutation.append(idx)

    if duplicate:
        repairs.append("dedupe")
    if out_of_range:
        repairs.append("out_of_range")
    if len(permutation) < num:
        if tokens:
            repairs.append("fill_missing")
        permutation.extend(i for i in range(num) if i not in seen)
    return ParsedPermutation(permutation=permutation, repairs=repairs)
