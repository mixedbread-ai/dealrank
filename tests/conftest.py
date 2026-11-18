from __future__ import annotations

import pytest
from support import doc, true_rank

from dealrank import FakeLLM


@pytest.fixture
def passages() -> list[str]:
    """100 candidates whose retriever order is also the true order."""
    return [doc(i) for i in range(100)]


@pytest.fixture
def perfect_llm() -> FakeLLM:
    """A consistent ranker that knows the true order."""
    return FakeLLM(lambda _q, p: -true_rank(p))
