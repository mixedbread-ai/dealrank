"""The released-BlitzRank adapter, offline.

These tests need the optional ``blitzrank`` package (and the LiteLLM it
depends on) and are skipped without it. No request leaves the process: the
ranker is a :class:`~dealrank.FakeLLM`, or ``litellm.acompletion`` is
replaced by a fake that answers like a consistent model.
"""

from __future__ import annotations

import asyncio
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("blitzrank")

from dealrank import FakeLLM, deal
from repro.blitzrank_package import BlitzRankPackage

N = 100


def _docs(seed: int = 7) -> tuple[list[str], list[int]]:
    """Passages whose merit is the number they carry (smaller is better), in a shuffled order."""
    merit = list(range(N))
    random.Random(seed).shuffle(merit)
    return [f"passage {m} about the query" for m in merit], merit


def _merit(text: str) -> int:
    return int(re.search(r"passage (\d+)", text).group(1))


def _fake_llm() -> FakeLLM:
    return FakeLLM(lambda _q, p: -_merit(p))


def _run(schedule: BlitzRankPackage, docs: list[str], llm: Any) -> Any:
    return asyncio.run(schedule.arerank("which passage", docs, llm=llm))


def test_package_certifies_the_true_top_with_a_consistent_ranker() -> None:
    docs, merit = _docs()
    result = _run(BlitzRankPackage(window=20, top_m=10), docs, _fake_llm())
    assert sorted(result.ranking) == list(range(N))
    assert [merit[i] for i in result.ranking[:10]] == list(range(10))
    assert result.round_widths == [1] * result.num_calls
    assert result.diagnostics["certified"] is True
    assert len(result.diagnostics["calls"]) == result.num_calls
    assert result.diagnostics["calls"][:5] == [list(range(j * 20, (j + 1) * 20)) for j in range(5)]
    for call in (c for r in result.rounds for c in r):
        assert call.started_at is not None and call.started_at <= call.ended_at


def test_concurrent_cover_changes_only_the_rounds() -> None:
    docs, _ = _docs()
    sequential = _run(BlitzRankPackage(window=20, top_m=10), docs, _fake_llm())
    concurrent = _run(BlitzRankPackage(window=20, top_m=10, concurrent_cover=True), docs, _fake_llm())
    assert concurrent.ranking == sequential.ranking
    assert concurrent.diagnostics["calls"] == sequential.diagnostics["calls"]
    assert concurrent.round_widths[0] == 5
    assert concurrent.num_calls == sequential.num_calls
    assert concurrent.num_rounds == sequential.num_rounds - 4


def test_dealt_input_opens_on_dealrank_hands() -> None:
    docs, merit = _docs()
    llm = FakeLLM(lambda _q, p: -_merit(p), delay_s=0.01)
    result = _run(BlitzRankPackage(window=20, top_m=10, input_order="dealt", concurrent_cover=True), docs, llm)
    assert sorted(map(sorted, result.diagnostics["calls"][:5])) == sorted(deal(N, 5))
    assert llm.max_in_flight == 5
    assert [merit[i] for i in result.ranking[:10]] == list(range(10))


def _fake_acompletion(seen: list[dict[str, Any]]) -> Any:
    async def acompletion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        messages = kwargs["messages"]
        num = (len(messages) - 4) // 2
        turns = [m["content"] for m in messages[3 : 3 + 2 * num : 2]]
        order = sorted(range(num), key=lambda j: _merit(turns[j]))
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=None)
        choice = SimpleNamespace(message=SimpleNamespace(content=" > ".join(f"[{j + 1}]" for j in order)))
        return SimpleNamespace(choices=[choice], usage=usage, model=kwargs["model"])

    return acompletion


def test_decoding_regime_is_injected_at_the_litellm_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    import litellm

    from dealrank.llm.openai_compatible import OpenAICompatible

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion(seen))
    llm = OpenAICompatible.openrouter(
        "google/gemini-3-flash-preview",
        api_key="unused",
        extra_body={"reasoning": {"enabled": False}, "provider": {"ignore": ["SomeProvider"]}},
    )
    docs, merit = _docs()
    result = _run(BlitzRankPackage(window=20, top_m=10), docs, llm)
    assert [merit[i] for i in result.ranking[:10]] == list(range(10))
    assert len(seen) == result.num_calls
    first = seen[0]
    assert first["model"] == "openrouter/google/gemini-3-flash-preview"
    assert first["temperature"] == 0.0
    assert first["seed"] == 42
    assert first["max_tokens"] == 1024
    assert first["extra_body"]["reasoning"] == {"enabled": False}
    assert first["extra_body"]["provider"] == {"ignore": ["SomeProvider"]}
    assert first["timeout"] == 60.0  # the package's own timeout
    assert result.rounds[0][0].prompt_tokens == 100


def test_local_server_is_reached_through_the_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    import litellm

    from dealrank.llm.openai_compatible import OpenAICompatible

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion(seen))
    llm = OpenAICompatible.vllm("Qwen/Qwen3.8-27B", base_url="http://127.0.0.1:8000/v1")
    docs, _ = _docs()
    _run(BlitzRankPackage(window=20, top_m=10, concurrent_cover=True), docs, llm)
    assert seen[0]["model"] == "openai/Qwen/Qwen3.8-27B"
    assert seen[0]["api_base"] == "http://127.0.0.1:8000/v1"
    assert seen[0]["api_key"] == "EMPTY"
    assert seen[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
