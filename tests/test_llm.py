from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from support import doc

from dealrank import DealRank, FakeLLM, ResponseCache
from dealrank.llm.base import Completion, ListwiseLLM
from dealrank.prompt import Message, rankgpt_messages


class Echo(ListwiseLLM):
    """Answers every call with a fixed text and counts requests."""

    model = "echo"

    def __init__(self, text: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.text = text
        self.sent: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> Completion:
        self.sent.append(messages)
        return Completion(text=self.text, prompt_tokens=10, completion_tokens=3, latency_s=0.5)


def test_rank_parses_and_reports_usage() -> None:
    llm = Echo("[2] > [1]")
    call = asyncio.run(llm.rank("q", ["a", "b"]))
    assert call.permutation == [1, 0]
    assert (call.prompt_tokens, call.completion_tokens, call.latency_s) == (10, 3, 0.5)
    assert llm.sent[0] == rankgpt_messages("q", ["a", "b"])


def test_cache_serves_identical_requests(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "responses.sqlite")
    llm = Echo("[2] > [1]", cache=cache)
    first = asyncio.run(llm.rank("q", ["a", "b"]))
    second = asyncio.run(llm.rank("q", ["a", "b"]))
    assert not first.cached and second.cached
    assert second.permutation == [1, 0]
    assert len(llm.sent) == 1
    asyncio.run(llm.rank("q", ["b", "a"]))
    assert len(llm.sent) == 2


def test_request_key_is_sha256_of_model_messages_and_params() -> None:
    llm = Echo("")
    payload = json.dumps({"model": "echo", "messages": rankgpt_messages("q", ["a"])}, sort_keys=True)
    assert llm.request_key("q", ["a"]) == hashlib.sha256(payload.encode()).hexdigest()


def test_openai_compatible_request_params() -> None:
    pytest.importorskip("openai")
    from dealrank.llm import OpenAICompatible

    router = OpenAICompatible.openrouter("some/model", api_key="unused")
    assert router.request_params() == {
        "max_tokens": 1024,
        "extra_body": {"reasoning": {"enabled": False}},
        "seed": 42,
        "temperature": 0.0,
    }
    local = OpenAICompatible.vllm("Qwen/Qwen3.8-27B")
    assert local.request_params()["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_fake_llm_concurrency_is_bounded() -> None:
    llm = FakeLLM(delay_s=0.02, max_concurrency=2)
    DealRank(llm).rerank("q", [doc(i) for i in range(100)])
    assert llm.max_in_flight == 2


def test_rerank_works_inside_a_running_event_loop() -> None:
    async def inner() -> list[int]:
        return DealRank(FakeLLM()).rerank("q", [doc(i) for i in range(40)]).ranking

    assert sorted(asyncio.run(inner())) == list(range(40))


def test_import_needs_no_optional_dependency() -> None:
    code = (
        "import sys, dealrank\n"
        "from dealrank import DealRank, BlitzRank, deal\n"
        "heavy = [m for m in ('openai', 'litellm', 'numpy', 'httpx') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_openai_key_is_never_sent_to_another_server(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    from dealrank.llm import OpenAICompatible

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert OpenAICompatible.openrouter("some/model")._resolved_key() == "EMPTY"
    assert OpenAICompatible.vllm("some/model")._resolved_key() == "EMPTY"
    assert OpenAICompatible("gpt-5.5")._resolved_key() == "openai-secret"
