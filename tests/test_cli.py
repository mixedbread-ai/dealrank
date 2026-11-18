from __future__ import annotations

import json
from pathlib import Path

import pytest

from dealrank.cli import main


def test_deal_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["deal", "10", "--hands", "3"]) == 0
    assert capsys.readouterr().out.splitlines() == ["hand   1: 1 4 7 10", "hand   2: 2 5 8", "hand   3: 3 6 9"]


def test_rerank_command_offline(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    with open(queries, "w") as handle:
        for qid in ("q1", "q2"):
            candidates = [{"docid": f"{qid}-d{i}", "text": f"passage {i}"} for i in range(45)]
            handle.write(json.dumps({"qid": qid, "query": "cards", "candidates": candidates}) + "\n")
    out = tmp_path / "out.jsonl"
    assert main(["rerank", str(queries), "--fake", "-o", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["qid"] for r in records] == ["q1", "q2"]
    assert records[0]["rounds"] == 2 and records[0]["calls"] == 4
    assert sorted(records[0]["ranking"]) == sorted(f"q1-d{i}" for i in range(45))


def test_serve_vllm_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve-vllm", "Qwen/Qwen3.8-27B", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--tensor-parallel-size 2" in out
    assert "--disable-prefix-caching" in out
    assert "--dtype bfloat16" in out


def _built_llm(argv: list[str]):
    from dealrank.cli import _build_llm, build_parser

    return _build_llm(build_parser().parse_args(["rerank", "-", *argv]))


def test_openai_key_is_not_sent_to_a_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    llm = _built_llm(["--model", "m", "--base-url", "http://127.0.0.1:9/v1"])
    assert llm.api_key is None
    assert llm._resolved_key() == "EMPTY"


def test_custom_base_url_takes_a_key_only_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("MY_SERVER_KEY", "server-key")
    llm = _built_llm(["--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--api-key-env", "MY_SERVER_KEY"])
    assert llm._resolved_key() == "server-key"
    monkeypatch.delenv("MY_SERVER_KEY")
    with pytest.raises(SystemExit, match="MY_SERVER_KEY is not set"):
        _built_llm(["--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--api-key-env", "MY_SERVER_KEY"])


def test_openai_api_still_reads_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    llm = _built_llm(["--model", "gpt-x"])
    assert llm.base_url is None
    assert llm._resolved_key() == "openai-secret"
