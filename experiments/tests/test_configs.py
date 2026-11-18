"""Every config parses, every schedule spec builds, and every model entry is complete."""

from __future__ import annotations

from pathlib import Path

import pytest

from repro.config import CONFIGS, load_config
from repro.models import build_llm, cost_usd, entry, load_models
from repro.schedules import build_schedule

EXPERIMENT_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if p.name != "models.yaml")
PAPER_STEMS = {
    "table1_gemini",
    "table1_gemini_repeats",
    "table1_local_timing",
    "table2_models",
    "table3_depth",
    "fig1_api_timing",
    "fig3_split_ablation",
    "dense_gemini",
    "quota_sweep",
    "local_latency",
    "load",
}


def test_every_paper_experiment_has_a_config() -> None:
    assert {p.stem for p in EXPERIMENT_CONFIGS} == PAPER_STEMS


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_config_parses_and_every_schedule_builds(path: Path) -> None:
    experiment = load_config(path)
    names = [build_schedule(spec).name for spec in experiment.schedules]
    assert len(set(names)) == len(names)
    assert experiment.grids or experiment.timing
    cells = experiment.cells()
    assert len({c.path for c in cells}) == len(cells)
    models = load_models()
    for arm in experiment.arms():
        assert arm.llm in models


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_config_header_says_what_it_reproduces_and_how(path: Path) -> None:
    header = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        header.append(line)
    text = "\n".join(header)
    assert header and ("python -m repro.run" in text or "python -m repro.timing" in text)
    assert "Expected" in text


def test_model_entries_are_complete() -> None:
    for name, spec in load_models().items():
        assert {"label", "backend", "id", "price_per_m"} <= set(spec), name
        assert spec["backend"] in ("openrouter", "vllm"), name
    local = entry("qwen3.8-27b")
    assert local["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert local["serve"]["tensor_parallel_size"] == 2
    assert local["serve"]["prefix_caching"] is False
    assert cost_usd("gemini-3-flash", 1_000_000, 0) == pytest.approx(0.5)


def test_rankers_carry_the_decoding_regime_and_routing() -> None:
    pytest.importorskip("openai")
    kimi = build_llm("kimi-k2.6")
    assert kimi.extra_body == {"reasoning": {"enabled": False}, "provider": {"ignore": ["DigitalOcean", "Venice"]}}
    assert (kimi.temperature, kimi.seed, kimi.max_tokens, kimi.max_passage_words) == (0.0, 42, 1024, 300)
    glm = build_llm("glm-5.2")
    assert glm.extra_body["provider"] == {"ignore": ["GMICloud", "DigitalOcean"]}
    local = build_llm("qwen3.8-27b", base_url="http://127.0.0.1:9000/v1", max_concurrency=7)
    assert local.base_url == "http://127.0.0.1:9000/v1"
    assert local.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert local.max_concurrency == 7
