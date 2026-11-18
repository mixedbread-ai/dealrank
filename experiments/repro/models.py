"""The paper's LLMs, from ``configs/models.yaml``, as dealrank rankers.

>>> sorted(load_models())[:3]
['deepseek-v4-flash', 'gemini-3-flash', 'glm-5.2']
>>> round(cost_usd("gemini-3-flash", 2_000_000, 100_000), 2)
1.3
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from dealrank import ListwiseLLM

MODELS_YAML = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"

#: Overrides the ``base_url`` of local models, e.g. ``http://127.0.0.1:8001/v1``.
VLLM_URL_ENV = "DEALRANK_VLLM_URL"


@cache
def _read(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_models(path: Path = MODELS_YAML) -> dict[str, dict[str, Any]]:
    """Every entry of ``models.yaml``, by key."""
    return dict(_read(path)["models"])


def decoding(path: Path = MODELS_YAML) -> dict[str, Any]:
    """The decoding regime shared by every model."""
    return dict(_read(path)["decoding"])


def entry(name: str, path: Path = MODELS_YAML) -> dict[str, Any]:
    models = load_models(path)
    if name not in models:
        raise KeyError(f"no model {name!r} in {path.name}; expected one of {sorted(models)}")
    return dict(models[name])


def is_local(name: str) -> bool:
    """Whether ``name`` is served by a local vLLM server."""
    return entry(name)["backend"] == "vllm"


def build_llm(name: str, *, base_url: str | None = None, **overrides: Any) -> ListwiseLLM:
    """A :class:`~dealrank.llm.OpenAICompatible` ranker for the model ``name``.

    API models go through OpenRouter with the entry's reasoning setting and
    provider routing; local models go to a vLLM server at ``base_url``, the
    ``DEALRANK_VLLM_URL`` environment variable or the entry's ``base_url``,
    in that order. ``overrides`` are passed to the constructor, e.g.
    ``max_concurrency`` or ``attempts``.
    """
    from dealrank.llm.openai_compatible import OpenAICompatible

    spec = entry(name)
    regime = decoding()
    kwargs: dict[str, Any] = {
        "temperature": regime["temperature"],
        "seed": regime["seed"],
        "max_tokens": regime["max_output_tokens"],
        "max_passage_words": regime["max_passage_words"],
    }
    kwargs.update(overrides)
    if spec["backend"] == "openrouter":
        extra_body: dict[str, Any] = {"reasoning": dict(spec.get("reasoning") or {"enabled": False})}
        if spec.get("provider"):
            extra_body["provider"] = dict(spec["provider"])
        return OpenAICompatible.openrouter(spec["id"], extra_body=extra_body, **kwargs)
    if spec["backend"] == "vllm":
        url = base_url or os.environ.get(VLLM_URL_ENV) or spec["base_url"]
        return OpenAICompatible.vllm(spec["id"], base_url=url, **kwargs)
    raise ValueError(f"{name}: unknown backend {spec['backend']!r}")


def cost_usd(name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """The OpenRouter price of the given tokens for model ``name``."""
    price = entry(name).get("price_per_m") or {}
    return (prompt_tokens * price.get("input", 0.0) + completion_tokens * price.get("output", 0.0)) / 1e6
