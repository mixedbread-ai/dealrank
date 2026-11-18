"""LLM backends.

:class:`ListwiseLLM` is the interface every schedule calls. Backends:

* :class:`OpenAICompatible` for OpenAI, OpenRouter, vLLM or any other
  OpenAI-compatible server (extra ``openai``);
* :class:`LiteLLM` for any provider LiteLLM supports (extra ``litellm``);
* :class:`FakeLLM`, an offline ranker for tests and dry runs.

The optional backends are imported on first use, so ``import dealrank`` works
with no extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dealrank.llm.base import Completion, ListwiseLLM
from dealrank.llm.cache import ResponseCache
from dealrank.llm.fake import FakeLLM

if TYPE_CHECKING:
    from dealrank.llm.litellm_backend import LiteLLM
    from dealrank.llm.openai_compatible import OpenAICompatible

__all__ = ["Completion", "FakeLLM", "ListwiseLLM", "LiteLLM", "OpenAICompatible", "ResponseCache"]


def __getattr__(name: str) -> Any:
    if name == "OpenAICompatible":
        from dealrank.llm.openai_compatible import OpenAICompatible

        return OpenAICompatible
    if name == "LiteLLM":
        from dealrank.llm.litellm_backend import LiteLLM

        return LiteLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
