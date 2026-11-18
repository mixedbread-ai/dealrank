"""Helpers for serving a model locally with vLLM.

The paper's local runs serve the model with ``vllm serve`` in bf16 over two
GPUs (tensor parallelism 2) with prefix caching off, so that repeated
passages across calls cannot be served from cache and timings compare
schedules on equal terms. :func:`vllm_command` builds that command line.
vLLM itself is not a dependency of the core package (see the ``vllm`` extra).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request


def vllm_command(
    model: str,
    *,
    port: int = 8000,
    host: str = "127.0.0.1",
    tensor_parallel_size: int = 2,
    dtype: str = "bfloat16",
    prefix_caching: bool = False,
    max_model_len: int = 65536,
    gpu_memory_utilization: float = 0.90,
    seed: int = 42,
    revision: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The ``vllm serve`` argument list for the paper's local setting."""
    cmd = ["vllm", "serve", model]
    if revision:
        cmd += ["--revision", revision]
    cmd += [
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--enable-prompt-tokens-details",
    ]
    if not prefix_caching:
        cmd.append("--disable-prefix-caching")
    return cmd + list(extra_args or [])


def wait_until_healthy(base_url: str, timeout_s: float = 1800.0, poll_s: float = 5.0) -> None:
    """Block until ``{server}/health`` answers 200, or raise ``TimeoutError``.

    ``base_url`` may end in ``/v1``, as passed to an OpenAI client.
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{root}/health", timeout=poll_s) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"no healthy server at {root} after {timeout_s:.0f} s")
