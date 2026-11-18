"""Start the local vLLM server for a model of configs/models.yaml, with the paper's settings.

    python scripts/serve_vllm.py qwen3.8-27b --gpus 0,1 --port 8000
    python scripts/serve_vllm.py qwen3.8-27b --gpus 2,3 --port 8001 --dry-run

The settings come from the model's ``serve`` entry: bf16, tensor parallelism
over the two GPUs given, prefix caching off, a 65,536-token context, 90% of
GPU memory and seed 42, at the pinned revision. The paper's runs used vLLM
0.33. Point the experiments at the server with ``--base-url`` or the
``DEALRANK_VLLM_URL`` environment variable when it is not on port 8000.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dealrank.serve import vllm_command, wait_until_healthy
from repro.models import entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="a local model of configs/models.yaml, e.g. qwen3.8-27b")
    parser.add_argument("--gpus", required=True, help="comma-separated GPU ids, one per tensor-parallel rank")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    parser.add_argument("--wait", action="store_true", help="return once the server answers /health")
    args = parser.parse_args(argv)

    spec = entry(args.model)
    if spec["backend"] != "vllm":
        parser.error(f"{args.model} is not served locally")
    serve = spec["serve"]
    gpus = [g for g in args.gpus.split(",") if g.strip()]
    if len(gpus) != serve["tensor_parallel_size"]:
        parser.error(f"{args.model} runs on {serve['tensor_parallel_size']} GPUs, got {len(gpus)}")
    cmd = vllm_command(
        spec["id"],
        port=args.port,
        host=args.host,
        tensor_parallel_size=serve["tensor_parallel_size"],
        dtype=serve["dtype"],
        prefix_caching=serve["prefix_caching"],
        max_model_len=serve["max_model_len"],
        gpu_memory_utilization=serve["gpu_memory_utilization"],
        seed=serve["seed"],
        revision=spec.get("revision"),
    )
    # Long schedules keep a connection idle between rounds; keep it open.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus), "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300"}
    if args.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {shlex.join(cmd)}")
        return 0
    if not args.wait:
        os.execvpe(cmd[0], cmd, env)
    process = subprocess.Popen(cmd, env=env)
    try:
        wait_until_healthy(f"http://{args.host}:{args.port}/v1")
    except TimeoutError:
        process.terminate()
        raise
    print(f"serving {spec['id']} at http://{args.host}:{args.port}/v1 (pid {process.pid})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
