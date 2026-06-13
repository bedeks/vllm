# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare normal rollout outputs with experimental artifact export.

This benchmark intentionally forces the classic V1 GPU model runner because the
RFC spike hooks that runner first.
"""

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--keep-logprobs-in-output",
        action="store_true",
        help="Export artifacts but keep normal RequestOutput logprobs too.",
    )
    return parser.parse_args()


def make_prompts(num_prompts: int) -> list[str]:
    base_prompts = [
        "Write a short numbered plan for testing a distributed inference change.",
        "Explain why moving bulky rollout artifacts off the control path helps RL.",
        "Summarize the tradeoffs of file backed artifact export for a prototype.",
        "Give three debugging checks for generated token log probabilities.",
    ]
    return [
        f"{base_prompts[i % len(base_prompts)]}\nExample id: {i}\nAnswer:"
        for i in range(num_prompts)
    ]


def logprobs_payload(logprobs: Any) -> Any:
    if logprobs is None:
        return None
    if hasattr(logprobs, "start_indices"):
        return {
            "start_indices": list(logprobs.start_indices),
            "end_indices": list(logprobs.end_indices),
            "token_ids": list(logprobs.token_ids),
            "logprobs": list(logprobs.logprobs),
            "ranks": list(logprobs.ranks),
            "decoded_tokens": list(logprobs.decoded_tokens),
        }

    payload = []
    for position_logprobs in logprobs:
        if position_logprobs is None:
            payload.append(None)
            continue
        payload.append(
            {
                str(token_id): {
                    "logprob": float(logprob.logprob),
                    "rank": int(logprob.rank),
                    "decoded_token": logprob.decoded_token,
                }
                for token_id, logprob in position_logprobs.items()
            }
        )
    return payload


def output_payload_size(outputs: Any) -> int:
    payload = []
    for output in outputs:
        payload.append(
            {
                "request_id": output.request_id,
                "prompt_token_ids": output.prompt_token_ids,
                "prompt_logprobs": logprobs_payload(output.prompt_logprobs),
                "outputs": [
                    {
                        "index": completion.index,
                        "token_ids": list(completion.token_ids),
                        "logprobs": logprobs_payload(completion.logprobs),
                        "cumulative_logprob": completion.cumulative_logprob,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    }
                    for completion in output.outputs
                ],
                "finished": output.finished,
                "artifact_transfer_params": output.artifact_transfer_params,
            }
        )
    return len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())


def sampled_logprobs(output: Any) -> list[float]:
    completion = output.outputs[0]
    if completion.logprobs is None:
        return []
    values: list[float] = []
    for token_id, position_logprobs in zip(completion.token_ids, completion.logprobs):
        logprob = position_logprobs.get(token_id)
        if logprob is None:
            logprob = next(iter(position_logprobs.values()))
        values.append(float(logprob.logprob))
    return values


def load_artifact_chunks(output: Any) -> tuple[list[int], list[float], int]:
    params = output.artifact_transfer_params
    if not params:
        raise AssertionError("artifact_transfer_params missing from output")

    token_ids: list[int] = []
    logprobs: list[float] = []
    artifact_bytes = 0
    for chunk in params["chunks"]:
        path = Path(chunk["path"])
        artifact_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            assert metadata["format_version"] == 1
            if "token_ids" in data:
                token_ids.extend(data["token_ids"].astype(np.int64).tolist())
            if "logprobs" in data:
                logprobs.extend(data["logprobs"][:, 0].astype(np.float64).tolist())
    return token_ids, logprobs, artifact_bytes


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
    except Exception:
        return


def main() -> None:
    args = parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    artifact_parent = Path(
        args.artifact_dir or tempfile.mkdtemp(prefix="vllm-artifact-bench-")
    ).expanduser()
    artifact_dir = artifact_parent / f"run-{uuid.uuid4().hex}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": False,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    prompts = make_prompts(args.num_prompts)

    base_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
    )
    artifact_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
        extra_args={
            "artifact_transfer": {
                "enabled": True,
                "backend": "file",
                "path": str(artifact_dir),
                "fields": ["token_ids", "logprobs", "prompt_logprobs"],
                "exclude_from_request_output": not args.keep_logprobs_in_output,
            }
        },
    )

    synchronize_cuda()
    start = time.perf_counter()
    baseline_outputs = llm.generate(prompts, base_params, use_tqdm=False)
    synchronize_cuda()
    baseline_s = time.perf_counter() - start

    synchronize_cuda()
    start = time.perf_counter()
    artifact_outputs = llm.generate(prompts, artifact_params, use_tqdm=False)
    synchronize_cuda()
    artifact_s = time.perf_counter() - start

    artifact_bytes = 0
    for i, (baseline, artifact) in enumerate(zip(baseline_outputs, artifact_outputs)):
        baseline_token_ids = list(baseline.outputs[0].token_ids)
        artifact_token_ids, artifact_sample_logprobs, output_artifact_bytes = (
            load_artifact_chunks(artifact)
        )
        artifact_bytes += output_artifact_bytes
        if baseline_token_ids != artifact_token_ids:
            raise AssertionError(f"token mismatch for prompt {i}")

        baseline_sample_logprobs = sampled_logprobs(baseline)
        if baseline_sample_logprobs and not np.allclose(
            np.asarray(baseline_sample_logprobs),
            np.asarray(artifact_sample_logprobs),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(f"sample logprob mismatch for prompt {i}")

    baseline_output_bytes = output_payload_size(baseline_outputs)
    artifact_output_bytes = output_payload_size(artifact_outputs)
    result = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "prompt_logprobs": args.prompt_logprobs,
        "baseline_seconds": baseline_s,
        "artifact_seconds": artifact_s,
        "baseline_request_output_payload_bytes": baseline_output_bytes,
        "artifact_request_output_payload_bytes": artifact_output_bytes,
        "artifact_file_bytes": artifact_bytes,
        "artifact_dir": str(artifact_dir),
        "verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
