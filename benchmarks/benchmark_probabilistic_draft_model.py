#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
from dataclasses import asdict, dataclass

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def build_prompts(num_prompts: int) -> list[str]:
    base_prompts = [
        "Write one short sentence about Mars.",
        "Summarize why the sky appears blue in one sentence.",
        "Give a concise explanation of quicksort.",
        "Name three uses of linear algebra in machine learning.",
        "Write a short haiku about debugging distributed systems.",
        "Answer: what is the capital of Japan?",
        "Explain speculative decoding in one paragraph.",
        "List two risks of stale caches in distributed systems.",
    ]
    prompts = []
    for idx in range(num_prompts):
        template = base_prompts[idx % len(base_prompts)]
        prompts.append(f"{template} Prompt variant #{idx}.")
    return prompts


def compute_acceptance_rate(metrics) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    num_draft_tokens = name2metric["vllm:spec_decode_num_draft_tokens"].value
    num_accepted_tokens = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    return (
        float("nan")
        if num_draft_tokens == 0
        else num_accepted_tokens / num_draft_tokens
    )


def compute_acceptance_len(metrics) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    num_drafts = name2metric["vllm:spec_decode_num_drafts"].value
    num_accepted_tokens = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    return 1.0 if num_drafts == 0 else 1.0 + (num_accepted_tokens / num_drafts)


@dataclass
class RunResult:
    mode: str
    acceptance_rate: float
    acceptance_len: float


def run_case(args: argparse.Namespace, disable_draft_probs: bool) -> RunResult:
    original_method = GPUModelRunner._get_spec_decode_draft_probs
    if disable_draft_probs:
        GPUModelRunner._get_spec_decode_draft_probs = lambda self, metadata: None

    llm = None
    try:
        llm = LLM(
            model=args.target_model,
            speculative_config={
                "method": "draft_model",
                "model": args.draft_model,
                "num_speculative_tokens": args.num_speculative_tokens,
                "rejection_sample_method": "probabilistic",
                "enforce_eager": args.enforce_eager,
            },
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            disable_log_stats=False,
        )
        llm.generate(
            build_prompts(args.num_prompts),
            SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ignore_eos=False,
            ),
        )
        metrics = llm.get_metrics()
        return RunResult(
            mode="baseline" if disable_draft_probs else "fixed",
            acceptance_rate=compute_acceptance_rate(metrics),
            acceptance_len=compute_acceptance_len(metrics),
        )
    finally:
        GPUModelRunner._get_spec_decode_draft_probs = original_method
        if llm is not None:
            del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark probabilistic draft-model speculative decoding on the "
            "legacy V1 GPU model runner and compare the fixed path against the "
            "pre-fix behavior."
        )
    )
    parser.add_argument(
        "--target-model",
        default="Qwen/Qwen3-1.7B",
        help="Target model name or path.",
    )
    parser.add_argument(
        "--draft-model",
        default="Qwen/Qwen3-0.6B",
        help="Draft model name or path.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Number of draft tokens to propose per step.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=64,
        help="Number of prompts to generate for each run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum output tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling temperature. Use a non-zero value to exercise "
            "probabilistic rejection."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model length for both target and draft model.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help="Maximum number of concurrent sequences.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization passed to the engine.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for easier debugging.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "fixed", "both"),
        default="both",
        help="Which implementation to run.",
    )
    parser.add_argument(
        "--expect-improvement",
        action="store_true",
        help="Exit non-zero if the fixed path does not improve acceptance metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")

    requested_modes = (
        ("baseline", True),
        ("fixed", False),
    ) if args.mode == "both" else ((args.mode, args.mode == "baseline"),)

    results = [
        run_case(args, disable_draft_probs)
        for _, disable_draft_probs in requested_modes
    ]

    print("Probabilistic draft-model benchmark results:")
    for result in results:
        print(f"  {result.mode}: {asdict(result)}")

    if len(results) == 2:
        baseline, fixed = results
        delta_rate = fixed.acceptance_rate - baseline.acceptance_rate
        delta_len = fixed.acceptance_len - baseline.acceptance_len
        print(
            "  delta:",
            {
                "acceptance_rate": delta_rate,
                "acceptance_len": delta_len,
            },
        )
        if args.expect_improvement and (
            fixed.acceptance_rate <= baseline.acceptance_rate
            or fixed.acceptance_len <= baseline.acceptance_len
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
