#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM

from repro_issue_39451_weight_digest_harness import (
    apply_patch_to_trainer,
    build_deterministic_patch,
    get_server_weight_digest_map,
    init_trainer_group,
    run_sync_phase,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

SESSION = requests.Session()
SESSION.trust_env = False


def generate_one_token_outputs(
    base_url: str,
    model_name: str,
    max_tokens: int,
) -> list[str]:
    outputs = []
    for prompt in PROMPTS:
        resp = SESSION.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        outputs.append(data["choices"][0]["text"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["dense_patched", "sparse_patched"],
        required=True,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--outputs-json", required=True)
    parser.add_argument("--mutate-fraction", type=float, default=0.003)
    parser.add_argument("--mutate-seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.outputs_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(0)
    print(f"Loading training model on cuda:0: {args.model_name}")
    train_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
    ).to("cuda:0")

    param_list = list(train_model.named_parameters())
    param_map = dict(param_list)
    total_numel = sum(param.numel() for _, param in param_list)

    baseline_outputs = generate_one_token_outputs(
        args.base_url,
        args.model_name,
        args.max_tokens,
    )

    model_update_group = init_trainer_group(args.base_url)

    embed_name = "model.embed_tokens.weight"
    if embed_name not in param_map:
        raise ValueError(f"Required param missing: {embed_name}")

    target_changed_numel = max(1, int(total_numel * args.mutate_fraction))
    patch, patch_digest = build_deterministic_patch(
        param_name=embed_name,
        param=param_map[embed_name],
        target_changed_numel=target_changed_numel,
        seed=args.mutate_seed,
    )
    apply_patch_to_trainer(param_map[embed_name], patch)

    if args.phase == "dense_patched":
        metrics = run_sync_phase(
            send_fn=lambda: NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(param_list),
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=model_update_group,
                    packed=True,
                ),
            ),
            update_kwargs={
                "base_url": args.base_url,
                "names": [name for name, _ in param_list],
                "dtype_names": [str(param.dtype).split(".")[-1] for _, param in param_list],
                "shapes": [list(param.shape) for _, param in param_list],
                "packed": True,
                "is_checkpoint_format": True,
                "update_kind": "dense",
            },
        )
    else:
        metrics = run_sync_phase(
            send_fn=lambda: NCCLWeightTransferEngine.trainer_send_sparse_weights(
                iterator=iter([patch]),
                trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group),
            ),
            update_kwargs={
                "base_url": args.base_url,
                "names": [patch.name],
                "dtype_names": [str(patch.values.dtype).split(".")[-1]],
                "shapes": [list(param_map[patch.name].shape)],
                "packed": False,
                "is_checkpoint_format": False,
                "update_kind": "sparse_flat",
                "nnz_list": [patch.indices.numel()],
                "indices_dtype_name": "int32",
            },
        )

    digest_map = get_server_weight_digest_map(args.base_url)
    after_outputs = generate_one_token_outputs(
        args.base_url,
        args.model_name,
        args.max_tokens,
    )

    result = {
        "phase": args.phase,
        "model_name": args.model_name,
        "trainer_patch_digest": patch_digest,
        "patched_param": patch.name,
        "changed_numel": patch.indices.numel(),
        "baseline_outputs": baseline_outputs,
        "after_outputs": after_outputs,
        "digests": digest_map,
        "num_params": len(digest_map),
        "max_tokens": args.max_tokens,
        "update_request_ms": metrics["update_request_ms"],
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
