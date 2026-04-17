#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.base import SparseWeightPatch
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

SESSION = requests.Session()
SESSION.trust_env = False


def get_world_size(base_url: str) -> int:
    resp = SESSION.get(f"{base_url}/get_world_size", timeout=10)
    resp.raise_for_status()
    return resp.json()["world_size"]


def init_weight_transfer_engine(
    base_url: str,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
) -> None:
    resp = SESSION.post(
        f"{base_url}/init_weight_transfer_engine",
        json={
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
            }
        },
        timeout=300,
    )
    resp.raise_for_status()


def update_weights(
    base_url: str,
    *,
    names: list[str],
    dtype_names: list[str],
    shapes: list[list[int]],
    packed: bool,
    is_checkpoint_format: bool,
    update_kind: str,
    nnz_list: list[int] | None = None,
    indices_dtype_name: str | None = None,
) -> None:
    update_info = {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
        "packed": packed,
        "is_checkpoint_format": is_checkpoint_format,
        "update_kind": update_kind,
    }
    if nnz_list is not None:
        update_info["nnz_list"] = nnz_list
    if indices_dtype_name is not None:
        update_info["indices_dtype_name"] = indices_dtype_name

    resp = SESSION.post(
        f"{base_url}/update_weights",
        json={"update_info": update_info},
        timeout=600,
    )
    resp.raise_for_status()


def pause_generation(base_url: str) -> None:
    resp = SESSION.post(f"{base_url}/pause", timeout=120)
    resp.raise_for_status()


def resume_generation(base_url: str) -> None:
    resp = SESSION.post(f"{base_url}/resume", timeout=120)
    resp.raise_for_status()


def timed_update_weights(out: dict, **kwargs) -> None:
    t0 = time.perf_counter()
    try:
        update_weights(**kwargs)
    except Exception as exc:
        out["error"] = repr(exc)
    finally:
        out["update_request_ms"] = round((time.perf_counter() - t0) * 1000, 2)


def run_sync_phase(send_fn, update_kwargs: dict) -> dict:
    pause_generation(update_kwargs["base_url"])

    phase_metrics: dict[str, object] = {}
    update_thread = threading.Thread(
        target=timed_update_weights,
        kwargs={"out": phase_metrics, **update_kwargs},
        daemon=True,
    )
    update_thread.start()

    send_fn()
    torch.cuda.synchronize()

    update_thread.join()
    if phase_metrics.get("error") is not None:
        raise RuntimeError(f"update_weights failed: {phase_metrics['error']}")

    resume_generation(update_kwargs["base_url"])
    return phase_metrics


def build_deterministic_patch(
    *,
    param_name: str,
    param: torch.nn.Parameter,
    target_changed_numel: int,
    seed: int,
) -> tuple[SparseWeightPatch, str, int, int]:
    if param.ndim != 2:
        raise ValueError(f"{param_name} must be a 2D embedding matrix")

    target_changed_numel = min(target_changed_numel, param.numel())
    vocab_size, hidden_size = param.shape
    rows_needed = max(1, min(vocab_size, math.ceil(target_changed_numel / hidden_size)))

    if rows_needed >= vocab_size:
        start_row = 0
    else:
        start_row = seed % (vocab_size - rows_needed + 1)

    row_tensor = torch.arange(
        start_row,
        start_row + rows_needed,
        device=param.device,
        dtype=torch.long,
    )
    col_tensor = torch.arange(hidden_size, device=param.device, dtype=torch.long)
    flat_indices = (row_tensor[:, None] * hidden_size + col_tensor[None, :]).reshape(-1)
    flat_indices = flat_indices[:target_changed_numel]

    flat_param = param.data.view(-1)
    original_values = flat_param.index_select(0, flat_indices)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    delta_cpu = torch.randn(flat_indices.numel(), generator=gen, dtype=torch.float32)
    delta = (delta_cpu * 1e-3).to(device=param.device, dtype=param.dtype)
    new_values = original_values + delta

    patch = SparseWeightPatch(
        name=param_name,
        indices=flat_indices.to(torch.int32).contiguous(),
        values=new_values.contiguous(),
    )

    digest = hashlib.sha256(
        patch.indices.cpu().numpy().tobytes()
        + patch.values.float().cpu().numpy().tobytes()
    ).hexdigest()

    return patch, digest, int(flat_indices[0].item()), flat_indices.numel()


def apply_patch_to_trainer(param: torch.nn.Parameter, patch: SparseWeightPatch) -> None:
    flat_param = param.data.view(-1)
    flat_param.index_copy_(0, patch.indices.to(torch.long), patch.values)
    torch.cuda.synchronize()


def get_server_weight_slice_digest(
    base_url: str,
    *,
    name: str,
    flat_start: int,
    flat_length: int,
) -> dict:
    resp = SESSION.post(
        f"{base_url}/debug/get_weight_slice_digest",
        json={
            "name": name,
            "flat_start": flat_start,
            "flat_length": flat_length,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dense_reference", "sparse_candidate"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--outputs-json", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mutate-fraction", type=float, default=0.003)
    parser.add_argument("--mutate-seed", type=int, default=1234)
    args = parser.parse_args()

    torch.cuda.set_device(0)

    print(f"Loading training model on cuda:0: {args.model_name}")
    train_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
    ).to("cuda:0")

    param_list = list(train_model.named_parameters())
    param_map = dict(param_list)
    total_numel = sum(p.numel() for _, p in param_list)

    embed_name = "model.embed_tokens.weight"
    if embed_name not in param_map:
        raise ValueError(f"Required param missing: {embed_name}")
    embed_param = param_map[embed_name]

    inference_world_size = get_world_size(args.base_url)
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()

    print(f"Initializing weight transfer: master={master_address}:{master_port}, world_size={world_size}")

    init_thread = threading.Thread(
        target=init_weight_transfer_engine,
        args=(args.base_url, master_address, master_port, 1, world_size),
        daemon=True,
    )
    init_thread.start()
    model_update_group = NCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
        }
    )
    init_thread.join()

    target_changed_numel = max(1, int(total_numel * args.mutate_fraction))
    patch, patch_digest, flat_start, flat_length = build_deterministic_patch(
        param_name=embed_name,
        param=embed_param,
        target_changed_numel=target_changed_numel,
        seed=args.mutate_seed,
    )

    apply_patch_to_trainer(embed_param, patch)

    if args.phase == "dense_reference":
        metrics = run_sync_phase(
            send_fn=lambda: NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(param_list),
                trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group, packed=True),
            ),
            update_kwargs={
                "base_url": args.base_url,
                "names": [name for name, _ in param_list],
                "dtype_names": [str(p.dtype).split(".")[-1] for _, p in param_list],
                "shapes": [list(p.shape) for _, p in param_list],
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
                "shapes": [list(embed_param.shape)],
                "packed": False,
                "is_checkpoint_format": False,
                "update_kind": "sparse_flat",
                "nnz_list": [patch.indices.numel()],
                "indices_dtype_name": "int32",
            },
        )

    server_slice_digest = get_server_weight_slice_digest(
        args.base_url,
        name=patch.name,
        flat_start=flat_start,
        flat_length=flat_length,
    )

    out = {
        "phase": args.phase,
        "run_id": args.run_id,
        "model_name": args.model_name,
        "trainer_patch_digest": patch_digest,
        "server_slice_digest": server_slice_digest["digest"],
        "patched_param": patch.name,
        "changed_numel": patch.indices.numel(),
        "changed_fraction": patch.indices.numel() / total_numel,
        "flat_start": flat_start,
        "flat_length": flat_length,
        "update_request_ms": metrics["update_request_ms"],
    }

    output_path = Path(args.outputs_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
