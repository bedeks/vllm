# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.config as config
import vllm.envs as envs
from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

pytestmark = pytest.mark.skip_global_cleanup

BATCH_INVARIANT_ENV_KEYS = (
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
    "VLLM_USE_AOT_COMPILE",
)


class DummyAttentionImpl(AttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass


class NonBatchInvariantBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "NON_BATCH_INVARIANT"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return DummyAttentionImpl

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class BatchInvariantBackend(NonBatchInvariantBackend):
    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True


def test_validate_configuration_rejects_batch_invariant_unsupported_backend():
    invalid_reasons = NonBatchInvariantBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
        is_batch_invariant=True,
    )

    assert invalid_reasons == ["batch invariance not supported"]


def test_validate_configuration_accepts_batch_invariant_supported_backend():
    invalid_reasons = BatchInvariantBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
        is_batch_invariant=True,
    )

    assert invalid_reasons == []


def test_get_attn_backend_threads_batch_invariance(monkeypatch):
    _cached_get_attn_backend.cache_clear()
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(
        config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            cache_config=None,
            speculative_config=None,
            attention_config=SimpleNamespace(backend=None),
        ),
    )
    captured = {}

    def fake_get_attn_backend_cls(
        backend,
        attn_selector_config,
        num_heads=None,
    ):
        captured["config"] = attn_selector_config
        return AttentionBackendEnum.TRITON_ATTN.get_path()

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        fake_get_attn_backend_cls,
    )

    backend = get_attn_backend(64, torch.float16, None)

    assert captured["config"].is_batch_invariant is True
    assert backend.get_name() == "TRITON_ATTN"


def test_override_envs_for_invariance_allows_auto_selected_backend():
    original_env = {key: os.environ.get(key) for key in BATCH_INVARIANT_ENV_KEYS}

    try:
        override_envs_for_invariance(None)

        assert os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
        assert os.environ["NCCL_LAUNCH_MODE"] == "GROUP"
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_override_envs_for_invariance_uses_backend_capability(monkeypatch):
    original_env = {key: os.environ.get(key) for key in BATCH_INVARIANT_ENV_KEYS}
    original_get_class = AttentionBackendEnum.get_class

    def fake_get_class(self):
        if self == AttentionBackendEnum.FLASHINFER:
            return BatchInvariantBackend
        return original_get_class(self)

    monkeypatch.setattr(AttentionBackendEnum, "get_class", fake_get_class)

    try:
        override_envs_for_invariance(AttentionBackendEnum.FLASHINFER)

        assert os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_override_envs_for_invariance_rejects_unsupported_backend(monkeypatch):
    monkeypatch.setattr(
        AttentionBackendEnum,
        "get_class",
        lambda self: NonBatchInvariantBackend,
    )

    with pytest.raises(RuntimeError, match="requires an attention backend"):
        override_envs_for_invariance(AttentionBackendEnum.FLASHINFER)
