# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass, field
from typing import Any

ARTIFACT_TRANSFER_EXTRA_ARGS_KEY = "artifact_transfer"
DEFAULT_ARTIFACT_FIELDS = ("token_ids", "logprobs", "prompt_logprobs")


@dataclass(frozen=True)
class ArtifactTransferRequestConfig:
    """Per-request rollout artifact export options.

    This is intentionally request-scoped for the RFC spike: offline callers can
    opt individual rollout requests into artifact export through
    ``SamplingParams.extra_args`` without changing server-wide configuration.
    """

    backend: str = "file"
    path: str = "/tmp/vllm-rollout-artifacts"
    fields: tuple[str, ...] = DEFAULT_ARTIFACT_FIELDS
    exclude_from_request_output: bool = True


@dataclass
class ArtifactHandle:
    """Lightweight reference to an exported rollout artifact chunk."""

    backend: str
    artifact_id: str
    path: str
    format: str
    fields: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "format": self.format,
            "fields": self.fields,
            "metadata": self.metadata,
        }


@dataclass
class ArtifactConnectorOutput:
    """Worker-to-scheduler artifact export result for one engine step."""

    artifact_handles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.artifact_handles


def get_artifact_transfer_config(
    extra_args: dict[str, Any] | None,
) -> ArtifactTransferRequestConfig | None:
    if not extra_args:
        return None

    raw_config = extra_args.get(ARTIFACT_TRANSFER_EXTRA_ARGS_KEY)
    if raw_config is None:
        # Short alias for quick local experiments.
        raw_config = extra_args.get("rollout_artifact")
    if raw_config is None or raw_config is False:
        return None
    if raw_config is True:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError(
            "artifact_transfer must be a bool or a dictionary in "
            "SamplingParams.extra_args"
        )

    enabled = raw_config.get("enabled", True)
    if not enabled:
        return None

    backend = str(raw_config.get("backend", "file"))
    if backend != "file":
        raise ValueError(
            "Only artifact_transfer backend='file' is implemented in this spike"
        )

    path = str(
        raw_config.get(
            "path",
            os.environ.get("VLLM_ARTIFACT_TRANSFER_DIR", "/tmp/vllm-rollout-artifacts"),
        )
    )

    raw_fields: Any = raw_config.get("fields", DEFAULT_ARTIFACT_FIELDS)
    fields: tuple[str, ...]
    if isinstance(raw_fields, str):
        fields = (raw_fields,)
    else:
        fields = tuple(str(field_name) for field_name in raw_fields)

    exclude_from_request_output = bool(
        raw_config.get(
            "exclude_from_request_output",
            raw_config.get("omit_logprobs_from_output", True),
        )
    )

    return ArtifactTransferRequestConfig(
        backend=backend,
        path=path,
        fields=fields,
        exclude_from_request_output=exclude_from_request_output,
    )
