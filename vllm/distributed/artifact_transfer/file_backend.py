# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vllm.distributed.artifact_transfer.protocol import ArtifactHandle


class FileArtifactBackend:
    """Local file backend for rollout artifact transfer experiments."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_artifact(
        self,
        *,
        request_id: str,
        step_index: int,
        arrays: dict[str, npt.ArrayLike],
        metadata: dict[str, Any],
    ) -> ArtifactHandle:
        artifact_id = uuid.uuid4().hex
        path = self.root / f"{artifact_id}.npz"
        fields = sorted(arrays)
        metadata = {
            **metadata,
            "request_id": request_id,
            "step_index": step_index,
            "created_at_unix_s": time.time(),
        }

        np_arrays = {name: np.asarray(value) for name, value in arrays.items()}
        np_arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez(path, **np_arrays)

        return ArtifactHandle(
            backend="file",
            artifact_id=artifact_id,
            path=str(path),
            format="npz",
            fields=fields,
            metadata=metadata,
        )
