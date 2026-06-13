# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np

from vllm.distributed.artifact_transfer import (
    FileArtifactBackend,
    get_artifact_transfer_config,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor


def test_artifact_transfer_config_from_extra_args(tmp_path):
    config = get_artifact_transfer_config(
        {
            "artifact_transfer": {
                "enabled": True,
                "path": str(tmp_path),
                "fields": ["token_ids", "logprobs"],
                "exclude_from_request_output": False,
            }
        }
    )

    assert config is not None
    assert config.backend == "file"
    assert config.path == str(tmp_path)
    assert config.fields == ("token_ids", "logprobs")
    assert not config.exclude_from_request_output


def test_file_artifact_backend_round_trip(tmp_path):
    backend = FileArtifactBackend(str(tmp_path))
    handle = backend.put_artifact(
        request_id="req-1",
        step_index=3,
        arrays={
            "token_ids": np.asarray([1, 2, 3], dtype=np.int64),
            "logprobs": np.asarray([[-0.1], [-0.2], [-0.3]], dtype=np.float32),
        },
        metadata={"format_version": 1},
    )

    handle_dict = handle.to_dict()
    assert handle_dict["backend"] == "file"
    assert handle_dict["fields"] == ["logprobs", "token_ids"]

    with np.load(handle.path, allow_pickle=False) as data:
        assert data["token_ids"].tolist() == [1, 2, 3]
        assert data["logprobs"].shape == (3, 1)
        metadata = json.loads(str(data["metadata_json"].item()))
        assert metadata["request_id"] == "req-1"
        assert metadata["step_index"] == 3


def test_request_output_carries_artifact_transfer_params():
    params = {"version": 1, "chunks": [{"path": "/tmp/chunk.npz"}]}
    output = RequestOutput(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=" world",
                token_ids=[2],
                cumulative_logprob=None,
                logprobs=None,
            )
        ],
        finished=True,
        artifact_transfer_params=params,
    )

    assert output.artifact_transfer_params == params


def test_output_processor_copies_artifact_transfer_params():
    params = {"version": 1, "chunks": [{"path": "/tmp/chunk.npz"}]}
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    request.external_req_id = "external-req-1"
    processor.add_request(request, prompt=None)

    processed = processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="req-1",
                new_token_ids=[2],
                finish_reason=FinishReason.LENGTH,
                artifact_transfer_params=params,
            )
        ]
    )

    assert len(processed.request_outputs) == 1
    assert processed.request_outputs[0].artifact_transfer_params == params
