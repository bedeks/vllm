#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/workspace/.venv/bin/python}"

source /workspace/.venv/bin/activate
unset LD_LIBRARY_PATH

export VLLM_HOST_IP="${VLLM_HOST_IP:-$(ip -4 addr show eth0 | awk '/inet /{print $2}' | cut -d/ -f1)}"
export VLLM_SERVER_DEV_MODE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVER_GPU="${SERVER_GPU:-1}"
TRAINER_GPU="${TRAINER_GPU:-0}"
MUTATE_FRACTION="${MUTATE_FRACTION:-0.003}"
MUTATE_SEED="${MUTATE_SEED:-1234}"
MAX_TOKENS="${MAX_TOKENS:-1}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
DENSE_PORT="${DENSE_PORT:-18000}"
SPARSE_PORT="${SPARSE_PORT:-18001}"

LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

SERVER_PID=""
MON_PID=""

cleanup() {
  set +e
  if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
  if [[ -n "${MON_PID}" ]]; then kill "${MON_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

start_server() {
  local port="$1"
  local logfile="$2"
  CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
  vllm serve "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --dtype bfloat16 \
    --enforce-eager \
    --weight-transfer-config '{"backend":"nccl"}' \
    > "${logfile}" 2>&1 &
  SERVER_PID=$!
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    sleep 3
  fi
}

wait_ready() {
  local port="$1"
  local logfile="$2"
  local ready=0
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${port}/get_world_size" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done

  if [[ "${ready}" != "1" ]]; then
    echo "Server on port ${port} did not become ready. Last log lines:"
    tail -n 120 "${logfile}" || true
    exit 1
  fi
}

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader \
  -l 1 > "${LOG_DIR}/gpu_samples.csv" &
MON_PID=$!

start_server "${DENSE_PORT}" "${LOG_DIR}/server_dense_patched.log"
wait_ready "${DENSE_PORT}" "${LOG_DIR}/server_dense_patched.log"

CUDA_VISIBLE_DEVICES="${TRAINER_GPU}" \
"${VENV_PYTHON}" "${REPO_ROOT}/tools/repro/repro_issue_39451_generation_equivalence_harness.py" \
  --phase dense_patched \
  --base-url "http://127.0.0.1:${DENSE_PORT}" \
  --model-name "${MODEL_NAME}" \
  --outputs-json "${LOG_DIR}/dense_generation.json" \
  --mutate-fraction "${MUTATE_FRACTION}" \
  --mutate-seed "${MUTATE_SEED}" \
  --max-tokens "${MAX_TOKENS}" \
  > "${LOG_DIR}/trainer_dense_patched.log" 2>&1

stop_server

start_server "${SPARSE_PORT}" "${LOG_DIR}/server_sparse_patched.log"
wait_ready "${SPARSE_PORT}" "${LOG_DIR}/server_sparse_patched.log"

CUDA_VISIBLE_DEVICES="${TRAINER_GPU}" \
"${VENV_PYTHON}" "${REPO_ROOT}/tools/repro/repro_issue_39451_generation_equivalence_harness.py" \
  --phase sparse_patched \
  --base-url "http://127.0.0.1:${SPARSE_PORT}" \
  --model-name "${MODEL_NAME}" \
  --outputs-json "${LOG_DIR}/sparse_generation.json" \
  --mutate-fraction "${MUTATE_FRACTION}" \
  --mutate-seed "${MUTATE_SEED}" \
  --max-tokens "${MAX_TOKENS}" \
  > "${LOG_DIR}/trainer_sparse_patched.log" 2>&1

"${VENV_PYTHON}" - <<PY
import json
from pathlib import Path

def diff_keys(lhs: dict[str, str], rhs: dict[str, str]) -> list[str]:
    all_keys = sorted(set(lhs) | set(rhs))
    return [key for key in all_keys if lhs.get(key) != rhs.get(key)]

log_dir = Path("${LOG_DIR}")
dense = json.loads((log_dir / "dense_generation.json").read_text())
sparse = json.loads((log_dir / "sparse_generation.json").read_text())

baseline_equal = dense["baseline_outputs"] == sparse["baseline_outputs"]
trainer_patch_equal = dense["trainer_patch_digest"] == sparse["trainer_patch_digest"]
digest_map_equal = dense["digests"] == sparse["digests"]
after_equal = dense["after_outputs"] == sparse["after_outputs"]
differing_params = diff_keys(dense["digests"], sparse["digests"])

print("baseline_equal =", baseline_equal)
print("trainer_patch_equal =", trainer_patch_equal)
print("digest_map_equal =", digest_map_equal)
print("after_equal =", after_equal)
print("differing_param_count =", len(differing_params))
print("differing_params_first10 =", differing_params[:10])
print("dense_after_outputs =", dense["after_outputs"])
print("sparse_after_outputs =", sparse["after_outputs"])

if not baseline_equal:
    raise SystemExit("Fresh-server baseline outputs differed")
if not trainer_patch_equal:
    raise SystemExit("Dense and sparse runs built different trainer patches")
if not digest_map_equal:
    raise SystemExit("Dense and sparse runs produced different digest maps")
if not after_equal:
    raise SystemExit("Dense and sparse 1-token outputs differed despite equal digest maps")
PY

tail -n 80 "${LOG_DIR}/trainer_dense_patched.log" || true
echo
tail -n 80 "${LOG_DIR}/trainer_sparse_patched.log" || true
echo
echo "Logs:"
echo "  ${LOG_DIR}/server_dense_patched.log"
echo "  ${LOG_DIR}/server_sparse_patched.log"
echo "  ${LOG_DIR}/trainer_dense_patched.log"
echo "  ${LOG_DIR}/trainer_sparse_patched.log"
echo "  ${LOG_DIR}/dense_generation.json"
echo "  ${LOG_DIR}/sparse_generation.json"
echo "  ${LOG_DIR}/gpu_samples.csv"
