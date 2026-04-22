#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN_DIR="${VENV_BIN_DIR:-${REPO_ROOT}/.venv/bin}"
VENV_PYTHON="${VENV_PYTHON:-${VENV_BIN_DIR}/python}"

source "${VENV_BIN_DIR}/activate"
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
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
BASELINE_PORT="${BASELINE_PORT:-18000}"
DENSE_NOOP_PORT="${DENSE_NOOP_PORT:-18001}"
DENSE_PATCHED_PORT="${DENSE_PATCHED_PORT:-18002}"
SPARSE_PATCHED_PORT="${SPARSE_PATCHED_PORT:-18003}"

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

run_phase() {
  local phase="$1"
  local port="$2"
  local server_log="$3"
  local trainer_log="$4"
  local output_json="$5"

  start_server "${port}" "${server_log}"
  wait_ready "${port}" "${server_log}"

  if [[ "${phase}" == "baseline_fresh" ]]; then
    "${VENV_PYTHON}" "${REPO_ROOT}/tools/repro/repro_issue_39451_weight_digest_harness.py" \
      --phase "${phase}" \
      --base-url "http://127.0.0.1:${port}" \
      --model-name "${MODEL_NAME}" \
      --outputs-json "${output_json}" \
      > "${trainer_log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${TRAINER_GPU}" \
    "${VENV_PYTHON}" "${REPO_ROOT}/tools/repro/repro_issue_39451_weight_digest_harness.py" \
      --phase "${phase}" \
      --base-url "http://127.0.0.1:${port}" \
      --model-name "${MODEL_NAME}" \
      --outputs-json "${output_json}" \
      --mutate-fraction "${MUTATE_FRACTION}" \
      --mutate-seed "${MUTATE_SEED}" \
      > "${trainer_log}" 2>&1
  fi

  stop_server
}

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader \
  -l 1 > "${LOG_DIR}/gpu_samples.csv" &
MON_PID=$!

run_phase \
  baseline_fresh \
  "${BASELINE_PORT}" \
  "${LOG_DIR}/server_baseline_fresh.log" \
  "${LOG_DIR}/trainer_baseline_fresh.log" \
  "${LOG_DIR}/baseline_fresh_digests.json"

run_phase \
  dense_noop \
  "${DENSE_NOOP_PORT}" \
  "${LOG_DIR}/server_dense_noop.log" \
  "${LOG_DIR}/trainer_dense_noop.log" \
  "${LOG_DIR}/dense_noop_digests.json"

run_phase \
  dense_patched \
  "${DENSE_PATCHED_PORT}" \
  "${LOG_DIR}/server_dense_patched.log" \
  "${LOG_DIR}/trainer_dense_patched.log" \
  "${LOG_DIR}/dense_patched_digests.json"

run_phase \
  sparse_patched \
  "${SPARSE_PATCHED_PORT}" \
  "${LOG_DIR}/server_sparse_patched.log" \
  "${LOG_DIR}/trainer_sparse_patched.log" \
  "${LOG_DIR}/sparse_patched_digests.json"

"${VENV_PYTHON}" - <<PY
import json
from pathlib import Path

def diff_keys(lhs: dict[str, str], rhs: dict[str, str]) -> list[str]:
    all_keys = sorted(set(lhs) | set(rhs))
    return [key for key in all_keys if lhs.get(key) != rhs.get(key)]

log_dir = Path("${LOG_DIR}")
baseline = json.loads((log_dir / "baseline_fresh_digests.json").read_text())
dense_noop = json.loads((log_dir / "dense_noop_digests.json").read_text())
dense_patched = json.loads((log_dir / "dense_patched_digests.json").read_text())
sparse_patched = json.loads((log_dir / "sparse_patched_digests.json").read_text())

baseline_vs_dense_noop = diff_keys(baseline["digests"], dense_noop["digests"])
dense_vs_sparse = diff_keys(dense_patched["digests"], sparse_patched["digests"])
trainer_patch_equal = (
    dense_patched.get("trainer_patch_digest")
    == sparse_patched.get("trainer_patch_digest")
)

print("trainer_patch_equal =", trainer_patch_equal)
print("fresh_equals_dense_noop =", len(baseline_vs_dense_noop) == 0)
print("dense_patched_equals_sparse_patched =", len(dense_vs_sparse) == 0)
print("fresh_vs_dense_noop_diff_count =", len(baseline_vs_dense_noop))
print("dense_vs_sparse_diff_count =", len(dense_vs_sparse))
print("fresh_vs_dense_noop_first10 =", baseline_vs_dense_noop[:10])
print("dense_vs_sparse_first10 =", dense_vs_sparse[:10])

if not trainer_patch_equal:
    raise SystemExit("Dense patched and sparse patched built different trainer patches")
if baseline_vs_dense_noop:
    raise SystemExit("Dense no-op changed the model relative to a fresh server")
if dense_vs_sparse:
    raise SystemExit("Dense patched and sparse patched produced different parameter digest maps")
PY

for log_file in \
  "${LOG_DIR}/trainer_baseline_fresh.log" \
  "${LOG_DIR}/trainer_dense_noop.log" \
  "${LOG_DIR}/trainer_dense_patched.log" \
  "${LOG_DIR}/trainer_sparse_patched.log"
do
  tail -n 80 "${log_file}" || true
  echo
done

echo "Logs:"
echo "  ${LOG_DIR}/server_baseline_fresh.log"
echo "  ${LOG_DIR}/server_dense_noop.log"
echo "  ${LOG_DIR}/server_dense_patched.log"
echo "  ${LOG_DIR}/server_sparse_patched.log"
echo "  ${LOG_DIR}/trainer_baseline_fresh.log"
echo "  ${LOG_DIR}/trainer_dense_noop.log"
echo "  ${LOG_DIR}/trainer_dense_patched.log"
echo "  ${LOG_DIR}/trainer_sparse_patched.log"
echo "  ${LOG_DIR}/baseline_fresh_digests.json"
echo "  ${LOG_DIR}/dense_noop_digests.json"
echo "  ${LOG_DIR}/dense_patched_digests.json"
echo "  ${LOG_DIR}/sparse_patched_digests.json"
echo "  ${LOG_DIR}/gpu_samples.csv"
