#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
runner="${script_dir}/run_ccrr_v2_target_guarded.sh"
resource_repo_dir="${CCRR_RESOURCE_REPO_DIR:-/home/md0/ly/MSHNet_CCRR}"
queue_root="${CCRR_QUEUE_SAVE_ROOT:-${repo_dir}/repro_runs/formal}"
python_bin="${CCRR_PYTHON:-${resource_repo_dir}/.venv/bin/python}"
poll_seconds="${CCRR_QUEUE_POLL_SECONDS:-30}"
gpu_candidates="${CCRR_QUEUE_GPU_CANDIDATES:-0}"
wait_for_global_lock="${CCRR_QUEUE_WAIT_FOR_GLOBAL_LOCK:-1}"

if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CCRR_QUEUE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ "${wait_for_global_lock}" != "0" && "${wait_for_global_lock}" != "1" ]]; then
  echo "CCRR_QUEUE_WAIT_FOR_GLOBAL_LOCK must be 0 or 1" >&2
  exit 2
fi
if [[ ! -x "${runner}" ]]; then
  echo "TG-SCA runner is unavailable: ${runner}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the one-GPU queue" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for a race-free experiment queue" >&2
  exit 2
fi

lock_dir="${XDG_RUNTIME_DIR:-/tmp}"
queue_lock="${lock_dir}/mshnet-tg1-cross-dataset-queue-${UID}.lock"
training_lock="${lock_dir}/mshnet-ccrr-training-${UID}.lock"
exec 8>"${queue_lock}"
if ! flock -n 8; then
  echo "Another TG1 cross-dataset queue is already active: ${queue_lock}" >&2
  exit 3
fi

# In serial mode, wait for the active formal run and retain its global lock
# across both queued datasets.  In intentional cross-dataset parallel mode,
# the queue relies on its per-GPU lock and each child opts into concurrency.
if [[ "${wait_for_global_lock}" == "1" ]]; then
  exec 9>"${training_lock}"
  while ! flock -n 9; do
    echo "[$(date --iso-8601=seconds)] Another formal run is active; dataset queue is waiting." >&2
    sleep "${poll_seconds}"
  done
else
  echo "[$(date --iso-8601=seconds)] Cross-dataset parallel mode enabled; using a dedicated physical GPU." >&2
fi

gpu_is_idle() {
  local gpu="$1"
  local active_pids
  if ! active_pids="$(
      nvidia-smi \
        --id="${gpu}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"; then
    echo "[$(date --iso-8601=seconds)] Failed to query physical GPU ${gpu}; treating it as busy." >&2
    return 1
  fi
  [[ -z "${active_pids}" ]]
}

select_one_gpu() {
  local gpu
  while true; do
    for gpu in ${gpu_candidates}; do
      if [[ "${gpu}" =~ ^[0-9]+$ ]] && gpu_is_idle "${gpu}"; then
        printf '%s\n' "${gpu}"
        return 0
      fi
    done
    echo "[$(date --iso-8601=seconds)] GPU ${gpu_candidates} busy; queue is waiting." >&2
    sleep "${poll_seconds}"
  done
}

wait_for_selected_gpu() {
  local gpu="$1"
  while ! gpu_is_idle "${gpu}"; do
    echo "[$(date --iso-8601=seconds)] Physical GPU ${gpu} is busy; queue is waiting." >&2
    sleep "${poll_seconds}"
  done
}

run_dataset() {
  local dataset="$1"
  local gpu="$2"
  local save_dir="${queue_root}/TG1_${dataset}_seed42"

  echo "[$(date --iso-8601=seconds)] Starting TG1 ${dataset} on physical GPU ${gpu}." >&2
  CCRR_ALLOW_CONCURRENT_TRAINING=1 \
  CCRR_RESOURCE_REPO_DIR="${resource_repo_dir}" \
  CCRR_DATASET_NAME="${dataset}" \
  CCRR_DATASET_DIR="${resource_repo_dir}/datasets/${dataset}" \
  CCRR_BASELINE_WEIGHT="${resource_repo_dir}/checkpoints/baseline/${dataset}_mshnet_baseline.pth" \
  CCRR_TG_EXPERIMENT=TG1 \
  CCRR_GUARD_ALPHA=0.0 \
  CCRR_TARGET_TAIL_WEIGHT=0.0 \
  CCRR_FP_VALUE_BETA=0.0 \
  CCRR_GPU="${gpu}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CCRR_SAVE_DIR="${save_dir}" \
  CCRR_PYTHON="${python_bin}" \
    "${runner}"
  echo "[$(date --iso-8601=seconds)] Completed TG1 ${dataset} on physical GPU ${gpu}." >&2
}

selected_gpu="$(select_one_gpu)"
gpu_lock="${lock_dir}/mshnet-physical-gpu-${selected_gpu}-${UID}.lock"
exec 7>"${gpu_lock}"
if ! flock -n 7; then
  echo "Physical GPU ${selected_gpu} is reserved by another managed queue." >&2
  exit 3
fi
wait_for_selected_gpu "${selected_gpu}"
echo "[$(date --iso-8601=seconds)] Queue assigned to physical GPU ${selected_gpu}." >&2

# Run the smaller remaining dataset first; IRSTD starts automatically only
# after NUAA finishes successfully on the same selected physical GPU.
run_dataset NUAA-SIRST "${selected_gpu}"
wait_for_selected_gpu "${selected_gpu}"
run_dataset IRSTD-1K "${selected_gpu}"
