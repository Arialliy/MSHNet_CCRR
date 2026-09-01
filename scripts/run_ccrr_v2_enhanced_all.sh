#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
resource_repo_dir="${CCRR_RESOURCE_REPO_DIR:-${repo_dir}}"
if [[ ! -d "${resource_repo_dir}/datasets" ]]; then
  sibling_repo_dir="${repo_dir%_v2_enhanced}"
  if [[ -d "${sibling_repo_dir}/datasets" ]]; then
    resource_repo_dir="${sibling_repo_dir}"
  fi
fi

datasets_dir="${CCRR_DATASETS_DIR:-${resource_repo_dir}/datasets}"
baseline_dir="${CCRR_BASELINE_DIR:-${resource_repo_dir}/checkpoints/baseline}"
experiment="${CCRR_EXPERIMENT:-E5}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v2_enhanced/${experiment}}"
python_bin="${CCRR_PYTHON:-${resource_repo_dir}/.venv/bin/python}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-${CCRR_GPU:-0}}"

if [[ -n "${CCRR_DATASET_DIR:-}" ]]; then
  echo "CCRR_DATASET_DIR is ambiguous for the three-dataset runner; set CCRR_DATASETS_DIR instead." >&2
  exit 2
fi
if [[ -n "${CCRR_BASELINE_WEIGHT:-}" ]]; then
  echo "CCRR_BASELINE_WEIGHT is ambiguous for the three-dataset runner; set CCRR_BASELINE_DIR instead." >&2
  exit 2
fi

for dataset_name in IRSTD-1K NUAA-SIRST NUDT-SIRST; do
  CCRR_DATASET_NAME="${dataset_name}" \
  CCRR_DATASET_DIR="${datasets_dir}/${dataset_name}" \
  CCRR_BASELINE_WEIGHT="${baseline_dir}/${dataset_name}_mshnet_baseline.pth" \
  CCRR_SAVE_DIR="${save_dir}" \
  CCRR_PYTHON="${python_bin}" \
  CUDA_VISIBLE_DEVICES="${gpu_devices}" \
    "${script_dir}/run_ccrr_v2_enhanced.sh"
done
