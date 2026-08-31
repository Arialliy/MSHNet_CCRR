#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

datasets_dir="${CCRR_DATASETS_DIR:-${repo_dir}/datasets}"
baseline_dir="${CCRR_BASELINE_DIR:-${repo_dir}/checkpoints/baseline}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v2_selective_component}"
python_bin="${CCRR_PYTHON:-${repo_dir}/.venv/bin/python}"
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
  dataset_dir="${datasets_dir}/${dataset_name}"
  weight_path="${baseline_dir}/${dataset_name}_mshnet_baseline.pth"

  CCRR_DATASET_NAME="${dataset_name}" \
  CCRR_DATASET_DIR="${dataset_dir}" \
  CCRR_BASELINE_WEIGHT="${weight_path}" \
  CCRR_SAVE_DIR="${save_dir}" \
  CCRR_PYTHON="${python_bin}" \
  CUDA_VISIBLE_DEVICES="${gpu_devices}" \
    "${script_dir}/run_ccrr_v2_selective_component.sh"
done
