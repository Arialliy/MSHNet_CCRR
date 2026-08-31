#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
gpu_index="${CCRR_GPU:-0}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v1_threshold_aware}"

for dataset_name in IRSTD-1K NUAA-SIRST NUDT-SIRST; do
  CCRR_DATASET_NAME="${dataset_name}" \
  CCRR_GPU="${gpu_index}" \
  CCRR_SAVE_DIR="${save_dir}" \
    "${script_dir}/run_ccrr_v1_threshold_aware.sh"
done
