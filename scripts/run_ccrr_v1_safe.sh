#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
dataset_dir="${CCRR_DATASET_DIR:-${repo_dir}/datasets/IRSTD-1K}"
weight_path="${CCRR_BASELINE_WEIGHT:-${repo_dir}/checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v1_safe}"
python_bin="${CCRR_PYTHON:-${repo_dir}/.venv/bin/python}"
gpu_index="${CCRR_GPU:-0}"

CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -u "${repo_dir}/train.py" \
  --dataset-dir "${dataset_dir}" \
  --weight-path "${weight_path}" \
  --enable-ccrr \
  --ccrr-stage head_only \
  --ccrr-num-classes 2 \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --hidden-dim 64 \
  --roi-size 7 \
  --context-scale 3.0 \
  --min-context-size 15 \
  --ccrr-dropout 0.3 \
  --max-suppression 1.5 \
  --gate-margin 0.5 \
  --gate-temperature 0.1 \
  --ccrr-lr 3e-4 \
  --weight-decay 1e-3 \
  --scheduler cosine \
  --eta-min 1e-6 \
  --lambda-refined 0.5 \
  --lambda-candidate 1.0 \
  --lambda-calibration 0.05 \
  --lambda-preservation 1.0 \
  --label-smoothing 0.05 \
  --target-allowed-peak-drop 0.01 \
  --clutter-peak-ceiling 0.45 \
  --easy-negative-weight 0.5 \
  --hard-negative-weight 2.0 \
  --hardness-gamma 2.0 \
  --epochs 1000 \
  --test-start-epoch 500 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --save-dir "${save_dir}"
