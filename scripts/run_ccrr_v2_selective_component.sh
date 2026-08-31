#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

dataset_name="${CCRR_DATASET_NAME:-IRSTD-1K}"
case "${dataset_name}" in
  IRSTD-1K|NUAA-SIRST|NUDT-SIRST) ;;
  *)
    echo "Unsupported CCRR_DATASET_NAME: ${dataset_name}" >&2
    echo "Expected one of: IRSTD-1K, NUAA-SIRST, NUDT-SIRST" >&2
    exit 2
    ;;
esac

dataset_dir="${CCRR_DATASET_DIR:-${repo_dir}/datasets/${dataset_name}}"
weight_path="${CCRR_BASELINE_WEIGHT:-${repo_dir}/checkpoints/baseline/${dataset_name}_mshnet_baseline.pth}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v2_selective_component}"
python_bin="${CCRR_PYTHON:-${repo_dir}/.venv/bin/python}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-${CCRR_GPU:-0}}"

resolved_dataset_name="$(basename "${dataset_dir%/}")"
if [[ "${resolved_dataset_name}" != "${dataset_name}" ]]; then
  echo "Dataset name mismatch: CCRR_DATASET_NAME=${dataset_name}, dataset directory basename=${resolved_dataset_name}" >&2
  echo "The official manifests are resolved from the dataset directory basename." >&2
  exit 2
fi

if [[ ! -d "${dataset_dir}" ]]; then
  echo "Dataset directory does not exist: ${dataset_dir}" >&2
  exit 2
fi

for split in train test; do
  manifest="${dataset_dir}/img_idx/${split}_${dataset_name}.txt"
  if [[ ! -s "${manifest}" ]]; then
    echo "Missing or empty official ${split} manifest: ${manifest}" >&2
    exit 2
  fi
done

if [[ ! -f "${weight_path}" ]]; then
  echo "Baseline weight does not exist: ${weight_path}" >&2
  echo "Set CCRR_BASELINE_WEIGHT to the matching ${dataset_name} baseline checkpoint." >&2
  exit 2
fi

if [[ "${python_bin}" == */* ]]; then
  if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable is unavailable: ${python_bin}" >&2
    echo "Set CCRR_PYTHON to a valid Python executable." >&2
    exit 2
  fi
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python command is unavailable: ${python_bin}" >&2
  echo "Set CCRR_PYTHON to a valid Python executable." >&2
  exit 2
fi

mkdir -p "${save_dir}"

echo "Starting SCA-CCRR v2: dataset=${dataset_name}, baseline=${weight_path}, GPU=${gpu_devices}" >&2

CUDA_VISIBLE_DEVICES="${gpu_devices}" "${python_bin}" -u "${repo_dir}/train.py" \
  --dataset-dir "${dataset_dir}" \
  --weight-path "${weight_path}" \
  --enable-ccrr \
  --ccrr-version v2_selective_component \
  --ccrr-stage head_only \
  --ccrr-num-classes 2 \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --positive-iou 0.3 \
  --center-distance 3.0 \
  --min-candidate-area 1 \
  --max-candidate-area 1024 \
  --sca-feature-channels 32 \
  --sca-roi-size 15 \
  --hidden-dim 64 \
  --context-scale 3.0 \
  --min-context-size 15 \
  --ccrr-dropout 0.3 \
  --risk-threshold 2.0 \
  --quality-veto-threshold 0.2 \
  --risk-alpha 1.0 \
  --action-temperature 0.05 \
  --remove-threshold 0.45 \
  --ccrr-lr 3e-4 \
  --weight-decay 1e-3 \
  --scheduler cosine \
  --eta-min 1e-6 \
  --lambda-refined 0.5 \
  --lambda-clutter-cls 1.0 \
  --lambda-quality 1.0 \
  --lambda-action-risk 1.0 \
  --lambda-rank 0.1 \
  --target-harm-weight 20.0 \
  --missed-clutter-weight 1.0 \
  --rank-margin 0.5 \
  --quality-iou-weight 0.5 \
  --quality-center-sigma 3.0 \
  --clutter-focal-gamma 2.0 \
  --clutter-positive-alpha 0.75 \
  --epochs 1000 \
  --test-start-epoch 500 \
  --batch-size 4 \
  --num-workers 4 \
  --max-train-batches 0 \
  --max-test-batches 0 \
  --seed 42 \
  --device cuda \
  --save-dir "${save_dir}"
