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

dataset_name="${CCRR_DATASET_NAME:-IRSTD-1K}"
case "${dataset_name}" in
  IRSTD-1K|NUAA-SIRST|NUDT-SIRST) ;;
  *)
    echo "Unsupported CCRR_DATASET_NAME: ${dataset_name}" >&2
    echo "Expected one of: IRSTD-1K, NUAA-SIRST, NUDT-SIRST" >&2
    exit 2
    ;;
esac

dataset_dir="${CCRR_DATASET_DIR:-${resource_repo_dir}/datasets/${dataset_name}}"
weight_path="${CCRR_BASELINE_WEIGHT:-${resource_repo_dir}/checkpoints/baseline/${dataset_name}_mshnet_baseline.pth}"
experiment="${CCRR_EXPERIMENT:-E5}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v2_enhanced/${experiment}}"
python_bin="${CCRR_PYTHON:-${resource_repo_dir}/.venv/bin/python}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-${CCRR_GPU:-0}}"
case "${experiment}" in
  E1)
    default_pooling="avg_max_topk"
    default_target_tail_weight="0.0"
    default_fp_value_beta="0.0"
    ;;
  E2)
    default_pooling="avg"
    default_target_tail_weight="5.0"
    default_fp_value_beta="0.0"
    ;;
  E3)
    default_pooling="avg"
    default_target_tail_weight="0.0"
    default_fp_value_beta="2.0"
    ;;
  E4)
    default_pooling="avg_max_topk"
    default_target_tail_weight="5.0"
    default_fp_value_beta="0.0"
    ;;
  E5)
    default_pooling="avg_max_topk"
    default_target_tail_weight="5.0"
    default_fp_value_beta="2.0"
    ;;
  *)
    echo "Unsupported CCRR_EXPERIMENT: ${experiment}; expected E1, E2, E3, E4, or E5" >&2
    exit 2
    ;;
esac
pooling="${CCRR_CANDIDATE_POOLING:-${default_pooling}}"
target_tail_weight="${CCRR_TARGET_TAIL_WEIGHT:-${default_target_tail_weight}}"
fp_value_beta="${CCRR_FP_VALUE_BETA:-${default_fp_value_beta}}"

case "${pooling}" in
  avg|avg_max_topk) ;;
  *)
    echo "Unsupported CCRR_CANDIDATE_POOLING: ${pooling}" >&2
    exit 2
    ;;
esac

resolved_dataset_name="$(basename "${dataset_dir%/}")"
if [[ "${resolved_dataset_name}" != "${dataset_name}" ]]; then
  echo "Dataset name mismatch: CCRR_DATASET_NAME=${dataset_name}, dataset directory basename=${resolved_dataset_name}" >&2
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

if [[ "${dataset_name}" == "NUDT-SIRST" && "${CCRR_ALLOW_CONCURRENT_NUDT:-0}" != "1" ]]; then
  existing_nudt_active=0
  if command -v pgrep >/dev/null 2>&1 \
    && pgrep -f -- '[t]rain\.py.*--dataset-dir[ =]+[^ ]*/NUDT-SIRST([ /]|$)' >/dev/null; then
    existing_nudt_active=1
  fi
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl --user is-active --quiet mshnet-sca-v2-nudt.service 2>/dev/null; then
      existing_nudt_active=1
    elif systemctl is-active --quiet mshnet-sca-v2-nudt.service 2>/dev/null; then
      existing_nudt_active=1
    fi
  fi
  if [[ "${existing_nudt_active}" == "1" ]]; then
    echo "The existing formal NUDT service is still active; enhanced NUDT was not started." >&2
    echo "Wait for it to finish, or set CCRR_ALLOW_CONCURRENT_NUDT=1 only if concurrent execution is intentional." >&2
    exit 3
  fi
fi

if [[ "${python_bin}" == */* ]]; then
  if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable is unavailable: ${python_bin}" >&2
    exit 2
  fi
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python command is unavailable: ${python_bin}" >&2
  exit 2
fi

mkdir -p "${save_dir}"

echo "Starting SCA-CCRR V2-Enhanced ${experiment}: dataset=${dataset_name}, pooling=${pooling}, tail=${target_tail_weight}, fp_value_beta=${fp_value_beta}, GPU=${gpu_devices}" >&2

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
  --candidate-pooling "${pooling}" \
  --candidate-topk-ratio 0.125 \
  --candidate-minimum-topk 1 \
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
  --target-tail-weight "${target_tail_weight}" \
  --target-tail-temperature 0.1 \
  --missed-clutter-weight 1.0 \
  --fp-value-beta "${fp_value_beta}" \
  --fp-value-max 3.0 \
  --rank-margin 0.5 \
  --quality-iou-weight 0.5 \
  --quality-center-sigma 3.0 \
  --clutter-focal-gamma 2.0 \
  --clutter-positive-alpha 0.75 \
  --save-best-pareto \
  --epochs 1000 \
  --test-start-epoch 500 \
  --batch-size 4 \
  --num-workers 4 \
  --max-train-batches 0 \
  --max-test-batches 0 \
  --seed 42 \
  --device cuda \
  --save-dir "${save_dir}"
