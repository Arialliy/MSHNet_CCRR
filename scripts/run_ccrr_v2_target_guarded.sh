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

dataset_name="${CCRR_DATASET_NAME:-NUDT-SIRST}"
case "${dataset_name}" in
  IRSTD-1K|NUAA-SIRST|NUDT-SIRST) ;;
  *)
    echo "Unsupported CCRR_DATASET_NAME: ${dataset_name}" >&2
    exit 2
    ;;
esac

experiment="${CCRR_TG_EXPERIMENT:-TG1}"
case "${experiment}" in
  TG1)
    default_guard_alpha="0.0"
    default_target_tail_weight="0.0"
    default_fp_value_beta="0.0"
    ;;
  TG2)
    default_guard_alpha="1.0"
    default_target_tail_weight="0.0"
    default_fp_value_beta="0.0"
    ;;
  TG_FULL)
    default_guard_alpha="1.0"
    default_target_tail_weight="5.0"
    default_fp_value_beta="2.0"
    ;;
  *)
    echo "Unsupported CCRR_TG_EXPERIMENT: ${experiment}; expected TG1, TG2, or TG_FULL" >&2
    exit 2
    ;;
esac

dataset_dir="${CCRR_DATASET_DIR:-${resource_repo_dir}/datasets/${dataset_name}}"
weight_path="${CCRR_BASELINE_WEIGHT:-${resource_repo_dir}/checkpoints/baseline/${dataset_name}_mshnet_baseline.pth}"
save_dir="${CCRR_SAVE_DIR:-${repo_dir}/repro_runs/ccrr_v2_target_guarded/${experiment}/${dataset_name}_seed42}"
python_bin="${CCRR_PYTHON:-${resource_repo_dir}/.venv/bin/python}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-${CCRR_GPU:-0}}"
guard_alpha="${CCRR_GUARD_ALPHA:-${default_guard_alpha}}"
target_tail_weight="${CCRR_TARGET_TAIL_WEIGHT:-${default_target_tail_weight}}"
fp_value_beta="${CCRR_FP_VALUE_BETA:-${default_fp_value_beta}}"

if [[ "$(basename "${dataset_dir%/}")" != "${dataset_name}" ]]; then
  echo "Dataset name and directory basename do not match" >&2
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
  exit 2
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

if [[ "${CCRR_ALLOW_CONCURRENT_TRAINING:-0}" != "1" ]]; then
  if command -v flock >/dev/null 2>&1; then
    training_lock_dir="${XDG_RUNTIME_DIR:-/tmp}"
    training_lock="${training_lock_dir}/mshnet-ccrr-training-${UID}.lock"
    exec 9>"${training_lock}"
    if ! flock -n 9; then
      echo "Another TG-SCA launcher holds ${training_lock}; TG-SCA was not started." >&2
      exit 3
    fi
  fi
  if command -v pgrep >/dev/null 2>&1 \
    && pgrep -f -- '([t]rain|[m]ain)\.py.*--(enable-ccrr|dataset-dir)' >/dev/null; then
    echo "Another MSHNet/CCRR training task is active; TG-SCA was not started." >&2
    echo "Set CCRR_ALLOW_CONCURRENT_TRAINING=1 only for an intentional concurrent run." >&2
    exit 3
  fi
fi

mkdir -p "${save_dir}"
echo "Starting TG-SCA ${experiment}: dataset=${dataset_name}, guard_alpha=${guard_alpha}, tail=${target_tail_weight}, fp_beta=${fp_value_beta}, GPU=${gpu_devices}" >&2

CUDA_VISIBLE_DEVICES="${gpu_devices}" "${python_bin}" -u "${repo_dir}/train.py" \
  --dataset-dir "${dataset_dir}" \
  --weight-path "${weight_path}" \
  --enable-ccrr \
  --ccrr-version v2_target_guarded_component \
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
  --candidate-pooling avg_max_topk \
  --candidate-topk-ratio 0.125 \
  --candidate-minimum-topk 1 \
  --risk-threshold 2.0 \
  --quality-veto-threshold 0.2 \
  --guard-veto-threshold 0.2 \
  --risk-alpha 1.0 \
  --guard-alpha "${guard_alpha}" \
  --action-temperature 0.05 \
  --quality-temperature 0.05 \
  --guard-temperature 0.05 \
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
  --lambda-target-guard 2.0 \
  --target-harm-weight 20.0 \
  --target-tail-weight "${target_tail_weight}" \
  --target-tail-temperature 0.1 \
  --missed-clutter-weight 1.0 \
  --fp-value-beta "${fp_value_beta}" \
  --fp-value-max 3.0 \
  --target-guard-positive-weight 4.0 \
  --target-guard-fn-weight 2.0 \
  --target-guard-tail-temperature 0.1 \
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
