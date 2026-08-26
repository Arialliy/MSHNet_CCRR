# MSHNet-CCRR execution guide

This repository implements the Candidate--Context Reliability Rectification
(CCRR) workflow described in the
[design and implementation guide](Paper1_MSHNet_CCRR_方案总结与代码修改步骤.md).
The baseline path remains the default; CCRR is enabled only with
`--enable-ccrr`.

Datasets, checkpoints, candidate banks, metrics, predictions and run outputs
are local inputs or generated artifacts and are intentionally not versioned.
The paths below describe the expected local workspace layout.

## Frozen settings

- Source MSHNet commit: `46cdfd46802629da51f70124662af7335be74b56`
- Candidate-generation threshold: `0.2`
- High-confidence/hard-negative threshold: `0.5`
- Hard-negative score: `coarse_peak`
- Positive match: instance IoU >= `0.3`, or the candidate center lies inside
  the same GT instance **and** its centroid distance is <= `3` pixels
- MVP reliability classes: target/clutter; uncertain candidates are ignored
- Core/context ROI sizes: `1x` and `3x`; ROIAlign output: `7x7`
- Dataset indices: `datasets/<dataset>/img_idx/train_*.txt` and `test_*.txt`
- Training length: 1,000 epochs; test runs every epoch from zero-based epoch 500
- No validation split is created

## Verified baseline

The verification run used frozen epoch-799 checkpoints stored locally under
`checkpoints/baseline/`.  Full 256x256 evaluation at probability threshold
0.5 produced:

| Dataset | mIoU | nIoU | Pd | Fa / 1e6 pixels |
|---|---:|---:|---:|---:|
| IRSTD-1K | 0.687814 | 0.612314 | 0.942177 | 8.8061 |
| NUAA-SIRST | 0.737422 | 0.722778 | 0.961977 | 24.0290 |
| NUDT-SIRST | 0.806957 | 0.826626 | 0.959788 | 11.6050 |

The IRSTD-1K legacy evaluator reports `8.1988`; the corrected one-to-one
component matcher finds eight additional false-alarm pixels and reports
`8.8061`.  NUAA-SIRST and NUDT-SIRST are unchanged by this correction.
The evaluation commands write machine-readable results to
`baseline_metrics/final_main_regression_<dataset>.json` and binary predictions
to `baseline_predictions/` in the local workspace.

Run a baseline regression with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth \
  --device cuda \
  --metrics-output baseline_metrics/main_regression_IRSTD-1K.json
```

## Go/No-Go diagnosis

The 0.5 high-confidence operating point passes the Go gate on all three test
sets:

| Dataset | candidates | target | clutter | uncertain | high-conf. FP | affected images | FPPI@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 294 | 98 | 22 | 174 | 192 | 145 | 0.9552 |
| NUAA-SIRST | 250 | 184 | 3 | 63 | 63 | 60 | 0.2944 |
| NUDT-SIRST | 923 | 712 | 10 | 201 | 205 | 178 | 0.3087 |

The diagnostic command writes strict-label JSON and FROC points to
`baseline_diagnosis/<dataset>/test/`.  To reproduce one set:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_baseline.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth \
  --output-dir baseline_diagnosis/IRSTD-1K/test \
  --split test \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --positive-iou 0.3 \
  --center-distance 3 \
  --device cuda
```

## Offline candidate bank and temperature baseline

Build deterministic, augmentation-free candidate banks with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/build_candidate_bank.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth \
  --output-dir candidate_bank/IRSTD-1K/train \
  --splits train \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --positive-iou 0.3 \
  --center-distance 3 \
  --batch-size 8 \
  --device cuda
```

Build the test bank with the same command after changing the output to
`candidate_bank/IRSTD-1K/test` and `--splits test`.  Existing banks are
replaced only when `--overwrite` is supplied.

The local IRSTD-1K verification bank contained 1,164 candidates: 419 target,
57 clutter and 688 uncertain.  During binary MVP training, uncertain
candidates are ignored and inverse-frequency target/clutter weights are
derived from bank metadata unless `--candidate-class-weights` is supplied
explicitly.

Fit the monotone temperature baseline without image leakage:

```bash
.venv/bin/python scripts/fit_temperature.py \
  --candidate-json candidate_bank/IRSTD-1K/train/train_candidates.json \
  --output-json candidate_bank/IRSTD-1K/train/temperature_calibration.json \
  --calibration-fraction 0.5 \
  --seed 42
```

Temperature scaling can improve ECE/Brier/NLL, but cannot change ordering,
FROC, or the set of operating points attainable by sweeping a threshold.

## Frozen-backbone binary CCRR MVP

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth \
  --enable-ccrr \
  --ccrr-stage head_only \
  --candidate-bank candidate_bank/IRSTD-1K/train/train_candidates.json \
  --test-candidate-bank candidate_bank/IRSTD-1K/test/test_candidates.json \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --batch-size 4 \
  --epochs 1000 \
  --test-start-epoch 500 \
  --device cuda \
  --save-dir repro_runs/ccrr_head_only
```

In `head_only`, the MSHNet parameters and BatchNorm statistics are frozen.
The loss is exactly:

```text
L = L_base
  + lambda_refined * L_refined
  + lambda_candidate * L_weighted_CE
  + lambda_calibration * L_Brier
  + lambda_preservation * L_preservation
```

Omit both bank arguments to generate candidates online.  A bounded integration
run can use `--max-train-batches 2 --max-test-batches 5 --test-start-epoch 0`.

Epochs 0--499 save only the latest `weight.pkl` and `checkpoint.pkl`.  From
epoch 500 onward the test set is evaluated after every epoch and the run also
saves `best_pd.pkl` and `best_miou.pkl` independently.  These are structured
weight artifacts: each includes the state dict, selected metric/epoch,
inference/evaluation settings, candidate-bank hashes and baseline ancestry.
Loading a CCRR artifact with a different model, threshold, dataset manifest or
test-bank configuration fails explicitly.

## Joint fine-tuning

After the frozen-head gate succeeds, use `--ccrr-stage joint`.  CCRR uses
learning rate `1e-3`; `decoder_0`, `output_0` and `final` use `1e-4`; earlier
layers remain frozen.  Joint training must start from a complete head-only
CCRR artifact and uses online candidates because it changes the proposal
logits; offline train/test banks are intentionally rejected for this stage.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path PATH_TO_HEAD_ONLY_BEST \
  --enable-ccrr \
  --ccrr-stage joint \
  --epochs 1000 \
  --test-start-epoch 500 \
  --device cuda
```

## Evaluation and resume

Evaluate a full CCRR weight:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path PATH_TO_CCRR_WEIGHT \
  --enable-ccrr \
  --candidate-bank candidate_bank/IRSTD-1K/test/test_candidates.json \
  --hard-negative-threshold 0.5 \
  --device cuda \
  --metrics-output PATH_TO_METRICS_JSON
```

Resume with the same architecture and candidate settings:

```bash
.venv/bin/python train.py \
  --dataset-dir datasets/IRSTD-1K \
  --enable-ccrr \
  --ccrr-stage head_only \
  --candidate-bank candidate_bank/IRSTD-1K/train/train_candidates.json \
  --test-candidate-bank candidate_bank/IRSTD-1K/test/test_candidates.json \
  --if-checkpoint true \
  --resume-path PATH_TO_CHECKPOINT \
  --epochs 1000 \
  --test-start-epoch 500
```

Each scheduled test reports coarse/refined mIoU, nIoU, Pd and Fa, full
segmentation FROC/FPPI/Fa-at-fixed-Pd curves, plus paired raw-vs-CCRR candidate
ECE, Brier, NLL, risk--coverage/AURC and FROC.

## Verification status

- Full test suite: `59 passed`
- Full IRSTD-1K baseline regression: exact match to the stored baseline metrics
- Baseline train/test entrypoints: exercised
- CCRR explicit, online and empty-candidate forward/backward paths: exercised
- Frozen-head and joint optimizer paths: exercised
- Weight save/load and checkpoint resume: exercised
- Structured dual-best save/reload, strict config rejection and checkpoint
  continuation: exercised on a bounded GPU pipeline run

The one-epoch CCRR result is not a paper result.  Under the legacy evaluator,
it kept Pd unchanged and slightly reduced Fa on IRSTD-1K, but formal claims
require convergence, three datasets, ablations and repeated seeds.  The
requested no-validation schedule selects `best_pd`/`best_miou` on the test set,
so those weights are explicitly test-selected rather than an unbiased one-shot
test estimate.
