# MSHNet-CCRR: Candidate--Context Reliability Rectification for Infrared Small Target Detection

## Notice! 📰
First of all, thank you to all relevant workers for your attention. Recently, many people have discovered some obvious errors in the code, so we re-checked, modified and debugged the code. Surprisingly, we unexpectedly obtained a pretty good result on the IRSTD-1k data set. The results are published below for your reference.
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.87 | 92.86 | 8.88 | [new_weights](https://drive.google.com/file/d/1CSDwQG8xg7hv0_oGKa4NCEWUiMRU7eIs/view?usp=sharing) |

## Overview
![](assert/overview.png)

## Introduction
This repository extends the official implementation of the CVPR 2024 paper [Infrared Small Target Detection with Scale and Location Sensitivity](https://arxiv.org/abs/2403.19366) with an opt-in Candidate--Context Reliability Rectification (CCRR) workflow. The original MSHNet path remains the default.

In this paper, we first propose a novel Scale and Location Sensitive (SLS) loss to handle the limitations of existing losses: 1) for scale sensitivity, we compute a weight for the IoU loss based on target scales to help the detector distinguish targets with different scales: 2) for location sensitivity, we introduce a penalty term based on the center points of targets to help the detector localize targets more precisely. Then, we design a simple Multi-Scale Head to the plain U-Net (MSHNet). By applying SLS loss to each scale of the predictions, our MSHNet outperforms existing state-of-the-art methods by a large margin. In addition, the detection performance of existing detectors can be further improved when trained with our SLS loss, demonstrating the effectiveness and generalization of our SLS loss. The contribution of this paper are as follows:

1. We propose a novel scale and location sensitive loss for infrared small target detection, which helps detectors distinguish objects with different scales and locations.
   
2. We propose a simple but effective detector which achieves SOTA performance without bells and whistles.
   
3. We apply our loss to existing detectors and show that the detection performance can be further boosted.

## Training
The training command is very simple like this:
```
python main.py --dataset-dir DATASET_DIR --batch-size 4 --epochs 1000 --test-start-epoch 500 --lr 0.05 --mode train
```

For example:
```
python main.py --dataset-dir datasets/IRSTD-1K --batch-size 4 --epochs 1000 --test-start-epoch 500 --lr 0.05 --mode train
```

This repo also provides separate entrypoints:
```
.venv/bin/python train.py --dataset-dir datasets/IRSTD-1K --batch-size 4 --epochs 1000 --test-start-epoch 500 --lr 0.05 --device cuda
.venv/bin/python train.py --dataset-dir datasets/NUDT-SIRST --batch-size 4 --epochs 1000 --test-start-epoch 500 --lr 0.05 --device cuda
```

Training checkpoints and best weights are saved under `repro_runs/` by default.
Both `--base-size` and `--crop-size` must be positive multiples of 16.

For a fresh checkout, create a virtual environment and install all runtime dependencies with:
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Testing
You can test the model with the following command:
```
python main.py --dataset-dir datasets/IRSTD-1K --batch-size 4 --mode test --weight-path /path/to/MSHNet_weight.tar
```

Or use the separate testing entrypoint:
```
.venv/bin/python test.py --dataset-dir datasets/IRSTD-1K --weight-path repro_runs/IRSTD-1K-baseline-YYYY-MM-DD-HH-MM-SS/weight.pkl --device cuda
```

The dataset loader requires the official `img_idx/train_<dataset>.txt` and
`img_idx/test_<dataset>.txt` manifests; root-level `trainval.txt`/`test.txt`
files are not used as fallbacks.

## CCRR extension

This repository also contains the opt-in Candidate--Context Reliability
Rectification extension.  The current released experimental implementation is
SCA-CCRR V2-Enhanced.  It combines component-aligned actions, multi-level
features and a selective quality-veto head with masked Avg/Max/Top-K pooling,
target-tail protection, exact-component false-alarm value weighting and a
third non-inferior Pareto checkpoint.  Run the full E5 setting on one dataset
with:

```bash
CCRR_DATASET_NAME=IRSTD-1K CCRR_GPU=0 scripts/run_ccrr_v2_enhanced.sh
```

For all three datasets, use
[`scripts/run_ccrr_v2_enhanced_all.sh`](scripts/run_ccrr_v2_enhanced_all.sh).
The parameter reference is
[`configs/ccrr_v2_enhanced.yaml`](configs/ccrr_v2_enhanced.yaml).  Set
`CCRR_EXPERIMENT` to `E1`, `E2`, `E3`, `E4`, or `E5` for the registered
ablation matrix.

The runner uses the official train/test `img_idx`, 1,000 epochs, threshold 0.5,
no independent validation split, and test-based selection after every epoch
from epoch 500.  It retains `best_miou.pkl` and `best_pd.pkl`, writes
`best_pareto.pkl` only after all five non-inferiority constraints are met, and
records `pareto_status.json` once scheduled evaluation begins.  In an
independent worktree, `CCRR_RESOURCE_REPO_DIR` may point to the checkout
containing local datasets, baseline checkpoints and the virtual environment.

The target-safety prototype is available as the separate
`v2_target_guarded_component` variant.  It adds a high-recall target-presence
head and a soft-train/hard-evaluation veto without changing the E1--E5 module
or checkpoint schema.  The first protected-design test keeps Hybrid Pool while disabling
the previously inconclusive Tail and FP-value terms:

```bash
CCRR_DATASET_NAME=NUDT-SIRST CCRR_TG_EXPERIMENT=TG1 CCRR_GPU=0 \
  scripts/run_ccrr_v2_target_guarded.sh
```

`TG2` additionally places Guard evidence in the risk score, while `TG_FULL`
matches the full TargetGuard + Tail + FP-value configuration.  See
[`configs/ccrr_v2_target_guarded.yaml`](configs/ccrr_v2_target_guarded.yaml) and
[`configs/ccrr_v2_target_guarded_hybrid_guard.yaml`](configs/ccrr_v2_target_guarded_hybrid_guard.yaml).
TargetGuard Pareto selection additionally requires zero target deletion and at
least 0.99 Guard recall.

The base SCA-CCRR V2 remains reproducible through
`scripts/run_ccrr_v2_selective_component.sh`; V1.1, the
[V1 safe protocol](README_CCRR_V1.md) and [V0 execution guide](README_CCRR.md)
remain reproducibility paths.  The
[CCRR-V2 design guide](Paper1_CCRR_V2_全指标提升方案与代码修改指南.md)
records the staged roadmap; the bidirectional Bi-CCRR recovery branch remains
a proposal.  Baseline behavior remains the default.  Manual enhanced launches
must include `--enable-ccrr --ccrr-version v2_selective_component` plus the
enhanced pooling, loss and Pareto options captured by the runner.

Datasets, checkpoints, candidate banks, predictions and run outputs are local
artifacts and are intentionally excluded from version control.  Place or
generate them at the paths used by the commands in the execution guide.

## Visual Results
![](assert/visual_result.png)

## Quantitative Results
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.16 | 93.88 | 15.03 | [IRSTD-1k_weights](https://drive.google.com/file/d/1q3zfzJRczodGQb0dZ3y3KmLn0zz4F8ra/view?usp=drive_link) |
| NUDT-SIRST | 80.55 | 97.99 | 11.77 | [NUDT-SIRST_weights](https://drive.google.com/file/d/1uczanUIHePZqJA79RZu25fv9FNSHSDQZ/view?usp=drive_link) |


## Citation
**Please kindly cite the papers if this code is useful and helpful for your research.**

    @inproceedings{liu2024infrared,
      title={Infrared Small Target Detection with Scale and Location Sensitivity},
      author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
      booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
      year={2024}
    }
