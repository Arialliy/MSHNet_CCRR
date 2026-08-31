# CCRR-V1 safe protocol

> **Status:** This document records the V1 safe baseline. New threshold-aware
> experiments use `configs/ccrr_v1_threshold_aware.yaml` and
> `scripts/run_ccrr_v1_threshold_aware.sh`; the suppression-only V1 safety
> invariant is preserved.

CCRR-V1 treats the previous 1000-epoch head-only run as a V0 diagnostic. Do
not initialize joint training from a V0 best weight.

The V1 invariant is target-preserving clutter suppression:

- every proposal is supervised as target-presence or clutter;
- proposal score controls negative sample weight, not class membership;
- core and ring context share a low-capacity encoder;
- rectification is suppression-only, so refined logits cannot exceed coarse
  logits;
- no validation split is created or read;
- the existing test split is evaluated every epoch from epoch 500;
- test-selected `best_miou.pkl` and `best_pd.pkl` are saved as requested.

## IRSTD-1K split

The dataset loader accepts only the existing manifests under
`datasets/IRSTD-1K/img_idx/`:

- `train_IRSTD-1K.txt`: 800 training images;
- `test_IRSTD-1K.txt`: 201 test images.

It does not create a split or fall back to root-level lists. It rejects
missing manifests, duplicate names, train/test overlap, and missing image or
mask files.

## Training

Run from the repository root:

```bash
CCRR_GPU=1 scripts/run_ccrr_v1_safe.sh
```

The runner uses online proposals and training augmentation. It deliberately
does not accept the V0 offline candidate bank.

Primary artifacts are:

- `best_miou.pkl` and `best_pd.pkl`: test-selected models;
- `checkpoint.pkl`: exact resumable state including optimizer and scheduler;
- `metrics.jsonl`;
- `diagnostics/test_candidates_epoch_*.jsonl`.

Exact training continuation must use `checkpoint.pkl` with the same model,
evaluation and training configuration.  The checkpoint restores the model,
optimizer, scheduler and random-number-generator states.  The test-selected
`best_miou.pkl` and `best_pd.pkl` artifacts contain model weights for evaluation
and are deliberately rejected as exact-resume inputs.

Because model selection uses the test split by explicit project protocol,
these weights and reported metrics must be described as test-selected. Do not
proceed to `joint` unless refined Fa decreases while Pd remains stable and the
candidate/delta diagnostics pass.
