#!/usr/bin/env python3
"""Train and evaluate the MSHNet baseline or its CCRR extension.

The default command line remains a baseline MSHNet run.  CCRR is opt-in via
``--enable-ccrr`` and supports the frozen-backbone MVP (``head_only``) and a
small joint fine-tuning stage (``joint``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.measure import label as connected_components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adagrad, AdamW
from tqdm import tqdm

from model.MSHNet import MSHNet
from model.candidate_loss import CCRRLoss
from model.loss import AverageMeter, SLSIoULoss
from utils.candidate import (
    LABEL_NAMES,
    TARGET_LABEL,
    UNCERTAIN_LABEL,
    match_candidates_to_gt,
)
from utils.data import IRSTD_Dataset
from utils.detection_metric import SegmentationFROC
from utils.metric import PD_FA
from utils.reliability_metric import (
    candidate_brier_score,
    candidate_ece,
    candidate_nll,
    false_alarm_at_fixed_pd,
    fppi_at_fixed_pd,
    fppi_froc,
    risk_coverage_curve,
)


CANDIDATE_BANK_SCHEMA_VERSION = "mshnet-ccrr-candidate-bank/v1"
WEIGHT_SCHEMA_VERSION = "mshnet-ccrr-weight/v1"


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in ("yes", "true", "t", "1", "y"):
        return True
    if normalized in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args(default_mode: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Original MSHNet options.
    parser.add_argument("--dataset-dir", default="datasets/IRSTD-1K")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--test-start-epoch",
        type=int,
        default=500,
        help="Run the test set after every epoch whose zero-based index is >= this value.",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--multi-gpus", type=str2bool, default=False)
    parser.add_argument("--if-checkpoint", type=str2bool, default=False)
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--save-dir", default="repro_runs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=("train", "test"), default=default_mode or "train"
    )
    parser.add_argument("--weight-path", default="")
    parser.add_argument("--metrics-output", default="")

    # Candidate generation and matching.  The 0.5 hard-negative threshold is
    # the user-selected operating point; the permissive proposal threshold is
    # intentionally kept at 0.2.
    parser.add_argument("--enable-ccrr", action="store_true")
    parser.add_argument(
        "--ccrr-stage", choices=("head_only", "joint"), default="head_only"
    )
    parser.add_argument("--candidate-bank", default="")
    parser.add_argument(
        "--test-candidate-bank",
        "--val-candidate-bank",
        dest="test_candidate_bank",
        default="",
        help="Optional final-test bank; final testing falls back to online candidates.",
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.2)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-score",
        choices=("coarse_peak", "coarse_mean", "scale_peak", "scale_mean"),
        default="coarse_peak",
    )
    parser.add_argument("--positive-iou", type=float, default=0.3)
    parser.add_argument("--center-distance", type=float, default=3.0)
    parser.add_argument("--froc-bins", type=int, default=100)
    parser.add_argument("--min-candidate-area", type=int, default=1)
    parser.add_argument("--max-candidate-area", type=int, default=1024)

    # CCRR model and optimization.
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--max-delta", type=float, default=4.0)
    parser.add_argument("--ccrr-num-classes", type=int, choices=(2, 3), default=2)
    parser.add_argument("--ccrr-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-refined", type=float, default=1.0)
    parser.add_argument("--lambda-candidate", type=float, default=1.0)
    parser.add_argument("--lambda-calibration", type=float, default=0.1)
    parser.add_argument("--lambda-preservation", type=float, default=0.2)
    parser.add_argument("--clutter-margin", type=float, default=0.1)
    parser.add_argument(
        "--candidate-class-weights",
        type=float,
        nargs="+",
        default=None,
        help="One weight per class; default derives inverse-frequency weights from the train bank.",
    )

    # Bounded runs make integration checks reproducible without changing the
    # behavior of full experiments (zero means unlimited).
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--max-test-batches",
        "--max-val-batches",
        dest="max_test_batches",
        type=int,
        default=0,
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name_manifest_sha256(names: list[str]) -> str:
    payload = "\n".join(str(name) for name in names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def should_run_scheduled_test(epoch: int, test_start_epoch: int) -> bool:
    """The schedule is zero-based: epoch 500 is the first tested epoch."""

    return int(epoch) >= int(test_start_epoch)


class CandidateBank:
    """Decode the JSON/RLE artifact written by ``diagnose_baseline.py``."""

    def __init__(self, path: str, image_size: int) -> None:
        raw_payload = Path(path).read_bytes()
        payload = json.loads(raw_payload.decode("utf-8"))
        records = payload.get("candidates", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("candidate bank must contain a list of candidate records")
        self.path = path
        self.sha256 = hashlib.sha256(raw_payload).hexdigest()
        self.image_size = int(image_size)
        self.metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if isinstance(payload, dict) and isinstance(payload.get("images"), list):
            self.image_records: list[dict[str, Any]] | None = payload["images"]
            self.declared_image_name_list: list[str] | None = [
                str(record["name"])
                for record in payload["images"]
                if isinstance(record, dict) and "name" in record
            ]
            self.declared_image_names: set[str] | None = set(
                self.declared_image_name_list
            )
        else:
            self.image_records = None
            self.declared_image_name_list = None
            self.declared_image_names = None
        self.by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if not isinstance(record, dict) or "name" not in record:
                raise ValueError("every candidate-bank record must contain an image name")
            self.by_name[str(record["name"])].append(record)
        self.num_candidates = len(records)

    def validate_contract(
        self,
        args: argparse.Namespace,
        *,
        expected_split: str,
        expected_names: list[str],
    ) -> None:
        """Reject stale, partial, or differently configured offline banks."""

        required_metadata = {
            "schema_version": CANDIDATE_BANK_SCHEMA_VERSION,
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "split": expected_split,
            "candidate_threshold": float(args.candidate_threshold),
            "hard_negative_threshold": float(args.hard_negative_threshold),
            "candidate_score": args.candidate_score,
            "positive_iou": float(args.positive_iou),
            "center_distance": float(args.center_distance),
            "min_area": int(args.min_candidate_area),
            "max_area": int(args.max_candidate_area)
            if args.max_candidate_area
            else None,
            "base_size": int(args.base_size),
            "label_order": list(LABEL_NAMES),
            "box_format": "xyxy_half_open",
            "mask_encoding": "row_major_start_length_rle",
            "proposal_aggregation": "mean_sigmoid_multiscale",
            "component_connectivity": 8,
            "matching_rule": "iou_or_center_inside_and_centroid_distance",
        }
        differences: list[str] = []
        for key, expected in required_metadata.items():
            if key not in self.metadata:
                differences.append(f"{key}: missing (expected {expected!r})")
                continue
            actual = self.metadata[key]
            if isinstance(expected, float):
                try:
                    equal = math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = actual == expected
            if not equal:
                differences.append(f"{key}: {actual!r} != {expected!r}")
        if differences:
            raise ValueError(
                f"candidate bank {self.path!r} violates its metadata contract: "
                + "; ".join(differences)
            )
        weight_sha256 = self.metadata.get("weight_sha256")
        if not isinstance(weight_sha256, str) or len(weight_sha256) != 64:
            raise ValueError(
                f"candidate bank {self.path!r} has no valid baseline weight SHA256"
            )

        if self.declared_image_name_list is None:
            raise ValueError(
                f"candidate bank {self.path!r} has no complete image manifest"
            )
        if len(self.declared_image_name_list) != len(self.declared_image_names):
            raise ValueError(f"candidate bank {self.path!r} declares duplicate images")
        if self.declared_image_name_list != expected_names:
            expected_set = set(expected_names)
            missing = expected_set - self.declared_image_names
            extra = self.declared_image_names - expected_set
            order_mismatch = not missing and not extra
            raise ValueError(
                f"candidate bank {self.path!r} image manifest does not match "
                f"img_idx/{expected_split}: missing={len(missing)}, "
                f"extra={len(extra)}, order_mismatch={order_mismatch}"
            )
        if int(self.metadata.get("num_images", -1)) != len(expected_names):
            raise ValueError(
                f"candidate bank {self.path!r} metadata num_images is inconsistent"
            )
        if int(self.metadata.get("num_candidates", -1)) != self.num_candidates:
            raise ValueError(
                f"candidate bank {self.path!r} metadata num_candidates is inconsistent"
            )
        unknown_record_names = set(self.by_name) - self.declared_image_names
        if unknown_record_names:
            raise ValueError(
                f"candidate bank {self.path!r} has records outside its image manifest"
            )
        for image_record in self.image_records or []:
            name = str(image_record["name"])
            if int(image_record.get("num_candidates", -1)) != len(self.by_name[name]):
                raise ValueError(
                    f"candidate bank {self.path!r} has an inconsistent count for {name}"
                )

    @staticmethod
    def _decode_rle(runs: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
        flat = np.zeros(shape[0] * shape[1], dtype=np.bool_)
        for start, length in runs:
            start, length = int(start), int(length)
            if start < 0 or length < 0 or start + length > flat.size:
                raise ValueError("candidate-bank RLE is outside its declared mask shape")
            flat[start : start + length] = True
        return flat.reshape(shape)

    def get(self, names: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        masks: list[torch.Tensor] = []
        boxes: list[list[float]] = []
        fields: dict[str, list[float]] = {
            "scores": [],
            "coarse_peak_scores": [],
            "coarse_scores": [],
            "peak_scores": [],
        }
        label_ids: list[int] = []
        for batch_index, name in enumerate(names):
            for record in self.by_name.get(str(name), []):
                shape = tuple(int(value) for value in record.get(
                    "mask_shape", (self.image_size, self.image_size)
                ))
                if shape != (self.image_size, self.image_size):
                    raise ValueError(
                        f"candidate bank mask {shape} does not match --base-size "
                        f"{self.image_size}"
                    )
                masks.append(torch.from_numpy(self._decode_rle(record["mask_rle"], shape)))
                coordinates = [float(value) for value in record["box"][-4:]]
                boxes.append([float(batch_index), *coordinates])
                fields["scores"].append(float(record.get("scale_mean_score", record["score"])))
                fields["coarse_peak_scores"].append(
                    float(record.get("coarse_peak_score", record["score"]))
                )
                fields["coarse_scores"].append(
                    float(
                        record.get(
                            "coarse_mean_score",
                            record.get("coarse_score", record["score"]),
                        )
                    )
                )
                fields["peak_scores"].append(
                    float(
                        record.get(
                            "scale_peak_score",
                            record.get("peak_score", record["score"]),
                        )
                    )
                )
                label_ids.append(int(record.get("label_id", UNCERTAIN_LABEL)))

        if masks:
            mask_tensor = torch.stack(masks).to(device=device)
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        else:
            mask_tensor = torch.zeros(
                (0, self.image_size, self.image_size), dtype=torch.bool, device=device
            )
            box_tensor = torch.empty((0, 5), dtype=torch.float32, device=device)
        result = {
            "masks": mask_tensor,
            "candidate_masks": mask_tensor,
            "boxes": box_tensor,
            "batch_indices": box_tensor[:, 0].long(),
            "bank_labels": torch.tensor(label_ids, dtype=torch.long, device=device),
        }
        for key, values in fields.items():
            result[key] = torch.tensor(values, dtype=torch.float32, device=device)
        return result


class DetectionMetrics:
    """Original MSHNet metrics at its zero-logit (probability 0.5) threshold."""

    def __init__(self, image_size: int) -> None:
        self.pd_fa = PD_FA(1, 10, image_size)
        self.total_intersection = 0
        self.total_union = 0
        self.per_image_iou: list[float] = []

    def reset(self) -> None:
        self.pd_fa.reset()
        self.total_intersection = 0
        self.total_union = 0
        self.per_image_iou.clear()

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.shape[0] != 1:
            raise ValueError("PD/FA evaluation requires test batch size 1")
        prediction = logits > 0
        target = labels > 0
        intersection = int((prediction & target).sum().item())
        union = int((prediction | target).sum().item())
        self.total_intersection += intersection
        self.total_union += union
        self.per_image_iou.append(intersection / union if union else 1.0)
        self.pd_fa.update(logits, labels)

    @property
    def mean_iou(self) -> float:
        return (
            self.total_intersection / self.total_union
            if self.total_union
            else 1.0
        )

    def get(self, num_images: int) -> dict[str, float]:
        false_alarm, detection_probability = self.pd_fa.get(num_images)
        return {
            "mIoU": float(self.mean_iou),
            "nIoU": float(np.mean(self.per_image_iou)),
            "Pd": float(detection_probability[0]),
            "Fa_per_million_pixels": float(false_alarm[0] * 1_000_000),
        }


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.mode
        self.start_epoch = 0
        self.best_iou = -1.0
        self.best_pd = -1.0
        self.warm_epoch = args.warm_epoch
        self.parent_weight_sha256: str | None = None
        self.baseline_weight_sha256: str | None = None
        self._validate_args()
        self.device = self._select_device(args.device)
        self.train_candidate_bank = (
            CandidateBank(args.candidate_bank, args.base_size)
            if args.enable_ccrr and args.mode == "train" and args.candidate_bank
            else None
        )
        evaluation_bank_path = args.test_candidate_bank
        if args.mode == "test" and not evaluation_bank_path:
            evaluation_bank_path = args.candidate_bank
        self.eval_candidate_bank = (
            CandidateBank(evaluation_bank_path, args.base_size)
            if args.enable_ccrr and evaluation_bank_path
            else None
        )

        self.train_loader: Data.DataLoader | None = None
        if args.mode == "train":
            # Offline candidates are in the canonical resized coordinate
            # system, so the frozen-head stage deliberately disables random
            # geometric augmentation.
            if args.enable_ccrr and (
                args.ccrr_stage == "head_only" or args.candidate_bank
            ):
                train_source = IRSTD_Dataset(args, mode="test", split="train")
            else:
                train_source = IRSTD_Dataset(args, mode="train")
            trainset: Data.Dataset = train_source
            self.train_loader = Data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=self.device.type == "cuda",
            )
        testset = IRSTD_Dataset(args, mode="test", split="test")
        self.test_loader = Data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.split_summary = {
            "validation_source": None,
            "evaluation_source": "test",
            "used_for_model_selection": args.mode == "train",
            "num_train_images": len(self.train_loader.dataset)
            if self.train_loader is not None
            else None,
            "num_test_images": len(self.test_loader.dataset),
        }
        if self.train_candidate_bank is not None:
            if self.train_loader is None:
                raise RuntimeError("train candidate bank requires a training loader")
            self.train_candidate_bank.validate_contract(
                args,
                expected_split="train",
                expected_names=self._dataset_names(self.train_loader.dataset),
            )
        if self.eval_candidate_bank is not None:
            self.eval_candidate_bank.validate_contract(
                args,
                expected_split="test",
                expected_names=self._dataset_names(self.test_loader.dataset),
            )

        ccrr_config = None
        if args.enable_ccrr:
            ccrr_config = {
                "num_scales": 4,
                "roi_size": args.roi_size,
                "hidden_dim": args.hidden_dim,
                "context_scale": args.context_scale,
                "max_delta": args.max_delta,
                "num_classes": args.ccrr_num_classes,
            }
        model: nn.Module = MSHNet(3, ccrr_config=ccrr_config)
        if args.multi_gpus:
            if args.enable_ccrr:
                raise ValueError("CCRR currently requires --multi-gpus false")
            if self.device.type == "cuda" and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self._configure_trainable_parameters()
        self.optimizer = self._build_optimizer()

        self.loss_fun = SLSIoULoss()
        self.ccrr_loss: CCRRLoss | None = None
        if args.enable_ccrr:
            class_weights = self._resolve_candidate_class_weights()
            if len(class_weights) != args.ccrr_num_classes:
                if args.ccrr_num_classes == 3 and len(class_weights) == 2:
                    class_weights.append(0.0)
                else:
                    raise ValueError(
                        "--candidate-class-weights must have one value per CCRR class"
                    )
            self.ccrr_loss = CCRRLoss(
                class_weights=class_weights,
                ignore_index=-1,
                clutter_margin=args.clutter_margin,
                classification_weight=1.0,
                calibration_weight=1.0,
                preservation_weight=1.0,
            )
            args.resolved_candidate_class_weights = class_weights

        self.save_folder = self._new_save_folder()
        if args.mode == "train":
            if args.if_checkpoint:
                self._resume_checkpoint(args.resume_path or args.weight_path)
            elif args.weight_path:
                self._load_weight(
                    args.weight_path,
                    allow_missing_ccrr=(
                        args.enable_ccrr and args.ccrr_stage == "head_only"
                    ),
                )
            elif args.enable_ccrr:
                raise ValueError("CCRR training requires baseline --weight-path")
        else:
            if not args.weight_path:
                raise ValueError("--weight-path is required in test mode")
            self._load_weight(args.weight_path, allow_missing_ccrr=False)
            self.warm_epoch = -1

    def _validate_args(self) -> None:
        for argument in ("base_size", "crop_size"):
            size = getattr(self.args, argument)
            if size <= 0 or size % 16 != 0:
                raise ValueError(f"--{argument.replace('_', '-')} must be a positive multiple of 16")
        for argument in (
            "candidate_threshold",
            "hard_negative_threshold",
            "positive_iou",
        ):
            value = float(getattr(self.args, argument))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument.replace('_', '-')} must lie in [0, 1]")
        if self.args.center_distance < 0:
            raise ValueError("--center-distance must be non-negative")
        if self.args.froc_bins <= 0:
            raise ValueError("--froc-bins must be positive")
        if self.args.min_candidate_area < 1:
            raise ValueError("--min-candidate-area must be positive")
        if self.args.max_candidate_area and (
            self.args.max_candidate_area < self.args.min_candidate_area
        ):
            raise ValueError("--max-candidate-area must be zero or >= minimum area")
        if self.args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        if self.args.test_start_epoch < 0:
            raise ValueError("--test-start-epoch must be non-negative")
        if (
            self.args.mode == "train"
            and self.args.enable_ccrr
            and self.args.ccrr_stage == "joint"
            and (self.args.candidate_bank or self.args.test_candidate_bank)
        ):
            raise ValueError(
                "joint training updates proposal logits and therefore requires "
                "online candidates; omit --candidate-bank and --test-candidate-bank"
            )
        for name in (
            "lambda_refined",
            "lambda_candidate",
            "lambda_calibration",
            "lambda_preservation",
        ):
            if getattr(self.args, name) < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(requested)

    @staticmethod
    def _dataset_names(dataset: Data.Dataset) -> list[str]:
        if isinstance(dataset, Data.Subset):
            source_names = dataset.dataset.names
            return [str(source_names[index]) for index in dataset.indices]
        return [str(name) for name in dataset.names]

    def _state_model(self) -> MSHNet:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _dataset_contract(self, dataset: Data.Dataset) -> dict[str, Any]:
        source = dataset.dataset if isinstance(dataset, Data.Subset) else dataset
        names = self._dataset_names(dataset)
        split_path = Path(source.list_dir).resolve()
        try:
            split_name = str(split_path.relative_to(Path(self.args.dataset_dir).resolve()))
        except ValueError:
            split_name = str(split_path)
        return {
            "split_file": split_name,
            "num_images": len(names),
            "name_manifest_sha256": _name_manifest_sha256(names),
        }

    def _inference_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "schema_version": "mshnet-ccrr-inference/v1",
            "model_variant": "MSHNet+CCRR" if self.args.enable_ccrr else "MSHNet",
            "enable_ccrr": bool(self.args.enable_ccrr),
            "input_channels": 3,
            "feature_channels": 16,
            "num_scales": 4,
            "base_size": int(self.args.base_size),
            "preprocess": "resize_bilinear_imagenet_normalization/v1",
        }
        if self.args.enable_ccrr:
            config.update(
                {
                    "proposal_aggregation": "mean_sigmoid_multiscale",
                    "component_connectivity": 8,
                    "candidate_threshold": float(self.args.candidate_threshold),
                    "min_candidate_area": int(self.args.min_candidate_area),
                    "max_candidate_area": int(self.args.max_candidate_area),
                    "roi_size": int(self.args.roi_size),
                    "hidden_dim": int(self.args.hidden_dim),
                    "context_scale": float(self.args.context_scale),
                    "max_delta": float(self.args.max_delta),
                    "ccrr_num_classes": int(self.args.ccrr_num_classes),
                }
            )
        return config

    def _evaluation_config(self) -> dict[str, Any]:
        bank = self.eval_candidate_bank
        config: dict[str, Any] = {
            "schema_version": "mshnet-ccrr-evaluation/v1",
            "dataset_dir": str(Path(self.args.dataset_dir).resolve()),
            "test_split": self._dataset_contract(self.test_loader.dataset),
            "center_distance": float(self.args.center_distance),
            "probability_threshold": 0.5,
            "froc_bins": int(self.args.froc_bins),
        }
        if self.args.enable_ccrr:
            config.update(
                {
                    "candidate_source": "offline" if bank is not None else "online",
                    "candidate_bank_sha256": bank.sha256 if bank is not None else None,
                    "candidate_score": self.args.candidate_score,
                    "hard_negative_threshold": float(
                        self.args.hard_negative_threshold
                    ),
                    "positive_iou": float(self.args.positive_iou),
                }
            )
        return config

    def _training_config(self) -> dict[str, Any]:
        train_contract = (
            self._dataset_contract(self.train_loader.dataset)
            if self.train_loader is not None
            else None
        )
        bank = self.train_candidate_bank
        config: dict[str, Any] = {
            "schema_version": "mshnet-ccrr-training/v1",
            "batch_size": int(self.args.batch_size),
            "crop_size": int(self.args.crop_size),
            "warm_epoch": int(self.args.warm_epoch),
            "train_split": train_contract,
            "lr": float(self.args.lr),
        }
        if self.args.enable_ccrr:
            config.update(
                {
                    "ccrr_stage": self.args.ccrr_stage,
                    "candidate_source": "offline" if bank is not None else "online",
                    "candidate_bank_sha256": bank.sha256 if bank is not None else None,
                    "resolved_candidate_class_weights": list(
                        getattr(self.args, "resolved_candidate_class_weights", [])
                    ),
                    "ccrr_lr": float(self.args.ccrr_lr),
                    "backbone_lr": float(self.args.backbone_lr),
                    "weight_decay": float(self.args.weight_decay),
                    "lambda_refined": float(self.args.lambda_refined),
                    "lambda_candidate": float(self.args.lambda_candidate),
                    "lambda_calibration": float(self.args.lambda_calibration),
                    "lambda_preservation": float(self.args.lambda_preservation),
                    "clutter_margin": float(self.args.clutter_margin),
                }
            )
        return config

    @staticmethod
    def _config_differences(
        saved: Mapping[str, Any], current: Mapping[str, Any], prefix: str = ""
    ) -> list[str]:
        differences: list[str] = []
        for key, expected in current.items():
            qualified = f"{prefix}.{key}" if prefix else key
            if key not in saved:
                differences.append(f"{qualified}: missing")
                continue
            actual = saved[key]
            if isinstance(expected, Mapping):
                if not isinstance(actual, Mapping):
                    differences.append(f"{qualified}: {actual!r} != mapping")
                else:
                    differences.extend(
                        Trainer._config_differences(actual, expected, qualified)
                    )
            elif isinstance(expected, float):
                try:
                    equal = math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    equal = False
                if not equal:
                    differences.append(f"{qualified}: {actual!r} != {expected!r}")
            elif actual != expected:
                differences.append(f"{qualified}: {actual!r} != {expected!r}")
        return differences

    def _validate_weight_configuration(
        self,
        checkpoint: Any,
        path: str,
        *,
        baseline_initialization: bool,
        validate_evaluation: bool,
        validate_training: bool,
    ) -> None:
        if not isinstance(checkpoint, Mapping) or "inference_config" not in checkpoint:
            if self.args.enable_ccrr and not baseline_initialization:
                raise RuntimeError(
                    f"CCRR weight {path!r} lacks the required inference_config; "
                    "use a structured best/checkpoint artifact"
                )
            return
        if checkpoint.get("checkpoint_schema") != WEIGHT_SCHEMA_VERSION:
            raise RuntimeError(f"weight {path!r} has an unsupported checkpoint schema")

        saved_inference = checkpoint["inference_config"]
        current_inference = self._inference_config()
        if baseline_initialization:
            current_inference = {
                key: current_inference[key]
                for key in (
                    "input_channels",
                    "feature_channels",
                    "num_scales",
                    "base_size",
                    "preprocess",
                )
            }
        differences = self._config_differences(saved_inference, current_inference)
        if validate_evaluation:
            saved_evaluation = checkpoint.get("evaluation_config")
            if not isinstance(saved_evaluation, Mapping):
                differences.append("evaluation_config: missing")
            else:
                differences.extend(
                    self._config_differences(
                        saved_evaluation,
                        self._evaluation_config(),
                        "evaluation_config",
                    )
                )
        if validate_training:
            saved_training = checkpoint.get("training_config")
            if not isinstance(saved_training, Mapping):
                differences.append("training_config: missing")
            else:
                differences.extend(
                    self._config_differences(
                        saved_training,
                        self._training_config(),
                        "training_config",
                    )
                )
        if differences:
            raise RuntimeError(
                f"weight configuration mismatch for {path!r}: "
                + "; ".join(differences)
            )

    def _artifact_payload(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        selection_metric: str | None = None,
        selection_value: float | None = None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_schema": WEIGHT_SCHEMA_VERSION,
            "net": state_dict,
            "epoch": int(epoch),
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "inference_config": self._inference_config(),
            "evaluation_config": self._evaluation_config(),
            "training_config": self._training_config(),
            "provenance": {
                "parent_weight_sha256": self.parent_weight_sha256,
                "baseline_weight_sha256": self.baseline_weight_sha256,
                "train_candidate_bank_sha256": self.train_candidate_bank.sha256
                if self.train_candidate_bank is not None
                else None,
                "test_candidate_bank_sha256": self.eval_candidate_bank.sha256
                if self.eval_candidate_bank is not None
                else None,
            },
        }

    @staticmethod
    def _normalise_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "net" in checkpoint:
            state_dict = checkpoint["net"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint does not contain a state dictionary")
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
        return state_dict

    def _load_weight(self, path: str, *, allow_missing_ccrr: bool) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = self._normalise_state_dict(checkpoint)
        checkpoint_has_ccrr = any(key.startswith("ccrr.") for key in state_dict)
        baseline_initialization = allow_missing_ccrr and not checkpoint_has_ccrr
        if self.args.enable_ccrr and not allow_missing_ccrr and not checkpoint_has_ccrr:
            raise RuntimeError(
                "joint/test CCRR loading requires a complete head-only CCRR weight; "
                "a baseline-only state_dict is not a valid initializer"
            )
        self._validate_weight_configuration(
            checkpoint,
            path,
            baseline_initialization=baseline_initialization,
            validate_evaluation=self.mode == "test",
            validate_training=False,
        )
        if not allow_missing_ccrr:
            self._state_model().load_state_dict(state_dict, strict=True)
        else:
            incompatible = self._state_model().load_state_dict(state_dict, strict=False)
            missing_ccrr = [
                key for key in incompatible.missing_keys if key.startswith("ccrr.")
            ]
            invalid_missing = [
                key for key in incompatible.missing_keys if not key.startswith("ccrr.")
            ]
            if checkpoint_has_ccrr:
                invalid_missing.extend(missing_ccrr)
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "checkpoint is incompatible: missing={} unexpected={}".format(
                        invalid_missing, incompatible.unexpected_keys
                    )
                )
        self.parent_weight_sha256 = _file_sha256(path)
        provenance = checkpoint.get("provenance", {}) if isinstance(checkpoint, Mapping) else {}
        saved_baseline = provenance.get("baseline_weight_sha256")
        self.baseline_weight_sha256 = (
            str(saved_baseline)
            if saved_baseline
            else self.parent_weight_sha256
            if baseline_initialization
            else None
        )
        self._validate_candidate_bank_ancestry()

    def _resume_checkpoint(self, path: str) -> None:
        if not path:
            raise ValueError("--resume-path is required when --if-checkpoint true")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = self._normalise_state_dict(checkpoint)
        if self.args.enable_ccrr and not any(
            key.startswith("ccrr.") for key in state_dict
        ):
            raise RuntimeError("CCRR resume checkpoint does not contain CCRR parameters")
        self._validate_weight_configuration(
            checkpoint,
            path,
            baseline_initialization=False,
            validate_evaluation=True,
            validate_training=True,
        )
        self._state_model().load_state_dict(state_dict, strict=True)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if isinstance(checkpoint, dict):
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.best_iou = float(
                checkpoint.get("best_miou", checkpoint.get("iou", -1.0))
            )
            self.best_pd = float(checkpoint.get("best_pd", -1.0))
            rng_state = checkpoint.get("rng_state")
            if isinstance(rng_state, Mapping):
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"].cpu())
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(
                        [state.cpu() for state in rng_state["cuda"]]
                    )
            provenance = checkpoint.get("provenance", {})
            saved_baseline = provenance.get("baseline_weight_sha256")
            self.baseline_weight_sha256 = str(saved_baseline) if saved_baseline else None
        self.parent_weight_sha256 = _file_sha256(path)
        self._validate_candidate_bank_ancestry()
        self.save_folder = osp.dirname(path) or self.args.save_dir
        os.makedirs(self.save_folder, exist_ok=True)

    def _validate_candidate_bank_ancestry(self) -> None:
        banks = [
            ("train", self.train_candidate_bank),
            ("test", self.eval_candidate_bank),
        ]
        for split, bank in banks:
            if bank is None:
                continue
            bank_baseline = bank.metadata.get("weight_sha256")
            if not self.baseline_weight_sha256:
                raise RuntimeError(
                    f"cannot establish baseline ancestry for {split} candidate bank"
                )
            if bank_baseline != self.baseline_weight_sha256:
                raise RuntimeError(
                    f"{split} candidate bank was generated by baseline "
                    f"{bank_baseline}, but loaded CCRR lineage uses "
                    f"{self.baseline_weight_sha256}"
                )

    def _new_save_folder(self) -> str:
        dataset = Path(self.args.dataset_dir).name
        variant = "ccrr-" + self.args.ccrr_stage if self.args.enable_ccrr else "baseline"
        stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        folder = osp.join(self.args.save_dir, f"{dataset}-{variant}-{stamp}")
        if self.args.mode == "train" and not self.args.if_checkpoint:
            os.makedirs(folder, exist_ok=True)
        return folder

    def save_checkpoint(
        self, epoch: int, test_metrics: Mapping[str, Any] | None = None
    ) -> None:
        """Save the latest state independently of test-based best weights."""

        state_dict = self._state_model().state_dict()
        weight_artifact = self._artifact_payload(state_dict, epoch=epoch)
        torch.save(weight_artifact, osp.join(self.save_folder, "weight.pkl"))
        checkpoint = {
            **weight_artifact,
            "optimizer": self.optimizer.state_dict(),
            "best_miou": self.best_iou,
            "best_pd": self.best_pd,
            "iou": self.best_iou,
            "args": vars(self.args),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }
        if test_metrics is not None:
            checkpoint["test_metrics"] = dict(test_metrics)
        torch.save(checkpoint, osp.join(self.save_folder, "checkpoint.pkl"))

    def _configure_trainable_parameters(self) -> None:
        if self.mode != "train" or not self.args.enable_ccrr:
            return
        if self.args.ccrr_stage == "head_only":
            for name, parameter in self._state_model().named_parameters():
                parameter.requires_grad = name.startswith("ccrr.")
            return
        trainable_prefixes = ("ccrr.", "decoder_0.", "output_0.", "final.")
        for name, parameter in self._state_model().named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if not self.args.enable_ccrr:
            return Adagrad(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                lr=self.args.lr,
            )
        named_parameters = list(self._state_model().named_parameters())
        ccrr_parameters = [
            parameter
            for name, parameter in named_parameters
            if name.startswith("ccrr.") and parameter.requires_grad
        ]
        parameter_groups: list[dict[str, Any]] = [
            {"params": ccrr_parameters, "lr": self.args.ccrr_lr}
        ]
        other_parameters = [
            parameter
            for name, parameter in named_parameters
            if not name.startswith("ccrr.") and parameter.requires_grad
        ]
        if other_parameters:
            parameter_groups.append(
                {"params": other_parameters, "lr": self.args.backbone_lr}
            )
        return AdamW(parameter_groups, weight_decay=self.args.weight_decay)

    def _resolve_candidate_class_weights(self) -> list[float]:
        if self.args.candidate_class_weights is not None:
            return list(self.args.candidate_class_weights)
        class_names = ["target", "clutter"]
        if self.args.ccrr_num_classes == 3:
            class_names.append("uncertain")
        counts = {}
        if self.train_candidate_bank is not None:
            counts = self.train_candidate_bank.metadata.get("class_counts", {})
        frequencies = [float(counts.get(name, 0)) for name in class_names]
        if sum(value > 0 for value in frequencies) < 2:
            return [1.0] * self.args.ccrr_num_classes
        inverse = np.asarray(
            [1.0 / value if value > 0 else 0.0 for value in frequencies],
            dtype=np.float64,
        )
        inverse[inverse > 0] /= inverse[inverse > 0].mean()
        return inverse.tolist()

    def _candidate_score(self, candidates: Mapping[str, torch.Tensor]) -> torch.Tensor:
        key = {
            "coarse_peak": "coarse_peak_scores",
            "coarse_mean": "coarse_scores",
            "scale_peak": "peak_scores",
            "scale_mean": "scores",
        }[self.args.candidate_score]
        if key not in candidates:
            raise KeyError(f"candidate set does not contain score field {key!r}")
        return candidates[key]

    def _forward_batch(
        self,
        images: torch.Tensor,
        names: list[str],
        warm_flag: bool,
        candidate_bank: CandidateBank | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, torch.Tensor] | None]:
        if not self.args.enable_ccrr:
            return self.model(images, warm_flag=warm_flag), None
        candidates = (
            candidate_bank.get(names, self.device)
            if candidate_bank is not None
            else None
        )
        outputs = self.model(
            images,
            warm_flag=True,
            candidate_boxes=candidates,
            enable_ccrr=True,
            candidate_threshold=self.args.candidate_threshold,
            min_candidate_area=self.args.min_candidate_area,
            max_candidate_area=self.args.max_candidate_area or None,
        )
        return outputs, candidates or outputs["candidate_outputs"]

    def _label_candidates(
        self, candidates: Mapping[str, torch.Tensor], labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        masks = candidates.get("masks", candidates.get("candidate_masks"))
        if masks is None:
            raise KeyError("candidate set does not contain masks")
        label_input = {
            "masks": masks,
            "batch_indices": candidates["batch_indices"],
            "scores": self._candidate_score(candidates),
        }
        matching = match_candidates_to_gt(
            label_input,
            labels,
            positive_iou=self.args.positive_iou,
            hard_negative_threshold=self.args.hard_negative_threshold,
            center_distance=self.args.center_distance,
        )
        training_labels = matching["labels"].clone()
        if self.args.ccrr_num_classes == 2:
            training_labels[training_labels == UNCERTAIN_LABEL] = -1
        matching["training_labels"] = training_labels
        return matching

    def _detection_loss(
        self, outputs: Mapping[str, Any], labels: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        logits = outputs["coarse_logits"]
        losses = [self.loss_fun(logits, labels, self.warm_epoch, epoch)]
        for auxiliary in outputs["multi_scale_logits"]:
            if auxiliary.shape[-2:] == labels.shape[-2:]:
                scaled_labels = labels
            else:
                scale_h = labels.shape[-2] // auxiliary.shape[-2]
                scale_w = labels.shape[-1] // auxiliary.shape[-1]
                if (
                    scale_h == scale_w
                    and scale_h > 0
                    and labels.shape[-2] % auxiliary.shape[-2] == 0
                    and labels.shape[-1] % auxiliary.shape[-1] == 0
                ):
                    scaled_labels = F.max_pool2d(labels, scale_h, scale_h)
                else:
                    scaled_labels = F.adaptive_max_pool2d(
                        labels, auxiliary.shape[-2:]
                    )
            losses.append(
                self.loss_fun(auxiliary, scaled_labels, self.warm_epoch, epoch)
            )
        return torch.stack(losses).mean()

    def train(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("training loader is unavailable in test mode")
        self.model.train()
        if self.args.enable_ccrr:
            # Frozen BatchNorm buffers must stay frozen too.  Start from eval
            # mode, then opt only the modules trained in this stage back in.
            self.model.eval()
            state_model = self._state_model()
            if state_model.ccrr is None:
                raise RuntimeError("CCRR module is unavailable")
            state_model.ccrr.train()
            if self.args.ccrr_stage == "joint":
                state_model.decoder_0.train()
                state_model.output_0.train()
                state_model.final.train()
        meters = {name: AverageMeter() for name in (
            "total", "coarse", "refined", "classification", "calibration", "preservation"
        )}
        maximum = self.args.max_train_batches or len(self.train_loader)
        progress = tqdm(self.train_loader, total=min(len(self.train_loader), maximum))
        warm_flag = True if self.args.enable_ccrr else epoch > self.warm_epoch

        for step, batch in enumerate(progress):
            if self.args.max_train_batches and step >= self.args.max_train_batches:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["mask"].to(self.device, non_blocking=True)
            names = [str(name) for name in batch["name"]]
            outputs, candidates = self._forward_batch(
                images, names, warm_flag, self.train_candidate_bank
            )
            coarse_loss = self._detection_loss(outputs, labels, epoch)
            total_loss = coarse_loss
            terms: dict[str, torch.Tensor] = {}

            if self.args.enable_ccrr:
                if candidates is None or self.ccrr_loss is None:
                    raise RuntimeError("CCRR candidate/loss state is unavailable")
                matching = self._label_candidates(candidates, labels)
                refined_loss = self.loss_fun(
                    outputs["refined_logits"], labels, self.warm_epoch, epoch
                )
                terms = self.ccrr_loss(
                    outputs["candidate_outputs"],
                    matching["training_labels"],
                    coarse_logits=outputs["coarse_logits"],
                    refined_logits=outputs["refined_logits"],
                    candidate_masks=outputs["candidate_outputs"]["candidate_masks"],
                    candidate_batch_indices=outputs["candidate_outputs"]["batch_indices"],
                )
                total_loss = (
                    coarse_loss
                    + self.args.lambda_refined * refined_loss
                    + self.args.lambda_candidate * terms["classification"]
                    + self.args.lambda_calibration * terms["calibration"]
                    + self.args.lambda_preservation * terms["preservation"]
                )
                meters["refined"].update(refined_loss.item(), images.shape[0])
                for name in ("classification", "calibration", "preservation"):
                    meters[name].update(terms[name].item(), images.shape[0])

            self.optimizer.zero_grad(set_to_none=True)
            if total_loss.requires_grad:
                total_loss.backward()
                self.optimizer.step()
            meters["total"].update(total_loss.item(), images.shape[0])
            meters["coarse"].update(coarse_loss.item(), images.shape[0])
            progress.set_description(
                f"Epoch {epoch}, loss {meters['total'].avg:.4f}"
            )
        return {name: float(meter.avg) for name, meter in meters.items()}

    @staticmethod
    def _count_gt_instances(labels: torch.Tensor) -> int:
        count = 0
        for mask in labels[:, 0].detach().cpu().numpy() > 0:
            count += int(connected_components(mask, connectivity=2).max())
        return count

    def test(self, epoch: int) -> dict[str, Any]:
        self.model.eval()
        coarse_metrics = DetectionMetrics(self.args.base_size)
        refined_metrics = DetectionMetrics(self.args.base_size)
        froc_thresholds = np.unique(
            np.concatenate(
                (np.linspace(1.0, 0.0, self.args.froc_bins + 1), np.asarray([0.5]))
            )
        )[::-1]
        coarse_segmentation_froc = SegmentationFROC(
            froc_thresholds, center_distance=self.args.center_distance
        )
        refined_segmentation_froc = SegmentationFROC(
            froc_thresholds, center_distance=self.args.center_distance
        )
        probabilities: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
        paired_ccrr_probabilities: list[np.ndarray] = []
        raw_probabilities: list[np.ndarray] = []
        paired_labels: list[np.ndarray] = []
        detection_records: list[tuple[float, float, int, int, int]] = []
        total_gt_targets = 0
        num_images = 0
        maximum = self.args.max_test_batches or len(self.test_loader)
        progress = tqdm(self.test_loader, total=min(len(self.test_loader), maximum))

        with torch.inference_mode():
            for step, batch in enumerate(progress):
                if self.args.max_test_batches and step >= self.args.max_test_batches:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["mask"].to(self.device, non_blocking=True)
                names = [str(name) for name in batch["name"]]
                outputs, candidates = self._forward_batch(
                    images, names, warm_flag=True, candidate_bank=self.eval_candidate_bank
                )
                coarse_metrics.update(outputs["coarse_logits"], labels)
                refined_metrics.update(outputs["refined_logits"], labels)
                coarse_segmentation_froc.update(outputs["coarse_logits"], labels)
                refined_segmentation_froc.update(outputs["refined_logits"], labels)
                total_gt_targets += self._count_gt_instances(labels)

                if self.args.enable_ccrr:
                    if candidates is None:
                        raise RuntimeError("CCRR candidates are unavailable")
                    matching = self._label_candidates(candidates, labels)
                    candidate_outputs = outputs["candidate_outputs"]
                    probs = candidate_outputs["class_probs"].detach().cpu().numpy()
                    train_labels = matching["training_labels"].detach().cpu().numpy()
                    valid = train_labels != -1
                    if np.any(valid):
                        probabilities.append(probs[valid])
                        calibration_labels.append(train_labels[valid])

                    target_scores = candidate_outputs["target_scores"].detach().cpu().numpy()
                    raw_scores = self._candidate_score(candidates).detach().cpu().numpy()
                    raw_labels = matching["labels"].detach().cpu().numpy()
                    explicit = raw_labels != UNCERTAIN_LABEL
                    if np.any(explicit):
                        paired = probs[explicit, :2]
                        paired = paired / np.clip(
                            paired.sum(axis=1, keepdims=True), 1e-12, None
                        )
                        paired_ccrr_probabilities.append(paired)
                        raw_probabilities.append(
                            np.column_stack(
                                (raw_scores[explicit], 1.0 - raw_scores[explicit])
                            )
                        )
                        paired_labels.append(raw_labels[explicit])
                    matched_gt = matching["matched_gt_indices"].detach().cpu().numpy()
                    batch_indices = matching["batch_indices"].detach().cpu().numpy()
                    for index, score in enumerate(target_scores):
                        detection_records.append(
                            (
                                float(score),
                                float(raw_scores[index]),
                                num_images + int(batch_indices[index]),
                                int(matched_gt[index]),
                                int(raw_labels[index]),
                            )
                        )
                num_images += images.shape[0]
                progress.set_description(
                    f"Epoch {epoch}, IoU {refined_metrics.mean_iou:.4f}"
                )

        if num_images == 0:
            raise RuntimeError("no test images were evaluated")
        coarse_summary = coarse_metrics.get(num_images)
        refined_summary = refined_metrics.get(num_images)
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "num_images": num_images,
            "split": self.split_summary,
            "coarse": coarse_summary,
            "refined": refined_summary,
        }
        coarse_froc = coarse_segmentation_froc.get()
        refined_froc = refined_segmentation_froc.get()

        def install_correct_fixed_metrics(
            summary: dict[str, float], curve: Mapping[str, Any]
        ) -> None:
            half_index = int(np.flatnonzero(np.isclose(curve["thresholds"], 0.5))[0])
            summary["legacy_Pd"] = summary["Pd"]
            summary["legacy_Fa_per_million_pixels"] = summary[
                "Fa_per_million_pixels"
            ]
            summary["Pd"] = float(curve["Pd"][half_index])
            summary["FPPI"] = float(curve["FPPI"][half_index])
            summary["Fa_per_million_pixels"] = float(
                curve["Fa_per_million_pixels"][half_index]
            )

        install_correct_fixed_metrics(coarse_summary, coarse_froc)
        install_correct_fixed_metrics(refined_summary, refined_froc)

        def segmentation_curve_report(curve: Mapping[str, Any]) -> dict[str, Any]:
            return {
                **dict(curve),
                "FPPI_at_Pd_0.90": false_alarm_at_fixed_pd(
                    curve["Pd"], curve["FPPI"], 0.90
                ),
                "FPPI_at_Pd_0.95": false_alarm_at_fixed_pd(
                    curve["Pd"], curve["FPPI"], 0.95
                ),
                "Fa_at_Pd_0.90_per_million_pixels": false_alarm_at_fixed_pd(
                    curve["Pd"], curve["Fa_per_million_pixels"], 0.90
                ),
                "Fa_at_Pd_0.95_per_million_pixels": false_alarm_at_fixed_pd(
                    curve["Pd"], curve["Fa_per_million_pixels"], 0.95
                ),
            }

        metrics["segmentation_froc"] = {
            "matching": f"one-to-one centroid distance <= {self.args.center_distance}px",
            "coarse": segmentation_curve_report(coarse_froc),
            "refined": segmentation_curve_report(refined_froc),
        }

        if self.args.enable_ccrr:
            if probabilities:
                all_probabilities = np.concatenate(probabilities, axis=0)
                all_labels = np.concatenate(calibration_labels, axis=0)
                calibration = {
                    "num_labeled_candidates": int(all_labels.size),
                    "ECE": candidate_ece(
                        all_probabilities, all_labels, target_class=TARGET_LABEL
                    ),
                    "Brier": candidate_brier_score(all_probabilities, all_labels),
                    "NLL": candidate_nll(all_probabilities, all_labels),
                }
                selective_curve = risk_coverage_curve(
                    all_probabilities, all_labels
                )
            else:
                calibration = {
                    "num_labeled_candidates": 0,
                    "ECE": float("nan"),
                    "Brier": float("nan"),
                    "NLL": float("nan"),
                }
                selective_curve = risk_coverage_curve(
                    np.empty((0, self.args.ccrr_num_classes)),
                    np.empty((0,), dtype=np.int64),
                )
            calibration["AURC"] = selective_curve["aurc"]
            calibration["risk_coverage"] = {
                "thresholds": selective_curve["thresholds"],
                "coverage": selective_curve["coverage"],
                "risk": selective_curve["risk"],
            }

            if paired_labels:
                all_paired_labels = np.concatenate(paired_labels, axis=0)
                all_paired_ccrr = np.concatenate(paired_ccrr_probabilities, axis=0)
                all_raw_probabilities = np.concatenate(raw_probabilities, axis=0)

                def paired_calibration_report(
                    values: np.ndarray,
                ) -> dict[str, Any]:
                    risk_curve = risk_coverage_curve(values, all_paired_labels)
                    return {
                        "num_candidates": int(all_paired_labels.size),
                        "ECE": candidate_ece(
                            values, all_paired_labels, target_class=TARGET_LABEL
                        ),
                        "Brier": candidate_brier_score(values, all_paired_labels),
                        "NLL": candidate_nll(values, all_paired_labels),
                        "AURC": risk_curve["aurc"],
                        "risk_coverage": {
                            "thresholds": risk_curve["thresholds"],
                            "coverage": risk_curve["coverage"],
                            "risk": risk_curve["risk"],
                        },
                    }

                paired_calibration = {
                    "scope": "same explicit target/clutter candidates; uncertain excluded",
                    "raw": paired_calibration_report(all_raw_probabilities),
                    "ccrr": paired_calibration_report(all_paired_ccrr),
                }
            else:
                paired_calibration = {
                    "scope": "same explicit target/clutter candidates; uncertain excluded",
                    "raw": {"num_candidates": 0},
                    "ccrr": {"num_candidates": 0},
                }

            scores = np.asarray([record[0] for record in detection_records], dtype=np.float64)
            raw_scores = np.asarray(
                [record[1] for record in detection_records], dtype=np.float64
            )
            grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
            for index, (_, _, image_index, gt_index, label_id) in enumerate(detection_records):
                if label_id == TARGET_LABEL and gt_index >= 0:
                    grouped[(image_index, gt_index)].append(index)

            def true_positive_flags(candidate_scores: np.ndarray) -> np.ndarray:
                flags = np.zeros(candidate_scores.shape[0], dtype=bool)
                for candidate_indices in grouped.values():
                    best = max(
                        candidate_indices, key=lambda index: candidate_scores[index]
                    )
                    flags[best] = True
                return flags

            true_positive = true_positive_flags(scores)
            raw_true_positive = true_positive_flags(raw_scores)
            curve = fppi_froc(
                scores,
                true_positive,
                num_images=num_images,
                num_targets=total_gt_targets,
            )
            raw_curve = fppi_froc(
                raw_scores,
                raw_true_positive,
                num_images=num_images,
                num_targets=total_gt_targets,
            )
            at_half = fppi_froc(
                scores,
                true_positive,
                num_images=num_images,
                num_targets=total_gt_targets,
                thresholds=np.asarray([0.5]),
            )
            raw_at_half = fppi_froc(
                raw_scores,
                raw_true_positive,
                num_images=num_images,
                num_targets=total_gt_targets,
                thresholds=np.asarray([0.5]),
            )
            metrics["candidate"] = {
                **calibration,
                "calibration_scope": (
                    "CCRR head labels; binary MVP excludes uncertain candidates"
                ),
                "paired_calibration": paired_calibration,
                "num_candidates": int(scores.size),
                "num_gt_targets": int(total_gt_targets),
                "FPPI_at_reliability_0.5": float(at_half["fppi"][0]),
                "Pd_at_reliability_0.5": float(at_half["pd"][0]),
                "FPPI_at_Pd_0.90": fppi_at_fixed_pd(curve, 0.90),
                "FPPI_at_Pd_0.95": fppi_at_fixed_pd(curve, 0.95),
                "raw_FPPI_at_score_0.5": float(raw_at_half["fppi"][0]),
                "raw_Pd_at_score_0.5": float(raw_at_half["pd"][0]),
                "raw_FPPI_at_Pd_0.90": fppi_at_fixed_pd(raw_curve, 0.90),
                "raw_FPPI_at_Pd_0.95": fppi_at_fixed_pd(raw_curve, 0.95),
                "froc": {
                    "ccrr": {
                        "thresholds": curve["thresholds"],
                        "Pd": curve["pd"],
                        "FPPI": curve["fppi"],
                    },
                    "raw": {
                        "thresholds": raw_curve["thresholds"],
                        "Pd": raw_curve["pd"],
                        "FPPI": raw_curve["fppi"],
                    },
                },
            }

        if self.mode == "train":
            score_key = "refined" if self.args.enable_ccrr else "coarse"
            mean_iou = float(metrics[score_key]["mIoU"])
            detection_probability = float(metrics[score_key]["Pd"])
            if mean_iou > self.best_iou:
                self.best_iou = mean_iou
                torch.save(
                    self._artifact_payload(
                        self._state_model().state_dict(),
                        epoch=epoch,
                        selection_metric="mIoU",
                        selection_value=mean_iou,
                    ),
                    osp.join(self.save_folder, "best_miou.pkl"),
                )
            if detection_probability > self.best_pd:
                self.best_pd = detection_probability
                torch.save(
                    self._artifact_payload(
                        self._state_model().state_dict(),
                        epoch=epoch,
                        selection_metric="Pd",
                        selection_value=detection_probability,
                    ),
                    osp.join(self.save_folder, "best_pd.pkl"),
                )
            metrics["test_selection"] = {
                "best_mIoU": self.best_iou,
                "best_Pd": self.best_pd,
                "best_mIoU_weight": "best_miou.pkl",
                "best_Pd_weight": "best_pd.pkl",
            }

        metrics = _jsonable(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if self.args.metrics_output:
            output_path = Path(self.args.metrics_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if self.mode == "train":
            self.save_checkpoint(epoch, metrics)
            with open(osp.join(self.save_folder, "metrics.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        return metrics


def main(default_mode: str | None = None) -> None:
    args = parse_args(default_mode)
    seed_everything(args.seed)
    trainer = Trainer(args)
    if trainer.mode == "train":
        for epoch in range(trainer.start_epoch, args.epochs):
            training_metrics = trainer.train(epoch)
            training_record = _jsonable(
                {"epoch": epoch, "train": training_metrics}
            )
            print(json.dumps(training_record, indent=2))
            with open(
                osp.join(trainer.save_folder, "train_metrics.jsonl"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(training_record, ensure_ascii=False) + "\n")
            if should_run_scheduled_test(epoch, args.test_start_epoch):
                trainer.test(epoch)
            else:
                trainer.save_checkpoint(epoch)
    else:
        trainer.test(1)


if __name__ == "__main__":
    main()
