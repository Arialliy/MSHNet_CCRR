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
from skimage.measure import regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adagrad, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.MSHNet import MSHNet
from model.candidate_loss import (
    AsymmetricActionRiskLoss,
    CandidateBinaryFocalLoss,
    CandidateRankLoss,
    CCRRLoss,
    TargetQualityLoss,
)
from model.loss import AverageMeter, SLSIoULoss
from utils.candidate import (
    CLUTTER_LABEL,
    LABEL_NAMES,
    TARGET_LABEL,
    UNCERTAIN_LABEL,
    match_candidates_to_gt,
)
from utils.data import IRSTD_Dataset
from utils.detection_metric import (
    SegmentationFROC,
    match_prediction_components_to_gt,
    maximum_centroid_pairs,
)
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
WEIGHT_SCHEMA_VERSION = "mshnet-ccrr-weight/v2-safe"
SCA_WEIGHT_SCHEMA_VERSION = "mshnet-sca-ccrr-weight/v1"


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
    parser.add_argument(
        "--ccrr-version",
        choices=("v1_safe", "v1_threshold_aware", "v2_selective_component"),
        default="v1_safe",
        help=(
            "Keep v1_safe for exact historical reproduction; use "
            "v1_threshold_aware for the audited V1.1 action executor; use "
            "v2_selective_component for component-aligned quality-veto SCA-CCRR."
        ),
    )
    parser.add_argument("--candidate-bank", default="")
    parser.add_argument("--test-candidate-bank", default="")
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--min-context-size", type=float, default=15.0)
    parser.add_argument("--ccrr-dropout", type=float, default=0.3)
    parser.add_argument(
        "--max-delta", "--max-suppression", dest="max_delta", type=float, default=1.5
    )
    parser.add_argument("--gate-margin", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=0.1)
    parser.add_argument(
        "--clutter-action-threshold",
        "--action-threshold",
        dest="clutter_action_threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument("--action-temperature", type=float, default=0.05)
    parser.add_argument(
        "--max-action-suppression",
        type=float,
        default=0.0,
        help="V1.1 action cap in logits; 0 uses the audited exact/unbounded correction.",
    )
    parser.add_argument("--ccrr-num-classes", type=int, choices=(2, 3), default=2)
    parser.add_argument("--sca-roi-size", type=int, default=15)
    parser.add_argument("--sca-feature-channels", type=int, default=32)
    parser.add_argument("--risk-threshold", type=float, default=2.0)
    parser.add_argument("--quality-veto-threshold", type=float, default=0.20)
    parser.add_argument("--risk-alpha", type=float, default=1.0)
    parser.add_argument("--ccrr-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--lambda-refined", type=float, default=0.5)
    parser.add_argument("--lambda-candidate", type=float, default=1.0)
    parser.add_argument("--lambda-calibration", type=float, default=0.05)
    parser.add_argument(
        "--lambda-preservation",
        "--lambda-action",
        dest="lambda_preservation",
        type=float,
        default=1.0,
    )
    parser.add_argument("--clutter-margin", type=float, default=0.1)
    parser.add_argument("--easy-negative-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--hardness-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--clutter-focal-gamma", type=float, default=2.0)
    parser.add_argument("--clutter-positive-alpha", type=float, default=0.75)
    parser.add_argument("--lambda-clutter-cls", type=float, default=1.0)
    parser.add_argument("--lambda-quality", type=float, default=1.0)
    parser.add_argument("--lambda-action-risk", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--target-harm-weight", type=float, default=20.0)
    parser.add_argument("--missed-clutter-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--quality-iou-weight", type=float, default=0.5)
    parser.add_argument("--quality-center-sigma", type=float, default=3.0)
    parser.add_argument("--target-allowed-peak-drop", type=float, default=0.01)
    parser.add_argument(
        "--clutter-peak-ceiling",
        "--remove-threshold",
        dest="clutter_peak_ceiling",
        type=float,
        default=0.45,
    )
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
    parser.add_argument("--max-test-batches", type=int, default=0)
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


def _binary_candidate_metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Binary target/clutter metrics with clutter treated as positive."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {"num_candidates": 0}
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, prediction in zip(labels, predictions, strict=True):
        if truth in (TARGET_LABEL, CLUTTER_LABEL) and prediction in (
            TARGET_LABEL,
            CLUTTER_LABEL,
        ):
            confusion[int(truth), int(prediction)] += 1

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    target_recall = ratio(confusion[0, 0], confusion[0].sum())
    clutter_recall = ratio(confusion[1, 1], confusion[1].sum())
    clutter_precision = ratio(confusion[1, 1], confusion[:, 1].sum())
    clutter_f1 = ratio(
        2.0 * clutter_precision * clutter_recall,
        clutter_precision + clutter_recall,
    )

    positives = labels == CLUTTER_LABEL
    negatives = labels == TARGET_LABEL
    auroc = float("nan")
    auprc = float("nan")
    if positives.any() and negatives.any():
        scores = probabilities[:, CLUTTER_LABEL]
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_positive = positives[order]
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        )
        cumulative_tp = np.cumsum(sorted_positive)[group_ends].astype(np.float64)
        cumulative_fp = (
            np.cumsum(~sorted_positive)[group_ends].astype(np.float64)
        )
        recall = cumulative_tp / positives.sum()
        false_positive_rate = cumulative_fp / negatives.sum()
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
        auroc = float(
            np.trapezoid(
                np.r_[0.0, recall],
                np.r_[0.0, false_positive_rate],
            )
        )
        auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    recalls = np.asarray([target_recall, clutter_recall], dtype=np.float64)
    return {
        "num_candidates": int(labels.size),
        "confusion_matrix": confusion.tolist(),
        "target_recall": target_recall,
        "clutter_recall": clutter_recall,
        "clutter_precision": clutter_precision,
        "clutter_f1": clutter_f1,
        "balanced_accuracy": float(np.nanmean(recalls)),
        "AUROC_clutter": auroc,
        "AUPRC_clutter": auprc,
    }


def _delta_statistics(
    deltas: np.ndarray,
    labels: np.ndarray,
    max_suppression: float | None,
) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def one_class(mask: np.ndarray) -> dict[str, Any]:
        values = deltas[mask]
        if values.size == 0:
            return {"count": 0}
        summary = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_fraction": float(np.mean(values > 1e-8)),
            "negative_fraction": float(np.mean(values < -1e-8)),
        }
        summary["saturation_fraction"] = (
            float(
                np.mean(
                    np.isclose(np.abs(values), max_suppression, atol=1e-5)
                )
            )
            if max_suppression is not None
            else None
        )
        return summary

    target = one_class(labels == TARGET_LABEL)
    clutter = one_class(labels == CLUTTER_LABEL)
    return {
        "target": target,
        "clutter": clutter,
        "target_suppressed_fraction": target.get("negative_fraction", float("nan")),
        "clutter_unsuppressed_fraction": (
            1.0 - clutter["negative_fraction"]
            if "negative_fraction" in clutter
            else float("nan")
        ),
    }


def _detection_transition_counts(
    coarse_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    center_distance: float,
) -> dict[str, int]:
    totals = {
        "coarse_detected_refined_missed_targets": 0,
        "coarse_missed_refined_detected_targets": 0,
        "eliminated_fp_components": 0,
        "new_fp_components": 0,
    }
    coarse_binary = (coarse_logits[:, 0] > 0).detach().cpu().numpy()
    refined_binary = (refined_logits[:, 0] > 0).detach().cpu().numpy()
    truth_binary = (labels[:, 0] > 0).detach().cpu().numpy()
    for coarse_mask, refined_mask, truth_mask in zip(
        coarse_binary, refined_binary, truth_binary, strict=True
    ):
        gt_labels = connected_components(truth_mask, connectivity=2)
        coarse_labels = connected_components(coarse_mask, connectivity=2)
        refined_labels = connected_components(refined_mask, connectivity=2)

        def centroids(component_map: np.ndarray) -> list[np.ndarray]:
            return [
                np.argwhere(component_map == index).mean(axis=0)
                for index in range(1, int(component_map.max()) + 1)
            ]

        gt_centroids = centroids(gt_labels)
        coarse_centroids = centroids(coarse_labels)
        refined_centroids = centroids(refined_labels)
        coarse_pairs = maximum_centroid_pairs(
            gt_centroids, coarse_centroids, float(center_distance)
        )
        refined_pairs = maximum_centroid_pairs(
            gt_centroids, refined_centroids, float(center_distance)
        )
        coarse_detected = {gt_index for gt_index, _ in coarse_pairs}
        refined_detected = {gt_index for gt_index, _ in refined_pairs}
        totals["coarse_detected_refined_missed_targets"] += len(
            coarse_detected - refined_detected
        )
        totals["coarse_missed_refined_detected_targets"] += len(
            refined_detected - coarse_detected
        )

        coarse_matched_predictions = {
            prediction_index for _, prediction_index in coarse_pairs
        }
        refined_matched_predictions = {
            prediction_index for _, prediction_index in refined_pairs
        }
        coarse_fp = [
            prediction_index + 1
            for prediction_index in range(len(coarse_centroids))
            if prediction_index not in coarse_matched_predictions
        ]
        refined_fp = [
            prediction_index + 1
            for prediction_index in range(len(refined_centroids))
            if prediction_index not in refined_matched_predictions
        ]
        totals["eliminated_fp_components"] += sum(
            not any(
                np.any((coarse_labels == coarse_id) & (refined_labels == refined_id))
                for refined_id in refined_fp
            )
            for coarse_id in coarse_fp
        )
        totals["new_fp_components"] += sum(
            not any(
                np.any((refined_labels == refined_id) & (coarse_labels == coarse_id))
                for coarse_id in coarse_fp
            )
            for refined_id in refined_fp
        )
    return totals


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

    def __init__(self, image_size: int, center_distance: float = 3.0) -> None:
        self.pd_fa = PD_FA(1, 10, image_size)
        self.center_distance = float(center_distance)
        self.total_intersection = 0
        self.total_union = 0
        self.total_prediction_pixels = 0
        self.total_target_pixels = 0
        self.true_positives = 0
        self.false_positives = 0
        self.num_targets = 0
        self.matched_iou_sum = 0.0
        self.matched_iou_count = 0
        self.per_image_iou: list[float] = []

    def reset(self) -> None:
        self.pd_fa.reset()
        self.total_intersection = 0
        self.total_union = 0
        self.total_prediction_pixels = 0
        self.total_target_pixels = 0
        self.true_positives = 0
        self.false_positives = 0
        self.num_targets = 0
        self.matched_iou_sum = 0.0
        self.matched_iou_count = 0
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
        self.total_prediction_pixels += int(prediction.sum().item())
        self.total_target_pixels += int(target.sum().item())
        self.per_image_iou.append(intersection / union if union else 1.0)
        prediction_map = prediction[0, 0].detach().cpu().numpy()
        target_map = target[0, 0].detach().cpu().numpy()
        prediction_labels = connected_components(prediction_map, connectivity=2)
        target_labels = connected_components(target_map, connectivity=2)
        prediction_regions = regionprops(prediction_labels)
        target_regions = regionprops(target_labels)
        pairs = maximum_centroid_pairs(
            [np.asarray(region.centroid) for region in target_regions],
            [np.asarray(region.centroid) for region in prediction_regions],
            self.center_distance,
        )
        self.num_targets += len(target_regions)
        self.true_positives += len(pairs)
        self.false_positives += len(prediction_regions) - len(pairs)
        for target_index, prediction_index in pairs:
            target_component = target_labels == target_regions[target_index].label
            prediction_component = (
                prediction_labels == prediction_regions[prediction_index].label
            )
            component_intersection = int(
                np.count_nonzero(target_component & prediction_component)
            )
            component_union = int(
                np.count_nonzero(target_component | prediction_component)
            )
            self.matched_iou_sum += (
                component_intersection / component_union if component_union else 1.0
            )
            self.matched_iou_count += 1
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
        pixel_precision = (
            self.total_intersection / self.total_prediction_pixels
            if self.total_prediction_pixels
            else float("nan")
        )
        pixel_recall = (
            self.total_intersection / self.total_target_pixels
            if self.total_target_pixels
            else float("nan")
        )
        object_precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if self.true_positives + self.false_positives
            else float("nan")
        )
        object_recall = (
            self.true_positives / self.num_targets
            if self.num_targets
            else float("nan")
        )
        return {
            "mIoU": float(self.mean_iou),
            "nIoU": float(np.mean(self.per_image_iou)),
            "Pd": float(detection_probability[0]),
            "Fa_per_million_pixels": float(false_alarm[0] * 1_000_000),
            "pixel_precision": float(pixel_precision),
            "pixel_recall": float(pixel_recall),
            "pixel_f1": float(
                2.0 * pixel_precision * pixel_recall
                / (pixel_precision + pixel_recall)
                if pixel_precision + pixel_recall > 0
                else float("nan")
            ),
            "object_precision": float(object_precision),
            "object_f1": float(
                2.0 * object_precision * object_recall
                / (object_precision + object_recall)
                if object_precision + object_recall > 0
                else float("nan")
            ),
            "matched_target_iou": float(
                self.matched_iou_sum / self.matched_iou_count
                if self.matched_iou_count
                else float("nan")
            ),
        }


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        # Preserve compatibility with legacy tests and callers that build an
        # argparse-like namespace before the versioned CCRR CLI existed.
        if not hasattr(args, "ccrr_version"):
            args.ccrr_version = "v1_safe"
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
        test_bank_path = args.test_candidate_bank
        if args.mode == "test" and not test_bank_path:
            test_bank_path = args.candidate_bank
        self.test_candidate_bank = (
            CandidateBank(test_bank_path, args.base_size)
            if args.enable_ccrr and test_bank_path
            else None
        )

        self.train_loader: Data.DataLoader | None = None
        if args.mode == "train":
            # Only an offline bank requires canonical, augmentation-free
            # coordinates.  Online V1 candidates follow the augmented image.
            if args.enable_ccrr and args.candidate_bank:
                train_source = IRSTD_Dataset(args, mode="test", split="train")
            else:
                train_source = IRSTD_Dataset(args, mode="train", split="train")
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
            "evaluation_source": "official_test",
            "model_selection_source": "test" if args.mode == "train" else None,
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
        if self.test_candidate_bank is not None:
            self.test_candidate_bank.validate_contract(
                args,
                expected_split="test",
                expected_names=self._dataset_names(self.test_loader.dataset),
            )

        ccrr_config = None
        if args.enable_ccrr:
            if args.ccrr_version == "v2_selective_component":
                ccrr_config = {
                    "variant": "v2_selective_component",
                    "feature_channels": args.sca_feature_channels,
                    "num_scales": 4,
                    "roi_size": args.sca_roi_size,
                    "hidden_dim": args.hidden_dim,
                    "context_scale": args.context_scale,
                    "min_context_size": args.min_context_size,
                    "dropout": args.ccrr_dropout,
                    "risk_threshold": args.risk_threshold,
                    "quality_veto_threshold": args.quality_veto_threshold,
                    "risk_alpha": args.risk_alpha,
                    "action_temperature": args.action_temperature,
                    "remove_threshold": args.clutter_peak_ceiling,
                    "output_threshold": 0.5,
                }
            else:
                ccrr_config = {
                    "num_scales": 4,
                    "roi_size": args.roi_size,
                    "hidden_dim": args.hidden_dim,
                    "context_scale": args.context_scale,
                    "min_context_size": args.min_context_size,
                    "dropout": args.ccrr_dropout,
                    "max_delta": args.max_delta,
                    "gate_margin": args.gate_margin,
                    "gate_temperature": args.gate_temperature,
                    "num_classes": args.ccrr_num_classes,
                }
            if args.ccrr_version == "v1_threshold_aware":
                ccrr_config.update(
                    {
                        "rectifier": "threshold_aware",
                        "action_threshold": args.clutter_action_threshold,
                        "remove_threshold": args.clutter_peak_ceiling,
                        "action_temperature": args.action_temperature,
                        "max_suppression": (
                            args.max_action_suppression
                            if args.max_action_suppression > 0
                            else None
                        ),
                        "output_threshold": 0.5,
                    }
                )
        model: nn.Module = MSHNet(3, ccrr_config=ccrr_config)
        if args.multi_gpus:
            if args.enable_ccrr:
                raise ValueError("CCRR currently requires --multi-gpus false")
            if self.device.type == "cuda" and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self._configure_trainable_parameters()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.loss_fun = SLSIoULoss()
        self.ccrr_loss: CCRRLoss | None = None
        self.sca_clutter_loss: CandidateBinaryFocalLoss | None = None
        self.sca_quality_loss: TargetQualityLoss | None = None
        self.sca_risk_loss: AsymmetricActionRiskLoss | None = None
        self.sca_rank_loss: CandidateRankLoss | None = None
        if args.enable_ccrr and args.ccrr_version == "v2_selective_component":
            self.sca_clutter_loss = CandidateBinaryFocalLoss(
                gamma=args.clutter_focal_gamma,
                positive_alpha=args.clutter_positive_alpha,
                ignore_index=-1,
            )
            self.sca_quality_loss = TargetQualityLoss()
            self.sca_risk_loss = AsymmetricActionRiskLoss(
                target_harm_weight=args.target_harm_weight,
                missed_clutter_weight=args.missed_clutter_weight,
            )
            self.sca_rank_loss = CandidateRankLoss(margin=args.rank_margin)
            args.resolved_candidate_class_weights = []
        elif args.enable_ccrr:
            if args.ccrr_version == "v1_threshold_aware":
                class_weights = None
                resolved_class_weights: list[float] = []
            else:
                class_weights = self._resolve_candidate_class_weights()
                if len(class_weights) != args.ccrr_num_classes:
                    if args.ccrr_num_classes == 3 and len(class_weights) == 2:
                        class_weights.append(0.0)
                    else:
                        raise ValueError(
                            "--candidate-class-weights must have one value per CCRR class"
                        )
                resolved_class_weights = class_weights
            self.ccrr_loss = CCRRLoss(
                class_weights=class_weights,
                ignore_index=-1,
                clutter_margin=args.clutter_margin,
                label_smoothing=args.label_smoothing,
                allowed_target_peak_drop=args.target_allowed_peak_drop,
                clutter_peak_ceiling=args.clutter_peak_ceiling,
                classification_weight=1.0,
                calibration_weight=1.0,
                preservation_weight=1.0,
                classification_mode=(
                    "binary_focal"
                    if args.ccrr_version == "v1_threshold_aware"
                    else "cross_entropy"
                ),
                focal_gamma=args.clutter_focal_gamma,
                focal_positive_alpha=args.clutter_positive_alpha,
            )
            args.resolved_candidate_class_weights = resolved_class_weights

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
        if self.args.hidden_dim <= 0:
            raise ValueError("--hidden-dim must be positive")
        if self.args.sca_roi_size <= 0:
            raise ValueError("--sca-roi-size must be positive")
        if self.args.sca_feature_channels <= 0 or self.args.sca_feature_channels % 4:
            raise ValueError("--sca-feature-channels must be positive and divisible by 4")
        if self.args.min_context_size <= 0 or self.args.context_scale <= 0:
            raise ValueError("context sizes must be positive")
        if not 0.0 <= self.args.ccrr_dropout < 1.0:
            raise ValueError("--ccrr-dropout must lie in [0, 1)")
        if self.args.max_delta < 0:
            raise ValueError("--max-delta/--max-suppression must be non-negative")
        if self.args.gate_temperature <= 0:
            raise ValueError("--gate-temperature must be positive")
        if not 0.5 < self.args.clutter_action_threshold < 1.0:
            raise ValueError("--clutter-action-threshold must lie in (0.5, 1)")
        if self.args.action_temperature <= 0:
            raise ValueError("--action-temperature must be positive")
        if self.args.max_action_suppression < 0:
            raise ValueError("--max-action-suppression must be non-negative")
        if self.args.easy_negative_weight < 0 or self.args.hard_negative_weight < 0:
            raise ValueError("negative sample weights must be non-negative")
        if self.args.hard_negative_weight < self.args.easy_negative_weight:
            raise ValueError("--hard-negative-weight must be >= --easy-negative-weight")
        if self.args.hardness_gamma < 0:
            raise ValueError("--hardness-gamma must be non-negative")
        if not 0.0 <= self.args.label_smoothing <= 1.0:
            raise ValueError("--label-smoothing must lie in [0, 1]")
        if self.args.clutter_focal_gamma < 0:
            raise ValueError("--clutter-focal-gamma must be non-negative")
        if not 0.0 <= self.args.clutter_positive_alpha <= 1.0:
            raise ValueError("--clutter-positive-alpha must lie in [0, 1]")
        if not math.isfinite(self.args.risk_threshold):
            raise ValueError("--risk-threshold must be finite")
        if not 0.0 <= self.args.quality_veto_threshold <= 1.0:
            raise ValueError("--quality-veto-threshold must lie in [0,1]")
        if not math.isfinite(self.args.risk_alpha) or self.args.risk_alpha < 0:
            raise ValueError("--risk-alpha must be finite and non-negative")
        if not 0.0 <= self.args.quality_iou_weight <= 1.0:
            raise ValueError("--quality-iou-weight must lie in [0,1]")
        if not math.isfinite(self.args.quality_center_sigma) or self.args.quality_center_sigma <= 0:
            raise ValueError("--quality-center-sigma must be finite and positive")
        if self.args.target_harm_weight < 0 or self.args.missed_clutter_weight < 0:
            raise ValueError("SCA risk weights must be non-negative")
        if self.args.rank_margin < 0:
            raise ValueError("--rank-margin must be non-negative")
        if not 0.0 <= self.args.target_allowed_peak_drop <= 1.0:
            raise ValueError("--target-allowed-peak-drop must lie in [0, 1]")
        if not 0.0 <= self.args.clutter_peak_ceiling <= 1.0:
            raise ValueError("--clutter-peak-ceiling must lie in [0, 1]")
        if self.args.enable_ccrr and self.args.ccrr_version == "v1_threshold_aware":
            if self.args.ccrr_num_classes != 2:
                raise ValueError("v1_threshold_aware requires --ccrr-num-classes 2")
            if self.args.candidate_class_weights is not None:
                raise ValueError(
                    "v1_threshold_aware uses class-balanced focal loss; omit "
                    "--candidate-class-weights and tune --clutter-positive-alpha"
                )
            if not 0.0 < self.args.clutter_peak_ceiling < 0.5:
                raise ValueError(
                    "v1_threshold_aware requires --clutter-peak-ceiling in (0, 0.5)"
                )
            if self.args.candidate_bank or self.args.test_candidate_bank:
                raise ValueError(
                    "v1_threshold_aware uses online candidates; omit "
                    "--candidate-bank and --test-candidate-bank"
                )
        if self.args.enable_ccrr and self.args.ccrr_version == "v2_selective_component":
            if self.args.ccrr_num_classes != 2:
                raise ValueError("v2_selective_component requires --ccrr-num-classes 2")
            if self.args.candidate_bank or self.args.test_candidate_bank:
                raise ValueError(
                    "v2_selective_component uses online component-aligned candidates; "
                    "omit --candidate-bank and --test-candidate-bank"
                )
            if not math.isclose(self.args.candidate_threshold, 0.2, abs_tol=1e-12):
                raise ValueError("v2_selective_component fixes --candidate-threshold at 0.2")
            if not 0.0 < self.args.clutter_peak_ceiling < 0.5:
                raise ValueError(
                    "v2_selective_component requires --remove-threshold in (0,0.5)"
                )
        if (
            self.args.mode == "train"
            and self.args.enable_ccrr
            and self.args.ccrr_stage == "joint"
            and (
                self.args.candidate_bank
                or self.args.test_candidate_bank
            )
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
            "lambda_clutter_cls",
            "lambda_quality",
            "lambda_action_risk",
            "lambda_rank",
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

    def _weight_schema_version(self) -> str:
        if self.args.enable_ccrr and self.args.ccrr_version == "v2_selective_component":
            return SCA_WEIGHT_SCHEMA_VERSION
        return WEIGHT_SCHEMA_VERSION

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
            "schema_version": "mshnet-ccrr-inference/v2-safe",
            "model_variant": (
                "MSHNet+SCA-CCRR"
                if self.args.enable_ccrr
                and self.args.ccrr_version == "v2_selective_component"
                else "MSHNet+CCRR"
                if self.args.enable_ccrr
                else "MSHNet"
            ),
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
                    "roi_size": int(
                        self.args.sca_roi_size
                        if self.args.ccrr_version == "v2_selective_component"
                        else self.args.roi_size
                    ),
                    "hidden_dim": int(self.args.hidden_dim),
                    "context_scale": float(self.args.context_scale),
                    "min_context_size": float(self.args.min_context_size),
                    "ccrr_dropout": float(self.args.ccrr_dropout),
                    "max_delta": float(self.args.max_delta),
                    "rectifier": "suppression_only",
                    "gate_margin": float(self.args.gate_margin),
                    "gate_temperature": float(self.args.gate_temperature),
                    "ccrr_num_classes": int(self.args.ccrr_num_classes),
                }
            )
            if self.args.ccrr_version == "v1_threshold_aware":
                config.update(
                    {
                        "ccrr_version": "v1_threshold_aware",
                        "rectifier": "threshold_aware",
                        "clutter_action_threshold": float(
                            self.args.clutter_action_threshold
                        ),
                        "remove_threshold": float(
                            self.args.clutter_peak_ceiling
                        ),
                        "action_temperature": float(
                            self.args.action_temperature
                        ),
                        "max_action_suppression": (
                            float(self.args.max_action_suppression)
                            if self.args.max_action_suppression > 0
                            else None
                        ),
                        "output_probability_threshold": 0.5,
                    }
                )
            elif self.args.ccrr_version == "v2_selective_component":
                config.update(
                    {
                        "schema_version": "mshnet-sca-ccrr-inference/v1",
                        "ccrr_version": "v2_selective_component",
                        "ccrr_feature_channels": int(
                            self.args.sca_feature_channels
                        ),
                        "feature_sources": ["x_d0", "x_d1", "x_d2"],
                        "geometry_features": [
                            "log_area_fraction",
                            "normalized_width",
                            "normalized_height",
                            "compactness",
                            "coarse_peak_probability",
                            "coarse_mean_probability",
                            "scale_variance",
                            "core_ring_probability_contrast",
                        ],
                        "component_aligned": True,
                        "proposal_mask_use": "feature_encoding_only",
                        "action_mask_use": "label_suppression_and_evaluation",
                        "action_component_source": "coarse_probability_gt_0.5",
                        "rectifier": "component_aligned_threshold_aware",
                        "risk_threshold": float(self.args.risk_threshold),
                        "quality_veto_threshold": float(
                            self.args.quality_veto_threshold
                        ),
                        "risk_alpha": float(self.args.risk_alpha),
                        "action_temperature": float(
                            self.args.action_temperature
                        ),
                        "remove_threshold": float(
                            self.args.clutter_peak_ceiling
                        ),
                        "output_probability_threshold": 0.5,
                        "default_action": "keep",
                    }
                )
        return config

    def _evaluation_config(self) -> dict[str, Any]:
        bank = self.test_candidate_bank
        config: dict[str, Any] = {
            "schema_version": "mshnet-ccrr-evaluation/v2-safe",
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
            if self.args.ccrr_version in (
                "v1_threshold_aware",
                "v2_selective_component",
            ):
                test_source = self.test_loader.dataset
                if isinstance(test_source, Data.Subset):
                    test_source = test_source.dataset
                config.update(
                    {
                        "validation_source": None,
                        "development_selection_source": "official_test",
                        "max_test_batches": int(self.args.max_test_batches),
                        "raw_test_split_sha256": _file_sha256(
                            Path(test_source.list_dir)
                        ),
                    }
                )
            if self.args.ccrr_version == "v2_selective_component":
                config.update(
                    {
                        "schema_version": "mshnet-sca-ccrr-evaluation/v1",
                        "ccrr_version": "v2_selective_component",
                        "component_label_source": "exact_action_mask",
                        "component_matching": (
                            "8_connected_maximum_cardinality_centroid"
                        ),
                        "threshold_operator": ">",
                        "distance_operator": "<=",
                        "test_guided_development": True,
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
            "schema_version": "mshnet-ccrr-training/v2-safe",
            "batch_size": int(self.args.batch_size),
            "epochs": int(self.args.epochs),
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
                    "training_label_mode": "target_presence",
                    "easy_negative_weight": float(self.args.easy_negative_weight),
                    "hard_negative_weight": float(self.args.hard_negative_weight),
                    "hardness_gamma": float(self.args.hardness_gamma),
                    "label_smoothing": float(self.args.label_smoothing),
                    "target_allowed_peak_drop": float(
                        self.args.target_allowed_peak_drop
                    ),
                    "clutter_peak_ceiling": float(self.args.clutter_peak_ceiling),
                    "scheduler": self.args.scheduler,
                    "eta_min": float(self.args.eta_min),
                    "validation_split": None,
                    "test_start_epoch": int(self.args.test_start_epoch),
                }
            )
            if self.args.ccrr_version == "v1_threshold_aware":
                train_source = (
                    self.train_loader.dataset
                    if self.train_loader is not None
                    else None
                )
                if isinstance(train_source, Data.Subset):
                    train_source = train_source.dataset
                config.update(
                    {
                        "ccrr_version": "v1_threshold_aware",
                        "classification_loss": "class_balanced_binary_focal",
                        "label_smoothing": None,
                        "clutter_focal_gamma": float(
                            self.args.clutter_focal_gamma
                        ),
                        "clutter_positive_alpha": float(
                            self.args.clutter_positive_alpha
                        ),
                        "action_loss": "threshold_crossing_and_target_keep",
                        "action_probability_threshold": float(
                            self.args.clutter_action_threshold
                        ),
                        "max_train_batches": int(self.args.max_train_batches),
                        "num_workers": int(self.args.num_workers),
                        "raw_train_split_sha256": (
                            _file_sha256(Path(train_source.list_dir))
                            if train_source is not None
                            else None
                        ),
                    }
                )
            elif self.args.ccrr_version == "v2_selective_component":
                train_source = (
                    self.train_loader.dataset
                    if self.train_loader is not None
                    else None
                )
                if isinstance(train_source, Data.Subset):
                    train_source = train_source.dataset
                trainable_prefixes = ["ccrr.", "ccrr_feature_adapter."]
                if self.args.ccrr_stage == "joint":
                    trainable_prefixes.extend(
                        ["decoder_0.", "output_0.", "final."]
                    )
                config.update(
                    {
                        "schema_version": "mshnet-sca-ccrr-training/v1",
                        "ccrr_version": "v2_selective_component",
                        "training_label_mode": (
                            "exact_component_shared_evaluation_matching"
                        ),
                        "ambiguous_action": "keep_ignore",
                        "classification_loss": "class_balanced_binary_focal",
                        "label_smoothing": None,
                        "clutter_focal_gamma": float(
                            self.args.clutter_focal_gamma
                        ),
                        "clutter_positive_alpha": float(
                            self.args.clutter_positive_alpha
                        ),
                        "quality_loss": "smooth_l1",
                        "quality_iou_weight": float(
                            self.args.quality_iou_weight
                        ),
                        "quality_center_sigma": float(
                            self.args.quality_center_sigma
                        ),
                        "action_loss": "asymmetric_selective_risk",
                        "risk_threshold": float(self.args.risk_threshold),
                        "quality_veto_threshold": float(
                            self.args.quality_veto_threshold
                        ),
                        "risk_alpha": float(self.args.risk_alpha),
                        "action_temperature": float(
                            self.args.action_temperature
                        ),
                        "lambda_clutter_cls": float(
                            self.args.lambda_clutter_cls
                        ),
                        "lambda_quality": float(self.args.lambda_quality),
                        "lambda_action_risk": float(
                            self.args.lambda_action_risk
                        ),
                        "lambda_rank": float(self.args.lambda_rank),
                        "target_harm_weight": float(
                            self.args.target_harm_weight
                        ),
                        "missed_clutter_weight": float(
                            self.args.missed_clutter_weight
                        ),
                        "rank_margin": float(self.args.rank_margin),
                        "calibration_protocol": (
                            "pre_registered_fixed_risk_and_quality_thresholds"
                        ),
                        "development_selection_source": "official_test",
                        "max_train_batches": int(self.args.max_train_batches),
                        "num_workers": int(self.args.num_workers),
                        "raw_train_split_sha256": (
                            _file_sha256(Path(train_source.list_dir))
                            if train_source is not None
                            else None
                        ),
                        "trainable_prefixes": trainable_prefixes,
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
        expected_schema = (
            WEIGHT_SCHEMA_VERSION
            if baseline_initialization
            else self._weight_schema_version()
        )
        if checkpoint.get("checkpoint_schema") != expected_schema:
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
            "checkpoint_schema": self._weight_schema_version(),
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
                "test_candidate_bank_sha256": self.test_candidate_bank.sha256
                if self.test_candidate_bank is not None
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
            allowed_head_prefixes = ("ccrr.", "ccrr_feature_adapter.")
            missing_ccrr = [
                key
                for key in incompatible.missing_keys
                if key.startswith(allowed_head_prefixes)
            ]
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_head_prefixes)
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
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                "exact resume requires checkpoint.pkl; the supplied artifact "
                "is only a model state dictionary"
            )
        required_fields = {
            "checkpoint_schema",
            "net",
            "epoch",
            "optimizer",
            "scheduler",
            "best_miou",
            "best_pd",
            "args",
            "rng_state",
            "inference_config",
            "evaluation_config",
            "training_config",
            "provenance",
        }
        missing_fields = sorted(required_fields.difference(checkpoint))
        if missing_fields:
            raise RuntimeError(
                "exact resume requires the full checkpoint.pkl; missing fields: "
                + ", ".join(missing_fields)
            )
        if checkpoint.get("checkpoint_schema") != self._weight_schema_version():
            raise RuntimeError("exact resume checkpoint has an unsupported schema")
        if not isinstance(checkpoint["net"], Mapping):
            raise RuntimeError("exact resume checkpoint has an invalid net state")
        if not isinstance(checkpoint["optimizer"], Mapping):
            raise RuntimeError("exact resume checkpoint has an invalid optimizer state")
        if self.scheduler is not None and not isinstance(
            checkpoint["scheduler"], Mapping
        ):
            raise RuntimeError("exact resume checkpoint has no scheduler state")
        if self.scheduler is None and checkpoint["scheduler"] is not None:
            raise RuntimeError(
                "exact resume checkpoint contains a scheduler, but the current run does not"
            )
        rng_state = checkpoint["rng_state"]
        if not isinstance(rng_state, Mapping):
            raise RuntimeError("exact resume checkpoint has an invalid RNG state")
        missing_rng_fields = sorted(
            {"python", "numpy", "torch", "cuda"}.difference(rng_state)
        )
        if missing_rng_fields:
            raise RuntimeError(
                "exact resume checkpoint has an incomplete RNG state; missing fields: "
                + ", ".join(missing_rng_fields)
            )
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_iou = float(checkpoint["best_miou"])
        self.best_pd = float(checkpoint["best_pd"])
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng_state["cuda"]]
            )
        provenance = checkpoint["provenance"]
        saved_baseline = provenance.get("baseline_weight_sha256")
        self.baseline_weight_sha256 = str(saved_baseline) if saved_baseline else None
        self.parent_weight_sha256 = _file_sha256(path)
        self._validate_candidate_bank_ancestry()
        self.save_folder = osp.dirname(path) or self.args.save_dir
        os.makedirs(self.save_folder, exist_ok=True)

    def _validate_candidate_bank_ancestry(self) -> None:
        banks = [
            ("train", self.train_candidate_bank),
            ("test", self.test_candidate_bank),
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
        if self.args.enable_ccrr:
            version = (
                ""
                if self.args.ccrr_version == "v1_safe"
                else f"{self.args.ccrr_version}-"
            )
            variant = f"ccrr-{version}{self.args.ccrr_stage}"
        else:
            variant = "baseline"
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
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
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
            trainable_prefixes = ("ccrr.",)
            if self.args.ccrr_version == "v2_selective_component":
                trainable_prefixes += ("ccrr_feature_adapter.",)
            for name, parameter in self._state_model().named_parameters():
                parameter.requires_grad = name.startswith(trainable_prefixes)
            return
        trainable_prefixes = ("ccrr.", "decoder_0.", "output_0.", "final.")
        if self.args.ccrr_version == "v2_selective_component":
            trainable_prefixes += ("ccrr_feature_adapter.",)
        for name, parameter in self._state_model().named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if not self.args.enable_ccrr:
            return Adagrad(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                lr=self.args.lr,
            )
        named_parameters = list(self._state_model().named_parameters())
        head_prefixes = ("ccrr.",)
        if self.args.ccrr_version == "v2_selective_component":
            head_prefixes += ("ccrr_feature_adapter.",)
        ccrr_parameters = [
            parameter
            for name, parameter in named_parameters
            if name.startswith(head_prefixes) and parameter.requires_grad
        ]
        parameter_groups: list[dict[str, Any]] = [
            {"params": ccrr_parameters, "lr": self.args.ccrr_lr}
        ]
        other_parameters = [
            parameter
            for name, parameter in named_parameters
            if not name.startswith(head_prefixes) and parameter.requires_grad
        ]
        if other_parameters:
            parameter_groups.append(
                {"params": other_parameters, "lr": self.args.backbone_lr}
            )
        return AdamW(parameter_groups, weight_decay=self.args.weight_decay)

    def _build_scheduler(self) -> CosineAnnealingLR | None:
        if not self.args.enable_ccrr or self.args.scheduler == "none":
            return None
        return CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(self.args.epochs)),
            eta_min=float(self.args.eta_min),
        )

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
        if getattr(self.args, "ccrr_version", "v1_safe") == "v2_selective_component":
            key = {
                "coarse_peak": "coarse_peak_scores",
                "coarse_mean": "coarse_mean_scores",
                "scale_peak": "proposal_peak_scores",
                "scale_mean": "proposal_scores",
            }[self.args.candidate_score]
        else:
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

    def _label_sca_candidates(
        self, candidates: Mapping[str, torch.Tensor], labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Label exact action components with the public evaluation matcher."""

        action_masks = candidates.get("action_masks")
        if action_masks is None:
            raise KeyError("SCA candidate set does not contain action_masks")
        action_masks = action_masks.bool()
        batch_indices = candidates["batch_indices"].to(
            device=action_masks.device, dtype=torch.long
        )
        if action_masks.ndim != 3 or batch_indices.shape != (action_masks.shape[0],):
            raise ValueError("invalid SCA action mask or batch-index shape")
        if action_masks.shape[-2:] != labels.shape[-2:]:
            raise ValueError("SCA action masks and GT labels must share spatial size")

        count = action_masks.shape[0]
        device = action_masks.device
        strict_labels = torch.full(
            (count,), CLUTTER_LABEL, dtype=torch.long, device=device
        )
        matched_gt_indices = torch.full(
            (count,), -1, dtype=torch.long, device=device
        )
        matched_iou = torch.zeros((count,), dtype=torch.float32, device=device)
        centroid_distance = torch.full(
            (count,), float("inf"), dtype=torch.float32, device=device
        )
        quality_target = torch.zeros((count,), dtype=torch.float32, device=device)
        ambiguous = torch.zeros((count,), dtype=torch.bool, device=device)
        iou_weight = float(self.args.quality_iou_weight)
        sigma = float(self.args.quality_center_sigma)

        for batch_index in range(labels.shape[0]):
            positions = torch.nonzero(
                batch_indices == batch_index, as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            prediction_array = (
                action_masks[positions].detach().cpu().numpy().astype(bool)
            )
            target_array = (labels[batch_index, 0] > 0).detach().cpu().numpy()
            component_match = match_prediction_components_to_gt(
                prediction_array,
                target_array,
                center_distance=self.args.center_distance,
            )
            for local_index, position in enumerate(positions.tolist()):
                is_target = bool(component_match.is_tp_component[local_index])
                strict_labels[position] = TARGET_LABEL if is_target else CLUTTER_LABEL
                ambiguous[position] = bool(
                    component_match.ambiguous_keep[local_index]
                )
                gt_index = int(component_match.prediction_to_gt[local_index])
                matched_gt_indices[position] = gt_index
                if not is_target:
                    continue
                matched_iou[position] = float(
                    component_match.matched_component_iou[local_index]
                )
                centroid_distance[position] = float(
                    component_match.matched_centroid_distance[local_index]
                )
                prediction_mask = component_match.predictions.masks[local_index]
                prediction_centroid = component_match.predictions.centroids_yx[
                    local_index
                ]
                candidate_qualities = []
                for target_mask, target_centroid in zip(
                    component_match.targets.masks,
                    component_match.targets.centroids_yx,
                ):
                    intersection = int(
                        np.count_nonzero(prediction_mask & target_mask)
                    )
                    union = int(np.count_nonzero(prediction_mask | target_mask))
                    component_iou = intersection / union if union else 0.0
                    distance = float(
                        np.linalg.norm(prediction_centroid - target_centroid)
                    )
                    center_quality = math.exp(
                        -(distance * distance) / (2.0 * sigma * sigma)
                    )
                    candidate_qualities.append(
                        iou_weight * component_iou
                        + (1.0 - iou_weight) * center_quality
                    )
                quality_target[position] = max(candidate_qualities, default=0.0)

        training_labels = strict_labels.clone()
        training_labels[ambiguous] = -1
        is_target_component = (strict_labels == TARGET_LABEL) & ~ambiguous
        is_fp_component = (strict_labels == CLUTTER_LABEL) & ~ambiguous
        scores = self._candidate_score(candidates).clamp(0.0, 1.0)
        return {
            "labels": strict_labels,
            "strict_labels": strict_labels,
            "training_labels": training_labels,
            "batch_indices": batch_indices,
            "scores": scores,
            "sample_weights": torch.ones_like(scores),
            "matched_gt_indices": matched_gt_indices,
            "max_iou": matched_iou,
            "centroid_distance": centroid_distance,
            "center_match": strict_labels == TARGET_LABEL,
            "target_quality_gt": quality_target,
            "quality_valid": ~ambiguous,
            "ambiguous_keep": ambiguous,
            "is_target_component": is_target_component,
            "is_fp_component": is_fp_component,
        }

    def _label_candidates(
        self, candidates: Mapping[str, torch.Tensor], labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if getattr(self.args, "ccrr_version", "v1_safe") == "v2_selective_component":
            return self._label_sca_candidates(candidates, labels)
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
        strict_labels = matching["labels"].clone()
        has_target = (matching["max_iou"] > 0) | matching["center_match"]
        training_labels = torch.where(
            has_target,
            torch.full_like(strict_labels, TARGET_LABEL),
            torch.full_like(strict_labels, CLUTTER_LABEL),
        )
        scores = matching["scores"].clamp(0.0, 1.0)
        sample_weights = torch.ones_like(scores)
        clutter_mask = training_labels == CLUTTER_LABEL
        sample_weights[clutter_mask] = (
            self.args.easy_negative_weight
            + (
                self.args.hard_negative_weight
                - self.args.easy_negative_weight
            )
            * scores[clutter_mask].pow(self.args.hardness_gamma)
        )
        matching["strict_labels"] = strict_labels
        matching["training_labels"] = training_labels
        matching["sample_weights"] = sample_weights
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
            if self.args.ccrr_version == "v2_selective_component":
                if state_model.ccrr_feature_adapter is None:
                    raise RuntimeError("SCA feature adapter is unavailable")
                state_model.ccrr_feature_adapter.train()
            if self.args.ccrr_stage == "joint":
                state_model.decoder_0.train()
                state_model.output_0.train()
                state_model.final.train()
        meters = {
            name: AverageMeter()
            for name in (
                "total",
                "coarse",
                "refined",
                "classification",
                "calibration",
                "preservation",
                "quality",
                "action_risk",
                "target_harm",
                "missed_clutter",
                "rank",
            )
        }
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
                if candidates is None:
                    raise RuntimeError("CCRR candidate state is unavailable")
                matching = self._label_candidates(candidates, labels)
                refined_loss = self.loss_fun(
                    outputs["refined_logits"], labels, self.warm_epoch, epoch
                )
                candidate_outputs = outputs["candidate_outputs"]
                if self.args.ccrr_version == "v2_selective_component":
                    if any(
                        loss is None
                        for loss in (
                            self.sca_clutter_loss,
                            self.sca_quality_loss,
                            self.sca_risk_loss,
                            self.sca_rank_loss,
                        )
                    ):
                        raise RuntimeError("SCA loss state is unavailable")
                    classification = self.sca_clutter_loss(
                        candidate_outputs["class_logits"],
                        matching["training_labels"],
                    )
                    quality_valid = matching["quality_valid"]
                    quality = self.sca_quality_loss(
                        candidate_outputs["target_quality"][quality_valid],
                        matching["target_quality_gt"][quality_valid],
                    )
                    risk_terms = self.sca_risk_loss(
                        candidate_outputs["gates"],
                        matching["is_target_component"],
                        matching["is_fp_component"],
                    )
                    rank = self.sca_rank_loss(
                        candidate_outputs["risk_score"],
                        matching["is_target_component"],
                        matching["is_fp_component"],
                    )
                    terms = {
                        "classification": classification,
                        "quality": quality,
                        "action_risk": risk_terms["total"],
                        "target_harm": risk_terms["target_harm"],
                        "missed_clutter": risk_terms["missed_clutter"],
                        "rank": rank,
                    }
                    total_loss = (
                        coarse_loss
                        + self.args.lambda_refined * refined_loss
                        + self.args.lambda_clutter_cls * classification
                        + self.args.lambda_quality * quality
                        + self.args.lambda_action_risk * risk_terms["total"]
                        + self.args.lambda_rank * rank
                    )
                else:
                    if self.ccrr_loss is None:
                        raise RuntimeError("CCRR loss state is unavailable")
                    terms = self.ccrr_loss(
                        candidate_outputs,
                        matching["training_labels"],
                        coarse_logits=outputs["coarse_logits"],
                        refined_logits=outputs["refined_logits"],
                        candidate_masks=candidate_outputs["candidate_masks"],
                        candidate_batch_indices=candidate_outputs["batch_indices"],
                        sample_weights=matching["sample_weights"],
                    )
                    total_loss = (
                        coarse_loss
                        + self.args.lambda_refined * refined_loss
                        + self.args.lambda_candidate * terms["classification"]
                        + self.args.lambda_calibration * terms["calibration"]
                        + self.args.lambda_preservation * terms["preservation"]
                    )
                meters["refined"].update(refined_loss.item(), images.shape[0])
                for name in terms:
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
        split = "test"
        loader = self.test_loader
        candidate_bank = self.test_candidate_bank
        max_batches = self.args.max_test_batches
        select_weights = self.mode == "train"
        self.model.eval()
        coarse_metrics = DetectionMetrics(
            self.args.base_size, self.args.center_distance
        )
        refined_metrics = DetectionMetrics(
            self.args.base_size, self.args.center_distance
        )
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
        delta_values: list[np.ndarray] = []
        delta_labels: list[np.ndarray] = []
        candidate_records: list[dict[str, Any]] = []
        per_image_candidate_ids: dict[str, int] = defaultdict(int)
        detection_transitions: dict[str, int] = defaultdict(int)
        detection_records: list[tuple[float, float, int, int, int]] = []
        total_gt_targets = 0
        num_images = 0
        maximum = max_batches or len(loader)
        progress = tqdm(loader, total=min(len(loader), maximum))

        with torch.inference_mode():
            for step, batch in enumerate(progress):
                if max_batches and step >= max_batches:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["mask"].to(self.device, non_blocking=True)
                names = [str(name) for name in batch["name"]]
                outputs, candidates = self._forward_batch(
                    images, names, warm_flag=True, candidate_bank=candidate_bank
                )
                coarse_metrics.update(outputs["coarse_logits"], labels)
                refined_metrics.update(outputs["refined_logits"], labels)
                coarse_segmentation_froc.update(outputs["coarse_logits"], labels)
                refined_segmentation_froc.update(outputs["refined_logits"], labels)
                total_gt_targets += self._count_gt_instances(labels)
                transitions = _detection_transition_counts(
                    outputs["coarse_logits"],
                    outputs["refined_logits"],
                    labels,
                    self.args.center_distance,
                )
                for key, value in transitions.items():
                    detection_transitions[key] += int(value)

                if self.args.enable_ccrr:
                    if candidates is None:
                        raise RuntimeError("CCRR candidates are unavailable")
                    matching = self._label_candidates(candidates, labels)
                    candidate_outputs = outputs["candidate_outputs"]
                    probs = candidate_outputs["class_probs"].detach().cpu().numpy()
                    train_labels = matching["training_labels"].detach().cpu().numpy()
                    strict_labels = matching["strict_labels"].detach().cpu().numpy()
                    valid = train_labels != -1
                    if np.any(valid):
                        probabilities.append(probs[valid])
                        calibration_labels.append(train_labels[valid])

                    batch_indices_tensor = matching["batch_indices"]
                    candidate_masks_tensor = candidate_outputs[
                        "action_masks"
                        if self.args.ccrr_version == "v2_selective_component"
                        else "candidate_masks"
                    ].bool()
                    coarse_candidate_probabilities = outputs[
                        "coarse_logits"
                    ].sigmoid()[batch_indices_tensor, 0]
                    refined_candidate_probabilities = outputs[
                        "refined_logits"
                    ].sigmoid()[batch_indices_tensor, 0]
                    masked_coarse = coarse_candidate_probabilities.masked_fill(
                        ~candidate_masks_tensor, 0.0
                    )
                    masked_refined = refined_candidate_probabilities.masked_fill(
                        ~candidate_masks_tensor, 0.0
                    )
                    coarse_peaks = masked_coarse.flatten(1).amax(dim=1)
                    refined_peaks = masked_refined.flatten(1).amax(dim=1)
                    areas = candidate_masks_tensor.flatten(1).sum(dim=1).clamp_min(1)
                    coarse_means = masked_coarse.flatten(1).sum(dim=1) / areas
                    deltas_tensor = candidate_outputs["deltas"].detach()
                    gates_tensor = candidate_outputs.get("gates")
                    if gates_tensor is None:
                        gates_tensor = candidate_outputs.get("gate")
                    if gates_tensor is None:
                        gates_tensor = torch.zeros_like(deltas_tensor)
                    required_deltas_tensor = candidate_outputs.get(
                        "required_deltas", deltas_tensor
                    ).detach()
                    delta_values.append(deltas_tensor.cpu().numpy())
                    delta_labels.append(train_labels)

                    batch_indices_np = batch_indices_tensor.detach().cpu().numpy()
                    for index in range(len(train_labels)):
                        image_name = names[int(batch_indices_np[index])]
                        candidate_id = per_image_candidate_ids[image_name]
                        per_image_candidate_ids[image_name] += 1
                        coarse_detected = bool(coarse_peaks[index].item() > 0.5)
                        refined_detected = bool(refined_peaks[index].item() > 0.5)
                        record = {
                            "image": image_name,
                            "candidate_id": candidate_id,
                            "strict_label": LABEL_NAMES[int(strict_labels[index])],
                            "training_label": (
                                "keep_ignore"
                                if int(train_labels[index]) == -1
                                else LABEL_NAMES[int(train_labels[index])]
                            ),
                            "raw_peak": float(coarse_peaks[index].item()),
                            "raw_mean": float(coarse_means[index].item()),
                            "target_prob": float(probs[index, TARGET_LABEL]),
                            "clutter_prob": float(probs[index, CLUTTER_LABEL]),
                            "delta": float(deltas_tensor[index].item()),
                            "gate": float(gates_tensor[index].item()),
                            "coarse_detected": coarse_detected,
                            "refined_detected": refined_detected,
                        }
                        if self.args.ccrr_version == "v2_selective_component":
                            record.update(
                                {
                                    "proposal_id": int(
                                        candidate_outputs[
                                            "proposal_component_ids"
                                        ][index].item()
                                    ),
                                    "action_component_id": int(
                                        candidate_outputs[
                                            "action_component_local_ids"
                                        ][index].item()
                                    ),
                                    "proposal_to_component_iou": float(
                                        candidate_outputs[
                                            "proposal_to_action_iou"
                                        ][index].item()
                                    ),
                                    "proposal_is_fallback": bool(
                                        candidate_outputs[
                                            "proposal_is_fallback"
                                        ][index].item()
                                    ),
                                    "action_area": int(
                                        candidate_outputs["action_areas"][
                                            index
                                        ].item()
                                    ),
                                    "proposal_area": int(
                                        candidate_outputs["proposal_areas"][
                                            index
                                        ].item()
                                    ),
                                    "matched_gt_id": int(
                                        matching["matched_gt_indices"][
                                            index
                                        ].item()
                                    ),
                                    "target_quality": float(
                                        candidate_outputs["target_quality"][
                                            index
                                        ].item()
                                    ),
                                    "target_quality_gt": float(
                                        matching["target_quality_gt"][index].item()
                                    ),
                                    "risk_score": float(
                                        candidate_outputs["risk_score"][index].item()
                                    ),
                                    "quality_veto": bool(
                                        candidate_outputs["quality_veto"][
                                            index
                                        ].item()
                                    ),
                                    "ambiguous_keep": bool(
                                        matching["ambiguous_keep"][index].item()
                                    ),
                                }
                            )
                        if self.args.ccrr_version in (
                            "v1_threshold_aware",
                            "v2_selective_component",
                        ):
                            positive_support = (
                                candidate_masks_tensor[index]
                                & (coarse_candidate_probabilities[index] > 0.5)
                            )
                            refined_positive_support = (
                                candidate_masks_tensor[index]
                                & (refined_candidate_probabilities[index] > 0.5)
                            )
                            action_passed = bool(gates_tensor[index].item() >= 0.5)
                            record.update(
                                {
                                    "action_threshold_passed": action_passed,
                                    "peak_before": float(
                                        coarse_peaks[index].item()
                                    ),
                                    "peak_after": float(
                                        refined_peaks[index].item()
                                    ),
                                    "required_delta": float(
                                        required_deltas_tensor[index].item()
                                    ),
                                    "actual_delta": float(
                                        deltas_tensor[index].item()
                                    ),
                                    "crossed_output_threshold": bool(
                                        coarse_detected and not refined_detected
                                    ),
                                    "candidate_positive_support_eliminated": bool(
                                        action_passed
                                        and positive_support.any().item()
                                        and not refined_positive_support.any().item()
                                    ),
                                    "is_target": bool(
                                        strict_labels[index] == TARGET_LABEL
                                        if self.args.ccrr_version
                                        == "v2_selective_component"
                                        else train_labels[index]
                                        == TARGET_LABEL
                                    ),
                                    "is_clutter": bool(
                                        strict_labels[index] == CLUTTER_LABEL
                                        if self.args.ccrr_version
                                        == "v2_selective_component"
                                        else train_labels[index]
                                        == CLUTTER_LABEL
                                    ),
                                    "is_ambiguous": bool(
                                        self.args.ccrr_version
                                        == "v2_selective_component"
                                        and train_labels[index] == -1
                                    ),
                                    "component_eliminated": bool(
                                        coarse_detected and not refined_detected
                                    ),
                                    "target_component_eliminated": bool(
                                        (
                                            strict_labels[index]
                                            if self.args.ccrr_version
                                            == "v2_selective_component"
                                            else train_labels[index]
                                        )
                                        == TARGET_LABEL
                                        and coarse_detected
                                        and not refined_detected
                                    ),
                                }
                            )
                        candidate_records.append(record)

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
            raise RuntimeError(f"no {split} images were evaluated")
        coarse_summary = coarse_metrics.get(num_images)
        refined_summary = refined_metrics.get(num_images)
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "num_images": num_images,
            "split": {
                **self.split_summary,
                "evaluated_split": split,
                "used_for_model_selection": bool(select_weights),
            },
            "coarse": coarse_summary,
            "refined": refined_summary,
            "detection_transitions": dict(detection_transitions),
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
            true_positives = int(curve["true_positives"][half_index])
            false_positives = int(curve["false_positives"][half_index])
            false_negatives = int(curve["num_targets"]) - true_positives
            object_precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else float("nan")
            )
            summary.update(
                {
                    "true_positive_components": true_positives,
                    "false_positive_components": false_positives,
                    "false_negative_targets": false_negatives,
                    "false_alarm_pixels": int(
                        curve["false_alarm_pixels"][half_index]
                    ),
                    "object_precision": float(object_precision),
                    "object_f1": float(
                        2.0 * object_precision * summary["Pd"]
                        / (object_precision + summary["Pd"])
                        if object_precision + summary["Pd"] > 0
                        else float("nan")
                    ),
                }
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
                all_probabilities = np.empty(
                    (0, self.args.ccrr_num_classes), dtype=np.float64
                )
                all_labels = np.empty((0,), dtype=np.int64)
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
                    "all candidates; binary target-presence supervision"
                ),
                "paired_calibration": paired_calibration,
                "classification": _binary_candidate_metrics(
                    all_probabilities[:, :2], all_labels
                ),
                "delta_statistics": _delta_statistics(
                    np.concatenate(delta_values, axis=0)
                    if delta_values
                    else np.empty((0,), dtype=np.float64),
                    np.concatenate(delta_labels, axis=0)
                    if delta_labels
                    else np.empty((0,), dtype=np.int64),
                    (
                        None
                        if self.args.ccrr_version == "v2_selective_component"
                        or (
                            self.args.ccrr_version == "v1_threshold_aware"
                            and self.args.max_action_suppression == 0
                        )
                        else self.args.max_action_suppression
                        if self.args.ccrr_version == "v1_threshold_aware"
                        else self.args.max_delta
                    ),
                ),
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
            if self.args.ccrr_version in (
                "v1_threshold_aware",
                "v2_selective_component",
            ):
                actions = [
                    record
                    for record in candidate_records
                    if record.get("action_threshold_passed", False)
                ]
                coarse_positive_actions = [
                    record for record in actions if record["coarse_detected"]
                ]
                clutter_actions = [
                    record for record in actions if record["is_clutter"]
                ]
                target_actions = [
                    record for record in actions if record["is_target"]
                ]
                ambiguous_actions = [
                    record for record in actions if record["is_ambiguous"]
                ]
                crossed_actions = [
                    record
                    for record in coarse_positive_actions
                    if record["crossed_output_threshold"]
                ]
                support_eliminated = [
                    record
                    for record in coarse_positive_actions
                    if record["candidate_positive_support_eliminated"]
                ]
                eliminated_clutter = [
                    record
                    for record in clutter_actions
                    if record["component_eliminated"]
                ]
                eliminated_target = [
                    record
                    for record in target_actions
                    if record["target_component_eliminated"]
                ]

                def action_ratio(numerator: int, denominator: int) -> float:
                    return (
                        float(numerator / denominator)
                        if denominator
                        else float("nan")
                    )

                action_operating_point = {
                    "action_score": (
                        "selective_risk"
                        if self.args.ccrr_version == "v2_selective_component"
                        else "clutter_probability"
                    ),
                    "clutter_action_threshold": (
                        None
                        if self.args.ccrr_version == "v2_selective_component"
                        else float(self.args.clutter_action_threshold)
                    ),
                    "risk_threshold": (
                        float(self.args.risk_threshold)
                        if self.args.ccrr_version == "v2_selective_component"
                        else None
                    ),
                    "quality_veto_threshold": (
                        float(self.args.quality_veto_threshold)
                        if self.args.ccrr_version == "v2_selective_component"
                        else None
                    ),
                    "remove_threshold": float(
                        self.args.clutter_peak_ceiling
                    ),
                    "num_actions": len(actions),
                    "num_clutter_actions": len(clutter_actions),
                    "num_target_actions": len(target_actions),
                    "num_ambiguous_actions": len(ambiguous_actions),
                    "num_eliminated_fp_components": len(eliminated_clutter),
                    "num_eliminated_target_components": len(eliminated_target),
                    "fp_component_removal_efficiency": action_ratio(
                        len(eliminated_clutter), len(clutter_actions)
                    ),
                    "action_precision": action_ratio(
                        len(clutter_actions), len(actions)
                    ),
                    "num_actions_on_coarse_positive_candidates": len(
                        coarse_positive_actions
                    ),
                    "num_candidates_crossing_output_threshold": len(
                        crossed_actions
                    ),
                    "threshold_crossing_rate": action_ratio(
                        len(crossed_actions), len(coarse_positive_actions)
                    ),
                    "num_candidate_positive_supports_eliminated": len(
                        support_eliminated
                    ),
                    "support_removal_rate": action_ratio(
                        len(support_eliminated), len(coarse_positive_actions)
                    ),
                    "target_candidate_threshold_crossings": sum(
                        bool(record["crossed_output_threshold"])
                        for record in target_actions
                    ),
                    "component_transitions": dict(detection_transitions),
                }
                metrics["candidate"]["action_operating_point"] = (
                    action_operating_point
                )
                if self.args.ccrr_version == "v2_selective_component":
                    coarse_fppi = float(coarse_summary["FPPI"])
                    coarse_fa = float(
                        coarse_summary["Fa_per_million_pixels"]
                    )
                    relative_fppi_reduction = (
                        (coarse_fppi - float(refined_summary["FPPI"]))
                        / coarse_fppi
                        if coarse_fppi > 0
                        else 0.0
                    )
                    relative_fa_reduction = (
                        (
                            coarse_fa
                            - float(
                                refined_summary["Fa_per_million_pixels"]
                            )
                        )
                        / coarse_fa
                        if coarse_fa > 0
                        else 0.0
                    )
                    tolerance = 1e-12
                    sca_gate = {
                        "target_deletion_zero": len(eliminated_target) == 0,
                        "action_precision_at_least_0_90": (
                            len(actions) > 0
                            and len(clutter_actions) / len(actions) >= 0.90
                        ),
                        "fp_removal_efficiency_at_least_0_90": (
                            len(clutter_actions) > 0
                            and len(eliminated_clutter) / len(clutter_actions)
                            >= 0.90
                        ),
                        "fppi_relative_reduction_at_least_0_15": (
                            relative_fppi_reduction >= 0.15
                        ),
                        "fa_relative_reduction_at_least_0_10": (
                            relative_fa_reduction >= 0.10
                        ),
                        "pd_not_below_coarse": (
                            float(refined_summary["Pd"])
                            + tolerance
                            >= float(coarse_summary["Pd"])
                        ),
                        "miou_not_below_coarse": (
                            float(refined_summary["mIoU"])
                            + tolerance
                            >= float(coarse_summary["mIoU"])
                        ),
                        "niou_not_below_coarse": (
                            float(refined_summary["nIoU"])
                            + tolerance
                            >= float(coarse_summary["nIoU"])
                        ),
                    }
                    sca_gate["go"] = all(sca_gate.values())
                    metrics["candidate"]["sca_gate"] = {
                        **sca_gate,
                        "relative_fppi_reduction": relative_fppi_reduction,
                        "relative_fa_reduction": relative_fa_reduction,
                    }

        if self.mode == "train" and select_weights:
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

        if self.args.enable_ccrr:
            diagnostics_dir = Path(self.save_folder) / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            records_path = diagnostics_dir / f"{split}_candidates_epoch_{epoch:04d}.jsonl"
            records_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in candidate_records
                ),
                encoding="utf-8",
            )
            metrics["candidate_records"] = {
                "path": str(records_path),
                "count": len(candidate_records),
            }

        metrics = _jsonable(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if self.args.metrics_output and split == "test":
            output_path = Path(self.args.metrics_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if self.mode == "train":
            self.save_checkpoint(epoch, metrics)
            with open(
                osp.join(self.save_folder, "metrics.jsonl"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        else:
            with open(
                osp.join(self.save_folder, "test_metrics.jsonl"),
                "a",
                encoding="utf-8",
            ) as handle:
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
            if trainer.scheduler is not None:
                trainer.scheduler.step()
            if should_run_scheduled_test(epoch, args.test_start_epoch):
                trainer.test(epoch)
            else:
                trainer.save_checkpoint(epoch)
    else:
        trainer.test(1)


if __name__ == "__main__":
    main()
