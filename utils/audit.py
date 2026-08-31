"""Shared, deterministic metrics for CCRR upper-bound audits.

The project reports object detection with one-to-one centroid matching and
pixel false alarms from unmatched predicted components.  Audit scripts use
the same definition so their counterfactual results are directly comparable
with the training/test metrics at a fixed probability threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from utils.detection_metric import (
    component_detection_summary,
    extract_components,
    maximum_centroid_pairs,
)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def component_masks(binary_mask: Any) -> list[np.ndarray]:
    """Return deterministic 8-connected component masks for one image."""

    return list(extract_components(binary_mask).masks)


def component_centroid(mask: np.ndarray) -> np.ndarray:
    """Return a component centroid in ``(row, column)`` order."""

    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        raise ValueError("a component mask cannot be empty")
    return points.mean(axis=0)


def maximum_centroid_assignment(
    gt_components: list[np.ndarray],
    prediction_components: list[np.ndarray],
    maximum_distance: float,
) -> dict[int, int]:
    """Return a deterministic maximum-cardinality GT-to-prediction match."""

    gt_centroids = [component_centroid(mask) for mask in gt_components]
    prediction_centroids = [
        component_centroid(mask) for mask in prediction_components
    ]
    return dict(
        maximum_centroid_pairs(
            gt_centroids, prediction_centroids, float(maximum_distance)
        )
    )


def detection_snapshot(
    prediction_mask: Any,
    target_mask: Any,
    center_distance: float = 3.0,
) -> dict[str, Any]:
    """Describe one thresholded prediction using the project's Pd/FP rules."""

    prediction = np.asarray(prediction_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction_mask and target_mask must have shape [H,W]")
    if prediction.shape != target.shape:
        raise ValueError("prediction_mask and target_mask must have identical shapes")
    return component_detection_summary(
        prediction,
        target,
        center_distance=center_distance,
    )


@dataclass
class BinarySegmentationAccumulator:
    """Accumulate fixed-threshold segmentation and object metrics."""

    center_distance: float = 3.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.center_distance) or self.center_distance < 0:
            raise ValueError("center_distance must be finite and non-negative")
        self.total_intersection = 0
        self.total_union = 0
        self.image_ious: list[float] = []
        self.num_images = 0
        self.num_pixels = 0
        self.num_targets = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_alarm_pixels = 0

    def update(self, prediction_mask: Any, target_mask: Any) -> dict[str, Any]:
        prediction = np.asarray(prediction_mask, dtype=bool)
        target = np.asarray(target_mask, dtype=bool)
        snapshot = detection_snapshot(
            prediction, target, center_distance=self.center_distance
        )
        intersection = int(np.count_nonzero(prediction & target))
        union = int(np.count_nonzero(prediction | target))
        self.total_intersection += intersection
        self.total_union += union
        self.image_ious.append(intersection / union if union else 1.0)
        self.num_images += 1
        self.num_pixels += int(target.size)
        self.num_targets += len(snapshot["gt_components"])
        self.true_positives += len(snapshot["gt_to_prediction"])
        false_indices = snapshot["false_positive_indices"]
        self.false_positives += len(false_indices)
        self.false_alarm_pixels += sum(
            int(snapshot["prediction_components"][index].sum())
            for index in false_indices
        )
        return snapshot

    def get(self) -> dict[str, float | int]:
        return {
            "num_images": self.num_images,
            "num_targets": self.num_targets,
            "true_positives": self.true_positives,
            "false_positive_components": self.false_positives,
            "false_alarm_pixels": self.false_alarm_pixels,
            "mIoU": (
                self.total_intersection / self.total_union
                if self.total_union
                else 1.0
            ),
            "nIoU": float(np.mean(self.image_ious)) if self.image_ious else float("nan"),
            "Pd": (
                self.true_positives / self.num_targets
                if self.num_targets
                else float("nan")
            ),
            "FPPI": (
                self.false_positives / self.num_images
                if self.num_images
                else float("nan")
            ),
            "Fa_per_million_pixels": (
                self.false_alarm_pixels * 1_000_000 / self.num_pixels
                if self.num_pixels
                else float("nan")
            ),
        }


__all__ = [
    "BinarySegmentationAccumulator",
    "component_centroid",
    "component_masks",
    "detection_snapshot",
    "file_sha256",
    "maximum_centroid_assignment",
]
