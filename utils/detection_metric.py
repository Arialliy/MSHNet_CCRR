"""Object-level FROC and pixel false-alarm curves for segmentation logits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from skimage.measure import label as connected_components
from skimage.measure import regionprops
import torch


def _normalise_maps(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W]")
    return tensor


def _maximum_centroid_matching(
    gt_centroids: list[np.ndarray],
    prediction_centroids: list[np.ndarray],
    maximum_distance: float,
) -> set[int]:
    """Return prediction indices in a maximum-cardinality bipartite match."""

    adjacency: list[list[int]] = []
    for gt_centroid in gt_centroids:
        neighbors = [
            prediction_index
            for prediction_index, prediction_centroid in enumerate(prediction_centroids)
            if np.linalg.norm(prediction_centroid - gt_centroid) <= maximum_distance
        ]
        neighbors.sort(
            key=lambda index: float(
                np.linalg.norm(prediction_centroids[index] - gt_centroid)
            )
        )
        adjacency.append(neighbors)

    prediction_to_gt: dict[int, int] = {}

    def augment(gt_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[gt_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous_gt = prediction_to_gt.get(prediction_index)
            if previous_gt is None or augment(previous_gt, visited):
                prediction_to_gt[prediction_index] = gt_index
                return True
        return False

    # Sparse GTs first reduces needless rematching and remains deterministic.
    for gt_index in sorted(range(len(gt_centroids)), key=lambda index: len(adjacency[index])):
        augment(gt_index, set())
    return set(prediction_to_gt)


class SegmentationFROC:
    """Accumulate Pd, FPPI and false-alarm pixel rate over score thresholds.

    A predicted connected component is matched at most once to a GT component
    when their centroids are within ``center_distance`` pixels.  Unmatched
    predicted components contribute one false positive and all of their pixels
    to ``Fa``.  This avoids the legacy evaluator's ambiguity when two
    components happen to have the same area.
    """

    def __init__(
        self,
        thresholds: Sequence[float] | np.ndarray | None = None,
        *,
        center_distance: float = 3.0,
    ) -> None:
        if thresholds is None:
            thresholds = np.linspace(1.0, 0.0, 21)
        threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if threshold_array.size == 0:
            raise ValueError("thresholds cannot be empty")
        if not np.all(np.isfinite(threshold_array)) or np.any(threshold_array < 0) or np.any(
            threshold_array > 1
        ):
            raise ValueError("thresholds must be finite probabilities in [0, 1]")
        if not np.isfinite(center_distance) or center_distance < 0:
            raise ValueError("center_distance must be finite and non-negative")
        self.thresholds = threshold_array
        self.center_distance = float(center_distance)
        self.reset()

    def reset(self) -> None:
        size = self.thresholds.size
        self.true_positives = np.zeros(size, dtype=np.int64)
        self.false_positives = np.zeros(size, dtype=np.int64)
        self.false_alarm_pixels = np.zeros(size, dtype=np.int64)
        self.num_targets = 0
        self.num_images = 0
        self.num_pixels = 0

    def update(
        self,
        predictions: Any,
        targets: Any,
        *,
        from_logits: bool = True,
    ) -> None:
        prediction_tensor = _normalise_maps(predictions, "predictions")
        target_tensor = _normalise_maps(targets, "targets")
        if prediction_tensor.shape != target_tensor.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if from_logits:
            prediction_tensor = prediction_tensor.sigmoid()
        if not torch.isfinite(prediction_tensor).all():
            raise ValueError("predictions must be finite")
        if (prediction_tensor < 0).any() or (prediction_tensor > 1).any():
            raise ValueError("probability predictions must lie in [0, 1]")

        probabilities = prediction_tensor[:, 0].detach().cpu().numpy()
        target_masks = target_tensor[:, 0].detach().cpu().numpy() > 0
        for probability, target_mask in zip(probabilities, target_masks):
            gt_regions = regionprops(connected_components(target_mask, connectivity=2))
            gt_centroids = [np.asarray(region.centroid) for region in gt_regions]
            self.num_targets += len(gt_regions)
            self.num_images += 1
            self.num_pixels += int(target_mask.size)

            for threshold_index, threshold in enumerate(self.thresholds):
                prediction_mask = probability > threshold
                prediction_regions = regionprops(
                    connected_components(prediction_mask, connectivity=2)
                )
                prediction_centroids = [
                    np.asarray(region.centroid) for region in prediction_regions
                ]
                matched_predictions = _maximum_centroid_matching(
                    gt_centroids,
                    prediction_centroids,
                    self.center_distance,
                )
                self.true_positives[threshold_index] += len(matched_predictions)
                unmatched = [
                    index
                    for index in range(len(prediction_regions))
                    if index not in matched_predictions
                ]
                self.false_positives[threshold_index] += len(unmatched)
                self.false_alarm_pixels[threshold_index] += sum(
                    int(prediction_regions[index].area) for index in unmatched
                )

    def get(self) -> dict[str, Any]:
        if self.num_images == 0:
            raise RuntimeError("no images have been added")
        if self.num_targets:
            pd = self.true_positives.astype(np.float64) / self.num_targets
        else:
            pd = np.full(self.thresholds.shape, np.nan, dtype=np.float64)
        return {
            "thresholds": self.thresholds.copy(),
            "Pd": pd,
            "FPPI": self.false_positives.astype(np.float64) / self.num_images,
            "Fa_per_million_pixels": (
                self.false_alarm_pixels.astype(np.float64) / self.num_pixels * 1_000_000
            ),
            "true_positives": self.true_positives.copy(),
            "false_positives": self.false_positives.copy(),
            "false_alarm_pixels": self.false_alarm_pixels.copy(),
            "num_images": self.num_images,
            "num_targets": self.num_targets,
            "num_pixels": self.num_pixels,
        }


__all__ = ["SegmentationFROC"]
