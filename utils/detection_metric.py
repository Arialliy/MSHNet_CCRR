"""Object-level FROC and pixel false-alarm curves for segmentation logits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.measure import label as connected_components
from skimage.measure import regionprops
import torch


@dataclass(frozen=True)
class ComponentSet:
    """Deterministic 8-connected components for one binary image.

    Component ids are zero based and follow the scan order of
    :func:`skimage.measure.label`.  Bounding boxes use half-open ``yxyx``
    coordinates.
    """

    label_map: np.ndarray
    masks: tuple[np.ndarray, ...]
    centroids_yx: np.ndarray
    areas: np.ndarray
    bboxes_yxyx: np.ndarray

    def __len__(self) -> int:
        return len(self.masks)


@dataclass(frozen=True)
class ComponentMatch:
    """One-to-one component assignment shared by training and evaluation."""

    predictions: ComponentSet
    targets: ComponentSet
    pairs_gt_pred: tuple[tuple[int, int], ...]
    prediction_to_gt: np.ndarray
    gt_to_prediction: np.ndarray
    is_tp_component: np.ndarray
    is_fp_component: np.ndarray
    matched_centroid_distance: np.ndarray
    matched_component_iou: np.ndarray
    nearest_gt_id: np.ndarray
    nearest_gt_distance: np.ndarray
    max_iou_gt_id: np.ndarray
    max_gt_iou: np.ndarray
    ambiguous_keep: np.ndarray
    ambiguity_reasons: tuple[tuple[str, ...], ...]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_components(binary_mask: Any, *, connectivity: int = 8) -> ComponentSet:
    """Extract stable components from one 2-D binary map.

    The paper protocol fixes 8-connectivity.  ``connectivity`` remains an
    explicit argument so accidental use of a different definition fails
    loudly instead of silently changing Pd/FPPI.
    """

    if connectivity != 8:
        raise ValueError("the component protocol requires 8-connectivity")
    array = _as_numpy(binary_mask)
    if array.ndim != 2:
        raise ValueError("binary_mask must have shape [H,W]")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError("binary_mask must not contain NaN or infinite values")
    binary = array.astype(bool, copy=False)
    label_map = connected_components(binary, connectivity=2).astype(np.int32, copy=False)
    regions = regionprops(label_map)
    masks = tuple(label_map == region.label for region in regions)
    if regions:
        centroids = np.asarray([region.centroid for region in regions], dtype=np.float64)
        areas = np.asarray([int(region.area) for region in regions], dtype=np.int64)
        boxes = np.asarray([region.bbox for region in regions], dtype=np.int64)
    else:
        centroids = np.empty((0, 2), dtype=np.float64)
        areas = np.empty((0,), dtype=np.int64)
        boxes = np.empty((0, 4), dtype=np.int64)
    return ComponentSet(
        label_map=label_map,
        masks=masks,
        centroids_yx=centroids,
        areas=areas,
        bboxes_yxyx=boxes,
    )


def _component_set(value: ComponentSet | Any, name: str) -> ComponentSet:
    if isinstance(value, ComponentSet):
        return value
    array = _as_numpy(value)
    if array.ndim == 2:
        return extract_components(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a ComponentSet, [H,W], or [N,H,W]")

    number, height, width = array.shape
    masks: list[np.ndarray] = []
    label_map = np.zeros((height, width), dtype=np.int32)
    centroids = []
    areas = []
    boxes = []
    for component_id in range(number):
        mask = np.asarray(array[component_id], dtype=bool)
        extracted = extract_components(mask)
        if len(extracted) != 1:
            raise ValueError(f"each {name} item must contain exactly one component")
        if np.any((label_map != 0) & mask):
            raise ValueError(f"{name} components must not overlap")
        label_map[mask] = component_id + 1
        masks.append(mask)
        centroids.append(extracted.centroids_yx[0])
        areas.append(extracted.areas[0])
        boxes.append(extracted.bboxes_yxyx[0])
    return ComponentSet(
        label_map=label_map,
        masks=tuple(masks),
        centroids_yx=(
            np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
            if centroids
            else np.empty((0, 2), dtype=np.float64)
        ),
        areas=np.asarray(areas, dtype=np.int64),
        bboxes_yxyx=(
            np.asarray(boxes, dtype=np.int64).reshape(-1, 4)
            if boxes
            else np.empty((0, 4), dtype=np.int64)
        ),
    )


def _normalise_maps(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W]")
    return tensor


def maximum_centroid_pairs(
    gt_centroids: list[np.ndarray],
    prediction_centroids: list[np.ndarray],
    maximum_distance: float,
) -> list[tuple[int, int]]:
    """Return deterministic ``(gt, prediction)`` maximum-cardinality pairs."""

    if not np.isfinite(maximum_distance) or maximum_distance < 0:
        raise ValueError("maximum_distance must be finite and non-negative")

    adjacency: list[list[int]] = []
    for gt_centroid in gt_centroids:
        neighbors = [
            prediction_index
            for prediction_index, prediction_centroid in enumerate(prediction_centroids)
            if np.linalg.norm(prediction_centroid - gt_centroid) <= maximum_distance
        ]
        neighbors.sort(
            key=lambda index: (
                float(np.linalg.norm(prediction_centroids[index] - gt_centroid)),
                index,
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
    for gt_index in sorted(
        range(len(gt_centroids)), key=lambda index: (len(adjacency[index]), index)
    ):
        augment(gt_index, set())
    return sorted(
        (
            (gt_index, prediction_index)
            for prediction_index, gt_index in prediction_to_gt.items()
        ),
        key=lambda pair: pair[0],
    )


def _component_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def match_prediction_components_to_gt(
    prediction_masks: ComponentSet | Any,
    gt_masks: ComponentSet | Any,
    *,
    center_distance: float = 3.0,
    boundary_tolerance: float = 0.0,
) -> ComponentMatch:
    """Return the protocol's deterministic one-to-one TP/FP/FN assignment.

    Strict evaluation labels remain binary: every matched prediction is a TP
    component and every unmatched prediction is an FP component.  The
    ``ambiguous_keep`` field is only an optional training signal for merged,
    split, or distance-boundary cases; it never removes a component from the
    reported FPPI/Fa counts.
    """

    if not np.isfinite(center_distance) or center_distance < 0:
        raise ValueError("center_distance must be finite and non-negative")
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be finite and non-negative")
    predictions = _component_set(prediction_masks, "prediction_masks")
    targets = _component_set(gt_masks, "gt_masks")
    if predictions.label_map.shape != targets.label_map.shape:
        raise ValueError("prediction_masks and gt_masks must have identical shapes")

    pairs = tuple(
        maximum_centroid_pairs(
            list(targets.centroids_yx),
            list(predictions.centroids_yx),
            float(center_distance),
        )
    )
    number_of_predictions = len(predictions)
    number_of_targets = len(targets)
    prediction_to_gt = np.full(number_of_predictions, -1, dtype=np.int64)
    gt_to_prediction = np.full(number_of_targets, -1, dtype=np.int64)
    matched_distance = np.full(number_of_predictions, np.inf, dtype=np.float64)
    matched_iou = np.full(number_of_predictions, np.nan, dtype=np.float64)
    for gt_index, prediction_index in pairs:
        prediction_to_gt[prediction_index] = gt_index
        gt_to_prediction[gt_index] = prediction_index
        matched_distance[prediction_index] = float(
            np.linalg.norm(
                predictions.centroids_yx[prediction_index]
                - targets.centroids_yx[gt_index]
            )
        )
        matched_iou[prediction_index] = _component_iou(
            predictions.masks[prediction_index], targets.masks[gt_index]
        )

    nearest_gt_id = np.full(number_of_predictions, -1, dtype=np.int64)
    nearest_gt_distance = np.full(number_of_predictions, np.inf, dtype=np.float64)
    max_iou_gt_id = np.full(number_of_predictions, -1, dtype=np.int64)
    max_gt_iou = np.zeros(number_of_predictions, dtype=np.float64)
    distance_matrix = np.empty((number_of_predictions, number_of_targets), dtype=np.float64)
    iou_matrix = np.empty((number_of_predictions, number_of_targets), dtype=np.float64)
    for prediction_index in range(number_of_predictions):
        for gt_index in range(number_of_targets):
            distance_matrix[prediction_index, gt_index] = np.linalg.norm(
                predictions.centroids_yx[prediction_index]
                - targets.centroids_yx[gt_index]
            )
            iou_matrix[prediction_index, gt_index] = _component_iou(
                predictions.masks[prediction_index], targets.masks[gt_index]
            )
        if number_of_targets:
            nearest_gt_id[prediction_index] = int(
                np.argmin(distance_matrix[prediction_index])
            )
            nearest_gt_distance[prediction_index] = float(
                distance_matrix[prediction_index, nearest_gt_id[prediction_index]]
            )
            max_iou_gt_id[prediction_index] = int(
                np.argmax(iou_matrix[prediction_index])
            )
            max_gt_iou[prediction_index] = float(
                iou_matrix[prediction_index, max_iou_gt_id[prediction_index]]
            )

    eligible = distance_matrix <= float(center_distance)
    split_targets = eligible.sum(axis=0) > 1 if number_of_targets else np.zeros(0, dtype=bool)
    ambiguous = np.zeros(number_of_predictions, dtype=bool)
    reasons: list[tuple[str, ...]] = []
    for prediction_index in range(number_of_predictions):
        prediction_reasons = []
        if number_of_targets and int(eligible[prediction_index].sum()) > 1:
            prediction_reasons.append("merged_multiple_gt")
        if number_of_targets and np.any(eligible[prediction_index] & split_targets):
            prediction_reasons.append("split_multiple_predictions")
        if (
            number_of_targets
            and boundary_tolerance > 0
            and abs(nearest_gt_distance[prediction_index] - center_distance)
            <= boundary_tolerance
        ):
            prediction_reasons.append("distance_boundary")
        reasons.append(tuple(prediction_reasons))
        ambiguous[prediction_index] = bool(prediction_reasons)

    is_tp = prediction_to_gt >= 0
    return ComponentMatch(
        predictions=predictions,
        targets=targets,
        pairs_gt_pred=pairs,
        prediction_to_gt=prediction_to_gt,
        gt_to_prediction=gt_to_prediction,
        is_tp_component=is_tp,
        is_fp_component=~is_tp,
        matched_centroid_distance=matched_distance,
        matched_component_iou=matched_iou,
        nearest_gt_id=nearest_gt_id,
        nearest_gt_distance=nearest_gt_distance,
        max_iou_gt_id=max_iou_gt_id,
        max_gt_iou=max_gt_iou,
        ambiguous_keep=ambiguous,
        ambiguity_reasons=tuple(reasons),
    )


def component_detection_summary(
    prediction_or_match: ComponentMatch | ComponentSet | Any,
    target_mask: ComponentSet | Any | None = None,
    *,
    center_distance: float = 3.0,
    boundary_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Summarize one image using the same assignment as FROC and training."""

    if isinstance(prediction_or_match, ComponentMatch):
        if target_mask is not None:
            raise ValueError("target_mask must be omitted when a ComponentMatch is supplied")
        match = prediction_or_match
    else:
        if target_mask is None:
            raise ValueError("target_mask is required")
        match = match_prediction_components_to_gt(
            prediction_or_match,
            target_mask,
            center_distance=center_distance,
            boundary_tolerance=boundary_tolerance,
        )

    false_positive_indices = np.flatnonzero(match.is_fp_component).tolist()
    missed_gt_indices = np.flatnonzero(match.gt_to_prediction < 0).tolist()
    true_positives = int(match.is_tp_component.sum())
    false_positives = len(false_positive_indices)
    false_negatives = len(missed_gt_indices)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else float("nan")
    recall = true_positives / recall_denominator if recall_denominator else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    matched_ious = match.matched_component_iou[match.is_tp_component]
    return {
        # Backward-compatible audit fields.
        "gt_components": list(match.targets.masks),
        "prediction_components": list(match.predictions.masks),
        "gt_to_prediction": {
            int(gt_index): int(prediction_index)
            for gt_index, prediction_index in match.pairs_gt_pred
        },
        "prediction_to_gt": match.prediction_to_gt.copy(),
        "false_positive_indices": false_positive_indices,
        "missed_gt_indices": missed_gt_indices,
        # Explicit component-level evidence fields.
        "num_gt_components": len(match.targets),
        "num_prediction_components": len(match.predictions),
        "true_positive_components": true_positives,
        "false_positive_components": false_positives,
        "false_negative_targets": false_negatives,
        "false_alarm_pixels": int(
            match.predictions.areas[match.is_fp_component].sum()
        ),
        "matched_prediction_ids": np.flatnonzero(match.is_tp_component).tolist(),
        "fp_prediction_ids": false_positive_indices,
        "detected_gt_ids": np.flatnonzero(match.gt_to_prediction >= 0).tolist(),
        "missed_gt_ids": missed_gt_indices,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "mean_matched_component_iou": (
            float(matched_ious.mean()) if matched_ious.size else float("nan")
        ),
        "component_match": match,
    }


def _maximum_centroid_matching(
    gt_centroids: list[np.ndarray],
    prediction_centroids: list[np.ndarray],
    maximum_distance: float,
) -> set[int]:
    """Backward-compatible prediction-index view of centroid matching."""

    return {
        prediction_index
        for _, prediction_index in maximum_centroid_pairs(
            gt_centroids, prediction_centroids, maximum_distance
        )
    }


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
            target_components = extract_components(target_mask)
            self.num_targets += len(target_components)
            self.num_images += 1
            self.num_pixels += int(target_mask.size)

            for threshold_index, threshold in enumerate(self.thresholds):
                prediction_mask = probability > threshold
                match = match_prediction_components_to_gt(
                    prediction_mask,
                    target_components,
                    center_distance=self.center_distance,
                )
                summary = component_detection_summary(match)
                self.true_positives[threshold_index] += summary[
                    "true_positive_components"
                ]
                self.false_positives[threshold_index] += summary[
                    "false_positive_components"
                ]
                self.false_alarm_pixels[threshold_index] += summary[
                    "false_alarm_pixels"
                ]

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


__all__ = [
    "ComponentMatch",
    "ComponentSet",
    "SegmentationFROC",
    "component_detection_summary",
    "extract_components",
    "match_prediction_components_to_gt",
    "maximum_centroid_pairs",
]
