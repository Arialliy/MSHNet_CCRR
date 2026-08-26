"""Losses used to train the candidate reliability branch of CCRR.

The candidate classes follow one convention throughout the project:

``0 = target``, ``1 = clutter`` and ``2 = uncertain``.

The losses also support the binary MVP (target/clutter).  In that setting an
uncertain candidate can be assigned ``ignore_index`` (``-1`` by default), so
it contributes to neither the classification nor the calibration objective.
All losses return a differentiable zero for an empty candidate batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _zero_loss(reference: Tensor) -> Tensor:
    """Return a scalar zero that remains connected to ``reference``."""

    return reference.sum() * 0.0


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            "reduction must be one of {'none', 'mean', 'sum'}, "
            f"but got {reduction!r}"
        )


def _as_class_labels(labels: Tensor, num_classes: int) -> Tensor:
    """Normalize index or one-hot labels to a one-dimensional long tensor."""

    if labels.ndim == 2:
        if labels.shape[1] != num_classes:
            raise ValueError(
                "one-hot labels must have the same number of classes as logits: "
                f"got {labels.shape[1]} and {num_classes}"
            )
        labels = labels.argmax(dim=1)
    elif labels.ndim != 1:
        raise ValueError(
            f"candidate labels must have shape [N] or [N, C], got {tuple(labels.shape)}"
        )
    return labels.to(dtype=torch.long)


class CandidateClassificationLoss(nn.Module):
    """Weighted cross-entropy for target/clutter(/uncertain) candidates.

    Parameters
    ----------
    class_weights:
        Optional sequence with one weight per output class.  Both two-class
        and three-class reliability heads are supported.
    ignore_index:
        Label omitted from the loss.  ``-1`` is convenient for uncertain
        candidates while training the binary MVP.
    reduction:
        ``"mean"``, ``"sum"`` or ``"none"``.
    label_smoothing:
        Passed to :func:`torch.nn.functional.cross_entropy`.
    """

    def __init__(
        self,
        class_weights: Sequence[float] | Tensor | None = None,
        *,
        ignore_index: int = -1,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must lie in [0, 1]")

        weights = None
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() < 2:
                raise ValueError("class_weights must be a one-dimensional sequence")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("class_weights must be finite and non-negative")
        self.register_buffer("class_weights", weights)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, class_logits: Tensor, labels: Tensor) -> Tensor:
        if class_logits.ndim != 2:
            raise ValueError(
                f"class_logits must have shape [N, C], got {tuple(class_logits.shape)}"
            )
        num_candidates, num_classes = class_logits.shape
        if num_classes < 2:
            raise ValueError("class_logits must contain at least two classes")

        labels = _as_class_labels(labels, num_classes).to(class_logits.device)
        if labels.shape[0] != num_candidates:
            raise ValueError(
                "class_logits and labels disagree on candidate count: "
                f"{num_candidates} != {labels.shape[0]}"
            )

        valid = labels != self.ignore_index
        if valid.any():
            valid_labels = labels[valid]
            if (valid_labels < 0).any() or (valid_labels >= num_classes).any():
                raise ValueError(
                    f"labels must be in [0, {num_classes - 1}] or equal "
                    f"ignore_index={self.ignore_index}"
                )
        else:
            if self.reduction == "none":
                return class_logits.new_zeros((num_candidates,)) + class_logits.sum() * 0.0
            return _zero_loss(class_logits)

        weights = self.class_weights
        if weights is not None:
            if weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights has {weights.numel()} values, but logits have "
                    f"{num_classes} classes"
                )
            weights = weights.to(
                device=class_logits.device,
                dtype=class_logits.dtype,
            )

        if self.reduction == "none":
            return F.cross_entropy(
                class_logits,
                labels,
                weight=weights,
                ignore_index=self.ignore_index,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )

        if (
            self.reduction == "mean"
            and weights is not None
            and weights[labels[valid]].sum() <= 0
        ):
            # PyTorch divides by the selected target-weight sum for weighted
            # mean CE.  A batch containing only zero-weight classes would
            # otherwise produce NaN.
            return _zero_loss(class_logits)

        # Selecting valid rows avoids PyTorch's NaN result for an all-ignored
        # batch with mean reduction, while retaining the standard CE behavior.
        return F.cross_entropy(
            class_logits[valid],
            labels[valid],
            weight=weights,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class CandidateBrierLoss(nn.Module):
    """Multiclass Brier loss for candidate reliability calibration.

    For every candidate this computes the sum over classes specified in the
    design document, ``sum_k (p_k - y_k)^2``.  ``predictions`` are interpreted
    as logits by default; set ``from_logits=False`` when passing probabilities.
    """

    def __init__(
        self,
        *,
        from_logits: bool = True,
        ignore_index: int = -1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.from_logits = bool(from_logits)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        labels: Tensor,
        *,
        from_logits: bool | None = None,
    ) -> Tensor:
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions must have shape [N, C], got {tuple(predictions.shape)}"
            )
        num_candidates, num_classes = predictions.shape
        if num_classes < 2:
            raise ValueError("predictions must contain at least two classes")

        labels = _as_class_labels(labels, num_classes).to(predictions.device)
        if labels.shape[0] != num_candidates:
            raise ValueError(
                "predictions and labels disagree on candidate count: "
                f"{num_candidates} != {labels.shape[0]}"
            )

        valid = labels != self.ignore_index
        if valid.any():
            valid_labels = labels[valid]
            if (valid_labels < 0).any() or (valid_labels >= num_classes).any():
                raise ValueError(
                    f"labels must be in [0, {num_classes - 1}] or equal "
                    f"ignore_index={self.ignore_index}"
                )
        else:
            if self.reduction == "none":
                return predictions.new_zeros((num_candidates,)) + predictions.sum() * 0.0
            return _zero_loss(predictions)

        use_logits = self.from_logits if from_logits is None else bool(from_logits)
        probabilities = predictions.softmax(dim=1) if use_logits else predictions
        if not use_logits:
            if not torch.isfinite(probabilities).all():
                raise ValueError("candidate probabilities must be finite")
            if (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError("candidate probabilities must lie in [0, 1]")

        safe_labels = labels.clamp(min=0)
        target = F.one_hot(safe_labels, num_classes=num_classes).to(probabilities.dtype)
        per_candidate = (probabilities - target).square().sum(dim=1)
        per_candidate = per_candidate.masked_fill(~valid, 0.0)

        if self.reduction == "none":
            return per_candidate
        if self.reduction == "sum":
            return per_candidate.sum()
        return per_candidate[valid].mean()


class RectificationPreservationLoss(nn.Module):
    """Keep target responses while suppressing clutter after rectification.

    Target candidates incur ``relu(p_coarse - p_refined)``.  Clutter
    candidates incur ``relu(p_refined - p_coarse + margin)``.  Uncertain or
    ignored candidates are omitted.  Logits can be supplied as ``[B, 1, H,
    W]``, ``[B, H, W]`` or, for one image, ``[H, W]``.
    """

    def __init__(
        self,
        margin: float = 0.1,
        *,
        target_label: int = 0,
        clutter_label: int = 1,
        ignore_index: int = -1,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if target_label == clutter_label:
            raise ValueError("target_label and clutter_label must differ")
        self.margin = float(margin)
        self.target_label = int(target_label)
        self.clutter_label = int(clutter_label)
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)

    @staticmethod
    def _normalize_maps(maps: Tensor, name: str) -> Tensor:
        if maps.ndim == 2:
            maps = maps.unsqueeze(0).unsqueeze(0)
        elif maps.ndim == 3:
            maps = maps.unsqueeze(1)
        elif maps.ndim != 4:
            raise ValueError(f"{name} must have 2, 3 or 4 dimensions")
        if maps.shape[1] != 1:
            raise ValueError(f"{name} must have a singleton channel dimension")
        return maps

    @staticmethod
    def _normalize_masks(candidate_masks: Tensor) -> Tensor:
        if candidate_masks.ndim == 4 and candidate_masks.shape[1] == 1:
            candidate_masks = candidate_masks[:, 0]
        if candidate_masks.ndim != 3:
            raise ValueError(
                "candidate_masks must have shape [N, H, W] or [N, 1, H, W]"
            )
        return candidate_masks

    def forward(
        self,
        coarse_logits: Tensor,
        refined_logits: Tensor,
        candidate_masks: Tensor,
        labels: Tensor,
        candidate_batch_indices: Tensor | None = None,
    ) -> Tensor:
        coarse = self._normalize_maps(coarse_logits, "coarse_logits")
        refined = self._normalize_maps(refined_logits, "refined_logits")
        masks = self._normalize_masks(candidate_masks)

        if coarse.shape != refined.shape:
            raise ValueError(
                "coarse_logits and refined_logits must have identical shapes, got "
                f"{tuple(coarse.shape)} and {tuple(refined.shape)}"
            )
        if masks.shape[-2:] != coarse.shape[-2:]:
            raise ValueError(
                "candidate masks and prediction maps must have the same spatial size"
            )

        num_candidates = masks.shape[0]
        label_classes = (
            labels.shape[1]
            if labels.ndim == 2
            else max(self.target_label, self.clutter_label) + 1
        )
        labels = _as_class_labels(labels, label_classes)
        labels = labels.to(device=coarse.device)
        if labels.shape[0] != num_candidates:
            raise ValueError(
                "candidate_masks and labels disagree on candidate count: "
                f"{num_candidates} != {labels.shape[0]}"
            )

        if num_candidates == 0:
            return _zero_loss(coarse) + _zero_loss(refined)

        batch_size = coarse.shape[0]
        if candidate_batch_indices is None:
            if batch_size == 1:
                batch_indices = torch.zeros(
                    num_candidates, dtype=torch.long, device=coarse.device
                )
            elif batch_size == num_candidates:
                batch_indices = torch.arange(batch_size, device=coarse.device)
            else:
                raise ValueError(
                    "candidate_batch_indices is required when a batch contains "
                    "multiple images and a non-matching number of candidates"
                )
        else:
            batch_indices = torch.as_tensor(
                candidate_batch_indices, dtype=torch.long, device=coarse.device
            ).reshape(-1)
            if batch_indices.numel() != num_candidates:
                raise ValueError(
                    "candidate_batch_indices must contain one entry per candidate"
                )
            if (batch_indices < 0).any() or (batch_indices >= batch_size).any():
                raise ValueError("candidate_batch_indices contains an invalid batch index")

        masks = masks.to(device=coarse.device, dtype=coarse.dtype)
        if (masks < 0).any():
            raise ValueError("candidate_masks cannot contain negative weights")
        mask_area = masks.sum(dim=(1, 2))
        nonempty = mask_area > 0

        coarse_values = coarse[batch_indices, 0]
        refined_values = refined[batch_indices, 0]
        if self.from_logits:
            coarse_values = coarse_values.sigmoid()
            refined_values = refined_values.sigmoid()

        denominator = mask_area.clamp_min(torch.finfo(coarse.dtype).eps)
        coarse_mean = (coarse_values * masks).sum(dim=(1, 2)) / denominator
        refined_mean = (refined_values * masks).sum(dim=(1, 2)) / denominator

        valid = labels != self.ignore_index
        target_mask = valid & nonempty & (labels == self.target_label)
        clutter_mask = valid & nonempty & (labels == self.clutter_label)

        zero = (coarse_mean.sum() + refined_mean.sum()) * 0.0
        target_loss = zero
        if target_mask.any():
            target_loss = F.relu(
                coarse_mean[target_mask] - refined_mean[target_mask]
            ).mean()

        clutter_loss = zero
        if clutter_mask.any():
            clutter_loss = F.relu(
                refined_mean[clutter_mask]
                - coarse_mean[clutter_mask]
                + self.margin
            ).mean()

        return target_loss + clutter_loss


class CCRRLoss(nn.Module):
    """Convenience wrapper returning all candidate-level CCRR loss terms.

    The returned mapping contains ``classification``, ``calibration``,
    ``preservation`` and their weighted ``total``.  Candidate masks and batch
    indices may be passed explicitly or read from ``candidate_outputs``.
    """

    def __init__(
        self,
        class_weights: Sequence[float] | Tensor | None = None,
        *,
        ignore_index: int = -1,
        clutter_margin: float = 0.1,
        classification_weight: float = 1.0,
        calibration_weight: float = 1.0,
        preservation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("classification_weight", classification_weight),
            ("calibration_weight", calibration_weight),
            ("preservation_weight", preservation_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.classification = CandidateClassificationLoss(
            class_weights,
            ignore_index=ignore_index,
        )
        self.calibration = CandidateBrierLoss(
            from_logits=True,
            ignore_index=ignore_index,
        )
        self.preservation = RectificationPreservationLoss(
            margin=clutter_margin,
            ignore_index=ignore_index,
        )
        self.classification_weight = float(classification_weight)
        self.calibration_weight = float(calibration_weight)
        self.preservation_weight = float(preservation_weight)

    def forward(
        self,
        candidate_outputs: Mapping[str, Any] | Tensor,
        candidate_labels: Tensor | Mapping[str, Any],
        coarse_logits: Tensor | None = None,
        refined_logits: Tensor | None = None,
        candidate_masks: Tensor | None = None,
        candidate_batch_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(candidate_labels, Mapping):
            if "labels" not in candidate_labels:
                raise KeyError("candidate_labels mapping must contain 'labels'")
            candidate_labels = candidate_labels["labels"]
        if not isinstance(candidate_labels, Tensor):
            raise TypeError("candidate_labels must be a tensor or a mapping containing one")

        if isinstance(candidate_outputs, Tensor):
            class_logits = candidate_outputs
            outputs: Mapping[str, Any] = {}
        elif isinstance(candidate_outputs, Mapping):
            outputs = candidate_outputs
            if "class_logits" not in outputs:
                raise KeyError("candidate_outputs must contain 'class_logits'")
            class_logits = outputs["class_logits"]
            if not isinstance(class_logits, Tensor):
                raise TypeError("candidate_outputs['class_logits'] must be a tensor")
        else:
            raise TypeError("candidate_outputs must be a tensor or mapping")

        classification = self.classification(class_logits, candidate_labels)
        calibration = self.calibration(class_logits, candidate_labels)

        if candidate_masks is None:
            candidate_masks = outputs.get("candidate_masks")
        if candidate_batch_indices is None:
            candidate_batch_indices = outputs.get("batch_indices")
            if candidate_batch_indices is None and isinstance(outputs.get("boxes"), Tensor):
                boxes = outputs["boxes"]
                if boxes.ndim == 2 and boxes.shape[1] == 5:
                    candidate_batch_indices = boxes[:, 0].to(dtype=torch.long)

        preservation = _zero_loss(class_logits)
        preservation_inputs = (coarse_logits, refined_logits, candidate_masks)
        if all(value is not None for value in preservation_inputs):
            preservation = self.preservation(
                coarse_logits,  # type: ignore[arg-type]
                refined_logits,  # type: ignore[arg-type]
                candidate_masks,  # type: ignore[arg-type]
                candidate_labels,
                candidate_batch_indices,
            )
        elif any(value is not None for value in preservation_inputs):
            raise ValueError(
                "coarse_logits, refined_logits and candidate_masks must either all "
                "be provided or all be omitted"
            )

        total = (
            self.classification_weight * classification
            + self.calibration_weight * calibration
            + self.preservation_weight * preservation
        )
        return {
            "classification": classification,
            "calibration": calibration,
            "preservation": preservation,
            "total": total,
        }


__all__ = [
    "CandidateClassificationLoss",
    "CandidateBrierLoss",
    "RectificationPreservationLoss",
    "CCRRLoss",
]
