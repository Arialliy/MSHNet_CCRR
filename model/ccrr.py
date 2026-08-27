"""Candidate--Context Reliability Rectification (CCRR).

This module is deliberately independent from :mod:`model.MSHNet`.  It consumes
the high-resolution decoder feature, the coarse prediction, multi-scale
predictions, and an externally produced candidate set.  Candidate boxes use
half-open ``xyxy`` coordinates in the coarse-logit coordinate system.

The public forward API accepts either a flat representation

* ``candidate_boxes``: ``[N, 5]`` (batch index, x1, y1, x2, y2), and
* ``candidate_masks``: ``[N, H, W]``;

or per-image lists of ``[N_i, 4]`` boxes and ``[N_i, H, W]`` masks.  Both
forms are normalised to the flat representation in ``candidate_outputs``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def _empty_boxes(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.empty((0, 5), device=device, dtype=dtype)


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _normalise_boxes(
    candidate_boxes: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert supported box layouts to a flat ``[N, 5]`` tensor."""

    if candidate_boxes is None:
        return _empty_boxes(device, dtype)

    if isinstance(candidate_boxes, Mapping):
        if "boxes" not in candidate_boxes:
            raise KeyError("candidate_boxes mapping must contain a 'boxes' key")
        candidate_boxes = candidate_boxes["boxes"]

    # Dense Python sequences (including ``[[x1, y1, x2, y2], ...]``) can be
    # handled exactly like tensors.  Ragged per-image sequences take the path
    # below instead.
    dense_boxes: Tensor | None = None
    if isinstance(candidate_boxes, Tensor):
        dense_boxes = candidate_boxes.to(device=device, dtype=dtype)
    elif isinstance(candidate_boxes, Sequence):
        if len(candidate_boxes) == 0:
            return _empty_boxes(device, dtype)
        try:
            dense_boxes = torch.as_tensor(candidate_boxes, device=device, dtype=dtype)
        except (TypeError, ValueError, RuntimeError):
            dense_boxes = None

    if dense_boxes is not None:
        if dense_boxes.numel() == 0:
            return _empty_boxes(device, dtype)
        if dense_boxes.ndim == 1:
            if dense_boxes.shape[0] not in (4, 5):
                raise ValueError("a single candidate box must contain 4 or 5 values")
            dense_boxes = dense_boxes.unsqueeze(0)
        if dense_boxes.ndim == 2:
            if dense_boxes.shape[1] == 5:
                boxes = dense_boxes
            elif dense_boxes.shape[1] == 4:
                if batch_size != 1:
                    raise ValueError(
                        "flat [N,4] boxes are ambiguous for a multi-image batch; "
                        "use [N,5] boxes or a per-image list"
                    )
                batch_column = dense_boxes.new_zeros((dense_boxes.shape[0], 1))
                boxes = torch.cat((batch_column, dense_boxes), dim=1)
            else:
                raise ValueError("candidate_boxes must have 4 or 5 columns")
            return boxes
        if dense_boxes.ndim == 3:
            if dense_boxes.shape[0] != batch_size or dense_boxes.shape[2] not in (4, 5):
                raise ValueError(
                    "batched candidate_boxes must have shape [B,N,4] or [B,N,5]"
                )
            flattened = []
            for batch_index in range(batch_size):
                image_boxes = dense_boxes[batch_index]
                coords = image_boxes[:, -4:]
                batch_column = coords.new_full((coords.shape[0], 1), float(batch_index))
                flattened.append(torch.cat((batch_column, coords), dim=1))
            return torch.cat(flattened, dim=0) if flattened else _empty_boxes(device, dtype)
        raise ValueError("unsupported candidate_boxes rank")

    if not isinstance(candidate_boxes, Sequence) or len(candidate_boxes) != batch_size:
        raise ValueError("ragged candidate_boxes must be a per-image sequence of length B")

    flattened = []
    for batch_index, image_boxes in enumerate(candidate_boxes):
        image_boxes = _as_tensor(image_boxes, device=device, dtype=dtype)
        if image_boxes.numel() == 0:
            continue
        if image_boxes.ndim == 1:
            image_boxes = image_boxes.unsqueeze(0)
        if image_boxes.ndim != 2 or image_boxes.shape[1] not in (4, 5):
            raise ValueError("each per-image box tensor must have shape [N_i,4] or [N_i,5]")
        coords = image_boxes[:, -4:]
        batch_column = coords.new_full((coords.shape[0], 1), float(batch_index))
        flattened.append(torch.cat((batch_column, coords), dim=1))
    return torch.cat(flattened, dim=0) if flattened else _empty_boxes(device, dtype)


def _normalise_masks(
    candidate_masks: Any,
    *,
    num_candidates: int,
    batch_size: int,
    output_hw: tuple[int, int],
    device: torch.device,
) -> Tensor:
    """Convert candidate masks to bool ``[N,H,W]`` at coarse resolution."""

    output_h, output_w = output_hw
    if num_candidates == 0:
        return torch.zeros((0, output_h, output_w), device=device, dtype=torch.bool)
    if candidate_masks is None:
        raise ValueError("candidate_masks are required when candidate_boxes are non-empty")
    if isinstance(candidate_masks, Mapping):
        if "masks" not in candidate_masks:
            raise KeyError("candidate_masks mapping must contain a 'masks' key")
        candidate_masks = candidate_masks["masks"]

    dense_masks: Tensor | None = None
    if isinstance(candidate_masks, Tensor):
        dense_masks = candidate_masks.to(device=device)
    elif isinstance(candidate_masks, Sequence):
        try:
            dense_masks = torch.as_tensor(candidate_masks, device=device)
        except (TypeError, ValueError, RuntimeError):
            dense_masks = None

    if dense_masks is None:
        if not isinstance(candidate_masks, Sequence) or len(candidate_masks) != batch_size:
            raise ValueError("ragged candidate_masks must be a per-image sequence of length B")
        pieces = []
        for image_masks in candidate_masks:
            image_masks = _as_tensor(image_masks, device=device, dtype=torch.float32)
            if image_masks.numel() == 0:
                continue
            if image_masks.ndim == 2:
                image_masks = image_masks.unsqueeze(0)
            if image_masks.ndim == 4 and image_masks.shape[1] == 1:
                image_masks = image_masks[:, 0]
            if image_masks.ndim != 3:
                raise ValueError("each per-image mask tensor must have shape [N_i,H,W]")
            pieces.append(image_masks)
        if not pieces:
            dense_masks = torch.empty((0, output_h, output_w), device=device)
        else:
            dense_masks = torch.cat(pieces, dim=0)

    if dense_masks.ndim == 2:
        dense_masks = dense_masks.unsqueeze(0)
    elif dense_masks.ndim == 4:
        if dense_masks.shape[0] == num_candidates and dense_masks.shape[1] == 1:
            dense_masks = dense_masks[:, 0]
        elif dense_masks.shape[0] == batch_size and (
            dense_masks.shape[0] * dense_masks.shape[1] == num_candidates
        ):
            dense_masks = dense_masks.reshape(
                num_candidates, dense_masks.shape[-2], dense_masks.shape[-1]
            )
        else:
            raise ValueError(
                "4-D candidate_masks must have shape [N,1,H,W] or [B,N_i,H,W]"
            )
    if dense_masks.ndim != 3 or dense_masks.shape[0] != num_candidates:
        raise ValueError(
            f"candidate mask count ({dense_masks.shape[0] if dense_masks.ndim else 0}) "
            f"does not match candidate box count ({num_candidates})"
        )

    if tuple(dense_masks.shape[-2:]) != output_hw:
        dense_masks = F.interpolate(
            dense_masks.unsqueeze(1).float(), size=output_hw, mode="nearest"
        )[:, 0]
    return dense_masks > 0.5


def _sanitize_boxes(boxes: Tensor, image_hw: tuple[int, int], batch_size: int) -> Tensor:
    """Validate batch indices and clamp ``xyxy`` coordinates to the image."""

    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("normalised boxes must have shape [N,5]")
    if boxes.numel() == 0:
        return boxes.reshape(0, 5)
    if not torch.isfinite(boxes).all():
        raise ValueError("candidate_boxes contain NaN or infinite coordinates")

    raw_batch_indices = boxes[:, 0]
    batch_indices = raw_batch_indices.round().long()
    if not torch.allclose(raw_batch_indices, batch_indices.to(raw_batch_indices.dtype)):
        raise ValueError("candidate box batch indices must be integers")
    if (batch_indices < 0).any() or (batch_indices >= batch_size).any():
        raise ValueError("candidate box batch index is outside the feature batch")

    image_h, image_w = image_hw
    coords = boxes[:, 1:].clone()
    x1 = coords[:, 0].clamp(0, image_w)
    y1 = coords[:, 1].clamp(0, image_h)
    x2 = coords[:, 2].clamp(0, image_w)
    y2 = coords[:, 3].clamp(0, image_h)
    if (x2 <= x1).any() or (y2 <= y1).any():
        raise ValueError("candidate boxes must have positive width and height after clipping")
    return torch.cat((batch_indices.to(boxes.dtype).unsqueeze(1),
                      torch.stack((x1, y1, x2, y2), dim=1)), dim=1)


def _as_scale_list(multi_scale_logits: Any, num_scales: int) -> list[Tensor]:
    if isinstance(multi_scale_logits, Tensor):
        if multi_scale_logits.ndim == 4:
            if num_scales == 1 and multi_scale_logits.shape[1] == 1:
                scales = [multi_scale_logits]
            elif multi_scale_logits.shape[1] == num_scales:
                scales = [multi_scale_logits[:, index : index + 1] for index in range(num_scales)]
            else:
                raise ValueError(
                    "4-D multi_scale_logits must be [B,L,H,W], with L=num_scales"
                )
        elif multi_scale_logits.ndim == 5:
            if multi_scale_logits.shape[0] == num_scales:
                scales = list(multi_scale_logits.unbind(0))
            elif multi_scale_logits.shape[1] == num_scales:
                scales = list(multi_scale_logits.unbind(1))
            else:
                raise ValueError("5-D multi_scale_logits has no num_scales dimension")
        else:
            raise ValueError("multi_scale_logits tensor must be 4-D or 5-D")
    elif isinstance(multi_scale_logits, Sequence):
        scales = list(multi_scale_logits)
    else:
        raise TypeError("multi_scale_logits must be a tensor or sequence of tensors")

    if len(scales) != num_scales:
        raise ValueError(f"expected {num_scales} scale logits, received {len(scales)}")
    return scales


def _extract_scale_features(
    multi_scale_logits: Any,
    candidate_masks: Tensor,
    batch_indices: Tensor,
    *,
    batch_size: int,
    output_hw: tuple[int, int],
    num_scales: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return per-candidate sigmoid means and cross-scale variance."""

    num_candidates = candidate_masks.shape[0]
    if num_candidates == 0:
        return torch.empty((0, num_scales + 1), device=device, dtype=dtype)

    masks = candidate_masks.to(device=device, dtype=dtype)
    areas = masks.flatten(1).sum(dim=1)
    valid = areas > 0
    denominator = areas.clamp_min(1)
    responses = []
    probability_sum = masks.new_zeros(masks.shape)
    squared_probability_sum = masks.new_zeros(masks.shape)
    for scale_logits in _as_scale_list(multi_scale_logits, num_scales):
        if not isinstance(scale_logits, Tensor):
            raise TypeError("each item in multi_scale_logits must be a tensor")
        if scale_logits.ndim == 3:
            scale_logits = scale_logits.unsqueeze(1)
        if scale_logits.ndim != 4 or scale_logits.shape[0] != batch_size:
            raise ValueError("each scale logit must have shape [B,1,H_l,W_l]")
        if scale_logits.shape[1] != 1:
            raise ValueError("CCRR expects one logit channel at each prediction scale")
        scale_logits = scale_logits.to(device=device, dtype=dtype)
        if tuple(scale_logits.shape[-2:]) != output_hw:
            scale_logits = F.interpolate(
                scale_logits, size=output_hw, mode="bilinear", align_corners=False
            )
        candidate_probabilities = scale_logits.sigmoid()[batch_indices, 0]
        probability_sum = probability_sum + candidate_probabilities
        squared_probability_sum = squared_probability_sum + candidate_probabilities.square()
        response = (candidate_probabilities * masks).flatten(1).sum(dim=1) / denominator
        responses.append(torch.where(valid, response, torch.zeros_like(response)))

    response_tensor = torch.stack(responses, dim=1)
    mean_probability = probability_sum / num_scales
    # P_var is computed per pixel first (population variance across scales),
    # then averaged inside the candidate.  This retains spatial disagreement
    # that would disappear if only the ROI means were compared.
    variance_map = (squared_probability_sum / num_scales - mean_probability.square()).clamp_min(0)
    scale_variance = (
        (variance_map * masks).flatten(1).sum(dim=1) / denominator
    ).unsqueeze(1)
    scale_variance = torch.where(
        valid.unsqueeze(1), scale_variance, torch.zeros_like(scale_variance)
    )
    return torch.cat((response_tensor, scale_variance), dim=1)


class CandidateContextEncoder(nn.Module):
    """Encode a candidate core and its surrounding (core-free) ring.

    Core and ring deliberately share one low-capacity encoder.  This makes
    their difference meaningful and avoids learning two independent feature
    spaces from the small candidate training set.
    """

    def __init__(
        self,
        feature_channels: int,
        num_scales: int = 4,
        roi_size: int | tuple[int, int] = 7,
        hidden_dim: int = 64,
        context_scale: float = 3.0,
        min_context_size: float = 15.0,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_dim <= 0 or num_scales <= 0:
            raise ValueError("feature_channels, hidden_dim, and num_scales must be positive")
        if context_scale <= 0:
            raise ValueError("context_scale must be positive")
        if min_context_size <= 0:
            raise ValueError("min_context_size must be positive")

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.roi_size = roi_size
        self.hidden_dim = hidden_dim
        self.context_scale = float(context_scale)
        self.min_context_size = float(min_context_size)
        self.output_dim = 4 * hidden_dim + num_scales + 1

        # Prefer four groups as proposed for V1, while retaining support for
        # small/non-multiple-of-four dimensions used by callers and tests.
        num_groups = next(
            groups for groups in range(min(4, hidden_dim), 0, -1)
            if hidden_dim % groups == 0
        )
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    # Read-only compatibility aliases for code that inspected the V0 module.
    # They do not register duplicate modules or duplicate state-dict entries.
    @property
    def core_encoder(self) -> nn.Module:
        return self.roi_encoder

    @property
    def context_encoder(self) -> nn.Module:
        return self.roi_encoder

    @staticmethod
    def _expand_boxes(
        boxes: Tensor,
        scale: float,
        image_hw: tuple[int, int],
        min_context_size: float = 0.0,
    ) -> Tensor:
        image_h, image_w = image_hw
        coords = boxes[:, 1:]
        centers_x = (coords[:, 0] + coords[:, 2]) * 0.5
        centers_y = (coords[:, 1] + coords[:, 3]) * 0.5
        widths = coords[:, 2] - coords[:, 0]
        heights = coords[:, 3] - coords[:, 1]
        expanded_widths = torch.maximum(
            widths * scale,
            widths.new_full(widths.shape, float(min_context_size)),
        )
        expanded_heights = torch.maximum(
            heights * scale,
            heights.new_full(heights.shape, float(min_context_size)),
        )
        expanded = torch.stack(
            (
                (centers_x - expanded_widths * 0.5).clamp(0, image_w),
                (centers_y - expanded_heights * 0.5).clamp(0, image_h),
                (centers_x + expanded_widths * 0.5).clamp(0, image_w),
                (centers_y + expanded_heights * 0.5).clamp(0, image_h),
            ),
            dim=1,
        )
        return torch.cat((boxes[:, :1], expanded), dim=1)

    @staticmethod
    def _masks_from_boxes(candidate_boxes: Tensor, image_hw: tuple[int, int]) -> Tensor:
        """Build rectangular masks for legacy encoder calls without masks."""

        image_h, image_w = image_hw
        rows = torch.arange(image_h, device=candidate_boxes.device)[None, :, None]
        columns = torch.arange(image_w, device=candidate_boxes.device)[None, None, :]
        coords = candidate_boxes[:, 1:]
        return (
            (columns >= coords[:, 0, None, None])
            & (columns < coords[:, 2, None, None])
            & (rows >= coords[:, 1, None, None])
            & (rows < coords[:, 3, None, None])
        )

    @staticmethod
    def _to_feature_coordinates(
        boxes: Tensor,
        image_hw: tuple[int, int],
        feature_hw: tuple[int, int],
    ) -> Tensor:
        image_h, image_w = image_hw
        feature_h, feature_w = feature_hw
        scaled = boxes.clone()
        scaled[:, (1, 3)] *= feature_w / image_w
        scaled[:, (2, 4)] *= feature_h / image_h
        return scaled

    def forward(
        self,
        feature_map: Tensor,
        candidate_boxes: Tensor,
        scale_features: Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        candidate_masks: Tensor | None = None,
    ) -> Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.feature_channels:
            raise ValueError(
                f"feature_map must have shape [B,{self.feature_channels},H,W]"
            )
        if candidate_boxes.ndim != 2 or candidate_boxes.shape[1] not in (4, 5):
            raise ValueError("candidate_boxes must have shape [N,4] or [N,5]")
        if candidate_boxes.shape[1] == 4:
            if feature_map.shape[0] != 1:
                raise ValueError("[N,4] boxes are only valid for a one-image feature batch")
            candidate_boxes = torch.cat(
                (candidate_boxes.new_zeros((candidate_boxes.shape[0], 1)), candidate_boxes), dim=1
            )
        candidate_boxes = candidate_boxes.to(device=feature_map.device, dtype=feature_map.dtype)
        if image_hw is None:
            image_hw = tuple(feature_map.shape[-2:])
        candidate_boxes = _sanitize_boxes(candidate_boxes, image_hw, feature_map.shape[0])

        num_candidates = candidate_boxes.shape[0]
        if scale_features is None:
            scale_features = feature_map.new_zeros((num_candidates, self.num_scales + 1))
        else:
            scale_features = scale_features.to(device=feature_map.device, dtype=feature_map.dtype)
        if scale_features.shape != (num_candidates, self.num_scales + 1):
            raise ValueError(
                f"scale_features must have shape [N,{self.num_scales + 1}]"
            )
        if num_candidates == 0:
            return feature_map.new_empty((0, self.output_dim))

        if candidate_masks is None:
            candidate_masks = self._masks_from_boxes(candidate_boxes, image_hw)
        else:
            candidate_masks = candidate_masks.to(device=feature_map.device)
            if candidate_masks.ndim == 4 and candidate_masks.shape[1] == 1:
                candidate_masks = candidate_masks[:, 0]
            if candidate_masks.ndim != 3 or candidate_masks.shape[0] != num_candidates:
                raise ValueError("candidate_masks must have shape [N,H,W]")
            if tuple(candidate_masks.shape[-2:]) != image_hw:
                candidate_masks = F.interpolate(
                    candidate_masks.unsqueeze(1).float(), size=image_hw, mode="nearest"
                )[:, 0]

        context_boxes = self._expand_boxes(
            candidate_boxes,
            self.context_scale,
            image_hw,
            self.min_context_size,
        )
        feature_hw = tuple(feature_map.shape[-2:])
        core_feature_boxes = self._to_feature_coordinates(candidate_boxes, image_hw, feature_hw)
        context_feature_boxes = self._to_feature_coordinates(context_boxes, image_hw, feature_hw)

        core_rois = roi_align(
            feature_map,
            core_feature_boxes,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        context_rois = roi_align(
            feature_map,
            context_feature_boxes,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

        # Each candidate mask is an independent one-item image for ROIAlign;
        # therefore the ROI batch indices must be [0, ..., N-1], not the
        # original feature-map batch indices stored in ``context_boxes``.
        mask_boxes = context_boxes.clone()
        mask_boxes[:, 0] = torch.arange(
            num_candidates, device=mask_boxes.device, dtype=mask_boxes.dtype
        )
        core_mask_in_context = roi_align(
            candidate_masks.to(dtype=feature_map.dtype).unsqueeze(1),
            mask_boxes,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).clamp(0.0, 1.0)
        # Remove every context sample touched by the aligned core mask.  A
        # binary exclusion avoids boundary interpolation leaking core signal
        # into what is described and tested as ring-only context.
        ring_mask = (core_mask_in_context <= 0.0).to(context_rois.dtype)
        ring_rois = context_rois * ring_mask

        core_features = self.roi_encoder(core_rois)
        context_features = self.roi_encoder(ring_rois)
        return torch.cat(
            (
                core_features,
                context_features,
                core_features - context_features,
                core_features * context_features,
                scale_features,
            ),
            dim=1,
        )


class ReliabilityHead(nn.Module):
    """Predict target/clutter, optionally with an uncertain third class."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
        zero_effect_initialization: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if num_classes not in (2, 3):
            raise ValueError("num_classes must be 2 (MVP) or 3 (with uncertain class)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if zero_effect_initialization:
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, relation_features: Tensor) -> Tensor:
        if relation_features.ndim != 2 or relation_features.shape[1] != self.input_dim:
            raise ValueError(f"relation_features must have shape [N,{self.input_dim}]")
        return self.classifier(relation_features)


class SafeClutterSuppressor(nn.Module):
    """Apply a bounded, confidence-gated, non-positive candidate residual.

    The computation is out-of-place and fully differentiable with respect to
    the input scores and logits.  Because every correction is non-positive,
    the module cannot create a new positive response.
    """

    def __init__(
        self,
        max_suppression: float = 1.5,
        gate_margin: float = 0.5,
        gate_temperature: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if max_suppression < 0:
            raise ValueError("max_suppression must be non-negative")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if eps <= 0 or eps >= 0.5:
            raise ValueError("eps must be in (0,0.5)")
        self.max_suppression = float(max_suppression)
        # Compatibility for diagnostics/checkpoints that refer to max_delta.
        self.max_delta = self.max_suppression
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)

    def forward(
        self,
        coarse_logits: Tensor,
        target_scores: Tensor,
        clutter_scores: Tensor,
        candidate_masks: Tensor,
        batch_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have shape [B,1,H,W]")
        if candidate_masks.ndim == 2:
            candidate_masks = candidate_masks.unsqueeze(0)
        if candidate_masks.ndim == 4 and candidate_masks.shape[1] == 1:
            candidate_masks = candidate_masks[:, 0]
        if candidate_masks.ndim != 3:
            raise ValueError("candidate_masks must have shape [N,H,W]")
        num_candidates = candidate_masks.shape[0]
        if target_scores.ndim != 1 or target_scores.shape[0] != num_candidates:
            raise ValueError("target_scores must have shape [N]")
        if clutter_scores.ndim != 1 or clutter_scores.shape[0] != num_candidates:
            raise ValueError("clutter_scores must have shape [N]")
        if batch_indices is None:
            if coarse_logits.shape[0] != 1 and num_candidates:
                raise ValueError("batch_indices are required for a multi-image batch")
            batch_indices = torch.zeros(
                num_candidates, device=coarse_logits.device, dtype=torch.long
            )
        else:
            batch_indices = batch_indices.to(device=coarse_logits.device, dtype=torch.long)
        if batch_indices.ndim != 1 or batch_indices.shape[0] != num_candidates:
            raise ValueError("batch_indices must have shape [N]")
        if num_candidates and (
            (batch_indices < 0).any() or (batch_indices >= coarse_logits.shape[0]).any()
        ):
            raise ValueError("batch_indices contain an index outside coarse_logits")

        if num_candidates == 0:
            empty = coarse_logits.new_empty((0,))
            return coarse_logits, empty, empty

        if tuple(candidate_masks.shape[-2:]) != tuple(coarse_logits.shape[-2:]):
            candidate_masks = F.interpolate(
                candidate_masks.unsqueeze(1).float(),
                size=coarse_logits.shape[-2:],
                mode="nearest",
            )[:, 0]
        masks = (candidate_masks > 0.5).to(
            device=coarse_logits.device, dtype=coarse_logits.dtype
        )
        scores = target_scores.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
        clutter_scores = clutter_scores.to(
            device=coarse_logits.device, dtype=coarse_logits.dtype
        )

        candidate_logits = coarse_logits[batch_indices, 0]
        areas = masks.flatten(1).sum(dim=1)
        valid = areas > 0
        confidence_margin = clutter_scores - scores
        evidence_gate = torch.sigmoid(
            (confidence_margin - self.gate_margin) / self.gate_temperature
        )
        # Subtract the equal-evidence baseline before applying target
        # protection.  Thus zero-initialized class logits (p_C == p_T) give
        # an exact identity mapping rather than a small accidental decrease.
        baseline_gate = torch.sigmoid(
            scores.new_tensor(-self.gate_margin / self.gate_temperature)
        )
        maximum_evidence_gate = torch.sigmoid(
            scores.new_tensor((1.0 - self.gate_margin) / self.gate_temperature)
        )
        normalized_evidence = (
            (evidence_gate - baseline_gate)
            / (maximum_evidence_gate - baseline_gate).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        gates = normalized_evidence * (1.0 - scores)
        deltas = -self.max_suppression * gates
        deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
        gates = torch.where(valid, gates, torch.zeros_like(gates))

        candidate_probabilities = candidate_logits.sigmoid()
        masked_probabilities = candidate_probabilities * masks
        peak_probabilities = masked_probabilities.flatten(1).amax(dim=1).clamp_min(self.eps)
        spatial_weights = masked_probabilities / peak_probabilities[:, None, None]
        per_candidate_correction = spatial_weights * deltas[:, None, None]

        correction = coarse_logits.new_zeros(
            (coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3])
        )
        correction = correction.index_add(0, batch_indices, per_candidate_correction)
        refined_logits = coarse_logits + correction.unsqueeze(1)
        return refined_logits, deltas, gates


class InstanceLogitRectifier(SafeClutterSuppressor):
    """Backward-compatible name/call form for the V1 safe suppressor.

    V0 callers supplied only target probability.  For binary reliability this
    uniquely determines clutter probability, so the legacy signature remains
    usable while inheriting V1's non-positive-delta guarantee.
    """

    def __init__(
        self,
        max_delta: float = 1.5,
        eps: float = 1e-6,
        gate_margin: float = 0.5,
        gate_temperature: float = 0.1,
    ) -> None:
        super().__init__(
            max_suppression=max_delta,
            gate_margin=gate_margin,
            gate_temperature=gate_temperature,
            eps=eps,
        )

    def forward(
        self,
        coarse_logits: Tensor,
        target_scores: Tensor,
        candidate_masks: Tensor,
        batch_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        refined_logits, deltas, _ = super().forward(
            coarse_logits,
            target_scores,
            1.0 - target_scores,
            candidate_masks,
            batch_indices,
        )
        return refined_logits, deltas


class CCRRModule(nn.Module):
    """Candidate--Context Reliability Rectification module."""

    def __init__(
        self,
        feature_channels: int,
        num_scales: int = 4,
        roi_size: int | tuple[int, int] = 7,
        hidden_dim: int = 64,
        context_scale: float = 3.0,
        min_context_size: float = 15.0,
        dropout: float = 0.3,
        max_delta: float = 1.5,
        max_suppression: float | None = None,
        gate_margin: float = 0.5,
        gate_temperature: float = 0.1,
        num_classes: int = 3,
        eps: float = 1e-6,
        zero_effect_initialization: bool = True,
        rectifier: str = "suppression_only",
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_classes = num_classes
        if rectifier != "suppression_only":
            raise ValueError("CCRR-V1 supports only rectifier='suppression_only'")
        self.encoder = CandidateContextEncoder(
            feature_channels=feature_channels,
            num_scales=num_scales,
            roi_size=roi_size,
            hidden_dim=hidden_dim,
            context_scale=context_scale,
            min_context_size=min_context_size,
        )
        self.reliability_head = ReliabilityHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            zero_effect_initialization=zero_effect_initialization,
        )
        suppression_limit = max_delta if max_suppression is None else max_suppression
        self.rectifier = SafeClutterSuppressor(
            max_suppression=suppression_limit,
            gate_margin=gate_margin,
            gate_temperature=gate_temperature,
            eps=eps,
        )

    def forward(
        self,
        feature_map: Tensor,
        coarse_logits: Tensor,
        multi_scale_logits: Any,
        candidate_boxes: Any,
        candidate_masks: Any = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B,C,H_f,W_f]")
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have shape [B,1,H,W]")
        if feature_map.shape[0] != coarse_logits.shape[0]:
            raise ValueError("feature_map and coarse_logits must have the same batch size")
        if feature_map.device != coarse_logits.device:
            raise ValueError("feature_map and coarse_logits must be on the same device")

        # A candidate dictionary returned by utils.candidate can be supplied as
        # candidate_boxes while leaving candidate_masks=None.
        if isinstance(candidate_boxes, Mapping):
            candidate_record = candidate_boxes
            if candidate_masks is None:
                candidate_masks = candidate_record.get("masks")
            candidate_boxes = candidate_record.get("boxes")

        batch_size = coarse_logits.shape[0]
        output_hw = tuple(coarse_logits.shape[-2:])
        boxes = _normalise_boxes(
            candidate_boxes,
            batch_size=batch_size,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        boxes = _sanitize_boxes(boxes, output_hw, batch_size)
        masks = _normalise_masks(
            candidate_masks,
            num_candidates=boxes.shape[0],
            batch_size=batch_size,
            output_hw=output_hw,
            device=coarse_logits.device,
        )
        batch_indices = boxes[:, 0].long()

        scale_features = _extract_scale_features(
            multi_scale_logits,
            masks,
            batch_indices,
            batch_size=batch_size,
            output_hw=output_hw,
            num_scales=self.num_scales,
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        relation_features = self.encoder(
            feature_map,
            boxes,
            candidate_masks=masks,
            scale_features=scale_features,
            image_hw=output_hw,
        )
        class_logits = self.reliability_head(relation_features)
        class_probs = class_logits.softmax(dim=1)
        target_scores = class_probs[:, 0]
        clutter_scores = class_probs[:, 1]
        if self.num_classes == 3:
            uncertain_scores = class_probs[:, 2]
        else:
            uncertain_scores = class_probs.new_zeros((class_probs.shape[0],))

        refined_logits, deltas, gates = self.rectifier(
            coarse_logits,
            target_scores,
            clutter_scores,
            masks,
            batch_indices,
        )
        candidate_outputs = {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "target_scores": target_scores,
            "clutter_scores": clutter_scores,
            "uncertain_scores": uncertain_scores,
            "deltas": deltas,
            "gates": gates,
            "boxes": boxes,
            "candidate_masks": masks,
            "batch_indices": batch_indices,
            "scale_features": scale_features,
            "relation_features": relation_features,
        }
        return refined_logits, candidate_outputs


__all__ = [
    "CandidateContextEncoder",
    "ReliabilityHead",
    "InstanceLogitRectifier",
    "SafeClutterSuppressor",
    "CCRRModule",
]
