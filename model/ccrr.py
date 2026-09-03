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
import math
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


class MaskedHybridPool(nn.Module):
    """Pool masked spatial features with average, maximum, and Top-K mean.

    The three statistics are projected back to a fixed-width representation.
    The projection starts from the masked-average branch so enabling this
    module does not inject a randomly initialised Max/Top-K contribution at
    the beginning of an enhanced run.
    """

    def __init__(
        self,
        channels: int,
        output_dim: int,
        topk_ratio: float = 0.125,
        minimum_topk: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0 or output_dim <= 0:
            raise ValueError("channels and output_dim must be positive")
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must lie in (0, 1]")
        if minimum_topk <= 0:
            raise ValueError("minimum_topk must be positive")

        self.channels = int(channels)
        self.output_dim = int(output_dim)
        self.topk_ratio = float(topk_ratio)
        self.minimum_topk = int(minimum_topk)
        self.projection = nn.Sequential(
            nn.Linear(3 * self.channels, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(inplace=True),
        )

        nn.init.zeros_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)
        with torch.no_grad():
            diagonal = min(self.channels, self.output_dim)
            self.projection[0].weight[:diagonal, :diagonal] = torch.eye(
                diagonal,
                device=self.projection[0].weight.device,
                dtype=self.projection[0].weight.dtype,
            )

    def forward(self, feature: Tensor, mask: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError("feature must have shape [N,C,H,W]")
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"feature must have {self.channels} channels, received {feature.shape[1]}"
            )
        if feature.shape[-2] <= 0 or feature.shape[-1] <= 0:
            raise ValueError("feature spatial dimensions must be positive")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [N,1,H,W]")
        if mask.shape[0] != feature.shape[0]:
            raise ValueError("feature and mask disagree on N")
        if mask.shape[-2:] != feature.shape[-2:]:
            raise ValueError("feature and mask spatial sizes differ")

        mask_bool = mask.to(device=feature.device) > 0.0
        mask_float = mask_bool.to(dtype=feature.dtype)
        masked_feature = torch.where(mask_bool, feature, torch.zeros_like(feature))

        count = mask_float.flatten(2).sum(dim=-1).clamp_min(1.0)
        average = masked_feature.flatten(2).sum(dim=-1) / count

        flattened = torch.where(
            mask_bool,
            feature,
            torch.full_like(feature, float("-inf")),
        ).flatten(2)
        maximum = flattened.amax(dim=-1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )

        valid_count = mask_bool.flatten(2).sum(dim=-1).squeeze(1)
        k_per_candidate = torch.ceil(
            valid_count.to(dtype=torch.float32) * self.topk_ratio
        ).to(dtype=torch.long)
        k_per_candidate = torch.maximum(
            k_per_candidate,
            torch.full_like(k_per_candidate, self.minimum_topk),
        ).clamp_max(feature.shape[-2] * feature.shape[-1])
        maximum_k = int(k_per_candidate.max().item()) if k_per_candidate.numel() else 1

        topk_values = flattened.topk(k=maximum_k, dim=-1).values
        rank = torch.arange(maximum_k, device=feature.device).view(1, 1, -1)
        selected_rank = rank < k_per_candidate.view(-1, 1, 1)
        finite = torch.isfinite(topk_values) & selected_rank
        topk_mean = torch.where(
            finite, topk_values, torch.zeros_like(topk_values)
        ).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)

        statistics = torch.cat((average, maximum, topk_mean), dim=1)
        output = self.projection(statistics)
        if not torch.isfinite(output).all():
            raise ValueError("hybrid pooled features are not finite")
        return output


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
        pooling_mode: str = "avg",
        topk_ratio: float = 0.125,
        minimum_topk: int = 1,
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
        self.pooling_mode = pooling_mode

        # Prefer four groups as proposed for V1, while retaining support for
        # small/non-multiple-of-four dimensions used by callers and tests.
        num_groups = next(
            groups for groups in range(min(4, hidden_dim), 0, -1)
            if hidden_dim % groups == 0
        )
        if pooling_mode == "avg":
            # Preserve the legacy module hierarchy and state-dict keys exactly.
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
            self.spatial_encoder = None
            self.hybrid_pool = None
        elif pooling_mode == "avg_max_topk":
            self.roi_encoder = None
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.hybrid_pool = MaskedHybridPool(
                channels=hidden_dim,
                output_dim=hidden_dim,
                topk_ratio=topk_ratio,
                minimum_topk=minimum_topk,
            )
        else:
            raise ValueError(f"unsupported pooling_mode={pooling_mode!r}")

    # Read-only compatibility aliases for code that inspected the V0 module.
    # They do not register duplicate modules or duplicate state-dict entries.
    @property
    def core_encoder(self) -> nn.Module:
        if self.roi_encoder is not None:
            return self.roi_encoder
        assert self.spatial_encoder is not None
        return self.spatial_encoder

    @property
    def context_encoder(self) -> nn.Module:
        return self.core_encoder

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

        if self.pooling_mode == "avg":
            assert self.roi_encoder is not None
            core_features = self.roi_encoder(core_rois)
            context_features = self.roi_encoder(ring_rois)
        else:
            core_mask_boxes = candidate_boxes.clone()
            core_mask_boxes[:, 0] = torch.arange(
                num_candidates,
                device=core_mask_boxes.device,
                dtype=core_mask_boxes.dtype,
            )
            core_mask_in_core = roi_align(
                candidate_masks.to(dtype=feature_map.dtype).unsqueeze(1),
                core_mask_boxes,
                output_size=self.roi_size,
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            ).clamp(0.0, 1.0)
            core_mask = (core_mask_in_core > 0.0).to(core_rois.dtype)

            assert self.spatial_encoder is not None
            assert self.hybrid_pool is not None
            core_encoded = self.spatial_encoder(core_rois * core_mask) * core_mask
            ring_encoded = self.spatial_encoder(context_rois * ring_mask) * ring_mask
            core_features = self.hybrid_pool(core_encoded, core_mask)
            context_features = self.hybrid_pool(ring_encoded, ring_mask)
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


class SelectiveReliabilityHead(nn.Module):
    """Predict clutter probability and target-presence quality separately.

    The two final layers are zero-initialised by default.  Consequently both
    probabilities start at ``0.5`` and, together with
    :class:`SelectiveRiskGate`, the SCA branch starts as an exact Keep/no-op
    plugin.  Keeping the two predictions separate is important: looking like
    clutter is not the same question as being safe to remove.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        zero_effect_initialization: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.clutter_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        if zero_effect_initialization:
            nn.init.zeros_(self.clutter_head.weight)
            nn.init.zeros_(self.clutter_head.bias)
            nn.init.zeros_(self.quality_head.weight)
            nn.init.zeros_(self.quality_head.bias)

    def forward(self, relation_features: Tensor) -> dict[str, Tensor]:
        if relation_features.ndim != 2 or relation_features.shape[1] != self.input_dim:
            raise ValueError(
                f"relation_features must have shape [N,{self.input_dim}]"
            )
        if not torch.isfinite(relation_features).all():
            raise ValueError("relation_features contain NaN or infinite values")

        shared_feature = self.shared(relation_features)
        clutter_logits = self.clutter_head(shared_feature).squeeze(1)
        target_quality_logits = self.quality_head(shared_feature).squeeze(1)
        clutter_probability = clutter_logits.sigmoid()
        target_quality = target_quality_logits.sigmoid()
        for name, value in (
            ("shared_feature", shared_feature),
            ("clutter_logits", clutter_logits),
            ("target_quality_logits", target_quality_logits),
            ("clutter_probability", clutter_probability),
            ("target_quality", target_quality),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return {
            "clutter_logits": clutter_logits,
            "target_quality_logits": target_quality_logits,
            "clutter_probability": clutter_probability,
            "target_quality": target_quality,
            "shared_feature": shared_feature,
        }


class TargetGuardedReliabilityHead(nn.Module):
    """Predict clutter, target quality, and target-presence evidence separately.

    The guard is intentionally independent from the continuous quality head:
    a tiny or fragmented component can have low shape/localisation quality and
    still contain a real target that must never be removed.  Zero-initialising
    all three heads preserves the exact Keep/no-op startup contract.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        zero_effect_initialization: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.clutter_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.target_guard_head = nn.Linear(hidden_dim, 1)
        if zero_effect_initialization:
            for head in (
                self.clutter_head,
                self.quality_head,
                self.target_guard_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, relation_features: Tensor) -> dict[str, Tensor]:
        if relation_features.ndim != 2 or relation_features.shape[1] != self.input_dim:
            raise ValueError(
                f"relation_features must have shape [N,{self.input_dim}]"
            )
        if not torch.isfinite(relation_features).all():
            raise ValueError("relation_features contain NaN or infinite values")

        shared_feature = self.shared(relation_features)
        clutter_logits = self.clutter_head(shared_feature).squeeze(1)
        target_quality_logits = self.quality_head(shared_feature).squeeze(1)
        target_guard_logits = self.target_guard_head(shared_feature).squeeze(1)
        clutter_probability = clutter_logits.sigmoid()
        target_quality = target_quality_logits.sigmoid()
        target_guard = target_guard_logits.sigmoid()
        for name, value in (
            ("shared_feature", shared_feature),
            ("clutter_logits", clutter_logits),
            ("target_quality_logits", target_quality_logits),
            ("target_guard_logits", target_guard_logits),
            ("clutter_probability", clutter_probability),
            ("target_quality", target_quality),
            ("target_guard", target_guard),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return {
            "clutter_logits": clutter_logits,
            "target_quality_logits": target_quality_logits,
            "target_guard_logits": target_guard_logits,
            "clutter_probability": clutter_probability,
            "target_quality": target_quality,
            "target_guard": target_guard,
            "shared_feature": shared_feature,
        }


class ActionMaskEncoder(nn.Module):
    """Encode exact output-component shape and coarse confidence.

    ``action_masks`` are sampled only here (and by the executor below).  They
    are deliberately not used by :class:`CandidateContextEncoder`, whose
    context and scale features are defined by the lower-threshold proposal
    masks instead.
    """

    def __init__(
        self,
        roi_size: int | tuple[int, int] = 7,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if isinstance(roi_size, int):
            if roi_size <= 0:
                raise ValueError("roi_size must be positive")
        elif (
            not isinstance(roi_size, tuple)
            or len(roi_size) != 2
            or any(size <= 0 for size in roi_size)
        ):
            raise ValueError("roi_size must be a positive int or a positive (h,w) tuple")

        self.roi_size = roi_size
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(hidden_dim)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    @staticmethod
    def boxes_from_masks(
        action_masks: Tensor,
        batch_indices: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return half-open component boxes without merging candidates."""

        if action_masks.ndim != 3:
            raise ValueError("action_masks must have shape [N,H,W]")
        num_candidates = action_masks.shape[0]
        if batch_indices.ndim != 1 or batch_indices.shape[0] != num_candidates:
            raise ValueError("batch_indices must have shape [N]")
        if num_candidates == 0:
            return torch.empty(
                (0, 5), device=action_masks.device, dtype=dtype
            )

        boxes = []
        for index in range(num_candidates):
            positions = torch.nonzero(action_masks[index], as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("every action mask must contain at least one pixel")
            y1, x1 = positions.amin(dim=0)
            y2, x2 = positions.amax(dim=0) + 1
            boxes.append(
                torch.stack(
                    (
                        batch_indices[index],
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )
            )
        return torch.stack(boxes).to(device=action_masks.device, dtype=dtype)

    def forward(
        self,
        coarse_logits: Tensor,
        action_masks: Tensor,
        action_boxes: Tensor | None = None,
        batch_indices: Tensor | None = None,
    ) -> Tensor:
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have shape [B,1,H,W]")
        if not torch.is_floating_point(coarse_logits):
            raise TypeError("coarse_logits must be floating point")
        if not torch.isfinite(coarse_logits).all():
            raise ValueError("coarse_logits contain NaN or infinite values")

        if action_masks.ndim == 2:
            action_masks = action_masks.unsqueeze(0)
        if action_masks.ndim == 4 and action_masks.shape[1] == 1:
            action_masks = action_masks[:, 0]
        if action_masks.ndim != 3:
            raise ValueError("action_masks must have shape [N,H,W]")
        if torch.is_floating_point(action_masks) and not torch.isfinite(action_masks).all():
            raise ValueError("action_masks contain NaN or infinite values")
        if tuple(action_masks.shape[-2:]) != tuple(coarse_logits.shape[-2:]):
            action_masks = F.interpolate(
                action_masks.unsqueeze(1).float(),
                size=coarse_logits.shape[-2:],
                mode="nearest",
            )[:, 0]
        masks = (action_masks > 0.5).to(device=coarse_logits.device)
        num_candidates = masks.shape[0]

        if batch_indices is None:
            if coarse_logits.shape[0] != 1 and num_candidates:
                raise ValueError(
                    "batch_indices are required for a multi-image coarse batch"
                )
            batch_indices = torch.zeros(
                num_candidates, device=coarse_logits.device, dtype=torch.long
            )
        else:
            batch_indices = batch_indices.to(
                device=coarse_logits.device, dtype=torch.long
            )
        if batch_indices.ndim != 1 or batch_indices.shape[0] != num_candidates:
            raise ValueError("batch_indices must have shape [N]")
        if num_candidates and (
            (batch_indices < 0).any()
            or (batch_indices >= coarse_logits.shape[0]).any()
        ):
            raise ValueError("batch_indices contain an index outside coarse_logits")
        if num_candidates == 0:
            return coarse_logits.new_empty((0, self.output_dim))

        if action_boxes is None:
            boxes = self.boxes_from_masks(
                masks,
                batch_indices,
                dtype=coarse_logits.dtype,
            )
        else:
            boxes = _normalise_boxes(
                action_boxes,
                batch_size=coarse_logits.shape[0],
                device=coarse_logits.device,
                dtype=coarse_logits.dtype,
            )
            if boxes.shape[0] != num_candidates:
                raise ValueError(
                    "action box count must match the number of action masks"
                )
            boxes = _sanitize_boxes(
                boxes,
                tuple(coarse_logits.shape[-2:]),
                coarse_logits.shape[0],
            )
            if not torch.equal(boxes[:, 0].long(), batch_indices):
                raise ValueError(
                    "action box batch indices must match batch_indices"
                )

        independent_mask_boxes = boxes.clone()
        independent_mask_boxes[:, 0] = torch.arange(
            num_candidates,
            device=boxes.device,
            dtype=boxes.dtype,
        )
        action_mask_roi = roi_align(
            masks.to(dtype=coarse_logits.dtype).unsqueeze(1),
            independent_mask_boxes,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).clamp(0.0, 1.0)
        coarse_probability_roi = roi_align(
            coarse_logits.sigmoid(),
            boxes,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).clamp(0.0, 1.0)
        encoded = self.mask_encoder(
            torch.cat((action_mask_roi, coarse_probability_roi), dim=1)
        )
        if encoded.shape != (num_candidates, self.output_dim):
            raise RuntimeError("ActionMaskEncoder produced an unexpected output shape")
        if not torch.isfinite(encoded).all():
            raise ValueError("action mask features contain NaN or infinite values")
        return encoded


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


class ThresholdAwareClutterSuppressor(nn.Module):
    """Suppress high-confidence clutter below a fixed output threshold.

    Unlike V1's bounded spatially weighted delta, a selected candidate gets a
    uniform correction over its currently-positive support.  With no cap, a
    hard-selected candidate is guaranteed to cross ``output_threshold`` even
    when the coarse logit is extremely large.
    """

    def __init__(
        self,
        action_threshold: float = 0.90,
        remove_threshold: float = 0.45,
        soft_temperature: float = 0.05,
        max_suppression: float | None = None,
        output_threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.5 < action_threshold < 1.0:
            raise ValueError("action_threshold must lie in (0.5,1)")
        if not 0.0 < remove_threshold < output_threshold:
            raise ValueError("remove_threshold must lie in (0,output_threshold)")
        if not 0.0 < output_threshold < 1.0:
            raise ValueError("output_threshold must lie in (0,1)")
        if soft_temperature <= 0:
            raise ValueError("soft_temperature must be positive")
        if max_suppression is not None and max_suppression <= 0:
            raise ValueError("max_suppression must be positive or None")
        if not 0 < eps < 0.5:
            raise ValueError("eps must lie in (0,0.5)")
        self.action_threshold = float(action_threshold)
        self.remove_threshold = float(remove_threshold)
        self.soft_temperature = float(soft_temperature)
        self.max_suppression = (
            None if max_suppression is None else float(max_suppression)
        )
        self.output_threshold = float(output_threshold)
        self.eps = float(eps)

    def forward(
        self,
        coarse_logits: Tensor,
        target_scores: Tensor,
        clutter_scores: Tensor,
        candidate_masks: Tensor,
        batch_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
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
            batch_indices = batch_indices.to(
                device=coarse_logits.device, dtype=torch.long
            )
        if batch_indices.ndim != 1 or batch_indices.shape[0] != num_candidates:
            raise ValueError("batch_indices must have shape [N]")
        if num_candidates and (
            (batch_indices < 0).any()
            or (batch_indices >= coarse_logits.shape[0]).any()
        ):
            raise ValueError("batch_indices contain an index outside coarse_logits")
        if num_candidates == 0:
            empty = coarse_logits.new_empty((0,))
            return {
                "refined_logits": coarse_logits,
                "deltas": empty,
                "gates": empty,
                "required_deltas": empty,
                "unclipped_required_deltas": empty,
                "peak_logits": empty,
                "active_support": candidate_masks.to(dtype=torch.bool),
            }

        if tuple(candidate_masks.shape[-2:]) != tuple(coarse_logits.shape[-2:]):
            candidate_masks = F.interpolate(
                candidate_masks.unsqueeze(1).float(),
                size=coarse_logits.shape[-2:],
                mode="nearest",
            )[:, 0]
        masks = (candidate_masks > 0.5).to(device=coarse_logits.device)
        target_scores = target_scores.to(
            device=coarse_logits.device, dtype=coarse_logits.dtype
        )
        clutter_scores = clutter_scores.to(
            device=coarse_logits.device, dtype=coarse_logits.dtype
        )
        candidate_logits = coarse_logits[batch_indices, 0]
        valid = masks.flatten(1).any(dim=1)
        negative_infinity = torch.full_like(candidate_logits, float("-inf"))
        peak_logits = torch.where(
            masks, candidate_logits, negative_infinity
        ).flatten(1).amax(dim=1)
        peak_logits = torch.where(valid, peak_logits, torch.zeros_like(peak_logits))
        remove_logit = torch.logit(
            peak_logits.new_tensor(self.remove_threshold), eps=self.eps
        )
        unclipped_required = (remove_logit - peak_logits).clamp(max=0.0)
        required = unclipped_required
        if self.max_suppression is not None:
            required = required.clamp(min=-self.max_suppression)

        raw_soft_gate = torch.sigmoid(
            (clutter_scores - self.action_threshold) / self.soft_temperature
        )
        # Preserve exact identity for the zero-initialized binary head
        # (p_target == p_clutter == 0.5) during training.
        identity_gate = torch.sigmoid(
            (
                torch.full_like(clutter_scores, 0.5)
                - self.action_threshold
            )
            / self.soft_temperature
        )
        maximum_gate = torch.sigmoid(
            (torch.ones_like(clutter_scores) - self.action_threshold)
            / self.soft_temperature
        )
        soft_gate = (
            (raw_soft_gate - identity_gate)
            / (maximum_gate - identity_gate).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        if self.training:
            gates = soft_gate
        else:
            gates = (clutter_scores >= self.action_threshold).to(
                dtype=coarse_logits.dtype
            )
        gates = torch.where(valid, gates, torch.zeros_like(gates))
        deltas = gates * required

        output_logit = torch.logit(
            coarse_logits.new_tensor(self.output_threshold), eps=self.eps
        )
        active_support = masks & (candidate_logits.detach() > output_logit)
        per_candidate_correction = (
            active_support.to(dtype=coarse_logits.dtype) * deltas[:, None, None]
        )
        correction = coarse_logits.new_zeros(
            (coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3])
        )
        correction = correction.index_add(
            0, batch_indices, per_candidate_correction
        )
        refined_logits = coarse_logits + correction.unsqueeze(1)
        return {
            "refined_logits": refined_logits,
            "deltas": deltas,
            "gates": gates,
            "required_deltas": required,
            "unclipped_required_deltas": unclipped_required,
            "peak_logits": peak_logits,
            "active_support": active_support,
        }


class SelectiveRiskGate(nn.Module):
    """Turn clutter and target-quality estimates into a selective action.

    Evaluation uses the hard risk/veto rule from the SCA design.  Training
    uses a differentiable risk gate, normalised so that the zero-initialised
    head (``p_clutter == q_target == 0.5``) produces an *exact* zero action.
    """

    def __init__(
        self,
        risk_threshold: float = 2.0,
        quality_veto_threshold: float = 0.5,
        risk_alpha: float = 1.0,
        temperature: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not math.isfinite(risk_threshold):
            raise ValueError("risk_threshold must be finite")
        if not 0.0 <= quality_veto_threshold <= 1.0:
            raise ValueError("quality_veto_threshold must lie in [0,1]")
        if not math.isfinite(risk_alpha) or risk_alpha < 0:
            raise ValueError("risk_alpha must be non-negative and finite")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0,0.5)")
        self.risk_threshold = float(risk_threshold)
        self.quality_veto_threshold = float(quality_veto_threshold)
        self.risk_alpha = float(risk_alpha)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        clutter_probability: Tensor,
        target_quality: Tensor,
    ) -> dict[str, Tensor]:
        if clutter_probability.ndim != 1:
            raise ValueError("clutter_probability must have shape [N]")
        if target_quality.shape != clutter_probability.shape:
            raise ValueError("target_quality must have the same shape as clutter_probability")
        if not torch.is_floating_point(clutter_probability) or not torch.is_floating_point(
            target_quality
        ):
            raise TypeError("clutter_probability and target_quality must be floating point")
        if not torch.isfinite(clutter_probability).all():
            raise ValueError("clutter_probability contains NaN or infinite values")
        if not torch.isfinite(target_quality).all():
            raise ValueError("target_quality contains NaN or infinite values")
        if (
            (clutter_probability < 0).any()
            or (clutter_probability > 1).any()
            or (target_quality < 0).any()
            or (target_quality > 1).any()
        ):
            raise ValueError("probability inputs must lie in [0,1]")

        clutter_probability = clutter_probability.clamp(self.eps, 1.0 - self.eps)
        target_quality = target_quality.clamp(self.eps, 1.0 - self.eps)
        clutter_logit = torch.logit(clutter_probability)
        quality_logit = torch.logit(target_quality)
        risk_score = clutter_logit - self.risk_alpha * quality_logit

        raw_soft_action = torch.sigmoid(
            (risk_score - self.risk_threshold) / self.temperature
        )
        # Equal evidence is the exact zero-effect state of the selective
        # heads.  Removing its logistic tail retains useful gradients at the
        # origin while preventing an accidental training-time modification.
        zero_risk_action = torch.sigmoid(
            risk_score.new_tensor(-self.risk_threshold / self.temperature)
        )
        soft_action = (
            (raw_soft_action - zero_risk_action)
            / (1.0 - zero_risk_action).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        quality_veto = (
            target_quality <= self.quality_veto_threshold
        ).to(dtype=soft_action.dtype)

        if self.training:
            gate = soft_action * quality_veto
        else:
            gate = (
                (risk_score >= self.risk_threshold)
                & (target_quality <= self.quality_veto_threshold)
            ).to(dtype=soft_action.dtype)
        for name, value in (
            ("gate", gate),
            ("risk_score", risk_score),
            ("quality_veto", quality_veto),
            ("soft_action", soft_action),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return {
            "gate": gate,
            "risk_score": risk_score,
            "quality_veto": quality_veto,
            "soft_action": soft_action,
            "clutter_logit": clutter_logit,
            "quality_logit": quality_logit,
        }


class TargetGuardedRiskGate(nn.Module):
    """Differentiable training veto and conservative hard evaluation rule."""

    def __init__(
        self,
        risk_threshold: float = 2.0,
        quality_veto_threshold: float = 0.2,
        guard_veto_threshold: float = 0.2,
        risk_alpha: float = 1.0,
        guard_alpha: float = 1.0,
        action_temperature: float = 0.05,
        quality_temperature: float = 0.05,
        guard_temperature: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not math.isfinite(risk_threshold):
            raise ValueError("risk_threshold must be finite")
        for name, value in (
            ("quality_veto_threshold", quality_veto_threshold),
            ("guard_veto_threshold", guard_veto_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name, value in (("risk_alpha", risk_alpha), ("guard_alpha", guard_alpha)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        for name, value in (
            ("action_temperature", action_temperature),
            ("quality_temperature", quality_temperature),
            ("guard_temperature", guard_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0,0.5)")

        self.risk_threshold = float(risk_threshold)
        self.quality_veto_threshold = float(quality_veto_threshold)
        self.guard_veto_threshold = float(guard_veto_threshold)
        self.risk_alpha = float(risk_alpha)
        self.guard_alpha = float(guard_alpha)
        self.action_temperature = float(action_temperature)
        self.quality_temperature = float(quality_temperature)
        self.guard_temperature = float(guard_temperature)
        self.eps = float(eps)

    def _validate_probability(self, name: str, value: Tensor) -> None:
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [N]")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        if (value < 0).any() or (value > 1).any():
            raise ValueError(f"{name} must lie in [0,1]")

    def forward(
        self,
        clutter_probability: Tensor,
        target_quality: Tensor,
        target_guard: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_probability("clutter_probability", clutter_probability)
        self._validate_probability("target_quality", target_quality)
        self._validate_probability("target_guard", target_guard)
        if target_quality.shape != clutter_probability.shape:
            raise ValueError("target_quality must have the same shape as clutter_probability")
        if target_guard.shape != clutter_probability.shape:
            raise ValueError("target_guard must have the same shape as clutter_probability")
        if target_quality.device != clutter_probability.device or target_guard.device != clutter_probability.device:
            raise ValueError("all probability inputs must be on the same device")

        clutter_logit = torch.logit(
            clutter_probability.clamp(self.eps, 1.0 - self.eps)
        )
        quality_logit = torch.logit(target_quality.clamp(self.eps, 1.0 - self.eps))
        guard_logit = torch.logit(target_guard.clamp(self.eps, 1.0 - self.eps))
        risk_score = (
            clutter_logit
            - self.risk_alpha * quality_logit
            - self.guard_alpha * guard_logit
        )

        raw_action = torch.sigmoid(
            (risk_score - self.risk_threshold) / self.action_temperature
        )
        zero_action = torch.sigmoid(
            risk_score.new_tensor(-self.risk_threshold / self.action_temperature)
        )
        soft_action = (
            (raw_action - zero_action) / (1.0 - zero_action).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        soft_quality_allow = torch.sigmoid(
            (self.quality_veto_threshold - target_quality)
            / self.quality_temperature
        )
        soft_guard_allow = torch.sigmoid(
            (self.guard_veto_threshold - target_guard) / self.guard_temperature
        )
        quality_veto = (target_quality <= self.quality_veto_threshold).to(
            dtype=soft_action.dtype
        )
        guard_veto = (target_guard <= self.guard_veto_threshold).to(
            dtype=soft_action.dtype
        )

        if self.training:
            gate = soft_action * soft_quality_allow * soft_guard_allow
        else:
            gate = (
                (risk_score >= self.risk_threshold)
                & (target_quality <= self.quality_veto_threshold)
                & (target_guard <= self.guard_veto_threshold)
            ).to(dtype=soft_action.dtype)
        outputs = {
            "gate": gate,
            "risk_score": risk_score,
            "soft_action": soft_action,
            "soft_quality_allow": soft_quality_allow,
            "soft_guard_allow": soft_guard_allow,
            "quality_veto": quality_veto,
            "guard_veto": guard_veto,
            "target_guard": target_guard,
            "clutter_logit": clutter_logit,
            "quality_logit": quality_logit,
            "guard_logit": guard_logit,
        }
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return outputs


class ComponentAlignedSuppressor(nn.Module):
    """Execute a supplied gate on exact 0.5-output components.

    This class contains no classifier threshold.  Selection belongs solely to
    :class:`SelectiveRiskGate`; this executor only moves the currently
    positive support of each selected component below ``remove_threshold``.
    """

    def __init__(
        self,
        remove_threshold: float = 0.45,
        output_threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < remove_threshold < output_threshold:
            raise ValueError("remove_threshold must lie in (0,output_threshold)")
        if not 0.0 < output_threshold < 1.0:
            raise ValueError("output_threshold must lie in (0,1)")
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0,0.5)")
        self.remove_threshold = float(remove_threshold)
        self.output_threshold = float(output_threshold)
        self.eps = float(eps)

    def forward(
        self,
        coarse_logits: Tensor,
        gates: Tensor,
        action_masks: Tensor,
        batch_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have shape [B,1,H,W]")
        if not torch.is_floating_point(coarse_logits):
            raise TypeError("coarse_logits must be floating point")
        if not torch.isfinite(coarse_logits).all():
            raise ValueError("coarse_logits contain NaN or infinite values")
        if action_masks.ndim == 2:
            action_masks = action_masks.unsqueeze(0)
        if action_masks.ndim == 4 and action_masks.shape[1] == 1:
            action_masks = action_masks[:, 0]
        if action_masks.ndim != 3:
            raise ValueError("action_masks must have shape [N,H,W]")
        if torch.is_floating_point(action_masks) and not torch.isfinite(action_masks).all():
            raise ValueError("action_masks contain NaN or infinite values")

        num_candidates = action_masks.shape[0]
        if gates.ndim != 1 or gates.shape[0] != num_candidates:
            raise ValueError("gates must have shape [N]")
        if not torch.is_floating_point(gates):
            raise TypeError("gates must be floating point")
        if not torch.isfinite(gates).all():
            raise ValueError("gates contain NaN or infinite values")
        if (gates < 0).any() or (gates > 1).any():
            raise ValueError("gates must lie in [0,1]")

        if batch_indices is None:
            if coarse_logits.shape[0] != 1 and num_candidates:
                raise ValueError("batch_indices are required for a multi-image batch")
            batch_indices = torch.zeros(
                num_candidates, device=coarse_logits.device, dtype=torch.long
            )
        else:
            batch_indices = batch_indices.to(
                device=coarse_logits.device, dtype=torch.long
            )
        if batch_indices.ndim != 1 or batch_indices.shape[0] != num_candidates:
            raise ValueError("batch_indices must have shape [N]")
        if num_candidates and (
            (batch_indices < 0).any()
            or (batch_indices >= coarse_logits.shape[0]).any()
        ):
            raise ValueError("batch_indices contain an index outside coarse_logits")
        if num_candidates == 0:
            empty = coarse_logits.new_empty((0,))
            return {
                "refined_logits": coarse_logits,
                "deltas": empty,
                "gates": gates.to(
                    device=coarse_logits.device, dtype=coarse_logits.dtype
                ),
                "required_deltas": empty,
                "unclipped_required_deltas": empty,
                "peak_logits": empty,
                "active_support": torch.zeros(
                    (0, *coarse_logits.shape[-2:]),
                    device=coarse_logits.device,
                    dtype=torch.bool,
                ),
            }

        if tuple(action_masks.shape[-2:]) != tuple(coarse_logits.shape[-2:]):
            action_masks = F.interpolate(
                action_masks.unsqueeze(1).float(),
                size=coarse_logits.shape[-2:],
                mode="nearest",
            )[:, 0]
        masks = (action_masks > 0.5).to(device=coarse_logits.device)
        if not masks.flatten(1).any(dim=1).all():
            raise ValueError("every action mask must contain at least one pixel")
        gates = gates.to(device=coarse_logits.device, dtype=coarse_logits.dtype)

        candidate_logits = coarse_logits[batch_indices, 0]
        negative_infinity = torch.full_like(candidate_logits, float("-inf"))
        peak_logits = torch.where(
            masks, candidate_logits, negative_infinity
        ).flatten(1).amax(dim=1)
        remove_logit = torch.logit(
            peak_logits.new_tensor(self.remove_threshold), eps=self.eps
        )
        required_deltas = (remove_logit - peak_logits).clamp(max=0.0)
        deltas = gates * required_deltas

        output_logit = torch.logit(
            coarse_logits.new_tensor(self.output_threshold), eps=self.eps
        )
        active_support = masks & (candidate_logits.detach() > output_logit)
        per_candidate_correction = (
            active_support.to(dtype=coarse_logits.dtype) * deltas[:, None, None]
        )
        correction = coarse_logits.new_zeros(
            (coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3])
        )
        correction = correction.index_add(
            0, batch_indices, per_candidate_correction
        )
        refined_logits = coarse_logits + correction.unsqueeze(1)
        for name, value in (
            ("peak_logits", peak_logits),
            ("required_deltas", required_deltas),
            ("deltas", deltas),
            ("refined_logits", refined_logits),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return {
            "refined_logits": refined_logits,
            "deltas": deltas,
            "gates": gates,
            "required_deltas": required_deltas,
            "unclipped_required_deltas": required_deltas,
            "peak_logits": peak_logits,
            "active_support": active_support,
        }


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
        action_threshold: float = 0.90,
        remove_threshold: float = 0.45,
        action_temperature: float = 0.05,
        output_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_classes = num_classes
        if rectifier not in ("suppression_only", "threshold_aware"):
            raise ValueError(
                "rectifier must be 'suppression_only' or 'threshold_aware'"
            )
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
        if rectifier == "suppression_only":
            suppression_limit = max_delta if max_suppression is None else max_suppression
            self.rectifier = SafeClutterSuppressor(
                max_suppression=suppression_limit,
                gate_margin=gate_margin,
                gate_temperature=gate_temperature,
                eps=eps,
            )
        else:
            self.rectifier = ThresholdAwareClutterSuppressor(
                action_threshold=action_threshold,
                remove_threshold=remove_threshold,
                soft_temperature=action_temperature,
                max_suppression=max_suppression,
                output_threshold=output_threshold,
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

        rectifier_outputs = self.rectifier(
            coarse_logits,
            target_scores,
            clutter_scores,
            masks,
            batch_indices,
        )
        if isinstance(rectifier_outputs, Mapping):
            refined_logits = rectifier_outputs["refined_logits"]
            deltas = rectifier_outputs["deltas"]
            gates = rectifier_outputs["gates"]
        else:
            refined_logits, deltas, gates = rectifier_outputs
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
        if isinstance(rectifier_outputs, Mapping):
            for key in (
                "required_deltas",
                "unclipped_required_deltas",
                "peak_logits",
                "active_support",
            ):
                candidate_outputs[key] = rectifier_outputs[key]
        return refined_logits, candidate_outputs


class SCACRRModule(nn.Module):
    """Selective Component-Aligned CCRR.

    The two mask roles are intentionally disjoint:

    * ``proposal_masks`` define core/ring and multi-scale encoder features;
    * ``action_masks`` define mask-quality features and are the only pixels
      the suppressor may modify.

    One proposal/action pair must be supplied for every final output
    component.  ``candidate_metadata`` is forwarded for auditing but never
    participates in the prediction, which keeps the model contract explicit.
    """

    def __init__(
        self,
        feature_channels: int,
        num_scales: int = 4,
        roi_size: int | tuple[int, int] = 7,
        hidden_dim: int = 64,
        context_scale: float = 3.0,
        min_context_size: float = 15.0,
        dropout: float = 0.3,
        risk_threshold: float = 2.0,
        quality_veto_threshold: float = 0.5,
        risk_alpha: float = 1.0,
        action_temperature: float = 0.05,
        remove_threshold: float = 0.45,
        output_threshold: float = 0.5,
        mask_hidden_dim: int = 16,
        eps: float = 1e-6,
        zero_effect_initialization: bool = True,
        pooling_mode: str = "avg",
        topk_ratio: float = 0.125,
        minimum_topk: int = 1,
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.num_classes = 2
        self.output_threshold = float(output_threshold)
        self.encoder = CandidateContextEncoder(
            feature_channels=feature_channels,
            num_scales=num_scales,
            roi_size=roi_size,
            hidden_dim=hidden_dim,
            context_scale=context_scale,
            min_context_size=min_context_size,
            pooling_mode=pooling_mode,
            topk_ratio=topk_ratio,
            minimum_topk=minimum_topk,
        )
        self.mask_encoder = ActionMaskEncoder(
            roi_size=roi_size,
            hidden_dim=mask_hidden_dim,
        )
        self.geometry_dim = 8
        self.reliability_head = SelectiveReliabilityHead(
            input_dim=(
                self.encoder.output_dim
                + self.mask_encoder.output_dim
                + self.geometry_dim
            ),
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_effect_initialization=zero_effect_initialization,
        )
        self.risk_gate = SelectiveRiskGate(
            risk_threshold=risk_threshold,
            quality_veto_threshold=quality_veto_threshold,
            risk_alpha=risk_alpha,
            temperature=action_temperature,
            eps=eps,
        )
        self.rectifier = ComponentAlignedSuppressor(
            remove_threshold=remove_threshold,
            output_threshold=output_threshold,
            eps=eps,
        )

    @property
    def selective_head(self) -> SelectiveReliabilityHead:
        """Read-only descriptive alias without duplicate state-dict keys."""

        return self.reliability_head

    @staticmethod
    def _validate_finite_payload(name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                SCACRRModule._validate_finite_payload(f"{name}.{key}", item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                SCACRRModule._validate_finite_payload(f"{name}[{index}]", item)

    @staticmethod
    def _geometry_features(
        coarse_logits: Tensor,
        proposal_masks: Tensor,
        action_masks: Tensor,
        action_boxes: Tensor,
        batch_indices: Tensor,
        scale_features: Tensor,
    ) -> Tensor:
        """Return bounded shape/confidence features for each exact component."""

        count, height, width = action_masks.shape
        if count == 0:
            return coarse_logits.new_empty((0, 8))
        probability = coarse_logits.sigmoid()[batch_indices, 0]
        action_float = action_masks.to(dtype=probability.dtype)
        action_area = action_float.flatten(1).sum(dim=1).clamp_min(1.0)
        total_area = float(height * width)
        area_log_fraction = torch.log1p(action_area) / math.log1p(total_area)

        box_width = (action_boxes[:, 3] - action_boxes[:, 1]).clamp_min(1.0)
        box_height = (action_boxes[:, 4] - action_boxes[:, 2]).clamp_min(1.0)
        bbox_area = (box_width * box_height).clamp_min(1.0)
        compactness = action_area / bbox_area

        masked_probability = probability * action_float
        action_peak = masked_probability.flatten(1).amax(dim=1)
        action_mean = masked_probability.flatten(1).sum(dim=1) / action_area

        ring_masks = proposal_masks & ~action_masks
        ring_float = ring_masks.to(dtype=probability.dtype)
        ring_area = ring_float.flatten(1).sum(dim=1)
        ring_mean = (probability * ring_float).flatten(1).sum(dim=1) / ring_area.clamp_min(1.0)
        ring_mean = torch.where(ring_area > 0, ring_mean, action_mean)
        core_ring_contrast = action_mean - ring_mean
        scale_variance = scale_features[:, -1]

        geometry = torch.stack(
            (
                area_log_fraction,
                box_width / float(width),
                box_height / float(height),
                compactness,
                action_peak,
                action_mean,
                scale_variance,
                core_ring_contrast,
            ),
            dim=1,
        )
        if geometry.shape != (count, 8) or not torch.isfinite(geometry).all():
            raise ValueError("component geometry features are invalid")
        return geometry

    def forward(
        self,
        feature_map: Tensor,
        coarse_logits: Tensor,
        multi_scale_logits: Any,
        proposal_boxes: Any,
        proposal_masks: Any = None,
        action_masks: Any = None,
        action_boxes: Any = None,
        candidate_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B,C,H_f,W_f]")
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError("coarse_logits must have shape [B,1,H,W]")
        if not torch.is_floating_point(feature_map) or not torch.is_floating_point(
            coarse_logits
        ):
            raise TypeError("feature_map and coarse_logits must be floating point")
        if feature_map.shape[0] != coarse_logits.shape[0]:
            raise ValueError("feature_map and coarse_logits must have the same batch size")
        if feature_map.device != coarse_logits.device:
            raise ValueError("feature_map and coarse_logits must be on the same device")
        if feature_map.dtype != coarse_logits.dtype:
            raise ValueError("feature_map and coarse_logits must have the same dtype")
        if not torch.isfinite(feature_map).all():
            raise ValueError("feature_map contains NaN or infinite values")
        if not torch.isfinite(coarse_logits).all():
            raise ValueError("coarse_logits contain NaN or infinite values")

        candidate_record: Mapping[str, Any] | None = None
        if isinstance(proposal_boxes, Mapping):
            candidate_record = proposal_boxes
            if proposal_masks is None:
                proposal_masks = candidate_record.get(
                    "proposal_masks", candidate_record.get("masks")
                )
            if action_masks is None:
                action_masks = candidate_record.get("action_masks")
            if action_boxes is None:
                action_boxes = candidate_record.get("action_boxes")
            proposal_boxes = candidate_record.get("boxes")
            if candidate_metadata is None:
                candidate_metadata = candidate_record
        if candidate_metadata is not None and not isinstance(candidate_metadata, Mapping):
            raise TypeError("candidate_metadata must be a mapping or None")

        self._validate_finite_payload("proposal_masks", proposal_masks)
        self._validate_finite_payload("action_masks", action_masks)
        scale_list = _as_scale_list(multi_scale_logits, self.num_scales)
        for index, scale_logits in enumerate(scale_list):
            if not isinstance(scale_logits, Tensor):
                raise TypeError("each item in multi_scale_logits must be a tensor")
            if not torch.is_floating_point(scale_logits):
                raise TypeError("multi_scale_logits must be floating point")
            if not torch.isfinite(scale_logits).all():
                raise ValueError(
                    f"multi_scale_logits[{index}] contains NaN or infinite values"
                )

        batch_size = coarse_logits.shape[0]
        output_hw = tuple(coarse_logits.shape[-2:])
        boxes = _normalise_boxes(
            proposal_boxes,
            batch_size=batch_size,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        boxes = _sanitize_boxes(boxes, output_hw, batch_size)
        num_candidates = boxes.shape[0]
        proposals = _normalise_masks(
            proposal_masks,
            num_candidates=num_candidates,
            batch_size=batch_size,
            output_hw=output_hw,
            device=coarse_logits.device,
        )
        actions = _normalise_masks(
            action_masks,
            num_candidates=num_candidates,
            batch_size=batch_size,
            output_hw=output_hw,
            device=coarse_logits.device,
        )
        if num_candidates and not proposals.flatten(1).any(dim=1).all():
            raise ValueError("every proposal mask must contain at least one pixel")
        if num_candidates and not actions.flatten(1).any(dim=1).all():
            raise ValueError("every action mask must contain at least one pixel")
        batch_indices = boxes[:, 0].long()

        if action_boxes is None:
            normalised_action_boxes = ActionMaskEncoder.boxes_from_masks(
                actions,
                batch_indices,
                dtype=feature_map.dtype,
            )
        else:
            normalised_action_boxes = _normalise_boxes(
                action_boxes,
                batch_size=batch_size,
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            if normalised_action_boxes.shape[0] != num_candidates:
                raise ValueError(
                    "action box count must match proposal/action mask count"
                )
            normalised_action_boxes = _sanitize_boxes(
                normalised_action_boxes, output_hw, batch_size
            )
            if not torch.equal(
                normalised_action_boxes[:, 0].long(), batch_indices
            ):
                raise ValueError(
                    "proposal and action boxes must have identical candidate ordering"
                )

        # Proposal masks are the sole source of scale and context features.
        scale_features = _extract_scale_features(
            scale_list,
            proposals,
            batch_indices,
            batch_size=batch_size,
            output_hw=output_hw,
            num_scales=self.num_scales,
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        proposal_relation_features = self.encoder(
            feature_map,
            boxes,
            candidate_masks=proposals,
            scale_features=scale_features,
            image_hw=output_hw,
        )
        # Exact action masks are used only for the shape/confidence branch.
        action_mask_features = self.mask_encoder(
            coarse_logits,
            actions,
            normalised_action_boxes,
            batch_indices,
        )
        geometry_features = self._geometry_features(
            coarse_logits,
            proposals,
            actions,
            normalised_action_boxes,
            batch_indices,
            scale_features,
        )
        relation_features = torch.cat(
            (
                proposal_relation_features,
                action_mask_features,
                geometry_features,
            ),
            dim=1,
        )
        if not torch.isfinite(relation_features).all():
            raise ValueError("relation_features contain NaN or infinite values")

        head_outputs = self.reliability_head(relation_features)
        clutter_scores = head_outputs["clutter_probability"]
        target_quality = head_outputs["target_quality"]
        target_scores = 1.0 - clutter_scores
        class_probs = torch.stack((target_scores, clutter_scores), dim=1)
        # Symmetric logits preserve softmax(class_logits) == class_probs while
        # retaining the head's scalar binary logit.
        class_logits = torch.stack(
            (
                -0.5 * head_outputs["clutter_logits"],
                0.5 * head_outputs["clutter_logits"],
            ),
            dim=1,
        )
        if "target_guard" in head_outputs:
            gate_outputs = self.risk_gate(
                clutter_scores,
                target_quality,
                head_outputs["target_guard"],
            )
        else:
            gate_outputs = self.risk_gate(clutter_scores, target_quality)
        rectifier_outputs = self.rectifier(
            coarse_logits,
            gate_outputs["gate"],
            actions,
            batch_indices,
        )

        candidate_outputs: dict[str, Any] = {
            # V1/V1.1-compatible fields.
            "class_logits": class_logits,
            "class_probs": class_probs,
            "target_scores": target_scores,
            "clutter_scores": clutter_scores,
            "uncertain_scores": clutter_scores.new_zeros((num_candidates,)),
            "deltas": rectifier_outputs["deltas"],
            "gates": rectifier_outputs["gates"],
            "boxes": boxes,
            "batch_indices": batch_indices,
            "scale_features": scale_features,
            "relation_features": relation_features,
            # SCA-specific, role-explicit fields.
            "proposal_boxes": boxes,
            "action_boxes": normalised_action_boxes,
            "proposal_masks": proposals,
            "action_masks": actions,
            # Compatibility aliases action candidates to their exact masks.
            "candidate_masks": actions,
            "proposal_relation_features": proposal_relation_features,
            "action_mask_features": action_mask_features,
            "geometry_features": geometry_features,
            "shared_feature": head_outputs["shared_feature"],
            "clutter_logits": head_outputs["clutter_logits"],
            "clutter_probability": clutter_scores,
            "target_quality_logits": head_outputs["target_quality_logits"],
            "quality_logits": head_outputs["target_quality_logits"],
            "target_quality": target_quality,
            "risk_score": gate_outputs["risk_score"],
            "quality_veto": gate_outputs["quality_veto"],
            "soft_action": gate_outputs["soft_action"],
            "required_deltas": rectifier_outputs["required_deltas"],
            "unclipped_required_deltas": rectifier_outputs[
                "unclipped_required_deltas"
            ],
            "peak_logits": rectifier_outputs["peak_logits"],
            "active_support": rectifier_outputs["active_support"],
        }
        for key in (
            "target_guard_logits",
            "target_guard",
            "guard_logit",
            "guard_veto",
            "soft_quality_allow",
            "soft_guard_allow",
        ):
            if key in head_outputs:
                candidate_outputs[key] = head_outputs[key]
            elif key in gate_outputs:
                candidate_outputs[key] = gate_outputs[key]
        if candidate_metadata is not None:
            candidate_outputs["candidate_metadata"] = candidate_metadata
            # Preserve useful per-candidate audit identifiers/scores at the
            # top level without allowing metadata to overwrite model fields.
            for key, value in candidate_metadata.items():
                if key not in candidate_outputs and isinstance(value, Tensor):
                    candidate_outputs[key] = value.to(device=coarse_logits.device)
        return rectifier_outputs["refined_logits"], candidate_outputs


class TargetGuardedSCACRRModule(SCACRRModule):
    """SCA-CCRR with an independent high-recall target-presence veto."""

    def __init__(
        self,
        feature_channels: int,
        num_scales: int = 4,
        roi_size: int | tuple[int, int] = 7,
        hidden_dim: int = 64,
        context_scale: float = 3.0,
        min_context_size: float = 15.0,
        dropout: float = 0.3,
        risk_threshold: float = 2.0,
        quality_veto_threshold: float = 0.2,
        guard_veto_threshold: float = 0.2,
        risk_alpha: float = 1.0,
        guard_alpha: float = 1.0,
        action_temperature: float = 0.05,
        quality_temperature: float = 0.05,
        guard_temperature: float = 0.05,
        remove_threshold: float = 0.45,
        output_threshold: float = 0.5,
        mask_hidden_dim: int = 16,
        eps: float = 1e-6,
        zero_effect_initialization: bool = True,
        pooling_mode: str = "avg_max_topk",
        topk_ratio: float = 0.125,
        minimum_topk: int = 1,
    ) -> None:
        super().__init__(
            feature_channels=feature_channels,
            num_scales=num_scales,
            roi_size=roi_size,
            hidden_dim=hidden_dim,
            context_scale=context_scale,
            min_context_size=min_context_size,
            dropout=dropout,
            risk_threshold=risk_threshold,
            quality_veto_threshold=quality_veto_threshold,
            risk_alpha=risk_alpha,
            action_temperature=action_temperature,
            remove_threshold=remove_threshold,
            output_threshold=output_threshold,
            mask_hidden_dim=mask_hidden_dim,
            eps=eps,
            zero_effect_initialization=zero_effect_initialization,
            pooling_mode=pooling_mode,
            topk_ratio=topk_ratio,
            minimum_topk=minimum_topk,
        )
        relation_dim = (
            self.encoder.output_dim
            + self.mask_encoder.output_dim
            + self.geometry_dim
        )
        base_head = self.reliability_head
        # Reuse the already-created SCA shared/clutter/quality modules so a
        # fixed seed gives TG-SCA exactly the same common initialization as
        # E1.  Constructing the extra zero-initialized head inside fork_rng
        # also prevents an otherwise irrelevant RNG offset in data shuffling
        # and dropout streams.
        with torch.random.fork_rng(devices=[]):
            guarded_head = TargetGuardedReliabilityHead(
                input_dim=relation_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                zero_effect_initialization=zero_effect_initialization,
            )
        guarded_head.shared = base_head.shared
        guarded_head.clutter_head = base_head.clutter_head
        guarded_head.quality_head = base_head.quality_head
        self.reliability_head = guarded_head
        self.risk_gate = TargetGuardedRiskGate(
            risk_threshold=risk_threshold,
            quality_veto_threshold=quality_veto_threshold,
            guard_veto_threshold=guard_veto_threshold,
            risk_alpha=risk_alpha,
            guard_alpha=guard_alpha,
            action_temperature=action_temperature,
            quality_temperature=quality_temperature,
            guard_temperature=guard_temperature,
            eps=eps,
        )


__all__ = [
    "MaskedHybridPool",
    "CandidateContextEncoder",
    "ReliabilityHead",
    "SelectiveReliabilityHead",
    "TargetGuardedReliabilityHead",
    "ActionMaskEncoder",
    "InstanceLogitRectifier",
    "SafeClutterSuppressor",
    "ThresholdAwareClutterSuppressor",
    "SelectiveRiskGate",
    "TargetGuardedRiskGate",
    "ComponentAlignedSuppressor",
    "CCRRModule",
    "SCACRRModule",
    "TargetGuardedSCACRRModule",
]
