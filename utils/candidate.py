"""Candidate generation and bookkeeping utilities for CCRR.

The module uses one flattened candidate dimension throughout.  A candidate
dictionary has the following core fields::

    masks:          bool tensor [N, H, W]
    boxes:          float32 tensor [N, 5], (batch, x1, y1, x2, y2)
    batch_indices:  int64 tensor [N]

Box coordinates use half-open image bounds, i.e. ``x2`` and ``y2`` are one
past the final mask pixel.  This is also a valid input format for
``torchvision.ops.roi_align``.  Empty candidate sets retain all dimensions.

Reliability labels always use the order expected by CCRR's three-way head:
``TARGET_LABEL=0``, ``CLUTTER_LABEL=1``, ``UNCERTAIN_LABEL=2``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label as connected_component_labels


TARGET_LABEL = 0
CLUTTER_LABEL = 1
UNCERTAIN_LABEL = 2
LABEL_NAMES = ("target", "clutter", "uncertain")


def _as_binary_logits(logits: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize binary logits to [B, 1, H, W]."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not logits.is_floating_point():
        raise TypeError(f"{name} must contain floating-point logits")
    if logits.ndim == 2:
        logits = logits[None, None]
    elif logits.ndim == 3:
        logits = logits[:, None]
    elif logits.ndim != 4:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W]")
    if logits.shape[1] != 1:
        raise ValueError(f"{name} must have exactly one channel, got {logits.shape[1]}")
    return logits


def _logits_sequence(multi_scale_logits: Sequence[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(multi_scale_logits, torch.Tensor):
        return [multi_scale_logits]
    if not isinstance(multi_scale_logits, Sequence):
        raise TypeError("multi_scale_logits must be a tensor or a sequence of tensors")
    return list(multi_scale_logits)


def _flatten_candidate_masks(
    candidate_masks: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return support, batch indices, original values, and optional scores.

    Tensor layouts are interpreted as [H,W], [N,H,W] (single image), or
    [B,K,H,W].  A sequence contains one [K,H,W] tensor per image.  A mapping
    may explicitly provide ``masks``/``candidate_masks`` and
    ``batch_indices``; this is the unambiguous representation for variable
    numbers of candidates in a batch.
    """
    explicit_batch_indices = None
    explicit_scores = None
    raw_masks = candidate_masks
    if isinstance(candidate_masks, Mapping):
        raw_masks = candidate_masks.get("masks", candidate_masks.get("candidate_masks"))
        if raw_masks is None:
            raise KeyError("candidate mapping must contain 'masks' or 'candidate_masks'")
        explicit_batch_indices = candidate_masks.get("batch_indices")
        explicit_scores = candidate_masks.get(
            "scores", candidate_masks.get("mean_scores", candidate_masks.get("candidate_scores"))
        )

    if isinstance(raw_masks, (list, tuple)):
        if not raw_masks:
            values = torch.empty((0, 0, 0), dtype=torch.bool)
            batch_indices = torch.empty((0,), dtype=torch.long)
        else:
            flattened = []
            indices = []
            image_hw = None
            target_device = None
            for batch_index, item in enumerate(raw_masks):
                tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.ndim != 3:
                    raise ValueError("each candidate-mask list item must have shape [H,W] or [K,H,W]")
                if image_hw is None:
                    image_hw = tuple(tensor.shape[-2:])
                    target_device = tensor.device
                elif tuple(tensor.shape[-2:]) != image_hw:
                    raise ValueError("all candidate masks must have the same spatial shape")
                tensor = tensor.to(device=target_device)
                flattened.append(tensor)
                indices.append(
                    torch.full((tensor.shape[0],), batch_index, dtype=torch.long, device=target_device)
                )
            values = torch.cat(flattened, dim=0)
            batch_indices = torch.cat(indices, dim=0)
    else:
        values = raw_masks if isinstance(raw_masks, torch.Tensor) else torch.as_tensor(raw_masks)
        if values.ndim == 1 and values.numel() == 0:
            values = values.reshape(0, 0, 0)
        if values.ndim == 2:
            values = values.unsqueeze(0)
            batch_indices = torch.zeros((1,), dtype=torch.long, device=values.device)
        elif values.ndim == 3:
            batch_indices = torch.zeros((values.shape[0],), dtype=torch.long, device=values.device)
        elif values.ndim == 4:
            batch_size, candidates_per_image, height, width = values.shape
            values = values.reshape(batch_size * candidates_per_image, height, width)
            batch_indices = torch.arange(batch_size, device=values.device).repeat_interleave(
                candidates_per_image
            )
        else:
            raise ValueError("candidate_masks must have shape [H,W], [N,H,W], or [B,K,H,W]")

    if explicit_batch_indices is not None:
        batch_indices = torch.as_tensor(explicit_batch_indices, dtype=torch.long, device=values.device)
        if batch_indices.ndim != 1 or batch_indices.shape[0] != values.shape[0]:
            raise ValueError("batch_indices must have shape [N]")
    if torch.any(batch_indices < 0):
        raise ValueError("batch_indices must be non-negative")

    scores = None
    if explicit_scores is not None:
        scores = torch.as_tensor(explicit_scores, device=values.device)
        if scores.ndim != 1 or scores.shape[0] != values.shape[0]:
            raise ValueError("candidate scores must have shape [N]")
        if not scores.is_floating_point():
            scores = scores.float()

    return values != 0, batch_indices, values, scores


def _component_arrays(binary_mask: torch.Tensor) -> list[np.ndarray]:
    """Find deterministic 8-connected components of one [H,W] mask."""
    labels = connected_component_labels(
        binary_mask.detach().to(device="cpu", dtype=torch.uint8).numpy(),
        connectivity=2,
    )
    return [labels == component_id for component_id in range(1, int(labels.max()) + 1)]


def generate_candidates(
    coarse_logits: torch.Tensor,
    multi_scale_logits: Sequence[torch.Tensor] | torch.Tensor,
    threshold_low: float,
    min_area: int,
    max_area: int | None,
) -> dict[str, torch.Tensor]:
    """Generate batched connected-component candidates from scale consensus.

    Each scale is resized to ``coarse_logits.shape[-2:]`` and converted to a
    probability.  Components are extracted from the mean probability map at
    ``P_mean > threshold_low`` and filtered with inclusive area bounds.

    Returns a dictionary whose candidate-wise tensors remain on the coarse
    logits device.  ``scores`` and ``peak_scores`` summarize the multi-scale
    mean map; ``coarse_scores`` summarizes the fused coarse prediction.
    ``scale_responses`` is [N,L], and ``scale_variance`` is the candidate mean
    of the pixel-wise population variance across L scales.
    """
    if not 0.0 <= float(threshold_low) <= 1.0:
        raise ValueError("threshold_low must be in [0, 1]")
    if int(min_area) != min_area or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    if max_area is not None and (int(max_area) != max_area or max_area < min_area):
        raise ValueError("max_area must be None or an integer >= min_area")

    coarse = _as_binary_logits(coarse_logits, "coarse_logits")
    batch_size, _, height, width = coarse.shape
    scales = _logits_sequence(multi_scale_logits)
    if not scales:
        scales = [coarse]

    resized_probabilities = []
    for scale_index, scale_logits in enumerate(scales):
        scale = _as_binary_logits(scale_logits, f"multi_scale_logits[{scale_index}]")
        if scale.shape[0] != batch_size:
            raise ValueError("all multi-scale logits must have the same batch size as coarse_logits")
        scale = scale.to(device=coarse.device, dtype=coarse.dtype)
        if scale.shape[-2:] != (height, width):
            scale = F.interpolate(scale, size=(height, width), mode="bilinear", align_corners=False)
        probability = torch.sigmoid(scale)
        resized_probabilities.append(probability[:, 0])

    scale_stack = torch.stack(resized_probabilities, dim=0)  # [L,B,H,W]
    mean_probability = scale_stack.mean(dim=0)
    variance_map = scale_stack.var(dim=0, unbiased=False)
    coarse_probability = torch.sigmoid(coarse[:, 0])

    masks = []
    batch_indices_list = []
    scores = []
    peak_scores = []
    coarse_scores = []
    coarse_peak_scores = []
    areas = []
    scale_responses = []
    scale_variances = []

    binary_maps = mean_probability > float(threshold_low)
    for batch_index in range(batch_size):
        for component_array in _component_arrays(binary_maps[batch_index]):
            area = int(component_array.sum())
            if area < min_area or (max_area is not None and area > max_area):
                continue
            mask = torch.as_tensor(component_array, dtype=torch.bool, device=coarse.device)
            mask_values = mask
            masks.append(mask)
            batch_indices_list.append(batch_index)
            areas.append(area)
            scores.append(mean_probability[batch_index][mask_values].mean())
            peak_scores.append(mean_probability[batch_index][mask_values].amax())
            coarse_scores.append(coarse_probability[batch_index][mask_values].mean())
            coarse_peak_scores.append(coarse_probability[batch_index][mask_values].amax())
            scale_responses.append(scale_stack[:, batch_index, mask_values].mean(dim=1))
            scale_variances.append(variance_map[batch_index][mask_values].mean())

    number_of_scales = len(scales)
    probability_dtype = mean_probability.dtype
    if masks:
        mask_tensor = torch.stack(masks, dim=0)
        batch_indices = torch.tensor(batch_indices_list, dtype=torch.long, device=coarse.device)
        area_tensor = torch.tensor(areas, dtype=torch.long, device=coarse.device)
        score_tensor = torch.stack(scores)
        peak_score_tensor = torch.stack(peak_scores)
        coarse_score_tensor = torch.stack(coarse_scores)
        coarse_peak_tensor = torch.stack(coarse_peak_scores)
        response_tensor = torch.stack(scale_responses)
        variance_tensor = torch.stack(scale_variances)
    else:
        mask_tensor = torch.empty((0, height, width), dtype=torch.bool, device=coarse.device)
        batch_indices = torch.empty((0,), dtype=torch.long, device=coarse.device)
        area_tensor = torch.empty((0,), dtype=torch.long, device=coarse.device)
        score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        peak_score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        coarse_score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        coarse_peak_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        response_tensor = torch.empty(
            (0, number_of_scales), dtype=probability_dtype, device=coarse.device
        )
        variance_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)

    boxes = masks_to_roi_boxes({"masks": mask_tensor, "batch_indices": batch_indices})
    return {
        "masks": mask_tensor,
        "candidate_masks": mask_tensor,
        "boxes": boxes,
        "batch_indices": batch_indices,
        "scores": score_tensor,
        "peak_scores": peak_score_tensor,
        "coarse_scores": coarse_score_tensor,
        "coarse_peak_scores": coarse_peak_tensor,
        "areas": area_tensor,
        "scale_responses": response_tensor,
        "scale_variance": variance_tensor,
        "mean_probability": mean_probability.unsqueeze(1),
        "scale_variance_map": variance_map.unsqueeze(1),
    }


def map_action_components_to_proposals(
    action_masks: torch.Tensor,
    action_batch_indices: torch.Tensor,
    proposal_masks: torch.Tensor,
    proposal_batch_indices: torch.Tensor,
    *,
    fallback_dilation: int = 1,
) -> dict[str, torch.Tensor]:
    """Map every exact output component to one feature proposal.

    Mapping is restricted to the same image and maximizes overlap pixels;
    ties retain proposal scan order.  An action without overlap receives a
    deterministic 8-neighborhood dilation fallback.  A proposal may describe
    multiple actions, but every action appears exactly once in the output.
    """

    if action_masks.ndim != 3 or proposal_masks.ndim != 3:
        raise ValueError("action_masks and proposal_masks must have shape [N,H,W]")
    if tuple(action_masks.shape[-2:]) != tuple(proposal_masks.shape[-2:]):
        raise ValueError("action_masks and proposal_masks must have the same spatial shape")
    if action_batch_indices.shape != (action_masks.shape[0],):
        raise ValueError("action_batch_indices must have shape [N_action]")
    if proposal_batch_indices.shape != (proposal_masks.shape[0],):
        raise ValueError("proposal_batch_indices must have shape [N_proposal]")
    if action_masks.device != proposal_masks.device:
        raise ValueError("action_masks and proposal_masks must be on the same device")
    if int(fallback_dilation) != fallback_dilation or fallback_dilation < 0:
        raise ValueError("fallback_dilation must be a non-negative integer")

    device = action_masks.device
    action_masks = action_masks.bool()
    proposal_masks = proposal_masks.bool()
    action_batch_indices = action_batch_indices.to(device=device, dtype=torch.long)
    proposal_batch_indices = proposal_batch_indices.to(device=device, dtype=torch.long)
    selected_proposals = []
    selected_ids = []
    fallback_flags = []
    overlap_pixels = []
    mapping_ious = []
    proposal_has_action_overlap = torch.zeros(
        proposal_masks.shape[0], dtype=torch.bool, device=device
    )

    for action_mask, batch_index in zip(action_masks, action_batch_indices):
        eligible_ids = torch.nonzero(
            proposal_batch_indices == batch_index, as_tuple=False
        ).flatten()
        chosen_id = -1
        chosen_overlap = 0
        if eligible_ids.numel():
            intersections = (
                proposal_masks[eligible_ids] & action_mask.unsqueeze(0)
            ).flatten(1).sum(dim=1)
            proposal_has_action_overlap[eligible_ids] |= intersections > 0
            best_position = int(torch.argmax(intersections).item())
            chosen_overlap = int(intersections[best_position].item())
            if chosen_overlap > 0:
                chosen_id = int(eligible_ids[best_position].item())

        if chosen_id >= 0:
            proposal_mask = proposal_masks[chosen_id]
            fallback = False
        else:
            if fallback_dilation:
                kernel_size = 2 * fallback_dilation + 1
                proposal_mask = F.max_pool2d(
                    action_mask[None, None].float(),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=fallback_dilation,
                )[0, 0] > 0
            else:
                proposal_mask = action_mask.clone()
            fallback = True
            chosen_overlap = int(action_mask.sum().item())

        intersection = int((proposal_mask & action_mask).sum().item())
        union = int((proposal_mask | action_mask).sum().item())
        selected_proposals.append(proposal_mask)
        selected_ids.append(chosen_id)
        fallback_flags.append(fallback)
        overlap_pixels.append(chosen_overlap)
        mapping_ious.append(intersection / union if union else 0.0)

    height, width = action_masks.shape[-2:]
    if selected_proposals:
        selected_tensor = torch.stack(selected_proposals)
    else:
        selected_tensor = torch.empty((0, height, width), dtype=torch.bool, device=device)
    return {
        "proposal_masks": selected_tensor,
        "proposal_component_ids": torch.tensor(
            selected_ids, dtype=torch.long, device=device
        ),
        "proposal_is_fallback": torch.tensor(
            fallback_flags, dtype=torch.bool, device=device
        ),
        "proposal_action_overlap_pixels": torch.tensor(
            overlap_pixels, dtype=torch.long, device=device
        ),
        "proposal_to_action_iou": torch.tensor(
            mapping_ious, dtype=torch.float32, device=device
        ),
        "raw_proposal_has_action_overlap": proposal_has_action_overlap,
    }


def generate_component_aligned_candidates(
    coarse_logits: torch.Tensor,
    multi_scale_logits: Sequence[torch.Tensor] | torch.Tensor,
    proposal_threshold: float = 0.2,
    output_threshold: float = 0.5,
    min_area: int = 1,
    max_area: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create one action candidate per exact final-output component.

    ``proposal_masks`` are low-threshold multi-scale regions used only for
    feature/context encoding.  ``action_masks`` are the complete 8-connected
    components of ``sigmoid(coarse_logits) > output_threshold`` and are the
    only masks that may be labelled, suppressed, or counted for FPPI.

    Area bounds apply only to feature proposals.  They never remove an action
    component from the final 0.5 output partition.
    """

    if not 0.0 <= float(output_threshold) <= 1.0:
        raise ValueError("output_threshold must be in [0,1]")
    coarse = _as_binary_logits(coarse_logits, "coarse_logits")
    batch_size, _, height, width = coarse.shape
    raw_proposals = generate_candidates(
        coarse_logits=coarse,
        multi_scale_logits=multi_scale_logits,
        threshold_low=proposal_threshold,
        min_area=min_area,
        max_area=max_area,
    )

    action_masks_list = []
    action_batch_indices_list = []
    action_local_ids = []
    output_binary = coarse.sigmoid()[:, 0] > float(output_threshold)
    for batch_index in range(batch_size):
        for local_id, component in enumerate(
            _component_arrays(output_binary[batch_index]), start=1
        ):
            action_masks_list.append(
                torch.as_tensor(component, dtype=torch.bool, device=coarse.device)
            )
            action_batch_indices_list.append(batch_index)
            action_local_ids.append(local_id)

    if action_masks_list:
        action_masks = torch.stack(action_masks_list)
        batch_indices = torch.tensor(
            action_batch_indices_list, dtype=torch.long, device=coarse.device
        )
    else:
        action_masks = torch.empty(
            (0, height, width), dtype=torch.bool, device=coarse.device
        )
        batch_indices = torch.empty((0,), dtype=torch.long, device=coarse.device)

    mapping = map_action_components_to_proposals(
        action_masks,
        batch_indices,
        raw_proposals["masks"],
        raw_proposals["batch_indices"],
        fallback_dilation=1,
    )
    proposal_masks = mapping["proposal_masks"]
    proposal_boxes = masks_to_roi_boxes(
        {"masks": proposal_masks, "batch_indices": batch_indices}
    )
    action_boxes = masks_to_roi_boxes(
        {"masks": action_masks, "batch_indices": batch_indices}
    )

    effective_scales = _logits_sequence(multi_scale_logits)
    if not effective_scales:
        effective_scales = [coarse]
    scale_features = extract_scale_features(
        effective_scales,
        {"masks": proposal_masks, "batch_indices": batch_indices},
    )
    scale_responses = scale_features[:, :-1]
    scale_variance = scale_features[:, -1]

    coarse_probability = coarse.sigmoid()[:, 0]
    mean_probability = raw_proposals["mean_probability"][:, 0]
    coarse_peak_scores = []
    coarse_mean_scores = []
    proposal_scores = []
    proposal_peak_scores = []
    action_areas = []
    proposal_areas = []
    for action_mask, proposal_mask, batch_index in zip(
        action_masks, proposal_masks, batch_indices
    ):
        action_values = coarse_probability[batch_index][action_mask]
        proposal_values = mean_probability[batch_index][proposal_mask]
        coarse_peak_scores.append(action_values.amax())
        coarse_mean_scores.append(action_values.mean())
        proposal_scores.append(proposal_values.mean())
        proposal_peak_scores.append(proposal_values.amax())
        action_areas.append(int(action_mask.sum().item()))
        proposal_areas.append(int(proposal_mask.sum().item()))

    probability_dtype = coarse.dtype
    if action_masks.shape[0]:
        coarse_peak_tensor = torch.stack(coarse_peak_scores)
        coarse_mean_tensor = torch.stack(coarse_mean_scores)
        proposal_score_tensor = torch.stack(proposal_scores)
        proposal_peak_tensor = torch.stack(proposal_peak_scores)
    else:
        coarse_peak_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        coarse_mean_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        proposal_score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        proposal_peak_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)

    raw_counts = torch.bincount(
        raw_proposals["batch_indices"], minlength=batch_size
    )
    inactive_counts = torch.bincount(
        raw_proposals["batch_indices"][
            ~mapping["raw_proposal_has_action_overlap"]
        ],
        minlength=batch_size,
    )
    number_of_actions = action_masks.shape[0]
    return {
        "boxes": proposal_boxes,
        "proposal_boxes": proposal_boxes,
        "action_boxes": action_boxes,
        "proposal_masks": proposal_masks,
        "action_masks": action_masks,
        "batch_indices": batch_indices,
        "action_component_ids": torch.arange(
            number_of_actions, dtype=torch.long, device=coarse.device
        ),
        "action_component_local_ids": torch.tensor(
            action_local_ids, dtype=torch.long, device=coarse.device
        ),
        "proposal_component_ids": mapping["proposal_component_ids"],
        "proposal_is_fallback": mapping["proposal_is_fallback"],
        "proposal_action_overlap_pixels": mapping[
            "proposal_action_overlap_pixels"
        ],
        "proposal_to_action_iou": mapping["proposal_to_action_iou"],
        "coarse_peak_scores": coarse_peak_tensor,
        "coarse_mean_scores": coarse_mean_tensor,
        "proposal_scores": proposal_score_tensor,
        "proposal_peak_scores": proposal_peak_tensor,
        "action_areas": torch.tensor(
            action_areas, dtype=torch.long, device=coarse.device
        ),
        "proposal_areas": torch.tensor(
            proposal_areas, dtype=torch.long, device=coarse.device
        ),
        "scale_responses": scale_responses,
        "scale_variance": scale_variance,
        "mean_probability": raw_proposals["mean_probability"],
        "scale_variance_map": raw_proposals["scale_variance_map"],
        "num_raw_proposals_per_image": raw_counts,
        "num_inactive_raw_proposals_per_image": inactive_counts,
        "raw_proposal_masks": raw_proposals["masks"],
        "raw_proposal_boxes": raw_proposals["boxes"],
        "raw_proposal_batch_indices": raw_proposals["batch_indices"],
        "raw_proposal_has_action_overlap": mapping[
            "raw_proposal_has_action_overlap"
        ],
    }


def generate_recovery_candidates(
    coarse_logits: torch.Tensor,
    multi_scale_logits: Sequence[torch.Tensor] | torch.Tensor,
    threshold_low: float = 0.05,
    threshold_high: float = 0.5,
    local_max_kernel: int = 5,
    proposal_size: int = 15,
    max_candidates_per_image: int = 32,
) -> dict[str, torch.Tensor]:
    """Generate deterministic low-score square proposals around local peaks.

    The evidence map is the maximum probability across the fused coarse output
    and every resized scale output.  A peak is eligible only when its evidence
    is strictly above ``threshold_low`` while the fused coarse probability at
    that location is strictly below ``threshold_high``.  This targets weak or
    scale-specific responses without duplicating already-positive coarse
    detections.  No GT information is used.
    """

    if not 0.0 <= float(threshold_low) <= 1.0:
        raise ValueError("threshold_low must be in [0,1]")
    if not 0.0 <= float(threshold_high) <= 1.0:
        raise ValueError("threshold_high must be in [0,1]")
    if threshold_low >= threshold_high:
        raise ValueError("threshold_low must be below threshold_high")
    if int(local_max_kernel) != local_max_kernel or local_max_kernel < 1 or local_max_kernel % 2 == 0:
        raise ValueError("local_max_kernel must be a positive odd integer")
    if int(proposal_size) != proposal_size or proposal_size < 1 or proposal_size % 2 == 0:
        raise ValueError("proposal_size must be a positive odd integer")
    if int(max_candidates_per_image) != max_candidates_per_image or max_candidates_per_image < 1:
        raise ValueError("max_candidates_per_image must be a positive integer")

    coarse = _as_binary_logits(coarse_logits, "coarse_logits")
    batch_size, _, height, width = coarse.shape
    scales = _logits_sequence(multi_scale_logits)
    if not scales:
        scales = [coarse]
    resized_logits = []
    for scale_index, scale_logits in enumerate(scales):
        scale = _as_binary_logits(scale_logits, f"multi_scale_logits[{scale_index}]")
        if scale.shape[0] != batch_size:
            raise ValueError("all multi-scale logits must have the same batch size as coarse_logits")
        scale = scale.to(device=coarse.device, dtype=coarse.dtype)
        if scale.shape[-2:] != (height, width):
            scale = F.interpolate(scale, size=(height, width), mode="bilinear", align_corners=False)
        resized_logits.append(scale[:, 0])

    scale_logit_stack = torch.stack(resized_logits, dim=0)  # [L,B,H,W]
    scale_stack = scale_logit_stack.sigmoid()
    coarse_probability = coarse.sigmoid()[:, 0]
    all_logits = torch.cat((coarse[:, 0].unsqueeze(0), scale_logit_stack), dim=0)
    evidence_logits, source_map = all_logits.max(dim=0)
    evidence_map = evidence_logits.sigmoid()
    pooled = F.max_pool2d(
        evidence_logits.unsqueeze(1),
        kernel_size=local_max_kernel,
        stride=1,
        padding=local_max_kernel // 2,
    )[:, 0]
    eligible = (
        (evidence_map > float(threshold_low))
        & (coarse_probability < float(threshold_high))
        & (evidence_logits == pooled)
    )

    masks = []
    batch_indices_list = []
    proposal_scores = []
    coarse_scores = []
    coarse_peak_scores = []
    areas = []
    scale_responses = []
    scale_variances = []
    source_scales = []
    peak_coordinates = []
    peaks_before_limit = torch.zeros(
        (batch_size,), dtype=torch.long, device=coarse.device
    )
    half_size = proposal_size // 2
    scale_variance_map = scale_stack.var(dim=0, unbiased=False)

    for batch_index in range(batch_size):
        plateau_components = _component_arrays(eligible[batch_index])
        peaks = []
        for plateau in plateau_components:
            plateau_tensor = torch.as_tensor(
                plateau, dtype=torch.bool, device=coarse.device
            )
            flat_indices = torch.nonzero(plateau_tensor.flatten(), as_tuple=False).flatten()
            plateau_values = evidence_map[batch_index].flatten()[flat_indices]
            best_flat = int(flat_indices[torch.argmax(plateau_values)].item())
            peak_y, peak_x = divmod(best_flat, width)
            peaks.append(
                (
                    -float(evidence_map[batch_index, peak_y, peak_x].item()),
                    peak_y,
                    peak_x,
                )
            )
        peaks.sort()
        peaks_before_limit[batch_index] = len(peaks)
        for negative_score, peak_y, peak_x in peaks[:max_candidates_per_image]:
            y1 = max(0, peak_y - half_size)
            y2 = min(height, peak_y + half_size + 1)
            x1 = max(0, peak_x - half_size)
            x2 = min(width, peak_x + half_size + 1)
            mask = torch.zeros((height, width), dtype=torch.bool, device=coarse.device)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            batch_indices_list.append(batch_index)
            proposal_scores.append(evidence_map[batch_index, peak_y, peak_x])
            coarse_scores.append(coarse_probability[batch_index][mask].mean())
            coarse_peak_scores.append(coarse_probability[batch_index][mask].amax())
            areas.append(int(mask.sum().item()))
            scale_responses.append(scale_stack[:, batch_index, mask].mean(dim=1))
            scale_variances.append(scale_variance_map[batch_index, mask].mean())
            # -1 denotes the fused coarse response; 0..L-1 are scale heads.
            source_scales.append(int(source_map[batch_index, peak_y, peak_x].item()) - 1)
            peak_coordinates.append((peak_y, peak_x))

    number_of_scales = len(scales)
    probability_dtype = coarse.dtype
    if masks:
        mask_tensor = torch.stack(masks)
        batch_indices = torch.tensor(batch_indices_list, dtype=torch.long, device=coarse.device)
        score_tensor = torch.stack(proposal_scores)
        coarse_score_tensor = torch.stack(coarse_scores)
        coarse_peak_tensor = torch.stack(coarse_peak_scores)
        area_tensor = torch.tensor(areas, dtype=torch.long, device=coarse.device)
        scale_response_tensor = torch.stack(scale_responses)
        scale_variance_tensor = torch.stack(scale_variances)
        source_scale_tensor = torch.tensor(source_scales, dtype=torch.long, device=coarse.device)
        peak_tensor = torch.tensor(peak_coordinates, dtype=torch.long, device=coarse.device)
    else:
        mask_tensor = torch.empty((0, height, width), dtype=torch.bool, device=coarse.device)
        batch_indices = torch.empty((0,), dtype=torch.long, device=coarse.device)
        score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        coarse_score_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        coarse_peak_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        area_tensor = torch.empty((0,), dtype=torch.long, device=coarse.device)
        scale_response_tensor = torch.empty(
            (0, number_of_scales), dtype=probability_dtype, device=coarse.device
        )
        scale_variance_tensor = torch.empty((0,), dtype=probability_dtype, device=coarse.device)
        source_scale_tensor = torch.empty((0,), dtype=torch.long, device=coarse.device)
        peak_tensor = torch.empty((0, 2), dtype=torch.long, device=coarse.device)

    boxes = masks_to_roi_boxes({"masks": mask_tensor, "batch_indices": batch_indices})
    return {
        "masks": mask_tensor,
        "candidate_masks": mask_tensor,
        "boxes": boxes,
        "batch_indices": batch_indices,
        "scores": score_tensor,
        "proposal_scores": score_tensor,
        "peak_scores": score_tensor,
        "coarse_scores": coarse_score_tensor,
        "coarse_peak_scores": coarse_peak_tensor,
        "areas": area_tensor,
        "scale_responses": scale_response_tensor,
        "scale_variance": scale_variance_tensor,
        "source_scale": source_scale_tensor,
        "peak_yx": peak_tensor,
        "stream_ids": torch.ones_like(batch_indices),
        "num_peaks_before_limit": peaks_before_limit,
        "max_probability": evidence_map.unsqueeze(1),
        "mean_probability": scale_stack.mean(dim=0).unsqueeze(1),
        "scale_variance_map": scale_variance_map.unsqueeze(1),
    }


def _gt_instances_by_batch(
    gt_masks: Any,
    image_hw: tuple[int, int],
    device: torch.device,
    minimum_batches: int,
) -> list[list[torch.Tensor]]:
    """Convert semantic or instance GT layouts to per-image instances."""
    height, width = image_hw
    explicit_indices = None
    raw_gt = gt_masks
    if isinstance(gt_masks, Mapping):
        raw_gt = gt_masks.get("masks", gt_masks.get("gt_masks"))
        if raw_gt is None:
            raise KeyError("GT mapping must contain 'masks' or 'gt_masks'")
        explicit_indices = gt_masks.get("batch_indices")

    if explicit_indices is not None:
        tensor = raw_gt if isinstance(raw_gt, torch.Tensor) else torch.as_tensor(raw_gt)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != image_hw:
            raise ValueError("instance GT masks must have shape [M,H,W]")
        indices = torch.as_tensor(explicit_indices, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() != tensor.shape[0] or torch.any(indices < 0):
            raise ValueError("GT batch_indices must be non-negative with shape [M]")
        count = max(minimum_batches, int(indices.max().item()) + 1 if indices.numel() else 0)
        groups: list[list[torch.Tensor]] = [[] for _ in range(count)]
        for mask, index in zip(tensor, indices.tolist()):
            binary = (mask != 0).to(device=device)
            if binary.any():
                groups[index].append(binary)
        return groups

    if isinstance(raw_gt, (list, tuple)):
        groups = [[] for _ in range(max(minimum_batches, len(raw_gt)))]
        for batch_index, item in enumerate(raw_gt):
            tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            if tensor.ndim == 2:
                if tuple(tensor.shape) != image_hw:
                    raise ValueError("GT masks and candidate masks must have the same spatial shape")
                for component in _component_arrays(tensor != 0):
                    groups[batch_index].append(torch.as_tensor(component, device=device, dtype=torch.bool))
            elif tensor.ndim == 3 and tuple(tensor.shape[-2:]) == image_hw:
                groups[batch_index].extend(
                    (mask != 0).to(device=device) for mask in tensor if torch.any(mask != 0)
                )
            else:
                raise ValueError("each GT list item must have shape [H,W] or [M,H,W]")
        return groups

    tensor = raw_gt if isinstance(raw_gt, torch.Tensor) else torch.as_tensor(raw_gt)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    elif tensor.ndim != 4:
        raise ValueError("gt_masks must have shape [H,W], [B,H,W], or [B,M,H,W]")
    if tuple(tensor.shape[-2:]) != image_hw:
        raise ValueError("GT masks and candidate masks must have the same spatial shape")

    groups = [[] for _ in range(max(minimum_batches, tensor.shape[0]))]
    for batch_index in range(tensor.shape[0]):
        if tensor.shape[1] == 1:
            for component in _component_arrays(tensor[batch_index, 0] != 0):
                groups[batch_index].append(torch.as_tensor(component, device=device, dtype=torch.bool))
        else:
            groups[batch_index].extend(
                (mask != 0).to(device=device)
                for mask in tensor[batch_index]
                if torch.any(mask != 0)
            )
    return groups


def match_candidates_to_gt(
    candidate_masks: Any,
    gt_masks: Any,
    positive_iou: float,
    hard_negative_threshold: float,
    center_distance: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Assign target/clutter/uncertain labels to flattened candidates.

    A candidate is target (0) when its best instance IoU reaches
    ``positive_iou`` or its centroid falls inside a GT instance *and* is within
    ``center_distance`` pixels of that same GT centroid.  A zero-overlap
    candidate with score at least
    ``hard_negative_threshold`` is clutter (1); all remaining candidates are
    uncertain (2).  Candidate mappings should provide ``scores``.  For bare
    binary masks, each non-empty mask has an implicit score of one.
    """
    if not 0.0 <= float(positive_iou) <= 1.0:
        raise ValueError("positive_iou must be in [0, 1]")
    if not 0.0 <= float(hard_negative_threshold) <= 1.0:
        raise ValueError("hard_negative_threshold must be in [0, 1]")
    if float(center_distance) < 0:
        raise ValueError("center_distance must be non-negative")

    masks, batch_indices, _, supplied_scores = _flatten_candidate_masks(candidate_masks)
    number_of_candidates, height, width = masks.shape
    device = masks.device
    if supplied_scores is None:
        scores = masks.flatten(1).any(dim=1).to(dtype=torch.float32)
    else:
        scores = supplied_scores.to(device=device, dtype=torch.float32)

    required_batches = int(batch_indices.max().item()) + 1 if batch_indices.numel() else 0
    gt_instances = _gt_instances_by_batch(
        gt_masks, (height, width), device=device, minimum_batches=required_batches
    )

    max_ious = torch.zeros((number_of_candidates,), dtype=torch.float32, device=device)
    matched_indices = torch.full((number_of_candidates,), -1, dtype=torch.long, device=device)
    center_matches = torch.zeros((number_of_candidates,), dtype=torch.bool, device=device)
    centroid_distances = torch.full(
        (number_of_candidates,), float("inf"), dtype=torch.float32, device=device
    )
    for candidate_index in range(number_of_candidates):
        batch_index = int(batch_indices[candidate_index].item())
        instances = gt_instances[batch_index] if batch_index < len(gt_instances) else []
        candidate = masks[candidate_index]
        if not instances or not torch.any(candidate):
            continue
        instance_stack = torch.stack(instances)
        intersection = (instance_stack & candidate).flatten(1).sum(dim=1).float()
        union = (instance_stack | candidate).flatten(1).sum(dim=1).float()
        ious = intersection / union.clamp_min(1.0)
        best_iou, best_index = ious.max(dim=0)
        max_ious[candidate_index] = best_iou
        candidate_points = torch.nonzero(candidate, as_tuple=False).float()
        instance_centroids = torch.stack(
            [torch.nonzero(instance, as_tuple=False).float().mean(dim=0) for instance in instances]
        )
        candidate_centroid = candidate_points.mean(dim=0)
        distances = torch.linalg.vector_norm(
            instance_centroids - candidate_centroid.unsqueeze(0), dim=1
        )
        nearest_distance = distances.min()
        centroid_distances[candidate_index] = nearest_distance
        center_y = int(candidate_centroid[0].round().clamp(0, height - 1).item())
        center_x = int(candidate_centroid[1].round().clamp(0, width - 1).item())
        center_inside = instance_stack[:, center_y, center_x]
        eligible_center_matches = center_inside & (
            distances <= float(center_distance)
        )
        center_match = bool(eligible_center_matches.any())
        center_matches[candidate_index] = center_match
        if best_iou > 0:
            matched_indices[candidate_index] = best_index
        elif center_match:
            eligible_distances = distances.masked_fill(
                ~eligible_center_matches, float("inf")
            )
            matched_indices[candidate_index] = eligible_distances.argmin()

    labels = torch.full(
        (number_of_candidates,), UNCERTAIN_LABEL, dtype=torch.long, device=device
    )
    positive_match = (max_ious >= float(positive_iou)) | center_matches
    labels[positive_match] = TARGET_LABEL
    hard_negative = (
        (max_ious == 0)
        & ~positive_match
        & (scores >= float(hard_negative_threshold))
    )
    labels[hard_negative] = CLUTTER_LABEL
    return {
        "labels": labels,
        "max_iou": max_ious,
        "matched_gt_indices": matched_indices,
        "center_match": center_matches,
        "centroid_distance": centroid_distances,
        "batch_indices": batch_indices,
        "scores": scores,
    }


def masks_to_roi_boxes(candidate_masks: Any) -> torch.Tensor:
    """Convert masks to [N,5] ROIAlign boxes on the masks' device.

    Empty masks in a non-empty collection produce a zero-area box while
    preserving candidate order.  An empty collection returns float32 [0,5].
    """
    masks, batch_indices, _, _ = _flatten_candidate_masks(candidate_masks)
    number_of_candidates = masks.shape[0]
    boxes = torch.zeros((number_of_candidates, 5), dtype=torch.float32, device=masks.device)
    if number_of_candidates == 0:
        return boxes
    boxes[:, 0] = batch_indices.float()
    for candidate_index, mask in enumerate(masks):
        coordinates = torch.nonzero(mask, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        y1, x1 = coordinates.amin(dim=0)
        y2, x2 = coordinates.amax(dim=0) + 1
        boxes[candidate_index, 1:] = torch.stack((x1, y1, x2, y2)).float()
    return boxes


def expand_boxes(
    boxes: torch.Tensor,
    scale: float,
    image_hw: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    """Scale [N,4] or [N,5] xyxy boxes about their centers and clip them."""
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    if boxes.ndim != 2 or boxes.shape[1] not in (4, 5):
        raise ValueError("boxes must have shape [N,4] or [N,5]")
    if float(scale) <= 0:
        raise ValueError("scale must be positive")
    if len(image_hw) != 2:
        raise ValueError("image_hw must be (height, width)")
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    output_dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
    expanded = boxes.to(dtype=output_dtype).clone()
    coordinates = expanded[:, -4:]
    center_x = (coordinates[:, 0] + coordinates[:, 2]) * 0.5
    center_y = (coordinates[:, 1] + coordinates[:, 3]) * 0.5
    half_width = (coordinates[:, 2] - coordinates[:, 0]) * (float(scale) * 0.5)
    half_height = (coordinates[:, 3] - coordinates[:, 1]) * (float(scale) * 0.5)
    coordinates[:, 0] = (center_x - half_width).clamp(0, width)
    coordinates[:, 1] = (center_y - half_height).clamp(0, height)
    coordinates[:, 2] = (center_x + half_width).clamp(0, width)
    coordinates[:, 3] = (center_y + half_height).clamp(0, height)
    return expanded


def extract_scale_features(
    multi_scale_logits: Sequence[torch.Tensor] | torch.Tensor,
    candidate_masks: Any,
) -> torch.Tensor:
    """Return [N,L+1] candidate scale features.

    The first L columns are mean sigmoid responses inside each candidate.  The
    last column is the candidate mean of the per-pixel population variance
    across scales.  Resizing is bilinear with ``align_corners=False``.  The
    result is differentiable with respect to logits and resides on the first
    scale's device.  Empty candidates return [0,L+1].
    """
    scales = _logits_sequence(multi_scale_logits)
    if not scales:
        raise ValueError("multi_scale_logits must contain at least one scale")
    normalized = [_as_binary_logits(scale, f"multi_scale_logits[{index}]") for index, scale in enumerate(scales)]
    reference = normalized[0]
    batch_size = reference.shape[0]
    for scale in normalized[1:]:
        if scale.shape[0] != batch_size:
            raise ValueError("all multi-scale logits must have the same batch size")

    masks, batch_indices, _, _ = _flatten_candidate_masks(candidate_masks)
    number_of_candidates, height, width = masks.shape
    if number_of_candidates == 0:
        return reference.new_empty((0, len(scales) + 1))
    if height == 0 or width == 0:
        raise ValueError("non-empty candidates must have a non-empty spatial shape")
    if int(batch_indices.max().item()) >= batch_size:
        raise ValueError("candidate batch index exceeds the logits batch size")

    resized_probabilities = []
    for scale in normalized:
        scale = scale.to(device=reference.device, dtype=reference.dtype)
        if scale.shape[-2:] != (height, width):
            scale = F.interpolate(scale, size=(height, width), mode="bilinear", align_corners=False)
        probability = torch.sigmoid(scale)
        resized_probabilities.append(probability[:, 0])
    scale_stack = torch.stack(resized_probabilities, dim=0)  # [L,B,H,W]

    batch_indices = batch_indices.to(device=reference.device)
    masks_float = masks.to(device=reference.device, dtype=reference.dtype)
    selected = scale_stack[:, batch_indices]  # [L,N,H,W]
    denominator = masks_float.flatten(1).sum(dim=1).clamp_min(1)
    responses = (selected * masks_float.unsqueeze(0)).flatten(2).sum(dim=2)
    responses = (responses / denominator.unsqueeze(0)).transpose(0, 1)
    pixel_variance = selected.var(dim=0, unbiased=False)
    variance = (pixel_variance * masks_float).flatten(1).sum(dim=1) / denominator
    return torch.cat((responses, variance.unsqueeze(1)), dim=1)


__all__ = [
    "TARGET_LABEL",
    "CLUTTER_LABEL",
    "UNCERTAIN_LABEL",
    "LABEL_NAMES",
    "generate_component_aligned_candidates",
    "generate_candidates",
    "generate_recovery_candidates",
    "map_action_components_to_proposals",
    "match_candidates_to_gt",
    "masks_to_roi_boxes",
    "expand_boxes",
    "extract_scale_features",
]
