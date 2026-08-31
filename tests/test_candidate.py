import math

import pytest
import torch
import torch.nn.functional as F

from utils.candidate import (
    CLUTTER_LABEL,
    TARGET_LABEL,
    UNCERTAIN_LABEL,
    expand_boxes,
    extract_scale_features,
    generate_candidates,
    generate_recovery_candidates,
    masks_to_roi_boxes,
    match_candidates_to_gt,
)


def _binary_logits(masks: torch.Tensor, magnitude: float = 20.0) -> torch.Tensor:
    return torch.where(masks, torch.tensor(magnitude), torch.tensor(-magnitude)).float()


def test_generate_candidates_supports_batch_and_multiple_scales():
    support = torch.zeros((2, 1, 8, 8), dtype=torch.bool)
    support[0, 0, 1:3, 1:3] = True
    support[1, 0, 4:6, 5:7] = True
    strong = _binary_logits(support)
    weak_resized_scale = torch.full((2, 1, 1, 1), -20.0)
    coarse = _binary_logits(support)

    result = generate_candidates(
        coarse,
        [strong, strong.clone(), weak_resized_scale],
        threshold_low=0.6,
        min_area=2,
        max_area=8,
    )

    assert result["masks"].shape == (2, 8, 8)
    assert result["masks"].dtype == torch.bool
    assert result["batch_indices"].tolist() == [0, 1]
    assert result["areas"].tolist() == [4, 4]
    assert result["boxes"].tolist() == [
        [0.0, 1.0, 1.0, 3.0, 3.0],
        [1.0, 5.0, 4.0, 7.0, 6.0],
    ]
    assert result["scale_responses"].shape == (2, 3)
    assert torch.allclose(result["scores"], torch.full((2,), 2.0 / 3.0), atol=1e-5)
    assert result["scale_variance"].shape == (2,)
    assert result["mean_probability"].shape == (2, 1, 8, 8)
    assert all(value.device == coarse.device for value in result.values())


def test_generate_candidates_has_shape_stable_empty_output():
    logits = torch.full((3, 1, 7, 9), -20.0)
    result = generate_candidates(logits, [logits], 0.2, 1, None)

    assert result["masks"].shape == (0, 7, 9)
    assert result["boxes"].shape == (0, 5)
    assert result["batch_indices"].shape == (0,)
    assert result["scale_responses"].shape == (0, 1)
    assert result["scale_variance"].shape == (0,)


def test_generate_recovery_candidates_uses_local_max_and_strict_score_band():
    def logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    coarse = torch.full((1, 1, 9, 9), logit(0.01))
    scale = torch.full_like(coarse, logit(0.01))
    scale[0, 0, 4, 4] = logit(0.4)
    scale[0, 0, 1, 1] = logit(0.05)  # Strictly equal to low: excluded.
    coarse[0, 0, 7, 7] = logit(0.6)
    scale[0, 0, 7, 7] = logit(0.9)  # Coarse already positive: excluded.

    result = generate_recovery_candidates(
        coarse,
        [scale],
        threshold_low=0.05,
        threshold_high=0.5,
        local_max_kernel=3,
        proposal_size=5,
        max_candidates_per_image=8,
    )

    assert result["peak_yx"].tolist() == [[4, 4]]
    assert result["areas"].tolist() == [25]
    assert result["stream_ids"].tolist() == [1]
    assert result["source_scale"].tolist() == [0]
    assert result["num_peaks_before_limit"].tolist() == [1]
    assert result["proposal_scores"].item() == pytest.approx(0.4)


def test_generate_recovery_candidates_includes_coarse_weak_peak_and_caps_count():
    def logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    coarse = torch.full((1, 1, 9, 9), logit(0.01))
    coarse[0, 0, 2, 2] = logit(0.4)
    coarse[0, 0, 6, 6] = logit(0.3)
    scale = torch.full_like(coarse, logit(0.01))

    result = generate_recovery_candidates(
        coarse,
        [scale],
        threshold_low=0.05,
        threshold_high=0.5,
        local_max_kernel=3,
        proposal_size=3,
        max_candidates_per_image=1,
    )

    assert result["num_peaks_before_limit"].tolist() == [2]
    assert result["peak_yx"].tolist() == [[2, 2]]
    assert result["source_scale"].tolist() == [-1]
    assert result["boxes"].tolist() == [[0.0, 1.0, 1.0, 4.0, 4.0]]


def test_generate_recovery_candidates_has_stable_empty_shapes():
    logits = torch.full((2, 1, 8, 8), -20.0)

    result = generate_recovery_candidates(logits, [logits], threshold_low=0.05)

    assert result["masks"].shape == (0, 8, 8)
    assert result["boxes"].shape == (0, 5)
    assert result["peak_yx"].shape == (0, 2)
    assert result["scale_responses"].shape == (0, 1)
    assert result["num_peaks_before_limit"].tolist() == [0, 0]


def test_match_candidates_to_gt_uses_three_state_label_order():
    masks = torch.zeros((3, 6, 6), dtype=torch.bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 4:6, 4:6] = True
    masks[2, 1:4, 1:4] = True
    candidates = {
        "masks": masks,
        "batch_indices": torch.tensor([0, 0, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.95]),
    }
    gt = torch.zeros((2, 1, 6, 6), dtype=torch.bool)
    gt[0, 0, 1:3, 1:3] = True
    gt[1, 0, 1, 1] = True

    result = match_candidates_to_gt(
        candidates,
        gt,
        positive_iou=0.5,
        hard_negative_threshold=0.7,
        center_distance=0.0,
    )

    assert (TARGET_LABEL, CLUTTER_LABEL, UNCERTAIN_LABEL) == (0, 1, 2)
    assert result["labels"].tolist() == [0, 1, 2]
    assert torch.allclose(result["max_iou"], torch.tensor([1.0, 0.0, 1.0 / 9.0]))
    assert result["matched_gt_indices"].tolist() == [0, -1, 0]
    assert result["batch_indices"].tolist() == [0, 0, 1]


def test_match_candidates_accepts_centroid_distance_fallback():
    candidate = torch.zeros((1, 8, 8), dtype=torch.bool)
    candidate[0, 2:6, 2:6] = True
    gt = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    gt[0, 0, 4, 4] = True

    result = match_candidates_to_gt(
        {"masks": candidate, "scores": torch.tensor([0.9])},
        gt,
        positive_iou=0.5,
        hard_negative_threshold=0.7,
        center_distance=3.0,
    )

    assert result["labels"].tolist() == [TARGET_LABEL]
    assert result["center_match"].tolist() == [True]
    assert result["batch_indices"].tolist() == [0]


def test_match_candidates_rejects_nearby_centroid_outside_gt():
    candidate = torch.zeros((1, 8, 8), dtype=torch.bool)
    candidate[0, 2:6, 2:6] = True
    gt = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    gt[0, 0, 1, 4] = True

    result = match_candidates_to_gt(
        {"masks": candidate, "scores": torch.tensor([0.9])},
        gt,
        positive_iou=0.5,
        hard_negative_threshold=0.7,
        center_distance=3.0,
    )

    assert result["centroid_distance"].item() < 3.0
    assert result["center_match"].tolist() == [False]
    assert result["labels"].tolist() == [CLUTTER_LABEL]


def test_match_candidates_to_gt_handles_no_candidates_and_no_gt():
    candidates = {
        "masks": torch.empty((0, 5, 5), dtype=torch.bool),
        "batch_indices": torch.empty((0,), dtype=torch.long),
        "scores": torch.empty((0,)),
    }
    gt = torch.zeros((2, 1, 5, 5), dtype=torch.bool)

    result = match_candidates_to_gt(candidates, gt, 0.5, 0.7)

    assert result["labels"].shape == (0,)
    assert result["max_iou"].shape == (0,)
    assert result["matched_gt_indices"].shape == (0,)


def test_masks_to_roi_boxes_accepts_batched_layout_and_empty_set():
    masks = torch.zeros((2, 2, 6, 7), dtype=torch.bool)
    masks[0, 0, 1:3, 2:5] = True
    masks[1, 1, 4:6, 6] = True

    boxes = masks_to_roi_boxes(masks)

    assert boxes.shape == (4, 5)
    assert boxes.tolist() == [
        [0.0, 2.0, 1.0, 5.0, 3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 6.0, 4.0, 7.0, 6.0],
    ]
    assert masks_to_roi_boxes(torch.empty((0, 6, 7), dtype=torch.bool)).shape == (0, 5)


def test_expand_boxes_preserves_batch_column_and_clips_to_image():
    boxes = torch.tensor(
        [[3.0, 2.0, 2.0, 4.0, 4.0], [1.0, 0.0, 0.0, 2.0, 2.0]]
    )

    expanded = expand_boxes(boxes, scale=3.0, image_hw=(6, 7))

    assert torch.allclose(
        expanded,
        torch.tensor([[3.0, 0.0, 0.0, 6.0, 6.0], [1.0, 0.0, 0.0, 4.0, 4.0]]),
    )
    assert expand_boxes(torch.empty((0, 5)), 2.0, (6, 7)).shape == (0, 5)
    with pytest.raises(ValueError, match="positive"):
        expand_boxes(boxes, 0.0, (6, 7))


def test_extract_scale_features_is_batched_resized_and_differentiable():
    def logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    scale_0 = torch.tensor([logit(0.2), logit(0.6)]).reshape(2, 1, 1, 1)
    scale_0 = scale_0.expand(2, 1, 4, 4).clone().requires_grad_()
    scale_1 = torch.tensor([logit(0.4), logit(0.8)]).reshape(2, 1, 1, 1)
    scale_1.requires_grad_()
    masks = torch.ones((2, 4, 4), dtype=torch.bool)
    candidates = {"masks": masks, "batch_indices": torch.tensor([0, 1])}

    features = extract_scale_features([scale_0, scale_1], candidates)

    expected = torch.tensor([[0.2, 0.4, 0.01], [0.6, 0.8, 0.01]])
    assert features.shape == (2, 3)
    assert torch.allclose(features, expected, atol=1e-6)
    features.sum().backward()
    assert scale_0.grad is not None
    assert scale_1.grad is not None

    empty = {"masks": torch.empty((0, 4, 4), dtype=torch.bool)}
    assert extract_scale_features([scale_0, scale_1], empty).shape == (0, 3)


def test_extract_scale_features_upsamples_logits_before_sigmoid():
    scale = torch.tensor([[[[-4.0, 4.0], [4.0, -4.0]]]])
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, 1, 1] = True

    features = extract_scale_features([scale], mask)

    resized_logits = F.interpolate(scale, size=(4, 4), mode="bilinear", align_corners=False)
    expected_response = resized_logits.sigmoid()[0, 0, 1, 1]
    wrong_order = F.interpolate(
        scale.sigmoid(), size=(4, 4), mode="bilinear", align_corners=False
    )[0, 0, 1, 1]
    torch.testing.assert_close(features[0, 0], expected_response)
    torch.testing.assert_close(features[0, 1], torch.tensor(0.0))
    assert not torch.isclose(expected_response, wrong_order)
