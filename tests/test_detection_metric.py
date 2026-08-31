import numpy as np
import pytest
import torch

from utils.detection_metric import (
    SegmentationFROC,
    component_detection_summary,
    extract_components,
    match_prediction_components_to_gt,
    maximum_centroid_pairs,
)


def test_segmentation_froc_does_not_drop_equal_area_false_positive():
    logits = torch.full((1, 1, 8, 8), -10.0)
    logits[0, 0, 1, 1] = 10.0
    logits[0, 0, 6, 6] = 10.0
    target = torch.zeros_like(logits)
    target[0, 0, 1, 1] = 1.0

    metric = SegmentationFROC([0.5])
    metric.update(logits, target)
    curve = metric.get()

    np.testing.assert_allclose(curve["Pd"], [1.0])
    np.testing.assert_allclose(curve["FPPI"], [1.0])
    np.testing.assert_array_equal(curve["false_alarm_pixels"], [1])
    assert curve["Fa_per_million_pixels"][0] == pytest.approx(1e6 / 64)


def test_segmentation_froc_probability_thresholds_and_empty_gt():
    probabilities = torch.full((1, 1, 8, 8), 0.01)
    probabilities[0, 0, 1, 1] = 0.8
    probabilities[0, 0, 6, 6] = 0.6
    target = torch.zeros_like(probabilities)
    target[0, 0, 1, 1] = 1.0

    metric = SegmentationFROC([0.9, 0.5, 0.1])
    metric.update(probabilities, target, from_logits=False)
    curve = metric.get()
    np.testing.assert_allclose(curve["Pd"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(curve["FPPI"], [0.0, 1.0, 1.0])

    target_free = SegmentationFROC([0.5])
    target_free.update(probabilities, torch.zeros_like(target), from_logits=False)
    assert np.isnan(target_free.get()["Pd"]).all()


def test_public_centroid_pairs_reject_invalid_distance():
    with pytest.raises(ValueError, match="non-negative"):
        maximum_centroid_pairs([], [], -1.0)


def test_public_components_are_stable_eight_connected_records():
    binary = np.zeros((5, 6), dtype=bool)
    binary[1, 1] = True
    binary[2, 2] = True
    binary[4, 5] = True

    components = extract_components(binary)

    assert len(components) == 2
    assert components.areas.tolist() == [2, 1]
    assert components.bboxes_yxyx.tolist() == [[1, 1, 3, 3], [4, 5, 5, 6]]
    np.testing.assert_allclose(components.centroids_yx, [[1.5, 1.5], [4.0, 5.0]])
    assert np.array_equal(components.label_map == 1, components.masks[0])
    with pytest.raises(ValueError, match="8-connectivity"):
        extract_components(binary, connectivity=4)


def test_component_match_exposes_strict_labels_and_ambiguity_without_hiding_fp():
    target = np.zeros((9, 9), dtype=bool)
    target[4, 2] = True
    target[4, 6] = True
    prediction = np.zeros_like(target)
    prediction[4, 3:6] = True  # Within distance of both GT components.
    prediction[0, 0] = True

    match = match_prediction_components_to_gt(
        prediction, target, center_distance=3.0
    )
    summary = component_detection_summary(match)

    assert match.is_tp_component.tolist() == [False, True]
    assert match.is_fp_component.tolist() == [True, False]
    assert match.ambiguous_keep.tolist() == [False, True]
    assert "merged_multiple_gt" in match.ambiguity_reasons[1]
    assert summary["true_positive_components"] == 1
    assert summary["false_positive_components"] == 1
    assert summary["false_negative_targets"] == 1
    assert summary["false_alarm_pixels"] == 1


def test_component_match_distance_boundary_is_inclusive():
    target = np.zeros((9, 9), dtype=bool)
    target[4, 4] = True
    at_boundary = np.zeros_like(target)
    at_boundary[4, 7] = True
    outside = np.zeros_like(target)
    outside[0, 0] = True

    assert match_prediction_components_to_gt(
        at_boundary, target, center_distance=3.0
    ).is_tp_component.tolist() == [True]
    assert match_prediction_components_to_gt(
        outside, target, center_distance=3.0
    ).is_fp_component.tolist() == [True]
