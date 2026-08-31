import math

import numpy as np
import pytest
import torch

from scripts.audit_fp_upper_bound import (
    oracle_target_presence,
    threshold_aware_suppression,
)
from scripts.audit_missed_targets import resize_probabilities
from utils.audit import (
    BinarySegmentationAccumulator,
    component_masks,
    maximum_centroid_assignment,
)
from utils.detection_metric import maximum_centroid_pairs


def test_maximum_centroid_pairs_uses_augmenting_path_not_greedy_match():
    gt = [np.array([0.0, 0.0]), np.array([0.0, 1.0])]
    predictions = [np.array([0.0, 0.5]), np.array([0.0, -1.0])]

    pairs = maximum_centroid_pairs(gt, predictions, maximum_distance=1.1)

    assert pairs == [(0, 1), (1, 0)]


def test_component_masks_use_eight_connectivity_and_assignment_is_one_to_one():
    diagonal = np.eye(2, dtype=bool)
    assert len(component_masks(diagonal)) == 1

    gt = []
    for column in (1, 3):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, column] = True
        gt.append(mask)
    proposal = np.zeros((5, 5), dtype=bool)
    proposal[2, 2] = True

    assignment = maximum_centroid_assignment(gt, [proposal], maximum_distance=2.0)

    assert len(assignment) == 1
    assert list(assignment.values()) == [0]


def test_binary_audit_metrics_keep_equal_area_false_positive():
    prediction = np.zeros((8, 8), dtype=bool)
    prediction[1, 1] = True
    prediction[6, 6] = True
    target = np.zeros_like(prediction)
    target[1, 1] = True
    metric = BinarySegmentationAccumulator(center_distance=3.0)

    snapshot = metric.update(prediction, target)
    result = metric.get()

    assert len(snapshot["false_positive_indices"]) == 1
    assert result["Pd"] == 1.0
    assert result["FPPI"] == 1.0
    assert result["false_alarm_pixels"] == 1
    assert result["Fa_per_million_pixels"] == pytest.approx(1_000_000 / 64)


def test_oracle_target_presence_protects_any_gt_overlap():
    candidate_masks = torch.zeros((2, 6, 6), dtype=torch.bool)
    candidate_masks[0, 1:3, 1:3] = True
    candidate_masks[1, 4, 4] = True
    candidates = {
        "masks": candidate_masks,
        "batch_indices": torch.tensor([0, 0]),
        "coarse_peak_scores": torch.tensor([0.9, 0.9]),
    }
    target = torch.zeros((1, 1, 6, 6))
    target[0, 0, 2, 2] = 1.0

    protected = oracle_target_presence(
        candidates, target, positive_iou=0.9, center_distance=0.0
    )

    assert protected.tolist() == [True, False]


def test_threshold_aware_suppression_removes_full_fp_without_touching_target():
    logits = torch.full((1, 1, 5, 5), -10.0)
    logits[0, 0, 1, 1] = 10.0
    logits[0, 0, 4, 4] = 10.0
    masks = torch.zeros((1, 5, 5), dtype=torch.bool)
    masks[0, 4, 4] = True

    refined, oracle_mask = threshold_aware_suppression(
        logits,
        masks,
        torch.tensor([0]),
        torch.tensor([True]),
        remove_threshold=0.45,
    )

    assert oracle_mask[0, 4, 4]
    assert refined[0, 0, 1, 1] == logits[0, 0, 1, 1]
    assert refined.sigmoid()[0, 0, 4, 4] == pytest.approx(0.45)
    assert not bool((refined.sigmoid() > 0.5)[0, 0, 4, 4])


def test_multiscale_audit_resizes_logits_before_sigmoid():
    first = torch.tensor([[[[-4.0, 4.0], [4.0, -4.0]]]])
    second = torch.full((1, 1, 4, 4), math.log(0.25 / 0.75))

    probabilities = resize_probabilities([first, second], (4, 4))

    expected_first = torch.nn.functional.interpolate(
        first, size=(4, 4), mode="bilinear", align_corners=False
    ).sigmoid()[0, 0]
    torch.testing.assert_close(probabilities[0], expected_first)
    torch.testing.assert_close(probabilities[1], torch.full((4, 4), 0.25))
