import math

import numpy as np
import pytest
import torch

from utils.reliability_metric import (
    candidate_brier_score,
    candidate_ece,
    candidate_nll,
    false_alarm_at_fixed_pd,
    false_positives_per_image,
    fppi_at_fixed_pd,
    fppi_froc,
    risk_coverage_curve,
)


def test_candidate_calibration_metrics_match_hand_calculation():
    probabilities = np.array(
        [[0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1, 0])

    # All top confidences are in the upper of two bins.  Accuracy is 1/2 and
    # mean confidence is (0.8 + 0.6 + 0.8 + 0.7) / 4 = 0.725.
    assert candidate_ece(probabilities, labels, n_bins=2) == pytest.approx(0.225)

    expected_brier = np.mean([0.08, 0.72, 0.08, 0.98])
    assert candidate_brier_score(probabilities, labels) == pytest.approx(
        expected_brier
    )

    expected_nll = -np.log([0.8, 0.4, 0.8, 0.3]).mean()
    assert candidate_nll(probabilities, labels) == pytest.approx(expected_nll)


def test_target_class_ece_and_bin_details():
    probabilities = torch.tensor(
        [[0.9, 0.1, 0.0], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 0])

    details = candidate_ece(
        probabilities,
        labels,
        n_bins=2,
        target_class=0,
        return_bins=True,
    )

    # Lower bin: confidence=.2, target frequency=1. Upper bin: confidence=.75,
    # target frequency=.5. Weighted gap = (1/3)*.8 + (2/3)*.25.
    assert details["ece"] == pytest.approx(13.0 / 30.0)
    np.testing.assert_array_equal(details["bin_count"], [1, 2])
    assert details["num_candidates"] == 3


def test_calibration_metrics_ignore_invalid_labels():
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
    labels = np.array([0, -1, 1])

    expected_probabilities = probabilities[[0, 2]]
    expected_labels = labels[[0, 2]]
    assert candidate_brier_score(probabilities, labels) == pytest.approx(
        candidate_brier_score(expected_probabilities, expected_labels)
    )
    assert candidate_nll(probabilities, labels) == pytest.approx(
        candidate_nll(expected_probabilities, expected_labels)
    )


def test_empty_candidate_calibration_metrics_are_explicitly_undefined():
    probabilities = np.empty((0, 3), dtype=np.float64)
    labels = np.empty((0,), dtype=np.int64)

    assert math.isnan(candidate_ece(probabilities, labels))
    assert math.isnan(candidate_brier_score(probabilities, labels))
    assert math.isnan(candidate_nll(probabilities, labels))
    assert candidate_brier_score(probabilities, labels, reduction="sum") == 0.0
    assert candidate_nll(probabilities, labels, reduction="none").shape == (0,)

    details = candidate_ece(probabilities, labels, return_bins=True)
    assert details["num_candidates"] == 0
    assert details["bin_count"].sum() == 0


def test_fppi_and_froc_operating_points():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    is_true_positive = np.array([True, False, True, False])
    thresholds = np.array([0.95, 0.85, 0.75, 0.65, 0.55])

    curve = fppi_froc(
        scores,
        is_true_positive,
        num_images=2,
        num_targets=2,
        thresholds=thresholds,
    )

    np.testing.assert_allclose(curve["pd"], [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(curve["fppi"], [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(curve["true_positives"], [0, 1, 1, 2, 2])
    np.testing.assert_array_equal(curve["false_positives"], [0, 0, 1, 1, 2])
    assert false_positives_per_image([False, True, True], 4) == 0.5
    assert fppi_at_fixed_pd(curve, 0.9) == 0.5


def test_fixed_pd_false_alarm_discrete_interpolated_and_unreachable():
    pd = np.array([0.0, 0.5, 0.5, 1.0])
    fa = np.array([0.0, 0.0, 0.5, 0.5])

    assert false_alarm_at_fixed_pd(pd, fa, 0.75) == 0.5
    assert false_alarm_at_fixed_pd(pd, fa, 0.75, interpolate=True) == pytest.approx(
        0.25
    )
    assert math.isnan(false_alarm_at_fixed_pd(pd[:3], fa[:3], 0.95))


def test_froc_supports_no_candidates():
    curve = fppi_froc(
        np.empty((0,)),
        np.empty((0,), dtype=bool),
        num_images=7,
        num_targets=3,
    )

    np.testing.assert_array_equal(curve["thresholds"], [np.inf])
    np.testing.assert_array_equal(curve["pd"], [0.0])
    np.testing.assert_array_equal(curve["fppi"], [0.0])
    assert math.isnan(fppi_at_fixed_pd(curve, 0.9))


def test_target_free_froc_marks_pd_undefined_but_keeps_false_alarm_rate():
    curve = fppi_froc(
        scores=np.array([0.4]),
        is_true_positive=np.array([False]),
        num_images=2,
        num_targets=0,
        thresholds=np.array([0.5, 0.3]),
    )

    assert np.isnan(curve["pd"]).all()
    np.testing.assert_allclose(curve["fppi"], [0.0, 0.5])


def test_risk_coverage_curve_matches_selective_classification_definition():
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.45, 0.55]]
    )
    labels = np.asarray([0, 1, 0, 1])

    curve = risk_coverage_curve(probabilities, labels)

    np.testing.assert_allclose(curve["coverage"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(curve["risk"], [0.0, 0.0, 0.5, 1.0 / 3.0, 0.25])
    assert curve["aurc"] == pytest.approx(
        np.sum(
            np.diff(curve["coverage"])
            * (curve["risk"][:-1] + curve["risk"][1:])
            * 0.5
        )
    )


def test_risk_coverage_groups_ties_and_handles_empty_candidates():
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    labels = np.asarray([0, 0, -1])
    curve = risk_coverage_curve(probabilities, labels)
    np.testing.assert_allclose(curve["coverage"], [0.0, 1.0])
    np.testing.assert_allclose(curve["risk"], [0.0, 0.5])

    empty = risk_coverage_curve(np.empty((0, 2)), np.empty((0,), dtype=np.int64))
    assert empty["num_candidates"] == 0
    assert math.isnan(empty["aurc"])
