import numpy as np
import pytest
import torch

from utils.detection_metric import SegmentationFROC


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
