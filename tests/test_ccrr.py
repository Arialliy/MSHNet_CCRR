import math

import pytest
import torch

from model.ccrr import (
    CCRRModule,
    CandidateContextEncoder,
    InstanceLogitRectifier,
    ReliabilityHead,
)


def _two_image_candidates(height=16, width=16):
    boxes = torch.tensor(
        [
            [0, 2, 3, 7, 8],
            [1, 9, 8, 13, 14],
        ],
        dtype=torch.float32,
    )
    masks = torch.zeros((2, height, width), dtype=torch.bool)
    masks[0, 3:8, 2:7] = True
    masks[1, 8:14, 9:13] = True
    return boxes, masks


def test_candidate_context_encoder_uses_roi_features_and_backpropagates():
    torch.manual_seed(1)
    encoder = CandidateContextEncoder(
        feature_channels=4,
        num_scales=2,
        roi_size=5,
        hidden_dim=8,
        context_scale=3.0,
    )
    feature_map = torch.randn(2, 4, 8, 8, requires_grad=True)
    boxes, _ = _two_image_candidates()
    scale_features = torch.randn(2, 3)

    relation = encoder(
        feature_map,
        boxes,
        scale_features=scale_features,
        image_hw=(16, 16),
    )

    assert relation.shape == (2, 4 * 8 + 3)
    torch.testing.assert_close(relation[:, -3:], scale_features)
    relation.square().mean().backward()
    assert feature_map.grad is not None
    assert feature_map.grad.abs().sum() > 0


@pytest.mark.parametrize("num_classes", [2, 3])
def test_reliability_head_supports_mvp_and_uncertain_variants(num_classes):
    head = ReliabilityHead(input_dim=11, hidden_dim=7, num_classes=num_classes)
    logits = head(torch.randn(5, 11))
    assert logits.shape == (5, num_classes)


def test_rectifier_is_differentiable_and_changes_only_candidate_pixels():
    rectifier = InstanceLogitRectifier(max_delta=4.0)
    coarse = torch.zeros(2, 1, 10, 10, requires_grad=True)
    scores = torch.tensor([0.8, 0.2], requires_grad=True)
    masks = torch.zeros(2, 10, 10, dtype=torch.bool)
    masks[0, 1:4, 2:6] = True
    masks[1, 6:9, 5:8] = True
    batch_indices = torch.tensor([0, 1])

    refined, deltas = rectifier(coarse, scores, masks, batch_indices)

    expected_delta = math.log(4.0)
    torch.testing.assert_close(
        deltas.detach(), torch.tensor([expected_delta, -expected_delta]), atol=1e-6, rtol=1e-6
    )
    changed = (refined.detach() != coarse.detach()).squeeze(1)
    expected_changed = torch.zeros_like(changed)
    expected_changed[0] = masks[0]
    expected_changed[1] = masks[1]
    assert torch.equal(changed, expected_changed)
    assert torch.equal(refined.detach()[0, 0][~masks[0]], coarse.detach()[0, 0][~masks[0]])
    assert torch.equal(refined.detach()[1, 0][~masks[1]], coarse.detach()[1, 0][~masks[1]])

    refined.square().sum().backward()
    assert coarse.grad is not None and coarse.grad.abs().sum() > 0
    assert scores.grad is not None and scores.grad.abs().sum() > 0


def test_ccrr_module_batch_outputs_scale_features_and_gradients():
    torch.manual_seed(7)
    module = CCRRModule(
        feature_channels=4,
        num_scales=2,
        roi_size=3,
        hidden_dim=8,
        context_scale=2.5,
        num_classes=3,
    )
    feature_map = torch.randn(2, 4, 16, 16, requires_grad=True)
    coarse = (torch.randn(2, 1, 16, 16) * 0.1).requires_grad_()
    multi_scale_logits = [
        torch.zeros(2, 1, 16, 16),
        torch.ones(2, 1, 8, 8),
    ]
    boxes, masks = _two_image_candidates()

    refined, outputs = module(
        feature_map,
        coarse,
        multi_scale_logits,
        boxes,
        masks,
    )

    required = {
        "class_logits",
        "class_probs",
        "target_scores",
        "clutter_scores",
        "uncertain_scores",
        "deltas",
        "boxes",
    }
    assert required.issubset(outputs)
    assert refined.shape == coarse.shape
    assert outputs["class_logits"].shape == (2, 3)
    assert outputs["class_probs"].shape == (2, 3)
    assert outputs["deltas"].shape == (2,)
    assert outputs["boxes"].shape == (2, 5)
    assert outputs["candidate_masks"].shape == (2, 16, 16)
    assert outputs["batch_indices"].tolist() == [0, 1]
    torch.testing.assert_close(outputs["class_probs"].sum(1), torch.ones(2))

    expected_scale_means = torch.tensor([[0.5, torch.sigmoid(torch.tensor(1.0))]]).repeat(2, 1)
    torch.testing.assert_close(outputs["scale_features"][:, :2], expected_scale_means)
    expected_variance = expected_scale_means.var(dim=1, unbiased=False)
    torch.testing.assert_close(outputs["scale_features"][:, 2], expected_variance)

    union_by_batch = torch.zeros((2, 16, 16), dtype=torch.bool)
    union_by_batch[0] = masks[0]
    union_by_batch[1] = masks[1]
    difference = (refined - coarse).detach().squeeze(1)
    assert torch.count_nonzero(difference[~union_by_batch]) == 0

    refined.square().mean().backward()
    assert feature_map.grad is not None and feature_map.grad.abs().sum() > 0
    assert coarse.grad is not None and coarse.grad.abs().sum() > 0
    assert module.reliability_head.classifier[0].weight.grad is not None
    assert module.reliability_head.classifier[0].weight.grad.abs().sum() > 0


def test_ccrr_module_accepts_per_image_candidate_lists():
    module = CCRRModule(feature_channels=2, num_scales=1, hidden_dim=4, roi_size=3)
    feature_map = torch.randn(2, 2, 12, 12)
    coarse = torch.randn(2, 1, 12, 12)
    scales = [torch.randn(2, 1, 6, 6)]
    boxes = [
        torch.tensor([[1.0, 2.0, 5.0, 6.0]]),
        torch.tensor([[6.0, 5.0, 10.0, 9.0]]),
    ]
    masks = [torch.zeros(1, 12, 12), torch.zeros(1, 12, 12)]
    masks[0][0, 2:6, 1:5] = 1
    masks[1][0, 5:9, 6:10] = 1

    refined, outputs = module(feature_map, coarse, scales, boxes, masks)

    assert refined.shape == coarse.shape
    assert outputs["boxes"][:, 0].long().tolist() == [0, 1]
    assert outputs["batch_indices"].tolist() == [0, 1]


def test_scale_variance_keeps_pixelwise_cross_scale_disagreement():
    module = CCRRModule(feature_channels=2, num_scales=2, hidden_dim=4, roi_size=3)
    feature_map = torch.randn(1, 2, 4, 4)
    coarse = torch.zeros(1, 1, 4, 4)
    first_probability = torch.full((1, 1, 4, 4), 0.5)
    second_probability = torch.full((1, 1, 4, 4), 0.5)
    first_probability[0, 0, 1:3, 1:3] = torch.tensor([[0.2, 0.8], [0.8, 0.2]])
    second_probability[0, 0, 1:3, 1:3] = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    scales = [torch.logit(first_probability), torch.logit(second_probability)]
    boxes = torch.tensor([[0, 1, 1, 3, 3]], dtype=torch.float32)
    masks = torch.zeros(1, 4, 4, dtype=torch.bool)
    masks[0, 1:3, 1:3] = True

    _, outputs = module(feature_map, coarse, scales, boxes, masks)

    # Both ROI means are 0.5, but every ROI pixel disagrees by +/-0.3.
    torch.testing.assert_close(outputs["scale_features"][0, :2], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(outputs["scale_features"][0, 2], torch.tensor(0.09))


def test_binary_mvp_exposes_zero_uncertain_scores():
    module = CCRRModule(
        feature_channels=2,
        num_scales=1,
        hidden_dim=4,
        roi_size=3,
        num_classes=2,
    )
    feature_map = torch.randn(1, 2, 8, 8)
    coarse = torch.zeros(1, 1, 8, 8)
    boxes = torch.tensor([[0, 2, 2, 5, 5]], dtype=torch.float32)
    masks = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks[0, 2:5, 2:5] = True

    _, outputs = module(feature_map, coarse, [coarse], boxes, masks)

    assert outputs["class_logits"].shape == (1, 2)
    assert outputs["class_probs"].shape == (1, 2)
    assert torch.equal(outputs["uncertain_scores"], torch.zeros(1))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_empty_candidates_are_a_no_op_with_stable_shapes(batch_size):
    module = CCRRModule(feature_channels=3, num_scales=2, hidden_dim=4, roi_size=3)
    feature_map = torch.randn(batch_size, 3, 8, 8, requires_grad=True)
    coarse = torch.randn(batch_size, 1, 8, 8, requires_grad=True)
    scales = [torch.randn(batch_size, 1, 8, 8), torch.randn(batch_size, 1, 4, 4)]
    boxes = torch.empty(0, 5)
    masks = torch.empty(0, 8, 8, dtype=torch.bool)

    refined, outputs = module(feature_map, coarse, scales, boxes, masks)

    assert refined is coarse
    assert outputs["class_logits"].shape == (0, 3)
    assert outputs["class_probs"].shape == (0, 3)
    assert outputs["target_scores"].shape == (0,)
    assert outputs["clutter_scores"].shape == (0,)
    assert outputs["uncertain_scores"].shape == (0,)
    assert outputs["deltas"].shape == (0,)
    assert outputs["boxes"].shape == (0, 5)
    assert outputs["candidate_masks"].shape == (0, 8, 8)
    assert outputs["scale_features"].shape == (0, 3)
