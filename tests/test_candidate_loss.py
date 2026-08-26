import torch
import torch.nn.functional as F

from model.candidate_loss import (
    CCRRLoss,
    CandidateBrierLoss,
    CandidateClassificationLoss,
    RectificationPreservationLoss,
)


def test_weighted_classification_matches_cross_entropy_for_three_classes():
    logits = torch.tensor(
        [[2.0, -0.5, 0.2], [-1.0, 1.5, 0.3], [0.1, 0.2, 1.1]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2])
    weights = torch.tensor([1.0, 2.0, 0.5])

    actual = CandidateClassificationLoss(weights)(logits, labels)
    expected = F.cross_entropy(logits, labels, weight=weights)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_binary_losses_ignore_minus_one_candidates():
    logits = torch.tensor([[2.0, -1.0], [-3.0, 3.0], [0.5, -0.5]])
    labels = torch.tensor([0, -1, 1])

    classification = CandidateClassificationLoss(ignore_index=-1)(logits, labels)
    expected_classification = F.cross_entropy(logits[[0, 2]], labels[[0, 2]])
    torch.testing.assert_close(classification, expected_classification)

    calibration = CandidateBrierLoss(ignore_index=-1)(logits, labels)
    probabilities = logits[[0, 2]].softmax(dim=1)
    expected_calibration = (
        probabilities - F.one_hot(labels[[0, 2]], num_classes=2)
    ).square().sum(dim=1).mean()
    torch.testing.assert_close(calibration, expected_calibration)


def test_zero_weight_only_class_returns_finite_differentiable_zero():
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([1])

    loss = CandidateClassificationLoss([1.0, 0.0])(logits, labels)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_brier_loss_accepts_probabilities_for_two_and_three_classes():
    binary_probabilities = torch.tensor([[0.8, 0.2], [0.25, 0.75]])
    binary_labels = torch.tensor([0, 1])
    binary_loss = CandidateBrierLoss(from_logits=False)(
        binary_probabilities, binary_labels
    )
    torch.testing.assert_close(binary_loss, torch.tensor(0.1025))

    three_class_probabilities = torch.tensor([[0.7, 0.2, 0.1]])
    three_class_labels = torch.tensor([0])
    three_class_loss = CandidateBrierLoss(from_logits=False)(
        three_class_probabilities, three_class_labels
    )
    torch.testing.assert_close(three_class_loss, torch.tensor(0.14))


def test_preservation_loss_uses_candidate_batch_indices_and_class_policy():
    coarse_probability = torch.tensor([0.8, 0.8]).view(2, 1, 1, 1).expand(2, 1, 2, 2)
    refined_probability = torch.tensor([0.7, 0.75]).view(2, 1, 1, 1).expand(2, 1, 2, 2)
    coarse_logits = torch.logit(coarse_probability)
    refined_logits = torch.logit(refined_probability)
    masks = torch.ones((2, 2, 2), dtype=torch.bool)
    labels = torch.tensor([0, 1])
    batch_indices = torch.tensor([0, 1])

    loss = RectificationPreservationLoss(margin=0.1)(
        coarse_logits,
        refined_logits,
        masks,
        labels,
        batch_indices,
    )

    # Target: relu(0.8 - 0.7) = 0.1.
    # Clutter: relu(0.75 - 0.8 + 0.1) = 0.05.
    torch.testing.assert_close(loss, torch.tensor(0.15), atol=1e-6, rtol=1e-6)


def test_preservation_ignores_uncertain_and_zero_area_masks():
    coarse = torch.zeros((1, 1, 2, 2), requires_grad=True)
    refined = torch.ones((1, 1, 2, 2), requires_grad=True)
    masks = torch.stack(
        (torch.ones((2, 2), dtype=torch.bool), torch.zeros((2, 2), dtype=torch.bool))
    )
    labels = torch.tensor([2, 1])

    loss = RectificationPreservationLoss()(coarse, refined, masks, labels)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert coarse.grad is not None
    assert refined.grad is not None


def test_ccrr_loss_reads_masks_and_batch_indices_from_candidate_outputs():
    class_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    masks = torch.ones((2, 2, 2), dtype=torch.bool)
    outputs = {
        "class_logits": class_logits,
        "candidate_masks": masks,
        "batch_indices": torch.tensor([0, 1]),
    }
    labels = torch.tensor([0, 1])
    coarse = torch.zeros((2, 1, 2, 2), requires_grad=True)
    refined = torch.zeros((2, 1, 2, 2), requires_grad=True)

    losses = CCRRLoss(calibration_weight=0.2, preservation_weight=0.3)(
        outputs,
        labels,
        coarse,
        refined,
    )

    assert set(losses) == {
        "classification",
        "calibration",
        "preservation",
        "total",
    }
    expected_total = (
        losses["classification"]
        + 0.2 * losses["calibration"]
        + 0.3 * losses["preservation"]
    )
    torch.testing.assert_close(losses["total"], expected_total)
    losses["total"].backward()
    assert class_logits.grad is not None
    assert coarse.grad is not None
    assert refined.grad is not None


def test_all_candidate_losses_support_empty_batches_and_backward():
    logits = torch.empty((0, 3), requires_grad=True)
    labels = torch.empty((0,), dtype=torch.long)
    masks = torch.empty((0, 4, 5), dtype=torch.bool)
    coarse = torch.randn((2, 1, 4, 5), requires_grad=True)
    refined = torch.randn((2, 1, 4, 5), requires_grad=True)

    classification = CandidateClassificationLoss()(logits, labels)
    calibration = CandidateBrierLoss()(logits, labels)
    preservation = RectificationPreservationLoss()(
        coarse,
        refined,
        masks,
        labels,
        torch.empty((0,), dtype=torch.long),
    )
    total = classification + calibration + preservation

    torch.testing.assert_close(total, torch.tensor(0.0))
    total.backward()
    assert logits.grad is not None
    assert coarse.grad is not None
    assert refined.grad is not None
