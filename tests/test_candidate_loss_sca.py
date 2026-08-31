import pytest
import torch
import torch.nn.functional as F

from model.candidate_loss import (
    AsymmetricActionRiskLoss,
    CandidateRankLoss,
    TargetQualityLoss,
)


def test_target_quality_loss_matches_smooth_l1_and_backpropagates():
    prediction = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
    target = torch.tensor([0.0, 0.7, 1.0])

    actual = TargetQualityLoss(beta=0.5)(prediction, target)
    expected = F.smooth_l1_loss(prediction, target, beta=0.5)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_target_quality_loss_supports_all_reductions():
    prediction = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.0, 1.0])
    per_candidate = F.smooth_l1_loss(prediction, target, reduction="none")

    torch.testing.assert_close(
        TargetQualityLoss(reduction="none")(prediction, target), per_candidate
    )
    torch.testing.assert_close(
        TargetQualityLoss(reduction="sum")(prediction, target), per_candidate.sum()
    )


def test_asymmetric_action_risk_matches_design_equation():
    gates = torch.tensor([0.2, 0.6, 0.9, 0.4], requires_grad=True)
    is_target = torch.tensor([True, True, False, False])
    is_fp = torch.tensor([False, False, True, True])

    losses = AsymmetricActionRiskLoss(
        target_harm_weight=20.0,
        missed_clutter_weight=1.0,
    )(gates, is_target, is_fp)

    target_harm = gates[:2].mean()
    missed_clutter = (1.0 - gates[2:]).mean()
    torch.testing.assert_close(losses["target_harm"], target_harm)
    torch.testing.assert_close(losses["missed_clutter"], missed_clutter)
    torch.testing.assert_close(losses["total"], 20.0 * target_harm + missed_clutter)
    losses["total"].backward()
    assert gates.grad is not None and torch.isfinite(gates.grad).all()


def test_asymmetric_action_risk_ignores_unassigned_candidates():
    gates = torch.tensor([0.25, 0.75, 0.99])
    losses = AsymmetricActionRiskLoss(2.0, 3.0)(
        gates,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(losses["total"], torch.tensor(1.25))


def test_candidate_rank_loss_matches_all_fp_target_pairs():
    risk = torch.tensor([0.2, 0.8, 1.0, -0.1], requires_grad=True)
    is_target = torch.tensor([True, False, False, True])
    is_fp = torch.tensor([False, True, True, False])

    actual = CandidateRankLoss(margin=0.5)(risk, is_target, is_fp)
    target_risk = risk[[0, 3]]
    fp_risk = risk[[1, 2]]
    expected = F.relu(0.5 - fp_risk[:, None] + target_risk[None, :]).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert risk.grad is not None and torch.isfinite(risk.grad).all()


@pytest.mark.parametrize("missing_class", ["target", "fp", "both"])
def test_sca_losses_are_differentiable_for_empty_or_missing_sets(missing_class):
    size = 0 if missing_class == "both" else 2
    quality = torch.zeros(size, requires_grad=True)
    quality_target = torch.zeros(size)
    quality_loss = TargetQualityLoss()(quality, quality_target)

    gates = torch.full((size,), 0.5, requires_grad=True)
    risk = torch.arange(size, dtype=torch.float32, requires_grad=True)
    if missing_class == "target":
        is_target = torch.zeros(size, dtype=torch.bool)
        is_fp = torch.ones(size, dtype=torch.bool)
    elif missing_class == "fp":
        is_target = torch.ones(size, dtype=torch.bool)
        is_fp = torch.zeros(size, dtype=torch.bool)
    else:
        is_target = torch.zeros(size, dtype=torch.bool)
        is_fp = torch.zeros(size, dtype=torch.bool)

    action = AsymmetricActionRiskLoss()(gates, is_target, is_fp)["total"]
    rank = CandidateRankLoss()(risk, is_target, is_fp)
    total = quality_loss + action + rank

    assert torch.isfinite(total)
    total.backward()
    assert quality.grad is not None
    assert gates.grad is not None
    assert risk.grad is not None


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (torch.tensor([[0.5]]), torch.tensor([0.5]), r"shape \[N\]"),
        (torch.tensor([1.1]), torch.tensor([0.5]), r"lie in \[0, 1\]"),
        (torch.tensor([0.5]), torch.tensor([float("nan")]), "must be finite"),
        (torch.tensor([0.5]), torch.tensor([0.5, 0.6]), "identical shapes"),
    ],
)
def test_target_quality_loss_rejects_invalid_inputs(prediction, target, message):
    with pytest.raises(ValueError, match=message):
        TargetQualityLoss()(prediction, target)


def test_action_and_rank_losses_reject_overlapping_or_non_boolean_membership():
    values = torch.tensor([0.2, 0.8])
    overlap = torch.tensor([True, False])
    with pytest.raises(ValueError, match="cannot be both"):
        AsymmetricActionRiskLoss()(values, overlap, overlap)

    with pytest.raises(TypeError, match="torch.bool"):
        CandidateRankLoss()(
            values,
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TargetQualityLoss(beta=0.0), "beta"),
        (lambda: AsymmetricActionRiskLoss(target_harm_weight=-1.0), "target_harm"),
        (lambda: AsymmetricActionRiskLoss(missed_clutter_weight=float("inf")), "missed_clutter"),
        (lambda: CandidateRankLoss(margin=-0.1), "margin"),
    ],
)
def test_sca_loss_hyperparameters_are_validated(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
