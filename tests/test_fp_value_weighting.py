import pytest
import torch

from model.candidate_loss import AsymmetricActionRiskLoss


def test_weighted_fp_loss_matches_normalized_value_equation():
    gates = torch.tensor([0.1, 0.2, 0.8, 0.4], requires_grad=True)
    is_target = torch.tensor([True, False, False, False])
    is_fp = torch.tensor([False, True, True, True])
    weights = torch.tensor([99.0, 1.0, 2.0, 3.0])

    terms = AsymmetricActionRiskLoss(
        target_harm_weight=2.0,
        missed_clutter_weight=3.0,
    )(gates, is_target, is_fp, fp_value_weights=weights)

    expected_fp = (weights[is_fp] * (1.0 - gates[is_fp])).sum() / weights[
        is_fp
    ].sum()
    torch.testing.assert_close(terms["missed_clutter"], expected_fp)
    torch.testing.assert_close(
        terms["mean_fp_value_weight"], weights[is_fp].mean()
    )
    torch.testing.assert_close(
        terms["total"], 2.0 * gates[is_target].mean() + 3.0 * expected_fp
    )
    terms["total"].backward()
    assert gates.grad is not None and torch.isfinite(gates.grad).all()


def test_risk_loss_uses_weights_only_on_fp_branch():
    gates = torch.tensor([0.25, 0.75, 0.2, 0.8])
    is_target = torch.tensor([True, True, False, False])
    is_fp = ~is_target
    criterion = AsymmetricActionRiskLoss(target_tail_weight=5.0)
    weights_a = torch.tensor([1.0, 1.0, 1.0, 3.0])
    weights_b = torch.tensor([1000.0, 0.001, 1.0, 3.0])

    terms_a = criterion(gates, is_target, is_fp, weights_a)
    terms_b = criterion(gates, is_target, is_fp, weights_b)

    torch.testing.assert_close(terms_a["target_harm"], terms_b["target_harm"])
    torch.testing.assert_close(terms_a["target_tail"], terms_b["target_tail"])
    torch.testing.assert_close(
        terms_a["missed_clutter"], terms_b["missed_clutter"]
    )
    torch.testing.assert_close(terms_a["total"], terms_b["total"])


def test_equal_fp_weights_reproduce_old_loss():
    gates = torch.tensor([0.2, 0.6, 0.9, 0.4])
    is_target = torch.tensor([True, True, False, False])
    is_fp = ~is_target
    criterion = AsymmetricActionRiskLoss(2.0, 3.0)

    legacy = criterion(gates, is_target, is_fp)
    equally_weighted = criterion(
        gates,
        is_target,
        is_fp,
        fp_value_weights=torch.ones_like(gates),
    )

    torch.testing.assert_close(equally_weighted["total"], legacy["total"])
    torch.testing.assert_close(
        equally_weighted["missed_clutter"], legacy["missed_clutter"]
    )
    torch.testing.assert_close(
        equally_weighted["mean_fp_value_weight"], gates.new_ones(())
    )


def test_no_fp_batch_returns_differentiable_zero_terms():
    gates = torch.tensor([0.2, 0.8], requires_grad=True)
    is_target = torch.ones(2, dtype=torch.bool)
    is_fp = torch.zeros(2, dtype=torch.bool)

    terms = AsymmetricActionRiskLoss()(gates, is_target, is_fp)

    torch.testing.assert_close(terms["missed_clutter"], gates.new_zeros(()))
    torch.testing.assert_close(
        terms["mean_fp_value_weight"], gates.new_zeros(())
    )
    terms["missed_clutter"].backward()
    assert gates.grad is not None
    torch.testing.assert_close(gates.grad, torch.zeros_like(gates))


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        ([1.0, 1.0], TypeError, "tensor or None"),
        (torch.ones(2, 1), ValueError, "shape"),
        (torch.ones(2, dtype=torch.long), TypeError, "floating point"),
        (torch.tensor([1.0, float("nan")]), ValueError, "finite"),
        (torch.tensor([1.0, -0.1]), ValueError, "non-negative"),
    ],
)
def test_fp_value_weights_are_strictly_validated(weights, error, message):
    gates = torch.tensor([0.2, 0.8])
    is_target = torch.tensor([True, False])
    is_fp = ~is_target

    with pytest.raises(error, match=message):
        AsymmetricActionRiskLoss()(
            gates, is_target, is_fp, fp_value_weights=weights
        )
