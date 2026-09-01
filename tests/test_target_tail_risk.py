import pytest
import torch

from model.candidate_loss import AsymmetricActionRiskLoss


def _target_only_masks(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(size, dtype=torch.bool), torch.zeros(size, dtype=torch.bool)


def test_tail_is_zero_for_zero_target_gates():
    gates = torch.zeros(5, requires_grad=True)
    is_target, is_fp = _target_only_masks(gates.numel())

    terms = AsymmetricActionRiskLoss(
        target_tail_weight=5.0,
        target_tail_temperature=0.1,
    )(gates, is_target, is_fp)

    torch.testing.assert_close(terms["target_tail"], gates.new_zeros(()))
    torch.testing.assert_close(terms["total"], gates.new_zeros(()))
    terms["total"].backward()
    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()


def test_one_high_target_gate_increases_tail():
    is_target, is_fp = _target_only_masks(8)
    criterion = AsymmetricActionRiskLoss(
        target_tail_weight=1.0,
        target_tail_temperature=0.1,
    )

    safe = criterion(torch.zeros(8), is_target, is_fp)["target_tail"]
    one_spike = criterion(
        torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        is_target,
        is_fp,
    )["target_tail"]

    assert one_spike > safe


def test_tail_penalty_exceeds_mean_for_rare_spike():
    gates = torch.tensor(
        [0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        requires_grad=True,
    )
    is_target, is_fp = _target_only_masks(gates.numel())

    terms = AsymmetricActionRiskLoss(
        target_tail_weight=5.0,
        target_tail_temperature=0.1,
    )(gates, is_target, is_fp)

    assert terms["target_tail"] > terms["target_harm"]
    terms["target_tail"].backward()
    assert gates.grad is not None
    assert gates.grad[0] > gates.grad[1:].max()


def test_no_target_batch_returns_differentiable_zero():
    gates = torch.tensor([0.2, 0.8], requires_grad=True)
    is_target = torch.zeros(2, dtype=torch.bool)
    is_fp = torch.ones(2, dtype=torch.bool)

    tail = AsymmetricActionRiskLoss(target_tail_weight=5.0)(
        gates, is_target, is_fp
    )["target_tail"]

    torch.testing.assert_close(tail, gates.new_zeros(()))
    tail.backward()
    assert gates.grad is not None
    torch.testing.assert_close(gates.grad, torch.zeros_like(gates))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_tail_weight": -1.0}, "target_tail_weight"),
        ({"target_tail_weight": float("inf")}, "target_tail_weight"),
        ({"target_tail_temperature": 0.0}, "target_tail_temperature"),
        ({"target_tail_temperature": float("nan")}, "target_tail_temperature"),
    ],
)
def test_target_tail_hyperparameters_are_strictly_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AsymmetricActionRiskLoss(**kwargs)


def test_disabled_tail_preserves_legacy_value_and_empty_state_dict():
    gates = torch.tensor([0.2, 0.6, 0.9, 0.4])
    is_target = torch.tensor([True, True, False, False])
    is_fp = ~is_target
    criterion = AsymmetricActionRiskLoss(
        target_harm_weight=20.0,
        missed_clutter_weight=1.0,
    )

    terms = criterion(gates, is_target, is_fp)
    legacy = 20.0 * gates[is_target].mean() + (1.0 - gates[is_fp]).mean()

    assert torch.equal(terms["total"], legacy)
    assert criterion.target_tail_weight == 0.0
    assert criterion.state_dict() == {}
    criterion.load_state_dict({}, strict=True)
