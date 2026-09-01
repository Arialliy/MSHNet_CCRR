from copy import deepcopy

import pytest

from utils.pareto import (
    BestParetoState,
    is_pareto_feasible,
    pareto_constraints,
    pareto_key,
)


@pytest.fixture
def coarse_metrics():
    return {
        "mIoU": 0.70,
        "nIoU": 0.65,
        "Pd": 0.90,
        "FPPI": 0.20,
        "Fa_per_million_pixels": 10.0,
        "object_f1": 0.85,
    }


def feasible_refined(coarse_metrics):
    refined = dict(coarse_metrics)
    refined.update(
        {
            "mIoU": 0.71,
            "nIoU": 0.66,
            "Pd": 0.91,
            "FPPI": 0.15,
            "Fa_per_million_pixels": 8.0,
            "object_f1": 0.90,
        }
    )
    return refined


@pytest.mark.parametrize(
    ("metric", "bad_value", "constraint"),
    [
        ("mIoU", 0.70 - 1e-6, "mIoU_not_below"),
        ("nIoU", 0.65 - 1e-6, "nIoU_not_below"),
        ("Pd", 0.90 - 1e-6, "Pd_not_below"),
        ("FPPI", 0.20 + 1e-6, "FPPI_not_above"),
        (
            "Fa_per_million_pixels",
            10.0 + 1e-6,
            "Fa_not_above",
        ),
    ],
)
def test_each_non_inferiority_violation_is_rejected(
    coarse_metrics, metric, bad_value, constraint
):
    refined = dict(coarse_metrics)
    refined[metric] = bad_value

    constraints = pareto_constraints(coarse_metrics, refined)

    assert constraints[constraint] is False
    assert sum(not accepted for accepted in constraints.values()) == 1
    assert not is_pareto_feasible(coarse_metrics, refined)


def test_default_tolerance_accepts_only_roundoff_sized_differences(coarse_metrics):
    within = dict(coarse_metrics)
    within["mIoU"] -= 0.5e-12
    within["FPPI"] += 0.5e-12
    outside = dict(coarse_metrics)
    outside["Pd"] -= 2e-12

    assert is_pareto_feasible(coarse_metrics, within)
    assert not is_pareto_feasible(coarse_metrics, outside)


def test_pareto_key_has_locked_lexicographic_order(coarse_metrics):
    refined = feasible_refined(coarse_metrics)

    assert pareto_key(refined) == (-0.15, -8.0, 0.90, 0.71, 0.66)

    lower_fppi = dict(refined, FPPI=0.14, object_f1=0.10)
    lower_fa = dict(refined, Fa_per_million_pixels=7.0, object_f1=0.10)
    higher_object_f1 = dict(refined, object_f1=0.91, mIoU=0.1)
    assert pareto_key(lower_fppi) > pareto_key(refined)
    assert pareto_key(lower_fa) > pareto_key(refined)
    assert pareto_key(higher_object_f1) > pareto_key(refined)


def test_state_updates_only_for_a_strictly_better_feasible_candidate(
    coarse_metrics,
):
    state = BestParetoState()
    infeasible = dict(coarse_metrics, Pd=coarse_metrics["Pd"] - 1e-4)
    first = feasible_refined(coarse_metrics)
    worse_key = dict(first, Fa_per_million_pixels=9.0)
    better_key = dict(first, FPPI=0.14, object_f1=0.80)

    assert not state.consider(500, coarse_metrics, infeasible)
    assert not state.found
    assert state.consider(501, coarse_metrics, first)
    assert state.found
    assert not state.consider(502, coarse_metrics, worse_key)
    assert not state.consider(503, coarse_metrics, dict(first))
    assert state.epoch == 501
    assert state.consider(504, coarse_metrics, better_key)
    assert state.epoch == 504
    assert state.key == pareto_key(better_key)


def test_state_snapshots_and_serialization_do_not_alias_inputs(coarse_metrics):
    refined = feasible_refined(coarse_metrics)
    state = BestParetoState()
    assert state.consider(17, coarse_metrics, refined)

    expected_metrics = {
        "coarse": deepcopy(coarse_metrics),
        "refined": deepcopy(refined),
    }
    payload = state.state_dict()
    coarse_metrics["mIoU"] = -1.0
    refined["mIoU"] = -1.0
    payload["metrics"]["refined"]["mIoU"] = -2.0

    assert state.metrics == expected_metrics
    assert state.state_dict()["key"] == list(state.key)


def test_state_round_trip_restores_tuple_key_and_metrics(coarse_metrics):
    original = BestParetoState()
    refined = feasible_refined(coarse_metrics)
    assert original.consider(601, coarse_metrics, refined)

    restored = BestParetoState.from_checkpoint(
        {"epoch": 999, "best_pareto": original.state_dict()}
    )

    assert restored.key == original.key
    assert isinstance(restored.key, tuple)
    assert restored.epoch == 601
    assert restored.metrics == original.metrics
    assert restored.state_dict() == original.state_dict()


def test_old_checkpoint_without_pareto_state_loads_as_empty():
    restored = BestParetoState.from_checkpoint(
        {"epoch": 499, "best_miou": 0.7, "best_pd": 0.9}
    )

    assert not restored.found
    assert restored.key is None
    assert restored.epoch is None
    assert restored.metrics is None
    assert restored.state_dict() == {"key": None, "epoch": None, "metrics": None}


def test_invalid_tolerance_and_nonfinite_metrics_are_rejected(coarse_metrics):
    refined = feasible_refined(coarse_metrics)

    with pytest.raises(ValueError, match="tolerance"):
        is_pareto_feasible(coarse_metrics, refined, tolerance=-1.0)
    with pytest.raises(ValueError, match="object_f1"):
        pareto_key(dict(refined, object_f1=float("nan")))
