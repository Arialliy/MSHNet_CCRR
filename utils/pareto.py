"""Pure helpers for selecting a non-inferior Pareto checkpoint.

The selection rule is intentionally independent from checkpoint file I/O.  A
caller can use :class:`BestParetoState` while training, put ``state_dict()``
under the ``best_pareto`` checkpoint key, and decide separately when or where
to save model weights.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, TypeAlias


DEFAULT_TOLERANCE = 1e-12
CHECKPOINT_STATE_KEY = "best_pareto"

ParetoKey: TypeAlias = tuple[float, float, float, float, float]


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    """Return one finite scalar metric or fail loudly on invalid evidence."""

    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"metric {name!r} must be finite, got {value!r}")
    return value


def _validated_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return value


def pareto_constraints(
    coarse_metrics: Mapping[str, Any],
    refined_metrics: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, bool]:
    """Evaluate every locked non-inferiority constraint.

    ``mIoU``, ``nIoU`` and ``Pd`` may not decrease, while ``FPPI`` and
    ``Fa_per_million_pixels`` may not increase.  The returned names are stable
    and suitable for inclusion in checkpoint selection details.
    """

    tolerance = _validated_tolerance(tolerance)
    coarse_miou = _metric(coarse_metrics, "mIoU")
    coarse_niou = _metric(coarse_metrics, "nIoU")
    coarse_pd = _metric(coarse_metrics, "Pd")
    coarse_fppi = _metric(coarse_metrics, "FPPI")
    coarse_fa = _metric(coarse_metrics, "Fa_per_million_pixels")
    refined_miou = _metric(refined_metrics, "mIoU")
    refined_niou = _metric(refined_metrics, "nIoU")
    refined_pd = _metric(refined_metrics, "Pd")
    refined_fppi = _metric(refined_metrics, "FPPI")
    refined_fa = _metric(refined_metrics, "Fa_per_million_pixels")

    return {
        "mIoU_not_below": refined_miou + tolerance >= coarse_miou,
        "nIoU_not_below": refined_niou + tolerance >= coarse_niou,
        "Pd_not_below": refined_pd + tolerance >= coarse_pd,
        "FPPI_not_above": refined_fppi <= coarse_fppi + tolerance,
        "Fa_not_above": refined_fa <= coarse_fa + tolerance,
    }


def is_pareto_feasible(
    coarse_metrics: Mapping[str, Any],
    refined_metrics: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return whether refined metrics are non-inferior on all five metrics."""

    return all(
        pareto_constraints(
            coarse_metrics,
            refined_metrics,
            tolerance=tolerance,
        ).values()
    )


def pareto_key(refined_metrics: Mapping[str, Any]) -> ParetoKey:
    """Return the locked lexicographic ranking key for a feasible candidate."""

    return (
        -_metric(refined_metrics, "FPPI"),
        -_metric(refined_metrics, "Fa_per_million_pixels"),
        _metric(refined_metrics, "object_f1"),
        _metric(refined_metrics, "mIoU"),
        _metric(refined_metrics, "nIoU"),
    )


@dataclass
class BestParetoState:
    """Track the lexicographically best feasible checkpoint without file I/O."""

    key: ParetoKey | None = None
    epoch: int | None = None
    metrics: dict[str, dict[str, Any]] | None = None

    @property
    def found(self) -> bool:
        """Whether at least one Pareto-feasible candidate has been accepted."""

        return self.key is not None

    def consider(
        self,
        epoch: int,
        coarse_metrics: Mapping[str, Any],
        refined_metrics: Mapping[str, Any],
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> bool:
        """Accept a candidate only when feasible and strictly better by key.

        Returns ``True`` exactly when this state was updated.  Metric snapshots
        are deep-copied so later accumulator mutation cannot alter the recorded
        model-selection evidence.
        """

        if not is_pareto_feasible(
            coarse_metrics,
            refined_metrics,
            tolerance=tolerance,
        ):
            return False

        candidate_key = pareto_key(refined_metrics)
        if self.key is not None and candidate_key <= self.key:
            return False

        self.key = candidate_key
        self.epoch = int(epoch)
        self.metrics = {
            "coarse": deepcopy(dict(coarse_metrics)),
            "refined": deepcopy(dict(refined_metrics)),
        }
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return the serializable payload stored under ``best_pareto``."""

        return {
            "key": list(self.key) if self.key is not None else None,
            "epoch": self.epoch,
            "metrics": deepcopy(self.metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Load a new-format state; ``None`` is an empty legacy-compatible state."""

        if not isinstance(state, Mapping) or state.get("key") is None:
            self.key = None
            self.epoch = None
            self.metrics = None
            return

        raw_key = state["key"]
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) != 5:
            raise ValueError("best_pareto key must contain exactly five values")
        loaded_key = tuple(float(value) for value in raw_key)
        if not all(math.isfinite(value) for value in loaded_key):
            raise ValueError("best_pareto key values must be finite")

        epoch = state.get("epoch")
        metrics = state.get("metrics")
        if epoch is None or not isinstance(metrics, Mapping):
            raise ValueError(
                "a populated best_pareto state requires epoch and metrics"
            )

        self.key = loaded_key  # type: ignore[assignment]
        self.epoch = int(epoch)
        self.metrics = deepcopy(dict(metrics))

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any] | None
    ) -> "BestParetoState":
        """Construct from a serialized state, treating a missing state as empty."""

        instance = cls()
        instance.load_state_dict(state)
        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "BestParetoState":
        """Restore from a checkpoint while accepting old checkpoints unchanged."""

        return cls.from_state_dict(checkpoint.get(CHECKPOINT_STATE_KEY))


__all__ = [
    "BestParetoState",
    "CHECKPOINT_STATE_KEY",
    "DEFAULT_TOLERANCE",
    "ParetoKey",
    "is_pareto_feasible",
    "pareto_constraints",
    "pareto_key",
]
