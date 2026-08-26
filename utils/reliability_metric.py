"""Candidate calibration and false-alarm metrics for CCRR.

The calibration metrics operate on candidate class probabilities.  The FROC
helpers operate on a flat list of scored detections with a pre-computed
true-positive flag; this keeps matching policy separate from metric policy and
makes the implementation suitable for both online candidates and an offline
candidate bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # Torch is optional for callers that evaluate exported NumPy arrays.
    import torch
except ImportError:  # pragma: no cover - the project normally depends on torch
    torch = None  # type: ignore[assignment]


ArrayLike = Any


def _to_numpy(value: ArrayLike, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _prepare_labels(labels: ArrayLike) -> np.ndarray:
    array = _to_numpy(labels)
    if array.ndim == 2:
        array = array.argmax(axis=1)
    elif array.ndim != 1:
        raise ValueError(f"labels must have shape [N] or [N, C], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("labels must contain integer class indices")
    return array.astype(np.int64, copy=False)


def _prepare_probabilities(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    from_logits: bool,
    ignore_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid ``[N, C]`` probabilities and integer labels.

    A one-dimensional prediction array is treated as binary class-1 scores and
    converted to ``[1-p, p]``.  CCRR normally supplies an explicit ``[N, C]``
    matrix, which avoids any ambiguity about target class 0.
    """

    values = _to_numpy(predictions, dtype=np.float64)
    label_array = _prepare_labels(labels)

    if values.ndim == 1:
        if from_logits:
            # Stable sigmoid.
            positive = np.empty_like(values)
            nonnegative = values >= 0
            positive[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
            exp_values = np.exp(values[~nonnegative])
            positive[~nonnegative] = exp_values / (1.0 + exp_values)
        else:
            positive = values
        probabilities = np.column_stack((1.0 - positive, positive))
    elif values.ndim == 2:
        if values.shape[1] < 2:
            raise ValueError("predictions must contain at least two classes")
        if from_logits:
            shifted = values - np.max(values, axis=1, keepdims=True)
            exponentiated = np.exp(shifted)
            probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        else:
            probabilities = values
    else:
        raise ValueError(
            f"predictions must have shape [N] or [N, C], got {values.shape}"
        )

    if probabilities.shape[0] != label_array.shape[0]:
        raise ValueError(
            "predictions and labels disagree on candidate count: "
            f"{probabilities.shape[0]} != {label_array.shape[0]}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("candidate probabilities must be finite")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("candidate probabilities must lie in [0, 1]")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    if probabilities.shape[0] > 0:
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("each row of candidate probabilities must sum to one")

    valid = np.ones(label_array.shape[0], dtype=bool)
    if ignore_index is not None:
        valid &= label_array != int(ignore_index)
    valid_labels = label_array[valid]
    if valid_labels.size and (
        np.any(valid_labels < 0) or np.any(valid_labels >= probabilities.shape[1])
    ):
        raise ValueError(
            f"labels must lie in [0, {probabilities.shape[1] - 1}]"
            + (" or equal ignore_index" if ignore_index is not None else "")
        )
    return probabilities[valid], valid_labels


def candidate_ece(
    predictions: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 15,
    *,
    num_bins: int | None = None,
    target_class: int | None = None,
    from_logits: bool = False,
    ignore_index: int | None = -1,
    return_bins: bool = False,
) -> float | dict[str, Any]:
    """Compute candidate expected calibration error (ECE).

    With ``target_class=None`` (default), this is standard top-label ECE: the
    maximum class probability is calibrated against prediction correctness.
    Setting ``target_class=0`` instead evaluates the CCRR target reliability
    ``p(target)`` against the binary event that a candidate is a target.

    Empty/all-ignored candidate sets produce ``NaN`` because calibration is
    undefined without observations.  When ``return_bins=True``, counts and bin
    statistics are returned alongside ``ece``.
    """

    if num_bins is not None:
        n_bins = num_bins
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    n_bins = int(n_bins)

    probabilities, label_array = _prepare_probabilities(
        predictions,
        labels,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    num_candidates, num_classes = probabilities.shape
    if target_class is None:
        if num_candidates:
            predicted = probabilities.argmax(axis=1)
            confidence = probabilities[np.arange(num_candidates), predicted]
            outcome = predicted == label_array
        else:
            confidence = np.empty(0, dtype=np.float64)
            outcome = np.empty(0, dtype=bool)
    else:
        target_class = int(target_class)
        if not 0 <= target_class < num_classes:
            raise ValueError(f"target_class must lie in [0, {num_classes - 1}]")
        confidence = probabilities[:, target_class]
        outcome = label_array == target_class

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # searchsorted gives bins [0, 1/n), ..., with confidence==1 in the last bin.
    bin_index = np.searchsorted(edges[1:-1], confidence, side="right")
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.int64)
    bin_accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    bin_confidence = np.full(n_bins, np.nan, dtype=np.float64)

    for index in np.flatnonzero(counts):
        members = bin_index == index
        bin_accuracy[index] = outcome[members].mean()
        bin_confidence[index] = confidence[members].mean()

    if num_candidates == 0:
        ece = float("nan")
    else:
        populated = counts > 0
        gaps = np.abs(bin_accuracy[populated] - bin_confidence[populated])
        ece = float(np.sum((counts[populated] / num_candidates) * gaps))

    if return_bins:
        return {
            "ece": ece,
            "bin_edges": edges,
            "bin_count": counts,
            "bin_accuracy": bin_accuracy,
            "bin_confidence": bin_confidence,
            "num_candidates": num_candidates,
        }
    return ece


def candidate_brier_score(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    from_logits: bool = False,
    ignore_index: int | None = -1,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Compute the multiclass candidate Brier score.

    The per-candidate value is ``sum_k (p_k-y_k)^2``, matching the CCRR design
    document.  An empty mean is ``NaN``; ``reduction='none'`` returns an empty
    array and ``reduction='sum'`` returns zero.
    """

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
    probabilities, label_array = _prepare_probabilities(
        predictions,
        labels,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    one_hot = np.zeros_like(probabilities)
    if label_array.size:
        one_hot[np.arange(label_array.size), label_array] = 1.0
    per_candidate = np.square(probabilities - one_hot).sum(axis=1)

    if reduction == "none":
        return per_candidate
    if reduction == "sum":
        return float(per_candidate.sum())
    return float(per_candidate.mean()) if per_candidate.size else float("nan")


def candidate_nll(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    from_logits: bool = False,
    ignore_index: int | None = -1,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Compute candidate negative log-likelihood (NLL)."""

    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
    probabilities, label_array = _prepare_probabilities(
        predictions,
        labels,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    if label_array.size:
        true_probability = probabilities[np.arange(label_array.size), label_array]
        per_candidate = -np.log(np.clip(true_probability, eps, 1.0))
    else:
        per_candidate = np.empty(0, dtype=np.float64)

    if reduction == "none":
        return per_candidate
    if reduction == "sum":
        return float(per_candidate.sum())
    return float(per_candidate.mean()) if per_candidate.size else float("nan")


def false_positives_per_image(
    false_positives: int | float | Sequence[bool] | np.ndarray,
    num_images: int,
) -> float:
    """Return false positives per image (FPPI)."""

    if isinstance(num_images, bool) or int(num_images) != num_images or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    values = _to_numpy(false_positives)
    if values.ndim == 0:
        count = float(values)
    else:
        count = float(np.count_nonzero(values)) if values.dtype == bool else float(values.sum())
    if not np.isfinite(count) or count < 0:
        raise ValueError("false-positive count must be finite and non-negative")
    return count / int(num_images)


def fppi_froc(
    scores: ArrayLike,
    is_true_positive: ArrayLike,
    num_images: int,
    *,
    num_targets: int | None = None,
    thresholds: ArrayLike | None = None,
) -> dict[str, Any]:
    """Build a candidate-level FROC curve.

    Parameters
    ----------
    scores:
        One confidence score per predicted candidate (larger is better).
    is_true_positive:
        Boolean flag after the dataset's one-to-one candidate/GT matching.
        Every selected non-TP candidate is counted as a false positive.
    num_images:
        Number of evaluated images, including images with no candidates.
    num_targets:
        Total number of GT instances.  If omitted, the number of TP flags is
        used, which is suitable only when the supplied flags are one-to-one and
        cover every target at the lowest threshold.
    thresholds:
        Optional operating thresholds.  By default, ``+inf`` followed by all
        unique scores in descending order is used, so the curve includes the
        no-detection origin.

    Returns
    -------
    dict
        Arrays ``thresholds``, ``pd``, ``fppi``, ``true_positives`` and
        ``false_positives``.  Empty candidate sets still return a valid origin.
    """

    if isinstance(num_images, bool) or int(num_images) != num_images or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    num_images = int(num_images)

    score_array = _to_numpy(scores, dtype=np.float64).reshape(-1)
    true_positive = _to_numpy(is_true_positive).reshape(-1)
    if score_array.shape[0] != true_positive.shape[0]:
        raise ValueError("scores and is_true_positive must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("candidate scores must be finite")
    if true_positive.dtype != bool:
        if not np.all(np.isin(true_positive, (0, 1))):
            raise ValueError("is_true_positive must contain only boolean/0/1 values")
        true_positive = true_positive.astype(bool)

    observed_targets = int(true_positive.sum())
    if num_targets is None:
        num_targets = observed_targets
    if isinstance(num_targets, bool) or int(num_targets) != num_targets or num_targets < 0:
        raise ValueError("num_targets must be a non-negative integer")
    num_targets = int(num_targets)
    if observed_targets > num_targets:
        raise ValueError(
            "true-positive flags exceed num_targets; perform one-to-one GT matching "
            "before computing FROC"
        )

    if thresholds is None:
        threshold_array = np.concatenate(
            (np.array([np.inf]), np.unique(score_array)[::-1])
        )
    else:
        threshold_array = _to_numpy(thresholds, dtype=np.float64).reshape(-1)
        if threshold_array.size == 0:
            raise ValueError("thresholds cannot be empty")
        if np.any(np.isnan(threshold_array)):
            raise ValueError("thresholds cannot contain NaN")

    true_positive_counts = np.zeros(threshold_array.size, dtype=np.int64)
    false_positive_counts = np.zeros(threshold_array.size, dtype=np.int64)
    for index, threshold in enumerate(threshold_array):
        selected = score_array >= threshold
        true_positive_counts[index] = np.count_nonzero(selected & true_positive)
        false_positive_counts[index] = np.count_nonzero(selected & ~true_positive)

    if num_targets > 0:
        pd = true_positive_counts.astype(np.float64) / num_targets
    else:
        # Detection probability is undefined without GT targets.  NaN prevents
        # target-free datasets from silently appearing to have perfect recall.
        pd = np.full(threshold_array.size, np.nan, dtype=np.float64)
    fppi_values = false_positive_counts.astype(np.float64) / num_images

    return {
        "thresholds": threshold_array,
        "pd": pd,
        "fppi": fppi_values,
        "true_positives": true_positive_counts,
        "false_positives": false_positive_counts,
        "num_images": num_images,
        "num_targets": num_targets,
    }


def false_alarm_at_fixed_pd(
    pd: ArrayLike,
    false_alarm: ArrayLike,
    target_pd: float,
    *,
    interpolate: bool = False,
    unreachable_value: float = float("nan"),
) -> float:
    """Return the lowest false-alarm value that attains ``target_pd``.

    ``false_alarm`` can be FPPI, pixel-level Fa, or a raw false-positive count;
    the function preserves that unit.  With ``interpolate=False`` it chooses a
    realizable operating point.  Linear interpolation is available for smooth
    reporting tables.  ``unreachable_value`` is returned when the requested
    detection probability cannot be reached.
    """

    if not np.isfinite(target_pd) or not 0.0 <= target_pd <= 1.0:
        raise ValueError("target_pd must lie in [0, 1]")
    pd_array = _to_numpy(pd, dtype=np.float64).reshape(-1)
    false_alarm_array = _to_numpy(false_alarm, dtype=np.float64).reshape(-1)
    if pd_array.shape != false_alarm_array.shape:
        raise ValueError("pd and false_alarm must have the same shape")

    finite = np.isfinite(pd_array) & np.isfinite(false_alarm_array)
    pd_array = pd_array[finite]
    false_alarm_array = false_alarm_array[finite]
    if pd_array.size == 0 or pd_array.max() < target_pd:
        return float(unreachable_value)

    if not interpolate:
        return float(false_alarm_array[pd_array >= target_pd].min())

    order = np.argsort(pd_array, kind="stable")
    sorted_pd = pd_array[order]
    sorted_fa = false_alarm_array[order]
    unique_pd = np.unique(sorted_pd)
    # Multiple thresholds can have the same Pd.  Keep the least false-alarm
    # operating point before interpolation.
    best_fa = np.array(
        [sorted_fa[sorted_pd == probability].min() for probability in unique_pd],
        dtype=np.float64,
    )
    if target_pd <= unique_pd[0]:
        return float(best_fa[0])
    return float(np.interp(target_pd, unique_pd, best_fa))


def fppi_at_fixed_pd(
    curve: Mapping[str, ArrayLike],
    target_pd: float,
    *,
    interpolate: bool = False,
    unreachable_value: float = float("nan"),
) -> float:
    """Convenience wrapper for ``Fa@Pd`` on an FPPI/FROC curve."""

    if "pd" not in curve or "fppi" not in curve:
        raise KeyError("curve must contain 'pd' and 'fppi'")
    return false_alarm_at_fixed_pd(
        curve["pd"],
        curve["fppi"],
        target_pd,
        interpolate=interpolate,
        unreachable_value=unreachable_value,
    )


def risk_coverage_curve(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    from_logits: bool = False,
    ignore_index: int | None = -1,
) -> dict[str, Any]:
    """Compute a tie-invariant selective-classification risk--coverage curve.

    Candidates are retained from highest to lowest top-class confidence.
    ``risk`` is the classification error rate among retained candidates and
    ``coverage`` is their fraction of all labeled candidates.  All candidates
    sharing a confidence are admitted together, so the curve does not depend
    on arbitrary ordering inside ties.  ``aurc`` is the trapezoidal area under
    the reported curve, including the conventional zero-risk origin.
    """

    probabilities, label_array = _prepare_probabilities(
        predictions,
        labels,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    num_candidates = int(label_array.size)
    if num_candidates == 0:
        return {
            "thresholds": np.asarray([np.inf], dtype=np.float64),
            "coverage": np.asarray([0.0], dtype=np.float64),
            "risk": np.asarray([0.0], dtype=np.float64),
            "selected": np.asarray([0], dtype=np.int64),
            "errors": np.asarray([0], dtype=np.int64),
            "aurc": float("nan"),
            "num_candidates": 0,
        }

    predicted_labels = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(num_candidates), predicted_labels]
    mistakes = predicted_labels != label_array
    thresholds = np.unique(confidence)[::-1]
    selected_counts = np.empty(thresholds.size, dtype=np.int64)
    error_counts = np.empty(thresholds.size, dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        selected = confidence >= threshold
        selected_counts[index] = int(np.count_nonzero(selected))
        error_counts[index] = int(np.count_nonzero(mistakes & selected))

    coverage = selected_counts.astype(np.float64) / num_candidates
    risk = error_counts.astype(np.float64) / selected_counts
    thresholds = np.concatenate((np.asarray([np.inf]), thresholds))
    coverage = np.concatenate((np.asarray([0.0]), coverage))
    risk = np.concatenate((np.asarray([0.0]), risk))
    selected_counts = np.concatenate((np.asarray([0]), selected_counts))
    error_counts = np.concatenate((np.asarray([0]), error_counts))
    return {
        "thresholds": thresholds,
        "coverage": coverage,
        "risk": risk,
        "selected": selected_counts,
        "errors": error_counts,
        "aurc": float(np.sum(np.diff(coverage) * (risk[:-1] + risk[1:]) * 0.5)),
        "num_candidates": num_candidates,
    }


# Descriptive aliases make the module convenient in scripts while retaining
# the paper terminology used above.
expected_calibration_error = candidate_ece
compute_candidate_ece = candidate_ece
brier_score = candidate_brier_score
candidate_brier = candidate_brier_score
compute_candidate_brier = candidate_brier_score
negative_log_likelihood = candidate_nll
compute_candidate_nll = candidate_nll
compute_fppi = false_positives_per_image
fppi = false_positives_per_image
compute_froc = fppi_froc
froc_curve = fppi_froc
fa_at_fixed_pd = false_alarm_at_fixed_pd
fixed_pd_fa = false_alarm_at_fixed_pd
fa_at_pd = false_alarm_at_fixed_pd
selective_risk_coverage = risk_coverage_curve


__all__ = [
    "candidate_ece",
    "candidate_brier_score",
    "candidate_nll",
    "false_positives_per_image",
    "fppi_froc",
    "false_alarm_at_fixed_pd",
    "fppi_at_fixed_pd",
    "risk_coverage_curve",
    "expected_calibration_error",
    "compute_candidate_ece",
    "brier_score",
    "candidate_brier",
    "compute_candidate_brier",
    "negative_log_likelihood",
    "compute_candidate_nll",
    "compute_fppi",
    "fppi",
    "compute_froc",
    "froc_curve",
    "fa_at_fixed_pd",
    "fixed_pd_fa",
    "fa_at_pd",
    "selective_risk_coverage",
]
