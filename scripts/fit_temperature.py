#!/usr/bin/env python3
"""Fit scalar temperature scaling for baseline candidate confidences.

The input is the ``*_candidates.json`` file written by
``scripts/diagnose_baseline.py``.  Only candidates explicitly labelled
``target`` or ``clutter`` are used; uncertain and incomplete records are never
silently converted to negatives.

Temperature scaling is a strictly monotone score transformation.  It can
improve ECE, Brier score and NLL, but it cannot improve ranking metrics (for
example ROC/AP/FROC) or the best operating point obtained by sweeping a
threshold.  The generated report records this limitation explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.reliability_metric import (
    candidate_brier_score,
    candidate_ece,
    candidate_nll,
)


TARGET_LABEL = 0
CLUTTER_LABEL = 1
EXPLICIT_LABEL_NAMES = {"target": TARGET_LABEL, "clutter": CLUTTER_LABEL}
ALL_LABEL_NAMES = {**EXPLICIT_LABEL_NAMES, "uncertain": 2}


@dataclass(frozen=True)
class CandidateSamples:
    """Flat explicit target/clutter samples loaded from a candidate bank."""

    scores: np.ndarray
    labels: np.ndarray
    image_names: np.ndarray

    def __len__(self) -> int:
        return int(self.scores.size)

    def subset(self, selector: np.ndarray) -> "CandidateSamples":
        selector = np.asarray(selector)
        return CandidateSamples(
            scores=self.scores[selector],
            labels=self.labels[selector],
            image_names=self.image_names[selector],
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-json",
        "--input-json",
        dest="candidate_json",
        required=True,
        help="Candidate JSON to evaluate (and split for fitting if --val-json is absent).",
    )
    parser.add_argument(
        "--val-json",
        "--validation-json",
        dest="val_json",
        default="",
        help="Optional independent validation candidate JSON used only to fit T.",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        default="",
        help="Report path (default: <candidate-json-dir>/temperature_calibration.json).",
    )
    parser.add_argument("--score-key", default="score")
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="Fraction of image names used to fit T when --val-json is absent.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--min-temperature", type=float, default=1e-2)
    parser.add_argument("--max-temperature", type=float, default=1e2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args(argv)


def load_candidate_json(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load either the diagnosis mapping or a bare candidate-record list."""

    candidate_path = Path(path)
    try:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"candidate JSON not found: {candidate_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid candidate JSON {candidate_path}: {error}") from error

    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        records = payload.get("candidates")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"metadata in {candidate_path} must be an object")
        metadata = raw_metadata
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            f"{candidate_path} must contain a candidate list or an object with 'candidates'"
        )
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"candidates in {candidate_path} must be a list of objects")
    return records, metadata


def _explicit_label(record: dict[str, Any], record_index: int) -> int | None:
    """Read an explicit label and reject contradictory name/id pairs."""

    label_from_name: int | None = None
    raw_name = record.get("label")
    if isinstance(raw_name, str):
        label_from_name = ALL_LABEL_NAMES.get(raw_name.strip().lower())

    label_from_id: int | None = None
    raw_id = record.get("label_id")
    if isinstance(raw_id, bool):
        raw_id = None
    if isinstance(raw_id, (int, float)) and math.isfinite(float(raw_id)):
        integer_id = int(raw_id)
        if float(integer_id) == float(raw_id):
            label_from_id = integer_id

    if (
        label_from_name is not None
        and label_from_id is not None
        and label_from_name != label_from_id
    ):
        raise ValueError(
            f"candidate record {record_index} has contradictory label and label_id"
        )
    resolved = label_from_name if label_from_name is not None else label_from_id
    return resolved if resolved in {TARGET_LABEL, CLUTTER_LABEL} else None


def extract_explicit_samples(
    records: Sequence[dict[str, Any]],
    *,
    score_key: str = "score",
    require_image_name: bool = True,
) -> CandidateSamples:
    """Extract only explicit target/clutter records from diagnosis output."""

    scores: list[float] = []
    labels: list[int] = []
    image_names: list[str] = []
    for index, record in enumerate(records):
        label = _explicit_label(record, index)
        if label is None:
            continue

        if score_key not in record:
            raise ValueError(
                f"explicit candidate record {index} is missing score key {score_key!r}"
            )
        raw_score = record[score_key]
        if isinstance(raw_score, bool):
            raise ValueError(f"candidate score at record {index} must be numeric")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"candidate score at record {index} must be numeric"
            ) from error
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"candidate score at record {index} must be finite and lie in [0, 1]"
            )

        raw_image_name = record.get("name")
        if require_image_name and (
            not isinstance(raw_image_name, str) or not raw_image_name.strip()
        ):
            raise ValueError(f"explicit candidate record {index} has no image name")
        image_name = str(raw_image_name) if raw_image_name is not None else ""
        scores.append(score)
        labels.append(label)
        image_names.append(image_name)

    return CandidateSamples(
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        image_names=np.asarray(image_names, dtype=str),
    )


def deterministic_image_split(
    samples: CandidateSamples,
    calibration_fraction: float = 0.5,
    *,
    seed: int = 0,
) -> tuple[CandidateSamples, CandidateSamples, list[str], list[str]]:
    """Split by stable SHA-256 ordering so no image leaks between partitions."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    unique_names = sorted(set(samples.image_names.tolist()))
    if len(unique_names) < 2:
        raise ValueError(
            "deterministic image split requires at least two image names; "
            "provide --val-json for an independent calibration set"
        )

    def stable_key(name: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest()
        return digest, name

    ordered_names = sorted(unique_names, key=stable_key)
    calibration_count = int(round(len(ordered_names) * calibration_fraction))
    calibration_count = min(max(calibration_count, 1), len(ordered_names) - 1)
    calibration_names = ordered_names[:calibration_count]
    evaluation_names = ordered_names[calibration_count:]
    calibration_set = set(calibration_names)
    calibration_mask = np.asarray(
        [name in calibration_set for name in samples.image_names], dtype=bool
    )
    return (
        samples.subset(calibration_mask),
        samples.subset(~calibration_mask),
        calibration_names,
        evaluation_names,
    )


def _candidate_logits(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def _binary_nll_from_logits(logits: np.ndarray, target: np.ndarray) -> float:
    values = np.maximum(logits, 0.0) - logits * target + np.log1p(
        np.exp(-np.abs(logits))
    )
    return float(values.mean())


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_temperature: float = 1e-2,
    max_temperature: float = 1e2,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """Fit the globally NLL-optimal bounded scalar temperature.

    Binary NLL is convex in inverse temperature ``a=1/T``.  Its derivative is
    monotone, so a bounded bisection finds the global optimum deterministically
    without adding a SciPy dependency.
    """

    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if score_array.shape != label_array.shape or score_array.size == 0:
        raise ValueError("scores and labels must be non-empty arrays of equal length")
    if not np.all(np.isfinite(score_array)) or np.any(score_array < 0) or np.any(score_array > 1):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    if not np.all(np.isin(label_array, (TARGET_LABEL, CLUTTER_LABEL))):
        raise ValueError("temperature fitting accepts only target/clutter labels 0/1")
    if np.unique(label_array).size < 2:
        raise ValueError("temperature fitting requires both target and clutter samples")
    if (
        not math.isfinite(min_temperature)
        or not math.isfinite(max_temperature)
        or min_temperature <= 0
        or max_temperature <= min_temperature
    ):
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    logits = _candidate_logits(score_array)
    target = (label_array == TARGET_LABEL).astype(np.float64)
    if np.max(np.abs(logits)) <= np.finfo(np.float64).eps:
        return 1.0

    def derivative(inverse_temperature: float) -> float:
        scaled = inverse_temperature * logits
        probability = np.empty_like(scaled)
        nonnegative = scaled >= 0
        probability[nonnegative] = 1.0 / (1.0 + np.exp(-scaled[nonnegative]))
        exponentiated = np.exp(scaled[~nonnegative])
        probability[~nonnegative] = exponentiated / (1.0 + exponentiated)
        return float(np.mean(logits * (probability - target)))

    lower = 1.0 / max_temperature
    upper = 1.0 / min_temperature
    lower_gradient = derivative(lower)
    upper_gradient = derivative(upper)
    if lower_gradient >= 0:
        inverse_temperature = lower
    elif upper_gradient <= 0:
        inverse_temperature = upper
    else:
        for _ in range(int(max_iter)):
            midpoint = (lower + upper) / 2.0
            gradient = derivative(midpoint)
            if abs(gradient) <= tolerance or upper - lower <= tolerance:
                lower = upper = midpoint
                break
            if gradient < 0:
                lower = midpoint
            else:
                upper = midpoint
        inverse_temperature = (lower + upper) / 2.0

    fitted = 1.0 / inverse_temperature
    # The bounded optimum should never be worse, but retaining T=1 on a tiny
    # floating-point regression makes this guarantee explicit.
    baseline_nll = _binary_nll_from_logits(logits, target)
    fitted_nll = _binary_nll_from_logits(logits / fitted, target)
    if fitted_nll > baseline_nll:
        return 1.0
    return float(fitted)


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Apply monotone scalar temperature scaling to target probabilities."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    score_array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(score_array)) or np.any(score_array < 0) or np.any(score_array > 1):
        raise ValueError("scores must be finite probabilities in [0, 1]")

    # Candidate banks store probabilities rather than the original logits.
    # Use the same finite endpoint convention as fitting so scores rounded to
    # exactly zero/one can still be softened by T > 1.
    logits = _candidate_logits(score_array)
    scaled = logits / temperature
    probabilities = np.empty_like(scaled)
    nonnegative = scaled >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scaled[nonnegative]))
    exponentiated = np.exp(scaled[~nonnegative])
    probabilities[~nonnegative] = exponentiated / (1.0 + exponentiated)
    return probabilities


def calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    probabilities = np.column_stack((scores, 1.0 - scores))
    return {
        "ECE": float(
            candidate_ece(probabilities, labels, n_bins=n_bins, target_class=TARGET_LABEL)
        ),
        "Brier": float(candidate_brier_score(probabilities, labels)),
        "NLL": float(candidate_nll(probabilities, labels)),
    }


def _sample_summary(samples: CandidateSamples) -> dict[str, Any]:
    return {
        "num_candidates": len(samples),
        "num_images": len(set(samples.image_names.tolist())),
        "class_counts": {
            "target": int(np.count_nonzero(samples.labels == TARGET_LABEL)),
            "clutter": int(np.count_nonzero(samples.labels == CLUTTER_LABEL)),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidate_json).resolve()
    records, _ = load_candidate_json(candidate_path)
    all_evaluation_samples = extract_explicit_samples(records, score_key=args.score_key)
    if len(all_evaluation_samples) == 0:
        raise ValueError(f"{candidate_path} contains no explicit target/clutter candidates")

    if args.val_json:
        validation_path = Path(args.val_json).resolve()
        validation_records, _ = load_candidate_json(validation_path)
        calibration_samples = extract_explicit_samples(
            validation_records, score_key=args.score_key
        )
        evaluation_samples = all_evaluation_samples
        split = {
            "mode": "independent_validation_file",
            "calibration_path": str(validation_path),
            "evaluation_path": str(candidate_path),
            "seed": None,
            "calibration_fraction": None,
        }
    else:
        (
            calibration_samples,
            evaluation_samples,
            calibration_names,
            evaluation_names,
        ) = deterministic_image_split(
            all_evaluation_samples,
            args.calibration_fraction,
            seed=args.seed,
        )
        split = {
            "mode": "deterministic_image_name_split",
            "source_path": str(candidate_path),
            "seed": int(args.seed),
            "calibration_fraction": float(args.calibration_fraction),
            "calibration_image_names": calibration_names,
            "evaluation_image_names": evaluation_names,
            "image_overlap": [],
        }

    if len(calibration_samples) == 0:
        raise ValueError("calibration partition contains no explicit target/clutter candidates")
    if len(evaluation_samples) == 0:
        raise ValueError("evaluation partition contains no explicit target/clutter candidates")

    temperature = fit_temperature(
        calibration_samples.scores,
        calibration_samples.labels,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )
    calibrated_fit_scores = apply_temperature(calibration_samples.scores, temperature)
    calibrated_evaluation_scores = apply_temperature(evaluation_samples.scores, temperature)

    fit_before = calibration_metrics(
        calibration_samples.scores, calibration_samples.labels, n_bins=args.n_bins
    )
    fit_after = calibration_metrics(
        calibrated_fit_scores, calibration_samples.labels, n_bins=args.n_bins
    )
    evaluation_before = calibration_metrics(
        evaluation_samples.scores, evaluation_samples.labels, n_bins=args.n_bins
    )
    evaluation_after = calibration_metrics(
        calibrated_evaluation_scores, evaluation_samples.labels, n_bins=args.n_bins
    )
    rank_before = np.argsort(-evaluation_samples.scores, kind="stable")
    rank_after = np.argsort(-calibrated_evaluation_scores, kind="stable")

    return {
        "temperature": temperature,
        "score_key": args.score_key,
        "n_bins": int(args.n_bins),
        "split": split,
        "calibration_fit": {
            **_sample_summary(calibration_samples),
            "before": fit_before,
            "after": fit_after,
            "delta_after_minus_before": {
                key: fit_after[key] - fit_before[key] for key in fit_before
            },
        },
        "evaluation": {
            **_sample_summary(evaluation_samples),
            "before": evaluation_before,
            "after": evaluation_after,
            "delta_after_minus_before": {
                key: evaluation_after[key] - evaluation_before[key]
                for key in evaluation_before
            },
        },
        "detection_metric_invariance": {
            "ranking_preserved": bool(np.array_equal(rank_before, rank_after)),
            "can_improve_ranking_or_threshold_sweep_metrics": False,
            "unchanged_metric_families": [
                "ROC/AUC",
                "precision-recall/AP",
                "FROC/FPPI-vs-Pd curve",
                "best metric selected by sweeping the score threshold",
            ],
            "explanation": (
                "Positive scalar temperature scaling is monotone: it changes "
                "confidence calibration but not candidate ordering. Therefore it "
                "cannot improve ranking metrics or the attainable operating points "
                "from a threshold sweep. A fixed numeric threshold may select a "
                "different point, but the same point is obtainable by remapping the "
                "threshold on the original scores."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_bins <= 0:
        raise ValueError("n_bins must be positive")
    report = build_report(args)
    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.candidate_json).resolve().parent / "temperature_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"wrote calibration report: {output_path}", file=sys.stderr)
    print(
        "NOTE: temperature scaling cannot improve ranking/FROC or the best "
        "threshold-swept detection operating point; it only calibrates confidence.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
