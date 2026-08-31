#!/usr/bin/env python3
"""Audit whether baseline-missed targets have recoverable weak proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.audit_common import (
    add_frozen_audit_arguments,
    build_official_test_loader,
    load_frozen_coarse_model,
    select_device,
)
from utils.audit import (
    BinarySegmentationAccumulator,
    component_centroid,
    maximum_centroid_assignment,
)
from utils.candidate import generate_candidates, generate_recovery_candidates
from utils.detection_metric import maximum_centroid_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_frozen_audit_arguments(parser)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--proposal-thresholds",
        type=float,
        nargs="+",
        default=(0.2, 0.1, 0.05, 0.02),
    )
    parser.add_argument("--center-distance", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=1024)
    parser.add_argument("--local-max-kernel", type=int, default=5)
    parser.add_argument("--recovery-proposal-size", type=int, default=15)
    parser.add_argument("--max-recovery-candidates", type=int, default=32)
    parser.add_argument("--minimum-coverage-at-0-05", type=float, default=0.60)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("--probability-threshold must be in [0,1]")
    if not args.proposal_thresholds:
        raise ValueError("--proposal-thresholds cannot be empty")
    if len(set(args.proposal_thresholds)) != len(args.proposal_thresholds):
        raise ValueError("--proposal-thresholds cannot contain duplicates")
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.proposal_thresholds):
        raise ValueError("proposal thresholds must lie in [0,1]")
    if args.center_distance < 0:
        raise ValueError("--center-distance must be non-negative")
    if args.min_area < 1 or args.max_area < args.min_area:
        raise ValueError("candidate area bounds are invalid")
    if args.local_max_kernel < 1 or args.local_max_kernel % 2 == 0:
        raise ValueError("--local-max-kernel must be a positive odd integer")
    if args.recovery_proposal_size < 1 or args.recovery_proposal_size % 2 == 0:
        raise ValueError("--recovery-proposal-size must be a positive odd integer")
    if args.max_recovery_candidates < 1:
        raise ValueError("--max-recovery-candidates must be positive")
    if not 0.0 <= args.minimum_coverage_at_0_05 <= 1.0:
        raise ValueError("--minimum-coverage-at-0-05 must be in [0,1]")


def threshold_key(value: float) -> str:
    return f"{float(value):g}"


def resize_probabilities(
    multi_scale_logits: list[torch.Tensor], output_hw: tuple[int, int]
) -> torch.Tensor:
    """Return ``[L,H,W]`` sigmoid probabilities after logit resizing."""

    resized = []
    for logits in multi_scale_logits:
        if tuple(logits.shape[-2:]) != output_hw:
            logits = F.interpolate(
                logits, size=output_hw, mode="bilinear", align_corners=False
            )
        resized.append(logits.sigmoid()[0, 0])
    return torch.stack(resized, dim=0)


def center_window_max(
    response: torch.Tensor, centroid_yx: np.ndarray, window_size: int
) -> float:
    if response.ndim != 2:
        raise ValueError("response must have shape [H,W]")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    center_y = int(round(float(centroid_yx[0])))
    center_x = int(round(float(centroid_yx[1])))
    radius = window_size // 2
    y1, y2 = max(0, center_y - radius), min(response.shape[0], center_y + radius + 1)
    x1, x2 = max(0, center_x - radius), min(response.shape[1], center_x + radius + 1)
    return float(response[y1:y2, x1:x2].max().item())


def feature_response_map(feature: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    """Create a descriptive RMS activation map without claiming separability."""

    response = feature[0].square().mean(dim=0, keepdim=True).sqrt()[None]
    if tuple(response.shape[-2:]) != output_hw:
        response = F.interpolate(
            response, size=output_hw, mode="bilinear", align_corners=False
        )
    return response[0, 0]


def proposal_detail(
    gt_mask: np.ndarray,
    gt_centroid: np.ndarray,
    proposal_masks: list[np.ndarray],
    matched_proposal: int | None,
    *,
    proposal_centroids: list[np.ndarray] | None = None,
    proposal_attributes: dict[str, list] | None = None,
) -> dict:
    if not proposal_masks:
        return {
            "matched": False,
            "proposal_id": None,
            "centroid_distance": None,
            "closest_centroid_distance": None,
            "area": None,
            "iou": None,
            "gt_coverage": None,
        }
    centroids = proposal_centroids or [
        component_centroid(mask) for mask in proposal_masks
    ]
    distances = [float(np.linalg.norm(centroid - gt_centroid)) for centroid in centroids]
    detail = {
        "matched": matched_proposal is not None,
        "proposal_id": matched_proposal,
        "centroid_distance": (
            distances[matched_proposal] if matched_proposal is not None else None
        ),
        "closest_centroid_distance": min(distances),
        "area": None,
        "iou": None,
        "gt_coverage": None,
    }
    if matched_proposal is not None:
        proposal = proposal_masks[matched_proposal]
        intersection = int(np.count_nonzero(gt_mask & proposal))
        union = int(np.count_nonzero(gt_mask | proposal))
        detail.update(
            {
                "area": int(proposal.sum()),
                "iou": intersection / union if union else 0.0,
                "gt_coverage": intersection / int(gt_mask.sum()),
            }
        )
        if proposal_attributes:
            for name, values in proposal_attributes.items():
                detail[name] = values[matched_proposal]
    return detail


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    _, loader, protocol = build_official_test_loader(
        args.dataset_dir,
        base_size=args.base_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        device=device,
    )
    model, weight_metadata = load_frozen_coarse_model(args.weight_path, device)

    captured_features: dict[str, torch.Tensor] = {}
    handles = []
    for feature_name, module in (
        ("x_d0", model.decoder_0),
        ("x_d1", model.decoder_1),
        ("x_d2", model.decoder_2),
    ):
        def capture(_module, _inputs, output, name=feature_name):
            captured_features[name] = output

        handles.append(module.register_forward_hook(capture))

    metric = BinarySegmentationAccumulator(args.center_distance)
    records: list[dict] = []
    threshold_keys = [threshold_key(value) for value in args.proposal_thresholds]
    stream_names = ("legacy_mean_components", "recovery_max_local_peaks")
    exact_recoverable = {
        stream: {key: 0 for key in threshold_keys} for stream in stream_names
    }
    union_recoverable = {
        stream: {key: 0 for key in threshold_keys} for stream in stream_names
    }
    proposal_counts = {
        stream: {key: 0 for key in threshold_keys} for stream in stream_names
    }
    recovery_peaks_before_limit = {key: 0 for key in threshold_keys}

    try:
        with torch.inference_mode():
            for image_index, batch in enumerate(tqdm(loader, desc="missed-target-audit")):
                captured_features.clear()
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                outputs = model(images, warm_flag=True)
                coarse_logits = outputs["coarse_logits"]
                output_hw = tuple(coarse_logits.shape[-2:])
                coarse_probability = coarse_logits.sigmoid()[0, 0]
                target_array = (targets[0, 0] > 0).cpu().numpy()
                prediction_array = (
                    coarse_probability > args.probability_threshold
                ).cpu().numpy()
                snapshot = metric.update(prediction_array, target_array)
                gt_components = snapshot["gt_components"]
                missed_indices = snapshot["missed_gt_indices"]
                if not missed_indices:
                    continue

                scale_probabilities = resize_probabilities(
                    outputs["multi_scale_logits"], output_hw
                )
                mean_scale_probability = scale_probabilities.mean(dim=0)
                max_scale_probability = scale_probabilities.max(dim=0).values
                feature_responses = {
                    name: feature_response_map(captured_features[name], output_hw)
                    for name in ("x_d0", "x_d1", "x_d2")
                }

                proposal_data: dict[str, dict[str, dict]] = {
                    stream: {} for stream in stream_names
                }
                gt_centroids = [component_centroid(mask) for mask in gt_components]
                for threshold, key in zip(args.proposal_thresholds, threshold_keys):
                    legacy_candidates = generate_candidates(
                        coarse_logits,
                        outputs["multi_scale_logits"],
                        threshold_low=threshold,
                        min_area=args.min_area,
                        max_area=args.max_area,
                    )
                    legacy_arrays = [
                        mask.cpu().numpy().astype(bool)
                        for mask in legacy_candidates["masks"]
                    ]
                    legacy_assignment = maximum_centroid_assignment(
                        gt_components, legacy_arrays, args.center_distance
                    )
                    proposal_counts["legacy_mean_components"][key] += len(
                        legacy_arrays
                    )
                    proposal_data["legacy_mean_components"][key] = {
                        "masks": legacy_arrays,
                        "assignment": legacy_assignment,
                        "centroids": [component_centroid(mask) for mask in legacy_arrays],
                        "attributes": {
                            "proposal_score": legacy_candidates["scores"].cpu().tolist(),
                        },
                    }

                    recovery_candidates = generate_recovery_candidates(
                        coarse_logits,
                        outputs["multi_scale_logits"],
                        threshold_low=threshold,
                        threshold_high=args.probability_threshold,
                        local_max_kernel=args.local_max_kernel,
                        proposal_size=args.recovery_proposal_size,
                        max_candidates_per_image=args.max_recovery_candidates,
                    )
                    recovery_arrays = [
                        mask.cpu().numpy().astype(bool)
                        for mask in recovery_candidates["masks"]
                    ]
                    recovery_centroids = [
                        np.asarray(point, dtype=np.float64)
                        for point in recovery_candidates["peak_yx"].cpu().tolist()
                    ]
                    recovery_assignment = dict(
                        maximum_centroid_pairs(
                            gt_centroids,
                            recovery_centroids,
                            args.center_distance,
                        )
                    )
                    proposal_counts["recovery_max_local_peaks"][key] += len(
                        recovery_arrays
                    )
                    recovery_peaks_before_limit[key] += int(
                        recovery_candidates["num_peaks_before_limit"].sum().item()
                    )
                    proposal_data["recovery_max_local_peaks"][key] = {
                        "masks": recovery_arrays,
                        "assignment": recovery_assignment,
                        "centroids": recovery_centroids,
                        "attributes": {
                            "proposal_score": recovery_candidates[
                                "proposal_scores"
                            ].cpu().tolist(),
                            "peak_yx": recovery_candidates["peak_yx"].cpu().tolist(),
                            "source_scale": recovery_candidates[
                                "source_scale"
                            ].cpu().tolist(),
                        },
                    }

                for gt_index in missed_indices:
                    gt_mask = gt_components[gt_index]
                    gt_tensor = torch.as_tensor(gt_mask, device=device)
                    centroid = component_centroid(gt_mask)
                    stream_proposals = {stream: {} for stream in stream_names}
                    recovered_keys = {stream: [] for stream in stream_names}
                    for stream in stream_names:
                        for key in threshold_keys:
                            data = proposal_data[stream][key]
                            proposal_id = data["assignment"].get(gt_index)
                            stream_proposals[stream][key] = proposal_detail(
                                gt_mask,
                                centroid,
                                data["masks"],
                                proposal_id,
                                proposal_centroids=data["centroids"],
                                proposal_attributes=data["attributes"],
                            )
                            if proposal_id is not None:
                                recovered_keys[stream].append(key)

                    response_windows = {}
                    for window_size in (3, 5, 9):
                        response_windows[str(window_size)] = {
                            "coarse": center_window_max(
                                coarse_probability, centroid, window_size
                            ),
                            "mean_scale": center_window_max(
                                mean_scale_probability, centroid, window_size
                            ),
                            "max_scale": center_window_max(
                                max_scale_probability, centroid, window_size
                            ),
                        }
                    feature_statistics = {}
                    for name, response in feature_responses.items():
                        image_mean = float(response.mean().item())
                        image_std = float(response.std(unbiased=False).item())
                        gt_peak = float(response[gt_tensor].max().item())
                        feature_statistics[name] = {
                            "gt_peak_rms_activation": gt_peak,
                            "gt_mean_rms_activation": float(
                                response[gt_tensor].mean().item()
                            ),
                            "image_mean_rms_activation": image_mean,
                            "image_std_rms_activation": image_std,
                            "gt_peak_zscore": (
                                (gt_peak - image_mean) / image_std
                                if image_std > 0
                                else None
                            ),
                        }
                    record = {
                        "image": batch["name"][0],
                        "image_index": image_index,
                        "gt_id": gt_index,
                        "gt_area": int(gt_mask.sum()),
                        "gt_centroid_yx": [float(value) for value in centroid],
                        "coarse_peak": float(coarse_probability[gt_tensor].max().item()),
                        "scale_peaks": [
                            float(scale[gt_tensor].max().item())
                            for scale in scale_probabilities
                        ],
                        "mean_scale_peak": float(
                            mean_scale_probability[gt_tensor].max().item()
                        ),
                        "max_scale_peak": float(
                            max_scale_probability[gt_tensor].max().item()
                        ),
                        "center_window_peaks": response_windows,
                        "feature_statistics": feature_statistics,
                        "legacy_mean_proposals": stream_proposals[
                            "legacy_mean_components"
                        ],
                        "proposals": stream_proposals[
                            "recovery_max_local_peaks"
                        ],
                        "recoverable_at_0.05": stream_proposals[
                            "recovery_max_local_peaks"
                        ].get("0.05", {}).get("matched", False),
                        "recoverable_any": bool(
                            recovered_keys["recovery_max_local_peaks"]
                        ),
                    }
                    records.append(record)
                    for stream in stream_names:
                        for key in threshold_keys:
                            if stream_proposals[stream][key]["matched"]:
                                exact_recoverable[stream][key] += 1

        # Recompute cumulative unions in threshold order from immutable records.
        for stream in stream_names:
            recovered_objects: set[tuple[int, int]] = set()
            record_key = (
                "legacy_mean_proposals"
                if stream == "legacy_mean_components"
                else "proposals"
            )
            for key in threshold_keys:
                for record in records:
                    if record[record_key][key]["matched"]:
                        recovered_objects.add(
                            (record["image_index"], record["gt_id"])
                        )
                union_recoverable[stream][key] = len(recovered_objects)
    finally:
        for handle in handles:
            handle.remove()

    baseline = metric.get()
    num_missed = len(records)
    coverage = {
        stream: {
            key: {
                "recoverable_missed_targets": exact_recoverable[stream][key],
                "coverage_fn": (
                    exact_recoverable[stream][key] / num_missed
                    if num_missed
                    else None
                ),
                "cumulative_recoverable_missed_targets": union_recoverable[
                    stream
                ][key],
                "cumulative_coverage_fn": (
                    union_recoverable[stream][key] / num_missed
                    if num_missed
                    else None
                ),
                "kept_proposals_on_images_with_missed_targets": proposal_counts[
                    stream
                ][key],
                **(
                    {"local_maxima_before_top_k": recovery_peaks_before_limit[key]}
                    if stream == "recovery_max_local_peaks"
                    else {}
                ),
            }
            for key in threshold_keys
        }
        for stream in stream_names
    }
    coverage_at_005 = coverage["recovery_max_local_peaks"].get("0.05", {}).get(
        "coverage_fn"
    )
    complete_test = protocol["evaluated_images"] == protocol["manifests"]["test"][
        "num_images"
    ]
    go_signal = bool(
        complete_test
        and coverage_at_005 is not None
        and coverage_at_005 >= args.minimum_coverage_at_0_05
    )
    recoverable_any = sum(record["recoverable_any"] for record in records)
    legacy_recoverable_any = sum(
        any(detail["matched"] for detail in record["legacy_mean_proposals"].values())
        for record in records
    )
    summary = {
        "schema_version": "mshnet-ccrr-missed-target-audit/v2",
        "protocol": {
            **protocol,
            "probability_threshold": args.probability_threshold,
            "proposal_thresholds": args.proposal_thresholds,
            "proposal_streams": {
                "legacy_mean_components": {
                    "aggregation": "mean(sigmoid(resized multi-scale logits))",
                    "support": "8-connected threshold components",
                    "area": [args.min_area, args.max_area],
                },
                "recovery_max_local_peaks": {
                    "aggregation": "max(sigmoid(coarse), sigmoid(resized multi-scale logits))",
                    "eligibility": (
                        "max evidence > proposal_threshold and "
                        f"coarse probability < {args.probability_threshold:g}"
                    ),
                    "local_max_kernel": args.local_max_kernel,
                    "proposal_size": args.recovery_proposal_size,
                    "max_candidates_per_image": args.max_recovery_candidates,
                    "gt_injection": False,
                },
            },
            "proposal_threshold_comparison": "strictly_greater_than",
            "connectivity": 8,
            "center_distance": args.center_distance,
            "matching": "all GT vs all proposals, maximum-cardinality one-to-one centroid matching",
            "feature_statistics": "descriptive RMS activations captured from actual decoder forward hooks; not a separability claim",
            "weight": weight_metadata,
        },
        "baseline": baseline,
        "missed_targets": {
            "count": num_missed,
            "legacy_mean_recoverable_at_any_requested_threshold": legacy_recoverable_any,
            "recoverable_at_any_requested_threshold": recoverable_any,
            "unrecoverable_at_all_requested_thresholds": num_missed
            - recoverable_any,
        },
        "coverage": coverage,
        "decision_gate": {
            "threshold": 0.05,
            "minimum_coverage": args.minimum_coverage_at_0_05,
            "observed_coverage": coverage_at_005,
            "complete_official_test": complete_test,
            "go_to_recovery_branch": go_signal,
        },
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "missed_targets.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
