#!/usr/bin/env python3
"""Sweep a frozen CCRR head on exact 0.5 output components.

The low-threshold multi-scale proposal is used only to obtain the frozen
classifier score. Every possible action is one complete 8-connected component
of ``sigmoid(coarse_logits) > 0.5``. Suppressing a selected mask removes that
whole component, exactly matching unbounded suppression to 0.45 at the fixed
0.5 evaluation point.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from model.MSHNet import MSHNet
from scripts.audit_common import (
    add_frozen_audit_arguments,
    build_official_test_loader,
    select_device,
    state_dict_from_checkpoint,
)
from utils.audit import BinarySegmentationAccumulator, file_sha256
from utils.candidate import generate_component_aligned_candidates
from utils.detection_metric import match_prediction_components_to_gt


DEFAULT_REPORT_THRESHOLDS = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_frozen_audit_arguments(parser)
    parser.add_argument(
        "--action-thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_REPORT_THRESHOLDS,
        help="Extra reporting thresholds. Unique active scores are swept too.",
    )
    parser.add_argument(
        "--fixed-thresholds-only",
        action="store_true",
        help="Do not add every unique active component score.",
    )
    parser.add_argument("--remove-threshold", type=float, default=0.45)
    parser.add_argument("--minimum-action-precision", type=float, default=0.80)
    parser.add_argument("--minimum-eliminated-fp", type=int, default=2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.action_thresholds)) != len(args.action_thresholds):
        raise ValueError("--action-thresholds cannot contain duplicates")
    if any(not 0.5 < value <= 1.0 for value in args.action_thresholds):
        raise ValueError("action thresholds must lie in (0.5,1]")
    if not 0.0 < args.remove_threshold < 0.5:
        raise ValueError("--remove-threshold must lie in (0,0.5)")
    if not 0.0 <= args.minimum_action_precision <= 1.0:
        raise ValueError("--minimum-action-precision must lie in [0,1]")
    if args.minimum_eliminated_fp < 1:
        raise ValueError("--minimum-eliminated-fp must be positive")


def load_v1_model(
    path_value: str | Path, device: torch.device, base_size: int
) -> tuple[MSHNet, dict[str, Any], dict[str, Any]]:
    path = Path(path_value).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("the action sweep requires a structured CCRR checkpoint")
    inference = checkpoint.get("inference_config")
    evaluation = checkpoint.get("evaluation_config")
    if not isinstance(inference, dict) or not isinstance(evaluation, dict):
        raise RuntimeError("CCRR checkpoint lacks inference/evaluation metadata")
    required = {
        "enable_ccrr": True,
        "base_size": base_size,
        "proposal_aggregation": "mean_sigmoid_multiscale",
        "component_connectivity": 8,
    }
    for key, expected in required.items():
        if inference.get(key) != expected:
            raise RuntimeError(
                f"checkpoint inference_config.{key}={inference.get(key)!r}, "
                f"expected {expected!r}"
            )
    if evaluation.get("probability_threshold") != 0.5:
        raise RuntimeError("checkpoint probability threshold is not the fixed 0.5")
    if evaluation.get("candidate_source") != "online":
        raise RuntimeError("the action sweep requires online candidates")
    if inference.get("output_probability_threshold", 0.5) != 0.5:
        raise RuntimeError("checkpoint action output threshold is not 0.5")

    ccrr_config = {
        "num_scales": int(inference["num_scales"]),
        "roi_size": int(inference["roi_size"]),
        "hidden_dim": int(inference["hidden_dim"]),
        "context_scale": float(inference["context_scale"]),
        "min_context_size": float(inference["min_context_size"]),
        "dropout": float(inference["ccrr_dropout"]),
        "max_delta": float(inference["max_delta"]),
        "gate_margin": float(inference["gate_margin"]),
        "gate_temperature": float(inference["gate_temperature"]),
        "num_classes": int(inference["ccrr_num_classes"]),
        # This audit consumes only the frozen encoder/head scores.
        "rectifier": "suppression_only",
    }
    model = MSHNet(3, ccrr_config=ccrr_config).to(device)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
    }
    return model, metadata, checkpoint


def safe_relative_reduction(baseline: float, refined: float) -> float:
    return float((baseline - refined) / baseline) if baseline else 0.0


def json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_thresholds(
    score_arrays: list[np.ndarray],
    report_thresholds: list[float] | tuple[float, ...],
    *,
    include_score_boundaries: bool,
) -> list[float]:
    thresholds = {1.0, *(float(value) for value in report_thresholds)}
    if include_score_boundaries:
        for scores in score_arrays:
            thresholds.update(
                float(value)
                for value in scores
                if 0.5 < float(value) < 1.0
            )
    return sorted(thresholds, reverse=True)


def score_candidates(
    model: MSHNet,
    feature_map: torch.Tensor,
    coarse_logits: torch.Tensor,
    multi_scale_logits: list[torch.Tensor],
    boxes: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.ccrr is None:
        raise RuntimeError("the checkpoint model has no CCRR module")
    _, outputs = model.ccrr(
        feature_map=feature_map,
        coarse_logits=coarse_logits,
        multi_scale_logits=multi_scale_logits,
        candidate_boxes=boxes,
        candidate_masks=masks,
    )
    return outputs


def evaluate_threshold(
    image_records: list[dict[str, Any]],
    threshold: float,
    *,
    center_distance: float,
    coarse_metrics: dict[str, float | int],
    complete_test: bool,
    minimum_action_precision: float,
    minimum_eliminated_fp: int,
) -> dict[str, Any]:
    metric = BinarySegmentationAccumulator(center_distance)
    actions = 0
    actions_on_tp = 0
    actions_on_fp = 0
    target_deletions = 0
    relabelled_or_new_fp = 0

    for record in image_records:
        selected = record["clutter_scores"] >= threshold
        selected_count = int(selected.sum())
        selected_tp = int(record["is_tp_component"][selected].sum())
        selected_fp = int(record["is_fp_component"][selected].sum())
        actions += selected_count
        actions_on_tp += selected_tp
        actions_on_fp += selected_fp

        refined = record["coarse_prediction"].copy()
        if selected_count:
            refined[np.any(record["action_masks"][selected], axis=0)] = False
        refined_snapshot = metric.update(refined, record["target"])
        refined_detected_gt = set(refined_snapshot["detected_gt_ids"])
        target_deletions += len(record["coarse_detected_gt"] - refined_detected_gt)
        expected_surviving_fp = record["coarse_fp_components"] - selected_fp
        relabelled_or_new_fp += max(
            0,
            int(refined_snapshot["false_positive_components"])
            - expected_surviving_fp,
        )

    refined_metrics = metric.get()
    action_precision = actions_on_fp / actions if actions else None
    removal_efficiency = 1.0 if actions_on_fp else None
    net_fp_reduction = int(coarse_metrics["false_positive_components"]) - int(
        refined_metrics["false_positive_components"]
    )
    relative_fppi_reduction = safe_relative_reduction(
        float(coarse_metrics["FPPI"]), float(refined_metrics["FPPI"])
    )
    relative_fa_reduction = safe_relative_reduction(
        float(coarse_metrics["Fa_per_million_pixels"]),
        float(refined_metrics["Fa_per_million_pixels"]),
    )
    pd_damage = float(coarse_metrics["Pd"]) - float(refined_metrics["Pd"])
    miou_damage = float(coarse_metrics["mIoU"]) - float(refined_metrics["mIoU"])
    niou_damage = float(coarse_metrics["nIoU"]) - float(refined_metrics["nIoU"])

    rescue_gates = {
        "complete_official_test": complete_test,
        "zero_target_deletion": target_deletions == 0,
        "minimum_eliminated_fp": actions_on_fp >= minimum_eliminated_fp,
        "minimum_action_precision": (
            action_precision is not None
            and action_precision >= minimum_action_precision
        ),
    }
    rescue_gates["go"] = bool(all(rescue_gates.values()))
    tolerance = 1e-12
    final_sca_gates = {
        "zero_target_deletion": target_deletions == 0,
        "action_precision_at_least_0_90": (
            action_precision is not None and action_precision >= 0.90
        ),
        "fp_removal_efficiency_at_least_0_90": (
            removal_efficiency is not None and removal_efficiency >= 0.90
        ),
        "eliminated_fp_at_least_4": actions_on_fp >= 4,
        "fppi_relative_reduction_at_least_0_15": relative_fppi_reduction >= 0.15,
        "fa_relative_reduction_at_least_0_10": relative_fa_reduction >= 0.10,
        "pd_not_below_baseline": pd_damage <= tolerance,
        "miou_not_below_baseline": miou_damage <= tolerance,
        "niou_not_below_baseline": niou_damage <= tolerance,
    }
    final_sca_gates["pass"] = bool(complete_test and all(final_sca_gates.values()))
    return {
        "action_threshold": float(threshold),
        "actions": actions,
        "actions_on_tp_components": actions_on_tp,
        "actions_on_fp_components": actions_on_fp,
        "action_precision": action_precision,
        "target_deletions": target_deletions,
        "unique_fp_components_acted": actions_on_fp,
        "original_fp_components_eliminated": actions_on_fp,
        "net_fp_components_reduced": net_fp_reduction,
        "fp_removal_efficiency": removal_efficiency,
        "new_or_relabelled_fp_components": relabelled_or_new_fp,
        "refined": refined_metrics,
        "relative_fppi_reduction": relative_fppi_reduction,
        "relative_fa_reduction": relative_fa_reduction,
        "pd_damage": pd_damage,
        "miou_damage": miou_damage,
        "niou_damage": niou_damage,
        "rescue_gate": rescue_gates,
        "final_sca_gate": final_sca_gates,
    }


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
    model, weight_metadata, checkpoint = load_v1_model(
        args.weight_path, device, args.base_size
    )
    inference = checkpoint["inference_config"]
    evaluation = checkpoint["evaluation_config"]
    proposal_threshold = float(inference["candidate_threshold"])
    min_area = int(inference["min_candidate_area"])
    raw_max_area = inference["max_candidate_area"]
    max_area = None if raw_max_area is None else int(raw_max_area)
    center_distance = float(evaluation["center_distance"])

    coarse_metric = BinarySegmentationAccumulator(center_distance)
    image_records: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    inactive_records: list[dict[str, Any]] = []
    total_raw_proposals = 0
    total_inactive_proposals = 0
    total_fallback_actions = 0

    with torch.inference_mode():
        for image_index, batch in enumerate(
            tqdm(loader, desc="exact-component-score-audit")
        ):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, warm_flag=True, enable_ccrr=False)
            coarse_logits = outputs["coarse_logits"]
            aligned = generate_component_aligned_candidates(
                coarse_logits=coarse_logits,
                multi_scale_logits=outputs["multi_scale_logits"],
                proposal_threshold=proposal_threshold,
                output_threshold=0.5,
                min_area=min_area,
                max_area=max_area,
            )
            aligned_scores = score_candidates(
                model,
                outputs["feature"],
                coarse_logits,
                outputs["multi_scale_logits"],
                aligned["proposal_boxes"],
                aligned["proposal_masks"],
            )
            raw_scores = score_candidates(
                model,
                outputs["feature"],
                coarse_logits,
                outputs["multi_scale_logits"],
                aligned["raw_proposal_boxes"],
                aligned["raw_proposal_masks"],
            )

            target_array = (targets[0, 0] > 0).cpu().numpy()
            coarse_prediction = (coarse_logits[0, 0] > 0).cpu().numpy()
            action_masks = aligned["action_masks"].cpu().numpy().astype(bool)
            if action_masks.shape[0]:
                if not np.array_equal(action_masks.any(axis=0), coarse_prediction):
                    raise RuntimeError("action masks do not partition the 0.5 output")
                if np.any(action_masks.astype(np.uint8).sum(axis=0) > 1):
                    raise RuntimeError("action masks overlap")
            elif coarse_prediction.any():
                raise RuntimeError("non-empty output has no action component")

            component_match = match_prediction_components_to_gt(
                action_masks,
                target_array,
                center_distance=center_distance,
            )
            coarse_snapshot = coarse_metric.update(coarse_prediction, target_array)
            evaluation_labels = np.asarray(
                [
                    index in coarse_snapshot["matched_prediction_ids"]
                    for index in range(action_masks.shape[0])
                ],
                dtype=bool,
            )
            if not np.array_equal(component_match.is_tp_component, evaluation_labels):
                raise RuntimeError("action labels diverge from evaluation matching")

            clutter_scores = aligned_scores["clutter_scores"].cpu().numpy().astype(np.float64)
            target_scores = aligned_scores["target_scores"].cpu().numpy().astype(np.float64)
            image_name = batch["name"][0]
            fallback = aligned["proposal_is_fallback"].cpu().numpy().astype(bool)
            total_fallback_actions += int(fallback.sum())
            total_raw_proposals += int(aligned["raw_proposal_masks"].shape[0])
            inactive = (~aligned["raw_proposal_has_action_overlap"]).cpu().numpy()
            total_inactive_proposals += int(inactive.sum())

            for component_id in range(action_masks.shape[0]):
                matched_gt = int(component_match.prediction_to_gt[component_id])
                action_records.append(
                    {
                        "image": image_name,
                        "image_index": image_index,
                        "action_component_id": int(
                            aligned["action_component_local_ids"][component_id].item()
                        ),
                        "proposal_id": int(
                            aligned["proposal_component_ids"][component_id].item()
                        ),
                        "proposal_is_fallback": bool(fallback[component_id]),
                        "proposal_to_component_iou": float(
                            aligned["proposal_to_action_iou"][component_id].item()
                        ),
                        "component_label": (
                            "target"
                            if component_match.is_tp_component[component_id]
                            else "clutter"
                        ),
                        "matched_gt_id": matched_gt if matched_gt >= 0 else None,
                        "centroid_distance": json_number(
                            component_match.matched_centroid_distance[component_id]
                        ),
                        "component_iou": json_number(
                            component_match.matched_component_iou[component_id]
                        ),
                        "ambiguous_keep": bool(
                            component_match.ambiguous_keep[component_id]
                        ),
                        "ambiguity_reasons": list(
                            component_match.ambiguity_reasons[component_id]
                        ),
                        "target_probability": float(target_scores[component_id]),
                        "clutter_probability": float(clutter_scores[component_id]),
                        "coarse_peak_probability": float(
                            aligned["coarse_peak_scores"][component_id].item()
                        ),
                        "coarse_mean_probability": float(
                            aligned["coarse_mean_scores"][component_id].item()
                        ),
                        "action_area": int(aligned["action_areas"][component_id].item()),
                        "proposal_area": int(
                            aligned["proposal_areas"][component_id].item()
                        ),
                    }
                )

            raw_clutter = raw_scores["clutter_scores"].cpu().numpy()
            raw_target = raw_scores["target_scores"].cpu().numpy()
            for proposal_id in np.flatnonzero(inactive):
                inactive_records.append(
                    {
                        "image": image_name,
                        "image_index": image_index,
                        "proposal_id": int(proposal_id),
                        "clutter_probability": float(raw_clutter[proposal_id]),
                        "target_probability": float(raw_target[proposal_id]),
                        "coarse_detected": False,
                        "actionable_at_output_threshold": False,
                    }
                )

            image_records.append(
                {
                    "target": target_array,
                    "coarse_prediction": coarse_prediction,
                    "action_masks": action_masks,
                    "clutter_scores": clutter_scores,
                    "is_tp_component": component_match.is_tp_component.copy(),
                    "is_fp_component": component_match.is_fp_component.copy(),
                    "coarse_detected_gt": set(coarse_snapshot["detected_gt_ids"]),
                    "coarse_fp_components": int(
                        coarse_snapshot["false_positive_components"]
                    ),
                }
            )

    coarse = coarse_metric.get()
    complete_test = protocol["evaluated_images"] == protocol["manifests"]["test"][
        "num_images"
    ]
    thresholds = build_thresholds(
        [record["clutter_scores"] for record in image_records],
        args.action_thresholds,
        include_score_boundaries=not args.fixed_thresholds_only,
    )
    sweep = [
        evaluate_threshold(
            image_records,
            threshold,
            center_distance=center_distance,
            coarse_metrics=coarse,
            complete_test=complete_test,
            minimum_action_precision=args.minimum_action_precision,
            minimum_eliminated_fp=args.minimum_eliminated_fp,
        )
        for threshold in tqdm(thresholds, desc="threshold-operating-points")
    ]

    passing = [item for item in sweep if item["rescue_gate"]["go"]]
    recommended = None
    if passing:
        recommended = min(
            passing,
            key=lambda item: (
                item["refined"]["Fa_per_million_pixels"],
                item["refined"]["FPPI"],
                -item["action_threshold"],
            ),
        )["action_threshold"]
    safe_points = [item for item in sweep if item["target_deletions"] == 0]
    best_safe = max(
        safe_points,
        key=lambda item: (
            item["original_fp_components_eliminated"],
            item["action_precision"] if item["action_precision"] is not None else -1.0,
            item["action_threshold"],
        ),
    )
    ranked_components = sorted(
        action_records, key=lambda item: item["clutter_probability"], reverse=True
    )
    ranked_inactive = sorted(
        inactive_records, key=lambda item: item["clutter_probability"], reverse=True
    )
    mapped_keys = {
        (item["image_index"], item["proposal_id"])
        for item in action_records
        if item["proposal_id"] >= 0
    }
    candidate_counts = {
        "exact_output_components": len(action_records),
        "target_components": sum(
            item["component_label"] == "target" for item in action_records
        ),
        "false_positive_components": sum(
            item["component_label"] == "clutter" for item in action_records
        ),
        "mapped_components": len(action_records) - total_fallback_actions,
        "fallback_components": total_fallback_actions,
        "raw_proposals": total_raw_proposals,
        "inactive_raw_proposals": total_inactive_proposals,
        "duplicate_proposal_mappings": (
            len(action_records) - total_fallback_actions - len(mapped_keys)
        ),
    }
    no_go_reason = None
    if not passing:
        if best_safe["original_fp_components_eliminated"] == 0:
            no_go_reason = (
                "No threshold eliminates an FP component with zero target deletion; "
                "the frozen head requires quality-veto/selective-risk retraining."
            )
        else:
            no_go_reason = (
                "No zero-target-deletion operating point meets the minimum eliminated-FP "
                "and action-precision gates; retrain the quality-veto/selective-risk head."
            )

    summary = {
        "schema_version": "mshnet-ccrr-exact-component-sweep/v2",
        "protocol": {
            **protocol,
            "test_guided_development_result": True,
            "independent_validation_split": False,
            "probability_threshold": 0.5,
            "threshold_operator": ">",
            "component_connectivity": 8,
            "center_distance": center_distance,
            "distance_operator": "<=",
            "matching": "deterministic_maximum_cardinality_centroid_v1",
            "proposal_threshold": proposal_threshold,
            "proposal_area": [min_area, max_area],
            "proposal_use": "feature_scoring_only",
            "action_unit": "exact_complete_coarse_probability_gt_0.5_component",
            "action_mask_use": "label_suppression_attribution_and_metrics",
            "fallback_proposal": "one_pixel_eight_neighborhood_dilation",
            "frozen_score": "legacy_target_clutter_head_on_mapped_feature_proposal",
            "gate_mode": "hard_clutter_probability_gte_threshold",
            "remove_threshold": args.remove_threshold,
            "suppression_equivalence": "whole-component deletion at fixed 0.5",
            "weight": weight_metadata,
        },
        "candidate_counts": candidate_counts,
        "coarse": coarse,
        "rescue_decision_thresholds": {
            "zero_target_deletion": True,
            "minimum_eliminated_fp": args.minimum_eliminated_fp,
            "minimum_action_precision": args.minimum_action_precision,
        },
        "rescue_decision": "GO" if passing else "NO_GO",
        "no_go_reason": no_go_reason,
        "recommended_test_guided_action_threshold": recommended,
        "best_zero_target_deletion_point": best_safe,
        "highest_scored_exact_components": ranked_components[:20],
        "highest_scored_inactive_proposals": ranked_inactive[:20],
        "num_thresholds_evaluated": len(sweep),
        "sweep": sweep,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "actions.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in action_records
        ),
        encoding="utf-8",
    )
    (output_dir / "inactive_proposals.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in inactive_records
        ),
        encoding="utf-8",
    )
    with (output_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "action_threshold", "actions", "actions_on_tp_components",
            "actions_on_fp_components", "action_precision", "target_deletions",
            "original_fp_components_eliminated", "Pd", "FPPI",
            "Fa_per_million_pixels", "mIoU", "nIoU", "rescue_go",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in sweep:
            writer.writerow(
                {
                    "action_threshold": item["action_threshold"],
                    "actions": item["actions"],
                    "actions_on_tp_components": item["actions_on_tp_components"],
                    "actions_on_fp_components": item["actions_on_fp_components"],
                    "action_precision": item["action_precision"],
                    "target_deletions": item["target_deletions"],
                    "original_fp_components_eliminated": item[
                        "original_fp_components_eliminated"
                    ],
                    "Pd": item["refined"]["Pd"],
                    "FPPI": item["refined"]["FPPI"],
                    "Fa_per_million_pixels": item["refined"][
                        "Fa_per_million_pixels"
                    ],
                    "mIoU": item["refined"]["mIoU"],
                    "nIoU": item["refined"]["nIoU"],
                    "rescue_go": item["rescue_gate"]["go"],
                }
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_counts": candidate_counts,
                "coarse": coarse,
                "rescue_decision": summary["rescue_decision"],
                "no_go_reason": no_go_reason,
                "recommended_test_guided_action_threshold": recommended,
                "best_zero_target_deletion_point": best_safe,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
