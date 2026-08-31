#!/usr/bin/env python3
"""Sweep hard clutter-action thresholds on a frozen CCRR-V1 checkpoint."""

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
from tqdm import tqdm

from model.MSHNet import MSHNet
from model.ccrr import ThresholdAwareClutterSuppressor
from scripts.audit_common import (
    add_frozen_audit_arguments,
    build_official_test_loader,
    select_device,
    state_dict_from_checkpoint,
)
from scripts.audit_fp_upper_bound import oracle_target_presence
from utils.audit import BinarySegmentationAccumulator, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_frozen_audit_arguments(parser)
    parser.add_argument(
        "--action-thresholds",
        type=float,
        nargs="+",
        default=(0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    )
    parser.add_argument("--remove-threshold", type=float, default=0.45)
    parser.add_argument(
        "--max-suppression",
        type=float,
        default=0.0,
        help="Zero means exact/unbounded threshold-aware suppression.",
    )
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--minimum-action-precision", type=float, default=0.85)
    parser.add_argument("--minimum-removal-rate", type=float, default=0.80)
    parser.add_argument("--minimum-fa-reduction", type=float, default=0.10)
    parser.add_argument("--maximum-pd-damage", type=float, default=0.002)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.action_thresholds:
        raise ValueError("--action-thresholds cannot be empty")
    if len(set(args.action_thresholds)) != len(args.action_thresholds):
        raise ValueError("--action-thresholds cannot contain duplicates")
    if any(not 0.5 < value < 1.0 for value in args.action_thresholds):
        raise ValueError("action thresholds must lie in (0.5,1)")
    if not 0.0 < args.remove_threshold < 0.5:
        raise ValueError("--remove-threshold must lie in (0,0.5)")
    if args.max_suppression < 0:
        raise ValueError("--max-suppression must be non-negative")
    if args.soft_temperature <= 0:
        raise ValueError("--soft-temperature must be positive")
    for name in (
        "minimum_action_precision",
        "minimum_removal_rate",
        "minimum_fa_reduction",
        "maximum_pd_damage",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1]")


def load_v1_model(
    path_value: str | Path, device: torch.device, base_size: int
) -> tuple[MSHNet, dict, dict]:
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
    }
    for key, expected in required.items():
        if inference.get(key) != expected:
            raise RuntimeError(
                f"checkpoint inference_config.{key}={inference.get(key)!r}, expected {expected!r}"
            )
    if evaluation.get("probability_threshold") != 0.5:
        raise RuntimeError("checkpoint probability threshold is not the fixed 0.5")
    if evaluation.get("candidate_source") != "online":
        raise RuntimeError("the V1 action sweep requires online candidates")

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
        "rectifier": "suppression_only",
    }
    model = MSHNet(3, ccrr_config=ccrr_config).to(device)
    state = state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
    }
    return model, metadata, checkpoint


def safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


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
    candidate_threshold = float(inference["candidate_threshold"])
    min_area = int(inference["min_candidate_area"])
    max_area = int(inference["max_candidate_area"])
    center_distance = float(evaluation["center_distance"])
    positive_iou = float(evaluation["positive_iou"])

    suppression_cap = args.max_suppression or None
    rectifiers = {}
    state = {}
    for threshold in args.action_thresholds:
        key = f"{threshold:g}"
        rectifier = ThresholdAwareClutterSuppressor(
            action_threshold=threshold,
            remove_threshold=args.remove_threshold,
            soft_temperature=args.soft_temperature,
            max_suppression=suppression_cap,
            output_threshold=0.5,
        ).to(device)
        rectifier.eval()
        rectifiers[key] = rectifier
        state[key] = {
            "metric": BinarySegmentationAccumulator(center_distance),
            "selected_candidates": 0,
            "selected_target_candidates": 0,
            "selected_clutter_candidates": 0,
            "active_selected_candidates": 0,
            "crossed_candidates": 0,
            "acted_fp_components": 0,
            "removed_acted_fp_components": 0,
            "required_deltas": [],
            "actual_deltas": [],
        }

    coarse_metric = BinarySegmentationAccumulator(center_distance)
    action_records: list[dict] = []
    total_candidates = 0
    total_target_candidates = 0
    total_clutter_candidates = 0

    with torch.inference_mode():
        for image_index, batch in enumerate(tqdm(loader, desc="action-threshold-sweep")):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            outputs = model(
                images,
                warm_flag=True,
                enable_ccrr=True,
                candidate_threshold=candidate_threshold,
                min_candidate_area=min_area,
                max_candidate_area=max_area,
            )
            coarse_logits = outputs["coarse_logits"]
            candidate_outputs = outputs["candidate_outputs"]
            candidate_masks = candidate_outputs["candidate_masks"]
            batch_indices = candidate_outputs["batch_indices"]
            target_scores = candidate_outputs["target_scores"]
            clutter_scores = candidate_outputs["clutter_scores"]
            generated_candidates = {
                "masks": candidate_masks,
                "batch_indices": batch_indices,
                "coarse_peak_scores": candidate_outputs["coarse_peak_scores"],
            }
            target_presence = oracle_target_presence(
                generated_candidates,
                targets,
                positive_iou=positive_iou,
                center_distance=center_distance,
            )
            clutter_presence = ~target_presence
            total_candidates += int(candidate_masks.shape[0])
            total_target_candidates += int(target_presence.sum().item())
            total_clutter_candidates += int(clutter_presence.sum().item())

            target_array = (targets[0, 0] > 0).cpu().numpy()
            coarse_prediction = (coarse_logits.sigmoid()[0, 0] > 0.5).cpu().numpy()
            coarse_snapshot = coarse_metric.update(coarse_prediction, target_array)
            fp_components = [
                coarse_snapshot["prediction_components"][index]
                for index in coarse_snapshot["false_positive_indices"]
            ]

            for key, rectifier in rectifiers.items():
                result = rectifier(
                    coarse_logits,
                    target_scores,
                    clutter_scores,
                    candidate_masks,
                    batch_indices,
                )
                selected = result["gates"] > 0.5
                active = selected & (result["peak_logits"] > 0)
                refined_logits = result["refined_logits"]
                refined_prediction = (
                    refined_logits.sigmoid()[0, 0] > 0.5
                ).cpu().numpy()
                state[key]["metric"].update(refined_prediction, target_array)
                state[key]["selected_candidates"] += int(selected.sum().item())
                state[key]["selected_target_candidates"] += int(
                    (selected & target_presence).sum().item()
                )
                state[key]["selected_clutter_candidates"] += int(
                    (selected & clutter_presence).sum().item()
                )
                state[key]["active_selected_candidates"] += int(active.sum().item())

                selected_union = np.zeros_like(target_array)
                for candidate_index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                    mask = candidate_masks[candidate_index].cpu().numpy().astype(bool)
                    selected_union |= mask
                    peak_after = float(
                        refined_logits[0, 0][candidate_masks[candidate_index]].max().item()
                    )
                    peak_before = float(result["peak_logits"][candidate_index].item())
                    crossed = peak_before > 0 and peak_after <= 0
                    state[key]["crossed_candidates"] += int(crossed)
                    required = float(
                        result["unclipped_required_deltas"][candidate_index].item()
                    )
                    actual = float(result["deltas"][candidate_index].item())
                    state[key]["required_deltas"].append(required)
                    state[key]["actual_deltas"].append(actual)
                    overlapped_fp = [
                        fp_index
                        for fp_index, fp_mask in enumerate(fp_components)
                        if np.any(fp_mask & mask)
                    ]
                    eliminated = bool(overlapped_fp) and all(
                        not np.any(fp_components[fp_index] & refined_prediction)
                        for fp_index in overlapped_fp
                    )
                    action_records.append(
                        {
                            "image": batch["name"][0],
                            "image_index": image_index,
                            "action_threshold": float(key),
                            "candidate_id": candidate_index,
                            "target_probability": float(target_scores[candidate_index].item()),
                            "clutter_probability": float(clutter_scores[candidate_index].item()),
                            "is_target": bool(target_presence[candidate_index].item()),
                            "is_clutter": bool(clutter_presence[candidate_index].item()),
                            "peak_before_logit": peak_before,
                            "peak_after_logit": peak_after,
                            "required_delta": required,
                            "actual_delta": actual,
                            "crossed_output_threshold": crossed,
                            "overlapped_coarse_fp_components": overlapped_fp,
                            "overlapped_fp_components_eliminated": eliminated,
                        }
                    )
                for fp_mask in fp_components:
                    if np.any(fp_mask & selected_union):
                        state[key]["acted_fp_components"] += 1
                        if not np.any(fp_mask & refined_prediction):
                            state[key]["removed_acted_fp_components"] += 1

    coarse = coarse_metric.get()
    complete_test = protocol["evaluated_images"] == protocol["manifests"]["test"][
        "num_images"
    ]
    sweep = []
    for key in (f"{value:g}" for value in args.action_thresholds):
        item = state[key]
        refined = item["metric"].get()
        selected_count = item["selected_candidates"]
        action_precision = safe_ratio(
            item["selected_clutter_candidates"], selected_count
        )
        removal_rate = safe_ratio(
            item["removed_acted_fp_components"], item["acted_fp_components"]
        )
        relative_fa_reduction = safe_ratio(
            float(coarse["Fa_per_million_pixels"])
            - float(refined["Fa_per_million_pixels"]),
            float(coarse["Fa_per_million_pixels"]),
        )
        pd_damage = float(coarse["Pd"]) - float(refined["Pd"])
        gates = {
            "action_precision_pass": action_precision
            >= args.minimum_action_precision,
            "target_deletion_pass": item["selected_target_candidates"] == 0
            and pd_damage <= args.maximum_pd_damage,
            "removal_rate_pass": removal_rate >= args.minimum_removal_rate,
            "fa_reduction_pass": relative_fa_reduction
            >= args.minimum_fa_reduction,
        }
        gates["go_to_short_run"] = bool(complete_test and all(gates.values()))
        sweep.append(
            {
                "action_threshold": float(key),
                "selected_candidates": selected_count,
                "selected_target_candidates": item["selected_target_candidates"],
                "selected_clutter_candidates": item["selected_clutter_candidates"],
                "active_selected_candidates": item["active_selected_candidates"],
                "crossed_output_threshold_candidates": item["crossed_candidates"],
                "action_precision": action_precision,
                "acted_fp_components": item["acted_fp_components"],
                "removed_acted_fp_components": item[
                    "removed_acted_fp_components"
                ],
                "removal_rate": removal_rate,
                "refined": refined,
                "relative_fa_reduction": relative_fa_reduction,
                "pd_damage": pd_damage,
                "decision_gates": gates,
            }
        )

    passing = [item for item in sweep if item["decision_gates"]["go_to_short_run"]]
    recommended = None
    if passing:
        recommended = min(
            passing,
            key=lambda item: (
                item["refined"]["Fa_per_million_pixels"],
                -item["refined"]["mIoU"],
                -item["action_threshold"],
            ),
        )["action_threshold"]
    summary = {
        "schema_version": "mshnet-ccrr-action-threshold-sweep/v1",
        "protocol": {
            **protocol,
            "test_guided_development_result": True,
            "probability_threshold": 0.5,
            "candidate_threshold": candidate_threshold,
            "candidate_area": [min_area, max_area],
            "candidate_source": "online",
            "center_distance": center_distance,
            "positive_iou": positive_iou,
            "rectifier": "threshold_aware_uniform_active_support",
            "gate_mode": "hard_inference",
            "remove_threshold": args.remove_threshold,
            "max_suppression": suppression_cap,
            "weight": weight_metadata,
        },
        "candidate_counts": {
            "total": total_candidates,
            "oracle_target": total_target_candidates,
            "oracle_clutter": total_clutter_candidates,
        },
        "coarse": coarse,
        "decision_thresholds": {
            "minimum_action_precision": args.minimum_action_precision,
            "minimum_removal_rate": args.minimum_removal_rate,
            "minimum_relative_fa_reduction": args.minimum_fa_reduction,
            "maximum_pd_damage": args.maximum_pd_damage,
        },
        "sweep": sweep,
        "recommended_test_guided_action_threshold": recommended,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "actions.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in action_records
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
