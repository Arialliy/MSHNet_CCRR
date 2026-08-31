#!/usr/bin/env python3
"""Audit FP proposal coverage and GT-oracle suppression upper bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from scripts.audit_common import (
    add_frozen_audit_arguments,
    build_official_test_loader,
    load_frozen_coarse_model,
    select_device,
)
from utils.audit import BinarySegmentationAccumulator, component_masks
from utils.candidate import generate_candidates, match_candidates_to_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_frozen_audit_arguments(parser)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-threshold", type=float, default=0.2)
    parser.add_argument("--remove-threshold", type=float, default=0.45)
    parser.add_argument("--positive-iou", type=float, default=0.3)
    parser.add_argument("--center-distance", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=1024)
    parser.add_argument("--minimum-fp-coverage", type=float, default=0.90)
    parser.add_argument("--minimum-fa-reduction", type=float, default=0.10)
    parser.add_argument("--maximum-pd-damage", type=float, default=0.002)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "probability_threshold",
        "candidate_threshold",
        "remove_threshold",
        "positive_iou",
        "minimum_fp_coverage",
        "minimum_fa_reduction",
        "maximum_pd_damage",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.remove_threshold >= args.probability_threshold:
        raise ValueError("--remove-threshold must be below --probability-threshold")
    if args.center_distance < 0:
        raise ValueError("--center-distance must be non-negative")
    if args.min_area < 1 or args.max_area < args.min_area:
        raise ValueError("candidate area bounds are invalid")


def oracle_target_presence(
    candidates: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    positive_iou: float,
    center_distance: float,
) -> torch.Tensor:
    """Protect every proposal with GT overlap or a valid center match."""

    label_candidates = dict(candidates)
    label_candidates["scores"] = candidates["coarse_peak_scores"]
    matching = match_candidates_to_gt(
        label_candidates,
        targets,
        positive_iou=positive_iou,
        hard_negative_threshold=0.0,
        center_distance=center_distance,
    )
    return (matching["max_iou"] > 0) | matching["center_match"]


def threshold_aware_suppression(
    coarse_logits: torch.Tensor,
    candidate_masks: torch.Tensor,
    batch_indices: torch.Tensor,
    suppress_candidates: torch.Tensor,
    remove_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp the union of selected proposal masks below the decision point."""

    if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
        raise ValueError("coarse_logits must have shape [B,1,H,W]")
    oracle_mask = torch.zeros_like(coarse_logits[:, 0], dtype=torch.bool)
    selected = torch.nonzero(suppress_candidates, as_tuple=False).flatten()
    for candidate_index in selected.tolist():
        batch_index = int(batch_indices[candidate_index].item())
        oracle_mask[batch_index] |= candidate_masks[candidate_index].bool()
    ceiling = torch.logit(
        coarse_logits.new_tensor(float(remove_threshold)), eps=1e-6
    )
    clamped = torch.minimum(coarse_logits, torch.full_like(coarse_logits, ceiling))
    refined = torch.where(oracle_mask[:, None], clamped, coarse_logits)
    if torch.any(refined > coarse_logits):
        raise AssertionError("oracle suppression must never increase a logit")
    if not torch.equal(refined[~oracle_mask[:, None]], coarse_logits[~oracle_mask[:, None]]):
        raise AssertionError("oracle suppression changed pixels outside proposals")
    return refined, oracle_mask


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "num_above_v1_limit_1_5": int(np.count_nonzero(array > 1.5)),
        "num_above_guide_limit_6_0": int(np.count_nonzero(array > 6.0)),
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
    model, weight_metadata = load_frozen_coarse_model(args.weight_path, device)

    coarse_metric = BinarySegmentationAccumulator(args.center_distance)
    oracle_metric = BinarySegmentationAccumulator(args.center_distance)
    fp_records: list[dict] = []
    candidate_count = 0
    target_candidate_count = 0
    clutter_candidate_count = 0
    all_candidate_covered = 0
    oracle_candidate_covered = 0
    fully_oracle_covered = 0
    protected_fp_components = 0
    total_fp_pixels = 0
    oracle_covered_fp_pixels = 0
    original_fully_removed = 0
    original_partially_reduced = 0
    original_unchanged = 0
    post_split_fragments = 0
    required_suppression: list[float] = []
    ceiling_logit = math.log(
        args.remove_threshold / (1.0 - args.remove_threshold)
    )

    with torch.inference_mode():
        for image_index, batch in enumerate(tqdm(loader, desc="fp-upper-bound")):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, warm_flag=True)
            coarse_logits = outputs["coarse_logits"]
            candidates = generate_candidates(
                coarse_logits,
                outputs["multi_scale_logits"],
                threshold_low=args.candidate_threshold,
                min_area=args.min_area,
                max_area=args.max_area,
            )
            target_presence = oracle_target_presence(
                candidates,
                targets,
                positive_iou=args.positive_iou,
                center_distance=args.center_distance,
            )
            clutter_candidates = ~target_presence
            oracle_logits, oracle_mask_tensor = threshold_aware_suppression(
                coarse_logits,
                candidates["masks"],
                candidates["batch_indices"],
                clutter_candidates,
                args.remove_threshold,
            )
            target_bool_tensor = targets[:, 0] > 0
            if torch.any(oracle_mask_tensor & target_bool_tensor):
                raise AssertionError("an oracle-clutter proposal overlaps GT pixels")

            probability = coarse_logits.sigmoid()[0, 0]
            oracle_probability = oracle_logits.sigmoid()[0, 0]
            target_array = target_bool_tensor[0].cpu().numpy()
            coarse_prediction = (
                probability > args.probability_threshold
            ).cpu().numpy()
            oracle_prediction = (
                oracle_probability > args.probability_threshold
            ).cpu().numpy()
            coarse_snapshot = coarse_metric.update(coarse_prediction, target_array)
            oracle_metric.update(oracle_prediction, target_array)

            masks = candidates["masks"].cpu().numpy().astype(bool)
            oracle_candidate_flags = clutter_candidates.cpu().numpy().astype(bool)
            all_union = np.any(masks, axis=0) if masks.size else np.zeros_like(target_array)
            oracle_union = oracle_mask_tensor[0].cpu().numpy()
            local_required: list[float] = []
            for candidate_index in np.flatnonzero(oracle_candidate_flags):
                mask = candidates["masks"][candidate_index]
                if not torch.any(mask):
                    continue
                peak_logit = float(coarse_logits[0, 0][mask].max().item())
                local_required.append(max(0.0, peak_logit - ceiling_logit))
            required_suppression.extend(local_required)

            candidate_count += int(masks.shape[0])
            target_candidate_count += int(target_presence.sum().item())
            clutter_candidate_count += int(clutter_candidates.sum().item())
            for fp_index in coarse_snapshot["false_positive_indices"]:
                fp_mask = coarse_snapshot["prediction_components"][fp_index]
                fp_area = int(fp_mask.sum())
                total_fp_pixels += fp_area
                all_overlap = masks[:, fp_mask].any(axis=1) if masks.size else np.zeros(0, bool)
                oracle_overlap = all_overlap & oracle_candidate_flags
                any_coverage = bool(all_overlap.any())
                oracle_coverage = bool(oracle_overlap.any())
                full_coverage = bool(np.all(oracle_union[fp_mask]))
                covered_pixels = int(np.count_nonzero(fp_mask & oracle_union))
                all_candidate_covered += int(any_coverage)
                oracle_candidate_covered += int(oracle_coverage)
                fully_oracle_covered += int(full_coverage)
                oracle_covered_fp_pixels += covered_pixels
                protected = any_coverage and not oracle_coverage
                protected_fp_components += int(protected)

                remaining_mask = fp_mask & oracle_prediction
                remaining_pixels = int(remaining_mask.sum())
                fragments = len(component_masks(remaining_mask))
                post_split_fragments += fragments
                if remaining_pixels == 0:
                    original_fully_removed += 1
                    transition = "removed"
                elif remaining_pixels < fp_area:
                    original_partially_reduced += 1
                    transition = "partially_reduced"
                else:
                    original_unchanged += 1
                    transition = "unchanged"
                fp_records.append(
                    {
                        "image": batch["name"][0],
                        "image_index": image_index,
                        "fp_component_id": fp_index,
                        "area": fp_area,
                        "candidate_indices": np.flatnonzero(all_overlap).tolist(),
                        "oracle_clutter_candidate_indices": np.flatnonzero(
                            oracle_overlap
                        ).tolist(),
                        "any_candidate_coverage": any_coverage,
                        "oracle_clutter_coverage": oracle_coverage,
                        "fully_oracle_covered": full_coverage,
                        "oracle_pixel_coverage": safe_ratio(covered_pixels, fp_area),
                        "protected_by_target_presence": protected,
                        "remaining_pixels": remaining_pixels,
                        "post_suppression_fragments": fragments,
                        "transition": transition,
                    }
                )

    coarse = coarse_metric.get()
    oracle = oracle_metric.get()
    num_fp = int(coarse["false_positive_components"])
    complete_test = protocol["evaluated_images"] == protocol["manifests"]["test"][
        "num_images"
    ]
    fa_reduction = safe_ratio(
        float(coarse["Fa_per_million_pixels"])
        - float(oracle["Fa_per_million_pixels"]),
        float(coarse["Fa_per_million_pixels"]),
    )
    pd_damage = float(coarse["Pd"]) - float(oracle["Pd"])
    coverage_rate = safe_ratio(all_candidate_covered, num_fp)
    oracle_elimination_requirement = min(20, num_fp)
    gates = {
        "complete_official_test": complete_test,
        "candidate_coverage_pass": coverage_rate >= args.minimum_fp_coverage,
        "oracle_elimination_pass": original_fully_removed
        >= oracle_elimination_requirement,
        "fa_reduction_pass": fa_reduction >= args.minimum_fa_reduction,
        "pd_safety_pass": pd_damage <= args.maximum_pd_damage,
    }
    gates["go_to_threshold_aware_rectifier"] = bool(
        complete_test
        and gates["candidate_coverage_pass"]
        and gates["oracle_elimination_pass"]
        and gates["fa_reduction_pass"]
        and gates["pd_safety_pass"]
    )
    summary = {
        "schema_version": "mshnet-ccrr-fp-upper-bound/v1",
        "protocol": {
            **protocol,
            "probability_threshold": args.probability_threshold,
            "candidate_threshold": args.candidate_threshold,
            "remove_threshold": args.remove_threshold,
            "proposal_aggregation": "mean(sigmoid(resized multi-scale logits))",
            "candidate_area": [args.min_area, args.max_area],
            "connectivity": 8,
            "center_distance": args.center_distance,
            "positive_iou": args.positive_iou,
            "oracle_rule": "protect any GT-overlap or valid-center candidate; suppress all others",
            "weight": weight_metadata,
        },
        "counts": {
            "candidates": candidate_count,
            "oracle_target_candidates": target_candidate_count,
            "oracle_clutter_candidates": clutter_candidate_count,
        },
        "coverage": {
            "coarse_fp_components": num_fp,
            "any_candidate_covered_components": all_candidate_covered,
            "any_candidate_component_rate": coverage_rate,
            "oracle_clutter_covered_components": oracle_candidate_covered,
            "oracle_clutter_component_rate": safe_ratio(
                oracle_candidate_covered, num_fp
            ),
            "fully_oracle_covered_components": fully_oracle_covered,
            "fully_oracle_covered_rate": safe_ratio(fully_oracle_covered, num_fp),
            "oracle_fp_pixel_coverage": safe_ratio(
                oracle_covered_fp_pixels, total_fp_pixels
            ),
            "protected_fp_components": protected_fp_components,
            "uncovered_fp_components": num_fp - all_candidate_covered,
        },
        "required_logit_suppression_to_0_45": distribution(required_suppression),
        "coarse": coarse,
        "oracle_adaptive": {
            **oracle,
            "fp_components_reduced_net": num_fp
            - int(oracle["false_positive_components"]),
            "fp_pixels_eliminated_net": int(coarse["false_alarm_pixels"])
            - int(oracle["false_alarm_pixels"]),
            "original_fp_components_fully_removed": original_fully_removed,
            "original_fp_components_partially_reduced": original_partially_reduced,
            "original_fp_components_unchanged": original_unchanged,
            "post_split_fragments_inside_original_fp_support": post_split_fragments,
            "relative_fa_reduction": fa_reduction,
            "pd_damage": pd_damage,
        },
        "decision_gates": {
            "minimum_candidate_coverage": args.minimum_fp_coverage,
            "minimum_oracle_eliminated_components": oracle_elimination_requirement,
            "minimum_relative_fa_reduction": args.minimum_fa_reduction,
            "maximum_pd_damage": args.maximum_pd_damage,
            **gates,
        },
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fp_components.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in fp_records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
