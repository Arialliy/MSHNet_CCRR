#!/usr/bin/env python3
"""Diagnose high-confidence candidate-level false alarms from a frozen MSHNet."""

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
from PIL import Image
from skimage.measure import label as connected_components
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.MSHNet import MSHNet
from utils.candidate import (
    CLUTTER_LABEL,
    LABEL_NAMES,
    TARGET_LABEL,
    UNCERTAIN_LABEL,
    generate_candidates,
    match_candidates_to_gt,
)
from utils.data import IRSTD_Dataset
from utils.reliability_metric import (
    candidate_brier_score,
    candidate_ece,
    candidate_nll,
    fppi_at_fixed_pd,
    fppi_froc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--candidate-threshold", type=float, default=0.2)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-score",
        choices=("coarse_peak", "coarse_mean", "scale_peak", "scale_mean"),
        default="coarse_peak",
        help="Candidate confidence used for hard-negative labels and FROC.",
    )
    parser.add_argument("--positive-iou", type=float, default=0.3)
    parser.add_argument("--center-distance", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=1024)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-crops", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def encode_mask_rle(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask.reshape(-1))
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, boundaries)
    return [[int(group[0]), int(group.size)] for group in groups]


def one_to_one_tp_flags(
    labels: np.ndarray,
    matched_gt: np.ndarray,
    batch_indices: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    flags = np.zeros(labels.shape[0], dtype=bool)
    for batch_index in np.unique(batch_indices):
        image_candidates = np.flatnonzero(
            (batch_indices == batch_index) & (labels == TARGET_LABEL) & (matched_gt >= 0)
        )
        for gt_index in np.unique(matched_gt[image_candidates]):
            matches = image_candidates[matched_gt[image_candidates] == gt_index]
            if matches.size:
                flags[matches[np.argmax(scores[matches])]] = True
    return flags


def finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None


def save_false_positive_crops(
    records: list[dict], dataset: IRSTD_Dataset, output_dir: Path, limit: int
) -> None:
    if limit <= 0:
        return
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    false_positives = [record for record in records if record["label"] == "clutter"]
    false_positives.sort(key=lambda record: record["score"], reverse=True)
    for rank, record in enumerate(false_positives[:limit]):
        image_path = dataset._resolve_image_path(dataset.imgs_dir, record["name"])
        image = Image.open(image_path).convert("RGB").resize(
            (dataset.base_size, dataset.base_size), Image.BILINEAR
        )
        _, x1, y1, x2, y2 = record["box"]
        pad = max(4, int(max(x2 - x1, y2 - y1)))
        crop_box = (
            max(0, int(x1) - pad),
            max(0, int(y1) - pad),
            min(dataset.base_size, int(x2) + pad),
            min(dataset.base_size, int(y2) + pad),
        )
        image.crop(crop_box).save(
            crop_dir / f"{rank:03d}_{record['name']}_{record['candidate_index']}.png"
        )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset_args = argparse.Namespace(
        dataset_dir=args.dataset_dir,
        base_size=args.base_size,
        crop_size=args.base_size,
    )
    dataset = IRSTD_Dataset(dataset_args, mode="test", split=args.split)
    evaluated_dataset = dataset
    if args.max_images > 0:
        evaluated_dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        evaluated_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MSHNet(3).to(device)
    checkpoint = torch.load(args.weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()

    records: list[dict] = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_matched_gt: list[int] = []
    all_batch_indices: list[int] = []
    total_gt_targets = 0

    with torch.inference_mode():
        for image_index, batch in enumerate(tqdm(loader, desc="candidate-diagnosis")):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, warm_flag=True)
            candidates = generate_candidates(
                coarse_logits=outputs["coarse_logits"],
                multi_scale_logits=outputs["multi_scale_logits"],
                threshold_low=args.candidate_threshold,
                min_area=args.min_area,
                max_area=args.max_area,
            )
            score_tensors = {
                "coarse_peak": candidates["coarse_peak_scores"],
                "coarse_mean": candidates["coarse_scores"],
                "scale_peak": candidates["peak_scores"],
                "scale_mean": candidates["scores"],
            }
            label_candidates = dict(candidates)
            label_candidates["scores"] = score_tensors[args.candidate_score]
            matching = match_candidates_to_gt(
                label_candidates,
                masks,
                positive_iou=args.positive_iou,
                hard_negative_threshold=args.hard_negative_threshold,
                center_distance=args.center_distance,
            )

            gt_array = masks[0, 0].detach().cpu().numpy() > 0
            total_gt_targets += int(connected_components(gt_array, connectivity=2).max())
            labels = matching["labels"].cpu().numpy()
            matched_gt = matching["matched_gt_indices"].cpu().numpy()
            scores = score_tensors[args.candidate_score].cpu().numpy()
            scale_mean_scores = candidates["scores"].cpu().numpy()
            boxes = candidates["boxes"].cpu().numpy()
            candidate_masks = candidates["masks"].cpu().numpy()
            batch_indices = np.full(labels.shape[0], image_index, dtype=np.int64)

            offset = len(all_scores)
            for local_index in range(labels.shape[0]):
                records.append(
                    {
                        "name": batch["name"][0],
                        "image_index": image_index,
                        "candidate_index": local_index,
                        "global_candidate_index": offset + local_index,
                        "box": [float(value) for value in boxes[local_index]],
                        "area": int(candidates["areas"][local_index].item()),
                        "score": float(scores[local_index]),
                        "scale_mean_score": float(scale_mean_scores[local_index]),
                        "peak_score": float(candidates["peak_scores"][local_index].item()),
                        "coarse_score": float(candidates["coarse_scores"][local_index].item()),
                        "coarse_peak_score": float(
                            candidates["coarse_peak_scores"][local_index].item()
                        ),
                        "scale_responses": [
                            float(value)
                            for value in candidates["scale_responses"][local_index].tolist()
                        ],
                        "scale_variance": float(
                            candidates["scale_variance"][local_index].item()
                        ),
                        "label_id": int(labels[local_index]),
                        "label": LABEL_NAMES[int(labels[local_index])],
                        "max_iou": float(matching["max_iou"][local_index].item()),
                        "matched_gt_index": int(matched_gt[local_index]),
                        "center_match": bool(matching["center_match"][local_index].item()),
                        "centroid_distance": finite_or_none(
                            matching["centroid_distance"][local_index].item()
                        ),
                        "mask_rle": encode_mask_rle(candidate_masks[local_index]),
                        "mask_shape": [args.base_size, args.base_size],
                    }
                )
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            all_matched_gt.extend(matched_gt.tolist())
            all_batch_indices.extend(batch_indices.tolist())

    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int64)
    matched_gt = np.asarray(all_matched_gt, dtype=np.int64)
    batch_indices = np.asarray(all_batch_indices, dtype=np.int64)
    tp_flags = one_to_one_tp_flags(labels, matched_gt, batch_indices, scores)
    thresholds = np.linspace(1.0, 0.0, 101)
    curve = fppi_froc(
        scores,
        tp_flags,
        num_images=len(evaluated_dataset),
        num_targets=total_gt_targets,
        thresholds=thresholds,
    )

    calibration_mask = labels != UNCERTAIN_LABEL
    calibration_scores = scores[calibration_mask]
    calibration_labels = labels[calibration_mask]
    probabilities = np.column_stack((calibration_scores, 1.0 - calibration_scores))
    class_counts = {
        name: int(np.count_nonzero(labels == index))
        for index, name in enumerate(LABEL_NAMES)
    }
    score_means = {
        name: finite_or_none(scores[labels == index].mean())
        if np.any(labels == index)
        else None
        for index, name in enumerate(LABEL_NAMES)
    }
    hard_false_positive = (scores >= args.hard_negative_threshold) & ~tp_flags
    images_with_hard_fp = len(set(batch_indices[hard_false_positive].tolist()))
    metrics = {
        "dataset_dir": args.dataset_dir,
        "weight_path": args.weight_path,
        "split": args.split,
        "num_images": len(evaluated_dataset),
        "num_gt_targets": total_gt_targets,
        "num_candidates": int(scores.size),
        "candidate_threshold": args.candidate_threshold,
        "candidate_score": args.candidate_score,
        "hard_negative_threshold": args.hard_negative_threshold,
        "positive_iou": args.positive_iou,
        "center_distance": args.center_distance,
        "class_counts": class_counts,
        "class_score_means": score_means,
        "high_confidence_false_positives": int(hard_false_positive.sum()),
        "images_with_high_confidence_false_positives": images_with_hard_fp,
        "high_confidence_fp_image_fraction": images_with_hard_fp / len(evaluated_dataset),
        "FPPI_at_hard_threshold": float(hard_false_positive.sum() / len(evaluated_dataset)),
        "Candidate_ECE": finite_or_none(
            candidate_ece(probabilities, calibration_labels, target_class=0)
        ),
        "Candidate_Brier": finite_or_none(
            candidate_brier_score(probabilities, calibration_labels)
        ),
        "Candidate_NLL": finite_or_none(candidate_nll(probabilities, calibration_labels)),
        "FPPI_at_Pd_0.90": finite_or_none(fppi_at_fixed_pd(curve, 0.90)),
        "FPPI_at_Pd_0.95": finite_or_none(fppi_at_fixed_pd(curve, 0.95)),
        "go_signal": bool(
            class_counts["target"] > 0
            and class_counts["clutter"] > 0
            and images_with_hard_fp > 0
        ),
        "froc": {
            "thresholds": curve["thresholds"].tolist(),
            "Pd": curve["pd"].tolist(),
            "FPPI": curve["fppi"].tolist(),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / f"{args.split}_candidates.json").write_text(
        json.dumps(
            {"metadata": {key: value for key, value in metrics.items() if key != "froc"},
             "candidates": records},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_false_positive_crops(records, dataset, output_dir, args.save_crops)
    print(json.dumps({key: value for key, value in metrics.items() if key != "froc"}, indent=2))


if __name__ == "__main__":
    main()
