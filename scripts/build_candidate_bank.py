#!/usr/bin/env python3
"""Build a deterministic offline CCRR candidate bank from a frozen MSHNet.

The script writes one ``<split>_candidates.json`` file per requested split and
a ``manifest.json`` in the output directory.  Candidate masks use row-major
run-length encoding, and boxes use half-open ``[x1, y1, x2, y2]`` coordinates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.MSHNet import MSHNet
from scripts.diagnose_baseline import (
    encode_mask_rle,
    finite_or_none,
    select_device,
    state_dict_from_checkpoint,
)
from utils.candidate import LABEL_NAMES, generate_candidates, match_candidates_to_gt
from utils.data import IRSTD_Dataset


SCHEMA_VERSION = "mshnet-ccrr-candidate-bank/v1"
SCORE_FIELDS = {
    "coarse_peak": "coarse_peak_scores",
    "coarse_mean": "coarse_scores",
    "scale_peak": "peak_scores",
    "scale_mean": "scores",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a probability in [0, 1]")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset root containing images/masks.")
    parser.add_argument("--weight-path", required=True, help="Frozen baseline checkpoint or weight file.")
    parser.add_argument("--output-dir", required=True, help="Directory for split banks and manifest.json.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Deterministic dataset splits to export (default: train test).",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=probability,
        default=0.2,
        help="Low multi-scale mean threshold used to form candidates (default: 0.2).",
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=probability,
        default=0.5,
        help="Minimum selected score for a zero-overlap clutter label (default: 0.5).",
    )
    parser.add_argument(
        "--candidate-score",
        choices=tuple(SCORE_FIELDS),
        default="coarse_peak",
        help="Score used for labels and saved as each candidate's primary score.",
    )
    parser.add_argument("--positive-iou", type=probability, default=0.3)
    parser.add_argument(
        "--center-distance",
        type=float,
        default=3.0,
        help="Maximum centroid distance in pixels for the auxiliary positive match.",
    )
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument(
        "--max-area",
        type=int,
        default=1024,
        help="Maximum candidate area; use 0 for no upper bound.",
    )
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Per-split image limit for smoke runs; 0 processes the whole split.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split bank files and manifest.json.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.center_distance < 0:
        parser.error("--center-distance must be non-negative")
    if args.min_area < 1:
        parser.error("--min-area must be positive")
    if args.max_area < 0:
        parser.error("--max-area must be non-negative")
    if args.max_area and args.max_area < args.min_area:
        parser.error("--max-area must be 0 or at least --min-area")
    if args.base_size <= 0:
        parser.error("--base-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.max_images < 0:
        parser.error("--max-images must be non-negative")
    # Preserve the requested order while avoiding duplicate output files.
    args.splits = list(dict.fromkeys(args.splits))
    return args


def select_candidate_scores(
    candidates: Mapping[str, torch.Tensor], score_name: str
) -> torch.Tensor:
    """Select the configured [N] confidence tensor from candidate output."""
    if score_name not in SCORE_FIELDS:
        raise ValueError(f"unknown candidate score {score_name!r}")
    field = SCORE_FIELDS[score_name]
    if field not in candidates:
        raise KeyError(f"candidate output does not contain {field!r}")
    scores = candidates[field]
    if scores.ndim != 1:
        raise ValueError(f"candidate score field {field!r} must have shape [N]")
    return scores


def decode_mask_rle(runs: Sequence[Sequence[int]], shape: Sequence[int]) -> np.ndarray:
    """Decode the bank's row-major RLE mask representation."""
    if len(shape) != 2 or int(shape[0]) < 0 or int(shape[1]) < 0:
        raise ValueError("shape must contain non-negative height and width")
    height, width = int(shape[0]), int(shape[1])
    flat = np.zeros(height * width, dtype=bool)
    for run in runs:
        if len(run) != 2:
            raise ValueError("each RLE run must contain [start, length]")
        start, length = int(run[0]), int(run[1])
        if start < 0 or length <= 0 or start + length > flat.size:
            raise ValueError("RLE run lies outside the requested mask shape")
        flat[start : start + length] = True
    return flat.reshape(height, width)


def _baseline_outputs(outputs: Any) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
    """Read both the CCRR-compatible dict and the original tuple interface."""
    if isinstance(outputs, Mapping):
        return outputs["multi_scale_logits"], outputs["coarse_logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1]
    raise TypeError("MSHNet must return an output mapping or (multi_scale_logits, coarse_logits)")


def candidate_records_from_batch(
    candidates: Mapping[str, torch.Tensor],
    matching: Mapping[str, torch.Tensor],
    image_names: Sequence[str],
    *,
    image_offset: int,
    candidate_offset: int,
    score_name: str,
    hard_negative_threshold: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Serialize one generated batch without retaining GPU tensors."""
    serialized_fields = (
        "masks",
        "boxes",
        "batch_indices",
        "areas",
        "scores",
        "peak_scores",
        "coarse_scores",
        "coarse_peak_scores",
        "scale_responses",
        "scale_variance",
    )
    missing = [field for field in serialized_fields if field not in candidates]
    if missing:
        raise KeyError(f"candidate output is missing fields: {', '.join(missing)}")
    candidate_cpu = {
        field: candidates[field].detach().cpu() for field in serialized_fields
    }
    scores = select_candidate_scores(candidate_cpu, score_name)
    masks = candidate_cpu["masks"].numpy().astype(bool, copy=False)
    boxes = candidate_cpu["boxes"]
    batch_indices = candidate_cpu["batch_indices"].long()
    labels = matching["labels"].detach().cpu().long()
    matched_gt = matching["matched_gt_indices"].detach().cpu().long()
    max_iou = matching["max_iou"].detach().cpu()
    center_match = matching.get("center_match")
    centroid_distance = matching.get("centroid_distance")
    if center_match is None:
        center_match = torch.zeros(labels.shape, dtype=torch.bool)
    else:
        center_match = center_match.detach().cpu().bool()
    if centroid_distance is None:
        centroid_distance = torch.full(labels.shape, float("inf"))
    else:
        centroid_distance = centroid_distance.detach().cpu()

    number_of_candidates = masks.shape[0]
    candidate_fields = (
        boxes,
        batch_indices,
        labels,
        matched_gt,
        max_iou,
        center_match,
        centroid_distance,
        candidate_cpu["areas"],
        candidate_cpu["scores"],
        candidate_cpu["peak_scores"],
        candidate_cpu["coarse_scores"],
        candidate_cpu["coarse_peak_scores"],
        candidate_cpu["scale_variance"],
    )
    if any(field.shape[0] != number_of_candidates for field in candidate_fields):
        raise ValueError("candidate and matching fields disagree on candidate count")
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("candidate boxes must have shape [N,5]")
    if len(image_names) == 0 and number_of_candidates:
        raise ValueError("image_names cannot be empty for a non-empty candidate batch")

    per_image_counts = [0 for _ in image_names]
    records: list[dict[str, Any]] = []
    for local_index in range(number_of_candidates):
        batch_index = int(batch_indices[local_index].item())
        if batch_index < 0 or batch_index >= len(image_names):
            raise ValueError("candidate batch index is outside image_names")
        if int(round(float(boxes[local_index, 0].item()))) != batch_index:
            raise ValueError("candidate box and batch_indices disagree")
        label_id = int(labels[local_index].item())
        if not 0 <= label_id < len(LABEL_NAMES):
            raise ValueError(f"candidate label {label_id} is outside LABEL_NAMES")
        candidate_index = per_image_counts[batch_index]
        per_image_counts[batch_index] += 1
        selected_score = float(scores[local_index].item())
        mask_shape = [int(masks.shape[1]), int(masks.shape[2])]
        records.append(
            {
                "name": str(image_names[batch_index]),
                "image_index": image_offset + batch_index,
                "candidate_index": candidate_index,
                "global_candidate_index": candidate_offset + local_index,
                "box": [float(value) for value in boxes[local_index, 1:].tolist()],
                "area": int(candidate_cpu["areas"][local_index].item()),
                "score": selected_score,
                "score_type": score_name,
                "scale_mean_score": float(candidate_cpu["scores"][local_index].item()),
                "scale_peak_score": float(candidate_cpu["peak_scores"][local_index].item()),
                "coarse_mean_score": float(candidate_cpu["coarse_scores"][local_index].item()),
                "coarse_peak_score": float(
                    candidate_cpu["coarse_peak_scores"][local_index].item()
                ),
                "scale_responses": [
                    float(value)
                    for value in candidate_cpu["scale_responses"][local_index].tolist()
                ],
                "scale_variance": float(candidate_cpu["scale_variance"][local_index].item()),
                "label_id": label_id,
                "label": LABEL_NAMES[label_id],
                "is_hard_negative": bool(
                    label_id == 1 and selected_score >= hard_negative_threshold
                ),
                "max_iou": float(max_iou[local_index].item()),
                "matched_gt_index": int(matched_gt[local_index].item()),
                "center_match": bool(center_match[local_index].item()),
                "centroid_distance": finite_or_none(centroid_distance[local_index].item()),
                "mask_rle": encode_mask_rle(masks[local_index]),
                "mask_shape": mask_shape,
            }
        )
    return records, per_image_counts


def build_split_bank(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run one deterministic split through the frozen baseline."""
    dataset_args = argparse.Namespace(
        dataset_dir=args.dataset_dir,
        base_size=args.base_size,
        crop_size=args.base_size,
    )
    # Deliberately use test mode for every split: offline banks must not depend
    # on random crop, scale, mirror, or blur augmentation.
    dataset = IRSTD_Dataset(dataset_args, mode="test", split=split)
    evaluated_dataset = dataset
    if args.max_images:
        evaluated_dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        evaluated_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    image_offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"candidate-bank:{split}", disable=args.quiet):
            images = batch["image"].to(device, non_blocking=True)
            gt_masks = batch["mask"].to(device, non_blocking=True)
            multi_scale_logits, coarse_logits = _baseline_outputs(
                model(images, warm_flag=True)
            )
            candidates = generate_candidates(
                coarse_logits=coarse_logits,
                multi_scale_logits=multi_scale_logits,
                threshold_low=args.candidate_threshold,
                min_area=args.min_area,
                max_area=args.max_area or None,
            )
            selected_scores = select_candidate_scores(candidates, args.candidate_score)
            candidates_for_matching = dict(candidates)
            candidates_for_matching["scores"] = selected_scores
            matching = match_candidates_to_gt(
                candidates_for_matching,
                gt_masks,
                positive_iou=args.positive_iou,
                hard_negative_threshold=args.hard_negative_threshold,
                center_distance=args.center_distance,
            )

            image_names = [str(name) for name in batch["name"]]
            batch_records, per_image_counts = candidate_records_from_batch(
                candidates,
                matching,
                image_names,
                image_offset=image_offset,
                candidate_offset=len(records),
                score_name=args.candidate_score,
                hard_negative_threshold=args.hard_negative_threshold,
            )
            records.extend(batch_records)
            for local_batch_index, (name, count) in enumerate(
                zip(image_names, per_image_counts)
            ):
                image_records.append(
                    {
                        "name": name,
                        "image_index": image_offset + local_batch_index,
                        "num_candidates": count,
                    }
                )
            image_offset += len(image_names)

    class_counts = {
        name: sum(record["label_id"] == label_id for record in records)
        for label_id, name in enumerate(LABEL_NAMES)
    }
    hard_negative_count = sum(record["is_hard_negative"] for record in records)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "weight_path": str(Path(args.weight_path).resolve()),
        "weight_sha256": args.weight_sha256,
        "split": split,
        "num_images": image_offset,
        "num_candidates": len(records),
        "class_counts": class_counts,
        "num_hard_negatives": hard_negative_count,
        "label_order": list(LABEL_NAMES),
        "candidate_threshold": args.candidate_threshold,
        "hard_negative_threshold": args.hard_negative_threshold,
        "candidate_score": args.candidate_score,
        "positive_iou": args.positive_iou,
        "center_distance": args.center_distance,
        "min_area": args.min_area,
        "max_area": args.max_area or None,
        "base_size": args.base_size,
        "box_format": "xyxy_half_open",
        "mask_encoding": "row_major_start_length_rle",
        "proposal_aggregation": "mean_sigmoid_multiscale",
        "component_connectivity": 8,
        "matching_rule": "iou_or_center_inside_and_centroid_distance",
    }
    return {"metadata": metadata, "images": image_records, "candidates": records}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON beside its destination and atomically replace on success."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _output_paths(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    return [output_dir / f"{split}_candidates.json" for split in args.splits] + [
        output_dir / "manifest.json"
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = _output_paths(args)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"candidate-bank output already exists: {formatted}; use --overwrite")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    checkpoint = torch.load(args.weight_path, map_location=device, weights_only=False)
    args.weight_sha256 = file_sha256(args.weight_path)
    model = MSHNet(3).to(device)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "weight_path": str(Path(args.weight_path).resolve()),
        "weight_sha256": args.weight_sha256,
        "candidate_score": args.candidate_score,
        "splits": {},
    }
    for split in args.splits:
        bank = build_split_bank(model, args=args, split=split, device=device)
        output_path = output_dir / f"{split}_candidates.json"
        write_json_atomic(output_path, bank)
        manifest["splits"][split] = {
            "path": output_path.name,
            "num_images": bank["metadata"]["num_images"],
            "num_candidates": bank["metadata"]["num_candidates"],
            "class_counts": bank["metadata"]["class_counts"],
            "num_hard_negatives": bank["metadata"]["num_hard_negatives"],
        }

    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
