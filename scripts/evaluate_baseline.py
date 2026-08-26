#!/usr/bin/env python3
"""Evaluate a frozen MSHNet baseline without enabling CCRR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.MSHNet import MSHNet
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset_args = argparse.Namespace(
        dataset_dir=args.dataset_dir,
        base_size=args.base_size,
        crop_size=args.base_size,
    )
    dataset = IRSTD_Dataset(dataset_args, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MSHNet(3).to(device)
    checkpoint = torch.load(args.weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()

    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    total_intersection = 0.0
    total_union = 0.0
    per_image_iou = []
    pd_fa = PD_FA(1, 10, args.base_size)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="baseline-eval"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, warm_flag=True)
            logits = outputs["coarse_logits"]
            predictions = logits > 0
            targets = masks > 0

            intersection = (predictions & targets).sum().item()
            union = (predictions | targets).sum().item()
            total_intersection += intersection
            total_union += union
            per_image_iou.append(intersection / union if union else 1.0)
            pd_fa.update(logits, masks)

            if prediction_dir is not None:
                array = predictions[0, 0].to(torch.uint8).cpu().numpy() * 255
                Image.fromarray(array).save(prediction_dir / (batch["name"][0] + ".png"))

    false_alarm, detection_probability = pd_fa.get(len(dataset))
    metrics = {
        "dataset_dir": args.dataset_dir,
        "weight_path": args.weight_path,
        "num_images": len(dataset),
        "threshold": 0.5,
        "mIoU": total_intersection / total_union if total_union else 1.0,
        "nIoU": float(np.mean(per_image_iou)),
        "Pd": float(detection_probability[0]),
        "Fa_per_million_pixels": float(false_alarm[0] * 1_000_000),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
