"""Common I/O and protocol checks for frozen CCRR audit scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from model.MSHNet import MSHNet
from utils.audit import file_sha256
from utils.data import IRSTD_Dataset


def add_frozen_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "net" in state:
        state = state["net"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) for key in state
    ):
        raise TypeError("checkpoint does not contain a string-keyed model state dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_frozen_coarse_model(
    weight_path: str | Path, device: torch.device
) -> tuple[MSHNet, dict[str, Any]]:
    path = Path(weight_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = state_dict_from_checkpoint(checkpoint)
    coarse_state = {
        key: value for key, value in state.items() if not key.startswith("ccrr.")
    }
    model = MSHNet(3).to(device)
    model.load_state_dict(coarse_state, strict=True)
    model.eval()
    metadata: dict[str, Any] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_contains_ccrr": len(coarse_state) != len(state),
    }
    if isinstance(checkpoint, dict):
        for key in ("epoch", "selection_metric", "selection_value"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    return model, metadata


def build_official_test_loader(
    dataset_dir: str | Path,
    *,
    base_size: int,
    num_workers: int,
    max_images: int,
    device: torch.device,
) -> tuple[IRSTD_Dataset, DataLoader, dict[str, Any]]:
    if base_size <= 0 or base_size % 16:
        raise ValueError("base_size must be a positive multiple of 16")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_images < 0:
        raise ValueError("max_images must be non-negative")
    root = Path(dataset_dir).resolve()
    dataset_args = argparse.Namespace(
        dataset_dir=str(root), base_size=base_size, crop_size=base_size
    )
    dataset = IRSTD_Dataset(dataset_args, mode="test", split="test")
    evaluated_dataset = dataset
    if max_images:
        evaluated_dataset = Subset(dataset, range(min(max_images, len(dataset))))
    loader = DataLoader(
        evaluated_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    manifests = {}
    dataset_name = root.name
    for split in ("train", "test"):
        manifest_path = root / "img_idx" / f"{split}_{dataset_name}.txt"
        names = IRSTD_Dataset._read_split_names(manifest_path)
        manifests[split] = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "num_images": len(names),
        }
    protocol = {
        "dataset_dir": str(root),
        "dataset": dataset_name,
        "split": "test",
        "validation_split": None,
        "test_is_development_and_selection_split": True,
        "manifests": manifests,
        "evaluated_images": len(evaluated_dataset),
        "base_size": base_size,
    }
    return dataset, loader, protocol


__all__ = [
    "add_frozen_audit_arguments",
    "build_official_test_loader",
    "load_frozen_coarse_model",
    "select_device",
    "state_dict_from_checkpoint",
]
