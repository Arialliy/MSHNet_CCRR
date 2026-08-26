import argparse
import json
from pathlib import Path
import sys

import pytest

from main import CandidateBank, parse_args, should_run_scheduled_test


def _bank_args(tmp_path):
    return argparse.Namespace(
        dataset_dir=str(tmp_path / "dataset"),
        candidate_threshold=0.2,
        hard_negative_threshold=0.5,
        candidate_score="coarse_peak",
        positive_iou=0.3,
        center_distance=3.0,
        min_candidate_area=1,
        max_candidate_area=1024,
        base_size=8,
    )


def _bank_payload(args):
    return {
        "metadata": {
            "schema_version": "mshnet-ccrr-candidate-bank/v1",
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "weight_sha256": "a" * 64,
            "split": "train",
            "num_images": 2,
            "num_candidates": 0,
            "candidate_threshold": 0.2,
            "hard_negative_threshold": 0.5,
            "candidate_score": "coarse_peak",
            "positive_iou": 0.3,
            "center_distance": 3.0,
            "min_area": 1,
            "max_area": 1024,
            "base_size": 8,
            "label_order": ["target", "clutter", "uncertain"],
            "box_format": "xyxy_half_open",
            "mask_encoding": "row_major_start_length_rle",
            "proposal_aggregation": "mean_sigmoid_multiscale",
            "component_connectivity": 8,
            "matching_rule": "iou_or_center_inside_and_centroid_distance",
        },
        "images": [
            {"name": "image-a", "num_candidates": 0},
            {"name": "image-b", "num_candidates": 0},
        ],
        "candidates": [],
    }


def test_epoch_500_is_first_scheduled_test_epoch():
    assert not should_run_scheduled_test(499, 500)
    assert should_run_scheduled_test(500, 500)
    assert should_run_scheduled_test(999, 500)


def test_requested_training_defaults_are_frozen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = parse_args()

    assert args.epochs == 1000
    assert args.test_start_epoch == 500
    assert args.hard_negative_threshold == 0.5
    assert args.candidate_threshold == 0.2
    assert args.froc_bins == 100


def test_candidate_bank_requires_exact_img_idx_manifest_and_metadata(tmp_path):
    args = _bank_args(tmp_path)
    path = tmp_path / "train_candidates.json"
    path.write_text(json.dumps(_bank_payload(args)), encoding="utf-8")
    bank = CandidateBank(str(path), image_size=8)

    bank.validate_contract(
        args, expected_split="train", expected_names=["image-a", "image-b"]
    )

    with pytest.raises(ValueError, match="image manifest"):
        bank.validate_contract(
            args, expected_split="train", expected_names=["image-b", "image-a"]
        )
    args.hard_negative_threshold = 0.6
    with pytest.raises(ValueError, match="hard_negative_threshold"):
        bank.validate_contract(
            args, expected_split="train", expected_names=["image-a", "image-b"]
        )
