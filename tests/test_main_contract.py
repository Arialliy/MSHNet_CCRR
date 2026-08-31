import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from main import (
    CandidateBank,
    Trainer,
    parse_args,
    should_run_scheduled_test,
)


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


def test_v1_safe_model_defaults_keep_requested_test_schedule(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = parse_args()

    assert args.hidden_dim == 64
    assert args.max_delta == 1.5
    assert args.ccrr_lr == pytest.approx(3e-4)
    assert args.epochs == 1000
    assert args.test_start_epoch == 500
    assert args.ccrr_version == "v1_safe"
    assert not hasattr(args, "val_start_epoch")


def test_threshold_aware_cli_and_inference_config_are_versioned(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--enable-ccrr",
            "--ccrr-version",
            "v1_threshold_aware",
        ],
    )
    args = parse_args()
    trainer = Trainer.__new__(Trainer)
    trainer.args = args

    config = trainer._inference_config()

    assert config["ccrr_version"] == "v1_threshold_aware"
    assert config["rectifier"] == "threshold_aware"
    assert config["clutter_action_threshold"] == pytest.approx(0.9)
    assert config["remove_threshold"] == pytest.approx(0.45)
    assert config["max_action_suppression"] is None


def test_v1_safe_inference_config_omits_new_version_fields(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--enable-ccrr"])
    args = parse_args()
    trainer = Trainer.__new__(Trainer)
    trainer.args = args

    config = trainer._inference_config()

    assert config["rectifier"] == "suppression_only"
    assert "ccrr_version" not in config
    assert "clutter_action_threshold" not in config


def test_resume_rejects_best_weight_without_optimizer_state(monkeypatch):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.scheduler = None
    best_weight = {
        "checkpoint_schema": "mshnet-ccrr-weight/v2-safe",
        "net": {},
        "epoch": 500,
        "inference_config": {},
        "evaluation_config": {},
        "training_config": {},
        "provenance": {},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: best_weight)

    with pytest.raises(RuntimeError, match="full checkpoint.pkl"):
        trainer._resume_checkpoint("best_miou.pkl")


def test_v1_presence_labels_supervise_low_score_zero_overlap_candidate():
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        positive_iou=0.3,
        hard_negative_threshold=0.5,
        center_distance=3.0,
        candidate_score="coarse_peak",
        easy_negative_weight=0.5,
        hard_negative_weight=2.0,
        hardness_gamma=2.0,
    )
    masks = torch.zeros((2, 8, 8), dtype=torch.bool)
    masks[0, 7, 7] = True
    masks[1, 0, 0] = True
    candidates = {
        "masks": masks,
        "batch_indices": torch.tensor([0, 0]),
        "coarse_peak_scores": torch.tensor([0.9, 0.3]),
    }
    ground_truth = torch.zeros((1, 1, 8, 8))
    ground_truth[0, 0, 7, 7] = 1

    matching = trainer._label_candidates(candidates, ground_truth)

    assert matching["strict_labels"].tolist() == [0, 2]
    assert matching["training_labels"].tolist() == [0, 1]
    assert not torch.any(matching["training_labels"] == -1)
    assert matching["sample_weights"][1].item() == pytest.approx(
        0.5 + 1.5 * 0.3**2
    )


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
