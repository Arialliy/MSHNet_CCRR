import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from main import (
    Trainer,
    _fp_value_weights_for_risk,
    _sca_enhanced_diagnostics,
    parse_args,
)
from model.MSHNet import MSHNet


def test_enhanced_cli_defaults_preserve_legacy_v2_behavior(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = parse_args()

    assert args.candidate_pooling == "avg"
    assert args.candidate_topk_ratio == pytest.approx(0.125)
    assert args.candidate_minimum_topk == 1
    assert args.target_tail_weight == 0.0
    assert args.target_tail_temperature == pytest.approx(0.1)
    assert args.fp_value_beta == 0.0
    assert args.fp_value_max == pytest.approx(3.0)
    assert args.save_best_pareto is False


def test_legacy_configs_accept_only_missing_enhanced_default_fields():
    legacy_inference = {}
    legacy_training = {}
    default_inference = {
        "candidate_pooling": "avg",
        "candidate_topk_ratio": 0.125,
        "candidate_minimum_topk": 1,
    }
    default_training = {
        **default_inference,
        "target_tail_weight": 0.0,
        "target_tail_temperature": 0.1,
        "fp_value_beta": 0.0,
        "fp_value_max": 3.0,
        "save_best_pareto": False,
    }

    assert Trainer._config_differences(legacy_inference, default_inference) == []
    assert (
        Trainer._config_differences(
            legacy_training,
            default_training,
            "training_config",
        )
        == []
    )
    assert Trainer._config_differences(
        legacy_inference,
        dict(default_inference, candidate_pooling="avg_max_topk"),
    ) == ["candidate_pooling: missing"]


def test_sca_component_value_uses_exact_area_and_keeps_targets_at_one():
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        candidate_score="coarse_peak",
        center_distance=0.0,
        quality_iou_weight=0.5,
        quality_center_sigma=3.0,
        fp_value_beta=2.0,
        fp_value_max=3.0,
    )
    masks = torch.zeros((3, 8, 8), dtype=torch.bool)
    masks[0, 1, 1] = True
    masks[1, 3, 3] = True
    masks[2, 4:6, 4:6] = True
    candidates = {
        "action_masks": masks,
        "batch_indices": torch.zeros(3, dtype=torch.long),
        "coarse_peak_scores": torch.full((3,), 0.8),
    }
    labels = torch.zeros((1, 1, 8, 8))
    labels[0, 0, 1, 1] = 1.0

    matching = trainer._label_sca_candidates(candidates, labels)

    assert matching["action_areas"].tolist() == [1.0, 1.0, 4.0]
    assert matching["is_target_component"].tolist() == [True, False, False]
    assert matching["fp_value_weights"][0].item() == pytest.approx(1.0)
    expected_small = 1.0 + 2.0 * math.log1p(1.0) / math.log1p(64.0)
    expected_large = 1.0 + 2.0 * math.log1p(4.0) / math.log1p(64.0)
    assert matching["fp_value_weights"][1].item() == pytest.approx(expected_small)
    assert matching["fp_value_weights"][2].item() == pytest.approx(expected_large)
    assert matching["fp_value_weights"][2] > matching["fp_value_weights"][1]


def test_disabled_fp_value_selects_bit_exact_legacy_risk_path():
    weights = torch.ones(3)

    assert _fp_value_weights_for_risk(weights, beta=0.0) is None
    assert _fp_value_weights_for_risk(weights, beta=2.0) is weights


def test_enhanced_diagnostics_report_target_tail_and_fp_area_bins():
    records = [
        {"is_target": True, "is_ambiguous": False, "gate": 0.0},
        {"is_target": True, "is_ambiguous": False, "gate": 0.8},
        {
            "is_clutter": True,
            "is_ambiguous": False,
            "gate": 1.0,
            "fp_value_weight": 1.2,
            "action_area": 1,
            "action_threshold_passed": True,
            "component_eliminated": True,
        },
        {
            "is_clutter": True,
            "is_ambiguous": False,
            "gate": 0.0,
            "fp_value_weight": 2.0,
            "action_area": 20,
            "action_threshold_passed": False,
            "component_eliminated": False,
        },
    ]

    diagnostics = _sca_enhanced_diagnostics(
        records, target_tail_temperature=0.1
    )

    assert diagnostics["target_gate"]["target_gate_max"] == pytest.approx(0.8)
    assert diagnostics["target_gate"]["num_target_gate_above_0.5"] == 1
    assert diagnostics["target_gate"]["target_tail_softmax"] > 0.5
    assert diagnostics["fp_value"]["fp_value_weight_max"] == pytest.approx(2.0)
    assert diagnostics["fp_area_bins"]["1_pixel"]["num_eliminated"] == 1
    assert diagnostics["fp_area_bins"]["17_64_pixels"]["num_actions"] == 0


def _write_tiny_dataset(root: Path) -> Path:
    dataset = root / "Tiny-SIRST"
    for folder in ("images", "masks", "img_idx"):
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    for name in ("train_image", "test_image"):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[7:9, 7:9] = 255
        Image.fromarray(image).save(dataset / "images" / f"{name}.png")
        Image.fromarray(mask).save(dataset / "masks" / f"{name}.png")
    (dataset / "img_idx" / "train_Tiny-SIRST.txt").write_text(
        "train_image\n", encoding="utf-8"
    )
    (dataset / "img_idx" / "test_Tiny-SIRST.txt").write_text(
        "test_image\n", encoding="utf-8"
    )
    return dataset


def test_test_selection_keeps_dual_best_and_adds_pareto(tmp_path, monkeypatch):
    dataset = _write_tiny_dataset(tmp_path)
    baseline = MSHNet(3)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.zero_()
        baseline.final.bias.fill_(1.0)
    baseline_path = tmp_path / "baseline.pth"
    torch.save(baseline.state_dict(), baseline_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset-dir",
            str(dataset),
            "--weight-path",
            str(baseline_path),
            "--save-dir",
            str(tmp_path / "runs"),
            "--enable-ccrr",
            "--ccrr-version",
            "v2_selective_component",
            "--candidate-pooling",
            "avg_max_topk",
            "--target-tail-weight",
            "5.0",
            "--fp-value-beta",
            "2.0",
            "--save-best-pareto",
            "--base-size",
            "16",
            "--crop-size",
            "16",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--max-test-batches",
            "1",
            "--device",
            "cpu",
        ],
    )
    trainer = Trainer(parse_args())

    metrics = trainer.test(500)

    save_folder = Path(trainer.save_folder)
    assert (save_folder / "best_miou.pkl").is_file()
    assert (save_folder / "best_pd.pkl").is_file()
    assert (save_folder / "best_pareto.pkl").is_file()
    status = json.loads((save_folder / "pareto_status.json").read_text())
    assert status["pareto_found"] is True
    assert status["best_pareto_epoch"] == 500
    assert metrics["test_selection"]["best_Pareto_weight"] == "best_pareto.pkl"
    pareto_artifact = torch.load(
        save_folder / "best_pareto.pkl",
        map_location="cpu",
        weights_only=False,
    )
    assert pareto_artifact["selection_metric"] == "pareto"
    assert pareto_artifact["selection_details"]["constraints"] == {
        "mIoU_not_below": True,
        "nIoU_not_below": True,
        "Pd_not_below": True,
        "FPPI_not_above": True,
        "Fa_not_above": True,
    }
