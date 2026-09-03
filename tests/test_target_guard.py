from types import SimpleNamespace
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from main import (
    SCA_WEIGHT_SCHEMA_VERSION,
    TARGET_GUARDED_SCA_WEIGHT_SCHEMA_VERSION,
    Trainer,
    _target_guard_diagnostics,
    _target_guard_pareto_constraints,
    parse_args,
)
from model.MSHNet import MSHNet
from model.candidate_loss import TargetGuardLoss
from model.ccrr import (
    SCACRRModule,
    TargetGuardedReliabilityHead,
    TargetGuardedRiskGate,
    TargetGuardedSCACRRModule,
)


def test_target_guarded_head_zero_initialization_outputs_three_probabilities():
    head = TargetGuardedReliabilityHead(
        input_dim=6,
        hidden_dim=4,
        dropout=0.0,
    )
    outputs = head(torch.randn(3, 6))

    assert outputs["clutter_probability"].shape == (3,)
    assert outputs["target_quality"].shape == (3,)
    assert outputs["target_guard"].shape == (3,)
    assert torch.equal(outputs["clutter_probability"], torch.full((3,), 0.5))
    assert torch.equal(outputs["target_quality"], torch.full((3,), 0.5))
    assert torch.equal(outputs["target_guard"], torch.full((3,), 0.5))


def test_eval_requires_risk_quality_and_guard_conditions():
    gate = TargetGuardedRiskGate().eval()
    clutter = torch.tensor([0.99, 0.99, 0.99, 0.05])
    quality = torch.tensor([0.10, 0.30, 0.10, 0.10])
    guard = torch.tensor([0.10, 0.10, 0.90, 0.10])

    outputs = gate(clutter, quality, guard)

    assert outputs["gate"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert outputs["quality_veto"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert outputs["guard_veto"].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_low_quality_target_is_protected_by_high_guard():
    gate = TargetGuardedRiskGate().eval()
    outputs = gate(
        clutter_probability=torch.tensor([0.99]),
        target_quality=torch.tensor([0.01]),
        target_guard=torch.tensor([0.99]),
    )

    assert outputs["risk_score"].item() >= gate.risk_threshold
    assert outputs["quality_veto"].item() == 1.0
    assert outputs["guard_veto"].item() == 0.0
    assert outputs["gate"].item() == 0.0


def test_train_time_guard_and_quality_veto_have_gradients():
    gate = TargetGuardedRiskGate().train()
    clutter = torch.tensor([0.30], requires_grad=True)
    quality = torch.tensor([0.18], requires_grad=True)
    guard = torch.tensor([0.25], requires_grad=True)

    gate(clutter, quality, guard)["gate"].sum().backward()

    for value in (clutter, quality, guard):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_equal_evidence_is_exact_training_noop():
    outputs = TargetGuardedRiskGate().train()(
        torch.tensor([0.5]),
        torch.tensor([0.5]),
        torch.tensor([0.5]),
    )

    assert torch.equal(outputs["soft_action"], torch.zeros(1))
    assert torch.equal(outputs["gate"], torch.zeros(1))


def test_target_guard_loss_prioritizes_positive_false_negatives_and_backpropagates():
    logits = torch.tensor([-2.0, 2.0, -1.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0])
    valid = torch.tensor([True, True, False])

    terms = TargetGuardLoss(
        positive_weight=4.0,
        false_negative_weight=2.0,
        tail_temperature=0.1,
    )(logits, targets, valid)

    assert terms["total"] > terms["bce"]
    assert terms["false_negative"] > 0
    assert terms["tail_false_negative"] > 0
    terms["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[2].item() == 0.0


def test_target_guard_loss_empty_valid_set_is_differentiable_zero():
    logits = torch.randn(2, requires_grad=True)
    terms = TargetGuardLoss()(logits, torch.zeros(2), torch.zeros(2, dtype=torch.bool))

    assert terms["total"].item() == 0.0
    terms["total"].backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_sca_labels_protect_tp_ambiguous_and_only_mark_safe_fp_zero():
    action_masks = torch.zeros((2, 9, 9), dtype=torch.bool)
    action_masks[0, 4, 3:6] = True
    action_masks[1, 0, 0] = True
    labels = torch.zeros((1, 1, 9, 9))
    labels[0, 0, 4, 2] = 1.0
    labels[0, 0, 4, 6] = 1.0
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        ccrr_version="v2_target_guarded_component",
        center_distance=3.0,
        candidate_score="coarse_peak",
        quality_iou_weight=0.5,
        quality_center_sigma=3.0,
        fp_value_beta=0.0,
        fp_value_max=3.0,
    )
    candidates = {
        "action_masks": action_masks,
        "batch_indices": torch.zeros(2, dtype=torch.long),
        "coarse_peak_scores": torch.tensor([0.9, 0.8]),
    }

    matching = trainer._label_sca_candidates(candidates, labels)

    assert matching["ambiguous_keep"].tolist() == [True, False]
    assert matching["target_guard_gt"].tolist() == [1.0, 0.0]
    assert matching["guard_valid"].tolist() == [True, True]
    assert matching["training_labels"].tolist() == [-1, 1]


def test_target_guarded_module_zero_initialization_is_exact_keep():
    module = TargetGuardedSCACRRModule(
        feature_channels=4,
        num_scales=2,
        roi_size=3,
        hidden_dim=8,
        mask_hidden_dim=4,
        dropout=0.0,
    ).train()
    feature = torch.randn(1, 4, 8, 8)
    coarse = torch.full((1, 1, 8, 8), -3.0)
    coarse[0, 0, 2:4, 2:4] = 2.0
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    mask[0, 2:4, 2:4] = True
    box = torch.tensor([[0.0, 2.0, 2.0, 4.0, 4.0]])

    refined, outputs = module(
        feature,
        coarse,
        [coarse, coarse],
        box,
        mask,
        mask,
    )

    assert torch.equal(refined, coarse)
    assert torch.equal(outputs["gates"], torch.zeros(1))
    assert torch.equal(outputs["target_guard"], torch.full((1,), 0.5))


def test_target_guarded_module_preserves_common_e1_initialization_and_rng():
    kwargs = {
        "feature_channels": 4,
        "num_scales": 2,
        "roi_size": 3,
        "hidden_dim": 8,
        "mask_hidden_dim": 4,
        "dropout": 0.0,
        "pooling_mode": "avg_max_topk",
    }
    torch.manual_seed(42)
    base = SCACRRModule(**kwargs)
    base_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(42)
    guarded = TargetGuardedSCACRRModule(**kwargs)
    guarded_rng = torch.random.get_rng_state().clone()

    guarded_state = guarded.state_dict()
    for key, value in base.state_dict().items():
        assert torch.equal(value, guarded_state[key])
    assert torch.equal(base_rng, guarded_rng)


def test_existing_sca_state_schema_has_no_guard_parameters():
    old = SCACRRModule(
        feature_channels=4,
        num_scales=2,
        roi_size=3,
        hidden_dim=8,
        mask_hidden_dim=4,
        dropout=0.0,
    )
    state = old.state_dict()

    assert not any("target_guard" in key for key in state)
    clone = SCACRRModule(
        feature_channels=4,
        num_scales=2,
        roi_size=3,
        hidden_dim=8,
        mask_hidden_dim=4,
        dropout=0.0,
    )
    clone.load_state_dict(state, strict=True)


def test_target_guard_uses_a_distinct_checkpoint_schema():
    assert TARGET_GUARDED_SCA_WEIGHT_SCHEMA_VERSION != SCA_WEIGHT_SCHEMA_VERSION


def test_target_guard_diagnostics_exclude_ambiguous_from_safe_fp_rate():
    records = [
        {
            "guard_valid": True,
            "target_guard_gt": 1.0,
            "target_guard": 0.9,
            "is_ambiguous": False,
            "target_component_eliminated": False,
        },
        {
            "guard_valid": True,
            "target_guard_gt": 0.0,
            "target_guard": 0.9,
            "is_ambiguous": False,
            "target_component_eliminated": False,
        },
        {
            "guard_valid": True,
            "target_guard_gt": 0.0,
            "target_guard": 0.9,
            "is_ambiguous": True,
            "target_component_eliminated": False,
        },
    ]

    diagnostics = _target_guard_diagnostics(
        records, guard_veto_threshold=0.2
    )

    assert diagnostics["target_guard_recall"] == pytest.approx(1.0)
    assert diagnostics["num_false_positive_components"] == 1
    assert diagnostics["target_guard_fp_protection_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("transition_deletions", "candidate_deleted", "guard_recall", "expected"),
    [
        (0, False, 1.0, True),
        (1, False, 1.0, False),
        (0, True, 1.0, False),
        (0, False, 0.98, False),
    ],
)
def test_target_guard_pareto_rejects_each_safety_failure(
    transition_deletions, candidate_deleted, guard_recall, expected
):
    constraints = _target_guard_pareto_constraints(
        {"base_non_inferior": True},
        {"coarse_detected_refined_missed_targets": transition_deletions},
        [{"target_component_eliminated": candidate_deleted}],
        {"target_guard_recall": guard_recall},
    )

    assert all(constraints.values()) is expected


def test_target_guarded_cli_and_model_route_are_distinct(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--enable-ccrr", "--ccrr-version", "v2_target_guarded_component"],
    )
    args = parse_args()
    trainer = Trainer.__new__(Trainer)
    trainer.args = args

    inference = trainer._inference_config()
    model = MSHNet(
        3,
        ccrr_config={
            "variant": args.ccrr_version,
            "feature_channels": 32,
            "num_scales": 4,
        },
    )

    assert inference["ccrr_version"] == "v2_target_guarded_component"
    assert inference["output_probability_threshold"] == pytest.approx(0.5)
    assert inference["guard_veto_threshold"] == pytest.approx(0.2)
    assert isinstance(model.ccrr, TargetGuardedSCACRRModule)


def _write_tiny_dataset(root: Path) -> Path:
    dataset = root / "Tiny-SIRST"
    for folder in ("images", "masks", "img_idx"):
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    for name in ("train_image", "test_image"):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
            dataset / "images" / f"{name}.png"
        )
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[7:9, 7:9] = 255
        Image.fromarray(mask).save(dataset / "masks" / f"{name}.png")
    (dataset / "img_idx" / "train_Tiny-SIRST.txt").write_text(
        "train_image\n", encoding="utf-8"
    )
    (dataset / "img_idx" / "test_Tiny-SIRST.txt").write_text(
        "test_image\n", encoding="utf-8"
    )
    return dataset


def test_target_guarded_trainer_smoke_logs_guard_and_saves_three_weights(
    tmp_path, monkeypatch
):
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
            "v2_target_guarded_component",
            "--candidate-pooling",
            "avg_max_topk",
            "--save-best-pareto",
            "--base-size",
            "16",
            "--crop-size",
            "16",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--max-train-batches",
            "1",
            "--max-test-batches",
            "1",
            "--device",
            "cpu",
        ],
    )
    trainer = Trainer(parse_args())

    train_metrics = trainer.train(0)
    test_metrics = trainer.test(500)

    assert train_metrics["target_guard"] > 0
    assert train_metrics["target_guard_bce"] > 0
    assert "target_guard" in test_metrics["candidate"]
    assert (
        test_metrics["pareto_selection"]["constraints"]["target_deletion_zero"]
        is True
    )
    save_folder = Path(trainer.save_folder)
    for filename in ("best_miou.pkl", "best_pd.pkl", "best_pareto.pkl"):
        assert (save_folder / filename).is_file()
    records = [
        json.loads(line)
        for line in (
            save_folder / "diagnostics" / "test_candidates_epoch_0500.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert "target_guard" in records[0]
    assert "protected_by" in records[0]
