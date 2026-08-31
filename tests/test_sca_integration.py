"""Integration contracts for selective component-aligned (SCA) CCRR."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from main import Trainer, parse_args
from model.MSHNet import MSHNet
from model.ccrr import SCACRRModule
from utils.candidate import generate_component_aligned_candidates
from utils.detection_metric import match_prediction_components_to_gt


HEIGHT = 8
WIDTH = 8


def _sca_module() -> SCACRRModule:
    return SCACRRModule(
        feature_channels=4,
        num_scales=2,
        roi_size=3,
        hidden_dim=8,
        mask_hidden_dim=4,
        dropout=0.0,
        risk_threshold=2.0,
        quality_veto_threshold=0.2,
        remove_threshold=0.45,
        output_threshold=0.5,
    )


def _two_image_inputs():
    feature_map = torch.linspace(
        -1.0, 1.0, 2 * 4 * HEIGHT * WIDTH
    ).reshape(2, 4, HEIGHT, WIDTH)
    coarse_logits = torch.full((2, 1, HEIGHT, WIDTH), -3.0)
    coarse_logits[0, 0, 2:4, 2:4] = 2.0
    coarse_logits[1, 0, 5:7, 5:7] = 2.0
    multi_scale_logits = [
        torch.linspace(-2.0, 2.0, 2 * HEIGHT * WIDTH).reshape(
            2, 1, HEIGHT, WIDTH
        ),
        torch.linspace(2.0, -2.0, 2 * 4 * 4).reshape(2, 1, 4, 4),
    ]
    proposal_masks = torch.zeros((2, HEIGHT, WIDTH), dtype=torch.bool)
    proposal_masks[0, 1:5, 1:5] = True
    proposal_masks[1, 4:8, 4:8] = True
    action_masks = torch.zeros_like(proposal_masks)
    action_masks[0, 2:4, 2:4] = True
    action_masks[1, 5:7, 5:7] = True
    proposal_boxes = torch.tensor(
        [[0, 1, 1, 5, 5], [1, 4, 4, 8, 8]], dtype=torch.float32
    )
    return (
        feature_map,
        coarse_logits,
        multi_scale_logits,
        proposal_boxes,
        proposal_masks,
        action_masks,
    )


def test_mshnet_fuses_three_decoder_levels_before_sca_and_generates_candidates():
    torch.manual_seed(2)
    model = MSHNet(
        1,
        ccrr_config={
            "variant": "v2_selective_component",
            "feature_channels": 32,
            "num_scales": 4,
            "roi_size": 3,
            "hidden_dim": 8,
            "mask_hidden_dim": 4,
            "dropout": 0.0,
            "quality_veto_threshold": 0.2,
        },
    ).eval()
    adapter = model.ccrr_feature_adapter
    assert isinstance(adapter, torch.nn.Sequential)
    assert isinstance(adapter[0], torch.nn.Conv2d)
    assert adapter[0].in_channels == 16 + 32 + 64
    assert adapter[0].out_channels == 32
    assert adapter[0].kernel_size == (1, 1)
    assert isinstance(adapter[1], torch.nn.GroupNorm)
    assert adapter[1].num_groups == 4
    assert isinstance(adapter[2], torch.nn.ReLU)

    # A deterministic positive fused output gives exactly one 0.5-output
    # component per image, while the zero-initialised SCA head must keep it.
    with torch.no_grad():
        model.final.weight.zero_()
        model.final.bias.fill_(1.0)
    adapter_inputs = []
    handle = adapter.register_forward_pre_hook(
        lambda _module, inputs: adapter_inputs.append(inputs[0].shape)
    )
    try:
        with torch.no_grad():
            outputs = model(
                torch.randn(2, 1, 16, 16),
                warm_flag=True,
                enable_ccrr=True,
                candidate_threshold=0.2,
            )
    finally:
        handle.remove()

    assert adapter_inputs == [torch.Size((2, 112, 16, 16))]
    candidates = outputs["candidate_outputs"]
    assert candidates["batch_indices"].tolist() == [0, 1]
    assert candidates["action_component_local_ids"].tolist() == [1, 1]
    assert candidates["action_masks"].shape == (2, 16, 16)
    assert candidates["action_masks"].all()
    assert torch.equal(candidates["candidate_masks"], candidates["action_masks"])
    assert torch.equal(outputs["refined_logits"], outputs["coarse_logits"])


def test_exact_action_components_survive_proposal_area_filter_and_use_fallback():
    coarse_logits = torch.full((1, 1, HEIGHT, WIDTH), -10.0)
    # The diagonal pair is one 8-connected component; the square is another.
    coarse_logits[0, 0, 1, 1] = 2.0
    coarse_logits[0, 0, 2, 2] = 2.0
    coarse_logits[0, 0, 5:7, 5:7] = 2.0
    full_scale = torch.full_like(coarse_logits, 10.0)

    candidates = generate_component_aligned_candidates(
        coarse_logits,
        [full_scale, full_scale],
        proposal_threshold=0.2,
        output_threshold=0.5,
        min_area=1,
        max_area=1,
    )

    exact_positive_partition = coarse_logits.sigmoid()[:, 0] > 0.5
    assert candidates["action_masks"].shape[0] == 2
    assert sorted(candidates["action_areas"].tolist()) == [2, 4]
    assert torch.equal(
        candidates["action_masks"].any(dim=0), exact_positive_partition[0]
    )
    # max_area filters the full-image feature proposal, never the exact
    # actions. Both candidates therefore receive the documented fallback.
    assert candidates["raw_proposal_masks"].shape[0] == 0
    assert candidates["proposal_is_fallback"].tolist() == [True, True]
    assert torch.all(candidates["proposal_areas"] >= candidates["action_areas"])


def test_proposal_and_action_masks_have_separate_feature_roles():
    torch.manual_seed(3)
    module = _sca_module().eval()
    feature_map = torch.arange(4 * HEIGHT * WIDTH, dtype=torch.float32).reshape(
        1, 4, HEIGHT, WIDTH
    ) / 10.0
    coarse_logits = torch.linspace(-2.0, 3.0, HEIGHT * WIDTH).reshape(
        1, 1, HEIGHT, WIDTH
    )
    multi_scale_logits = [
        torch.linspace(-3.0, 3.0, HEIGHT * WIDTH).reshape(
            1, 1, HEIGHT, WIDTH
        ),
        torch.linspace(3.0, -3.0, 16).reshape(1, 1, 4, 4),
    ]
    proposal_boxes = torch.tensor([[0, 0, 0, 8, 8]], dtype=torch.float32)
    first_proposal = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.bool)
    first_proposal[0, :4, :4] = True
    second_proposal = torch.zeros_like(first_proposal)
    second_proposal[0, 4:, 4:] = True
    first_action = torch.zeros_like(first_proposal)
    first_action[0, 2:4, 2:4] = True
    second_action = torch.zeros_like(first_action)
    second_action[0, 4:6, 4:6] = True

    def forward(proposal_masks, action_masks):
        return module(
            feature_map,
            coarse_logits,
            multi_scale_logits,
            proposal_boxes,
            proposal_masks,
            action_masks,
        )[1]

    baseline = forward(first_proposal, first_action)
    changed_proposal = forward(second_proposal, first_action)
    changed_action = forward(first_proposal, second_action)

    assert not torch.equal(
        baseline["proposal_relation_features"],
        changed_proposal["proposal_relation_features"],
    )
    assert torch.equal(
        baseline["action_mask_features"], changed_proposal["action_mask_features"]
    )
    assert torch.equal(
        baseline["proposal_relation_features"],
        changed_action["proposal_relation_features"],
    )
    assert not torch.equal(
        baseline["action_mask_features"], changed_action["action_mask_features"]
    )


@pytest.mark.parametrize("training", [True, False])
def test_zero_initialised_sca_is_exact_keep_for_batch(training):
    module = _sca_module().train(training)
    inputs = _two_image_inputs()

    refined_logits, candidates = module(*inputs)

    coarse_logits = inputs[1]
    assert torch.equal(refined_logits, coarse_logits)
    assert torch.equal(candidates["gates"], torch.zeros(2))
    assert torch.equal(candidates["deltas"], torch.zeros(2))
    assert torch.equal(candidates["class_probs"], torch.full((2, 2), 0.5))
    assert torch.equal(candidates["target_quality"], torch.full((2,), 0.5))
    assert candidates["batch_indices"].tolist() == [0, 1]


def test_forced_sca_action_only_deletes_exact_action_mask():
    module = _sca_module().eval()
    with torch.no_grad():
        module.reliability_head.clutter_head.weight.zero_()
        module.reliability_head.clutter_head.bias.fill_(12.0)
        module.reliability_head.quality_head.weight.zero_()
        module.reliability_head.quality_head.bias.fill_(-12.0)

    feature_map = torch.randn(1, 4, HEIGHT, WIDTH)
    coarse_logits = torch.full((1, 1, HEIGHT, WIDTH), -4.0)
    coarse_logits[0, 0, 2:5, 2:5] = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 2.0], [0.2, 0.4, 0.8]]
    )
    action_mask = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.bool)
    action_mask[0, 2:5, 2:5] = True
    proposal_mask = torch.zeros_like(action_mask)
    proposal_mask[0, 1:6, 1:6] = True
    proposal_box = torch.tensor([[0, 1, 1, 6, 6]], dtype=torch.float32)

    refined_logits, candidates = module(
        feature_map,
        coarse_logits,
        [coarse_logits, coarse_logits],
        proposal_box,
        proposal_mask,
        action_mask,
    )

    assert candidates["gates"].tolist() == [1.0]
    assert candidates["quality_veto"].tolist() == [1.0]
    assert refined_logits[0, 0][action_mask[0]].sigmoid().max() <= 0.45 + 1e-6
    assert torch.equal(
        refined_logits[0, 0][~action_mask[0]],
        coarse_logits[0, 0][~action_mask[0]],
    )
    assert torch.count_nonzero(
        refined_logits[0, 0][proposal_mask[0] & ~action_mask[0]]
        - coarse_logits[0, 0][proposal_mask[0] & ~action_mask[0]]
    ) == 0


def test_sca_empty_candidate_batch_is_shape_safe_exact_keep():
    module = _sca_module().eval()
    inputs = _two_image_inputs()
    feature_map, coarse_logits, multi_scale_logits = inputs[:3]
    empty_masks = torch.empty((0, HEIGHT, WIDTH), dtype=torch.bool)

    refined_logits, candidates = module(
        feature_map,
        coarse_logits,
        multi_scale_logits,
        torch.empty((0, 5)),
        empty_masks,
        empty_masks,
    )

    assert torch.equal(refined_logits, coarse_logits)
    assert candidates["class_logits"].shape == (0, 2)
    assert candidates["proposal_boxes"].shape == (0, 5)
    assert candidates["action_boxes"].shape == (0, 5)
    assert candidates["proposal_masks"].shape == (0, HEIGHT, WIDTH)
    assert candidates["action_masks"].shape == (0, HEIGHT, WIDTH)
    assert candidates["batch_indices"].shape == (0,)


def test_sca_labels_equal_public_component_matcher_for_each_batch():
    action_masks = torch.zeros((4, 12, 12), dtype=torch.bool)
    action_masks[0, 1:3, 1:3] = True  # batch 0 TP by overlap
    action_masks[1, 8:10, 1:3] = True  # batch 0 FP
    action_masks[2, 5:7, 5:7] = True  # batch 1 TP by centroid distance
    action_masks[3, 9:11, 9:11] = True  # batch 1 FP
    batch_indices = torch.tensor([0, 0, 1, 1])
    labels = torch.zeros((2, 1, 12, 12))
    labels[0, 0, 1:3, 1:3] = 1
    labels[1, 0, 5:7, 7:9] = 1
    candidates = {
        "action_masks": action_masks,
        "batch_indices": batch_indices,
        "coarse_peak_scores": torch.tensor([0.8, 0.7, 0.6, 0.9]),
    }
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        ccrr_version="v2_selective_component",
        center_distance=3.0,
        candidate_score="coarse_peak",
        quality_iou_weight=0.5,
        quality_center_sigma=3.0,
    )

    actual = trainer._label_sca_candidates(candidates, labels)

    expected_labels = torch.empty(4, dtype=torch.long)
    expected_ambiguous = torch.empty(4, dtype=torch.bool)
    expected_gt = torch.empty(4, dtype=torch.long)
    for batch_index in range(2):
        positions = torch.nonzero(batch_indices == batch_index).flatten()
        public_match = match_prediction_components_to_gt(
            action_masks[positions],
            labels[batch_index, 0].bool(),
            center_distance=3.0,
        )
        expected_labels[positions] = torch.from_numpy(
            np.where(public_match.is_tp_component, 0, 1)
        )
        expected_ambiguous[positions] = torch.from_numpy(
            public_match.ambiguous_keep
        )
        expected_gt[positions] = torch.from_numpy(public_match.prediction_to_gt)

    assert torch.equal(actual["strict_labels"], expected_labels)
    assert torch.equal(actual["ambiguous_keep"], expected_ambiguous)
    assert torch.equal(actual["matched_gt_indices"], expected_gt)
    assert actual["strict_labels"].tolist() == [0, 1, 0, 1]
    assert actual["target_quality_gt"][0].item() == pytest.approx(1.0)
    assert actual["target_quality_gt"][1].item() == 0.0
    assert actual["target_quality_gt"][3].item() == 0.0


def test_sca_cli_freezes_requested_protocol(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--enable-ccrr", "--ccrr-version", "v2_selective_component"],
    )
    args = parse_args()
    trainer = Trainer.__new__(Trainer)
    trainer.args = args

    inference = trainer._inference_config()
    assert args.epochs == 1000
    assert args.test_start_epoch == 500
    assert args.max_test_batches == 0
    assert args.candidate_threshold == pytest.approx(0.2)
    assert args.hard_negative_threshold == pytest.approx(0.5)
    assert args.ccrr_num_classes == 2
    assert not hasattr(args, "val_start_epoch")
    assert inference["ccrr_version"] == "v2_selective_component"
    assert inference["feature_sources"] == ["x_d0", "x_d1", "x_d2"]
    assert inference["action_component_source"] == "coarse_probability_gt_0.5"
    assert inference["output_probability_threshold"] == pytest.approx(0.5)
    assert inference["proposal_mask_use"] == "feature_encoding_only"
    assert inference["action_mask_use"] == "label_suppression_and_evaluation"
