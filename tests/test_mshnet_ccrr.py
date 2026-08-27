"""End-to-end tests for the MSHNet -> candidates -> CCRR interface."""

import pytest
import torch

from model.MSHNet import MSHNet


IMAGE_SIZE = 16
BATCH_SIZE = 2


def _make_model() -> MSHNet:
    return MSHNet(
        input_channels=1,
        ccrr_config={
            "hidden_dim": 8,
            "roi_size": 3,
            "num_classes": 2,
        },
    )


def _make_input() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)


def _explicit_candidates() -> tuple[torch.Tensor, torch.Tensor]:
    boxes = torch.tensor(
        [
            [0, 2, 2, 6, 6],
            [1, 9, 8, 14, 13],
        ],
        dtype=torch.float32,
    )
    masks = torch.zeros(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 8:13, 9:14] = True
    return boxes, masks


def _assert_common_output_contract(outputs, number_of_candidates):
    assert set(outputs) == {
        "feature",
        "multi_scale_logits",
        "coarse_logits",
        "refined_logits",
        "candidate_outputs",
    }
    assert outputs["feature"].shape == (BATCH_SIZE, 16, IMAGE_SIZE, IMAGE_SIZE)
    assert [tuple(logits.shape) for logits in outputs["multi_scale_logits"]] == [
        (BATCH_SIZE, 1, 16, 16),
        (BATCH_SIZE, 1, 8, 8),
        (BATCH_SIZE, 1, 4, 4),
        (BATCH_SIZE, 1, 2, 2),
    ]
    assert outputs["coarse_logits"].shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert outputs["refined_logits"].shape == outputs["coarse_logits"].shape

    candidates = outputs["candidate_outputs"]
    assert candidates is not None
    assert candidates["class_logits"].shape == (number_of_candidates, 2)
    assert candidates["class_probs"].shape == (number_of_candidates, 2)
    assert candidates["target_scores"].shape == (number_of_candidates,)
    assert candidates["clutter_scores"].shape == (number_of_candidates,)
    assert candidates["uncertain_scores"].shape == (number_of_candidates,)
    assert candidates["deltas"].shape == (number_of_candidates,)
    assert candidates["gates"].shape == (number_of_candidates,)
    assert candidates["boxes"].shape == (number_of_candidates, 5)
    assert candidates["candidate_masks"].shape == (
        number_of_candidates,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    assert candidates["batch_indices"].shape == (number_of_candidates,)
    assert candidates["scale_features"].shape == (number_of_candidates, 5)


def _backward_refined_output(model: MSHNet, images: torch.Tensor, outputs) -> None:
    # The refined detection objective alone must train both the base detector
    # and the reliability path when candidates are present.
    outputs["refined_logits"].square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    assert model.conv_init.weight.grad is not None
    assert torch.isfinite(model.conv_init.weight.grad).all()
    assert model.conv_init.weight.grad.abs().sum() > 0


def test_mshnet_explicit_candidates_forward_and_backward():
    torch.manual_seed(123)
    model = _make_model().train()
    images = _make_input()
    boxes, masks = _explicit_candidates()

    outputs = model(
        images,
        warm_flag=True,
        candidate_boxes=boxes,
        candidate_masks=masks,
        enable_ccrr=True,
    )

    _assert_common_output_contract(outputs, number_of_candidates=2)
    candidates = outputs["candidate_outputs"]
    torch.testing.assert_close(candidates["boxes"], boxes)
    assert torch.equal(candidates["candidate_masks"], masks)
    assert candidates["batch_indices"].tolist() == [0, 1]

    correction = (outputs["refined_logits"] - outputs["coarse_logits"]).squeeze(1)
    candidate_union = torch.zeros_like(correction, dtype=torch.bool)
    candidate_union[0] = masks[0]
    candidate_union[1] = masks[1]
    assert torch.count_nonzero(correction[~candidate_union]) == 0
    # Zero-initialized reliability logits are equal target/clutter evidence;
    # the zero-baseline gate therefore preserves the detector exactly.
    assert torch.count_nonzero(correction[candidate_union]) == 0
    assert torch.equal(candidates["gates"], torch.zeros(2))
    assert torch.equal(candidates["deltas"], torch.zeros(2))

    _backward_refined_output(model, images, outputs)
    reliability_weight = model.ccrr.reliability_head.classifier[-1].weight
    assert reliability_weight.grad is not None
    assert torch.isfinite(reliability_weight.grad).all()
    assert reliability_weight.grad.abs().sum() > 0


def test_mshnet_online_candidates_forward_and_backward():
    torch.manual_seed(123)
    model = _make_model().train()
    images = _make_input()

    # Every finite sigmoid probability is > 0, so threshold 0 deterministically
    # produces one full-image connected component for each batch element.
    outputs = model(
        images,
        warm_flag=True,
        enable_ccrr=True,
        candidate_threshold=0.0,
    )

    _assert_common_output_contract(outputs, number_of_candidates=BATCH_SIZE)
    candidates = outputs["candidate_outputs"]
    expected_boxes = torch.tensor(
        [
            [0, 0, 0, IMAGE_SIZE, IMAGE_SIZE],
            [1, 0, 0, IMAGE_SIZE, IMAGE_SIZE],
        ],
        dtype=candidates["boxes"].dtype,
    )
    torch.testing.assert_close(candidates["boxes"], expected_boxes)
    assert candidates["batch_indices"].tolist() == [0, 1]
    assert candidates["candidate_masks"].all()
    # These fields come from online candidate generation and are merged into
    # the CCRR candidate-output record.
    assert candidates["masks"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    assert candidates["areas"].tolist() == [IMAGE_SIZE**2, IMAGE_SIZE**2]
    assert candidates["scores"].shape == (BATCH_SIZE,)

    _backward_refined_output(model, images, outputs)
    reliability_weight = model.ccrr.reliability_head.classifier[-1].weight
    assert reliability_weight.grad is not None
    assert torch.isfinite(reliability_weight.grad).all()
    assert reliability_weight.grad.abs().sum() > 0


def test_mshnet_trained_clutter_head_can_only_suppress_candidates():
    model = _make_model().eval()
    with torch.no_grad():
        final_classifier = model.ccrr.reliability_head.classifier[-1]
        final_classifier.weight.zero_()
        final_classifier.bias.copy_(torch.tensor([-5.0, 5.0]))
    images = _make_input()
    boxes, masks = _explicit_candidates()

    outputs = model(
        images,
        warm_flag=True,
        candidate_boxes=boxes,
        candidate_masks=masks,
        enable_ccrr=True,
    )

    candidates = outputs["candidate_outputs"]
    assert torch.all(candidates["gates"] > 0.99)
    assert torch.all(candidates["deltas"] < -1.4)
    assert torch.all(outputs["refined_logits"] <= outputs["coarse_logits"])
    correction = (outputs["refined_logits"] - outputs["coarse_logits"]).squeeze(1)
    assert torch.count_nonzero(correction[0][~masks[0]]) == 0
    assert torch.count_nonzero(correction[1][~masks[1]]) == 0


@pytest.mark.parametrize("candidate_source", ["explicit", "online"])
def test_mshnet_empty_candidates_are_no_op_and_base_backward_works(candidate_source):
    torch.manual_seed(123)
    model = _make_model().train()
    images = _make_input()

    if candidate_source == "explicit":
        candidate_arguments = {
            "candidate_boxes": torch.empty((0, 5)),
            "candidate_masks": torch.empty(
                (0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool
            ),
        }
    else:
        # A sigmoid of finite logits is strictly below 1, so this yields no
        # online connected components.
        candidate_arguments = {"candidate_threshold": 1.0}

    outputs = model(
        images,
        warm_flag=True,
        enable_ccrr=True,
        **candidate_arguments,
    )

    _assert_common_output_contract(outputs, number_of_candidates=0)
    torch.testing.assert_close(outputs["refined_logits"], outputs["coarse_logits"])
    assert outputs["candidate_outputs"]["boxes"].shape == (0, 5)
    assert outputs["candidate_outputs"]["candidate_masks"].shape == (
        0,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    _backward_refined_output(model, images, outputs)
