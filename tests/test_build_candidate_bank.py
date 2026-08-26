import json

import numpy as np
import pytest
import torch

from scripts.build_candidate_bank import (
    SCHEMA_VERSION,
    candidate_records_from_batch,
    decode_mask_rle,
    parse_args,
    select_candidate_scores,
    write_json_atomic,
)


def _required_cli_args(tmp_path):
    return [
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--weight-path",
        str(tmp_path / "weight.pkl"),
        "--output-dir",
        str(tmp_path / "bank"),
    ]


def test_cli_defaults_follow_candidate_bank_design(tmp_path):
    args = parse_args(_required_cli_args(tmp_path))

    assert args.splits == ["train", "test"]
    assert args.candidate_threshold == 0.2
    assert args.hard_negative_threshold == 0.5
    assert args.candidate_score == "coarse_peak"
    assert args.positive_iou == 0.3
    assert args.center_distance == 3.0
    assert args.overwrite is False


def test_cli_rejects_invalid_area_range(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(_required_cli_args(tmp_path) + ["--min-area", "5", "--max-area", "4"])


def test_select_candidate_scores_uses_coarse_peak_by_default_contract():
    candidates = {
        "coarse_peak_scores": torch.tensor([0.91, 0.73]),
        "coarse_scores": torch.tensor([0.81, 0.63]),
        "peak_scores": torch.tensor([0.71, 0.53]),
        "scores": torch.tensor([0.61, 0.43]),
    }

    torch.testing.assert_close(
        select_candidate_scores(candidates, "coarse_peak"),
        torch.tensor([0.91, 0.73]),
    )
    with pytest.raises(ValueError, match="unknown"):
        select_candidate_scores(candidates, "not-a-score")


def test_candidate_records_preserve_batch_identity_scores_and_rle():
    masks = torch.zeros((2, 5, 6), dtype=torch.bool)
    masks[0, 1:3, 2:5] = True
    masks[1, 4, 0:2] = True
    candidates = {
        "masks": masks,
        "boxes": torch.tensor(
            [[0.0, 2.0, 1.0, 5.0, 3.0], [1.0, 0.0, 4.0, 2.0, 5.0]]
        ),
        "batch_indices": torch.tensor([0, 1]),
        "areas": torch.tensor([6, 2]),
        "scores": torch.tensor([0.61, 0.43]),
        "peak_scores": torch.tensor([0.71, 0.53]),
        "coarse_scores": torch.tensor([0.81, 0.63]),
        "coarse_peak_scores": torch.tensor([0.91, 0.73]),
        "scale_responses": torch.tensor([[0.6, 0.7], [0.4, 0.5]]),
        "scale_variance": torch.tensor([0.01, 0.02]),
    }
    matching = {
        "labels": torch.tensor([0, 1]),
        "matched_gt_indices": torch.tensor([2, -1]),
        "max_iou": torch.tensor([0.75, 0.0]),
        "center_match": torch.tensor([True, False]),
        "centroid_distance": torch.tensor([1.5, float("inf")]),
    }

    records, counts = candidate_records_from_batch(
        candidates,
        matching,
        ["image-a", "image-b"],
        image_offset=10,
        candidate_offset=20,
        score_name="coarse_peak",
        hard_negative_threshold=0.5,
    )

    assert counts == [1, 1]
    assert records[0]["image_index"] == 10
    assert records[1]["image_index"] == 11
    assert records[0]["global_candidate_index"] == 20
    assert records[1]["global_candidate_index"] == 21
    assert records[0]["box"] == [2.0, 1.0, 5.0, 3.0]
    assert records[0]["score"] == pytest.approx(0.91)
    assert records[0]["score_type"] == "coarse_peak"
    assert records[1]["label"] == "clutter"
    assert records[1]["is_hard_negative"] is True
    assert records[1]["centroid_distance"] is None
    np.testing.assert_array_equal(
        decode_mask_rle(records[0]["mask_rle"], records[0]["mask_shape"]),
        masks[0].numpy(),
    )


def test_empty_candidate_batch_keeps_all_images(tmp_path):
    candidates = {
        "masks": torch.empty((0, 4, 4), dtype=torch.bool),
        "boxes": torch.empty((0, 5)),
        "batch_indices": torch.empty((0,), dtype=torch.long),
        "areas": torch.empty((0,), dtype=torch.long),
        "scores": torch.empty((0,)),
        "peak_scores": torch.empty((0,)),
        "coarse_scores": torch.empty((0,)),
        "coarse_peak_scores": torch.empty((0,)),
        "scale_responses": torch.empty((0, 4)),
        "scale_variance": torch.empty((0,)),
    }
    matching = {
        "labels": torch.empty((0,), dtype=torch.long),
        "matched_gt_indices": torch.empty((0,), dtype=torch.long),
        "max_iou": torch.empty((0,)),
    }

    records, counts = candidate_records_from_batch(
        candidates,
        matching,
        ["a", "b"],
        image_offset=0,
        candidate_offset=0,
        score_name="coarse_peak",
        hard_negative_threshold=0.5,
    )

    assert records == []
    assert counts == [0, 0]

    output = tmp_path / "manifest.json"
    payload = {"schema_version": SCHEMA_VERSION, "splits": {}}
    write_json_atomic(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "manifest.json.tmp").exists()
