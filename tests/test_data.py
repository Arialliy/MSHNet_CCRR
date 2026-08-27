from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from utils.data import IRSTD_Dataset


def _args(dataset_dir):
    return SimpleNamespace(
        dataset_dir=str(dataset_dir),
        crop_size=8,
        base_size=8,
    )


def _write_manifest(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def _write_assets(dataset_dir, names):
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "masks").mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        image = np.full((8, 8, 3), index + 1, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 255
        Image.fromarray(image).save(dataset_dir / "images" / f"{name}.png")
        Image.fromarray(mask).save(dataset_dir / "masks" / f"{name}.png")


def _official_dataset(
    tmp_path,
    train_names=("train-a", "train-b"),
    test_names=("test-a",),
):
    dataset_dir = tmp_path / "IRSTD-1K"
    _write_assets(dataset_dir, [*train_names, *test_names])
    _write_manifest(
        dataset_dir / "img_idx" / "train_IRSTD-1K.txt", train_names
    )
    _write_manifest(dataset_dir / "img_idx" / "test_IRSTD-1K.txt", test_names)
    return dataset_dir


def test_official_train_and_test_manifests_are_loaded_exactly(tmp_path):
    dataset_dir = _official_dataset(tmp_path)

    train = IRSTD_Dataset(_args(dataset_dir), mode="test", split="train")
    test = IRSTD_Dataset(_args(dataset_dir), mode="test", split="test")
    sample = test[0]

    assert train.names == ["train-a", "train-b"]
    assert test.names == ["test-a"]
    assert train.list_dir.endswith("/img_idx/train_IRSTD-1K.txt")
    assert test.list_dir.endswith("/img_idx/test_IRSTD-1K.txt")
    assert set(sample) == {"image", "mask", "name"}
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert sample["name"] == "test-a"
    assert torch.isfinite(sample["image"]).all()
    assert set(sample["mask"].unique().tolist()).issubset({0.0, 1.0})


def test_validation_split_is_not_supported_even_if_a_val_file_exists(tmp_path):
    dataset_dir = _official_dataset(tmp_path)
    _write_manifest(dataset_dir / "img_idx" / "val_IRSTD-1K.txt", ["train-a"])

    with pytest.raises(ValueError, match="Unknown dataset mode: val"):
        IRSTD_Dataset(_args(dataset_dir), mode="val")
    with pytest.raises(ValueError, match="Only official train/test splits"):
        IRSTD_Dataset._find_split_file(dataset_dir, "val")


def test_root_level_manifests_are_never_used_as_fallback(tmp_path):
    dataset_dir = tmp_path / "IRSTD-1K"
    _write_assets(dataset_dir, ["train-a", "test-a"])
    _write_manifest(dataset_dir / "trainval.txt", ["train-a"])
    _write_manifest(dataset_dir / "test.txt", ["test-a"])

    with pytest.raises(
        FileNotFoundError, match=r"img_idx/train_IRSTD-1K\.txt"
    ):
        IRSTD_Dataset(_args(dataset_dir), mode="train")


def test_similarly_named_nonofficial_index_files_are_not_used(tmp_path):
    dataset_dir = tmp_path / "IRSTD-1K"
    _write_assets(dataset_dir, ["train-a", "test-a"])
    _write_manifest(dataset_dir / "img_idx" / "train_demo.txt", ["train-a"])
    _write_manifest(dataset_dir / "img_idx" / "test_demo.txt", ["test-a"])

    with pytest.raises(
        FileNotFoundError, match=r"img_idx/train_IRSTD-1K\.txt"
    ):
        IRSTD_Dataset(_args(dataset_dir), mode="train")


def test_contract_rejects_duplicate_names_after_extension_normalization(tmp_path):
    dataset_dir = _official_dataset(
        tmp_path, train_names=("train-a", "train-a.png"), test_names=("test-a",)
    )

    with pytest.raises(ValueError, match="Duplicate image names in train"):
        IRSTD_Dataset(_args(dataset_dir), mode="train")


def test_contract_rejects_official_train_test_overlap(tmp_path):
    dataset_dir = _official_dataset(
        tmp_path, train_names=("shared",), test_names=("shared.png",)
    )

    with pytest.raises(ValueError, match="split leakage between official train and test"):
        IRSTD_Dataset(_args(dataset_dir), mode="test")


@pytest.mark.parametrize(
    ("missing_kind", "expected_message"),
    (
        ("image", "Missing image referenced by train"),
        ("mask", "Missing mask referenced by train"),
    ),
)
def test_contract_rejects_missing_referenced_assets(
    tmp_path, missing_kind, expected_message
):
    dataset_dir = _official_dataset(tmp_path)
    directory = "images" if missing_kind == "image" else "masks"
    (dataset_dir / directory / "train-a.png").unlink()

    with pytest.raises(FileNotFoundError, match=expected_message):
        IRSTD_Dataset(_args(dataset_dir), mode="train")
