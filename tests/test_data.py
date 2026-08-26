from types import SimpleNamespace

import torch

from utils.data import IRSTD_Dataset


def _args():
    return SimpleNamespace(
        dataset_dir="datasets/IRSTD-1K",
        crop_size=256,
        base_size=256,
    )


def test_dataset_returns_traceable_dictionary():
    dataset = IRSTD_Dataset(_args(), mode="val")
    sample = dataset[0]

    assert set(sample) == {"image", "mask", "name"}
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["name"]
    assert torch.isfinite(sample["image"]).all()
    assert set(sample["mask"].unique().tolist()).issubset({0.0, 1.0})
    assert "/img_idx/test_IRSTD-1K.txt" in dataset.list_dir


def test_training_split_can_be_read_without_augmentation_for_diagnosis():
    dataset = IRSTD_Dataset(_args(), mode="val", split="train")

    assert len(dataset) == 800
    assert dataset[0]["image"].shape[-2:] == (256, 256)
    assert "/img_idx/train_IRSTD-1K.txt" in dataset.list_dir
