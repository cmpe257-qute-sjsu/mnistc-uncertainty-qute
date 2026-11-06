
"""Offline tests for Step 6 MNIST-C preparation."""

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

from data.prepare_mnistc import (
    CORRUPTIONS,
    CorruptedMNIST,
)


class FakeClean(Dataset):
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), 0, index


def test_corruption_count():
    assert len(CORRUPTIONS) == 15
    assert len(set(CORRUPTIONS)) == 15
    assert "identity" not in CORRUPTIONS


@pytest.fixture
def sample_corruption(tmp_path):
    folder = tmp_path / "shot_noise"
    folder.mkdir()

    images = np.zeros(
        (10000, 28, 28, 1), dtype=np.uint8
    )
    labels = np.zeros(10000, dtype=np.uint8)

    np.save(folder / "test_images.npy", images)
    np.save(folder / "test_labels.npy", labels)

    return folder


def test_corrupted_dataset(sample_corruption):
    dataset = CorruptedMNIST(
        sample_corruption,
        mean=0.130731,
        std=0.308,
    )

    assert len(dataset) == 10000

    image, label, sample_id = dataset[42]

    assert image.shape == (1, 28, 28)
    assert label == 0
    assert sample_id == 42
    assert torch.isfinite(image).all()


def test_stream_order(sample_corruption):
    clean = FakeClean()
    corrupted = CorruptedMNIST(
        sample_corruption,
        mean=0.130731,
        std=0.308,
    )

    stream = ConcatDataset([clean, corrupted])

    assert len(stream) == 20000
    assert stream[0][2] == 0
    assert stream[9999][2] == 9999
    assert stream[10000][2] == 0
    assert stream[19999][2] == 9999
def test_condition_loader_contract(monkeypatch):
    from data import prepare_mnistc

    class FakeDataset(Dataset):
        def __len__(self):
            return 10000

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), 0, index

    conditions = {
        "clean": FakeDataset(),
        **{
            name: FakeDataset()
            for name in prepare_mnistc.CORRUPTIONS
        },
    }

    monkeypatch.setattr(
        prepare_mnistc,
        "build_streams",
        lambda protocol: (conditions, {})
    )

    loaders, manifest = (
        prepare_mnistc.get_corruption_loaders(
            {"training": {"batch_size": 256}}
        )
    )

    assert len(loaders) == 16
    assert set(loaders) == set(conditions)

    for loader in loaders.values():
        assert len(loader.dataset) == 10000
        assert len(next(iter(loader))) == 3