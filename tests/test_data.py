
"""Unit tests for the QUTE MNIST data interface.

These tests do not download MNIST.
"""

import hashlib

import torch
from torch.utils.data import Dataset, Subset

from data import IndexedDataset, get_loaders
from data.prepare_mnist import make_split


class FakeDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), index % 10


def test_data_interface_exists():
    assert callable(get_loaders)


def test_canonical_split():
    protocol = {
        "data": {
            "train_size": 50000,
            "val_size": 10000,
            "split_seed": 0,
        }
    }

    train_idx, val_idx = make_split(protocol, 60000)

    assert len(train_idx) == 50000
    assert len(val_idx) == 10000
    assert set(train_idx.tolist()).isdisjoint(
        set(val_idx.tolist())
    )

    split_hash = hashlib.sha256(
        train_idx.numpy().tobytes()
    ).hexdigest()[:16]

    assert split_hash == "aaed0a23690e2e3c"


def test_indexed_dataset():
    dataset = FakeDataset()

    indexed = IndexedDataset(
        dataset,
        ids=list(range(100, 110))
    )

    image, label, sample_id = indexed[0]

    assert image.shape == (1, 28, 28)
    assert label == 0
    assert sample_id == 100


def test_validation_original_indices():
    dataset = FakeDataset()

    val_indices = [7, 2, 9]
    subset = Subset(dataset, val_indices)

    indexed = IndexedDataset(
        subset,
        ids=val_indices
    )

    assert indexed[0][2] == 7
    assert indexed[1][2] == 2
    assert indexed[2][2] == 9
