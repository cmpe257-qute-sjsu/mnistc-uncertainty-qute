
"""MNIST preparation for the CMPE 257 QUTE project."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MANIFEST_ROOT = DATA_ROOT / "manifests"

REFERENCE = {
    "split": "aaed0a23690e2e3c",
    "labels": "cae9caa451436cf3",
    "images": "521727b4f446b293",
    "mean": 0.130731,
}


def fingerprint(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


class IndexedDataset(Dataset):
    """Return (image, label, original sample_id)."""

    def __init__(self, dataset, ids):
        if len(dataset) != len(ids):
            raise ValueError("IDs must match dataset length")
        self.dataset = dataset
        self.ids = list(ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, int(self.ids[index])


def make_split(protocol, total):
    """Match tests/smoke_test.py exactly."""
    cfg = protocol["data"]
    train_size = int(cfg["train_size"])
    val_size = int(cfg["val_size"])
    seed = int(cfg["split_seed"])

    if train_size + val_size != total:
        raise ValueError("Split sizes do not cover MNIST")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)

    train_idx = permutation[:train_size]
    val_idx = permutation[train_size:train_size + val_size]

    assert len(torch.unique(permutation)) == total

    return train_idx, val_idx


def verify_reference(raw, train_idx, protocol):
    """Reproduce canonical smoke-test fingerprints."""
    split_hash = fingerprint(train_idx.numpy())

    train_images = raw.data[train_idx].float() / 255.0
    mean = train_images.mean().item()
    std = train_images.std().item()

    seed = int(protocol["data"]["split_seed"])
    batch_size = int(protocol["training"]["batch_size"])

    subset = Subset(raw, train_idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    images, labels = next(iter(loader))

    label_hash = fingerprint(labels.numpy())
    image_hash = fingerprint(images.numpy())

    print("\nFINGERPRINT")
    print("split indices  ", split_hash)
    print("batch labels   ", label_hash)
    print("batch images   ", image_hash)
    print("train mean     ", f"{mean:.6f}")
    print("train std      ", f"{std:.6f}")

    if split_hash != REFERENCE["split"]:
        raise AssertionError("Split fingerprint mismatch")

    if label_hash != REFERENCE["labels"]:
        raise AssertionError("Batch label fingerprint mismatch")

    if image_hash != REFERENCE["images"]:
        raise AssertionError("Batch image fingerprint mismatch")

    if abs(mean - REFERENCE["mean"]) >= 1e-6:
        raise AssertionError("Training mean mismatch")

    print("All reference checks passed.")

    return mean, std, split_hash


def save_manifests(protocol, train_idx, val_idx,
                   mean, std, split_hash):
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    split_path = MANIFEST_ROOT / "mnist_split.npz"

    if split_path.exists():
        with np.load(split_path) as existing:
            if not np.array_equal(
                existing["train_indices"], train_idx.numpy()
            ):
                raise ValueError("Saved training split differs")

            if not np.array_equal(
                existing["val_indices"], val_idx.numpy()
            ):
                raise ValueError("Saved validation split differs")
    else:
        np.savez(
            split_path,
            train_indices=train_idx.numpy(),
            val_indices=val_idx.numpy(),
        )

    manifest = {
        "config_version": protocol["config_version"],
        "dataset": "MNIST",
        "split_seed": protocol["data"]["split_seed"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "split_fingerprint": split_hash,
        "train_mean": mean,
        "train_std": std,
        "batch_size": protocol["training"]["batch_size"],
        "augmentation": protocol["data"]["augmentation"],
    }

    with open(
        MANIFEST_ROOT / "mnist_preprocessing.json", "w"
    ) as f:
        json.dump(manifest, f, indent=2)

    print("Manifests saved to:", MANIFEST_ROOT)


def get_loaders(protocol):
    """
    Return (train_loader, validation_loader, test_loader).

    protocol: dictionary loaded from configs/base.yaml

    Training batches:   (image, label)
    Validation batches: (image, label, sample_id)
    Test batches:       (image, label, sample_id)
    """
    if not isinstance(protocol, dict):
        raise TypeError("protocol must be a configuration dictionary")

    if protocol.get("config_version") != 2:
        raise ValueError(
            "Expected frozen protocol version 2; "
            "review compatibility before changing this."
        )

    cfg = protocol["data"]
    batch_size = int(protocol["training"]["batch_size"])
    seed = int(cfg["split_seed"])

    if cfg["dataset"] != "mnist":
        raise ValueError("Expected MNIST dataset")

    if cfg["normalize_from"] != "train_only":
        raise ValueError("Expected training-only normalization")

    # Raw data for canonical split and fingerprint checks.
    raw = MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    if len(raw) != 60000:
        raise ValueError("Unexpected MNIST training size")

    train_idx, val_idx = make_split(protocol, len(raw))

    mean, std, split_hash = verify_reference(
        raw, train_idx, protocol
    )

    aug = cfg["augmentation"]

    train_transform = transforms.Compose([
        transforms.RandomRotation(
            aug["random_rotation_degrees"]
        ),
        transforms.RandomCrop(
            28,
            padding=aug["random_crop_padding"]
        ),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    train_data = MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=train_transform,
    )

    eval_data = MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_data = MNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_subset = Subset(
        train_data, train_idx.tolist()
    )

    val_subset = IndexedDataset(
        Subset(eval_data, val_idx.tolist()),
        ids=val_idx.tolist(),
    )

    test_subset = IndexedDataset(
        test_data,
        ids=list(range(len(test_data))),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    save_manifests(
        protocol, train_idx, val_idx,
        mean, std, split_hash,
    )

    return train_loader, val_loader, test_loader


def main():
    import yaml

    with open(ROOT / "configs" / "base.yaml") as f:
        protocol = yaml.safe_load(f)

    train_loader, val_loader, test_loader = (
        get_loaders(protocol)
    )

    print("\nDATASET COUNTS")
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert len(train_batch) == 2
    assert len(val_batch) == 3
    assert len(test_batch) == 3

    print("Train batch shape:", train_batch[0].shape)
    print("Validation batch items:", len(val_batch))
    print("Test batch items:", len(test_batch))
    print("\nSTEP 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
