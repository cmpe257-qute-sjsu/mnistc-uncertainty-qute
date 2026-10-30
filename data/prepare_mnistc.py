
"""
Step 6: MNIST-C corruption preparation.
CMPE 257 QUTE project.

No model predictions are calculated here.
No MNIST-C data is used for training or tuning.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MNISTC_ROOT = DATA_ROOT / "mnist_c"
MANIFEST_ROOT = DATA_ROOT / "manifests"

CORRUPTIONS = (
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
)


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_protocol():
    with open(ROOT / "configs" / "base.yaml") as file:
        return yaml.safe_load(file)


def load_step5_statistics(protocol):
    path = MANIFEST_ROOT / "mnist_preprocessing.json"

    if not path.exists():
        raise FileNotFoundError(
            "Step 5 preprocessing manifest missing. "
            "Run python3 data/prepare_mnist.py first."
        )

    with open(path) as file:
        stats = json.load(file)

    if stats["config_version"] != protocol["config_version"]:
        raise ValueError("Step 5 config version mismatch")

    if stats["split_fingerprint"] != "aaed0a23690e2e3c":
        raise ValueError("Unexpected MNIST split fingerprint")

    return float(stats["train_mean"]), float(stats["train_std"])


class CorruptedMNIST(Dataset):
    """MNIST-C examples with original sample IDs."""

    def __init__(self, folder, mean, std):
        self.folder = Path(folder)

        self.images_path = self.folder / "test_images.npy"
        self.labels_path = self.folder / "test_labels.npy"

        self.images = np.load(
            self.images_path, mmap_mode="r"
        )
        self.labels = np.load(
            self.labels_path, mmap_mode="r"
        )

        if self.images.shape not in (
            (10000, 28, 28),
            (10000, 28, 28, 1),
        ):
            raise ValueError(
                f"Unexpected image shape: {self.images.shape}"
            )

        if self.labels.size != 10000:
            raise ValueError("Expected 10,000 labels")

        if self.images.dtype != np.uint8:
            raise ValueError("Expected uint8 images")

        labels = self.labels.reshape(-1)

        if not np.all((labels >= 0) & (labels <= 9)):
            raise ValueError("Invalid digit labels")

        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        image = np.asarray(
            self.images[index]
        ).reshape(28, 28)

        # ToTensor converts uint8 [0,255] into [0,1].
        image = Image.fromarray(image, mode="L")
        image = self.transform(image)

        label = int(self.labels[index])

        return image, label, int(index)


class CleanMNIST(Dataset):
    """Clean MNIST test examples with original sample IDs."""

    def __init__(self, mean, std):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        self.dataset = MNIST(
            root=str(DATA_ROOT),
            train=False,
            download=True,
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, int(label), int(index)


def find_dataset_root():
    """Accept either common extraction layout."""
    candidates = (
        MNISTC_ROOT,
        MNISTC_ROOT / "mnist_c",
    )

    for root in candidates:
        if (
            (root / "shot_noise" / "test_images.npy")
            .is_file()
        ):
            return root

    raise FileNotFoundError(
        "MNIST-C not found. Extract mnist_c.zip "
        "under data/mnist_c/"
    )


def build_streams(protocol):
    expected = int(
        protocol["data"]["corruptions_expected"]
    )

    if expected != len(CORRUPTIONS):
        raise ValueError("Corruption count mismatch")

    mean, std = load_step5_statistics(protocol)
    root = find_dataset_root()

    clean = CleanMNIST(mean, std)
    clean_labels = clean.dataset.targets.numpy()

    if len(clean) != 10000:
        raise ValueError("Expected 10,000 clean images")

    available = {
        directory.name
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "test_images.npy").exists()
    }

    if "identity" not in available:
        print("Note: identity folder not found.")

    selected = set(CORRUPTIONS)

    if not selected.issubset(available):
        missing = sorted(selected - available)
        raise ValueError(
            f"Missing MNIST-C corruptions: {missing}"
        )

    extra = available - selected - {"identity"}
    if extra:
        raise ValueError(
            f"Unexpected corruption folders: {sorted(extra)}"
        )

    manifest = {
        "config_version": protocol["config_version"],
        "dataset": "MNIST-C",
        "split_fingerprint": "aaed0a23690e2e3c",
        "normalization": {
            "mean": mean,
            "std": std,
        },
        "clean_count": 10000,
        "corruption_count": len(CORRUPTIONS),
        "identity_excluded": True,
        "stream_order": "clean_then_corrupted",
        "sample_id_policy": (
            "IDs 0..9999 within each segment; "
            "segment identifies clean or corruption"
        ),
        "streams": {},
    }

    streams = {}

    for name in CORRUPTIONS:
        print("Checking:", name)

        folder = root / name

        corrupted = CorruptedMNIST(
            folder, mean, std
        )

        if not np.array_equal(
            corrupted.labels, clean_labels
        ):
            raise AssertionError(
                f"Label ordering mismatch: {name}"
            )

        # Clean first, corruption second.
        # No shuffle or random interleaving.
        stream = ConcatDataset([clean, corrupted])

        assert len(stream) == 20000

        streams[name] = stream

        manifest["streams"][name] = {
            "total_samples": 20000,
            "segments": [
                {
                    "source": "mnist_test",
                    "start": 0,
                    "stop_exclusive": 10000,
                },
                {
                    "source": name,
                    "start": 10000,
                    "stop_exclusive": 20000,
                },
            ],
            "corruption_images_sha256": sha256_file(
                folder / "test_images.npy"
            ),
            "corruption_labels_sha256": sha256_file(
                folder / "test_labels.npy"
            ),
        }

    return streams, manifest


def save_manifest(manifest):
    MANIFEST_ROOT.mkdir(
        parents=True, exist_ok=True
    )

    path = MANIFEST_ROOT / "corruption_streams.json"

    with open(path, "w") as file:
        json.dump(manifest, file, indent=2)

    print("Saved:", path)


def get_corruption_loaders(protocol):
    """Return deterministic evaluation DataLoaders by corruption."""
    streams, manifest = build_streams(protocol)

    batch_size = int(
        protocol["training"]["batch_size"]
    )

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for name, dataset in streams.items()
    }

    return loaders, manifest


def main():
    protocol = load_protocol()
    loaders, manifest = get_corruption_loaders(protocol)

    save_manifest(manifest)

    print("\nMNIST-C verification results")
    print("Corruptions:", len(loaders))

    for name, loader in loaders.items():
        print(name, len(loader.dataset))

        assert len(loader.dataset) == 20000

    # Verify one batch without model inference.
    first_name = CORRUPTIONS[0]
    images, labels, sample_ids = next(
        iter(loaders[first_name])
    )

    assert images.shape[1:] == (1, 28, 28)
    assert len(images) == len(labels) == len(sample_ids)
    assert torch.isfinite(images).all()

    print("\nSTEP 6 DATA PREPARATION SUCCESSFUL")


if __name__ == "__main__":
    main()
