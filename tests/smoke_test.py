"""
Step 1 environment check for the CMPE 257 QUTE project.

Every teammate runs this and pastes the output into the team chat.
If the FINGERPRINT lines match across all three of us, Step 1 is done.

    python tests/smoke_test.py

This checks the environment, not the project's model. It downloads MNIST
once into ./data (which .gitignore should already exclude).
"""

import hashlib
import platform
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

SPLIT_SEED = 0
TRAIN_SIZE = 50000
VAL_SIZE = 10000
BATCH_SIZE = 256


def line(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    line("VERSIONS")
    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"numpy        {np.__version__}")
    print(f"platform     {platform.system()} {platform.machine()}")
    print(f"cuda         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device       {torch.cuda.get_device_name(0)}")

    # ---- fixed split, identical for everyone ----
    line("DATA")
    raw = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    print(f"train+val pool  {len(raw)}")

    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(raw), generator=g)
    train_idx = perm[:TRAIN_SIZE]
    val_idx = perm[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, "splits overlap"
    print(f"train / val     {len(train_idx)} / {len(val_idx)}  (no overlap)")

    # ---- training-set statistics, computed on the training subset only ----
    train_images = raw.data[train_idx].float() / 255.0
    mean = train_images.mean().item()
    std = train_images.std().item()
    print(f"train mean/std  {mean:.6f} / {std:.6f}")

    # ---- one deterministic batch ----
    line("BATCH")
    subset = torch.utils.data.Subset(raw, train_idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
        num_workers=0,
    )
    images, labels = next(iter(loader))
    print(f"images shape    {tuple(images.shape)}")
    print(f"labels shape    {tuple(labels.shape)}")
    print(f"label range     {labels.min().item()}-{labels.max().item()}")

    assert tuple(images.shape) == (BATCH_SIZE, 1, 28, 28), "unexpected image shape"
    assert labels.dtype == torch.int64, "labels should be int64"

    # ---- trivial forward pass, not the project model ----
    line("FORWARD PASS")
    torch.manual_seed(SPLIT_SEED)
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(8 * 14 * 14, 10),
    )
    net.eval()
    with torch.no_grad():
        logits = net(images)
        probs = torch.softmax(logits, dim=1)

    print(f"logits shape    {tuple(logits.shape)}")
    print(f"probs row sum   {probs.sum(dim=1).mean().item():.6f}  (should be 1.000000)")

    assert torch.isfinite(probs).all(), "non-finite probabilities"
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)

    # ---- fingerprints to compare across machines ----
    line("FINGERPRINT  (these must match across all three of us)")
    label_hash = hashlib.sha256(labels.numpy().tobytes()).hexdigest()[:16]
    image_hash = hashlib.sha256(
        np.ascontiguousarray(images.numpy()).tobytes()
    ).hexdigest()[:16]
    print(f"split indices   {hashlib.sha256(train_idx.numpy().tobytes()).hexdigest()[:16]}")
    print(f"batch labels    {label_hash}")
    print(f"batch images    {image_hash}")
    print(f"train mean      {mean:.6f}")

    line("RESULT")
    print("Environment OK. Paste the FINGERPRINT block into the team chat.")
    print("Fingerprints identical  -> Step 1 is done.")
    print("Fingerprints differ     -> stop and compare torchvision versions first.")
    print("\nNote: the forward-pass numbers can differ in the last decimal place")
    print("between CPU and GPU. The fingerprint lines above should not.")


if __name__ == "__main__":
    main()
