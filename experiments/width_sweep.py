"""Exploratory extension: pick a smaller CNN whose clean accuracy is near the paper's.

Our [8, 16, 32, 32] CNN reaches about 97% clean accuracy; the paper's MNIST model
reports about 91%. To test whether the results depend on base-model strength,
we retrain everything at a narrower width. This script only CHOOSES that width.

Rule (pre-registered in RUNLOG before running):
  - Train BASE, seed 11, under config v5 with only `model.channels` changed.
  - Score each width by final-epoch accuracy on the 10,000-image VALIDATION split.
    Never the test set, never MNIST-C.
  - Pick the width whose validation accuracy is closest to 0.91; on a tie, the
    smaller model.

It never edits configs/base.yaml. It writes:
  results/tables/width_sweep.csv      one row per width
  configs/small_cnn.yaml              v5 with the chosen channels and parameter count

Usage (repo root, after data is prepared as for any training run):
  python -m experiments.width_sweep
  python -m experiments.width_sweep --widths 4,8,16,16 2,4,8,8   # custom candidates
Re-running reuses finished checkpoints in checkpoints/width_sweep/ unless --overwrite.
"""

import argparse
import copy
import csv
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from baselines.base import (
    DEFAULT_PROTOCOL,
    build_cnn_model,
    fit_cnn,
    load_checkpoint,
    load_protocol,
    require_loaders,
    unpack_batch,
)
from models import CNNConfig, SharedCNN

ROOT = Path(__file__).resolve().parents[1]
TARGET_ACCURACY = 0.91
SWEEP_SEED = 11
DEFAULT_WIDTHS = ([8, 16, 32, 32], [6, 12, 24, 24], [4, 8, 16, 16], [3, 6, 12, 12], [2, 4, 8, 8])
SMALL_CONFIG_VERSION = 51  # "v5, small-CNN variant": keeps its checkpoints apart from v5's
CHECKPOINT_DIR = ROOT / "checkpoints" / "width_sweep"
TABLE_PATH = ROOT / "results" / "tables" / "width_sweep.csv"
SMALL_CONFIG_PATH = ROOT / "configs" / "small_cnn.yaml"


def count_parameters(protocol: dict, channels: List[int]) -> int:
    model_cfg = protocol["model"]
    config = CNNConfig(
        channels=list(channels),
        kernel_size=int(model_cfg["kernel_size"]),
        pool_after=list(model_cfg["pool_after"]),
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
        head=str(model_cfg["head"]),
    )
    return sum(p.numel() for p in SharedCNN(config).parameters())


def protocol_with_channels(protocol: dict, channels: List[int]) -> dict:
    """A copy of v5 that differs only in channel widths (and the count that checks them)."""
    variant = copy.deepcopy(protocol)
    variant["model"]["channels"] = list(channels)
    variant["model"]["parameters"] = count_parameters(protocol, channels)
    return variant


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        images, labels = unpack_batch(batch)
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


def width_tag(channels: Sequence[int]) -> str:
    return "-".join(str(c) for c in channels)


def choose(rows: List[dict]) -> dict:
    return min(rows, key=lambda r: (abs(r["val_accuracy"] - TARGET_ACCURACY), r["parameters"]))


def write_small_config(channels: List[int], parameters: int, rows: List[dict]) -> None:
    text = DEFAULT_PROTOCOL.read_text(encoding="utf-8")
    replacements = (
        (r"^config_version: \d+", f"config_version: {SMALL_CONFIG_VERSION}"),
        (r"^(\s*channels:) \[[^\]]*\]", rf"\1 [{', '.join(map(str, channels))}]"),
        (r"^(\s*parameters:) \d+", rf"\g<1> {parameters}"),
    )
    for pattern, replacement in replacements:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(f"could not find {pattern!r} in {DEFAULT_PROTOCOL}")
    sweep = ", ".join(f"{width_tag(r['channels'])}: {r['val_accuracy']:.4f}" for r in rows)
    header = (
        "# SMALL-CNN VARIANT of config v5 (exploratory extension, see RUNLOG).\n"
        "# Identical to configs/base.yaml v5 except config_version and the model\n"
        "# channels/parameters, chosen by experiments/width_sweep.py: BASE seed 11,\n"
        f"# final-epoch validation accuracy closest to {TARGET_ACCURACY}.\n"
        f"# Sweep (val acc): {sweep}\n"
        "# Run it in a separate git worktree with this file copied over configs/base.yaml.\n\n"
    )
    SMALL_CONFIG_PATH.write_text(header + text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--widths", nargs="+", default=None, help="e.g. 4,8,16,16 2,4,8,8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    if int(protocol["config_version"]) != 5:
        raise SystemExit(f"expected configs/base.yaml v5, found v{protocol['config_version']}")
    widths = [[int(c) for c in w.split(",")] for w in args.widths] if args.widths else [list(w) for w in DEFAULT_WIDTHS]
    train_loader, validation_loader, _test_loader = require_loaders(protocol)

    rows = []
    for channels in widths:
        variant = protocol_with_channels(protocol, channels)
        checkpoint = CHECKPOINT_DIR / f"base_w{width_tag(channels)}_seed{SWEEP_SEED}.pt"
        start = time.time()
        if checkpoint.exists() and not args.overwrite:
            model = build_cnn_model(variant, SWEEP_SEED, dropout_rate=0.0)
            payload = load_checkpoint(checkpoint)
            model.load_state_dict(payload["model_state_dict"])
            val_nll = float(payload["val_nll"])
            print(f"[{width_tag(channels)}] reusing {checkpoint}")
        else:
            model, history = fit_cnn(
                train_loader, validation_loader, SWEEP_SEED, variant,
                method="base", dropout_rate=0.0, checkpoint=checkpoint,
                device=args.device, overwrite=args.overwrite, verbose=True,
            )
            val_nll = history[-1]["val_nll"]
        row = {
            "channels": channels,
            "parameters": variant["model"]["parameters"],
            "val_accuracy": accuracy(model.to(args.device), validation_loader, args.device),
            "val_nll": val_nll,
            "minutes": round((time.time() - start) / 60, 1),
        }
        rows.append(row)
        print(f"[{width_tag(channels)}] params={row['parameters']} val_acc={row['val_accuracy']:.4f} val_nll={val_nll:.4f}")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channels", "parameters", "val_accuracy", "val_nll", "minutes", "chosen"])
        chosen = choose(rows)
        for r in rows:
            writer.writerow([width_tag(r["channels"]), r["parameters"], f"{r['val_accuracy']:.4f}",
                             f"{r['val_nll']:.4f}", r["minutes"], r is chosen])

    write_small_config(chosen["channels"], chosen["parameters"], rows)
    print(f"\nChosen: {chosen['channels']} ({chosen['parameters']} params, val acc {chosen['val_accuracy']:.4f})")
    print(f"Wrote {TABLE_PATH} and {SMALL_CONFIG_PATH}")
    if abs(chosen["val_accuracy"] - TARGET_ACCURACY) > 0.02:
        print("WARNING: no width landed within 2 points of 0.91. Record this in RUNLOG and "
              "add a narrower candidate with --widths before moving on.")


if __name__ == "__main__":
    main()
