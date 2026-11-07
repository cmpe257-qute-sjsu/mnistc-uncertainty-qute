"""Predict every MNIST / MNIST-C condition with the saved QUTE checkpoints.

    python -m qute.evaluate                    # QUTE and the ablation, all seeds
    python -m qute.evaluate --only qute
    python -m qute.evaluate --only ablation

No training happens here. Each checkpoint written by qute.train is rebuilt
from configs/base.yaml, loaded, and run once on each of the 16 condition
loaders from data.prepare_mnistc (clean + 15 corruptions). One prediction file
is written per method, seed and condition:

    results/preds/qute_seed11_fog.npz
    results/preds/qute_no_transfer_seed11_fog.npz

Files that already exist are kept, not overwritten. That includes the
*_clean.npz written at the end of training, which was produced from the same
test images with the same normalization. Pass --overwrite to redo everything.

Metrics are not computed here. Every number in the report is computed later
from these files, the same way for all four methods.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from baselines.base import load_checkpoint, load_protocol
from models.qute import QUTEModel

from .train import (
    build_qute_from_protocol,
    evaluate_qute,
    method_name,
    qute_checkpoint_path,
    qute_prediction_path,
)

EXPECTED_CONDITIONS = 16
EXPECTED_SAMPLES = 10000
VARIANTS = {"qute": (True,), "ablation": (False,), "both": (True, False)}


def load_qute_checkpoint(
    protocol: dict,
    seed: int,
    transfer: bool,
    checkpoint_dir="checkpoints",
    device: str = "cpu",
) -> Tuple[QUTEModel, dict]:
    """Rebuild a QUTE model from the protocol and load its saved weights.

    Refuses a checkpoint that was trained under a different config version or
    checkpoint rule, so predictions can never silently mix protocols.
    """
    path = qute_checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed, transfer)
    if not path.exists():
        command = "python -m qute.train" + ("" if transfer else " --no-transfer")
        raise FileNotFoundError(f"{path} not found. Train it first with: {command}")

    payload = load_checkpoint(path, map_location="cpu")

    expected = method_name(transfer)
    if payload.get("method") != expected:
        raise ValueError(f"{path} holds method {payload.get('method')!r}, expected {expected!r}")
    if int(payload.get("config_version", -1)) != int(protocol["config_version"]):
        raise ValueError(
            f"{path} was trained under config v{payload.get('config_version')}, "
            f"configs/base.yaml is v{protocol['config_version']}. Retrain before evaluating."
        )
    rule = protocol["training"]["checkpoint_selection"]
    if payload.get("checkpoint_selection") != rule:
        raise ValueError(
            f"{path} was saved with checkpoint_selection={payload.get('checkpoint_selection')!r}, "
            f"configs/base.yaml says {rule!r}"
        )
    if rule == "final_epoch" and int(payload["epoch"]) != int(protocol["training"]["epochs"]):
        raise ValueError(
            f"{path} saved epoch {payload['epoch']}, but final_epoch should be "
            f"epoch {protocol['training']['epochs']}"
        )

    model = build_qute_from_protocol(protocol, seed)
    model.load_state_dict(payload["model_state_dict"])
    model.freeze_base(False)
    return model.to(device).eval(), payload


def check_condition_loaders(loaders: Dict[str, DataLoader], expected_conditions=EXPECTED_CONDITIONS,
                            expected_samples=EXPECTED_SAMPLES) -> None:
    """The 16 loaders from data.prepare_mnistc, each with 10,000 images."""
    if "clean" not in loaders:
        raise ValueError(f"no 'clean' condition among {sorted(loaders)}")
    if len(loaders) != expected_conditions:
        raise ValueError(f"expected {expected_conditions} conditions, got {len(loaders)}: {sorted(loaders)}")
    for name, loader in loaders.items():
        if len(loader.dataset) != expected_samples:
            raise ValueError(f"{name} has {len(loader.dataset)} images, expected {expected_samples}")


def evaluate_conditions(
    model: QUTEModel,
    loaders: Dict[str, DataLoader],
    seed: int,
    protocol: dict,
    transfer: bool,
    preds_dir="results/preds",
    overwrite: bool = False,
    device: str = "cpu",
) -> Tuple[List[Path], List[Path]]:
    """Write one prediction file per condition. Returns (written, kept_existing)."""
    written, kept = [], []
    for condition, loader in loaders.items():
        path = qute_prediction_path(seed, condition, transfer, directory=preds_dir)
        if path.exists() and not overwrite:
            kept.append(path)
            continue
        probs, sample_ids, _members, _meta = evaluate_qute(
            model,
            loader,
            seed=seed,
            protocol=protocol,
            path=path,
            condition=condition,
            transfer=transfer,
            overwrite=overwrite,
            device=device,
        )
        n = len(loader.dataset)
        if tuple(probs.shape) != (n, 10):
            raise RuntimeError(f"{condition}: expected probabilities [{n}, 10], got {tuple(probs.shape)}")
        if len(set(sample_ids.tolist())) != n:
            raise RuntimeError(f"{condition}: sample ids are not unique")
        written.append(path)
    return written, kept


def run(
    protocol: dict,
    loaders: Dict[str, DataLoader],
    variants: Iterable[bool] = (True, False),
    checkpoint_dir="checkpoints",
    preds_dir="results/preds",
    overwrite: bool = False,
    device: str = "cpu",
) -> Dict[Tuple[str, int], Tuple[List[Path], List[Path]]]:
    """Every requested variant x every seed x every condition."""
    results = {}
    for transfer in variants:
        for seed in protocol["training"]["seeds"]:
            seed = int(seed)
            model, payload = load_qute_checkpoint(
                protocol, seed, transfer, checkpoint_dir=checkpoint_dir, device=device
            )
            written, kept = evaluate_conditions(
                model, loaders, seed, protocol, transfer,
                preds_dir=preds_dir, overwrite=overwrite, device=device,
            )
            name = method_name(transfer)
            print(f"{name} seed {seed} (epoch {payload['epoch']}): "
                  f"wrote {len(written)}, kept {len(kept)} existing")
            results[(name, seed)] = (written, kept)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Predict all 16 conditions from saved QUTE checkpoints")
    parser.add_argument("--only", choices=sorted(VARIANTS), default="both",
                        help="qute, ablation (no transfer), or both (default)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo prediction files that already exist")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--preds-dir", default="results/preds")
    args = parser.parse_args(argv)

    # Imported here so the module can be tested without MNIST-C on disk.
    from data.prepare_mnistc import get_corruption_loaders

    protocol = load_protocol()
    loaders, _manifest = get_corruption_loaders(protocol)
    check_condition_loaders(loaders)

    results = run(
        protocol,
        loaders,
        variants=VARIANTS[args.only],
        checkpoint_dir=args.checkpoint_dir,
        preds_dir=args.preds_dir,
        overwrite=args.overwrite,
        device=args.device,
    )
    total = sum(len(written) for written, _kept in results.values())
    print(f"\nDone. {total} prediction files written to {args.preds_dir}")


if __name__ == "__main__":
    main()
