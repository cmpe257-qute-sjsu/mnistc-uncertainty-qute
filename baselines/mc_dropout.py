"""Train and evaluate MC Dropout.

One SharedCNN per run seed, with dropout at the early-exit locations. The
matched run averages 2 stochastic passes and the reference run averages 20.
Both use the same checkpoint. Dropout is off while the validation checkpoint
is chosen, and on only inside predict_proba_mc.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch.nn as nn
from torch.utils.data import DataLoader

from eval.scores import with_confidence
from models import ensemble_proba, predict_proba_mc, save_predictions

from .base import (
    _refuse_existing,
    _require_indexed_eval_loader,
    fit_cnn,
    load_protocol,
    require_loaders,
)


def mc_checkpoint_path(directory, config_version: int, seed: int) -> Path:
    return Path(directory) / f"mc_dropout_config{config_version}_seed{seed}.pt"


def mc_prediction_path(
    seed: int, passes: int, condition: str = "clean", directory="results/preds"
) -> Path:
    return Path(directory) / f"mc_dropout_seed{seed}_{condition}_passes{passes}.npz"


def train_mc_dropout(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    seed: int,
    protocol: Optional[dict] = None,
    device: str = "cpu",
    checkpoint_dir="checkpoints",
    overwrite: bool = False,
    verbose: bool = False,
):
    """Train one dropout CNN. The rate comes from configs/base.yaml."""
    if protocol is None:
        protocol = load_protocol()
    if seed not in set(protocol["training"]["seeds"]):
        raise ValueError(
            f"seed {seed} is not in configs/base.yaml training.seeds {protocol['training']['seeds']}"
        )
    rate = float(protocol["baselines"]["mc_dropout"]["rate"])
    path = mc_checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed)
    return fit_cnn(
        train_loader,
        validation_loader,
        seed,
        protocol,
        method="mc_dropout",
        dropout_rate=rate,
        checkpoint=path,
        device=device,
        overwrite=overwrite,
        verbose=verbose,
        extra={"passes_headline": int(protocol["baselines"]["mc_dropout"]["passes_headline"])},
    )


def evaluate_mc_dropout(
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    passes: int,
    protocol: dict,
    path,
    condition: str = "clean",
    overwrite: bool = False,
    device: str = "cpu",
):
    """Average `passes` stochastic forward passes and save the probabilities."""
    if int(passes) < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    output = Path(path)
    _refuse_existing(output, overwrite)
    _require_indexed_eval_loader(loader, "evaluation loader")
    output.parent.mkdir(parents=True, exist_ok=True)
    mean_probs, sample_ids, member_probs = predict_proba_mc(
        model, loader, passes=int(passes), device=device, seed=seed
    )
    if tuple(member_probs.shape) != (mean_probs.shape[0], int(passes), mean_probs.shape[1]):
        raise ValueError(
            f"expected member probabilities [N, {passes}, C], got {tuple(member_probs.shape)}"
        )
    _confidence, extra = with_confidence(
        mean_probs,
        {
            "passes": int(passes),
            "dropout_rate": float(protocol["baselines"]["mc_dropout"]["rate"]),
            "parameters": int(protocol["model"]["parameters"]),
        },
    )
    metadata = save_predictions(
        output,
        mean_probs,
        sample_ids,
        method="mc_dropout",
        seed=seed,
        condition=condition,
        member_probs=member_probs,
        config_version=int(protocol["config_version"]),
        extra=extra,
    )
    return mean_probs, sample_ids, member_probs, metadata


def evaluate_mc_reference_and_headline(
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    protocol: dict,
    headline_path,
    reference_path,
    condition: str = "clean",
    overwrite: bool = False,
    device: str = "cpu",
):
    """Run the reference passes once. The headline file is the first of those passes."""
    mc = protocol["baselines"]["mc_dropout"]
    if not mc["headline_is_prefix_of_reference"]:
        raise ValueError(
            "baselines.mc_dropout.headline_is_prefix_of_reference is false. "
            "This trainer only saves the headline as the first passes of the reference run."
        )
    headline = int(mc["passes_headline"])
    reference = int(mc["passes_reference"])
    if headline > reference:
        raise ValueError(f"passes_headline {headline} is larger than passes_reference {reference}")
    _refuse_existing(Path(headline_path), overwrite)
    mean_probs, sample_ids, member_probs, reference_meta = evaluate_mc_dropout(
        model,
        loader,
        seed,
        reference,
        protocol,
        reference_path,
        condition=condition,
        overwrite=overwrite,
        device=device,
    )
    prefix = member_probs[:, :headline, :].contiguous()
    headline_probs = ensemble_proba(prefix)
    Path(headline_path).parent.mkdir(parents=True, exist_ok=True)
    _confidence, extra = with_confidence(
        headline_probs,
        {
            "passes": headline,
            "dropout_rate": float(mc["rate"]),
            "parameters": int(protocol["model"]["parameters"]),
            "taken_from": f"first {headline} of {reference} passes",
        },
    )
    headline_meta = save_predictions(
        headline_path,
        headline_probs,
        sample_ids,
        method="mc_dropout",
        seed=seed,
        condition=condition,
        member_probs=prefix,
        config_version=int(protocol["config_version"]),
        extra=extra,
    )
    return headline_probs, mean_probs, sample_ids, headline_meta, reference_meta


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train MC Dropout from configs/base.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    mc = protocol["baselines"]["mc_dropout"]
    train_loader, validation_loader, test_loader = require_loaders(protocol)
    for seed in protocol["training"]["seeds"]:
        model, _history = train_mc_dropout(
            train_loader,
            validation_loader,
            seed=int(seed),
            protocol=protocol,
            device=args.device,
            overwrite=args.overwrite,
            verbose=True,
        )
        headline_path = mc_prediction_path(int(seed), int(mc["passes_headline"]))
        reference_path = mc_prediction_path(int(seed), int(mc["passes_reference"]))
        evaluate_mc_reference_and_headline(
            model,
            test_loader,
            seed=int(seed),
            protocol=protocol,
            headline_path=headline_path,
            reference_path=reference_path,
            overwrite=args.overwrite,
            device=args.device,
        )
        print(f"saved predictions to {headline_path}")
        print(f"saved predictions to {reference_path}")


if __name__ == "__main__":
    main()
