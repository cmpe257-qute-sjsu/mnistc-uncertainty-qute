"""Train and evaluate the deep ensemble.

Each run seed trains members_reference independent CNNs. The headline result
averages the first members_headline of those same models. configs/base.yaml
sets that subset relationship, so this module does not train a second pair.

The yaml lists three run seeds and five reference members, and it does not
list a fifth initialization seed. Member k of run R is initialized with
R * 100 + k, counting k from 1. Run 11 therefore uses 1101 through 1105.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval.scores import with_confidence
from models import ensemble_proba, predict_proba, save_predictions, stack_members

from .base import (
    _refuse_existing,
    _require_indexed_eval_loader,
    fit_cnn,
    load_protocol,
    require_loaders,
)


def member_init_seeds(run_seed: int, n_members: int) -> List[int]:
    """Return n_members distinct initialization seeds for one ensemble run."""
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    seeds = [int(run_seed) * 100 + member for member in range(1, int(n_members) + 1)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"member seeds are not unique: {seeds}")
    return seeds


def ensemble_checkpoint_path(
    directory, config_version: int, run_seed: int, member_index: int, member_seed: int
) -> Path:
    name = (
        f"deep_ensemble_config{config_version}_run{run_seed}"
        f"_member{member_index}_seed{member_seed}.pt"
    )
    return Path(directory) / name


def ensemble_prediction_path(
    run_seed: int, n_members: int, condition: str = "clean", directory="results/preds"
) -> Path:
    return Path(directory) / f"deep_ensemble_seed{run_seed}_{condition}_members{n_members}.npz"


def _ensemble_settings(protocol: dict) -> tuple:
    block = protocol["baselines"]["deep_ensemble"]
    if not block["headline_is_subset_of_reference"]:
        raise ValueError(
            "baselines.deep_ensemble.headline_is_subset_of_reference is false. "
            "This trainer only implements the 2-member result as a subset of the 5-member run."
        )
    headline = int(block["members_headline"])
    reference = int(block["members_reference"])
    if headline < 1 or reference < headline:
        raise ValueError(
            f"members_headline={headline} must be at least 1 and no larger than "
            f"members_reference={reference}"
        )
    return headline, reference


def train_deep_ensemble(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    run_seed: int,
    protocol: Optional[dict] = None,
    device: str = "cpu",
    checkpoint_dir="checkpoints",
    overwrite: bool = False,
    verbose: bool = False,
):
    """Train the reference members for one run. Headline members are a prefix."""
    if protocol is None:
        protocol = load_protocol()
    if run_seed not in set(protocol["training"]["seeds"]):
        raise ValueError(
            f"run seed {run_seed} is not in configs/base.yaml training.seeds "
            f"{protocol['training']['seeds']}"
        )
    headline, reference = _ensemble_settings(protocol)
    init_seeds = member_init_seeds(run_seed, reference)
    models = []
    histories = []
    for member_index, member_seed in enumerate(init_seeds, start=1):
        path = ensemble_checkpoint_path(
            checkpoint_dir,
            int(protocol["config_version"]),
            int(run_seed),
            member_index,
            member_seed,
        )
        model, history = fit_cnn(
            train_loader,
            validation_loader,
            member_seed,
            protocol,
            method="deep_ensemble",
            dropout_rate=0.0,
            checkpoint=path,
            device=device,
            overwrite=overwrite,
            verbose=verbose,
            extra={
                "run_seed": int(run_seed),
                "member_index": member_index,
                "member_seed": member_seed,
                "members_headline": headline,
                "members_reference": reference,
            },
        )
        models.append(model)
        histories.append(history)
    return models, histories, init_seeds


def collect_member_probabilities(models: Sequence[nn.Module], loader: DataLoader, device: str = "cpu"):
    """One deterministic pass per member. All members must see the same sample ids."""
    _require_indexed_eval_loader(loader, "evaluation loader")
    per_member = []
    sample_ids = None
    for model in models:
        probs, ids = predict_proba(model, loader, device=device)
        if sample_ids is None:
            sample_ids = ids
        elif not torch.equal(sample_ids, ids):
            raise RuntimeError(
                "ensemble members saw different sample ids. The evaluation loader must use shuffle=False."
            )
        per_member.append(probs)
    stacked = stack_members(per_member)
    return stacked, sample_ids


def evaluate_deep_ensemble(
    models: Sequence[nn.Module],
    loader: DataLoader,
    run_seed: int,
    protocol: dict,
    headline_path,
    reference_path,
    condition: str = "clean",
    overwrite: bool = False,
    device: str = "cpu",
):
    """Save the headline average and the reference average from one member stack."""
    headline, reference = _ensemble_settings(protocol)
    if len(models) != reference:
        raise ValueError(f"expected {reference} members, got {len(models)}")
    for path in (headline_path, reference_path):
        _refuse_existing(Path(path), overwrite)

    stacked, sample_ids = collect_member_probabilities(models, loader, device=device)
    if stacked.shape[1] != reference:
        raise ValueError(f"expected member axis {reference}, got {stacked.shape[1]}")
    headline_members = stacked[:, :headline, :].contiguous()
    headline_probs = ensemble_proba(headline_members)
    reference_probs = ensemble_proba(stacked)
    config_version = int(protocol["config_version"])
    common = {
        "run_seed": int(run_seed),
        "parameters_per_member": int(protocol["model"]["parameters"]),
        "dropout_rate": 0.0,
        "member_seeds": member_init_seeds(run_seed, reference),
    }
    Path(headline_path).parent.mkdir(parents=True, exist_ok=True)
    _headline_confidence, headline_extra = with_confidence(
        headline_probs,
        {**common, "n_members_reported": headline, "role": "headline"},
    )
    _reference_confidence, reference_extra = with_confidence(
        reference_probs,
        {**common, "n_members_reported": reference, "role": "reference"},
    )
    headline_meta = save_predictions(
        headline_path,
        headline_probs,
        sample_ids,
        method="deep_ensemble",
        seed=int(run_seed),
        condition=condition,
        member_probs=headline_members,
        config_version=config_version,
        extra=headline_extra,
    )
    reference_meta = save_predictions(
        reference_path,
        reference_probs,
        sample_ids,
        method="deep_ensemble",
        seed=int(run_seed),
        condition=condition,
        member_probs=stacked,
        config_version=config_version,
        extra=reference_extra,
    )
    return headline_probs, reference_probs, sample_ids, headline_meta, reference_meta


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train deep ensembles from configs/base.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    headline, reference = _ensemble_settings(protocol)
    train_loader, validation_loader, test_loader = require_loaders(protocol)
    for run_seed in protocol["training"]["seeds"]:
        models, _histories, _seeds = train_deep_ensemble(
            train_loader,
            validation_loader,
            run_seed=int(run_seed),
            protocol=protocol,
            device=args.device,
            overwrite=args.overwrite,
            verbose=True,
        )
        headline_path = ensemble_prediction_path(int(run_seed), headline)
        reference_path = ensemble_prediction_path(int(run_seed), reference)
        evaluate_deep_ensemble(
            models,
            test_loader,
            run_seed=int(run_seed),
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
