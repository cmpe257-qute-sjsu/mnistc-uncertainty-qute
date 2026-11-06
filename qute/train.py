"""Train and evaluate the QUTE model, and its no-transfer ablation.

Owner: Smitha. Deliberately separate from baselines/fit_cnn, because QUTE's
batch loop differs in three ways that would make the shared trainer worse for
everyone if folded in:

  * weights are copied from each early exit into its output block before every
    batch, including through the frozen final epochs
  * the loss has three weighted parts instead of one cross-entropy
  * the base network freezes for the last 10% of epochs while the output
    blocks keep training

Everything else is borrowed from baselines.base so the comparison stays
honest: the same protocol validation, the same optimizer and schedule, the
same minibatch order, the same checkpoint rule, the same prediction schema and
the same confidence definition.

The ablation is this same trainer with transfer=False. Nothing else changes --
same architecture, same seeds, same data order -- so a difference in results
can only come from the mechanism.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval.scores import with_confidence
from models import ensemble_proba, save_predictions
from models.qute import (
    QUTEModel, build_qute, ev_weights, qute_config_from_protocol, qute_loss, tau_weights,
)
from models.shared_cnn import CNNConfig, build_model

from baselines.base import (
    CHECKPOINT_RULES,
    _refuse_existing,
    _require_indexed_eval_loader,
    assert_train_batch_size,
    fix_data_order,
    load_protocol,
    pin_training_randomness,
    require_loaders,
    set_seed,
    unpack_batch,
)

METHOD = "qute"
ABLATION_METHOD = "qute_no_transfer"
NLL_EPS = 1e-12


def method_name(transfer: bool) -> str:
    return METHOD if transfer else ABLATION_METHOD


def qute_checkpoint_path(directory, config_version: int, seed: int, transfer: bool = True) -> Path:
    return Path(directory) / f"{method_name(transfer)}_config{config_version}_seed{seed}.pt"


def qute_prediction_path(
    seed: int, condition: str = "clean", transfer: bool = True, directory="results/preds"
) -> Path:
    return Path(directory) / f"{method_name(transfer)}_seed{seed}_{condition}.npz"


def build_qute_from_protocol(protocol: dict, seed: int) -> QUTEModel:
    """Build the QUTE model from configs/base.yaml, with the seed applied.

    Mirrors baselines.base.build_cnn_model: same seeding, same architecture
    keys, and the same parameter-count guard on the base network.
    """
    model_cfg = protocol["model"]
    cnn_config = CNNConfig(
        channels=list(model_cfg["channels"]),
        kernel_size=int(model_cfg["kernel_size"]),
        pool_after=list(model_cfg["pool_after"]),
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
        dropout_rate=0.0,                     # QUTE's base never carries dropout
        head=str(model_cfg["head"]),
    )
    set_seed(seed)
    model = build_qute(cnn_config=cnn_config,
                       qute_config=qute_config_from_protocol(protocol),
                       seed=seed)

    base_params = sum(p.numel() for p in model.base.parameters())
    expected = int(model_cfg["parameters"])
    if base_params != expected:
        raise RuntimeError(
            f"base parameter count is {base_params}, configs/base.yaml "
            f"model.parameters is {expected}"
        )
    model.assert_no_aliasing()
    return model


@torch.no_grad()
def deployed_nll(model: QUTEModel, loader: DataLoader, device: torch.device) -> float:
    """Mean NLL of the DEPLOYED prediction, logged every epoch (and used for
    selection under min_val_nll).

    The deployed prediction is the mean of the output blocks' probabilities, so
    this is -log(mean_prob[true class]) -- not cross-entropy on any single set
    of logits.
    """
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        images, labels = unpack_batch(batch)
        images, labels = images.to(device), labels.to(device)
        probs = torch.softmax(model.member_logits(images), dim=2).mean(dim=1)
        true_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        total += float(-torch.log(true_prob.clamp_min(NLL_EPS)).sum())
        count += labels.shape[0]
    if count == 0:
        raise RuntimeError("validation loader produced no examples")
    return total / count


def train_qute(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    seed: int,
    protocol: Optional[dict] = None,
    transfer: bool = True,
    device: str = "cpu",
    checkpoint_dir="checkpoints",
    overwrite: bool = False,
    verbose: bool = False,
) -> Tuple[QUTEModel, List[dict]]:
    """Train one QUTE model. transfer=False is the ablation.

    Per batch, in this order:
        copy theta -> phi (unless ablating), zero grads, forward every exit,
        weighted three-part loss, backward, step.

    From the first frozen epoch the base network stops updating while transfer
    continues, so the output blocks train in isolation against early-exit
    weights that are themselves still moving.
    """
    if protocol is None:
        protocol = load_protocol()
    if seed not in set(protocol["training"]["seeds"]):
        raise ValueError(
            f"seed {seed} is not in configs/base.yaml training.seeds "
            f"{protocol['training']['seeds']}"
        )

    training = protocol["training"]
    rule = training["checkpoint_selection"]
    if rule not in CHECKPOINT_RULES:
        raise ValueError(f"training.checkpoint_selection is {rule!r}; train_qute implements {list(CHECKPOINT_RULES)}")
    assert_train_batch_size(train_loader, int(training["batch_size"]))
    torch_device = torch.device(device)

    path = qute_checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed, transfer)
    _refuse_existing(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = build_qute_from_protocol(protocol, seed).to(torch_device)

    # Same hook as the baselines: build_qute reseeds the global RNG, so restart
    # the shuffle afterwards from split_seed. Without this, QUTE would see a
    # different minibatch order than BASE and the comparison would be confounded.
    fix_data_order(train_loader, protocol)
    fix_data_order(validation_loader, protocol)
    # Reseed the global RNG from split_seed so augmentation matches the baselines.
    pin_training_randomness(protocol)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["momentum"]), 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training["lr_decay_per_epoch"])
    )

    epochs = int(training["epochs"])
    freeze_from = model.freeze_epoch_start(epochs)
    transfer_in_freeze = bool(model.config.transfer_during_freeze)

    best_nll, best_state, best_epoch = float("inf"), None, None
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        frozen = epoch >= freeze_from
        model.freeze_base(frozen)
        model.train()

        current_lr = float(optimizer.param_groups[0]["lr"])
        sums = {"total": 0.0, "ee": 0.0, "ev": 0.0, "final": 0.0}
        seen = 0

        for batch in train_loader:
            images, labels = unpack_batch(batch)
            images, labels = images.to(torch_device), labels.to(torch_device)

            # The copy comes first, so the loss is computed against the
            # transferred weights. Reversing this trains a different method.
            if transfer and (not frozen or transfer_in_freeze):
                model.transfer_weights()

            optimizer.zero_grad(set_to_none=True)
            parts = qute_loss(model.forward_train(images), labels, model.config)
            parts["total"].backward()
            optimizer.step()

            n = labels.shape[0]
            for key in sums:
                sums[key] += float(parts[key].detach()) * n
            seen += n

        if seen == 0:
            raise RuntimeError("train loader produced no examples")

        val_nll = deployed_nll(model, validation_loader, torch_device)
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "frozen_base": frozen,
            "val_nll": val_nll,
            **{f"train_{k}": v / seen for k, v in sums.items()},
        }
        history.append(row)

        if verbose:
            print(
                f"{method_name(transfer)} seed {seed} epoch {epoch}/{epochs} "
                f"total={row['train_total']:.4f} ee={row['train_ee']:.4f} "
                f"ev={row['train_ev']:.4f} final={row['train_final']:.4f} "
                f"val_nll={val_nll:.4f} lr={current_lr:.6f}"
                + ("  [base frozen]" if frozen else "")
            )

        if rule == "min_val_nll" and val_nll < best_nll:
            best_nll, best_epoch = val_nll, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()
    if rule == "final_epoch":
        best_nll = history[-1]["val_nll"]
        best_epoch = history[-1]["epoch"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.freeze_base(False)
    model.load_state_dict(best_state)

    payload = {
        "method": method_name(transfer),
        "seed": seed,
        "transfer": transfer,
        "dropout_rate": 0.0,
        "config_version": int(protocol["config_version"]),
        "epoch": best_epoch,
        "checkpoint_selection": rule,
        "val_nll": best_nll,
        "val_nll_definition": "mean -log p(true class) of the averaged output blocks",
        "model_state_dict": best_state,
        "training": training,
        "model": protocol["model"],
        "qute": {
            **{k: v for k, v in protocol["qute"].items()},
            "ev_weights": ev_weights(model.config),
            "tau_weights": tau_weights(model.config),
            "freeze_from_epoch": freeze_from,
            "deployed_parameters": model.deployed_parameters(),
            "training_parameters": model.training_parameters(),
        },
    }
    torch.save(payload, path)
    return model, history


@torch.no_grad()
def predict_proba_qute(model: QUTEModel, loader: DataLoader, device: str = "cpu"):
    """Deployed prediction. Returns (mean_probs [N,C], sample_ids [N], members [N,K,C]).

    The output blocks are the ensemble members, exactly as the deep ensemble's
    networks are. Averaging happens over softmaxes via ensemble_proba -- never
    over logits, and never over predicted class IDs.
    """
    _require_indexed_eval_loader(loader, "evaluation loader")
    model.eval().to(device)
    member_chunks, id_chunks = [], []
    for images, _labels, idx in loader:
        logits = model.member_logits(images.to(device))
        member_chunks.append(torch.softmax(logits, dim=2).cpu())
        id_chunks.append(idx.cpu())

    members = torch.cat(member_chunks).float()
    sample_ids = torch.cat(id_chunks).long()
    return ensemble_proba(members), sample_ids, members


def evaluate_qute(
    model: QUTEModel,
    loader: DataLoader,
    seed: int,
    protocol: dict,
    path,
    condition: str = "clean",
    transfer: bool = True,
    overwrite: bool = False,
    device: str = "cpu",
):
    """Save probabilities and sample ids before any aggregate metric."""
    output = Path(path)
    _refuse_existing(output, overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)

    mean_probs, sample_ids, members = predict_proba_qute(model, loader, device=device)
    if members.shape[1] != model.config.K:
        raise ValueError(f"expected member axis {model.config.K}, got {members.shape[1]}")

    _confidence, extra = with_confidence(
        mean_probs,
        {
            "K": model.config.K,
            "transfer": transfer,
            "forward_passes": 1,
            "deployed_parameters": model.deployed_parameters(),
            "training_parameters": model.training_parameters(),
            "dropped_original_output_block": True,
        },
    )
    metadata = save_predictions(
        output,
        mean_probs,
        sample_ids,
        method=method_name(transfer),
        seed=seed,
        condition=condition,
        member_probs=members,
        config_version=int(protocol["config_version"]),
        extra=extra,
    )
    return mean_probs, sample_ids, members, metadata


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train QUTE from configs/base.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-transfer", action="store_true",
        help="ablation: disable the early-view weight transfer and nothing else",
    )
    args = parser.parse_args(argv)
    transfer = not args.no_transfer

    protocol = load_protocol()
    train_loader, validation_loader, test_loader = require_loaders(protocol)

    for seed in protocol["training"]["seeds"]:
        model, _history = train_qute(
            train_loader,
            validation_loader,
            seed=int(seed),
            protocol=protocol,
            transfer=transfer,
            device=args.device,
            overwrite=args.overwrite,
            verbose=True,
        )
        output = qute_prediction_path(int(seed), transfer=transfer)
        evaluate_qute(
            model,
            test_loader,
            seed=int(seed),
            protocol=protocol,
            path=output,
            transfer=transfer,
            overwrite=args.overwrite,
            device=args.device,
        )
        print(f"saved predictions to {output}")


if __name__ == "__main__":
    main()
