"""Train and evaluate the BASE CNN.

BASE is one SharedCNN per seed, with dropout left off. Settings come from
configs/base.yaml. This module does not build the MNIST split or the
preprocessing pipeline; those loaders belong in data/.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from eval.scores import with_confidence
from models import CNNConfig, build_model, predict_proba, save_predictions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "base.yaml"

# training.checkpoint_selection values fit_cnn implements.
#   min_val_nll: keep the epoch with the lowest validation NLL
#   final_epoch: keep the last epoch's weights, as the paper does (fixed 20 epochs)
# Validation NLL is computed and logged every epoch under both rules.
CHECKPOINT_RULES = ("min_val_nll", "final_epoch")


def load_protocol(path: Optional[Path] = None) -> dict:
    """Read configs/base.yaml. Training code must not override these values."""
    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL
    with protocol_path.open() as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError(f"{protocol_path} did not contain a mapping")
    _check_protocol(protocol)
    return protocol


def _check_protocol(protocol: dict) -> None:
    training = protocol.get("training")
    model = protocol.get("model")
    if "config_version" not in protocol or not isinstance(training, dict) or not isinstance(model, dict):
        raise ValueError("protocol is missing config_version, training, or model")
    required = (
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "momentum",
        "lr_decay_per_epoch",
        "seeds",
        "checkpoint_selection",
        "data_order",
    )
    if "num_workers" not in training:
        raise ValueError("training block is missing num_workers")
    missing = [key for key in required if key not in training]
    if missing:
        raise ValueError(f"training block is missing {missing}")
    if training["optimizer"] != "adam":
        raise ValueError(
            f"training.optimizer is {training['optimizer']!r}; BASE implements adam from configs/base.yaml"
        )
    if training["checkpoint_selection"] not in CHECKPOINT_RULES:
        raise ValueError(
            "training.checkpoint_selection is "
            f"{training['checkpoint_selection']!r}; this trainer implements {list(CHECKPOINT_RULES)}"
        )
    if training["data_order"] != "fixed_by_split_seed":
        raise ValueError(
            "training.data_order is "
            f"{training['data_order']!r}; this trainer only implements fixed_by_split_seed"
        )
    if "split_seed" not in protocol.get("data", {}):
        raise ValueError("data block is missing split_seed")
    for key in ("channels", "kernel_size", "pool_after", "use_batchnorm", "head", "parameters"):
        if key not in model:
            raise ValueError(f"model block is missing {key}")
    baselines = protocol.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("protocol is missing baselines")
    mc = baselines.get("mc_dropout")
    ensemble = baselines.get("deep_ensemble")
    if not isinstance(mc, dict) or not isinstance(ensemble, dict):
        raise ValueError("baselines block is missing mc_dropout or deep_ensemble")
    for key in ("rate", "passes_headline", "passes_reference", "headline_is_prefix_of_reference"):
        if key not in mc:
            raise ValueError(f"baselines.mc_dropout is missing {key}")
    for key in ("members_headline", "members_reference", "headline_is_subset_of_reference"):
        if key not in ensemble:
            raise ValueError(f"baselines.deep_ensemble is missing {key}")


def require_loaders(protocol: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build Abirami's loaders on demand from the protocol.

    data.get_loaders(protocol) must return
    (train_loader, validation_loader, test_loader). Constructing those loaders
    at import time would download MNIST as a side effect and freeze the batch
    size before the config can reach it.
    """
    try:
        import data as data_module
    except ImportError as exc:
        raise SystemExit(
            "Missing data.get_loaders(protocol). Import error: " + str(exc)
        ) from exc
    factory = getattr(data_module, "get_loaders", None)
    if not callable(factory):
        raise SystemExit(
            "Missing data.get_loaders(protocol). "
            "Build train_loader, validation_loader, and test_loader inside that "
            "function using training.batch_size from the protocol. "
            "Do not construct them at import time."
        )
    loaders = factory(protocol)
    if not isinstance(loaders, (tuple, list)) or len(loaders) != 3:
        raise SystemExit(
            "data.get_loaders(protocol) must return "
            "(train_loader, validation_loader, test_loader)."
        )
    return tuple(loaders)


def _random_sampler(loader: DataLoader):
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, BatchSampler):
        sampler = sampler.sampler
    if isinstance(sampler, RandomSampler):
        return sampler
    return None


def fix_data_order(loader: DataLoader, protocol: dict) -> None:
    """Restart a shuffled loader from data.split_seed.

    The model seed resets the global torch RNG inside build_model. Shuffle
    order has to come from an explicit generator, or ensemble members would
    differ by both initialization and minibatch order. A loader that does not
    shuffle is already fixed.
    """
    sampler = _random_sampler(loader)
    if sampler is None:
        return
    if sampler.generator is None:
        raise ValueError(
            "loader shuffles with the global RNG. data.get_loaders must pass "
            "generator=torch.Generator().manual_seed(split_seed) so every model "
            "seed and ensemble member sees the same data order."
        )
    sampler.generator.manual_seed(int(protocol["data"]["split_seed"]))


def pin_training_randomness(protocol: dict) -> None:
    """Reseed global RNGs for training-time augmentation after model init.

    build_cnn_model seeds from the model seed. Minibatch order is reset by
    fix_data_order. Augmentation (RandomCrop, RandomRotation, etc.) still
    draws from the global torch RNG unless we reseed it from data.split_seed,
    so every run seed and ensemble member sees the same augmented batches.
    """
    split_seed = int(protocol["data"]["split_seed"])
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)


def worker_init_fn(protocol: dict):
    """Pass to DataLoader(..., worker_init_fn=worker_init_fn(protocol)) when num_workers > 0."""

    split_seed = int(protocol["data"]["split_seed"])

    def init(worker_id: int) -> None:
        seed = split_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return init


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cnn_model(protocol: dict, seed: int, dropout_rate: float) -> nn.Module:
    """Build one SharedCNN from the yaml architecture and check its parameter count."""
    model_cfg = protocol["model"]
    config = CNNConfig(
        channels=list(model_cfg["channels"]),
        kernel_size=int(model_cfg["kernel_size"]),
        pool_after=list(model_cfg["pool_after"]),
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
        dropout_rate=float(dropout_rate),
        head=str(model_cfg["head"]),
    )
    set_seed(seed)
    model = build_model(config, seed=seed)
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = int(model_cfg["parameters"])
    if actual != expected:
        raise RuntimeError(
            f"parameter count is {actual}, configs/base.yaml model.parameters is {expected}"
        )
    dropout_layers = sum(1 for module in model.modules() if isinstance(module, nn.Dropout))
    if dropout_rate == 0.0 and dropout_layers != 0:
        raise RuntimeError(f"dropout_rate 0 must not contain dropout, found {dropout_layers}")
    exit_locations = len(model_cfg["pool_after"])
    if dropout_rate > 0.0 and dropout_layers != exit_locations:
        raise RuntimeError(
            f"expected dropout at {exit_locations} early-exit locations, found {dropout_layers}"
        )
    return model


def build_base_model(protocol: dict, seed: int) -> nn.Module:
    """Build one dropout-free SharedCNN."""
    return build_cnn_model(protocol, seed, dropout_rate=0.0)


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("expected a batch of (images, labels)")
    images, labels = batch[0], batch[1]
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise TypeError("images and labels must be tensors")
    return images, labels


def assert_supervised_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(f"expected images [N, 1, 28, 28], got {tuple(images.shape)}")
    if not images.dtype.is_floating_point:
        raise ValueError(f"expected float images, got {images.dtype}")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(f"expected labels [N], got {tuple(labels.shape)}")
    if labels.dtype != torch.int64:
        raise ValueError(f"expected int64 labels, got {labels.dtype}")
    if labels.numel() == 0:
        raise ValueError("empty batch")
    if int(labels.min()) < 0 or int(labels.max()) > 9:
        raise ValueError(
            f"labels must be in 0..9, got {int(labels.min())}..{int(labels.max())}"
        )


def assert_train_batch_size(loader: DataLoader, expected: int) -> None:
    actual = getattr(loader, "batch_size", None)
    if actual is not None and actual != expected:
        raise ValueError(
            f"train_loader batch_size is {actual}, configs/base.yaml training.batch_size is {expected}"
        )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images, labels = unpack_batch(batch)
            assert_supervised_batch(images, labels)
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if tuple(logits.shape) != (images.shape[0], 10):
                raise ValueError(f"expected logits [N, 10], got {tuple(logits.shape)}")
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
    if total_examples == 0:
        raise RuntimeError("loader produced no examples")
    return total_loss / total_examples


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """One training epoch. Loss is mean cross-entropy on logits."""
    return _run_epoch(model, loader, criterion, optimizer, torch.device(device))


def evaluate_nll(model, loader, criterion, device) -> float:
    """Mean validation NLL. Cross-entropy on logits is the NLL of the softmax."""
    return _run_epoch(model, loader, criterion, None, torch.device(device))


def checkpoint_path(directory, config_version: int, seed: int) -> Path:
    return Path(directory) / f"base_config{config_version}_seed{seed}.pt"


def prediction_path(seed: int, condition: str = "clean", directory="results/preds") -> Path:
    return Path(directory) / f"base_seed{seed}_{condition}.npz"


def _clone_state(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")


def fit_cnn(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    seed: int,
    protocol: dict,
    method: str,
    dropout_rate: float,
    checkpoint: Path,
    device: str = "cpu",
    overwrite: bool = False,
    verbose: bool = False,
    extra: Optional[dict] = None,
) -> Tuple[nn.Module, List[dict]]:
    """Train one SharedCNN and save the checkpoint picked by training.checkpoint_selection.

    min_val_nll keeps the epoch with the lowest validation NLL. final_epoch keeps the
    last epoch's weights. Validation NLL is logged in history every epoch either way.
    Validation runs in eval mode, so dropout is off while it is measured.
    MC Dropout turns dropout back on only at prediction time.
    """
    _check_protocol(protocol)
    assert_train_batch_size(train_loader, int(protocol["training"]["batch_size"]))
    training = protocol["training"]
    torch_device = torch.device(device)
    model = build_cnn_model(protocol, seed, dropout_rate).to(torch_device)
    fix_data_order(train_loader, protocol)
    fix_data_order(validation_loader, protocol)
    pin_training_randomness(protocol)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["momentum"]), 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training["lr_decay_per_epoch"])
    )

    path = Path(checkpoint)
    _refuse_existing(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)

    rule = training["checkpoint_selection"]
    best_nll = float("inf")
    best_state = None
    best_epoch = None
    history: List[dict] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_nll = train_one_epoch(model, train_loader, criterion, optimizer, torch_device)
        val_nll = evaluate_nll(model, validation_loader, criterion, torch_device)
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "lr": current_lr,
            }
        )
        if verbose:
            print(
                f"{method} seed {seed} epoch {epoch}/{training['epochs']} "
                f"train_nll={train_nll:.4f} val_nll={val_nll:.4f} lr={current_lr:.6f}"
            )
        if rule == "min_val_nll" and val_nll < best_nll:
            best_nll = val_nll
            best_epoch = epoch
            best_state = _clone_state(model)
        scheduler.step()

    if rule == "final_epoch":
        best_nll = history[-1]["val_nll"]
        best_epoch = history[-1]["epoch"]
        best_state = _clone_state(model)

    model.load_state_dict(best_state)
    payload = {
        "method": method,
        "seed": seed,
        "dropout_rate": float(dropout_rate),
        "config_version": int(protocol["config_version"]),
        "checkpoint_selection": rule,
        "epoch": best_epoch,
        "val_nll": best_nll,
        "model_state_dict": best_state,
        "training": training,
        "model": protocol["model"],
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return model, history


def train_base(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    seed: int,
    protocol: Optional[dict] = None,
    device: str = "cpu",
    checkpoint_dir="checkpoints",
    overwrite: bool = False,
    verbose: bool = False,
) -> Tuple[nn.Module, List[dict]]:
    """Train one BASE model and save the checkpoint picked by training.checkpoint_selection."""
    if protocol is None:
        protocol = load_protocol()
    else:
        _check_protocol(protocol)
    if seed not in set(protocol["training"]["seeds"]):
        raise ValueError(
            f"seed {seed} is not in configs/base.yaml training.seeds {protocol['training']['seeds']}"
        )
    path = checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed)
    return fit_cnn(
        train_loader,
        validation_loader,
        seed,
        protocol,
        method="base",
        dropout_rate=0.0,
        checkpoint=path,
        device=device,
        overwrite=overwrite,
        verbose=verbose,
    )


def _require_indexed_eval_loader(loader: DataLoader, name: str) -> None:
    if isinstance(getattr(loader, "sampler", None), RandomSampler):
        raise ValueError(f"{name} is shuffled. Evaluation loaders need shuffle=False.")
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__getitem__"):
        raise ValueError(f"{name} has no indexable dataset, so sample ids cannot be checked")
    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or len(sample) < 3:
        raise ValueError(
            f"{name} yields (image, label) without a sample id. Wrap the dataset as "
            "IndexedDataset(subset, ids=original_indices) before evaluation."
        )


def evaluate_base(
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    protocol: dict,
    path,
    condition: str = "clean",
    overwrite: bool = False,
    device: str = "cpu",
):
    """Run one deterministic pass and save probabilities before any aggregate metric."""
    output = Path(path)
    _refuse_existing(output, overwrite)
    _require_indexed_eval_loader(loader, "evaluation loader")
    output.parent.mkdir(parents=True, exist_ok=True)
    probs, sample_ids = predict_proba(model, loader, device=device)
    confidence, extra = with_confidence(
        probs,
        {
            "parameters": int(protocol["model"]["parameters"]),
            "dropout_rate": 0.0,
        },
    )
    metadata = save_predictions(
        output,
        probs,
        sample_ids,
        method="base",
        seed=seed,
        condition=condition,
        config_version=int(protocol["config_version"]),
        extra=extra,
    )
    return probs, sample_ids, confidence, metadata


def load_checkpoint(path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train BASE from configs/base.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    train_loader, validation_loader, test_loader = require_loaders(protocol)
    for seed in protocol["training"]["seeds"]:
        model, _history = train_base(
            train_loader,
            validation_loader,
            seed=int(seed),
            protocol=protocol,
            device=args.device,
            overwrite=args.overwrite,
            verbose=True,
        )
        output = prediction_path(int(seed))
        evaluate_base(
            model,
            test_loader,
            seed=int(seed),
            protocol=protocol,
            path=output,
            overwrite=args.overwrite,
            device=args.device,
        )
        print(f"saved predictions to {output}")


if __name__ == "__main__":
    main()
