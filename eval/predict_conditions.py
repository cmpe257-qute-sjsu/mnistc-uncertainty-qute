"""Predict every trained model on all 16 conditions (clean + 15 MNIST-C).

    python -m eval.predict_conditions                       # all baselines, all seeds
    python -m eval.predict_conditions --methods base --seeds 11
    python -m eval.predict_conditions --conditions clean shot_noise

Nothing is trained here. Each saved checkpoint is loaded once and run on each
of the 16 loaders from data.prepare_mnistc.get_corruption_loaders. Every
condition is predicted exactly once per model; streams and windows are built
later in eval from these files.

Output, one file per method, seed and condition, in results/preds/conditions/:

    base_seed{seed}_{condition}.npz
    mc_dropout_seed{seed}_{condition}_passes{2,20}.npz
    deep_ensemble_seed{seed}_{condition}_members{2,5}.npz

The clean condition is the original MNIST test set (the "clean" entry of
get_corruption_loaders, same images, normalization and sample IDs as the
training-time test loader). It is predicted again here so that all 16 files
are made the same way and carry labels and confidence. The clean-only files
written at training time in results/preds/ are left untouched.

Each file has the usual probs / sample_ids / metadata (and member_probs for
MC Dropout and the ensemble), plus:

    labels      [N] int64    true class, aligned to sample_ids
    confidence  [N] float32  max softmax probability of probs

MC Dropout passes are seeded with the model seed only (evaluate_mc_dropout
does this), not with a separate seed per condition. Every condition therefore
sees the same sequence of dropout masks (common random numbers), and rerunning
gives identical files. Describe it this way in the report.

Adding a method (e.g. QUTE): write a function with the same signature as
predict_base below and add it to METHODS.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.base import (
    build_cnn_model,
    checkpoint_path,
    evaluate_base,
    load_checkpoint,
    load_protocol,
    prediction_path,
)
from baselines.ensemble import (
    _ensemble_settings,
    ensemble_checkpoint_path,
    ensemble_prediction_path,
    evaluate_deep_ensemble,
    member_init_seeds,
)
from baselines.mc_dropout import (
    evaluate_mc_reference_and_headline,
    mc_checkpoint_path,
    mc_prediction_path,
)
from qute.evaluate import load_qute_checkpoint
from qute.train import evaluate_qute, qute_prediction_path

DEFAULT_OUT_DIR = Path("results/preds/conditions")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------- labels

def labels_by_id(loader: DataLoader) -> Dict[int, int]:
    """Map sample_id -> label by reading the loader once. No model involved."""
    mapping: Dict[int, int] = {}
    for batch in loader:
        if len(batch) < 3:
            raise ValueError("evaluation loader must yield (image, label, sample_id)")
        _images, labels, ids = batch[0], batch[1], batch[2]
        for sample_id, label in zip(ids.tolist(), labels.tolist()):
            if sample_id in mapping:
                raise ValueError(f"sample_id {sample_id} appears twice in one condition")
            mapping[int(sample_id)] = int(label)
    return mapping


def attach_labels(path: Path, label_map: Dict[int, int]) -> None:
    """Add `labels` and `confidence` arrays to a saved prediction file.

    Labels are looked up by sample_id, so they line up with probs whatever
    the order. The file is rewritten atomically.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    ids = arrays["sample_ids"]
    missing = [int(i) for i in ids if int(i) not in label_map]
    if missing:
        raise ValueError(f"{path}: {len(missing)} sample ids have no label, e.g. {missing[:5]}")
    if len(ids) != len(label_map):
        raise ValueError(f"{path}: {len(ids)} predictions but the loader has {len(label_map)} samples")
    arrays["labels"] = np.array([label_map[int(i)] for i in ids], dtype=np.int64)
    arrays["confidence"] = arrays["probs"].max(axis=1).astype(np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


# ---------------------------------------------------------------- checkpoints

def load_cnn(path: Path, protocol: dict, expected_method: str, device: str = "cpu") -> torch.nn.Module:
    """Rebuild a SharedCNN from a checkpoint and check it matches the protocol."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing checkpoint {path}. Train it first, or check --checkpoint-dir.")
    payload = load_checkpoint(path)
    if payload.get("method") != expected_method:
        raise ValueError(f"{path} is method {payload.get('method')!r}, expected {expected_method!r}")
    if int(payload["config_version"]) != int(protocol["config_version"]):
        raise ValueError(
            f"{path} was trained with config v{payload['config_version']}, "
            f"but configs/base.yaml is v{protocol['config_version']}"
        )
    expected_rule = protocol["training"]["checkpoint_selection"]
    rule = payload.get("checkpoint_selection")
    if rule is None:
        raise ValueError(f"{path} does not record checkpoint_selection; it predates config v5")
    if rule != expected_rule:
        raise ValueError(f"{path} used checkpoint_selection {rule!r}, protocol says {expected_rule!r}")
    if rule == "final_epoch" and int(payload["epoch"]) != int(protocol["training"]["epochs"]):
        raise ValueError(
            f"{path} saved epoch {payload['epoch']}, but final_epoch should keep epoch "
            f"{protocol['training']['epochs']}"
        )
    model = build_cnn_model(protocol, int(payload["seed"]), float(payload["dropout_rate"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------- methods

def predict_base(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    ckpt = checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed)
    model = load_cnn(ckpt, protocol, "base", device)
    written = []
    for condition, loader in loaders.items():
        path = prediction_path(seed, condition, directory=out_dir)
        if path.exists() and not overwrite:
            print(f"  skip  {path} (exists)")
            continue
        evaluate_base(model, loader, seed, protocol, path, condition=condition, overwrite=overwrite, device=device)
        attach_labels(path, label_maps[condition])
        written.append(path)
        print(f"  saved {path}")
    return written


def predict_mc_dropout(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    mc = protocol["baselines"]["mc_dropout"]
    ckpt = mc_checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed)
    model = load_cnn(ckpt, protocol, "mc_dropout", device)
    written = []
    for condition, loader in loaders.items():
        headline = mc_prediction_path(seed, int(mc["passes_headline"]), condition, directory=out_dir)
        reference = mc_prediction_path(seed, int(mc["passes_reference"]), condition, directory=out_dir)
        if headline.exists() and reference.exists() and not overwrite:
            print(f"  skip  {headline.name}, {reference.name} (exist)")
            continue
        # A half-finished pair is redone as a pair, so both come from one run of passes.
        evaluate_mc_reference_and_headline(
            model, loader, seed, protocol, headline, reference,
            condition=condition, overwrite=True, device=device,
        )
        for path in (headline, reference):
            attach_labels(path, label_maps[condition])
            written.append(path)
            print(f"  saved {path}")
    return written


def predict_deep_ensemble(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    n_headline, n_reference = _ensemble_settings(protocol)
    version = int(protocol["config_version"])
    models = [
        load_cnn(ensemble_checkpoint_path(checkpoint_dir, version, seed, index, member_seed),
                 protocol, "deep_ensemble", device)
        for index, member_seed in enumerate(member_init_seeds(seed, n_reference), start=1)
    ]
    written = []
    for condition, loader in loaders.items():
        headline = ensemble_prediction_path(seed, n_headline, condition, directory=out_dir)
        reference = ensemble_prediction_path(seed, n_reference, condition, directory=out_dir)
        if headline.exists() and reference.exists() and not overwrite:
            print(f"  skip  {headline.name}, {reference.name} (exist)")
            continue
        evaluate_deep_ensemble(
            models, loader, seed, protocol, headline, reference,
            condition=condition, overwrite=True, device=device,
        )
        for path in (headline, reference):
            attach_labels(path, label_maps[condition])
            written.append(path)
            print(f"  saved {path}")
    return written

def _predict_qute(transfer, protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    model, _payload = load_qute_checkpoint(protocol, seed, transfer, checkpoint_dir=checkpoint_dir, device=device)
    written = []
    for condition, loader in loaders.items():
        path = qute_prediction_path(seed, condition, transfer, directory=out_dir)
        if path.exists() and not overwrite:
            print(f"  skip  {path} (exists)")
            continue
        evaluate_qute(model, loader, seed=seed, protocol=protocol, path=path,
                      condition=condition, transfer=transfer, overwrite=overwrite, device=device)
        attach_labels(path, label_maps[condition])
        written.append(path)
        print(f"  saved {path}")
    return written


def predict_qute(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    return _predict_qute(True, protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device)


def predict_qute_no_transfer(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device) -> List[Path]:
    return _predict_qute(False, protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device)

# name -> function(protocol, seed, loaders, label_maps, out_dir, checkpoint_dir, overwrite, device)
METHODS: Dict[str, Callable] = {
    "base": predict_base,
    "mc_dropout": predict_mc_dropout,
    "deep_ensemble": predict_deep_ensemble,
    "qute": predict_qute,
    "qute_no_transfer": predict_qute_no_transfer,
}


# ---------------------------------------------------------------- driver

def run(
    protocol: dict,
    loaders: Dict[str, DataLoader],
    methods: Iterable[str],
    seeds: Iterable[int],
    out_dir=DEFAULT_OUT_DIR,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    overwrite: bool = False,
    device: str = "cpu",
) -> List[Path]:
    """Predict each (method, seed) on every loader. Returns the files written."""
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; known: {sorted(METHODS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_maps = {condition: labels_by_id(loader) for condition, loader in loaders.items()}
    written: List[Path] = []
    for method in methods:
        for seed in seeds:
            print(f"{method} seed {seed}: {len(loaders)} conditions")
            written += METHODS[method](
                protocol, int(seed), loaders, label_maps, out_dir, Path(checkpoint_dir), overwrite, device
            )
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Predict saved checkpoints on clean + 15 MNIST-C conditions")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: training.seeds")
    parser.add_argument("--conditions", nargs="+", default=None, help="default: all 16")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    from data.prepare_mnistc import get_corruption_loaders

    protocol = load_protocol()
    seeds = args.seeds or [int(s) for s in protocol["training"]["seeds"]]
    bad_seeds = [s for s in seeds if s not in protocol["training"]["seeds"]]
    if bad_seeds:
        raise SystemExit(f"seeds {bad_seeds} are not in configs/base.yaml training.seeds")

    loaders, _manifest = get_corruption_loaders(protocol)
    if len(loaders) != 16:
        raise SystemExit(f"expected 16 conditions, got {len(loaders)}")
    if args.conditions:
        unknown = [c for c in args.conditions if c not in loaders]
        if unknown:
            raise SystemExit(f"unknown conditions {unknown}; known: {list(loaders)}")
        loaders = {c: loaders[c] for c in args.conditions}

    written = run(protocol, loaders, args.methods, seeds, args.out_dir, args.checkpoint_dir,
                  args.overwrite, args.device)
    print(f"\n{len(written)} files written to {args.out_dir}")


if __name__ == "__main__":
    main()
