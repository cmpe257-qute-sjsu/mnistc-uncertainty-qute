"""
Mechanics checks for the BASE trainer.

    python tests/test_base.py

Uses a handful of synthetic images. It does not download MNIST or replace
the data loaders. The real 20-epoch run waits for data.train_loader,
data.validation_loader, and data.test_loader.
"""

import copy
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IndexedDataset, load_predictions
from baselines.base import (
    build_base_model,
    checkpoint_path,
    evaluate_base,
    fix_data_order,
    load_checkpoint,
    load_protocol,
    pin_training_randomness,
    require_loaders,
    train_base,
)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def fake_loader(n=16, batch_size=8, indexed=False, channels=1, shuffle=False):
    generator = torch.Generator().manual_seed(3)
    images = torch.rand(n, channels, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (n,), generator=generator)
    dataset = TensorDataset(images, labels)
    if indexed:
        dataset = IndexedDataset(dataset, ids=list(range(500, 500 + n)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def short_protocol():
    protocol = copy.deepcopy(load_protocol())
    protocol["training"]["epochs"] = 2
    protocol["training"]["batch_size"] = 8
    return protocol


print("\nPROTOCOL")
print("--------")

def t_protocol():
    protocol = load_protocol()
    training = protocol["training"]
    assert training["epochs"] == 20
    assert training["batch_size"] == 256
    assert training["optimizer"] == "adam"
    assert training["learning_rate"] == 0.001
    assert training["momentum"] == 0.9
    assert training["lr_decay_per_epoch"] == 0.99
    assert training["seeds"] == [11, 22, 33]
    assert training["checkpoint_selection"] == "min_val_nll"
    assert training["data_order"] == "fixed_by_split_seed"
    assert training["num_workers"] == 0
    assert protocol["config_version"] == 4
    assert protocol["model"]["parameters"] == 15466
check("base.yaml still has the frozen BASE settings", t_protocol)

def t_missing_loaders():
    protocol = load_protocol()
    fake = types.ModuleType("data")
    with mock.patch.dict(sys.modules, {"data": fake}):
        try:
            require_loaders(protocol)
        except SystemExit as exc:
            message = str(exc)
            assert "data.get_loaders(protocol)" in message, message
            return
    raise AssertionError("require_loaders should exit when get_loaders is missing")
check("missing data.get_loaders is reported by name", t_missing_loaders)

def t_augmentation_rng():
    protocol = load_protocol()
    build_base_model(protocol, 11)
    pin_training_randomness(protocol)
    first = torch.rand(4)
    build_base_model(protocol, 22)
    pin_training_randomness(protocol)
    second = torch.rand(4)
    assert torch.equal(first, second)
check("augmentation RNG is pinned to split_seed after model init", t_augmentation_rng)

def t_data_order():
    protocol = load_protocol()
    generator = torch.Generator().manual_seed(1)
    images = torch.arange(16).view(16, 1, 1, 1).float().expand(16, 1, 28, 28).contiguous()
    labels = torch.zeros(16, dtype=torch.int64)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True, generator=generator)
    fix_data_order(loader, protocol)
    first = next(iter(loader))[0].clone()
    fix_data_order(loader, protocol)
    second = next(iter(loader))[0]
    assert torch.equal(first, second)
    unfixed = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True)
    try:
        fix_data_order(unfixed, protocol)
    except ValueError as exc:
        assert "global RNG" in str(exc), exc
        return
    raise AssertionError("a shuffled loader without a generator was accepted")
check("shuffled loaders restart from split_seed and refuse the global RNG", t_data_order)


print("\nMODEL")
print("-----")

def t_base_model():
    protocol = load_protocol()
    first = build_base_model(protocol, 11)
    second = build_base_model(protocol, 11)
    third = build_base_model(protocol, 22)
    first_weights = torch.cat([parameter.detach().flatten() for parameter in first.parameters()])
    second_weights = torch.cat([parameter.detach().flatten() for parameter in second.parameters()])
    third_weights = torch.cat([parameter.detach().flatten() for parameter in third.parameters()])
    assert torch.equal(first_weights, second_weights)
    assert not torch.equal(first_weights, third_weights)
    assert sum(1 for module in first.modules() if isinstance(module, nn.Dropout)) == 0
    logits = first(torch.zeros(4, 1, 28, 28))
    assert logits.shape == (4, 10)
check("BASE reuses SharedCNN, repeats a seed, and returns logits", t_base_model)

def t_bad_batch():
    images = torch.rand(4, 3, 28, 28)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    with tempfile.TemporaryDirectory() as directory:
        try:
            train_base(loader, loader, seed=11, protocol=short_protocol(), checkpoint_dir=directory)
        except ValueError as exc:
            assert "[N, 1, 28, 28]" in str(exc), exc
            return
    raise AssertionError("wrong image shape was accepted")
check("training rejects images that are not [N, 1, 28, 28]", t_bad_batch)


print("\nTRAINING")
print("--------")

def t_train_and_checkpoint():
    protocol = short_protocol()
    with tempfile.TemporaryDirectory() as directory:
        model, history = train_base(
            fake_loader(),
            fake_loader(),
            seed=11,
            protocol=protocol,
            checkpoint_dir=directory,
        )
        assert [row["epoch"] for row in history] == [1, 2]
        assert abs(history[0]["lr"] - 0.001) < 1e-12
        assert abs(history[1]["lr"] - 0.001 * 0.99) < 1e-12
        saved = load_checkpoint(checkpoint_path(directory, protocol["config_version"], 11))
        assert saved["method"] == "base"
        assert saved["seed"] == 11
        assert saved["epoch"] == min(history, key=lambda row: row["val_nll"])["epoch"]
        assert abs(saved["val_nll"] - min(row["val_nll"] for row in history)) < 1e-8
        assert saved["config_version"] == protocol["config_version"]
        model.load_state_dict(saved["model_state_dict"])
check("two epochs decay the learning rate and keep the best validation NLL", t_train_and_checkpoint)

def t_no_overwrite():
    protocol = short_protocol()
    with tempfile.TemporaryDirectory() as directory:
        train_base(fake_loader(), fake_loader(), seed=11, protocol=protocol, checkpoint_dir=directory)
        try:
            train_base(fake_loader(), fake_loader(), seed=11, protocol=protocol, checkpoint_dir=directory)
        except FileExistsError:
            return
    raise AssertionError("an existing checkpoint was overwritten")
check("an existing checkpoint is left in place", t_no_overwrite)

def t_predictions():
    protocol = short_protocol()
    with tempfile.TemporaryDirectory() as directory:
        model, _history = train_base(
            fake_loader(),
            fake_loader(),
            seed=22,
            protocol=protocol,
            checkpoint_dir=directory,
        )
        output = Path(directory) / "base_seed22_clean.npz"
        probs, sample_ids, confidence, metadata = evaluate_base(
            model,
            fake_loader(indexed=True),
            seed=22,
            protocol=protocol,
            path=output,
        )
        assert probs.shape == (16, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)
        assert sample_ids.tolist() == list(range(500, 516))
        assert torch.equal(confidence, probs.max(dim=1).values)
        loaded_probs, loaded_ids, loaded_meta, _members = load_predictions(output)
        assert torch.allclose(probs, loaded_probs, atol=1e-6)
        assert torch.equal(sample_ids, loaded_ids)
        assert loaded_meta["method"] == "base"
        assert loaded_meta["confidence"] == "max_softmax_probability"
        assert metadata["n_members"] == 1
        try:
            evaluate_base(model, fake_loader(indexed=True), seed=22, protocol=protocol, path=output)
        except FileExistsError:
            return
    raise AssertionError("an existing prediction file was overwritten")
check("clean predictions keep sample ids and max-softmax confidence", t_predictions)

def t_unindexed_eval():
    protocol = load_protocol()
    model = build_base_model(protocol, 11)
    try:
        evaluate_base(model, fake_loader(indexed=False), seed=11, protocol=protocol, path="unused.npz")
    except ValueError as exc:
        assert "sample id" in str(exc), exc
        return
    raise AssertionError("evaluation accepted a loader without sample ids")
check("evaluation refuses a loader that has no sample ids", t_unindexed_eval)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nBASE trainer checks pass.")
print("Real training still needs data.get_loaders(protocol).")
