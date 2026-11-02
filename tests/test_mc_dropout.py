"""
Mechanics checks for MC Dropout.

    python tests/test_mc_dropout.py

Synthetic images only. The saved headline file is the first passes of the
reference run, averaged with ensemble_proba.
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ensemble_proba, load_predictions
from baselines.mc_dropout import (
    evaluate_mc_reference_and_headline,
    mc_checkpoint_path,
    train_mc_dropout,
)
from baselines.base import load_checkpoint, load_protocol

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def fake_loader(n=16, batch_size=8, indexed=False):
    generator = torch.Generator().manual_seed(3)
    images = torch.rand(n, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (n,), generator=generator)
    dataset = TensorDataset(images, labels)
    if indexed:
        from models import IndexedDataset
        dataset = IndexedDataset(dataset, ids=list(range(700, 700 + n)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def short_protocol():
    protocol = copy.deepcopy(load_protocol())
    protocol["training"]["epochs"] = 1
    protocol["training"]["batch_size"] = 8
    protocol["baselines"]["mc_dropout"]["passes_headline"] = 2
    protocol["baselines"]["mc_dropout"]["passes_reference"] = 4
    return protocol


print("\nPROTOCOL")
print("--------")

def t_protocol():
    mc = load_protocol()["baselines"]["mc_dropout"]
    assert mc["rate"] == 0.1
    assert mc["passes_headline"] == 2
    assert mc["passes_reference"] == 20
    assert mc["headline_is_prefix_of_reference"] is True
    assert mc["locations"] == "early_exit_locations"
check("MC Dropout protocol is 0.1 dropout, 2 passes, and 20 passes", t_protocol)


print("\nTRAIN AND PREDICT")
print("-----------------")

def t_dropout_and_prefix():
    protocol = short_protocol()
    with tempfile.TemporaryDirectory() as directory:
        model, _history = train_mc_dropout(
            fake_loader(),
            fake_loader(),
            seed=11,
            protocol=protocol,
            checkpoint_dir=directory,
        )
        dropout_layers = sum(1 for module in model.modules() if isinstance(module, nn.Dropout))
        assert dropout_layers == 2, dropout_layers
        saved = load_checkpoint(mc_checkpoint_path(directory, protocol["config_version"], 11))
        assert saved["method"] == "mc_dropout"
        assert saved["dropout_rate"] == 0.1
        headline_path = Path(directory) / "headline.npz"
        reference_path = Path(directory) / "reference.npz"
        headline, reference, ids, _hmeta, rmeta = evaluate_mc_reference_and_headline(
            model,
            fake_loader(indexed=True),
            seed=11,
            protocol=protocol,
            headline_path=headline_path,
            reference_path=reference_path,
        )
        _hprobs, _hids, _hmeta, hmembers = load_predictions(headline_path)
        _rprobs, rids, _rmeta, rmembers = load_predictions(reference_path)
        assert headline.shape == (16, 10)
        assert reference.shape == (16, 10)
        assert rmembers.shape == (16, 4, 10)
        assert hmembers.shape == (16, 2, 10)
        assert torch.allclose(hmembers, rmembers[:, :2, :], atol=1e-6)
        assert torch.allclose(headline, ensemble_proba(rmembers[:, :2, :].contiguous()), atol=1e-6)
        assert torch.allclose(reference, ensemble_proba(rmembers), atol=1e-6)
        assert not torch.equal(rmembers[:, 0, :], rmembers[:, 1, :])
        assert ids.tolist() == list(range(700, 716))
        assert torch.equal(ids, rids)
        assert rmeta["passes"] == 4
        assert rmeta["n_members"] == 4
        assert rmeta["confidence"] == "max_softmax_probability"
        assert _hmeta["confidence"] == "max_softmax_probability"
check("headline MC file is the first passes of the reference run", t_dropout_and_prefix)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nMC Dropout checks pass.")
