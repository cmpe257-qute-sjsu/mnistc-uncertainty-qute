"""
Mechanics checks for the deep ensemble.

    python tests/test_ensemble.py

Trains the reference members on synthetic images. The 2-member file is the
first members of that same stack, averaged with ensemble_proba.
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ensemble_proba, load_predictions
from baselines.base import load_protocol
from baselines.ensemble import (
    ensemble_checkpoint_path,
    evaluate_deep_ensemble,
    member_init_seeds,
    train_deep_ensemble,
)
from baselines.base import load_checkpoint

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
    generator = torch.Generator().manual_seed(5)
    images = torch.rand(n, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (n,), generator=generator)
    dataset = TensorDataset(images, labels)
    if indexed:
        from models import IndexedDataset
        dataset = IndexedDataset(dataset, ids=list(range(800, 800 + n)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def short_protocol():
    protocol = copy.deepcopy(load_protocol())
    protocol["training"]["epochs"] = 1
    protocol["training"]["batch_size"] = 8
    return protocol


print("\nPROTOCOL")
print("--------")

def t_protocol():
    block = load_protocol()["baselines"]["deep_ensemble"]
    assert block["members_headline"] == 2
    assert block["members_reference"] == 5
    assert block["headline_is_subset_of_reference"] is True
    assert member_init_seeds(11, 5) == [1101, 1102, 1103, 1104, 1105]
    assert len(set(member_init_seeds(11, 5) + member_init_seeds(22, 5))) == 10
check("ensemble protocol is 2 headline members inside a 5-member run", t_protocol)


print("\nTRAIN AND PREDICT")
print("-----------------")

def t_subset():
    protocol = short_protocol()
    with tempfile.TemporaryDirectory() as directory:
        models, _histories, seeds = train_deep_ensemble(
            fake_loader(),
            fake_loader(),
            run_seed=11,
            protocol=protocol,
            checkpoint_dir=directory,
        )
        assert seeds == [1101, 1102, 1103, 1104, 1105]
        assert len(models) == 5
        first = load_checkpoint(
            ensemble_checkpoint_path(directory, protocol["config_version"], 11, 1, 1101)
        )
        second = load_checkpoint(
            ensemble_checkpoint_path(directory, protocol["config_version"], 11, 2, 1102)
        )
        assert first["method"] == "deep_ensemble"
        assert first["dropout_rate"] == 0.0
        assert not torch.equal(
            first["model_state_dict"]["classifier.weight"],
            second["model_state_dict"]["classifier.weight"],
        )
        headline_path = Path(directory) / "members2.npz"
        reference_path = Path(directory) / "members5.npz"
        headline, reference, ids, hmeta, rmeta = evaluate_deep_ensemble(
            models,
            fake_loader(indexed=True),
            run_seed=11,
            protocol=protocol,
            headline_path=headline_path,
            reference_path=reference_path,
        )
        _hp, _hi, _hm, hmembers = load_predictions(headline_path)
        _rp, _ri, _rm, rmembers = load_predictions(reference_path)
        assert headline.shape == (16, 10)
        assert reference.shape == (16, 10)
        assert hmembers.shape == (16, 2, 10)
        assert rmembers.shape == (16, 5, 10)
        assert torch.allclose(hmembers, rmembers[:, :2, :], atol=1e-6)
        assert torch.allclose(headline, ensemble_proba(rmembers[:, :2, :].contiguous()), atol=1e-6)
        assert torch.allclose(reference, ensemble_proba(rmembers), atol=1e-6)
        assert ids.tolist() == list(range(800, 816))
        assert hmeta["role"] == "headline"
        assert rmeta["role"] == "reference"
        assert hmeta["confidence"] == "max_softmax_probability"
        assert rmeta["confidence"] == "max_softmax_probability"
        assert rmeta["n_members"] == 5
check("2-member ensemble is the first two of the five trained members", t_subset)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nDeep ensemble checks pass.")
