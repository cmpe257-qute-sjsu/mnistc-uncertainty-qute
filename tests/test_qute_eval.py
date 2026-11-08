"""
Checks for QUTE evaluation through eval/predict_conditions.py.

    python tests/test_qute_eval.py

Synthetic images and a 2-epoch model, no MNIST or MNIST-C needed. These verify
that a checkpoint reloads into exactly the model that was trained, that the
shared prediction script writes QUTE and ablation files with the agreed names,
labels and confidence, that existing files are skipped, and that a checkpoint
from a different protocol is refused.
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IndexedDataset, load_predictions
from baselines.base import load_protocol
from eval.metrics import metrics_from_file
from eval.predict_conditions import METHODS, run
from qute.train import predict_proba_qute, train_qute
from qute.evaluate import load_qute_checkpoint

PASS, FAIL = [], []
BATCH, N, EVAL_N = 8, 24, 12
SEED = 11


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def short_protocol(epochs=2):
    p = copy.deepcopy(load_protocol())
    p["training"]["epochs"] = epochs
    p["training"]["batch_size"] = BATCH
    return p


def train_loaders(seed=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (N,), generator=g)
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=BATCH, shuffle=True,
                       generator=torch.Generator().manual_seed(0))
    val = DataLoader(IndexedDataset(ds, ids=list(range(600, 600 + N))),
                     batch_size=BATCH, shuffle=False)
    return train, val


def condition_loaders(names=("clean", "fog")):
    """Stand-ins for get_corruption_loaders: same labels and ids, different images."""
    g = torch.Generator().manual_seed(100)
    labels = torch.randint(0, 10, (EVAL_N,), generator=g)
    loaders = {}
    for name in names:
        x = torch.rand(EVAL_N, 1, 28, 28, generator=g)
        ds = IndexedDataset(TensorDataset(x, labels), ids=list(range(EVAL_N)))
        loaders[name] = DataLoader(ds, batch_size=5, shuffle=False)
    return loaders, labels.numpy()


def train_both(protocol, checkpoint_dir):
    train, val = train_loaders()
    on, _ = train_qute(train, val, seed=SEED, protocol=protocol, transfer=True, checkpoint_dir=checkpoint_dir)
    off, _ = train_qute(train, val, seed=SEED, protocol=protocol, transfer=False, checkpoint_dir=checkpoint_dir)
    return on, off


print("\nLOADING CHECKPOINTS")
print("-------------------")

def t_reload_matches_trained_model():
    p = short_protocol()
    loaders, _ = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        trained, _ = train_both(p, d)
        reloaded, payload = load_qute_checkpoint(p, SEED, True, checkpoint_dir=d)
    a, ids_a, _ = predict_proba_qute(trained, loaders["fog"])
    b, ids_b, _ = predict_proba_qute(reloaded, loaders["fog"])
    assert torch.equal(ids_a, ids_b)
    assert torch.allclose(a, b, atol=1e-7), "reloaded model predicts differently"
    assert payload["epoch"] == p["training"]["epochs"]
check("a reloaded checkpoint predicts exactly like the trained model", t_reload_matches_trained_model)

def t_missing_checkpoint_names_the_command():
    p = short_protocol()
    with tempfile.TemporaryDirectory() as d:
        try:
            load_qute_checkpoint(p, SEED, False, checkpoint_dir=d)
        except FileNotFoundError as exc:
            assert "--no-transfer" in str(exc), exc
            return
    raise AssertionError("a missing checkpoint was not reported")
check("a missing checkpoint says which training command to run", t_missing_checkpoint_names_the_command)

def t_refuses_other_protocol():
    p = short_protocol()
    with tempfile.TemporaryDirectory() as d:
        train_both(p, d)
        other = copy.deepcopy(p)
        other["training"]["checkpoint_selection"] = "min_val_nll"
        try:
            load_qute_checkpoint(other, SEED, True, checkpoint_dir=d)
        except ValueError as exc:
            assert "checkpoint_selection" in str(exc), exc
            return
    raise AssertionError("a checkpoint from a different rule was accepted")
check("a checkpoint saved under a different rule is refused", t_refuses_other_protocol)


print("\nSHARED PREDICTION SCRIPT")
print("------------------------")

def t_registered():
    assert "qute" in METHODS and "qute_no_transfer" in METHODS, sorted(METHODS)
check("qute and qute_no_transfer are registered in METHODS", t_registered)

def t_files_names_and_labels():
    p = short_protocol()
    loaders, labels = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_both(p, d)
        out = Path(d) / "out"
        written = run(p, loaders, ["qute", "qute_no_transfer"], [SEED], out, d)
        names = sorted(f.name for f in out.glob("*.npz"))
        expected = sorted(f"{m}_seed11_{c}.npz" for m in ("qute", "qute_no_transfer") for c in loaders)
        assert names == expected, names
        assert len(written) == 4
        assert not list(out.glob("*.tmp.npz")), "temporary files left behind"
        for name in names:
            method = "qute_no_transfer" if name.startswith("qute_no_transfer") else "qute"
            with np.load(out / name, allow_pickle=False) as data:
                assert {"probs", "sample_ids", "metadata", "labels", "confidence"} <= set(data.files), data.files
                assert data["labels"].dtype == np.int64
                assert np.array_equal(data["labels"], labels[data["sample_ids"]])
                assert np.allclose(data["confidence"], data["probs"].max(axis=1))
            probs, ids, meta, members = load_predictions(out / name)
            assert meta["method"] == method, meta["method"]
            assert meta["condition"] in name
            assert meta["forward_passes"] == 1
            assert tuple(members.shape) == (EVAL_N, 2, 10), members.shape
            assert metrics_from_file(out / name, protocol=p)["n"] == EVAL_N
check("QUTE and ablation files have the agreed names, labels and confidence", t_files_names_and_labels)

def t_skip_existing():
    p = short_protocol()
    loaders, _ = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_both(p, d)
        out = Path(d) / "out"
        first = run(p, loaders, ["qute"], [SEED], out, d)
        second = run(p, loaders, ["qute"], [SEED], out, d)
        third = run(p, loaders, ["qute"], [SEED], out, d, overwrite=True)
    assert len(first) == 2 and second == [] and len(third) == 2, (first, second, third)
check("existing QUTE files are skipped unless overwrite is set", t_skip_existing)

def t_reruns_identical():
    p = short_protocol()
    loaders, _ = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_both(p, d)
        run(p, loaders, ["qute"], [SEED], Path(d) / "a", d)
        run(p, loaders, ["qute"], [SEED], Path(d) / "b", d)
        with np.load(Path(d) / "a" / "qute_seed11_fog.npz") as a, \
             np.load(Path(d) / "b" / "qute_seed11_fog.npz") as b:
            assert np.array_equal(a["probs"], b["probs"])
check("QUTE predictions are deterministic across reruns", t_reruns_identical)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nQUTE evaluation checks pass.")
