"""
Checks for the QUTE evaluation script.

    python tests/test_qute_eval.py

Synthetic images and a 2-epoch model, no MNIST or MNIST-C needed. These verify
that a checkpoint reloads into exactly the model that was trained, that one
prediction file is written per condition with the right metadata, that existing
files are kept, and that a checkpoint from a different protocol is refused.
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IndexedDataset, load_predictions
from baselines.base import load_protocol
from qute.train import predict_proba_qute, qute_prediction_path, train_qute
from qute.evaluate import (
    check_condition_loaders,
    evaluate_conditions,
    load_qute_checkpoint,
    run,
)

PASS, FAIL = [], []
BATCH, N, EVAL_N = 8, 24, 12


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
    p["training"]["seeds"] = [11]
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
    """Stand-ins for get_corruption_loaders: one indexed loader per condition."""
    loaders = {}
    for i, name in enumerate(names):
        g = torch.Generator().manual_seed(100 + i)
        x = torch.rand(EVAL_N, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (EVAL_N,), generator=g)
        ds = IndexedDataset(TensorDataset(x, y), ids=list(range(EVAL_N)))
        loaders[name] = DataLoader(ds, batch_size=BATCH, shuffle=False)
    return loaders


print("\nLOADING CHECKPOINTS")
print("-------------------")

def t_reload_matches_trained_model():
    p = short_protocol()
    train, val = train_loaders()
    loaders = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        trained, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        reloaded, payload = load_qute_checkpoint(p, 11, True, checkpoint_dir=d)
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
            load_qute_checkpoint(p, 11, False, checkpoint_dir=d)
        except FileNotFoundError as exc:
            assert "--no-transfer" in str(exc), exc
            return
    raise AssertionError("a missing checkpoint was not reported")
check("a missing checkpoint says which training command to run", t_missing_checkpoint_names_the_command)

def t_refuses_other_protocol():
    p = short_protocol()
    train, val = train_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        other = copy.deepcopy(p)
        other["training"]["checkpoint_selection"] = "min_val_nll"
        try:
            load_qute_checkpoint(other, 11, True, checkpoint_dir=d)
        except ValueError as exc:
            assert "checkpoint_selection" in str(exc), exc
            return
    raise AssertionError("a checkpoint from a different rule was accepted")
check("a checkpoint saved under a different rule is refused", t_refuses_other_protocol)


print("\nPREDICTION FILES")
print("----------------")

def t_one_file_per_condition():
    p = short_protocol()
    train, val = train_loaders()
    loaders = condition_loaders(("clean", "fog", "zigzag"))
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        model, _ = load_qute_checkpoint(p, 11, True, checkpoint_dir=d)
        preds = Path(d) / "preds"
        written, kept = evaluate_conditions(model, loaders, 11, p, True, preds_dir=preds)
        assert len(written) == 3 and kept == [], (written, kept)
        for name in loaders:
            path = qute_prediction_path(11, name, True, directory=preds)
            probs, ids, meta, members = load_predictions(path)
            assert tuple(probs.shape) == (EVAL_N, 10)
            assert tuple(members.shape) == (EVAL_N, 2, 10)
            assert ids.tolist() == list(range(EVAL_N))
            assert meta["condition"] == name, meta["condition"]
            assert meta["method"] == "qute"
            assert meta["forward_passes"] == 1
check("one prediction file per condition, with the condition in its metadata", t_one_file_per_condition)

def t_existing_files_are_kept():
    p = short_protocol()
    train, val = train_loaders()
    loaders = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        model, _ = load_qute_checkpoint(p, 11, True, checkpoint_dir=d)
        preds = Path(d) / "preds"
        evaluate_conditions(model, loaders, 11, p, True, preds_dir=preds)
        written, kept = evaluate_conditions(model, loaders, 11, p, True, preds_dir=preds)
        assert written == [] and len(kept) == 2, (written, kept)
        written, kept = evaluate_conditions(model, loaders, 11, p, True, preds_dir=preds, overwrite=True)
        assert len(written) == 2 and kept == [], (written, kept)
check("existing files are kept, and --overwrite redoes them", t_existing_files_are_kept)

def t_run_covers_both_variants():
    p = short_protocol()
    train, val = train_loaders()
    loaders = condition_loaders()
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, transfer=True, checkpoint_dir=d)
        train_qute(train, val, seed=11, protocol=p, transfer=False, checkpoint_dir=d)
        preds = Path(d) / "preds"
        results = run(p, loaders, variants=(True, False), checkpoint_dir=d, preds_dir=preds)
        assert set(results) == {("qute", 11), ("qute_no_transfer", 11)}, set(results)
        for transfer, method in ((True, "qute"), (False, "qute_no_transfer")):
            for name in loaders:
                path = qute_prediction_path(11, name, transfer, directory=preds)
                assert path.exists(), path
                assert load_predictions(path)[2]["method"] == method
check("run writes files for QUTE and the ablation, kept apart by name", t_run_covers_both_variants)

def t_loader_check():
    good = condition_loaders(("clean", "fog"))
    check_condition_loaders(good, expected_conditions=2, expected_samples=EVAL_N)
    try:
        check_condition_loaders(condition_loaders(("fog", "zigzag")),
                                expected_conditions=2, expected_samples=EVAL_N)
    except ValueError as exc:
        assert "clean" in str(exc), exc
        return
    raise AssertionError("loaders without 'clean' were accepted")
check("the condition-loader check requires clean and the right counts", t_loader_check)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nQUTE evaluation checks pass. Run python -m qute.evaluate once training has finished.")
