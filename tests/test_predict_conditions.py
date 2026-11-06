"""
Checks for eval/predict_conditions.py.

    python tests/test_predict_conditions.py

Uses untrained checkpoints and tiny synthetic loaders for two conditions, so
it needs no MNIST-C and no real training. It checks file names, the added
labels / confidence arrays, that reruns match, skip / overwrite behaviour,
and that mismatched checkpoints are refused.
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import build_cnn_model, checkpoint_path, load_protocol
from baselines.ensemble import ensemble_checkpoint_path, member_init_seeds
from baselines.mc_dropout import mc_checkpoint_path
from eval.metrics import metrics_from_file
from eval.predict_conditions import METHODS, attach_labels, labels_by_id, run
from models import IndexedDataset, load_predictions

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


N = 12
SEED = 11


def fake_loaders():
    """Two conditions with the same labels and ids, different images."""
    generator = torch.Generator().manual_seed(5)
    labels = torch.randint(0, 10, (N,), generator=generator)
    loaders = {}
    for condition in ("clean", "shot_noise"):
        images = torch.rand(N, 1, 28, 28, generator=generator)
        dataset = IndexedDataset(TensorDataset(images, labels), ids=list(range(N)))
        loaders[condition] = DataLoader(dataset, batch_size=5, shuffle=False)
    return loaders, labels.numpy()


def save_ckpt(path, protocol, method, seed, dropout_rate):
    model = build_cnn_model(protocol, seed, dropout_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "seed": seed,
            "dropout_rate": dropout_rate,
            "config_version": int(protocol["config_version"]),
            "checkpoint_selection": protocol["training"]["checkpoint_selection"],
            "epoch": 20,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def write_checkpoints(directory, protocol):
    version = int(protocol["config_version"])
    save_ckpt(checkpoint_path(directory, version, SEED), protocol, "base", SEED, 0.0)
    rate = float(protocol["baselines"]["mc_dropout"]["rate"])
    save_ckpt(mc_checkpoint_path(directory, version, SEED), protocol, "mc_dropout", SEED, rate)
    reference = int(protocol["baselines"]["deep_ensemble"]["members_reference"])
    for index, member_seed in enumerate(member_init_seeds(SEED, reference), start=1):
        save_ckpt(ensemble_checkpoint_path(directory, version, SEED, index, member_seed),
                  protocol, "deep_ensemble", member_seed, 0.0)


EXPECTED = [
    "base_seed11_{c}.npz",
    "mc_dropout_seed11_{c}_passes2.npz",
    "mc_dropout_seed11_{c}_passes20.npz",
    "deep_ensemble_seed11_{c}_members2.npz",
    "deep_ensemble_seed11_{c}_members5.npz",
]


print("\nPREDICT ALL BASELINES ON TWO CONDITIONS")
print("---------------------------------------")

protocol = load_protocol()

def t_full_run():
    loaders, labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt, out = Path(d) / "ckpt", Path(d) / "out"
        write_checkpoints(ckpt, protocol)
        written = run(protocol, loaders, list(METHODS), [SEED], out, ckpt)
        names = sorted(p.name for p in out.glob("*.npz"))
        expected = sorted(t.format(c=c) for t in EXPECTED for c in ("clean", "shot_noise"))
        assert names == expected, names
        assert len(written) == 10
        assert not list(out.glob("*.tmp.npz")), "temporary files left behind"

        for name in names:
            with np.load(out / name, allow_pickle=False) as data:
                assert {"probs", "sample_ids", "metadata", "labels", "confidence"} <= set(data.files), data.files
                assert data["labels"].dtype == np.int64
                assert np.array_equal(data["labels"], labels[data["sample_ids"]])
                assert np.allclose(data["confidence"], data["probs"].max(axis=1))
            probs, ids, meta, _members = load_predictions(out / name)
            assert meta["condition"] in name and meta["seed"] == SEED
            assert metrics_from_file(out / name, protocol=protocol)["n"] == N
check("10 files with the agreed names, each with labels and confidence", t_full_run)

def t_member_shapes():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt, out = Path(d) / "ckpt", Path(d) / "out"
        write_checkpoints(ckpt, protocol)
        run(protocol, {"clean": loaders["clean"]}, ["mc_dropout", "deep_ensemble"], [SEED], out, ckpt)
        shapes = {}
        for name in ("mc_dropout_seed11_clean_passes2", "mc_dropout_seed11_clean_passes20",
                     "deep_ensemble_seed11_clean_members2", "deep_ensemble_seed11_clean_members5"):
            with np.load(out / f"{name}.npz") as data:
                shapes[name] = data["member_probs"].shape
        assert shapes["mc_dropout_seed11_clean_passes2"] == (N, 2, 10)
        assert shapes["mc_dropout_seed11_clean_passes20"] == (N, 20, 10)
        assert shapes["deep_ensemble_seed11_clean_members2"] == (N, 2, 10)
        assert shapes["deep_ensemble_seed11_clean_members5"] == (N, 5, 10)
        with np.load(out / "mc_dropout_seed11_clean_passes2.npz") as h, \
             np.load(out / "mc_dropout_seed11_clean_passes20.npz") as r:
            assert np.allclose(h["member_probs"], r["member_probs"][:, :2, :])
check("headline files are the first passes / members of the reference run", t_member_shapes)

def t_rerun_matches():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt"
        write_checkpoints(ckpt, protocol)
        run(protocol, loaders, ["mc_dropout"], [SEED], Path(d) / "a", ckpt)
        run(protocol, loaders, ["mc_dropout"], [SEED], Path(d) / "b", ckpt)
        for name in ("mc_dropout_seed11_shot_noise_passes20.npz", "mc_dropout_seed11_clean_passes2.npz"):
            with np.load(Path(d) / "a" / name) as a, np.load(Path(d) / "b" / name) as b:
                assert np.array_equal(a["probs"], b["probs"]), name
check("MC Dropout reruns give identical probabilities", t_rerun_matches)


print("\nSKIP AND OVERWRITE")
print("------------------")

def t_skip_existing():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt, out = Path(d) / "ckpt", Path(d) / "out"
        write_checkpoints(ckpt, protocol)
        first = run(protocol, loaders, ["base"], [SEED], out, ckpt)
        before = (out / "base_seed11_clean.npz").stat().st_mtime_ns
        second = run(protocol, loaders, ["base"], [SEED], out, ckpt)
        assert len(first) == 2 and second == [], second
        assert (out / "base_seed11_clean.npz").stat().st_mtime_ns == before
        third = run(protocol, loaders, ["base"], [SEED], out, ckpt, overwrite=True)
        assert len(third) == 2
check("existing files are skipped unless overwrite is set", t_skip_existing)

def t_half_pair_redone():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt, out = Path(d) / "ckpt", Path(d) / "out"
        write_checkpoints(ckpt, protocol)
        run(protocol, {"clean": loaders["clean"]}, ["deep_ensemble"], [SEED], out, ckpt)
        (out / "deep_ensemble_seed11_clean_members5.npz").unlink()
        written = run(protocol, {"clean": loaders["clean"]}, ["deep_ensemble"], [SEED], out, ckpt)
        assert sorted(p.name for p in written) == [
            "deep_ensemble_seed11_clean_members2.npz", "deep_ensemble_seed11_clean_members5.npz"]
check("a half-finished headline/reference pair is redone as a pair", t_half_pair_redone)


print("\nREFUSALS")
print("--------")

def expect(exc_type, text, fn):
    try:
        fn()
    except exc_type as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected {exc_type.__name__} mentioning {text!r}")

def t_missing_checkpoint():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        expect(FileNotFoundError, "missing checkpoint",
               lambda: run(protocol, loaders, ["base"], [SEED], Path(d) / "out", Path(d) / "none"))
check("a missing checkpoint is reported by path", t_missing_checkpoint)

def t_old_config():
    loaders, _labels = fake_loaders()
    old = copy.deepcopy(protocol)
    old["config_version"] = int(protocol["config_version"]) - 1
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt"
        save_ckpt(checkpoint_path(ckpt, int(protocol["config_version"]), SEED), old, "base", SEED, 0.0)
        expect(ValueError, "config v", lambda: run(protocol, loaders, ["base"], [SEED], Path(d) / "out", ckpt))
check("a checkpoint from another config version is refused", t_old_config)

def t_wrong_method():
    loaders, _labels = fake_loaders()
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt"
        save_ckpt(checkpoint_path(ckpt, int(protocol["config_version"]), SEED), protocol, "mc_dropout", SEED, 0.0)
        expect(ValueError, "expected 'base'", lambda: run(protocol, loaders, ["base"], [SEED], Path(d) / "out", ckpt))
check("a checkpoint for the wrong method is refused", t_wrong_method)

def t_unknown_method():
    loaders, _labels = fake_loaders()
    expect(ValueError, "unknown methods", lambda: run(protocol, loaders, ["qute"], [SEED]))
check("an unregistered method is refused", t_unknown_method)

def t_label_alignment():
    loaders, labels = fake_loaders()
    label_map = labels_by_id(loaders["clean"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.npz"
        order = np.arange(N)[::-1]
        probs = np.full((N, 10), 0.1, dtype=np.float32)
        np.savez(path, probs=probs, sample_ids=order.astype(np.int64), metadata=np.array("{}"))
        attach_labels(path, label_map)
        with np.load(path) as data:
            assert np.array_equal(data["labels"], labels[order])
        np.savez(path, probs=probs[:3], sample_ids=np.array([0, 1, 99]), metadata=np.array("{}"))
        expect(ValueError, "no label", lambda: attach_labels(path, label_map))
check("labels follow sample_ids, and unknown ids are refused", t_label_alignment)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
print("\nPrediction-driver checks pass.")
