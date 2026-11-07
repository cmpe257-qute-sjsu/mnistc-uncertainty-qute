"""
Checks for eval/metrics.py.

    python tests/test_metrics.py

Uses tiny hand-computed examples. Does not need MNIST, MNIST-C or checkpoints.

Worked example used below (2 classes, 4 examples):

    probs                 label  argmax  correct  confidence
    [0.9, 0.1]            0      0       yes      0.9
    [0.6, 0.4]            1      0       no       0.6
    [0.3, 0.7]            1      1       yes      0.7
    [0.2, 0.8]            1      1       yes      0.8

    accuracy = 3/4 = 0.75
    nll      = -(ln 0.9 + ln 0.4 + ln 0.7 + ln 0.8) / 4
    brier    = (0.02 + 0.72 + 0.18 + 0.08) / 4 = 0.25
    ECE, 2 equal-mass bins: {0.6 no, 0.7 yes} acc 0.5 conf 0.65 -> 0.15
                            {0.8 yes, 0.9 yes} acc 1.0 conf 0.85 -> 0.15
                            ECE = 0.5 * 0.15 + 0.5 * 0.15 = 0.15
    ECE, 2 equal-width bins: all four in (0.5, 1], acc 0.75 conf 0.75 -> 0.0
"""

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import load_protocol
from eval.metrics import accuracy, brier, calibration_metrics, ece, metrics_from_file, nll

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def close(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


PROBS = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
LABELS = np.array([0, 1, 1, 1])


def raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected ValueError mentioning {text!r}")


print("\nWORKED EXAMPLE")
print("--------------")

check("accuracy = 0.75", lambda: close(accuracy(PROBS, LABELS), 0.75))
check(
    "nll = mean -log p(true class)",
    lambda: close(nll(PROBS, LABELS), -(math.log(0.9) + math.log(0.4) + math.log(0.7) + math.log(0.8)) / 4),
)
check("brier = 0.25 (sum over classes, mean over examples)", lambda: close(brier(PROBS, LABELS), 0.25))
check("ece, 2 equal-mass bins = 0.15", lambda: close(ece(PROBS, LABELS, n_bins=2, binning="equal_mass"), 0.15))
check("ece, 2 equal-width bins = 0.0", lambda: close(ece(PROBS, LABELS, n_bins=2, binning="equal_width"), 0.0))
check("ece, 1 bin = |accuracy - mean confidence|", lambda: close(ece(PROBS, LABELS, n_bins=1), 0.0))


print("\nEDGE CASES")
print("----------")

def t_perfect():
    probs = np.eye(10)[[3, 1, 4, 1, 5]]
    labels = np.array([3, 1, 4, 1, 5])
    assert accuracy(probs, labels) == 1.0
    close(nll(probs, labels), 0.0)
    close(brier(probs, labels), 0.0)
    close(ece(probs, labels, n_bins=15), 0.0)
check("one-hot correct predictions score perfectly", t_perfect)

def t_confident_wrong():
    probs = np.eye(10)[[0]]
    labels = np.array([1])
    assert accuracy(probs, labels) == 0.0
    assert math.isfinite(nll(probs, labels)) and nll(probs, labels) > 20
    close(brier(probs, labels), 2.0)
    close(ece(probs, labels, n_bins=15), 1.0)
check("a confident wrong answer gives finite NLL, Brier 2, ECE 1", t_confident_wrong)

def t_more_bins_than_examples():
    close(ece(PROBS, LABELS, n_bins=15, binning="equal_mass"),
          ece(PROBS, LABELS, n_bins=4, binning="equal_mass"))
check("equal-mass ECE with more bins than examples ignores empty bins", t_more_bins_than_examples)

def t_ties_are_reproducible():
    rng = np.random.default_rng(0)
    probs = np.full((300, 10), 0.05)
    probs[:, 0] = 0.55
    labels = rng.integers(0, 10, 300)
    first = ece(probs, labels, n_bins=15)
    second = ece(probs, labels, n_bins=15)
    assert first == second
    close(first, abs((labels == 0).mean() - 0.55))
check("tied confidences give a reproducible ECE", t_ties_are_reproducible)

def t_equal_mass_bin_sizes():
    from eval.metrics import _bin_index
    bins = _bin_index(np.linspace(0, 1, 10000), 15, "equal_mass")
    counts = np.bincount(bins, minlength=15)
    assert counts.min() >= 666 and counts.max() <= 667, counts
check("15 equal-mass bins over 10k examples hold 666 or 667 each", t_equal_mass_bin_sizes)


print("\nINPUT CHECKS")
print("------------")

check("rows that do not sum to 1 are rejected",
      lambda: raises(lambda: accuracy(np.array([[0.5, 0.2]]), np.array([0])), "sum to 1"))
check("labels out of range are rejected",
      lambda: raises(lambda: accuracy(PROBS, np.array([0, 1, 2, 1])), "labels must be in"))
check("label count must match",
      lambda: raises(lambda: accuracy(PROBS, np.array([0, 1])), "expected labels"))
check("float labels are rejected",
      lambda: raises(lambda: accuracy(PROBS, LABELS.astype(float)), "integers"))
check("unknown binning is rejected",
      lambda: raises(lambda: ece(PROBS, LABELS, binning="adaptive"), "binning"))


print("\nPROTOCOL AND FILES")
print("------------------")

def t_protocol_settings():
    protocol = load_protocol()
    assert protocol["evaluation"]["ece_bins"] == 15
    assert protocol["evaluation"]["ece_binning"] == "equal_mass"
    result = calibration_metrics(PROBS, LABELS, protocol)
    close(result["ece"], ece(PROBS, LABELS, n_bins=15, binning="equal_mass"))
    assert set(result) == {"n", "accuracy", "nll", "brier", "ece"}
    assert result["n"] == 4
check("calibration_metrics reads ECE settings from base.yaml", t_protocol_settings)

def write_npz(path, with_labels):
    metadata = {"method": "base", "seed": 11, "condition": "clean", "n_members": 1, "config_version": 5}
    arrays = {
        "probs": PROBS.astype(np.float32),
        "sample_ids": np.arange(4, dtype=np.int64),
        "metadata": np.array(json.dumps(metadata)),
    }
    if with_labels:
        arrays["labels"] = LABELS.astype(np.int64)
    np.savez(path, **arrays)

def t_file_with_labels():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "base_seed11_clean.npz"
        write_npz(path, with_labels=True)
        result = metrics_from_file(path)
    close(result["accuracy"], 0.75)
    assert result["method"] == "base" and result["seed"] == 11 and result["condition"] == "clean"
check("a file with labels is scored and keeps its metadata", t_file_with_labels)

def t_file_without_labels():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "base_seed11_clean.npz"
        write_npz(path, with_labels=False)
        raises(lambda: metrics_from_file(path), "no labels")
        close(metrics_from_file(path, labels=LABELS)["accuracy"], 0.75)
check("a file without labels needs labels passed in", t_file_without_labels)

def t_label_mismatch():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "base_seed11_clean.npz"
        write_npz(path, with_labels=True)
        raises(lambda: metrics_from_file(path, labels=np.array([1, 1, 1, 1])), "disagree")
check("passed labels that disagree with the file are rejected", t_label_mismatch)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
print("\nCalibration metric checks pass.")
