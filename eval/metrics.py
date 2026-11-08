"""Calibration metrics computed from saved prediction files.

Every method (BASE, MC Dropout, deep ensemble, QUTE) is scored with the same
functions. Each takes the averaged probabilities a method deploys, [N, 10],
and the true labels, [N]. Nothing here runs a model.

Definitions
-----------
accuracy  fraction of examples whose argmax class equals the label
nll       mean of -log p(true class); p is clipped at 1e-12 so a confident
          wrong answer gives a large but finite value
brier     multi-class Brier score: mean over examples of
          sum_k (p_k - onehot_k)^2. Range 0 (perfect) to 2
ece       expected calibration error on max-softmax confidence:
          sum_b (n_b / N) * |accuracy_b - mean_confidence_b|

ECE binning follows configs/base.yaml (evaluation.ece_bins, ece_binning).
Our protocol uses 15 equal-mass bins: examples are sorted by confidence and
split into 15 groups of (nearly) equal size. The paper uses equal-width bins,
which is a recorded deviation; equal_width is available here so the two can be
compared in the report.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

NLL_EPS = 1e-12
BINNINGS = ("equal_mass", "equal_width")


def _as_numpy(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError(f"expected probabilities [N, C], got shape {probs.shape}")
    if labels.shape != (probs.shape[0],):
        raise ValueError(f"expected labels [{probs.shape[0]}], got shape {labels.shape}")
    if probs.shape[0] == 0:
        raise ValueError("no examples to score")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {labels.dtype}")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError(f"labels must be in [0, {probs.shape[1] - 1}]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probability rows do not sum to 1")
    return probs, labels.astype(np.int64)


def accuracy(probs, labels) -> float:
    probs, labels = _as_numpy(probs, labels)
    return float((probs.argmax(axis=1) == labels).mean())


def nll(probs, labels) -> float:
    probs, labels = _as_numpy(probs, labels)
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, NLL_EPS, 1.0)).mean())


def brier(probs, labels) -> float:
    probs, labels = _as_numpy(probs, labels)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _bin_index(confidence: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """Bin number (0 .. n_bins-1) for each example."""
    if binning == "equal_mass":
        # Stable sort, so ties keep their file order and the result is reproducible.
        order = np.argsort(confidence, kind="mergesort")
        bins = np.empty(len(confidence), dtype=np.int64)
        for b, members in enumerate(np.array_split(order, n_bins)):
            bins[members] = b
        return bins
    if binning == "equal_width":
        # Bins (0, 1/n], ..., ((n-1)/n, 1]; confidence 0 falls in the first bin.
        return np.clip(np.ceil(confidence * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    raise ValueError(f"binning must be one of {BINNINGS}, got {binning!r}")


def ece(probs, labels, n_bins: int = 15, binning: str = "equal_mass") -> float:
    probs, labels = _as_numpy(probs, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = _bin_index(confidence, n_bins, binning)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total / len(labels))


def calibration_metrics(probs, labels, protocol: Optional[dict] = None) -> Dict[str, float]:
    """accuracy, nll, brier and ece for one prediction set.

    ECE settings come from protocol["evaluation"] when a protocol is given,
    otherwise from configs/base.yaml.
    """
    if protocol is None:
        from baselines.base import load_protocol

        protocol = load_protocol()
    evaluation = protocol["evaluation"]
    n_bins = int(evaluation["ece_bins"])
    binning = evaluation["ece_binning"]
    return {
        "n": int(len(labels)),
        "accuracy": accuracy(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "ece": ece(probs, labels, n_bins=n_bins, binning=binning),
    }


def metrics_from_file(path, labels=None, protocol: Optional[dict] = None) -> Dict:
    """Score one saved prediction file.

    Uses the file's own `labels` array when it has one. Older files (the
    clean-only training outputs) have none; pass `labels` indexed by the
    file's sample_ids in that case.
    """
    import json

    data = np.load(Path(path), allow_pickle=False)
    probs = data["probs"]
    if "labels" in data:
        file_labels = data["labels"]
        if labels is not None and not np.array_equal(np.asarray(labels), file_labels):
            raise ValueError(f"{path}: labels passed in disagree with the labels stored in the file")
        labels = file_labels
    elif labels is None:
        raise ValueError(f"{path} has no labels array; pass labels aligned to its sample_ids")
    metadata = json.loads(str(data["metadata"]))
    result = calibration_metrics(probs, labels, protocol)
    for key in ("method", "seed", "condition", "n_members", "config_version", "git_commit"):
        if key in metadata:
            result[key] = metadata[key]
    result["file"] = str(path)
    return result
