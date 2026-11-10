
"""Step 16: Accuracy-drop detection and AUPRC."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

WINDOW_SIZE = 100
STRIDE = 1


def window_means(values, size=WINDOW_SIZE):
    """Compute stride-1 means for all complete windows."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional array")
    if len(values) < size:
        raise ValueError("Not enough samples for one window")

    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[size:] - cumulative[:-size]) / size


def load_predictions(path):
    """Load and order predictions by original sample ID."""
    with np.load(path, allow_pickle=False) as data:
        probs = np.asarray(data["probs"])
        ids = np.asarray(data["sample_ids"])
        labels = np.asarray(data["labels"])
        confidence = np.asarray(data["confidence"])

    if probs.shape != (10000, 10):
        raise ValueError(f"{path}: unexpected probs shape {probs.shape}")

    if ids.shape != (10000,) or labels.shape != (10000,):
        raise ValueError(f"{path}: invalid IDs or labels")

    if confidence.shape != (10000,):
        raise ValueError(f"{path}: invalid confidence shape")

    order = np.argsort(ids)
    ids = ids[order]

    if not np.array_equal(ids, np.arange(10000)):
        raise ValueError(f"{path}: missing or duplicate sample IDs")

    probs = probs[order]
    labels = labels[order]
    confidence = confidence[order]

    if not np.allclose(confidence, probs.max(axis=1), atol=1e-5):
        raise ValueError(f"{path}: confidence does not match probs")

    return {
        "ids": ids,
        "labels": labels,
        "predictions": probs.argmax(axis=1),
        "confidence": confidence,
    }


def alarm_mask(mean_confidence, rho):
    """Alarm when mean confidence is strictly below rho."""
    return np.asarray(mean_confidence) < rho


def evaluate_detection(clean, corrupted):
    """Evaluate one clean-to-corrupted prediction stream."""

    # Verify paired conditions use the same underlying test labels.
    if not np.array_equal(clean["ids"], corrupted["ids"]):
        raise ValueError("Clean/corrupted sample ID mismatch")

    if not np.array_equal(clean["labels"], corrupted["labels"]):
        raise ValueError("Clean/corrupted label mismatch")

    clean_correct = (
        clean["predictions"] == clean["labels"]
    ).astype(float)

    corrupted_correct = (
        corrupted["predictions"] == corrupted["labels"]
    ).astype(float)

    # Reference distribution uses clean-only windows.
    clean_reference_accuracy = window_means(clean_correct)

    mu = float(np.mean(clean_reference_accuracy))
    sigma = float(np.std(clean_reference_accuracy))
    threshold = mu - 3.0 * sigma

    # Evaluation stream: clean first, then corrupted.
    correct = np.concatenate(
        (clean_correct, corrupted_correct)
    )
    confidence = np.concatenate(
        (clean["confidence"], corrupted["confidence"])
    )

    window_accuracy = window_means(correct)
    mean_confidence = window_means(confidence)

    events = window_accuracy < threshold
    positives = int(events.sum())

    # Low confidence represents stronger degradation evidence.
    risk_score = 1.0 - mean_confidence

    if positives == 0:
        ap = None
    else:
        ap = float(average_precision_score(events, risk_score))

    sweep = []

    for rho in np.linspace(0.0, 1.0, 11):
        alarms = alarm_mask(mean_confidence, rho)

        tp = int(np.sum(alarms & events))
        fp = int(np.sum(alarms & ~events))

        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / positives if positives else None

        sweep.append({
            "rho": round(float(rho), 1),
            "alarms": int(alarms.sum()),
            "precision": precision,
            "recall": recall,
        })

    return {
        "clean_reference_mean": mu,
        "clean_reference_std": sigma,
        "degradation_threshold": threshold,
        "total_windows": len(events),
        "degradation_events": positives,
        "average_precision": ap,
        "rho_sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        required=True,
        help="Path to clean condition NPZ",
    )
    parser.add_argument(
        "--corrupted",
        required=True,
        help="Path to corrupted condition NPZ",
    )
    args = parser.parse_args()

    clean = load_predictions(Path(args.clean))
    corrupted = load_predictions(Path(args.corrupted))

    result = evaluate_detection(clean, corrupted)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
