import numpy as np
import pytest

from eval.detection import evaluate_detection
from eval.export_demo_data import RHO_GRID, calibration_entry, detection_stream
from eval.metrics import ece


def _fake(seed, accuracy, sharpness):
    rng = np.random.default_rng(seed)
    labels = np.random.default_rng(0).integers(0, 10, 10000)
    logits = rng.normal(0, 1, (10000, 10))
    correct = rng.random(10000) < accuracy
    target = np.where(correct, labels, (labels + 1) % 10)
    logits[np.arange(10000), target] += sharpness
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs, labels


def _stream_input(probs, labels):
    return {"ids": np.arange(10000), "labels": labels, "predictions": probs.argmax(axis=1), "confidence": probs.max(axis=1)}


@pytest.fixture
def stream():
    clean = _stream_input(*_fake(1, 0.97, 6.0))
    corrupted = _stream_input(*_fake(2, 0.55, 3.0))
    return clean, corrupted


def test_detection_matches_step16(stream):
    clean, corrupted = stream
    ours = detection_stream(clean, corrupted, full=True)
    reference = evaluate_detection(clean, corrupted)
    assert ours["ap"] == pytest.approx(reference["average_precision"])
    assert ours["events"] == reference["degradation_events"]
    assert ours["threshold"] == pytest.approx(reference["degradation_threshold"], abs=1e-4)
    for point in reference["rho_sweep"]:
        i = int(np.flatnonzero(np.isclose(RHO_GRID, point["rho"]))[0])
        if point["precision"] is not None:
            assert ours["precision"][i] == pytest.approx(point["precision"], abs=1e-4)
        assert ours["recall"][i] == pytest.approx(point["recall"], abs=1e-4)


def test_detection_delay_and_false_alarms(stream):
    clean, corrupted = stream
    ours = detection_stream(clean, corrupted, full=True)
    assert ours["delay"][0] is None  # rho = 0 never alarms
    assert ours["clean_false_alarm_rate"][-1] == 1.0  # rho = 1 always alarms
    delays = [d for d in ours["delay"] if d is not None]
    assert all(d >= 1 for d in delays)
    assert delays == sorted(delays, reverse=True)  # a higher rho alarms no later


def test_calibration_entry_matches_metrics():
    probs, labels = _fake(3, 0.8, 4.0)
    entry = calibration_entry(probs, labels)
    assert entry["ece"] == pytest.approx(ece(probs, labels, n_bins=15, binning="equal_mass"), abs=1e-4)
    gap = np.mean([abs(acc - conf) for conf, acc in entry["reliability"]])
    assert gap == pytest.approx(entry["ece"], abs=2e-3)  # equal-mass bins: ECE is the mean gap
    assert sum(entry["hist_correct"]) + sum(entry["hist_wrong"]) == 10000
