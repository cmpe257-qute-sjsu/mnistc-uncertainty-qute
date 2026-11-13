
import numpy as np

from eval.detection import (
    alarm_mask,
    evaluate_detection,
    window_means,
)


def make_condition(correct=True, confidence=0.9):
    labels = np.zeros(10000, dtype=int)
    predictions = np.zeros(10000, dtype=int)

    if not correct:
        predictions[:] = 1

    return {
        "ids": np.arange(10000),
        "labels": labels,
        "predictions": predictions,
        "confidence": np.full(10000, confidence),
    }


def test_window_boundaries():
    values = np.arange(20000, dtype=float)
    means = window_means(values)

    assert len(means) == 19901
    assert np.isclose(means[0], 49.5)
    assert np.isclose(means[-1], 19949.5)


def test_degradation_detected():
    clean = make_condition(correct=True, confidence=0.99)
    corrupted = make_condition(correct=False, confidence=0.55)

    result = evaluate_detection(clean, corrupted)

    assert result["total_windows"] == 19901
    assert result["degradation_events"] > 0
    assert result["average_precision"] is not None
    assert len(result["rho_sweep"]) == 11


def test_no_events_is_undefined():
    clean = make_condition()
    corrupted = make_condition()

    result = evaluate_detection(clean, corrupted)

    assert result["degradation_events"] == 0
    assert result["average_precision"] is None


def test_strict_alarm_threshold():
    means = np.array([0.2, 0.5, 0.8])
    alarms = alarm_mask(means, 0.5)

    assert alarms.tolist() == [True, False, False]


def test_sample_identity_mismatch():
    import pytest

    clean = make_condition()
    corrupted = make_condition()
    corrupted["ids"] = corrupted["ids"] + 1

    with pytest.raises(ValueError):
        evaluate_detection(clean, corrupted)
