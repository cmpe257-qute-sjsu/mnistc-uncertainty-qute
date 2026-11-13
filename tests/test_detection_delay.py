
"""Tests for clean-calibrated detection delay."""

import numpy as np
import pytest

from eval.detection_delay import evaluate_delay, summarize


def make_predictions(confidence, labels=None):
    """Create synthetic predictions for testing."""
    confidence = np.asarray(confidence, dtype=float)

    if labels is None:
        labels = np.zeros(len(confidence), dtype=int)

    return {
        "ids": np.arange(len(confidence)),
        "labels": np.asarray(labels),
        "confidence": confidence,
    }


def test_clean_threshold_first_percentile():
    """Threshold uses only clean-window confidence."""
    clean = make_predictions(np.full(10000, 0.9))
    corrupted = make_predictions(np.full(10000, 0.2))

    result = evaluate_delay(clean, corrupted)

    assert result["tau"] == pytest.approx(0.9)
    assert result["clean_windows"] == 9901


def test_first_alarm_delay():
    """First window ending after onset triggers at delay 1."""
    clean = make_predictions(np.full(10000, 0.9))
    corrupted = make_predictions(np.full(10000, 0.1))

    result = evaluate_delay(clean, corrupted)

    assert result["detected"] is True
    assert result["delay_samples"] == 1
    assert result["fully_corrupted_windows"] == 9901
    assert result["detection_rate"] == pytest.approx(1.0)


def test_no_detection():
    """Corruption that does not lower confidence is undetected."""
    clean = make_predictions(np.full(10000, 0.9))
    corrupted = make_predictions(np.full(10000, 0.95))

    result = evaluate_delay(clean, corrupted)

    assert result["detected"] is False
    assert result["delay_samples"] is None
    assert result["detection_rate"] == pytest.approx(0.0)


def test_strict_alarm_threshold():
    """Confidence equal to threshold does not alarm."""
    clean = make_predictions(np.full(10000, 0.9))
    corrupted = make_predictions(np.full(10000, 0.9))

    result = evaluate_delay(clean, corrupted)

    assert result["detected"] is False
    assert result["detection_rate"] == pytest.approx(0.0)


def test_sample_identity_mismatch():
    """Paired streams must use identical sample IDs and labels."""
    clean = make_predictions(np.full(10000, 0.9))
    corrupted = make_predictions(np.full(10000, 0.2))

    corrupted["ids"] = corrupted["ids"][::-1]

    with pytest.raises(ValueError, match="sample ID mismatch"):
        evaluate_delay(clean, corrupted)


def test_summary_excludes_undetected_from_median():
    """Median uses detected streams; coverage includes all."""
    rows = [
        {
            "variant": "qute",
            "severity": "mild",
            "delay_samples": 10,
            "detection_rate": 0.8,
        },
        {
            "variant": "qute",
            "severity": "mild",
            "delay_samples": None,
            "detection_rate": 0.0,
        },
    ]

    summary = summarize(rows)

    overall = next(
        row for row in summary
        if row["variant"] == "qute"
        and row["severity"] == "overall"
    )

    assert overall["streams"] == 2
    assert overall["detected_streams"] == 1
    assert overall["undetected_streams"] == 1
    assert overall["detection_coverage"] == pytest.approx(0.5)
    assert overall["median_delay_samples"] == pytest.approx(10.0)
    assert overall["mean_detection_rate"] == pytest.approx(0.4)
