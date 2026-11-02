"""Prediction scores shared by the baselines and QUTE."""

import torch

CONFIDENCE_NAME = "max_softmax_probability"


def max_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Confidence is the maximum softmax probability of each example."""
    if probs.ndim != 2 or probs.shape[1] != 10:
        raise ValueError(f"expected probabilities [N, 10], got {tuple(probs.shape)}")
    return probs.max(dim=1).values


def with_confidence(probs: torch.Tensor, extra: dict) -> tuple:
    """Attach the shared confidence definition to prediction metadata."""
    confidence = max_confidence(probs)
    if confidence.shape != (probs.shape[0],):
        raise ValueError(f"expected confidence [N], got {tuple(confidence.shape)}")
    merged = dict(extra)
    merged["confidence"] = CONFIDENCE_NAME
    return confidence, merged
