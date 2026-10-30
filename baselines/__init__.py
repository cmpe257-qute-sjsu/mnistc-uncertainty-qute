"""BASE, MC Dropout, and deep-ensemble baselines."""

from eval.scores import max_confidence

from .base import (
    build_base_model,
    evaluate_base,
    load_protocol,
    prediction_path,
    require_loaders,
    train_base,
)
from .ensemble import (
    evaluate_deep_ensemble,
    member_init_seeds,
    train_deep_ensemble,
)
from .mc_dropout import (
    evaluate_mc_dropout,
    evaluate_mc_reference_and_headline,
    train_mc_dropout,
)

__all__ = [
    "build_base_model",
    "evaluate_base",
    "evaluate_deep_ensemble",
    "evaluate_mc_dropout",
    "evaluate_mc_reference_and_headline",
    "load_protocol",
    "max_confidence",
    "member_init_seeds",
    "prediction_path",
    "require_loaders",
    "train_base",
    "train_deep_ensemble",
    "train_mc_dropout",
]
