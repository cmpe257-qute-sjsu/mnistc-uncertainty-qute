"""Shared model definitions for the CMPE 257 QUTE project."""

from .shared_cnn import (
    MODEL_VERSION,
    TAP_NAMES,
    CNNConfig,
    SharedCNN,
    build_model,
    enable_mc_dropout,
)
from .interface import (
    SCHEMA_VERSION,
    IndexedDataset,
    align_to,
    ensemble_proba,
    load_predictions,
    predict_proba,
    predict_proba_mc,
    save_predictions,
    stack_members,
)

__all__ = [
    "MODEL_VERSION", "TAP_NAMES", "CNNConfig", "SharedCNN", "build_model",
    "enable_mc_dropout", "SCHEMA_VERSION", "IndexedDataset", "align_to",
    "ensemble_proba", "load_predictions", "predict_proba", "predict_proba_mc",
    "save_predictions", "stack_members",
]
