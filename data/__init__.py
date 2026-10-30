
"""Public data-loading interface for the QUTE project."""

from .prepare_mnist import IndexedDataset, get_loaders

__all__ = ["IndexedDataset", "get_loaders"]
