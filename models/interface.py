"""
Prediction interface for the CMPE 257 QUTE project.

Every method produces predictions through these functions, so that the metric
code downstream never has to care which model made them.

The contract
------------
A prediction file holds, for one (method, seed, condition):

    probs         float32 [N, 10]     averaged class probabilities
    sample_ids    int64   [N]         dataset indices, so order never matters
    member_probs  float32 [N, K, 10]  optional, per ensemble member
    metadata      dict                method, seed, condition, versions, commit

Two rules that are easy to get wrong:

    * Average PROBABILITIES across members, never logits and never predicted
      class IDs. ensemble_proba enforces the axis.
    * Keep sample_ids with the probabilities. Joining on position breaks the
      moment someone shuffles a loader.
"""

import json
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .shared_cnn import MODEL_VERSION, enable_mc_dropout

SCHEMA_VERSION = 1


class IndexedDataset(Dataset):
    """Wrap a dataset so every item carries its index.

    Use this for every evaluation loader. It is what makes sample_ids real
    rather than "whatever order the loader happened to use".
    """

    def __init__(self, base: Dataset, ids: Optional[Sequence[int]] = None):
        self.base = base
        self.ids = list(range(len(base))) if ids is None else list(ids)
        if len(self.ids) != len(base):
            raise ValueError("ids length does not match dataset length")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x, y = item[0], item[1]
        return x, y, self.ids[i]


def _check_probs(probs: torch.Tensor) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be [N, C], got shape {tuple(probs.shape)}")
    if not torch.isfinite(probs).all():
        raise ValueError("probs contain NaN or inf")
    if (probs < 0).any():
        raise ValueError("probs contain negative values")
    row_sums = probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        worst = (row_sums - 1).abs().max().item()
        raise ValueError(f"probs rows do not sum to 1 (worst error {worst:.2e})")


@torch.no_grad()
def predict_proba(model, loader, device="cpu"):
    """Single deterministic forward pass. Used by BASE and each ensemble member.

    Returns (probs [N, C] float32, sample_ids [N] int64), both torch tensors.
    """
    model.eval().to(device)
    chunks, ids = [], []
    for x, _y, idx in loader:
        logits = model(x.to(device))
        chunks.append(torch.softmax(logits, dim=1).cpu())
        ids.append(idx.cpu())

    probs = torch.cat(chunks).float()
    sample_ids = torch.cat(ids).long()
    _check_probs(probs)
    return probs, sample_ids


@torch.no_grad()
def predict_proba_mc(model, loader, passes: int, device="cpu", seed: Optional[int] = None):
    """MC Dropout: `passes` stochastic forward passes, averaged.

    Returns (mean_probs [N, C], sample_ids [N], member_probs [N, passes, C]).
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if seed is not None:
        torch.manual_seed(seed)

    enable_mc_dropout(model)
    model.to(device)

    members, sample_ids = [], None
    for p in range(passes):
        chunks, ids = [], []
        for x, _y, idx in loader:
            logits = model(x.to(device))
            chunks.append(torch.softmax(logits, dim=1).cpu())
            ids.append(idx.cpu())
        members.append(torch.cat(chunks).float())
        current = torch.cat(ids).long()
        if sample_ids is None:
            sample_ids = current
        elif not torch.equal(sample_ids, current):
            raise RuntimeError(
                "sample order changed between MC passes -- the evaluation "
                "loader must have shuffle=False"
            )

    member_probs = torch.stack(members, dim=1)        # [N, passes, C]
    mean_probs = ensemble_proba(member_probs)
    return mean_probs, sample_ids, member_probs


def ensemble_proba(member_probs: torch.Tensor) -> torch.Tensor:
    """Average across ENSEMBLE MEMBERS, not across classes.

    Input  [N, K, C]  ->  output [N, C]

    The axis is the whole point of this function. Averaging the wrong one
    produces something that still looks like a probability vector, which is
    why it is worth having a named function rather than a bare .mean().
    """
    if member_probs.ndim != 3:
        raise ValueError(
            f"expected [N, K, C], got {tuple(member_probs.shape)}. "
            "If you have a list of [N, C] tensors, torch.stack(..., dim=1) first."
        )
    mean = member_probs.mean(dim=1)        # dim=1 is K
    _check_probs(mean)
    return mean


def stack_members(member_list: List[torch.Tensor]) -> torch.Tensor:
    """List of K tensors, each [N, C]  ->  [N, K, C]. For deep ensembles."""
    if not member_list:
        raise ValueError("no members given")
    shapes = {tuple(m.shape) for m in member_list}
    if len(shapes) != 1:
        raise ValueError(f"members have different shapes: {shapes}")
    return torch.stack(member_list, dim=1)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_predictions(
    path, probs, sample_ids, method: str, seed: int, condition: str,
    member_probs=None, config_version: int = 1, extra: Optional[Dict] = None,
):
    """Write one prediction file. `condition` is "clean" or a corruption name."""
    _check_probs(probs)
    if len(sample_ids) != len(probs):
        raise ValueError("sample_ids and probs have different lengths")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "config_version": config_version,
        "method": method,
        "seed": seed,
        "condition": condition,
        "n_samples": int(len(probs)),
        "n_classes": int(probs.shape[1]),
        "n_members": int(member_probs.shape[1]) if member_probs is not None else 1,
        "git_commit": _git_commit(),
    }
    if extra:
        metadata.update(extra)

    arrays = {
        "probs": probs.numpy().astype(np.float32),
        "sample_ids": sample_ids.numpy().astype(np.int64),
        "metadata": np.array(json.dumps(metadata)),
    }
    if member_probs is not None:
        arrays["member_probs"] = member_probs.numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)
    return metadata


def load_predictions(path):
    """Read a prediction file back. Returns (probs, sample_ids, metadata, member_probs)."""
    d = np.load(path, allow_pickle=False)
    probs = torch.from_numpy(d["probs"])
    sample_ids = torch.from_numpy(d["sample_ids"])
    metadata = json.loads(str(d["metadata"]))
    member_probs = torch.from_numpy(d["member_probs"]) if "member_probs" in d else None
    _check_probs(probs)
    return probs, sample_ids, metadata, member_probs


def align_to(probs: torch.Tensor, sample_ids: torch.Tensor, reference_ids: torch.Tensor):
    """Reorder predictions to match a reference ID order.

    Use this before comparing two methods. Never assume two files are already
    in the same order.
    """
    lookup = {int(s): i for i, s in enumerate(sample_ids.tolist())}
    missing = [int(r) for r in reference_ids.tolist() if int(r) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} reference IDs missing, e.g. {missing[:5]}")
    order = torch.tensor([lookup[int(r)] for r in reference_ids.tolist()], dtype=torch.long)
    return probs[order]
