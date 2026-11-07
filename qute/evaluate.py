"""Load a saved QUTE checkpoint for evaluation.

Prediction on the 16 conditions is driven by eval/predict_conditions.py, the
same script that predicts the baselines:

    python -m eval.predict_conditions --methods qute qute_no_transfer

This module only rebuilds a QUTE model from configs/base.yaml and loads its
saved weights, refusing a checkpoint that was trained under a different config
version or checkpoint rule, so predictions can never silently mix protocols.
"""

from typing import Tuple

from baselines.base import load_checkpoint
from models.qute import QUTEModel

from .train import build_qute_from_protocol, method_name, qute_checkpoint_path


def load_qute_checkpoint(
    protocol: dict,
    seed: int,
    transfer: bool,
    checkpoint_dir="checkpoints",
    device: str = "cpu",
) -> Tuple[QUTEModel, dict]:
    """Rebuild a QUTE model (or the no-transfer ablation) and load its weights."""
    path = qute_checkpoint_path(checkpoint_dir, int(protocol["config_version"]), seed, transfer)
    if not path.exists():
        command = "python -m qute.train" + ("" if transfer else " --no-transfer")
        raise FileNotFoundError(f"missing checkpoint {path}. Train it first with: {command}")

    payload = load_checkpoint(path, map_location="cpu")

    expected = method_name(transfer)
    if payload.get("method") != expected:
        raise ValueError(f"{path} holds method {payload.get('method')!r}, expected {expected!r}")
    if int(payload.get("config_version", -1)) != int(protocol["config_version"]):
        raise ValueError(
            f"{path} was trained under config v{payload.get('config_version')}, "
            f"configs/base.yaml is v{protocol['config_version']}. Retrain before evaluating."
        )
    rule = protocol["training"]["checkpoint_selection"]
    if payload.get("checkpoint_selection") != rule:
        raise ValueError(
            f"{path} was saved with checkpoint_selection={payload.get('checkpoint_selection')!r}, "
            f"configs/base.yaml says {rule!r}"
        )
    if rule == "final_epoch" and int(payload["epoch"]) != int(protocol["training"]["epochs"]):
        raise ValueError(
            f"{path} saved epoch {payload['epoch']}, but final_epoch should be "
            f"epoch {protocol['training']['epochs']}"
        )

    model = build_qute_from_protocol(protocol, seed)
    model.load_state_dict(payload["model_state_dict"])
    model.freeze_base(False)
    return model.to(device).eval(), payload
