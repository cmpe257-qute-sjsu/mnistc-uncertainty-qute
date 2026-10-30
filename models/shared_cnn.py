"""
Shared 4-layer CNN for the CMPE 257 QUTE project.

This is the common base network. Every method in the study uses it:

    BASE            SharedCNN(dropout_rate=0.0)
    MC Dropout      SharedCNN(dropout_rate=0.1)      <- dropout at the EE locations
    Deep ensemble   SharedCNN(dropout_rate=0.0) x N, different seeds
    QUTE            SharedCNN as the base, with early exits attached at the taps

Owner: Smitha. If you need this to change, say so in the team chat and bump
model_version -- Anusha's baselines and the QUTE model both depend on it.

Two things make this reusable rather than just "a CNN":

1. Named feature taps after conv layers 1 and 2. QUTE attaches its early exits
   there, and MC Dropout puts its dropout at the same places. The paper does
   this deliberately so the two methods are comparable.

2. The dropout_rate argument. Anusha does NOT need to write a separate
   dropout model -- passing a rate puts dropout in exactly the right spots.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

MODEL_VERSION = 1
TAP_NAMES = ("layer1", "layer2")


@dataclass
class CNNConfig:
    """Mirrors the `model:` block of configs/base.yaml."""
    in_channels: int = 1
    num_classes: int = 10
    channels: List[int] = field(default_factory=lambda: [8, 16, 32, 32])
    kernel_size: int = 3
    pool_after: List[int] = field(default_factory=lambda: [0, 1])  # conv indices
    use_batchnorm: bool = False
    dropout_rate: float = 0.0        # > 0 only for MC Dropout
    head: str = "gap"                # "gap" or "flatten"

    def __post_init__(self):
        if len(self.channels) != 4:
            raise ValueError(f"expected 4 conv layers, got {len(self.channels)}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate out of range: {self.dropout_rate}")
        if self.head not in ("gap", "flatten"):
            raise ValueError(f"unknown head: {self.head}")


class SharedCNN(nn.Module):
    """4 conv layers, feature taps after layers 1 and 2.

    forward(x)                      -> logits [N, num_classes]
    forward(x, return_taps=True)    -> (logits, {"layer1": t, "layer2": t})
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__()
        self.config = config or CNNConfig()
        c = self.config

        blocks = []
        in_ch = c.in_channels
        for i, out_ch in enumerate(c.channels):
            layers = [nn.Conv2d(in_ch, out_ch, c.kernel_size, padding=c.kernel_size // 2)]
            if c.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if i in c.pool_after:
                layers.append(nn.MaxPool2d(2))
            # Dropout sits at the EE locations only, matching the paper.
            if c.dropout_rate > 0.0 and i in c.pool_after:
                layers.append(nn.Dropout(c.dropout_rate))
            blocks.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.block1, self.block2, self.block3, self.block4 = blocks

        if c.head == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = c.channels[-1]
        else:
            self.pool = nn.Identity()
            side = 28 // (2 ** len(c.pool_after))
            feat_dim = c.channels[-1] * side * side

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(feat_dim, c.num_classes)
        self.feature_dim = feat_dim

    def forward(self, x, return_taps: bool = False):
        taps: Dict[str, torch.Tensor] = {}

        x = self.block1(x)
        taps["layer1"] = x
        x = self.block2(x)
        taps["layer2"] = x
        x = self.block3(x)
        x = self.block4(x)

        logits = self.classifier(self.flatten(self.pool(x)))
        return (logits, taps) if return_taps else logits

    def tap_shapes(self, input_size=(1, 1, 28, 28)) -> Dict[str, tuple]:
        """Shapes at each tap. QUTE needs these to size its early exits."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, taps = self.forward(torch.zeros(*input_size), return_taps=True)
        if was_training:
            self.train()
        return {k: tuple(v.shape) for k, v in taps.items()}

    def describe(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "model_version": MODEL_VERSION,
            "parameters": total,
            "approx_fp32_kib": round(total * 4 / 1024, 1),
            "feature_dim": self.feature_dim,
            "taps": self.tap_shapes(),
            "config": asdict(self.config),
        }


def build_model(config: Optional[CNNConfig] = None, seed: Optional[int] = None) -> SharedCNN:
    """Build a model with reproducible initialization.

    Deep-ensemble members must each get a DIFFERENT seed. Never build one model
    and copy it into several members.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return SharedCNN(config)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but reactivate dropout.

    This is what MC Dropout needs: batchnorm and everything else behave as at
    inference, only dropout stays stochastic. Calling model.train() instead
    would also change batchnorm, which is not what we want.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
