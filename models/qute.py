"""
QUTE architecture: early-exit-assisted ensemble on the shared CNN.

Owner: Smitha. This is the model only -- the training loop lives in
qute/train.py, because QUTE's per-batch weight copy and three-term loss do not
fit inside the shared fit_cnn.

The shape constraint that makes the whole thing work
----------------------------------------------------
The paper's early-view assistance copies the weights of each early-exit layer
h_theta into the matching output-block layer h_phi before every training batch.
A copy only works if the two tensors have identical shape.

Both are single depthwise convolutions. A depthwise conv over C channels has
weight shape [C, 1, k, k] -- it depends on the channel count, not on the spatial
size. h_phi sits on the base network's final features f_D, which have 32
channels here, so h_phi.weight is [32, 1, 3, 3].

Each early exit therefore needs 32 channels at h_theta too, which is what the
pointwise convolution in front of it is for: 8 -> 32 at exit 1, 16 -> 32 at
exit 2. The spatial sizes still differ (14x14 vs 7x7 vs 7x7) and that is fine,
because depthwise weights carry no spatial extent.

What is trained and what is deployed
------------------------------------
Training graph:  base network + K early exits + K output blocks + the base's
                 own classifier (the "final exit", f_out).
Deployed graph:  base network + K output blocks.

Both the early exits AND the base's original classifier are removed at
inference. The paper drops the original classifier because its overconfidence
degrades calibration -- not merely because it is redundant. Forgetting that is
the easiest silent mistake in this implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_cnn import CNNConfig, SharedCNN, build_model

QUTE_VERSION = 1


@dataclass
class QUTEConfig:
    """Mirrors the `qute:` block of configs/base.yaml."""
    K: int = 2
    early_exit_after_layers: List[int] = field(default_factory=lambda: [1, 2])
    w_ev_0: float = 3.0
    w_ev_delta: float = 0.5
    tau: Optional[List[float]] = None          # EE loss weights; see tau_weights()
    kernel_size: int = 3
    freeze_base_last_fraction: float = 0.10
    transfer_during_freeze: bool = True
    drop_original_output_block: bool = True

    def __post_init__(self):
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if len(self.early_exit_after_layers) != self.K:
            raise ValueError(
                f"K={self.K} but early_exit_after_layers has "
                f"{len(self.early_exit_after_layers)} entries"
            )
        if sorted(set(self.early_exit_after_layers)) != sorted(self.early_exit_after_layers):
            raise ValueError("early_exit_after_layers must be distinct")
        for layer in self.early_exit_after_layers:
            if layer not in (1, 2, 3):
                raise ValueError(
                    f"early exit after layer {layer} is not a tap on a 4-layer CNN"
                )
        if self.tau is not None and len(self.tau) != self.K:
            raise ValueError(f"tau has {len(self.tau)} entries, expected {self.K}")


def ev_weights(config: QUTEConfig) -> List[float]:
    """Output-block loss weights: w_EV(0) = w_ev_0, w_EV(k) = w_EV(k-1) + delta.

    [paper] w_ev_0 = 3 and delta = 0.5, so K = 2 gives [3.0, 3.5]. The paper
    tested w_ev_0 in {2, 3, 4, 5} and chose 3, because larger values let the
    output-block losses drown out the early exits.
    """
    return [config.w_ev_0 + k * config.w_ev_delta for k in range(config.K)]


def tau_weights(config: QUTEConfig, depth: int = 4) -> List[float]:
    """Early-exit loss weights.

    [ours] The paper applies a weighting tau_k to the early-exit losses and
    cites Kaya et al. (2019) for it, but never gives the values. We default to
    the relative depth of each exit, which is what that paper does: an exit
    after layer 1 of 4 gets 0.25, after layer 2 gets 0.50.

    This is a deviation and belongs in the report. Override it by setting
    `qute.tau` in configs/base.yaml.
    """
    if config.tau is not None:
        return list(config.tau)
    return [layer / depth for layer in config.early_exit_after_layers]


class EarlyExit(nn.Module):
    """g_theta_k -- training only, removed before deployment.

    pointwise conv (match channels) -> h_theta (depthwise) -> ReLU -> GAP -> linear

    Only h_theta is ever copied. The pointwise conv and the classifier are the
    early exit's own and stay behind when it is discarded.
    """

    def __init__(self, in_channels: int, match_channels: int, kernel_size: int, num_classes: int):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, match_channels, kernel_size=1)
        self.h_theta = nn.Conv2d(
            match_channels, match_channels, kernel_size,
            padding=kernel_size // 2, groups=match_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(match_channels, num_classes)

    def forward(self, tap: torch.Tensor) -> torch.Tensor:
        x = self.pointwise(tap)
        x = F.relu(self.h_theta(x), inplace=True)
        return self.classifier(torch.flatten(self.pool(x), 1))


class OutputBlock(nn.Module):
    """f_out_k -- deployed. Reads the base network's final features f_D.

    h_phi (depthwise) -> ReLU -> GAP -> linear

    h_phi is overwritten from the matching h_theta before every training batch,
    so its own learned values never persist. What persists is the adaptation of
    the layers feeding it.
    """

    def __init__(self, channels: int, kernel_size: int, num_classes: int):
        super().__init__()
        self.h_phi = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.h_phi(features), inplace=True)
        return self.classifier(torch.flatten(self.pool(x), 1))


class QUTEModel(nn.Module):
    """Shared CNN + K early exits (training) + K output blocks (deployed)."""

    def __init__(self, base: SharedCNN, config: Optional[QUTEConfig] = None):
        super().__init__()
        self.base = base
        self.config = config or QUTEConfig()
        cfg = self.config

        if base.config.dropout_rate != 0.0:
            raise ValueError(
                f"QUTE's base network must have dropout_rate 0, got "
                f"{base.config.dropout_rate}. Dropout belongs to MC Dropout only."
            )

        channels = base.config.channels
        final_channels = channels[-1]

        # Spatial size is irrelevant to the copy, but channel count is not.
        self.early_exits = nn.ModuleList([
            EarlyExit(
                in_channels=channels[layer - 1],
                match_channels=final_channels,
                kernel_size=cfg.kernel_size,
                num_classes=base.config.num_classes,
            )
            for layer in cfg.early_exit_after_layers
        ])
        self.output_blocks = nn.ModuleList([
            OutputBlock(
                channels=final_channels,
                kernel_size=cfg.kernel_size,
                num_classes=base.config.num_classes,
            )
            for _ in range(cfg.K)
        ])

        self._check_transfer_shapes()

    # ---- base network, run block by block so f_D is reachable ----

    def _base_features(self, x: torch.Tensor):
        """Return (taps_by_layer, f_D).

        SharedCNN.forward exposes layer1 and layer2 but not the final block's
        output, and f_D is what the output blocks read. Running the blocks here
        avoids changing the shared module that the baselines depend on.
        """
        taps = {}
        x = self.base.block1(x)
        taps[1] = x
        x = self.base.block2(x)
        taps[2] = x
        x = self.base.block3(x)
        taps[3] = x
        f_d = self.base.block4(x)
        return taps, f_d

    def _final_exit_logits(self, f_d: torch.Tensor) -> torch.Tensor:
        """The base network's own classifier, f_out. Trained, never deployed."""
        return self.base.classifier(self.base.flatten(self.base.pool(f_d)))

    # ---- forward passes ----

    def forward_train(self, x: torch.Tensor) -> Dict[str, object]:
        """Every exit, for the three-term loss.

        Returns {"ee": [K logits], "ev": [K logits], "final": logits}.
        """
        taps, f_d = self._base_features(x)
        ee_logits = [
            exit_block(taps[layer])
            for exit_block, layer in zip(self.early_exits, self.config.early_exit_after_layers)
        ]
        ev_logits = [block(f_d) for block in self.output_blocks]
        return {"ee": ee_logits, "ev": ev_logits, "final": self._final_exit_logits(f_d)}

    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Deployed path. Returns [N, K, num_classes].

        Early exits and the original classifier are not touched. Average the
        softmax of these with models.ensemble_proba -- never the logits, and
        never the predicted class IDs.
        """
        _taps, f_d = self._base_features(x)
        return torch.stack([block(f_d) for block in self.output_blocks], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of the output blocks' probabilities, [N, num_classes].

        This returns PROBABILITIES, not logits, because averaging K softmaxes is
        not the same as the softmax of averaged logits. Do not wrap this in
        another softmax. For the standard prediction path use
        qute.predict.predict_proba_qute instead.
        """
        return torch.softmax(self.member_logits(x), dim=2).mean(dim=1)

    # ---- weight transfer ----

    def _check_transfer_shapes(self) -> None:
        for k, (exit_block, out_block) in enumerate(zip(self.early_exits, self.output_blocks)):
            theta, phi = exit_block.h_theta, out_block.h_phi
            if theta.weight.shape != phi.weight.shape:
                raise RuntimeError(
                    f"exit {k}: h_theta weight {tuple(theta.weight.shape)} does not match "
                    f"h_phi {tuple(phi.weight.shape)}. The pointwise conv must map the tap "
                    f"to {phi.weight.shape[0]} channels."
                )
            if (theta.bias is None) != (phi.bias is None):
                raise RuntimeError(f"exit {k}: h_theta and h_phi disagree about bias")

    @torch.no_grad()
    def transfer_weights(self) -> None:
        """phi_k <- theta_k. Call before EVERY training batch, including during
        the final frozen epochs.

        copy_ writes values into the existing tensor. Assigning
        `phi.weight = theta.weight` would alias the two parameters into one,
        which trains but is not the method -- assert_no_aliasing checks for it.
        """
        for exit_block, out_block in zip(self.early_exits, self.output_blocks):
            out_block.h_phi.weight.copy_(exit_block.h_theta.weight)
            if out_block.h_phi.bias is not None:
                out_block.h_phi.bias.copy_(exit_block.h_theta.bias)

    def assert_no_aliasing(self) -> None:
        """Fail if h_phi and h_theta share storage rather than values."""
        for k, (exit_block, out_block) in enumerate(zip(self.early_exits, self.output_blocks)):
            theta, phi = exit_block.h_theta.weight, out_block.h_phi.weight
            if theta.data_ptr() == phi.data_ptr():
                raise RuntimeError(
                    f"exit {k}: h_theta and h_phi share storage. Use copy_, not assignment."
                )

    def transfer_matches(self, atol: float = 0.0) -> bool:
        """True when every h_phi currently equals its h_theta."""
        for exit_block, out_block in zip(self.early_exits, self.output_blocks):
            if not torch.allclose(out_block.h_phi.weight, exit_block.h_theta.weight, atol=atol):
                return False
            if out_block.h_phi.bias is not None:
                if not torch.allclose(out_block.h_phi.bias, exit_block.h_theta.bias, atol=atol):
                    return False
        return True

    # ---- final training stage ----

    def freeze_base(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the base network.

        [paper] The base is frozen for the last 10% of epochs while weight
        transfer continues, so the output blocks train in isolation. With 20
        epochs that is epochs 19 and 20.
        """
        for parameter in self.base.parameters():
            parameter.requires_grad_(not frozen)

    def freeze_epoch_start(self, epochs: int) -> int:
        """First epoch (1-based) of the frozen stage."""
        frozen = max(1, round(epochs * self.config.freeze_base_last_fraction))
        return epochs - frozen + 1

    # ---- reporting ----

    def deployed_parameters(self) -> int:
        """Parameters that survive to inference.

        Base network, minus its original classifier, plus the output blocks.
        Early exits are excluded -- they are training-only.
        """
        base_total = sum(p.numel() for p in self.base.parameters())
        original_head = sum(p.numel() for p in self.base.classifier.parameters())
        blocks = sum(p.numel() for p in self.output_blocks.parameters())
        return base_total - original_head + blocks

    def training_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {
            "qute_version": QUTE_VERSION,
            "K": self.config.K,
            "early_exit_after_layers": list(self.config.early_exit_after_layers),
            "ev_weights": ev_weights(self.config),
            "tau_weights": tau_weights(self.config),
            "training_parameters": self.training_parameters(),
            "deployed_parameters": self.deployed_parameters(),
            "deployed_approx_fp32_kib": round(self.deployed_parameters() * 4 / 1024, 1),
            "transfer_weight_shape": tuple(self.output_blocks[0].h_phi.weight.shape),
        }


def qute_loss(
    outputs: Dict[str, object],
    labels: torch.Tensor,
    config: QUTEConfig,
    depth: int = 4,
) -> Dict[str, torch.Tensor]:
    """L = sum_k tau_k * CE(EE_k) + sum_k w_EV_k * CE(EV_k) + CE(final exit).

    [paper] Equation 2. The three parts are returned alongside the total so the
    trainer can log them -- a weight-transfer bug usually shows up as the EV
    terms behaving differently from the EE terms.
    """
    taus = tau_weights(config, depth=depth)
    ws = ev_weights(config)

    ee_loss = sum(
        tau * F.cross_entropy(logits, labels)
        for tau, logits in zip(taus, outputs["ee"])
    )
    ev_loss = sum(
        w * F.cross_entropy(logits, labels)
        for w, logits in zip(ws, outputs["ev"])
    )
    final_loss = F.cross_entropy(outputs["final"], labels)

    return {
        "total": ee_loss + ev_loss + final_loss,
        "ee": ee_loss,
        "ev": ev_loss,
        "final": final_loss,
    }


def build_qute(
    cnn_config: Optional[CNNConfig] = None,
    qute_config: Optional[QUTEConfig] = None,
    seed: Optional[int] = None,
) -> QUTEModel:
    """Build a QUTE model with reproducible initialization."""
    if seed is not None:
        torch.manual_seed(seed)
    base = build_model(cnn_config)
    return QUTEModel(base, qute_config)


def qute_config_from_protocol(protocol: dict) -> QUTEConfig:
    """Read the `qute:` block of configs/base.yaml."""
    block = protocol["qute"]
    return QUTEConfig(
        K=int(block["K"]),
        early_exit_after_layers=list(block["early_exit_after_layers"]),
        w_ev_0=float(block["w_ev_0"]),
        w_ev_delta=float(block["w_ev_delta"]),
        tau=list(block["tau"]) if block.get("tau") is not None else None,
        freeze_base_last_fraction=float(block["freeze_base_last_fraction"]),
        transfer_during_freeze=bool(block["transfer_during_freeze"]),
        drop_original_output_block=bool(block["drop_original_output_block"]),
    )
