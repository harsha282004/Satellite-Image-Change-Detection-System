"""Attention gates for the decoder skip connections (Phase 8 research experiment).

Standard additive attention gate (Oktay et al., "Attention U-Net", 2018): before concatenating a
skip connection into the decoder, the skip is re-weighted per-pixel by a gating signal computed
from the coarser decoder feature map at that stage. Intuitively, the gate learns to suppress
skip-connection regions the decoder's current context considers irrelevant, and pass through
regions it considers relevant — a targeted, well-established technique rather than a novel or
exotic mechanism (DEVELOPMENT_RULES.md Rule 5: prefer simple, justified improvements).

`AttentionUp` is a drop-in alternative to `models.unet.Up`, reusing `DoubleConv` from that module
(Rule 6) so the only new logic here is the gate itself.
"""
import torch
import torch.nn as nn

from models.unet import DoubleConv


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.relu(self.w_gate(gate) + self.w_skip(skip))
        attention = self.psi(attention)  # (N, 1, H, W) in [0, 1]
        return skip * attention


class AttentionUp(nn.Module):
    """Upscale, gate the skip connection with an AttentionGate, concatenate, then double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention_gate = AttentionGate(gate_channels=in_channels // 2, skip_channels=skip_channels)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                       diff_h // 2, diff_h - diff_h // 2])
        gated_skip = self.attention_gate(gate=x, skip=skip)
        x = torch.cat([gated_skip, x], dim=1)
        return self.conv(x)
