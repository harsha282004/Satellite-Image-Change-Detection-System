"""Shared-weight encoder for the Siamese U-Net (Phase 5).

Reuses DoubleConv/Down from models/unet.py rather than duplicating them (DEVELOPMENT_RULES.md
Rule 6). Weight sharing is achieved simply by construction: a single SiameseEncoder instance is
called on both the before and after images in models/siamese_unet.py, so both passes use the
exact same parameters — not just the same architecture.
"""
import torch
import torch.nn as nn

from models.unet import DoubleConv, Down


class SiameseEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
