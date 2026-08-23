"""Baseline U-Net for change detection (Phase 4).

Operates on a single fused input: the before and after images concatenated channel-wise
(3 + 3 = 6 input channels), producing a single-channel binary change-mask logit map. This is
the reference baseline; the encoder-sharing Siamese architecture (Phase 5) is the primary
project contribution.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscale, concatenate skip connection, then double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle any off-by-one spatial mismatch from odd input sizes.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net, 4 downsampling stages, configurable width via `base_channels`.

    Default `base_channels=32` (rather than the original paper's 64) to keep the model
    tractable on the 6 GB GPU identified in Phase 1 at 256x256 input resolution.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)  # raw logits — pair with BCEWithLogitsLoss / sigmoid at inference


class BaselineChangeUNet(nn.Module):
    """Wraps UNet to take (before, after) image pairs and concatenate them into a fused
    6-channel input, matching the Phase 4 "single fused input" baseline design."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.unet = UNet(in_channels=6, out_channels=1, base_channels=base_channels)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([before, after], dim=1)
        return self.unet(fused)
