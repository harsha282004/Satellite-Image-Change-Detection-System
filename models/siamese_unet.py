"""Siamese U-Net (Phase 5) — the project's primary change-detection architecture.

    before -> shared encoder -> features_before ---\
                                                      compare -> U-Net decoder -> change-mask logits
    after  -> shared encoder -> features_after  ---/

Unlike the Phase 4 baseline (which concatenates before/after into one 6-channel input before any
convolution), this model runs the *same* encoder (identical weights) on the before and after
images separately, then explicitly compares the resulting feature maps at every scale — including
the bottleneck — before decoding. The comparison is configurable, per PROJECT_CONTEXT.md:

    "diff"        - absolute difference of the two feature maps
    "concat"      - channel-wise concatenation of the two feature maps
    "diff_concat" - concatenation of both feature maps AND their absolute difference

The decoder reuses the `Up` block from models/unet.py (Rule 6 — don't duplicate working code);
its skip-connection channel counts are derived from the comparison mode via `comparison_channels`.
"""
import torch
import torch.nn as nn

from models.attention import AttentionUp
from models.siamese_encoder import SiameseEncoder
from models.unet import Up

COMPARISON_MODES = ("diff", "concat", "diff_concat")


def compare_features(before_feat: torch.Tensor, after_feat: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "diff":
        return torch.abs(before_feat - after_feat)
    if mode == "concat":
        return torch.cat([before_feat, after_feat], dim=1)
    if mode == "diff_concat":
        diff = torch.abs(before_feat - after_feat)
        return torch.cat([before_feat, after_feat, diff], dim=1)
    raise ValueError(f"Unknown comparison mode: {mode!r} (expected one of {COMPARISON_MODES})")


def comparison_channels(feature_channels: int, mode: str) -> int:
    if mode == "diff":
        return feature_channels
    if mode == "concat":
        return feature_channels * 2
    if mode == "diff_concat":
        return feature_channels * 3
    raise ValueError(f"Unknown comparison mode: {mode!r} (expected one of {COMPARISON_MODES})")


class SiameseUNet(nn.Module):
    """
    `use_attention=True` (Phase 8 research experiment) replaces each decoder `Up` block with
    `AttentionUp` (models/attention.py), gating each skip connection by the coarser decoder
    context before concatenation, per Oktay et al.'s Attention U-Net. Default False reproduces
    the exact Phase 5 architecture unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        comparison_mode: str = "diff_concat",
        use_attention: bool = False,
    ):
        super().__init__()
        if comparison_mode not in COMPARISON_MODES:
            raise ValueError(f"Unknown comparison mode: {comparison_mode!r} (expected one of {COMPARISON_MODES})")
        self.comparison_mode = comparison_mode
        self.use_attention = use_attention

        c = base_channels
        self.encoder = SiameseEncoder(in_channels=in_channels, base_channels=c)

        cc1 = comparison_channels(c, comparison_mode)
        cc2 = comparison_channels(c * 2, comparison_mode)
        cc3 = comparison_channels(c * 4, comparison_mode)
        cc4 = comparison_channels(c * 8, comparison_mode)
        cc5 = comparison_channels(c * 16, comparison_mode)

        up_cls = AttentionUp if use_attention else Up
        self.up1 = up_cls(cc5, cc4, c * 8)
        self.up2 = up_cls(c * 8, cc3, c * 4)
        self.up3 = up_cls(c * 4, cc2, c * 2)
        self.up4 = up_cls(c * 2, cc1, c)

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        b1, b2, b3, b4, b5 = self.encoder(before)
        a1, a2, a3, a4, a5 = self.encoder(after)

        f1 = compare_features(b1, a1, self.comparison_mode)
        f2 = compare_features(b2, a2, self.comparison_mode)
        f3 = compare_features(b3, a3, self.comparison_mode)
        f4 = compare_features(b4, a4, self.comparison_mode)
        f5 = compare_features(b5, a5, self.comparison_mode)

        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        return self.out_conv(x)  # raw logits, same convention as BaselineChangeUNet
