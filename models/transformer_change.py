"""Phase 20: Transformer-based Siamese change detection architecture — a research comparison
against the CNN-based Siamese U-Net (models/siamese_unet.py), which remains this project's
primary/production model (see README.md "Results"). This model is never used to replace it; it
exists solely to produce a real, measured answer to "does a self-attention encoder outperform the
CNN encoder on this task and this dataset size", reported honestly either way.

    before -> patch embed -> [shared transformer encoder blocks] -> tokens_before ---\
                                                                                        compare -> CNN decoder -> logits
    after  -> patch embed -> [shared transformer encoder blocks] -> tokens_after  ---/

The encoder is a genuine multi-head self-attention Transformer (`nn.TransformerEncoder`) operating
over a single-scale patch grid (patch_size=16 on a 256x256 input -> 16x16=256 tokens) with a
learnable positional embedding — global receptive field via self-attention across all tokens from
the first layer, which a CNN's local convolutions lack. Weight sharing between the before/after
branches is by construction: one `TransformerEncoder` module instance is called on each image
(same pattern as `models/siamese_encoder.py::SiameseEncoder`).

Because attention here is single-scale, the decoder cannot reuse the CNN U-Net's multi-scale skip
connections; instead it progressively upsamples the compared token grid back to full resolution
with transposed-convolution blocks. Feature comparison reuses `models/siamese_unet.py`'s
`compare_features`/`comparison_channels` (Rule 6 — don't duplicate working code).
"""
import torch
import torch.nn as nn

from models.siamese_unet import COMPARISON_MODES, comparison_channels, compare_features


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        _, _, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return tokens, (h, w)


class TransformerEncoder(nn.Module):
    """Shared-weight transformer branch: patch embed + learnable positional embedding + a stack
    of standard pre-norm multi-head self-attention encoder layers."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, num_layers: int,
                 num_heads: int, mlp_ratio: float, dropout: float, num_patches: int):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        tokens, (h, w) = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        return tokens, (h, w)


class DecoderBlock(nn.Module):
    """Upsample-by-2 + conv block, used to progressively decode the compared token grid back to
    full input resolution — there are no multi-scale skip connections here, unlike the CNN U-Net,
    since the transformer encoder above operates at a single patch-grid scale."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class TransformerChangeDetector(nn.Module):
    """Phase 20 research-comparison architecture. `image_size` must be divisible by `patch_size`
    (default 256/16 -> a 16x16=256-token grid, matching this project's standard 256x256 input)."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        image_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        comparison_mode: str = "diff_concat",
    ):
        super().__init__()
        if comparison_mode not in COMPARISON_MODES:
            raise ValueError(f"Unknown comparison mode: {comparison_mode!r} (expected one of {COMPARISON_MODES})")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

        self.comparison_mode = comparison_mode
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size * self.grid_size

        self.encoder = TransformerEncoder(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size,
            num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            num_patches=num_patches,
        )

        cc = comparison_channels(embed_dim, comparison_mode)

        # patch_size=16 -> 4 upsampling stages of x2 (16 -> 32 -> 64 -> 128 -> 256), matching
        # log2(patch_size); only valid for power-of-2 patch sizes, which every config here uses.
        num_up_stages = patch_size.bit_length() - 1
        channels = [cc] + [max(embed_dim // (2 ** i), 16) for i in range(1, num_up_stages + 1)]
        self.decoder = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1]) for i in range(num_up_stages)
        ])
        self.out_conv = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        tokens_before, (h, w) = self.encoder(before)
        tokens_after, _ = self.encoder(after)

        grid_before = tokens_before.transpose(1, 2).reshape(tokens_before.size(0), -1, h, w)
        grid_after = tokens_after.transpose(1, 2).reshape(tokens_after.size(0), -1, h, w)
        x = compare_features(grid_before, grid_after, self.comparison_mode)

        for block in self.decoder:
            x = block(x)
        return self.out_conv(x)  # raw logits, same convention as every other model in this project
