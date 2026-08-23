# ARCHITECTURE.md

Describes the architecture **as actually implemented**. Per `DEVELOPMENT_RULES.md`, this document
does not describe planned-but-unbuilt functionality as if it exists.

## Pipeline (implemented so far: Phases 0-4)

```
Before image ─┐
              ├─> Preprocessing (resize, normalize) ─> Fused 6-channel input ─> U-Net ─> Binary change-mask logits
After image  ─┘
```

`src/data/preprocessing.py` resizes before/after images to a fixed square (`image_size`, 256 by
default) with linear interpolation, binarizes the ground-truth mask (threshold 127, see
`docs/DATASET.md`), and normalizes images to `[0,1]`. `src/data/augmentation.py`'s
`PairedAugmentor` applies identical spatial transforms (flip/rotate/scale-crop) to the before
image, after image, and mask, plus independent brightness jitter to the images only.

## Baseline model: single-fused-input U-Net (`models/unet.py`, `BaselineChangeUNet`)

**Status: Implemented (Phase 4).**

```
before (3xHxW) ─┐
                ├─ concat (channel dim) ─> fused (6xHxW) ─> UNet ─> logits (1xHxW)
after  (3xHxW) ─┘
```

`BaselineChangeUNet` concatenates the before and after images along the channel dimension into a
single 6-channel input and feeds it through a standard U-Net (`UNet` class). This is a
deliberately simple reference baseline — it does not share an encoder between the two time steps,
so it cannot explicitly compare before/after features; it only sees their channel-wise
concatenation. This limitation is exactly what the Siamese architecture (Phase 5, not yet
implemented) is designed to address.

### U-Net structure
- 4 downsampling stages (`DoubleConv` + `MaxPool2d`), symmetric 4-stage decoder
  (`ConvTranspose2d` upsampling + skip-connection concat + `DoubleConv`)
- `DoubleConv` = (Conv3x3 → BatchNorm → ReLU) × 2
- Configurable width via `base_channels` (default 32, not the original U-Net paper's 64 — chosen
  for tractability on the 6 GB GPU identified in Phase 1 at 256×256 input resolution)
- Output: single-channel raw logits (paired with `BCEWithLogitsLoss` / sigmoid at inference, never
  a raw probability output directly from the model)
- **Measured parameter count: 7,763,905** (`base_channels=32`, printed by
  `src/training/train.py` at training start)

### Losses (`models/losses.py`)
- `BCEWithLogitsLoss` (`"bce"`)
- `DiceLoss` (`"dice"`) — soft Dice computed from sigmoid probabilities
- `BCEDiceLoss` (`"bce_dice"`, **used for the baseline**) — equal-weighted sum of the two, since
  Dice alone can be unstable early in training on a heavily class-imbalanced mask (see
  `docs/DATASET.md`, ~4-5% changed pixels)

### Training configuration actually used (`configs/baseline.yaml`)
```yaml
image_size: 256
batch_size: 8
epochs: 30
optimizer: adam
learning_rate: 0.0001
loss: bce_dice
seed: 42
```

## Siamese U-Net (`models/siamese_encoder.py`, `models/siamese_unet.py`)

**Status: Implemented (Phase 5). This is the project's primary architecture.**

```
before ─> shared encoder ─> (b1,b2,b3,b4,b5) ─┐
                                               ├─ compare at each scale ─> U-Net decoder ─> logits (1xHxW)
after  ─> shared encoder ─> (a1,a2,a3,a4,a5) ─┘
```

Unlike the baseline, the before and after images are never concatenated before convolution.
Instead, `SiameseEncoder` (a single `nn.Module` instance, structurally identical to the baseline
U-Net's encoder — 4 downsampling `DoubleConv`+`MaxPool2d` stages plus the initial `DoubleConv`) is
called twice, once on `before` and once on `after`, producing feature maps at 5 scales for each.
Because it's the *same* module instance called twice, both passes literally share every weight —
this is not merely the same architecture applied independently, verified in
`tests/test_siamese_unet.py::test_siamese_unet_shares_encoder_weights_with_baseline_style_parameter_count`.

The corresponding before/after feature maps are then explicitly compared at every scale
(`compare_features`, `models/siamese_unet.py`), via one of three configurable modes:

| mode | operation | comparison channels (given `c` feature channels) |
|------|-----------|----|
| `diff` | `abs(before_feat - after_feat)` | `c` |
| `concat` | `cat([before_feat, after_feat])` | `2c` |
| `diff_concat` | `cat([before_feat, after_feat, abs(before_feat - after_feat)])` | `3c` |

The 5 compared feature maps (bottleneck + 4 skip levels) then feed a standard U-Net decoder —
reusing the exact `Up` block class from `models/unet.py` (Rule 6: no duplicated decoder code),
with skip-connection channel counts derived from the comparison mode via `comparison_channels()`.

### Verified properties (`tests/test_siamese_unet.py`, 13 tests)
- Forward pass produces correct `(N, 1, H, W)` logit shape for all 3 comparison modes
- Backward pass + optimizer step actually changes model parameters, for all 3 comparison modes
- `diff` mode is symmetric in before/after (as expected for `abs(a-b)`); `concat` mode is
  **not** symmetric — swapping before/after changes the output, confirming the model can actually
  distinguish temporal direction when the comparison mode preserves it
- Encoder parameters appear exactly once in `named_parameters()` (genuinely shared, not duplicated
  per branch)

### Configuration actually used for the primary trained run (`configs/siamese.yaml`)
```yaml
model:
  type: siamese_unet
  base_channels: 32
  comparison_mode: diff_concat  # richest signal; all 3 modes fit comfortably in the 6GB GPU's VRAM
image_size: 256
batch_size: 8
epochs: 30
optimizer: adam
learning_rate: 0.0001
loss: bce_dice
seed: 42
```

**Measured parameter counts** (`base_channels=32`, printed at training start / from
`scripts/` benchmarking, real numbers, not estimated): `diff`=7,763,041; `concat`=10,709,345;
`diff_concat`=14,704,225. **Measured peak training VRAM** (batch_size=8, image_size=256, RTX 4050):
`diff`=2.21GB, `concat`=2.38GB, `diff_concat`=2.79GB — all three comfortably within budget.
Real train/val/test metrics for the trained `diff_concat` configuration are in
`DEVELOPMENT_LOG.md` (Phase 5 entry), not fabricated or estimated here.

## Evaluation metrics (`src/evaluation/metrics.py`)

**Status: Implemented (Phase 4).** `MetricAccumulator` accumulates confusion-matrix counts
(TP/FP/FN/TN) across an entire split before computing IoU/Dice/Precision/Recall/F1/Accuracy —
correct under class imbalance, since per-batch metric averaging would let near-empty-mask batches
skew the result. See `docs/EVALUATION.md` (written in Phase 7) for the full rigorous evaluation
methodology and results; real Phase 4 baseline test-set numbers are recorded in
`DEVELOPMENT_LOG.md`.
