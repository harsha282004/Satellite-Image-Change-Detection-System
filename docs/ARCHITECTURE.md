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

## Siamese U-Net (Phase 5)

**Status: Planned, not yet implemented.** Will share encoder weights between the before/after
branches and compare their extracted features explicitly (configurable: absolute difference,
concatenation, or both), rather than relying on a single fused 6-channel input. Documented here in
advance per `PROJECT_CONTEXT.md`; this section will be rewritten to describe the real
implementation once Phase 5 is built, not before.

## Evaluation metrics (`src/evaluation/metrics.py`)

**Status: Implemented (Phase 4).** `MetricAccumulator` accumulates confusion-matrix counts
(TP/FP/FN/TN) across an entire split before computing IoU/Dice/Precision/Recall/F1/Accuracy —
correct under class imbalance, since per-batch metric averaging would let near-empty-mask batches
skew the result. See `docs/EVALUATION.md` (written in Phase 7) for the full rigorous evaluation
methodology and results; real Phase 4 baseline test-set numbers are recorded in
`DEVELOPMENT_LOG.md`.
