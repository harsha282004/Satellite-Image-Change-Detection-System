# EVALUATION.md

Rigorous evaluation of the models implemented so far (baseline U-Net, Phase 4; Siamese U-Net,
Phase 5) on the held-out LEVIR-CD test split. Every number in this document is real, measured
output from `src/evaluation/evaluate.py` — see `outputs/metrics/*_test_metrics.json` for the raw
machine-readable reports and `DEVELOPMENT_LOG.md` (Phase 4-7 entries) for the full commands and
transcripts that produced them. Nothing here is estimated or fabricated (`DEVELOPMENT_RULES.md`
Rule 3).

## Methodology

### Test set
The LEVIR-CD **official test split**, 128 before/after/mask triplets, acquired and verified in
Phase 2 (`docs/DATASET.md`). This split was never used for training or for checkpoint selection —
checkpoints are selected by best **validation** IoU during training
(`src/training/trainer.py::Trainer.fit`), then evaluated on test exactly once per reported run.
There is no train/test geographic overlap: the official LEVIR-CD authors' split is used as-is,
with no re-shuffling (see `docs/DATASET.md` "Leakage prevention / split methodology").

### Metrics
Six metrics are always reported together: **IoU, Dice, Precision, Recall, F1, Accuracy**
(`src/evaluation/metrics.py::MetricAccumulator`). Per `PROJECT_CONTEXT.md`, **accuracy alone is
not a sufficient metric on this task**: Phase 2 measured only ~4.2-5.1% of pixels as "changed"
across the LEVIR-CD splits, so a model that always predicts "no change" would already score
>94% accuracy while being useless. IoU/Dice/F1 are the metrics that actually reflect change-mask
quality under this imbalance.

Metrics are computed by accumulating confusion-matrix counts (TP/FP/FN/TN) across the **entire**
test split before computing ratios — not by averaging per-batch metrics, which would let
batches with few/no changed pixels skew the result under this class imbalance.

### Checkpoint selection and reproducibility caveat
Both models were trained for a fixed 30 epochs (`configs/baseline.yaml`, `configs/siamese.yaml`),
Adam (lr=1e-4), BCE+Dice loss, batch size 8, image size 256, seed 42, on the same NVIDIA RTX 4050
Laptop GPU (Phase 1). **Phase 6 found that training on this GPU is not bit-exact reproducible even
with a fixed seed** (non-deterministic cuDNN convolution algorithms) — re-running the identical
baseline config produced a different "best" epoch and a real precision/recall shift at that
checkpoint, despite near-identical aggregate IoU/Dice. The numbers below are from one training run
each (the baseline's *restored* run, after a Phase 6 incident — see `DEVELOPMENT_LOG.md` Phase 6).
They should be read as representative, not as exact, irreproducible-to-the-decimal values. A
multi-seed comparison is deferred to Phase 8 if judged worthwhile.

## Quantitative results (real, measured)

| Metric | Baseline U-Net (Phase 4) | Siamese U-Net (Phase 5) |
|---|---|---|
| IoU | 0.6234 | **0.6442** |
| Dice | 0.7680 | **0.7836** |
| Precision | 0.7333 | **0.7982** |
| Recall | **0.8062** | 0.7695 |
| F1 | 0.7680 | **0.7836** |
| Accuracy | 0.9752 | **0.9784** |

Raw confusion-matrix counts (test set, 128 × 256 × 256 = 8,388,608 pixels total):

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| Baseline U-Net | 344,426 | 125,241 | 82,806 | 7,836,135 |
| Siamese U-Net | 328,735 | 83,099 | 98,497 | 7,878,277 |

### Interpretation
The Siamese U-Net outperforms the baseline on IoU, Dice, Precision, F1, and Accuracy, but the
baseline has a real, meaningfully higher recall (+0.0367). This is a genuine precision/recall
tradeoff, not measurement noise: the Siamese model produces roughly 42,000 fewer false positives
than the baseline on the test set (83,099 vs. 125,241), at the cost of about 15,700 more false
negatives (98,497 vs. 82,806). In other words, the Siamese model is more conservative — it predicts
"change" less liberally, catching fewer true changes but with far less false-positive noise. Net
effect on the metrics that matter most given the class imbalance (IoU, Dice, F1): Siamese wins.

This is consistent with the architectural difference: the Siamese model can explicitly compare
same-location before/after features at every decoder scale (via `diff_concat`), while the baseline
only sees the raw channel-wise concatenation of the two images and must implicitly learn to
extract and compare features on its own — plausibly making it more prone to spurious
"change-looking" predictions.

## Qualitative results

Full 6-sample prediction grids (Before / After / Ground Truth / Prediction / Overlay /
Diff-with-FP-yellow-FN-blue) for both models:
- `outputs/visualizations/baseline_unet_test_predictions.png`
- `outputs/visualizations/siamese_unet_diff_concat_test_predictions.png`

Both grids use the same 5 evenly-spaced test-set indices, enabling direct visual comparison.
Observations from manual inspection (Phase 4/5/7):
- Both models correctly predict near-empty masks on genuinely no-change scenes (forest/rural
  scenes with no new construction) — no gross over-triggering.
- Both models correctly detect large, obvious changes (new subdivisions, parking lots).
- The baseline visibly produces more scattered false-positive (yellow) speckling around
  true-positive regions than the Siamese model, consistent with its lower measured precision.
- Neither model perfectly recovers small/thin building outlines — both show some false negatives
  (blue) at object edges, consistent with imperfect but reasonable boundary precision for a
  30-epoch, untuned baseline-scale training run.

## Training curves

- `outputs/visualizations/baseline_unet_training_curves.png`
- `outputs/visualizations/siamese_unet_diff_concat_training_curves.png`

Both show smoothly decreasing train/val loss and increasing train/val IoU/Dice across all 30
epochs, with no divergence or instability. Validation curves (IoU, Dice, and especially
Precision/Recall/F1) are visibly noisier epoch-to-epoch than training curves — expected given the
validation split is only 64 images, so per-epoch validation metrics have higher variance than the
445-image training metrics. Neither model's validation IoU had clearly plateaued by epoch 30 (see
`DEVELOPMENT_LOG.md` Phase 4/5 "Known issues") — both are likely undertrained relative to full
convergence, a deliberate scope choice for these milestones rather than a tuned final result.

## Limitations of this evaluation (see also `docs/LIMITATIONS.md`, Phase 12)

- **Single run per model, single seed.** No confidence intervals or variance estimates. Given
  Phase 6's finding that this GPU's training is not bit-exact reproducible, some of the
  baseline-vs-Siamese gap could narrow or shift under a different seed — the direction of the
  IoU/Dice/F1 result (Siamese ahead) is plausible and architecturally motivated, but has not been
  confirmed across multiple seeds.
- **Fixed, untuned hyperparameters** for both models (same config values, no learning-rate
  schedule, no hyperparameter search). This evaluation compares two architectures under one
  reasonable-but-unoptimized training recipe, not each architecture's best achievable performance.
- **Benchmark-only.** This evaluation uses only the LEVIR-CD benchmark test split — a curated,
  pre-registered, single-sensor (Google Earth, 0.5m/pixel) dataset. It says nothing yet about
  performance on real-world, differently-sourced imagery (e.g. Sentinel-2) — that is Phase 11's
  explicit, separately-caveated real-world demonstration.
- **Threshold fixed at 0.5** (sigmoid probability → binary prediction). No precision/recall
  operating-point analysis (e.g. an ROC or PR curve sweeping the threshold) has been performed —
  the recall/precision tradeoff observed above is only characterized at this one operating point.
- **Only the `diff_concat` Siamese comparison mode was trained.** `diff` and `concat` are
  implemented and unit-tested (Phase 5) but not evaluated on real data — deferred to Phase 8 as an
  ablation study, not required for this milestone.

## Status summary

| Item | Status |
|---|---|
| IoU / Dice / Precision / Recall / F1 / Accuracy on held-out test set | **Implemented, measured** |
| Qualitative before/after/GT/prediction/overlay/diff visualizations | **Implemented, measured** (5 examples each model) |
| Training curves (loss, IoU, Dice vs. epoch) | **Implemented, measured** |
| Baseline vs. Siamese comparison | **Implemented, measured** (single run/seed each) |
| Multi-seed / confidence-interval comparison | Planned — Phase 8, if warranted |
| Attention / Transformer variant comparison | Planned — Phase 8, if warranted |
| Real-world (Sentinel-2) evaluation | Planned — Phase 11, explicitly separate from this benchmark evaluation |
| Precision/recall operating-point (threshold) analysis | Not yet implemented |
