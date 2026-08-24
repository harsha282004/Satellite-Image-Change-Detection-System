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
- **Threshold fixed at 0.5** for the numbers in this section (measured before Phase 15). Phase 15
  below performs the threshold sweep/operating-point analysis this originally called "not yet
  implemented" — see that section for the real result (spoiler: the model is quite insensitive to
  threshold choice in this range, so this limitation turned out to matter little in practice).
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
| Precision/recall operating-point (threshold) analysis | **Implemented, measured** — see Phase 15 below |

---

## Phase 15 — Confidence, Probability, Threshold Optimization, and Robustness

All results below use **the current best model** (`siamese_unet_diff_concat_attention_e100`,
Phase 13 Experiment C) — not the Phase 4-8 models this document's earlier sections describe.

### 15.1 — Prediction probability maps

The model outputs raw logits; `sigmoid(logits)` gives a per-pixel value in [0,1] —
`src/inference/predict.py::predict_probability`. This is displayed and referred to everywhere in
this project as **"prediction probability"**, never **"confidence"** — no calibration study (e.g.
checking whether pixels the model outputs 0.8 for are actually correct ~80% of the time, via
reliability diagrams / Expected Calibration Error) has been run. Until one is, "probability" means
only "the sigmoid-activated model output," not a verified statistical confidence level.

Representative examples (`outputs/visualizations/probability_maps/`, generated by
`scripts/generate_probability_maps.py`): a dense-change scene, a dramatic large-scale change
scene, and a no-change scene. Visual observation: the model's probability maps are strongly
bimodal/saturated (values cluster near 0.0 or 1.0, with a thin uncertain band at building edges)
rather than being smoothly graded across [0,1] — consistent with a well-converged binary
classifier, though this is a qualitative observation from 3 examples, not a systematic study.

### 15.2 — Threshold optimization (real, measured)

Swept 9 thresholds (0.30-0.70, step 0.05) on the **validation set only** (`scripts/
threshold_optimization.py`, `src/evaluation/threshold_analysis.py`), selected the threshold with
the highest validation IoU, then evaluated that one threshold once on the **test set** — the test
set was never used to choose the threshold (Rule 3). Full data:
`outputs/metrics/threshold_analysis.csv`; curves: `outputs/visualizations/threshold_analysis.png`.

| Threshold | Val IoU | Val Dice | Val Precision | Val Recall | Val F1 |
|---|---|---|---|---|---|
| 0.30 | 0.7175 | 0.8355 | 0.8162 | 0.8557 | 0.8355 |
| 0.35 | 0.7189 | 0.8365 | 0.8246 | 0.8487 | 0.8365 |
| **0.40 (selected)** | **0.7196** | **0.8369** | 0.8322 | 0.8416 | **0.8369** |
| 0.45 | 0.7194 | 0.8368 | 0.8389 | 0.8346 | 0.8368 |
| 0.50 (untuned default) | 0.7188 | 0.8364 | 0.8454 | 0.8275 | 0.8364 |
| 0.55 | 0.7184 | 0.8361 | 0.8520 | 0.8208 | 0.8361 |
| 0.60 | 0.7173 | 0.8354 | 0.8586 | 0.8135 | 0.8354 |
| 0.65 | 0.7156 | 0.8343 | 0.8655 | 0.8052 | 0.8343 |
| 0.70 | 0.7131 | 0.8325 | 0.8729 | 0.7957 | 0.8325 |

The curves are textbook-shaped and monotonic where expected (precision rises, recall falls, as
threshold increases) — confirms the sweep is implemented correctly, not an artifact.

**Test-set result at the selected threshold, reported honestly rather than spun as an
improvement:**

| | Test IoU | Test Dice | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---|---|---|---|---|---|
| Threshold 0.40 (validation-selected) | 0.7122 | 0.8319 | 0.8263 | 0.8376 | 0.8319 | 0.9828 |
| Threshold 0.50 (untuned default) | 0.7123 | 0.8320 | 0.8402 | 0.8239 | 0.8320 | 0.9830 |

**The "optimized" threshold does not actually improve test performance** — the difference
(0.7122 vs. 0.7123 IoU) is a tie within noise, not a real gain. This is the honest, correct way
to report this: the validation sweep shows the model's IoU/Dice/F1 are essentially flat across the
whole 0.30-0.70 range (max-min spread of only 0.0065 IoU on validation), meaning the model is
robust to threshold choice in this range rather than having a sharply peaked optimum. The 0.40
selection reflects validation-set noise at this granularity, not a real, generalizable
improvement — a genuine, useful finding (the model doesn't need threshold tuning to perform well)
even though it isn't the finding one might have hoped for. **The dashboard still defaults to 0.40
for this model** (since it is validation-optimal, however marginally) but the threshold remains
fully user-adjustable, and this section documents plainly that the difference from 0.50 is not
practically meaningful.

### 15.3 — Dashboard integration

`dashboard/app.py`: added a "Prediction Probability" panel (viridis heatmap) alongside the
existing before/after/mask/overlay panels, and the threshold slider now defaults to the
validation-optimized value **only when the currently selected model matches the checkpoint that
threshold was swept for** — every other model falls back to the untuned 0.5 default rather than
silently applying a threshold tuned for a different model's output distribution
(`load_selected_threshold()`). All UI text uses "prediction probability"; a dedicated capability-
table row states explicitly that formal calibration is **not implemented**.

### 15.4 — Robustness testing (real, measured)

`scripts/robustness_analysis.py` (perturbations: `src/evaluation/robustness.py`) ran 10 real test
images with ground truth through 6 controlled perturbations, each applied to the **after** image
only (simulating date-to-date variation between the two captures — the more realistic scenario
than perturbing both images identically). IoU is measured against the same, unperturbed ground
truth mask. Full data: `outputs/metrics/robustness_analysis.csv`,
`outputs/metrics/robustness_summary.json`.

| Perturbation | Mean IoU degradation (10 images) |
|---|---|
| Gaussian noise (σ=15) | **+0.0098** (minimal — robust) |
| Brightness +30% | +0.0200 (minimal) |
| Contrast +30% | +0.0005 (negligible) |
| Contrast −30% | **+0.1048** (substantial) |
| Shift 5px (simulated misregistration) | **+0.1178** (substantial) |
| Brightness −30% | **+0.1198** (substantial) |

**A real, honest vulnerability was found, not glossed over:** the model is notably sensitive to
*darkening* (brightness −30%), *reduced contrast*, and *small misregistration* — each costs
roughly 0.10-0.12 IoU on average across the 10-image sample — while it is comparatively robust to
sensor noise and brightening. The single worst case
(`outputs/visualizations/robustness/worst_case_test_101.png`) shows a **0.4234 IoU drop** under
30%-reduced contrast: the model correctly detects most changed buildings on the original image but
misses the large majority of them (mostly false negatives) once contrast is reduced. This is a
genuine limitation, illustrated with a real, saved example, not a hypothetical caveat.

**Interpretation, stated as a hypothesis:** darkening/reduced-contrast and misregistration both
directly attack the specific visual cues (building edges, roof-color contrast against surrounding
terrain) the model was trained to recognize at LEVIR-CD's clean, well-exposed, pre-registered
imagery — this is consistent with, though not proof of, the general concern already documented in
`docs/LIMITATIONS.md` about registration error and lighting sensitivity being real, unaddressed
risk factors for this model, now with a first controlled measurement of their approximate size
rather than only a qualitative concern.

### Phase 15 status summary

| Item | Status |
|---|---|
| Probability maps (visualized, saved) | **Implemented, measured** |
| Threshold sweep (validation-only selection, test-set confirmation) | **Implemented, measured** — found the model is threshold-insensitive in 0.30-0.70 |
| Dashboard probability-map display + validated default threshold | **Implemented** |
| Robustness testing (brightness/contrast/noise/shift) | **Implemented, measured** — found real sensitivity to darkening/contrast/misregistration |
| Formal probability calibration (reliability diagrams, ECE) | **Not implemented** — explicitly named as a gap, not silently assumed |
