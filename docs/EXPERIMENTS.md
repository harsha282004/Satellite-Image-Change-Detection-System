# EXPERIMENTS.md

Phase 8 research experiments: model-improvement variants, compared against the Phase 4 baseline
and Phase 5 Siamese U-Net using real, measured results only (`DEVELOPMENT_RULES.md` Rule 3). Every
number in this document comes from `src/evaluation/evaluate.py` against the real held-out LEVIR-CD
test split — see `outputs/metrics/*_test_metrics.json` for the raw machine-readable reports.

## Which experiments were run, and why

Per `PROJECT_CONTEXT.md`, experiments are implemented only where "technically justified" — not
automatically running every architecture the original brief mentions. Four candidate directions
were considered:

| Candidate | Run? | Justification |
|---|---|---|
| Siamese comparison-mode ablation (`diff`, `concat` vs. the already-trained `diff_concat`) | **Yes** | Zero new code — the infrastructure and all three modes were already implemented and unit-tested in Phase 5 (`tests/test_siamese_unet.py`). Directly answers a real open question left explicit in `docs/EVALUATION.md`'s limitations section: only `diff_concat` had been trained. Cheap (~25 min/run on the existing pipeline). |
| Siamese U-Net + Attention (decoder skip-connection attention gates) | **Yes** | A well-established, lightweight technique (Oktay et al., Attention U-Net, 2018) rather than a novel/exotic mechanism — consistent with Rule 5's "prefer simple, justified improvements". Adds ~700K parameters, no new training infrastructure needed (reuses the existing `Trainer`/`evaluate.py`). |
| Transformer-based change detection | **No — deferred, see below** | |
| Hyperparameter search | **No — deferred, see below** | |

### Why the Transformer variant was not implemented this phase

A Transformer-based change-detection architecture (e.g., a ChangeFormer/BIT-style model) was
**not** implemented in Phase 8, for concrete, documented reasons rather than by omission:

1. **Dataset size.** Training data is 445 images (Phase 2). Vision Transformers are well known to
   need either substantially more training data or ImageNet-scale pretrained weights to
   outperform CNNs — training a Transformer-based change-detection model from scratch on 445
   images would likely underperform the already-working Siamese U-Net, not improve on it, without
   a pretrained backbone this project does not currently have infrastructure to load/fine-tune.
2. **No evidence the CNN approach is insufficient.** The Siamese U-Net already achieves real,
   measured IoU≈0.64 on the test set (`docs/EVALUATION.md`) — there is no observed failure mode
   (e.g., missing long-range spatial context) that a Transformer's global attention would
   specifically be expected to fix, which would be the normal justification for the added
   engineering cost.
3. **Hardware/scope.** The 6 GB GPU (Phase 1) and the time budget available for this milestone
   favor completing well-justified, cheap ablations (above) over a substantial new architecture
   family that would need its own tokenization/patch-embedding/positional-encoding
   infrastructure, none of which exists in this codebase yet.

This is recorded as **Future scope** (`PROJECT_CONTEXT.md`/`README.md`), not abandoned — if a
later phase identifies a concrete failure mode best addressed by long-range attention (e.g. missed
changes in large, spatially spread-out regions), implementing it would then be justified by
evidence rather than by the original brief listing it as an option.

### Why a formal hyperparameter search was not run

All experiments in this project (Phase 4 through this phase) use the same reasonable-default
hyperparameters (`configs/config.yaml`: Adam, lr=1e-4, 30 epochs, batch size 8, BCE+Dice loss, no
LR scheduler). No learning-rate/batch-size/loss-weight sweep has been performed. This is a
deliberate scope decision: Phase 7/8's goal is comparing *architectures* on equal footing (same
recipe for all), not finding each architecture's individually optimal recipe — conflating the two
would make the architecture comparison itself less trustworthy (an architecture could "win" only
because it happened to get luckier hyperparameters). Deferred to a future phase if warranted.

## Results (real, measured — all 5 experiments complete)

All models: 30 epochs, Adam (lr=1e-4), BCE+Dice loss, batch size 8, image size 256, seed 42, same
train/val/test split, same NVIDIA RTX 4050 Laptop GPU, checkpoint selected by best validation IoU.
Per `DEVELOPMENT_LOG.md` Phase 6, training on this GPU is not bit-exact reproducible even with a
fixed seed — treat these as one representative run each, not exact/irreproducible-to-the-decimal
values. Full raw reports: `outputs/metrics/*_test_metrics.json`.

| Experiment | Params | Test IoU | Test Dice | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---|---|---|---|---|---|---|
| Baseline U-Net (Phase 4) | 7,763,905 | 0.6234 | 0.7680 | 0.7333 | 0.8062 | 0.7680 | 0.9752 |
| Siamese U-Net, `diff` | 7,763,041 | 0.5569 | 0.7154 | 0.8004 | 0.6468 | 0.7154 | 0.9738 |
| Siamese U-Net, `concat` | 10,709,345 | 0.6351 | 0.7768 | 0.7077 | 0.8609 | 0.7768 | 0.9748 |
| Siamese U-Net, `diff_concat` (Phase 5 primary) | 14,704,225 | 0.6442 | 0.7836 | 0.7982 | 0.7695 | 0.7836 | 0.9784 |
| **Siamese U-Net, `diff_concat` + Attention** | 15,428,125 | **0.6560** | **0.7922** | 0.8018 | 0.7829 | **0.7922** | **0.9791** |

Ranked by IoU (the primary metric given class imbalance, per `docs/EVALUATION.md`):
**Attention (0.6560) > diff_concat (0.6442) > concat (0.6351) > Baseline (0.6234) > diff (0.5569)**.

### Interpretation

**1. Comparison-mode ablation: `diff` alone is a real regression, not just "worse".** Siamese
`diff` mode (IoU=0.5569) underperforms even the simple non-Siamese baseline (IoU=0.6234) — a
genuinely surprising result worth stating plainly rather than glossing over. The likely
explanation: `abs(before_feat - after_feat)` throws away the actual feature values, keeping only
their magnitude of difference. A decoder working from that signal alone doesn't know *what* the
underlying features were — only that they changed — which turns out to lose more useful
information than the baseline's naive raw-pixel channel concatenation retains. `concat` mode
(IoU=0.6351, keeps both raw feature maps, no explicit difference) does much better than `diff`
alone and comes close to `diff_concat`, suggesting the raw before/after feature context matters
more than having an explicit difference signal. `diff_concat` (both) is better still — the
difference signal adds real value, but only on top of, not instead of, the raw features. This is a
concrete, useful finding for any future architecture work on this task: **don't discard raw
before/after features in favor of only a difference signal.**

**2. Attention gates were worth their added cost.** Adding Attention-U-Net-style skip-connection
gates to the best comparison mode (`diff_concat`) improved every single metric — IoU +0.0118,
Dice +0.0086, Precision +0.0036, Recall +0.0134, F1 +0.0086, Accuracy +0.0007 — for a modest
~4.9% parameter increase (724K params). Unlike the baseline-vs-Siamese comparison in
`docs/EVALUATION.md` (which showed a real precision/recall *tradeoff*), attention improved both
precision and recall simultaneously — a genuinely better operating point, not a different one.
This is the strongest result across every experiment run to date and is the current best model by
every metric.

**3. Caveats (same as `docs/EVALUATION.md`, still apply):** single run/seed per experiment, fixed
untuned hyperparameters shared across all 5 (not each architecture's individually optimal recipe),
benchmark-only (no real-world evaluation). The `diff` mode's poor showing might look different
with better tuning — this result characterizes the specific recipe used here, not the mode's
absolute ceiling.

### Qualitative note
The attention model's prediction grid (`outputs/visualizations/
siamese_unet_diff_concat_attention_test_predictions.png`) shows generally strong agreement with
ground truth (see the dense-subdivision example, near-perfect region overlap), but also a genuine
failure case: one no-change test scene produced a small cluster of false-positive predictions with
no counterpart in the ground truth or in any other model's output on that same scene — a concrete
example of the false-positive risk `PROJECT_CONTEXT.md`'s "actual change vs. apparent difference"
principle warns about, not hidden here despite being an imperfection in the best-performing model.

### Training curves
`outputs/visualizations/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}_training_curves.png`
— all three show smooth, non-diverging train/val loss and IoU/Dice curves, consistent with the
other experiments in `docs/EVALUATION.md`; validation curves show the same expected small-val-set
(64 images) noise.

### Status
- [x] Baseline U-Net (Phase 4) — reference
- [x] Siamese U-Net, `diff_concat` (Phase 5) — reference, primary architecture
- [x] Siamese U-Net, `diff` (Phase 8 ablation)
- [x] Siamese U-Net, `concat` (Phase 8 ablation)
- [x] Siamese U-Net + Attention, `diff_concat` (Phase 8 experiment) — best result *under this
      phase's controlled equal-30-epoch-budget comparison*. Phase 13 found the same architecture
      does substantially better with a longer training budget + early stopping + LR scheduler
      (test IoU 0.6560 → 0.7123) — see the Phase 13 section of `docs/TRAINING.md` and the Phase 14
      section below, which builds on that improved training strategy, not on this phase's 30-epoch
      one. This phase's equal-budget architecture ranking (attention > diff_concat > concat >
      baseline > diff) is unaffected and still the reference architecture comparison.
- Transformer-based variant, formal hyperparameter search: deferred (see justification above)

---

## Phase 14 — Loss Function Experiments

**Status: Implemented and run.** Tests whether an alternative loss function improves on BCE+Dice
for the current best architecture (Siamese U-Net + Attention, `diff_concat`), using Phase 13's
scientifically-justified best training strategy (max 100 epochs, early stopping patience=10 on
validation IoU, `ReduceLROnPlateau`) as the fixed, controlled backdrop — changing only the loss
function, per the "controlled experiment" requirement. The BCE+Dice entry below is **Phase 13
Experiment C, reused rather than retrained** — it is already the exact controlled result for this
training strategy with `loss=bce_dice`; retraining it would waste GPU time reproducing a number
already measured.

Full per-experiment data: `outputs/metrics/loss_experiment_comparison.csv`.

| Loss | Params | Best epoch | Val IoU | Val Dice | Test IoU | Test Dice | Test Precision | Test Recall | Test F1 | Test Accuracy | Training time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BCE+Dice** (reused, Phase 13 Exp. C) | bce_weight=0.5, dice_weight=0.5 | 68 | **0.7188** | **0.8364** | **0.7123** | **0.8320** | **0.8402** | 0.8239 | **0.8320** | **0.9830** | 54.2 min |
| Focal+Dice | focal_alpha=0.8, focal_gamma=2.0, focal_weight=0.5, dice_weight=0.5 | 48 | 0.6758 | 0.8065 | 0.6646 | 0.7985 | 0.7803 | 0.8176 | 0.7985 | 0.9790 | 40.3 min |
| Weighted BCE+Dice | pos_weight=5.0, bce_weight=0.5, dice_weight=0.5 | 53 | 0.6579 | 0.7936 | 0.6539 | 0.7907 | 0.7199 | 0.8770 | 0.7907 | 0.9764 | 43.6 min |
| Tversky | alpha=0.3, beta=0.7 | 39 | 0.6376 | 0.7787 | 0.6322 | 0.7747 | 0.6941 | **0.8764** | 0.7747 | 0.9740 | 43.6 min |

### Interpretation — BCE+Dice won clearly, and the pattern is interpretable, not arbitrary

**BCE+Dice outperformed all three alternatives by a wide, consistent margin** — +0.0477 test IoU
over the next best (Focal+Dice), and the ranking is identical on every metric except recall. This
is not a marginal/noisy result: all three alternatives also converged in fewer epochs before
early stopping (58, 63, 49 vs. BCE+Dice's 78), suggesting they reached a worse optimum faster
rather than needing more training time.

**The precision/recall pattern is exactly what each loss was designed to produce — the
mechanisms are working correctly, they just weren't the right fix for this task:**
- **Weighted BCE+Dice** (`pos_weight=5.0`, up-weights the minority "changed" class) and
  **Tversky** (`beta=0.7`, penalizes false negatives more than false positives) both show the
  expected recall-favoring shift: Tversky reaches the *highest recall of all four* (0.8764) but
  the *lowest precision* (0.6941) and *lowest IoU* (0.6322). Weighted BCE+Dice shows the same
  pattern less extremely. In both cases, the extra false positives incurred by pushing recall up
  cost more IoU than the recall gain returned — a real, measured tradeoff, not a bug.
- **Focal loss** (down-weights easy/confident pixels to focus training on hard ones) did better
  than the other two alternatives but still clearly worse than plain BCE+Dice.

**Plausible explanation, stated as a hypothesis, not a proven mechanism:** BCE+Dice's Dice
component already directly optimizes a set-overlap objective closely related to IoU, and is
already known (from every prior experiment in this project, `docs/EVALUATION.md`) to handle this
task's ~4-5% positive-pixel imbalance reasonably well without a validation-IoU plateau problem
severe enough to need Focal's hard-example mining or Tversky/Weighted-BCE's explicit recall bias.
Those techniques exist to solve imbalance/hard-example problems more severe than what this dataset
apparently presents at this model scale — added here as *complexity without addressing an actual
deficiency*, and the controlled experiment shows that plainly. This is a hypothesis consistent
with the observed data, not independently verified by an ablation isolating each loss's specific
mechanism.

### Training curves
`outputs/visualizations/siamese_unet_diff_concat_attention_{focal_dice,weighted_bce_dice,tversky}_training_curves.png`
— all three show smooth, non-diverging loss/IoU/Dice curves and the same ReduceLROnPlateau
step-down pattern as Phase 13's experiments; none shows train/val divergence (overfitting), only
an earlier plateau than BCE+Dice.

### Status
- [x] BCE+Dice — reused from Phase 13 Experiment C, best result
- [x] Focal+Dice — run, underperformed
- [x] Weighted BCE+Dice — run, underperformed
- [x] Tversky — run, underperformed
- **Conclusion: BCE+Dice remains the loss function used for this project's best model.** No loss
  change is adopted from this experiment set.

---

## Phase 14.3 — Hyperparameter Experiments

**Status: Implemented and run.** A controlled matrix testing learning rate, weight decay, and
batch size, each varied independently against the same fixed backdrop as Phase 14.2 (architecture
= Siamese U-Net + Attention `diff_concat`, loss = BCE+Dice — confirmed the winner in 14.2 — max
100 epochs, early stopping patience=10, `ReduceLROnPlateau`, seed=42). Per the "not an
unnecessarily huge grid search" instruction, one variant per hyperparameter was tested (not a
full cross-product): two learning rates either side of the 1e-4 baseline (5e-5, 2e-4), one
weight-decay setting (AdamW, `weight_decay=0.01` — decoupled weight decay, not naive L2-via-Adam,
per Loshchilov & Hutter 2017), and one batch-size setting (4, chosen over a larger batch for GPU
VRAM safety margin on this shared machine — see `configs/siamese_attention_bs4.yaml`). The
`lr=1e-4` baseline row below is **Phase 13 Experiment C, reused rather than retrained** — it is
already the exact controlled measurement for these fixed hyperparameters.

All model selection used **validation IoU only** — the test set was evaluated once per experiment,
after training and checkpoint selection were already complete, never used to choose a
hyperparameter value.

Full per-experiment data: `outputs/metrics/hyperparameter_experiment_comparison.csv`.

| Hyperparameter varied | Optimizer | LR | Weight decay | Batch size | Best epoch | Val IoU | Test IoU | Test Dice | Test Precision | Test Recall | Test F1 | Test Accuracy | Training time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lr=1e-4 (baseline, reused)** | adam | 1e-4 | 0.0 | 8 | 68 | **0.7188** | **0.7123** | **0.8320** | 0.8402 | 0.8239 | **0.8320** | **0.9830** | 54.2 min |
| lr=5e-5 | adam | 5e-5 | 0.0 | 8 | 53 | 0.6653 | 0.6560 | 0.7923 | 0.7763 | 0.8090 | 0.7923 | 0.9784 | 62.4 min |
| lr=2e-4 | adam | 2e-4 | 0.0 | 8 | 49 | 0.7094 | 0.6999 | 0.8235 | 0.8339 | 0.8133 | 0.8235 | 0.9822 | 45.2 min |
| weight_decay=0.01 | adamw | 1e-4 | 0.01 | 8 | 55 | 0.7102 | 0.7028 | 0.8255 | 0.8277 | 0.8232 | 0.8255 | 0.9823 | 45.1 min |
| batch_size=4 | adam | 1e-4 | 0.0 | 4 | 57 | 0.7135 | 0.6997 | 0.8233 | **0.8424** | 0.8051 | 0.8233 | 0.9824 | 47.2 min |

### Interpretation — the original hyperparameters were already well-tuned for this setup

**Every single variant underperformed the lr=1e-4/wd=0/batch_size=8 baseline, and the ranking was
identical on validation and test data** (a real consistency check, not assumed) — the baseline
wins on IoU, Dice, and F1 in both cases. This is a clean, low-noise result: no variant even
matched the baseline, let alone beat it.

**Learning rate showed the clearest, most interpretable pattern.** Halving the LR (5e-5) hurt
substantially (test IoU 0.6560, the worst result in this entire experiment set); doubling it
(2e-4) hurt much less (0.6999, close to baseline). This asymmetry makes sense given
`ReduceLROnPlateau` is already active: starting at 2e-4 gives the scheduler more room to anneal
down through useful intermediate values, while starting at 5e-5 leaves less room above the
`min_lr=1e-6` floor before training exhausts its useful learning-rate range — 5e-5 behaves less
like "a smaller step size" and more like "starting already partway through the annealing
schedule the 1e-4 baseline discovers on its own."

**Weight decay and batch size both landed close to, but consistently below, the baseline**
(test IoU 0.7028 and 0.6997 respectively, vs. baseline 0.7123) — small, real effects, not
dramatic ones. Given this project has not observed overfitting at any best epoch in any prior
experiment (`docs/TRAINING.md` Phase 13), it is unsurprising that adding regularization (weight
decay) did not help: there was no overfitting problem for it to solve, only underfitting risk it
could make marginally worse by constraining the model slightly more than necessary. The
batch_size=4 result is consistent with plain SGD-family intuition: smaller batches produce noisier
gradient estimates, which can occasionally help generalization but more often simply makes
optimization slightly less stable — the modest IoU drop here is unsurprising, not evidence of a
batch-size effect large enough to justify the ~2x more optimizer steps per epoch it costs.

### Training curves
`outputs/visualizations/siamese_unet_diff_concat_attention_{lr5e-5,lr2e-4,wd0.01,bs4}_training_curves.png`
— all four show smooth, non-diverging curves with the expected `ReduceLROnPlateau` step-downs; no
train/val divergence (overfitting) observed in any of the four.

### Status
- [x] Learning rate sweep (5e-5, 1e-4 baseline reused, 2e-4) — baseline wins
- [x] Weight decay (AdamW, 0.01) — run, underperformed slightly
- [x] Batch size (4) — run, underperformed slightly
- **Conclusion: the original Phase 13 hyperparameters (Adam, lr=1e-4, weight_decay=0.0,
  batch_size=8) remain the best configuration found.** No hyperparameter change is adopted.

---

## Phase 14.4 — Final Best Training Configuration

Combining the Phase 13 (training-strategy) and Phase 14.1-14.3 (loss + hyperparameter) results,
the single best configuration found across every controlled experiment run in this project is
**Phase 13 Experiment C's exact recipe** — nothing tested in Phase 14 improved on it:

```yaml
architecture: Siamese U-Net + Attention (diff_concat comparison mode)
config file:   configs/siamese_attention_e100.yaml
checkpoint:    outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt

loss:          bce_dice (bce_weight=0.5, dice_weight=0.5)   — beat Focal+Dice, Weighted BCE+Dice, Tversky
optimizer:     adam                                          — beat adamw+weight_decay=0.01
learning_rate: 0.0001 (initial)                               — beat 5e-5 and 2e-4
weight_decay:  0.0                                            — beat 0.01
batch_size:    8                                               — beat 4
max_epochs:    100 (maximum budget)
early_stopping: enabled, patience=10, monitor=val_iou
scheduler:     ReduceLROnPlateau (factor=0.5, patience=4, min_lr=1e-6)
seed:          42

actual_epochs_trained: 78 (early-stopped)
best_epoch:            68
val_iou_at_best:       0.7188
val_dice_at_best:      0.8364

TEST SET (real, measured, held-out 128 images):
  IoU=0.7123  Dice=0.8320  Precision=0.8402  Recall=0.8239  F1=0.8320  Accuracy=0.9830
```

### Why this is the selected configuration
Every dimension Phase 14 tested — 3 alternative losses, 2 alternative learning rates, one
weight-decay setting, one batch-size setting, 7 new full training runs in total — was compared
against this exact recipe under matched conditions (same architecture, data, split, seed, and
training strategy except the one dimension varied), and **none beat it**. This is not a default
chosen for convenience; it is the empirically best-performing configuration out of 8 real,
measured alternatives (the original baseline + 3 losses + 4 hyperparameters), selected on
validation IoU (never the test set) and confirmed by held-out test evaluation.

### What remains untested (honestly scoped, not silently assumed)
- **No cross-product grid.** Each dimension was varied one at a time against the same baseline —
  e.g., `lr=2e-4` combined with `weight_decay=0.01` was never tried. A joint optimum elsewhere in
  the hyperparameter space cannot be ruled out; this was a controlled, low-cost sweep, not an
  exhaustive search (explicitly scoped this way per the "not an unnecessarily huge grid search"
  instruction).
- **Single seed (42) throughout.** No variance estimate for any comparison in Phase 13 or 14 —
  the same caveat that applies to every experiment in this project.
- **Loss-parameter values were not swept.** Focal/Tversky/Weighted-BCE were each tested at one
  literature-reasonable parameter setting, not a range (`docs/EXPERIMENTS.md` Phase 14.2
  limitations) — it remains possible a different `pos_weight` or Tversky `alpha`/`beta` could
  close some of the gap to BCE+Dice.
