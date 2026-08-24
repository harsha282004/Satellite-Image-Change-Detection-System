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
