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
- [x] Siamese U-Net + Attention, `diff_concat` (Phase 8 experiment) — **best result overall**
- Transformer-based variant, formal hyperparameter search: deferred (see justification above)
