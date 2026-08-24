# TRAINING.md

How every model in this project was actually trained. Describes the implemented pipeline
(`src/training/`) and the real configuration/commands used to produce every result in
`docs/EVALUATION.md` and `docs/EXPERIMENTS.md` — not a general guide to hypothetical training
options.

## Pipeline (`src/training/`)

- `train.py` — entry point. Loads a YAML config (with a one-level `extends:` base-config merge —
  every experiment config extends `configs/config.yaml`), sets `random`/`numpy`/`torch`/
  `torch.cuda` seeds, resolves the device (`"auto"` → CUDA if available, else CPU), builds the
  dataset/dataloaders/model/optimizer/loss from the config, and runs `Trainer.fit()`.
- `trainer.py` — `Trainer`: one training epoch, one validation pass (via `validate.py`, shared
  with the standalone test-set evaluation in `src/evaluation/evaluate.py`, so training-time
  validation and final test evaluation use identical metric computation), saves `last.pt` every
  epoch and `best.pt` whenever validation IoU improves, logs every epoch's metrics.
- `checkpoint.py` — every checkpoint bundles the model state, optimizer state, epoch, validation
  metrics at that epoch, and the **full config used to produce it** — enough to know exactly how
  any checkpoint was produced (Rule 7), even without a matching git commit.
- `logger.py` — `ExperimentLogger` writes `outputs/experiments/<name>/history.csv` (one row per
  epoch, immediately flushed) and a final `history.json`. `src/visualization/plots.py` reads
  `history.csv` to render the training-curve PNGs referenced throughout `docs/EVALUATION.md` and
  `docs/EXPERIMENTS.md`.

## Command

```bash
python -m src.training.train --config configs/<experiment>.yaml [--epochs N]
```

`--epochs` overrides the config's epoch count — used for 1-2 epoch smoke tests before every real
run in this project (confirms the pipeline works end-to-end — config loading, dataloaders, model,
optimizer, checkpointing, logging — before committing 20-30 minutes of GPU time to a full run).

## Configuration actually used

Every experiment in this project (`configs/baseline.yaml`, `configs/siamese.yaml`,
`configs/siamese_diff.yaml`, `configs/siamese_concat.yaml`, `configs/siamese_attention.yaml`)
extends the same base (`configs/config.yaml`) and shares this training recipe — deliberately, so
architecture comparisons in `docs/EXPERIMENTS.md` are apples-to-apples rather than confounded by
different hyperparameters per model:

```yaml
dataset:
  image_size: 256
dataloader:
  batch_size: 8
  num_workers: 0
training:
  epochs: 30
  learning_rate: 0.0001
  optimizer: adam
  scheduler: none      # no LR scheduler implemented/used — explicit, not silently absent
  loss: bce_dice
  seed: 42
device: auto           # cuda if available (Phase 1 confirmed an RTX 4050 Laptop GPU), else cpu
```

Only `model:` (type, `base_channels`, `comparison_mode`, `use_attention`) and `experiment_name`
differ between configs — see each file for the exact per-experiment values, and
`docs/ARCHITECTURE.md` for what each model type means.

## Why these specific choices
- **`image_size=256`, not LEVIR-CD's native 1024.** Chosen for VRAM/compute tractability on the
  6 GB GPU identified in Phase 1 (`docs/ARCHITECTURE.md`).
- **`batch_size=8`.** Verified to fit comfortably within the GPU's VRAM for every model variant,
  including the largest (Siamese+Attention, `diff_concat`, ~2.8 GB peak — `docs/EXPERIMENTS.md`
  benchmarking table, Phase 5 entry in `DEVELOPMENT_LOG.md`).
- **`loss: bce_dice`.** BCE alone is a common default; Dice is added specifically because the task
  is heavily class-imbalanced (~4-5% changed pixels, `docs/DATASET.md`) and Dice loss is less
  dominated by the majority (unchanged) class than BCE alone.
- **`epochs=30`, fixed across every experiment.** A deliberately bounded budget for a controlled
  comparison, not a claim that any model had converged — validation IoU was still trending upward
  at epoch 30 for most runs (`DEVELOPMENT_LOG.md` Phase 4/5/8 "Known issues";
  `docs/LIMITATIONS.md`).
- **No LR scheduler, no hyperparameter search.** Same reasoning as above — isolates the
  architecture comparison from confounding hyperparameter differences (`docs/EXPERIMENTS.md`
  "Why a formal hyperparameter search was not run").

## Reproducibility — what's guaranteed and what isn't

Every checkpoint's bundled config + the fixed seed is enough to *rerun* the exact same training
recipe. **It is not enough to reproduce bit-identical results.** Phase 6 discovered, by actually
testing it (not assuming it), that training on this GPU is not bit-exact reproducible even with an
identical seed and config — non-deterministic cuDNN convolution algorithm selection causes
numerical drift starting from epoch 2, which compounds enough over 30 epochs to select a different
"best" checkpoint with a measurably different precision/recall balance
(`DEVELOPMENT_LOG.md` Phase 6, full incident and finding). `torch.use_deterministic_algorithms`
was deliberately left off (it costs training speed); this tradeoff is documented, not hidden.
Anyone re-running the commands in this document should expect results close to, but not
necessarily identical to, the numbers in `docs/EVALUATION.md`/`docs/EXPERIMENTS.md`.

## Checkpoints and experiment tracking

`outputs/checkpoints/<experiment_name>/{best,last}.pt` and
`outputs/experiments/<experiment_name>/{history.csv,history.json}` are produced for every
experiment (gitignored — regenerable by rerunning the training command, not committed since they
total several hundred MB across all 5 experiments). **Always use a distinct `experiment_name`
for any new run** — reusing an existing tracked experiment's name overwrites its checkpoint and
history, which is exactly what happened accidentally during Phase 6's reproducibility testing and
had to be fixed by retraining (`DEVELOPMENT_LOG.md` Phase 6).

---

## Phase 13 — Advanced Training Strategy: early stopping, LR scheduling, longer training

### Motivation
Every Phase 4-8 experiment used a *fixed* 30-epoch budget, deliberately, to keep the Phase 8
architecture comparison controlled (see "Why these specific choices" above). But
`DEVELOPMENT_LOG.md`'s Phase 4/5/8 entries repeatedly noted validation IoU was still climbing at
epoch 30 — a documented *undertraining* concern, not a claim that any architecture had converged.
Phase 13's purpose was to test that concern experimentally rather than leave it as a caveat:
**does the winning architecture (Siamese U-Net + Attention, `diff_concat`) actually improve with
more training, and if so, does it eventually overfit or plateau?**

### What was added (`src/training/trainer.py`, `src/training/train.py`)
- **Early stopping** (`Trainer`, new `early_stopping` constructor arg): monitors validation IoU;
  if it does not improve for `patience` consecutive epochs, training stops. The best-val-IoU
  checkpoint is always retained regardless of when training stops — early stopping only shortens
  *how long training continues after* the best epoch, never discards the best result.
- **`ReduceLROnPlateau` scheduler** (`train.py::build_scheduler`): halves the learning rate
  (`factor=0.5`) after `patience=4` epochs without validation-IoU improvement, down to a
  `min_lr=1e-6` floor. Current LR is logged every epoch (`lr` column in `history.csv`) and plotted
  (`src/visualization/plots.py`, new conditional LR subplot — only rendered for experiments that
  logged an `lr` column, so the 5 pre-Phase-13 experiments' plots are byte-for-byte unchanged).
- **AdamW + configurable weight decay** (`train.py::build_optimizer`): Adam remains the default,
  unchanged for every existing config; AdamW is available as `training.optimizer: adamw` for
  future experiments (Phase 14).
- **Backward compatibility, verified not just assumed:** every new `Trainer`/`build_*` parameter
  defaults to "off" (no scheduler, no early stopping, `weight_decay=0.0`), so any config written
  before Phase 13 trains identically to how it did before. Verified two ways: (1) 18 new unit
  tests (`tests/test_trainer.py`) using a mocked `validate()` to deterministically check early-
  stopping/scheduler control flow and a regression test proving training runs the full epoch count
  when early stopping is unconfigured; (2) a real smoke run of the pre-existing
  `configs/baseline.yaml` (under a throwaway experiment name) reproducing the *exact* epoch-1
  `train_loss=0.7457` seen in every prior baseline run.

### Experiments run

All three share: architecture=`siamese_unet_diff_concat_attention` (Siamese U-Net, `diff_concat`,
`use_attention=true`), dataset/split (LEVIR-CD, 445/64/128, Phase 2), optimizer=Adam, initial
lr=1e-4, weight_decay=0.0, batch_size=8, loss=BCE+Dice, seed=42, image_size=256 — identical to
Experiment A/Phase 8 in every respect except the ones being tested (max epochs, early stopping,
LR scheduler). Full per-experiment stats: `outputs/metrics/training_experiment_comparison.csv`.

| | Experiment A (Phase 8, preserved) | Experiment B | Experiment C |
|---|---|---|---|
| Config | `configs/siamese_attention.yaml` | `configs/siamese_attention_e60.yaml` | `configs/siamese_attention_e100.yaml` |
| Max epochs | 30 | 60 | 100 |
| Early stopping | Not implemented at the time | Enabled, patience=10 | Enabled, patience=10 |
| LR scheduler | None (constant 1e-4) | ReduceLROnPlateau | ReduceLROnPlateau |
| **Actual epochs trained** | 30 | **60** | **78** |
| Early stopped? | N/A (no such feature yet) | **No** — hit the 60-epoch ceiling, still improving | **Yes** — no val IoU improvement for 10 epochs after epoch 68 |
| Best epoch | 26 | 60 | 68 |
| Best val IoU | 0.6702 | 0.7106 | **0.7188** |
| Best val Dice | 0.8026 | 0.8308 | **0.8364** |
| LR at best epoch | 1e-4 (constant) | 2.5e-5 (reduced twice) | 1.25e-5 (reduced three times) |
| Test IoU | 0.6560 | 0.7031 | **0.7123** |
| Test Dice | 0.7922 | 0.8257 | **0.8320** |
| Test Precision | 0.8018 | 0.8176 | **0.8402** |
| Test Recall | 0.7829 | **0.8339** | 0.8239 |
| Test F1 | 0.7922 | 0.8257 | **0.8320** |
| Test Accuracy | 0.9791 | 0.9821 | **0.9830** |
| Training time | not precisely measured (pre-Phase-13 timing code) | 3491.9s (58.2 min) | 3253.9s (54.2 min) |

Checkpoints, kept fully separate per Rule 7/11: `outputs/checkpoints/siamese_unet_diff_concat_attention{,_e60,_e100}/best.pt`. Experiment A's checkpoint, history, and test-metrics JSON were never
touched — confirmed by re-reading them unchanged after B and C finished.

### Interpretation — was Experiment A undertrained? Yes, substantially.

**Experiment A was genuinely undertrained.** Giving the identical architecture/data/optimizer more
epochs plus an LR scheduler improved test IoU from 0.6560 to 0.7123 — a **+0.0563 absolute
improvement (+8.6% relative)**, far larger than any of the Phase 8 architecture-choice effects
(e.g. adding attention gates to `diff_concat` only gained +0.0118 IoU in Phase 8's equal-budget
comparison). Training strategy mattered more than architecture choice, for this model.

**The mechanism is visible in the training curves, not just inferred from the final numbers.**
`outputs/visualizations/siamese_unet_diff_concat_attention_e60_training_curves.png`'s LR panel
shows two step-downs (epoch 39: 1e-4→5e-5; epoch 58: 5e-5→2.5e-5), and both are followed by a
visible acceleration in validation IoU/Dice improvement — the scheduler was doing real, observable
work, not just a hyperparameter that happened to be set.

**Overfitting was NOT observed in either B or C.** At every experiment's best epoch, train and
validation metrics track closely (e.g. Experiment C, epoch 68: train_iou=0.7238 vs.
val_iou=0.7188 — within noise, val not trailing train by a meaningful margin). Experiment C's
early stopping triggered because validation IoU plateaued (oscillating around 0.70-0.72 for the
final ~10 epochs, `outputs/experiments/siamese_unet_diff_concat_attention_e100/history.csv`), not
because it started climbing on train while falling on val.

**Experiment B is a real, unresolved open question, reported honestly rather than smoothed over:**
B ran its full 60-epoch budget without early stopping ever triggering, and its last epoch (60) was
still its best epoch — meaning the 60-epoch ceiling, not a genuine plateau, is what ended B's
training. It is not known whether B would have kept improving with a higher ceiling; that question
is exactly what Experiment C (max 100) was designed to probe, and C's early stopping at epoch 78
(best epoch 68) does show the improvement trend genuinely leveling off eventually — but only
confirmed for *this* architecture, seed, and hyperparameter recipe, not as a general claim.

### Known limitations of this experiment set
- **Single seed (42) for all three.** No variance/confidence-interval estimate — consistent with
  every other experiment in this project (`docs/LIMITATIONS.md`), and the same GPU-non-determinism
  caveat from Phase 6 applies here too.
- **Experiment A's training time was not precisely measured** — `time.time()` wall-clock
  measurement was added to `train.py` in Phase 13 itself, so it did not exist when Experiment A was
  originally trained in Phase 8. Reported as `NOT_PRECISELY_MEASURED_PRE_PHASE13` in the CSV rather
  than estimated or backfilled.
- **Early-stopping/scheduler patience values (10 and 4) were not themselves tuned** — they are the
  literal defaults specified for this phase, not the result of a search. A different patience could
  plausibly change where C stops.
- **This is one architecture.** Whether the baseline U-Net or the other Siamese comparison modes
  (`diff`, `concat`) would show the same magnitude of improvement from longer training + scheduling
  is untested — Phase 8's equal-30-epoch comparison remains the only controlled cross-architecture
  result in this project.
