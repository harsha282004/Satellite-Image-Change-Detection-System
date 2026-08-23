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
