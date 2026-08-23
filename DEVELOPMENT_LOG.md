# DEVELOPMENT_LOG.md

Running log of phase completions. Newest entry at the top. See `PROJECT_CONTEXT.md` for phase
definitions and `DEVELOPMENT_RULES.md` for the verification rules each entry must satisfy.

---

## PHASE 8 — Model Improvement & Research Experiments

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- **Scope decision, made and documented before writing any code** (per Rule 5 — only implement
  where technically justified, not automatically): ran the deferred Siamese comparison-mode
  ablation (`diff`, `concat` — cheap, infra already existed) and a Siamese + Attention variant
  (well-established, lightweight technique). Explicitly deferred a Transformer-based variant and a
  formal hyperparameter search, with concrete reasoning written into `docs/EXPERIMENTS.md` (small
  445-image training set unfavorable for training a Transformer from scratch, no observed CNN
  failure mode a Transformer would specifically fix, hardware/scope) rather than silently skipping
  them or building them speculatively.
- `models/attention.py`: `AttentionGate` (standard additive attention gate, Oktay et al. 2018) and
  `AttentionUp` (drop-in alternative to `models.unet.Up`, reusing `DoubleConv` — Rule 6).
- `models/siamese_unet.py`: added a `use_attention: bool = False` constructor parameter to
  `SiameseUNet` — swaps `Up` for `AttentionUp` at all 4 decoder stages when `True`. Verified the
  default (`False`) is architecturally unchanged from Phase 5
  (`tests/test_attention.py::test_siamese_unet_use_attention_false_matches_phase5_architecture`).
- `configs/siamese_diff.yaml`, `configs/siamese_concat.yaml`, `configs/siamese_attention.yaml` —
  distinct `experiment_name`s for each (explicitly avoiding the Phase 6 checkpoint-overwrite
  incident by never reusing an existing experiment's name).
- 6 new pytest tests (`tests/test_attention.py`): gate output shape/boundedness, `AttentionUp`
  forward shape, full `SiameseUNet(use_attention=True)` forward/backward/optimizer-step, and the
  use_attention True/False class-selection checks.
- Trained and evaluated all 3 new experiments for real, on the held-out test set, then wrote the
  full 5-experiment comparison, ablation interpretation, and a qualitative failure-case note into
  `docs/EXPERIMENTS.md`. Updated `docs/ARCHITECTURE.md` and `README.md` to reflect the new best
  model.

**FILES CREATED:**
- `models/attention.py`
- `configs/siamese_diff.yaml`, `configs/siamese_concat.yaml`, `configs/siamese_attention.yaml`
- `tests/test_attention.py`
- `outputs/checkpoints/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}/
  {best,last}.pt` (gitignored)
- `outputs/experiments/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}/
  {history.csv,history.json}` (gitignored)
- `outputs/metrics/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}_test_metrics.json` (gitignored)
- `outputs/visualizations/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}_test_predictions.png` (gitignored)
- `outputs/visualizations/{siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}_training_curves.png` (gitignored)

**FILES MODIFIED:**
- `models/siamese_unet.py` (added `use_attention` parameter)
- `docs/ARCHITECTURE.md` (new Attention section)
- `docs/EXPERIMENTS.md` (results filled in — was scaffolded with methodology/justification only)
- `README.md` (Results, Experiments, Future Scope sections updated)

**COMMANDS EXECUTED:**
- `pytest tests/ -q` (before and after the new code, 40 -> 47 tests)
- `python -m src.training.train --config configs/{siamese_diff,siamese_concat,siamese_attention}.yaml --epochs 1`
  (3 smoke tests, distinct experiment names confirmed before any full run)
- `python -m src.training.train --config configs/siamese_diff.yaml` (full 30-epoch run, background)
- `python -m src.evaluation.evaluate --config configs/siamese_diff.yaml --checkpoint outputs/checkpoints/siamese_unet_diff/best.pt`
- `python -m src.training.train --config configs/siamese_concat.yaml` (full 30-epoch run, background)
- `python -m src.evaluation.evaluate --config configs/siamese_concat.yaml --checkpoint outputs/checkpoints/siamese_unet_concat/best.pt`
- `python -m src.training.train --config configs/siamese_attention.yaml` (full 30-epoch run, background)
- `python -m src.evaluation.evaluate --config configs/siamese_attention.yaml --checkpoint outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt`
- `python -m src.visualization.plots --experiment {siamese_unet_diff,siamese_unet_concat,siamese_unet_diff_concat_attention}`

**TESTS:**
- `pytest tests/`: 47/47 passed (40 from Phase 7 + 6 new attention tests + the pre-existing
  `test_siamese_unet_shares_encoder_weights...` count was unaffected).
- 3 one-epoch smoke tests (one per new config) run before any full 30-epoch training, each
  confirmed to use a distinct `experiment_name` from every existing tracked experiment (the
  concrete lesson from the Phase 6 incident, applied here).
- 3 full 30-epoch training runs on the real LEVIR-CD train/val split, same protocol as Phase 4/5 —
  all converged smoothly (see training-curve PNGs), no divergence or NaN losses.
- 3 real, measured test-set evaluations on the held-out LEVIR-CD test split (same 128 samples,
  same checkpoint-selection-by-validation-IoU protocol as every prior experiment).
- Manually inspected all 3 new qualitative prediction grids, including the attention model's —
  found and documented one genuine false-positive cluster on a no-change scene (see
  `docs/EXPERIMENTS.md` "Qualitative note"), not hidden despite it being the best-performing model.

**RESULTS (actual, measured — full report and interpretation in `docs/EXPERIMENTS.md`; raw JSON:
`outputs/metrics/*_test_metrics.json`):**
```
Experiment                                Params        Test IoU   Dice     Precision  Recall   F1       Accuracy
Baseline U-Net                            7,763,905     0.6234     0.7680   0.7333     0.8062   0.7680   0.9752
Siamese U-Net, diff                       7,763,041     0.5569     0.7154   0.8004     0.6468   0.7154   0.9738
Siamese U-Net, concat                     10,709,345    0.6351     0.7768   0.7077     0.8609   0.7768   0.9748
Siamese U-Net, diff_concat (Phase 5)      14,704,225    0.6442     0.7836   0.7982     0.7695   0.7836   0.9784
Siamese U-Net, diff_concat + Attention    15,428,125    0.6560     0.7922   0.8018     0.7829   0.7922   0.9791  <- best
```
Two genuine, non-obvious findings (full reasoning in `docs/EXPERIMENTS.md`):
1. **`diff` alone underperforms the baseline** (0.5569 vs. 0.6234 IoU) — discarding raw before/
   after feature values in favor of only their difference loses more information than it gains.
2. **Attention gates improved every metric simultaneously** (not a tradeoff) over the best prior
   model, for a modest ~4.9% parameter increase.

**KNOWN ISSUES:**
- Same reproducibility caveat as Phase 6/7: single run/seed per experiment on a GPU that is not
  bit-exact reproducible even with a fixed seed.
- Fixed, shared, untuned hyperparameters across all 5 experiments (deliberate — see
  `docs/EXPERIMENTS.md`'s "Why a formal hyperparameter search was not run").
- One genuine false-positive failure case identified in the best model's qualitative grid,
  documented rather than cropped out of the discussion.
- Transformer-based variant and formal hyperparameter search deliberately deferred, with reasoning
  recorded (`docs/EXPERIMENTS.md`), not silently skipped.

**NEXT PHASE:**
- PHASE 9 — Change Region Analysis & Quantification: connected-component extraction from a
  predicted mask (using the best model found here — Siamese + Attention), per-region statistics
  (count, area, largest/average region), and physical-area conversion only where a pixel
  resolution/geotransform is actually available and the assumption is documented (LEVIR-CD's
  known 0.5 m/pixel resolution, Phase 2).

---

## PHASE 7 — Evaluation & Visualization

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Most of the underlying evaluation work this phase covers (test-set IoU/Dice/Precision/Recall/
  F1/Accuracy, qualitative before/after/GT/prediction/overlay/diff grids, training curves) was
  already implemented and actually run in Phases 4-6. Phase 7's concrete new work is consolidating
  it into a single rigorous, honest writeup:
- `docs/EVALUATION.md`: full evaluation methodology (test-set/leakage guarantees, class-imbalance
  rationale for why accuracy alone is insufficient, confusion-matrix-accumulation metric
  methodology, checkpoint-selection protocol, the Phase 6 reproducibility caveat), the real
  quantitative comparison table (restated from the corrected Phase 6 numbers) plus raw TP/FP/FN/TN
  counts, an interpretation section explaining the precision/recall tradeoff in terms of the actual
  false-positive/false-negative pixel counts, qualitative-results discussion, training-curve
  discussion (noting validation-curve noise from the small 64-image val set), and an explicit
  "Limitations of this evaluation" section (single seed, untuned hyperparameters, benchmark-only,
  fixed 0.5 threshold, only one Siamese comparison mode trained) plus a status-summary table
  distinguishing Implemented/Measured from Planned items.
- `README.md` Evaluation section updated to point to the new doc.

**FILES CREATED:**
- `docs/EVALUATION.md`

**FILES MODIFIED:**
- `README.md` (Evaluation section)

**COMMANDS EXECUTED:**
- Re-inspected `outputs/metrics/{baseline_unet,siamese_unet_diff_concat}_test_metrics.json` to
  confirm the numbers written into `docs/EVALUATION.md` exactly match the saved machine-readable
  reports (no transcription drift).
- Visually re-inspected `outputs/visualizations/baseline_unet_test_predictions.png` (regenerated
  post-Phase-6-restore) to confirm the qualitative discussion in `docs/EVALUATION.md` (more
  false-positive speckling than the Siamese grid) is accurate to the actual current image, not the
  pre-restore one.
- `pytest tests/ -q` — confirm no regression (no code changed this phase, documentation only).

**TESTS:**
- `pytest tests/`: 40/40 passed (unchanged from Phase 6 — this phase touched no source code).
- Cross-checked every number quoted in `docs/EVALUATION.md`'s quantitative table and confusion-
  matrix table character-for-character against the JSON metric files, per Rule 3 (never fabricate
  — and never let a transcription typo silently become a fabrication either).

**RESULTS:**
No new model training or metrics computation this phase — `docs/EVALUATION.md` is a rigorous
writeup of results already measured and recorded in the Phase 4/5/6 entries above. See those
entries (or `docs/EVALUATION.md` directly) for the real numbers.

**KNOWN ISSUES:**
- Carried forward from `docs/EVALUATION.md`'s own limitations section: single run/seed per model,
  untuned hyperparameters, benchmark-only (no real-world evaluation yet — Phase 11), fixed 0.5
  decision threshold (no PR-curve/operating-point sweep), only `diff_concat` Siamese mode
  evaluated (Phase 8 ablation deferred).

**NEXT PHASE:**
- PHASE 8 — Model Improvement & Research Experiments: only implement additional experiments
  (attention variant, Transformer variant, the untrained `diff`/`concat` Siamese comparison modes,
  hyperparameter tuning) where technically justified by hardware/time/dataset-size constraints and
  expected measurable value — not automatically, per `DEVELOPMENT_RULES.md` Rule 5. Record an
  honest experiment comparison table with real results for whatever is actually run.

---

## PHASE 6 — Training & Experiment Pipeline

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Most of the config-driven training/checkpointing/logging infrastructure this phase covers was
  actually already built and exercised in Phases 4-5 (`src/training/`, `configs/*.yaml`,
  `outputs/experiments/*/history.csv`+`history.json`, `outputs/checkpoints/*/{best,last}.pt` with
  full config bundled in). Phase 6's concrete new work:
- `src/visualization/plots.py`: `plot_training_curves(experiment_name)` reads an experiment's
  `history.csv` and renders a 2x2 grid — Epoch vs. Loss (train+val), Epoch vs. IoU (train+val),
  Epoch vs. Dice (train+val), and a bonus Epoch vs. Validation Precision/Recall/F1 panel — exactly
  the curves `PROJECT_CONTEXT.md`/the master spec calls for. Generated and visually inspected for
  both `baseline_unet` and `siamese_unet_diff_concat`.
- Added an explicit `scheduler: none` field to `configs/config.yaml` (inherited by both
  experiment configs) — documents that no LR scheduler is implemented/used, rather than silently
  omitting the field (Rule 4: distinguish implemented/not, don't imply unstated capability).
- **Reproducibility check (Rule 7) actually run, not assumed:** re-ran `configs/baseline.yaml`
  for 1 epoch twice with the same seed. Epoch-1 train metrics matched exactly
  (train_loss=0.7457, train_iou=0.1363 both times); validation metrics differed slightly
  (val_iou 0.2802 vs 0.2829) — evidence of GPU (cuDNN) convolution non-determinism, not a bug in
  the seeding code (`random`/`numpy`/`torch`/`torch.cuda` seeds are all set in `train.py`).
- **Incident during this check, documented rather than hidden:** the two 1-epoch smoke-test runs
  used the *same* `experiment_name` (`baseline_unet`) as the real Phase 4 run, so they overwrote
  its `best.pt`/`last.pt` checkpoints and `history.csv` with 1-epoch data. Caught immediately by
  re-inspecting `history.csv` (found only 2 lines instead of 31). Fixed by re-running
  `configs/baseline.yaml` for the full 30 epochs with the identical config/seed to restore a
  complete, real experiment.
- **This restore run surfaced a more significant reproducibility finding:** it was NOT bit-exact
  reproducible beyond epoch 1 — train metrics started drifting from epoch 2 onward (e.g. original
  epoch 2 train_loss=0.6741 vs. restored run's 0.6746), and the drift compounded over 30 epochs
  enough that a different epoch was selected as "best" by validation IoU (epoch 30 vs. the
  original's epoch 29), with a real precision/recall tradeoff shift at the selected checkpoint
  (restored: precision=0.7333/recall=0.8062 vs. original: precision=0.7681/recall=0.7703) despite
  similar aggregate IoU/Dice/F1. **All baseline numbers in this log and in README.md have been
  updated to the current, real, restored-run results** — see the Phase 4 entry's "Checkpoint
  overwrite" note and the corrected Phase 5 comparison table. The Phase 5 `siamese_unet_diff_concat`
  experiment was not touched by this incident and its original numbers stand unchanged.

**FILES CREATED:**
- `src/visualization/__init__.py`, `src/visualization/plots.py`
- `outputs/visualizations/{baseline_unet,siamese_unet_diff_concat}_training_curves.png` (gitignored)

**FILES MODIFIED:**
- `configs/config.yaml` (added explicit `scheduler: none`)
- `outputs/checkpoints/baseline_unet/{best,last}.pt`, `outputs/experiments/baseline_unet/
  {history.csv,history.json}`, `outputs/metrics/baseline_unet_test_metrics.json`,
  `outputs/visualizations/baseline_unet_test_predictions.png` — all regenerated by the restore
  retrain + re-evaluation (gitignored, but noting the real cause of the update)
- `DEVELOPMENT_LOG.md` Phase 4 and Phase 5 entries corrected in place with the restored-run's real
  numbers, with the overwrite incident and reproducibility finding documented rather than silently
  fixed
- `README.md` Results table corrected to match

**COMMANDS EXECUTED:**
- `python -m src.visualization.plots --experiment baseline_unet`
- `python -m src.visualization.plots --experiment siamese_unet_diff_concat`
- `python -c "torch.load(...); print(checkpoint['config'])"` — verified saved checkpoints contain
  a complete, sufficient config to reproduce the run's setup
- `python -m src.training.train --config configs/baseline.yaml --epochs 1` (x2, reproducibility
  check — this is what caused the overwrite)
- `python -m src.training.train --config configs/baseline.yaml` (full 30-epoch restore retrain)
- `python -m src.evaluation.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_unet/best.pt`
  (re-evaluation of the restored checkpoint)
- `pytest tests/ -q` (confirm nothing broken by config change)

**TESTS:**
- `pytest tests/`: 40/40 passed after the `scheduler: none` config change (config-merge logic
  exercised indirectly via `train.py`'s existing tests-adjacent smoke runs; no dedicated config
  unit test existed or was deemed necessary for a single added key — verified manually instead via
  `load_config()` output inspection, shown in the terminal transcript).
- Manually inspected both training-curve PNGs: both show smooth, non-diverging loss curves and
  IoU/Dice trending upward with the expected small-validation-set (64 samples) noise; no anomalies.
- Verified checkpoint reproducibility content directly: `best.pt`'s bundled `config` dict exactly
  matches the YAML config used to train it (including `seed`), confirming Rule 7's "every
  experiment must have enough information to reproduce it" — with the important, now-documented
  caveat that *exact numeric* reproduction on GPU is not guaranteed even with identical config+seed.

**RESULTS (actual, measured):**
```
Reproducibility check (2x 1-epoch runs, configs/baseline.yaml, seed=42):
  Run A: train_loss=0.7457  train_iou=0.1363  val_iou=0.2802
  Run B: train_loss=0.7457  train_iou=0.1363  val_iou=0.2829
  -> train metrics bit-exact at epoch 1; val metrics differ at the ~1e-3 level (GPU nondeterminism)

Baseline restore retrain (30 epochs, configs/baseline.yaml, seed=42) vs. original Phase 4 run:
  Original:  best=epoch29  test: IoU=0.6250 Dice=0.7692 Precision=0.7681 Recall=0.7703 Acc=0.9765
  Restored:  best=epoch30  test: IoU=0.6234 Dice=0.7680 Precision=0.7333 Recall=0.8062 Acc=0.9752
  -> aggregate metrics (IoU/Dice/F1/Accuracy) very close (within ~0.002-0.001); precision/recall
     shifted by a real, non-trivial amount (~0.03-0.04) due to a different epoch being selected best
```

**KNOWN ISSUES:**
- **Training on this GPU setup is not bit-exact reproducible even with a fixed seed**, due to
  non-deterministic cuDNN convolution algorithm selection (PyTorch does not enable
  `torch.use_deterministic_algorithms(True)`/`cudnn.deterministic=True` here, since those
  typically cost meaningful training speed — a tradeoff, not an oversight, but one now explicitly
  documented rather than silently assumed away). Practical consequence: re-running an experiment
  with the same config reproduces the same *overall training trajectory and ballpark performance*,
  but not the same numbers to the decimal place, and can select a different "best" checkpoint with
  a real precision/recall tradeoff shift. Anyone reproducing this project's results should expect
  numbers close to, but not necessarily identical to, those recorded here.
- No LR scheduler is used (`scheduler: none`, explicit in config) — both models trained with a
  constant learning rate for the full 30 epochs.
- This incident is a useful, concrete illustration of why Rule 6 ("smallest reasonable change",
  and by extension, "use a distinct experiment name for throwaway/smoke-test runs") matters —
  noted for future phases: **use a `*_smoketest` or similar distinct `experiment_name` for any
  future ad hoc verification runs, never the name of a real tracked experiment.**

**NEXT PHASE:**
- PHASE 7 — Evaluation & Visualization: the rigorous final evaluation writeup (`docs/EVALUATION.md`)
  comparing baseline vs. Siamese with the qualitative grids and training curves already produced in
  Phases 4-6, plus explicit discussion of the class-imbalance-aware metric choices and the
  reproducibility caveat surfaced in this phase.

---

## PHASE 5 — Siamese U-Net

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `models/siamese_encoder.py`: `SiameseEncoder`, structurally identical to the baseline U-Net's
  encoder (reuses `DoubleConv`/`Down` from `models/unet.py`, Rule 6). Weight sharing between the
  before/after branches is achieved by construction — a single `SiameseEncoder` instance is called
  twice in `SiameseUNet.forward` (once per image), so both passes use literally the same
  parameters, not just the same architecture.
- `models/siamese_unet.py`: `SiameseUNet` — runs the shared encoder on before/after separately,
  explicitly compares the resulting feature maps at every scale (bottleneck + 4 skip levels) via
  `compare_features()`, and decodes with the same reused `Up` block class from `models/unet.py`.
  Three configurable comparison modes implemented per `PROJECT_CONTEXT.md`: `diff` (absolute
  difference), `concat` (channel concat), `diff_concat` (both). `comparison_channels()` derives
  the correct decoder skip-channel counts for whichever mode is selected.
- `configs/siamese.yaml`: primary trained configuration, `comparison_mode: diff_concat` (chosen
  after benchmarking all 3 modes for parameter count / VRAM / speed — see Results below — since
  all three fit comfortably and `diff_concat` carries the richest signal).
- `docs/ARCHITECTURE.md` updated: Siamese section rewritten from "planned" to describe the actual
  implementation, with real measured parameter counts and VRAM numbers per comparison mode.
- 13 new pytest tests (`tests/test_siamese_unet.py`): shared-encoder determinism, comparison-mode
  channel-count correctness, `diff` symmetry vs. `concat`/`diff_concat` asymmetry (the model can
  actually tell before from after when the mode preserves order), forward-pass output shape and
  backward+optimizer-step parameter-change checks for all 3 comparison modes, invalid-mode error
  handling, and a check that encoder parameters appear exactly once (genuinely shared, not
  duplicated per branch).
- `src/training/train.py` cleaned up: removed the Phase-4-era `try/except` placeholder import for
  `SiameseUNet` now that it's implemented, wired directly.

**FILES CREATED:**
- `models/siamese_encoder.py`, `models/siamese_unet.py`
- `configs/siamese.yaml`
- `tests/test_siamese_unet.py`
- `outputs/checkpoints/siamese_unet_diff_concat/{best,last}.pt` (gitignored)
- `outputs/experiments/siamese_unet_diff_concat/{history.csv,history.json}` (gitignored)
- `outputs/metrics/siamese_unet_diff_concat_test_metrics.json` (gitignored)
- `outputs/visualizations/siamese_unet_diff_concat_test_predictions.png` (gitignored)

**FILES MODIFIED:**
- `src/training/train.py` (direct `SiameseUNet` import/wiring, placeholder removed)
- `docs/ARCHITECTURE.md` (Siamese section rewritten to describe the real implementation)

**COMMANDS EXECUTED:**
- `pytest tests/test_siamese_unet.py -v` (13 new tests)
- Benchmark script (inline `python -c`): instantiated `SiameseUNet` for all 3 comparison modes at
  `base_channels=32`, ran 3 real forward+backward+optimizer-step iterations at batch_size=8,
  image_size=256 on the GPU, measured parameter count, time/iter, and peak VRAM for each
- `pytest tests/ -v` (full suite, 40 tests, after adding Siamese tests)
- `python -m src.training.train --config configs/siamese.yaml --epochs 2` (smoke test)
- `python -m src.training.train --config configs/siamese.yaml` (real 30-epoch run, GPU, background)
- `python -m src.evaluation.evaluate --config configs/siamese.yaml --checkpoint outputs/checkpoints/siamese_unet_diff_concat/best.pt`

**TESTS:**
- `pytest tests/test_siamese_unet.py`: 13/13 passed.
- `pytest tests/` (full suite): 40/40 passed (27 from Phase 4 + 13 new).
- Real GPU benchmark (not estimated) of all 3 comparison modes, `base_channels=32`,
  batch_size=8, image_size=256:
  ```
  diff          params=7,763,041   time/iter=0.311s  peak_vram=2.21GB
  concat        params=10,709,345  time/iter=0.218s  peak_vram=2.38GB
  diff_concat   params=14,704,225  time/iter=0.254s  peak_vram=2.79GB
  ```
  All three comfortably within the 6GB GPU's budget (Phase 1) — `diff_concat` selected as the
  primary trained configuration.
- 2-epoch smoke test of the full `train.py` pipeline with `configs/siamese.yaml` before the full
  run — confirmed working end-to-end (checkpointing, logging, correct experiment directory).
- Full 30-epoch training run on the real LEVIR-CD train/val split (same data/split as the Phase 4
  baseline, for a fair comparison) — training loss and validation IoU improved across training
  without divergence or NaN losses.
- Real, measured test-set evaluation on the held-out LEVIR-CD test split (same 128 samples used
  for the Phase 4 baseline evaluation), using the best checkpoint (selected by validation IoU).
- Manually inspected the qualitative prediction grid (same 6 test sample indices as the Phase 4
  grid, for direct visual comparison) — predictions are visually cleaner than the baseline's, with
  fewer false-positive (yellow) blobs in the diff panel.

**RESULTS (actual, measured — full report: `outputs/metrics/siamese_unet_diff_concat_test_metrics.json`):**
```
Model: SiameseUNet (base_channels=32, comparison_mode=diff_concat), 14,704,225 parameters
Training: 30 epochs, Adam lr=1e-4, BCE+Dice loss, batch_size=8, image_size=256, seed=42
Best checkpoint: epoch 29 (selected by validation IoU) — same protocol as the Phase 4 baseline

Validation (at best checkpoint, epoch 29):
  IoU=0.6567  Dice=0.7928  Precision=0.8005  Recall=0.7852  Accuracy=0.9828

TEST SET (held out, 128 samples, real measured results):
  IoU=0.6442  Dice=0.7836  Precision=0.7982  Recall=0.7695  F1=0.7836  Accuracy=0.9784
```

### Baseline vs. Siamese — real, apples-to-apples comparison (identical data split, config
protocol, training budget, checkpoint-selection rule)

**NOTE (updated in Phase 6):** the baseline numbers below were revised after Phase 6's
reproducibility testing accidentally overwrote and required retraining the baseline checkpoint —
see the Phase 4 entry's "Checkpoint overwrite" known issue and the Phase 6 entry. The table below
reflects the current, real, restored-baseline-run numbers.

| Metric | Baseline U-Net (Phase 4) | Siamese U-Net (Phase 5) | Δ |
|---|---|---|---|
| IoU | 0.6234 | **0.6442** | +0.0208 |
| Dice | 0.7680 | **0.7836** | +0.0156 |
| Precision | 0.7333 | **0.7982** | +0.0649 |
| Recall | **0.8062** | 0.7695 | −0.0367 |
| F1 | 0.7680 | **0.7836** | +0.0156 |
| Accuracy | 0.9752 | **0.9784** | +0.0032 |

The Siamese U-Net outperforms the baseline on IoU, Dice, Precision, F1, and Accuracy, but the
baseline has meaningfully *higher* recall (+0.0367) — a real precision/recall tradeoff, not a tie
(an earlier draft of this comparison, before the Phase 6 retrain, had shown recall as
approximately tied; that was an artifact of the specific checkpoint selected on that run, not a
stable property — see Phase 6's reproducibility finding). Net effect: Siamese still wins on the
aggregate metrics that matter most given this task's class imbalance (IoU, Dice, F1), by predicting
more conservatively (fewer false positives, at the cost of some false negatives) — plausible given
it can explicitly compare same-location before/after features at every decoder scale, rather than
only seeing their raw channel-wise concatenation as the baseline does. This is a real, measured
result on one training run each (and, per Phase 6, not perfectly bit-reproducible even with a
fixed seed on this GPU) — not a hyperparameter-tuned or multi-seed comparison. Phase 8 is where a
more rigorous multi-experiment, multi-seed comparison (if warranted) would be conducted.

**KNOWN ISSUES:**
- Same caveat as Phase 4: validation IoU was still trending upward through epoch 29-30 — this is
  a fixed 30-epoch budget for a controlled comparison with the baseline, not a claim of full
  convergence for either model.
- Only the `diff_concat` comparison mode was trained to completion; `diff` and `concat` are fully
  implemented and tested (forward/backward/optimizer-step verified) but not yet trained end-to-end
  — that comparison is deferred to Phase 8 (research experiments) since it's an ablation, not a
  requirement for the Phase 5 milestone (having a working, evaluated Siamese architecture).
- Single run, single seed — no variance estimate across multiple seeds.

**NEXT PHASE:**
- PHASE 6 — Training & Experiment Pipeline: the config-driven training/checkpointing/logging
  infrastructure was actually already built and used in Phases 4-5 (`src/training/`,
  `configs/*.yaml`, `outputs/experiments/*/history.csv`); Phase 6 work here is primarily about
  training-curve visualization (epoch vs. loss/IoU/Dice plots from the existing history.csv files)
  and formalizing experiment tracking, ahead of Phase 7's full rigorous evaluation writeup.

---

## PHASE 4 — Baseline U-Net

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `models/unet.py`: standard 4-stage U-Net (`DoubleConv`/`Down`/`Up`, skip connections,
  `base_channels=32` for VRAM tractability) plus `BaselineChangeUNet`, which concatenates the
  before/after images into a fused 6-channel input (the Phase 4 "single fused input" baseline
  design, deliberately simpler than the Phase 5 Siamese architecture).
- `models/losses.py`: `DiceLoss`, `BCEDiceLoss` (used for the baseline), `get_loss()` factory;
  all operate on raw logits for numerical stability.
- `src/evaluation/metrics.py`: `MetricAccumulator` — accumulates TP/FP/FN/TN across a whole
  split (not per-batch averaging, which would be wrong under class imbalance) and computes
  IoU/Dice/Precision/Recall/F1/Accuracy.
- `src/training/`: `checkpoint.py` (save/load with epoch/metrics/config bundled in),
  `logger.py` (per-epoch CSV+JSON history), `validate.py` (shared val/test evaluation loop),
  `trainer.py` (`Trainer` class: train-one-epoch, fit loop, best-checkpoint-by-val-IoU
  selection — written model-agnostically so it's reused unchanged for Phase 5's Siamese model),
  `train.py` (YAML-config-driven entry point with `extends:` base-config merging, seeding, device
  auto-detection).
- `configs/config.yaml` (shared defaults) and `configs/baseline.yaml` (baseline experiment).
- `src/evaluation/evaluate.py`: test-set evaluation (real IoU/Dice/Precision/Recall/F1/Accuracy)
  plus a 6-column qualitative prediction grid (before/after/GT/prediction/overlay/diff with
  FP=yellow, FN=blue, TP=green coding).
- `docs/ARCHITECTURE.md`: documents the implemented baseline (explicitly marks the Siamese
  architecture as "Planned, not yet implemented" rather than describing it as built).
- 11 new pytest tests (`tests/test_model.py`, `tests/test_metrics.py`): forward pass and output
  shape, backward pass + optimizer step actually changes parameters, Dice loss correctness on a
  synthetic perfect prediction, metric correctness on synthetic perfect/wrong/accumulated cases.

**FILES CREATED:**
- `models/__init__.py`, `models/unet.py`, `models/losses.py`
- `src/evaluation/__init__.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`
- `src/training/__init__.py`, `src/training/checkpoint.py`, `src/training/logger.py`,
  `src/training/validate.py`, `src/training/trainer.py`, `src/training/train.py`
- `configs/config.yaml`, `configs/baseline.yaml`
- `docs/ARCHITECTURE.md`
- `tests/test_model.py`, `tests/test_metrics.py`
- `outputs/checkpoints/baseline_unet/{best,last}.pt` (gitignored)
- `outputs/experiments/baseline_unet/{history.csv,history.json}` (gitignored)
- `outputs/metrics/baseline_unet_test_metrics.json` (gitignored)
- `outputs/visualizations/baseline_unet_test_predictions.png` (gitignored)

**FILES MODIFIED:**
- None outside the above

**COMMANDS EXECUTED:**
- `pytest tests/ -v` (27 tests, before training)
- `python -m src.training.train --config configs/baseline.yaml --epochs 2` (smoke test)
- `python -m src.training.train --config configs/baseline.yaml --epochs 1` (timing check: ~50s/epoch)
- `python -m src.training.train --config configs/baseline.yaml` (real 30-epoch run, GPU, background)
- `python -m src.evaluation.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_unet/best.pt`

**TESTS:**
- `pytest tests/`: 27/27 passed (16 from Phase 3 + 11 new: `UNet`/`BaselineChangeUNet` forward
  shape, backward+optimizer-step parameter-change check, `DiceLoss`/`BCEDiceLoss` correctness,
  `get_loss` factory, `MetricAccumulator`/`confusion_counts`/`logits_to_binary_preds` correctness
  on synthetic perfect/inverted/multi-batch cases).
- 2-epoch smoke test of the full `train.py` pipeline (config loading, dataloaders, model,
  optimizer, checkpointing, CSV/JSON logging) before committing to a full run — confirmed working
  end-to-end.
- Full 30-epoch training run on the real LEVIR-CD train/val split (GPU, RTX 4050) — training loss
  and validation IoU both improved monotonically-ish across training (see history.csv), converging
  without divergence or NaN losses.
- Real, measured test-set evaluation (`src/evaluation/evaluate.py`) on the held-out LEVIR-CD test
  split (128 samples, never seen during training/validation) using the best checkpoint (selected by
  validation IoU, not by test performance — no test-set leakage into model selection).
- Manually inspected the qualitative prediction grid (6 test samples: before/after/GT/prediction/
  overlay/diff) — predictions visually track ground truth closely (majority true-positive/green in
  the diff panel), including correctly near-empty predictions on genuinely no-change scenes.

**RESULTS (actual, measured — full report: `outputs/metrics/baseline_unet_test_metrics.json`.
NOTE: this run was later accidentally overwritten and retrained; see the "Checkpoint overwrite"
known-issue note below and the Phase 6 entry — these are the final, currently-valid numbers):**
```
Model: BaselineChangeUNet (base_channels=32), 7,763,905 parameters
Training: 30 epochs, Adam lr=1e-4, BCE+Dice loss, batch_size=8, image_size=256, seed=42
Best checkpoint: epoch 30 (selected by validation IoU)

Validation (at best checkpoint, epoch 30):
  IoU=0.6152  Dice=0.7618  Precision=0.7227  Recall=0.8053  Accuracy=0.9789

TEST SET (held out, 128 samples, real measured results):
  IoU=0.6234  Dice=0.7680  Precision=0.7333  Recall=0.8062  F1=0.7680  Accuracy=0.9752
```
Training time: ~50 seconds/epoch on the RTX 4050 Laptop GPU (Phase 1), ~25 minutes total for 30
epochs. Inference/evaluation on the 128-sample test set: well under 1 minute.

**KNOWN ISSUES:**
- Training loss and validation IoU were still improving at epoch 30 — the model had likely not
  fully converged within the 30-epoch budget. This is an intentionally scoped baseline run, not a
  claim of a fully tuned/converged model; Phase 8 (research experiments) is where tuning/longer
  training is considered if it provides measurable value.
- The baseline's simple channel-concatenation design (vs. a true Siamese shared encoder) is a
  known architectural limitation, not a bug — it is the documented reason Phase 5 exists.
- No hyperparameter search was performed; `configs/baseline.yaml` values are reasonable defaults,
  not the result of tuning.
- **Checkpoint overwrite + non-bit-exact reproducibility (found in Phase 6):** during Phase 6
  reproducibility testing, a smoke-test run accidentally overwrote this experiment's checkpoint
  and history. It was restored by re-running `configs/baseline.yaml` with the identical config and
  seed. The restored run's epoch-1 train metrics matched the original run exactly, but numerical
  drift appeared from epoch 2 onward (GPU convolution algorithms are not bit-deterministic by
  default in PyTorch), compounding over training. The restored run selected epoch 30 as best
  (vs. epoch 29 originally) with a meaningfully different precision/recall balance (0.7333/0.8062
  vs. the original run's 0.7681/0.7703) despite similar aggregate IoU/Dice/F1/Accuracy. The numbers
  above are the current, real, restored-run results — see the Phase 6 entry for the full
  reproducibility finding and what it means for `DEVELOPMENT_RULES.md` Rule 7.

**NEXT PHASE:**
- PHASE 5 — Siamese U-Net: implement `models/siamese_encoder.py` and `models/siamese_unet.py`
  with a shared encoder between before/after branches and configurable feature comparison
  (absolute difference / concatenation / both), verify forward/backward/optimizer-step, then train
  and evaluate it with the same `Trainer`/`evaluate.py` pipeline built in Phase 4 for a real,
  apples-to-apples comparison against this baseline.

---

## PHASE 3 — Data Preprocessing & DataLoader

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/data/preprocessing.py`: image/mask loading (OpenCV), mask binarization (explicit
  threshold at 127, addressing the non-binary mask values found during Phase 2 verification),
  resize (linear for images, nearest-neighbor for masks to preserve binary values), [0,1]
  normalization, and numpy->CHW-float-tensor conversion.
- `src/data/augmentation.py`: `PairedAugmentor` applying identical spatial transforms (horizontal
  flip, vertical flip, 90-degree rotation, random-crop+resize "scale" jitter) to the before image,
  after image, and mask from one shared random draw per call, plus independent brightness jitter
  applied to the before/after images only (never the mask).
- `src/data/dataset.py`: `LEVIRCDDataset(Dataset)` — verifies A/B/label pairing at construction
  time (raises `ValueError` listing unpaired files, or `FileNotFoundError` if the split directory
  is missing), returns `(before, after, mask)` tensors per the pipeline above.
- `src/data/dataloader.py`: `get_dataloader()` wrapping `LEVIRCDDataset` in a `torch.utils.data.
  DataLoader`, defaulting to shuffle+augment on for `train` and off for `val`/`test`.
- `scripts/verify_dataloader.py`: end-to-end Phase 3 verification against the real LEVIR-CD data
  (train/val/test `dataset[0]`, an augmented sample, DataLoader batch shapes, and an augmentation
  visual sanity check grid).
- `tests/` pytest suite (16 tests) using small synthetic images (per the rule to avoid requiring
  the full multi-GB dataset for unit tests): `test_preprocessing.py`, `test_augmentation.py`
  (including a same-pixel-tracking test proving spatial augmentation moves the mask in lockstep
  with the images), `test_dataset.py` (pairing-error and missing-split-dir error paths, correct
  tensor shapes/dtypes, DataLoader batch shapes and train/val augment-default behavior).

**FILES CREATED:**
- `src/__init__.py`, `src/data/__init__.py`
- `src/data/preprocessing.py`, `src/data/augmentation.py`, `src/data/dataset.py`,
  `src/data/dataloader.py`
- `scripts/verify_dataloader.py`
- `tests/conftest.py`, `tests/test_preprocessing.py`, `tests/test_augmentation.py`,
  `tests/test_dataset.py`
- `outputs/visualizations/augmentation_samples.png` (gitignored)

**FILES MODIFIED:**
- `requirements.txt` (added `pytest`, `huggingface_hub`)

**COMMANDS EXECUTED:**
- `pip install pytest`
- `venv/Scripts/python.exe scripts/verify_dataloader.py`
- `venv/Scripts/python.exe -m pytest tests/ -v`

**TESTS:**
- `scripts/verify_dataloader.py` against the real dataset: asserts tensor shapes
  `(3,256,256)`/`(3,256,256)`/`(1,256,256)`, dtype `float32`, image value range `[0,1]`, and mask
  values `⊆ {0,1}` for `train[0]`, `val[0]`, `test[0]`, and an augmented `train[0]`; asserts
  DataLoader batch shapes for `train` (batch_size=4, drop_last=True, 111 batches) and `val`
  (batch_size=4, 16 batches); renders and visually inspected an augmentation sample grid.
- `pytest tests/` — 16/16 passed, covering mask binarization thresholding, normalization range,
  resize dimension/interpolation-mode correctness, tensor shape/dtype conversion, paired-spatial-
  augmentation correctness (marker-pixel tracking), zero-probability no-op augmentation, shape
  preservation under scale jitter, dataset length/pairing/error-path correctness, and DataLoader
  batch shapes and augment defaults.

**RESULTS (actual, measured):**
```
Real-data verification (scripts/verify_dataloader.py): ALL CHECKS PASSED
  train[0]: before/after=(3,256,256) float32 in [0,1], mask=(1,256,256) float32 {0,1} - OK
  val[0], test[0]: same shape/dtype/range checks - OK
  train[0] with augmentation: same shape/dtype/range checks - OK
  DataLoader(train, batch_size=4): 111 batches, shapes (4,3,256,256)/(4,3,256,256)/(4,1,256,256) - OK
  DataLoader(val, batch_size=4): 16 batches - OK
pytest tests/: 16 passed, 0 failed, 2.17s
```
Augmentation visual check confirmed by direct image inspection: flips/rotations/scale-jitter move
the mask's changed region in exact lockstep with the before/after images across 5 sampled
variants — no misalignment observed.

**KNOWN ISSUES:**
- `num_workers=0` used by default in `get_dataloader` (single-process loading) — adequate at
  current dataset size/throughput; can be revisited in Phase 6 if training-time data loading
  becomes a bottleneck.
- Default `image_size=256` (not the native 1024) chosen for compute/VRAM tractability on the
  6 GB GPU identified in Phase 1; this is a deliberate, documented choice, not a limitation
  discovered by accident.

**NEXT PHASE:**
- PHASE 4 — Baseline U-Net: implement `models/unet.py`, `models/losses.py` (BCE, Dice, BCE+Dice),
  evaluation metrics (IoU/Dice/Precision/Recall/F1/Accuracy), train the baseline on a
  single-fused-input formulation, and report real test-set metrics (never fabricated).

---

## PHASE 2 — Dataset Acquisition & Understanding

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Identified the official LEVIR-CD source (https://justchenhao.github.io/LEVIR/, Chen & Shi 2020)
  and confirmed its official download links (Google Drive / Baidu Drive) are manual-only —
  interactive cloud-drive folders with no stable unauthenticated direct-download URL, unsuitable
  for reliable unattended scripted download (Google Drive in particular gates large files behind
  a browser confirmation click).
- Per the stop-condition/acquisition rules in `DEVELOPMENT_RULES.md` / `PROJECT_CONTEXT.md`,
  located and used a documented mirror instead: Hugging Face dataset repo
  `satellite-image-deep-learning/LEVIR-CD`, which exposes `train.zip` / `val.zip` / `test.zip`
  as plain, individually addressable files downloadable via `huggingface_hub` with no
  authentication. This is documented as a mirror, not the original source, in `docs/DATASET.md`.
- Downloaded all three official splits (`val.zip` first to validate structure/legitimacy before
  committing to the larger downloads, then `test.zip`, then `train.zip`), verified each archive's
  byte size against the Hub-reported size before extraction (exact match on all three), and
  extracted them to `data/raw/levir_cd/{train,val,test}/{A,B,label}/`.
- Wrote `scripts/verify_dataset.py`: checks A/B/label pairing (missing/orphan detection),
  opens and decodes every image (corruption check), verifies dimensions/channels/format
  consistency, computes the changed-vs-unchanged pixel distribution, and renders a 4-column
  (before/after/mask/overlay) sample visualization grid per split.
- Ran full verification on all three splits; wrote `docs/DATASET.md` with the real measured
  results (source, mirror justification, structure, per-split counts, class-imbalance numbers,
  a mask-binarization caveat, and split/leakage methodology).
- Deleted the now-redundant zip archives after verified extraction (2.3 GB reclaimed); dataset
  now lives only as extracted files under `data/raw/levir_cd/` (gitignored — not committed).

**FILES CREATED:**
- `scripts/verify_dataset.py`
- `docs/DATASET.md`
- `outputs/metrics/dataset_verification.json` (full machine-readable verification report, gitignored)
- `outputs/visualizations/dataset_samples_{train,val,test}.png` (gitignored)
- `data/raw/levir_cd/{train,val,test}/{A,B,label}/*.png` (dataset itself, gitignored)

**FILES MODIFIED:**
- None outside the above (README/DEVELOPMENT_LOG updates are part of this same entry)

**COMMANDS EXECUTED:**
- `pip install huggingface_hub`
- `HfApi().list_repo_files(...)` / `repo_info(..., files_metadata=True)` — file listing and size check
- `hf_hub_download(...)` for `val.zip`, `test.zip`, `train.zip` (run sequentially, each in the
  background due to slow connection speed — observed ~250 KB/s)
- `python -c "zipfile...extractall(...)"` for each split
- `venv/Scripts/python.exe scripts/verify_dataset.py --root data/raw/levir_cd --splits val test train`
- `rm -rf data/raw/levir_cd_zips`

**TESTS:**
- Downloaded file byte sizes compared against Hub-reported sizes — exact match for all three
  (`val.zip` 246,152,048; `test.zip` 496,305,323; `train.zip` 1,721,956,862 bytes).
- `scripts/verify_dataset.py` run against all three splits: pairing check, corruption check
  (every file actually opened/decoded), dimension/channel/format consistency check, pixel-value
  range check, pixel-distribution computation.
- Manually inspected the generated sample-grid visualizations for all three splits — change masks
  visibly align with genuine new construction between before/after images in every sampled case.

**RESULTS (actual, measured — full report in `outputs/metrics/dataset_verification.json`):**
```
train: 445 samples, 445/445/445 paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.04589
val:    64 samples,  64/64/64  paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.04197
test:  128 samples, 128/128/128 paired, 0 corrupted, all 1024x1024, changed-pixel fraction 0.05094
total: 637 samples — exact match to the official LEVIR-CD dataset size (637 pairs)
```
Mask pixel values are not perfectly binary in all files (some contain anti-aliased edge values
like 156/254 alongside 0/255) — flagged for explicit thresholding in Phase 3 preprocessing.

**KNOWN ISSUES:**
- Download speed from the Hugging Face mirror was slow (~250 KB/s observed, unauthenticated
  requests); downloads took roughly 15 min (val), 30 min (test), and just over an hour (train).
  Not a correctness issue, just a environment/network characteristic worth knowing for future
  re-downloads.
- The dataset's official split is used as-is (no re-splitting performed), per the leakage-
  prevention methodology documented in `docs/DATASET.md`.

**NEXT PHASE:**
- PHASE 3 — Data Preprocessing & DataLoader: implement `src/data/dataset.py`,
  `src/data/preprocessing.py` (including mask binarization per the caveat above),
  `src/data/augmentation.py` (paired spatial augmentation applied identically to A/B/label), and
  `src/data/dataloader.py`; verify `dataset[0]` returns correctly-shaped/typed tensors and that
  DataLoader batching works before any model code is written.

---

## PHASE 1 — Environment & Project Setup

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Created a Python virtual environment at `venv/` (Python 3.13.0).
- Installed PyTorch 2.6.0 + torchvision 0.21.0 with CUDA 12.4 wheels
  (`--index-url https://download.pytorch.org/whl/cu124`), matched to the detected NVIDIA GPU.
- Installed core stack: NumPy, Pandas, OpenCV, Pillow, scikit-learn, PyYAML, Matplotlib, Plotly,
  python-dotenv, tqdm.
- Installed Streamlit (dashboard dependency, installed now for environment completeness; the
  dashboard itself is not built until Phase 10 per the project rules).
- Wrote `scripts/check_env.py`, an environment diagnostic script that reports Python/PyTorch/
  torchvision versions, CUDA availability, GPU name/VRAM, and runs a real tensor matmul on both
  the selected device and an explicit CPU fallback.
- Pinned exact installed versions into `requirements.txt` (replacing the Phase 0 unpinned
  placeholder), with a documented CPU-vs-CUDA install note.

**FILES CREATED:**
- `venv/` (virtual environment, gitignored)
- `scripts/check_env.py`
- `outputs/env_freeze.txt` (full `pip freeze` output, local diagnostic artifact, gitignored)

**FILES MODIFIED:**
- `requirements.txt` (unpinned placeholder -> exact pinned versions from the verified environment)

**COMMANDS EXECUTED:**
- `python -m venv venv`
- `venv/Scripts/python.exe -m pip install --upgrade pip`
- `venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- `venv/Scripts/python.exe -m pip install numpy pandas opencv-python Pillow scikit-learn PyYAML matplotlib plotly python-dotenv tqdm`
- `venv/Scripts/python.exe -m pip install streamlit`
- `nvidia-smi` (GPU presence check, prior to choosing the CUDA wheel)
- `venv/Scripts/python.exe scripts/check_env.py`
- `venv/Scripts/python.exe -m pip freeze > outputs/env_freeze.txt`

**TESTS:**
- Ran `scripts/check_env.py` end-to-end; it imports torch/torchvision/numpy/cv2/PIL, checks
  `torch.cuda.is_available()`, enumerates GPU properties, runs `matmul` on the selected device,
  and runs a second `matmul` explicitly pinned to CPU to confirm CPU fallback works independent
  of GPU presence.
- Verified `streamlit` imports and reports its version.

**RESULTS (actual, measured — see `scripts/check_env.py` output):**
```
Python:      3.13.0
PyTorch:     2.6.0+cu124
Torchvision: 0.21.0+cu124
CUDA available: True
CUDA version (torch build): 12.4
GPU count: 1
GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB total VRAM)
Selected device: cuda
NumPy: 2.5.2
OpenCV: 5.0.0
Pillow: 12.3.0
Tensor op test on cuda: matmul(4x4, 4x4) -> shape (4, 4), sum=12.9776
Tensor op test on cpu (explicit fallback check): shape (4, 4), sum=12.2680
streamlit: 1.62.0
```

**KNOWN ISSUES:**
- GPU has 6 GB total VRAM, of which ~2.5 GB was already in use by an unrelated local process
  (an Ollama server) at the time of this check, leaving roughly 3.5 GB free. This constrains
  batch size / image resolution for later training phases and will be revisited in Phase 6
  when real batch sizes are chosen.
- `outputs/env_freeze.txt` and `venv/` are local artifacts (gitignored), not committed.

**NEXT PHASE:**
- PHASE 2 — Dataset Acquisition & Understanding: obtain the LEVIR-CD benchmark dataset, verify
  structure/pairing/dimensions/channels/label values, check for missing/corrupted files, compute
  changed-vs-unchanged pixel distribution, document splitting methodology (leakage prevention),
  and write `docs/DATASET.md`.

---

## PHASE 0 — Project Initialization

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Full directory scaffold created (`data/`, `models/`, `src/*`, `dashboard/*`, `notebooks/`,
  `configs/`, `outputs/*`, `tests/`, `docs/`, `scripts/`).
- Core documentation created: `PROJECT_CONTEXT.md`, `DEVELOPMENT_RULES.md`, `DEVELOPMENT_LOG.md`,
  `README.md`.
- `.gitignore` and `requirements.txt` (placeholder, to be pinned in Phase 1) created.
- `.env.example` created.
- Git repository initialized with an initial commit.

**FILES CREATED:**
- `PROJECT_CONTEXT.md`
- `DEVELOPMENT_RULES.md`
- `DEVELOPMENT_LOG.md`
- `README.md`
- `.gitignore`
- `requirements.txt`
- `.env.example`
- Directory scaffold (empty dirs hold `.gitkeep` where needed)

**FILES MODIFIED:**
- None (fresh repository)

**COMMANDS EXECUTED:**
- `mkdir -p` for the full directory tree
- `git init`
- `git add` / `git commit` (see git log for exact commit)

**TESTS:**
- Verified directory tree with `find . -maxdepth 3 -type d`.
- Verified `git status` / `git log` show a clean initial commit.
- Verified Python 3.13.0 is available on PATH (`python --version`).

**RESULTS:**
- Repository structure matches the target layout in `PROJECT_CONTEXT.md`.
- Git initialized and working.
- Python 3.13.0 detected. PyTorch/CUDA/GPU/etc. NOT YET MEASURED — deferred to Phase 1.

**KNOWN ISSUES:**
- None.

**NEXT PHASE:**
- PHASE 1 — Environment & Project Setup: create virtual environment, install pinned dependencies,
  verify PyTorch/torchvision/CUDA, produce environment diagnostic report.
