# LIMITATIONS.md

A consolidated, honest account of this project's limitations, drawn from what was actually
discovered while building and evaluating it (Phases 2-11) — not a generic disclaimer list. Every
item below traces to a real, documented finding elsewhere in this repository, cited inline.

## Dataset limitations
- **LEVIR-CD is binary building-change only.** 637 image pairs, 20 regions in and around several
  Texas cities (Austin, Lakeway, Bee Cave, Buda, Kyle, Manor, Pflugerville, Dripping Springs,
  etc.), 2002-2018 (`docs/DATASET.md`). The model has never seen roads, vegetation, water, or any
  other change type as a labeled category — it cannot classify change type, only detect binary
  building change (`PROJECT_CONTEXT.md` capability-honesty rule; stated explicitly in the
  dashboard's Capabilities table, `dashboard/app.py`).
- **Strong class imbalance.** Only ~4.2-5.1% of pixels are labeled "changed" across the LEVIR-CD
  splits (`docs/DATASET.md`). This is why accuracy alone is never reported as a standalone metric
  anywhere in this project (`docs/EVALUATION.md`).
- **Mask binarization was required.** Raw LEVIR-CD mask files are not perfectly binary — some
  contain anti-aliased edge values like 156 or 254 alongside 0/255 (`docs/DATASET.md` "Mask
  binarization note"). Preprocessing explicitly thresholds at 127 (`src/data/preprocessing.py`);
  this threshold is a reasonable but unvalidated choice, not derived from a labeled sensitivity
  study.
- **Single geographic region.** All training data is from central Texas suburbs. Performance on
  architecturally or geographically different regions (different building materials, roof types,
  urban density, terrain) is untested.

## Model / training limitations
- **Not trained to full convergence.** Both the baseline and every Siamese variant were trained
  for a fixed 30-epoch budget; validation IoU was still trending upward at epoch 30 in most runs
  (`DEVELOPMENT_LOG.md` Phase 4/5/8 "Known issues"). Reported metrics reflect this specific,
  intentionally-scoped training budget, not each architecture's ceiling performance.
- **No formal hyperparameter search.** All experiments share one hyperparameter recipe (Adam,
  lr=1e-4, BCE+Dice loss, batch size 8, no LR scheduler) specifically so the *architecture*
  comparison would be apples-to-apples (`docs/EXPERIMENTS.md` "Why a formal hyperparameter search
  was not run"). No architecture's individually-optimal hyperparameters have been found.
- **Single run, single seed per experiment.** No confidence intervals or variance estimates exist
  for any reported metric.
- **GPU training is not bit-exact reproducible even with a fixed seed.** Re-running the identical
  baseline config produced measurably different results — same seed, same code, different
  outcome — due to non-deterministic cuDNN convolution algorithms
  (`DEVELOPMENT_LOG.md` Phase 6, the checkpoint-overwrite incident and its follow-up finding).
  `torch.use_deterministic_algorithms(True)` was deliberately not enabled, trading exact
  reproducibility for training speed — a documented tradeoff, not an oversight.
- **Fixed 0.5 decision threshold.** No ROC/precision-recall operating-point sweep has been
  performed (`docs/EVALUATION.md` "Limitations of this evaluation"); the precision/recall
  tradeoffs reported between models are only characterized at this one threshold.
- **Only one Siamese comparison mode (`diff_concat`) was combined with attention.** `diff` and
  `concat` alone were trained and evaluated without attention (`docs/EXPERIMENTS.md`); the
  attention+diff or attention+concat combinations were not tried.
- **A Transformer-based architecture was deliberately not built**, and no evidence exists either
  way about whether it would outperform the CNN-based Siamese U-Net on this task — the decision
  to defer it was based on dataset-size/scope reasoning (`docs/EXPERIMENTS.md`), not on a negative
  result from actually trying it.

## Evaluation limitations
- **Benchmark-only quantitative results.** IoU/Dice/Precision/Recall/F1/Accuracy
  (`docs/EVALUATION.md`) are measured exclusively on the LEVIR-CD held-out test split — a curated,
  pre-registered, single-sensor (0.5 m/pixel Google Earth composite imagery) dataset. These
  numbers say nothing directly about performance on other imagery.
- **False positives and false negatives are real and quantified, not hidden.** E.g. the baseline
  model's test-set confusion matrix: 125,241 false positives, 82,806 false negatives out of
  8,388,608 total pixels (`docs/EVALUATION.md`). The best model (Siamese+Attention) still has a
  documented qualitative false-positive failure case on one no-change test scene
  (`docs/EXPERIMENTS.md` "Qualitative note").

## Domain shift: benchmark vs. real-world (the largest, most consequential gap found)
- **Resolution: Sentinel-2 real-world imagery is 20x coarser than the training data** (10 m/pixel
  vs. LEVIR-CD's 0.5 m/pixel — `docs/REAL_WORLD_DEMO.md`). At 10 m/pixel, a typical house occupies
  a fraction of a pixel to a few pixels; the fine building-outline detail the model learned to
  recognize does not exist at that resolution.
- **Different sensor and radiometric processing.** Sentinel-2's multispectral instrument and
  atmospheric-correction pipeline differ from the Google Earth composite imagery LEVIR-CD is built
  from; no attempt has been made to quantify or correct for this.
- **No ground truth exists for the real-world demonstration**, so no accuracy metric can be or was
  computed for Sentinel-2 predictions — only a single qualitative example was examined
  (`docs/REAL_WORLD_DEMO.md`), which showed a genuine correct detection alongside several
  unverifiable predicted regions. This is one anecdotal data point, not a systematic real-world
  evaluation, and must not be generalized from.
- **Registration.** LEVIR-CD's before/after pairs are pre-registered by the dataset authors; no
  independent re-registration or alignment-quality check was performed for the Sentinel-2 imagery
  used in Phase 11 beyond Sentinel-2's own standard georeferencing.

## Multi-class change detection (Phase 19) — not implemented, dataset unobtainable, not a design choice
Phase 19 required a *properly labeled* multi-class change detection dataset (LEVIR-CD's binary
labels were never going to be misused to fabricate classes — see `PROJECT_CONTEXT.md`/
`DEVELOPMENT_RULES.md`). Three real candidate datasets were investigated on 2026-08-25:

| Dataset | What it offers | Why it could not be used here |
|---|---|---|
| SECOND (captain-whu, 4662 pairs, 512x512, 6 land-cover classes) | Exactly the right shape: small, patch-based, genuinely multi-class | Only distributed via Google Drive — no direct/resumable HTTP source; not reliably fetchable headless |
| HRSCD-Clean (`EPFL-ECEO/HRSCD_clean` on Hugging Face) | Real semantic segmentation masks alongside change masks, MIT-licensed | Distributed as a **single 60.25 GB zip** (confirmed via the Hub API's `files_metadata`) with no per-sample sharding |
| xView2/xBD (building damage, 4 severity classes — genuinely multi-class and arguably the best conceptual fit) | Real, widely-used, 4-class damage severity labels | Smallest mirror found (`Devansh25/xview2`) is still 3.85 GB; larger mirrors run 11-33 GB |

**The blocking constraint is this session's measured network throughput, not dataset scarcity.**
A direct, repeated HTTP range-request benchmark against Hugging Face's own CDN (not a slow mirror)
measured **~18.3 KB/s** — consistent with the ~10-22 KB/s observed installing `pyproj`/`pyogrio` in
Phase 18 (`DEVELOPMENT_LOG.md` Phase 18 "Known limitations"). At that rate the smallest viable
candidate (3.85 GB) would take **~2.4 days**; HRSCD-Clean's 60 GB zip would take **~38 days**. This
is not fixable by choosing a different dataset — any dataset of the size multi-class semantic
change detection genuinely requires (hundreds of MB to tens of GB of paired high-resolution
imagery + per-pixel semantic masks) hits the same wall.

**Decision (per the explicit autonomous-execution rule to document a genuine limitation and
continue rather than invent a solution):** Phase 19 is not implemented. LEVIR-CD's binary building-
change labels are not repurposed to fake multi-class output. The existing binary Siamese U-Net +
Attention model (`docs/EVALUATION.md`, `README.md` Results) is unaffected and remains this
project's primary result. If network conditions improve in a future session, `Devansh25/xview2`
(3.85 GB, real 4-class building-damage-severity labels) is the recommended first candidate to
revisit, since its class scheme (no damage/minor/major/destroyed) is a more natural fit for a
change-detection-derived pipeline than land-cover classes borrowed from a scene-segmentation task.

## Real-world input validation (Phase 22) — real, computed checks, deliberately not more than that
`src/realworld/validation.py` adds three genuinely computed diagnostic checks for arbitrary
(non-LEVIR-CD) before/after uploads, surfaced in the dashboard and `scripts/real_world_demo.py`:
- **Registration-offset estimate** (`cv2.phaseCorrelate`) — a real phase-correlation shift
  estimate between before/after, flagged above 3px. This is a diagnostic only: it does **not**
  align/correct the images, and phase correlation itself can be fooled by real (not just
  misregistration-caused) large-scale change between the two images.
- **Resolution-plausibility check** — flags imagery whose pixel size is ≥5x coarser than LEVIR-
  CD's 0.5 m/pixel training resolution (only computable when a real pixel size is known, e.g. from
  georeferenced raster metadata — never guessed for a plain upload).
- **Cloud/bright-region heuristic** — flags images where >5% of pixels are simultaneously bright
  and low-saturation. **Explicitly not a validated cloud detector**: no labeled cloud-mask data
  exists anywhere in this project to validate it against; it will miss thin/translucent cloud and
  can false-positive on bright rooftops, sand, snow, or plain overexposure.
- The exact required disclaimer — `"Model trained on LEVIR-CD imagery. Performance on this
  imagery has not been independently validated."` — is displayed wherever real-world/non-LEVIR-CD
  imagery is processed (dashboard upload flow, `scripts/real_world_demo.py`).
- **None of this constitutes accuracy validation.** These checks describe the *input*, not
  whether the resulting prediction is correct — no ground truth exists for arbitrary real-world
  uploads, so no accuracy claim is or can be made for them, benchmark or otherwise.

## Not implemented (explicitly, per Rule 4 — never implied as working)
- Actual image co-registration/alignment *correction* (Phase 22 adds an offset *estimate* only,
  not a correction) or a validated (non-heuristic) cloud/shadow detector — see above.
- Change-type classification (building vs. road vs. vegetation vs. water, etc.) — no such labels
  exist in the training data.
- Physical-area estimation for non-LEVIR-CD imagery — `src/analysis/area.py`'s pixel-size
  assumption is LEVIR-CD-specific and is deliberately not applied to Sentinel-2 predictions
  (`docs/REAL_WORLD_DEMO.md`, `scripts/real_world_demo.py`).
- Multi-seed statistical significance testing between architectures.
- ROC/precision-recall curve analysis at multiple decision thresholds.

## Scientific principle this project tries to uphold (and where it's been tested)
`PROJECT_CONTEXT.md` requires distinguishing **actual geographical change** from **apparent visual
difference** (clouds, shadows, seasonal vegetation, lighting, misalignment, sensor/resolution
differences). Two concrete, real data points exist for this:
- Phase 9's `test_99.png`: a genuinely no-change LEVIR-CD test scene with a strong seasonal
  lighting/vegetation difference (before: brown/dry; after: green/lush) — the model correctly
  predicted almost no change (3 tiny regions, 0.05% of the tile). One reassuring anecdotal example,
  not a systematic robustness evaluation.
- Phase 11's real-world demonstration: the model was not obviously confused by 5 years of
  real-world seasonal/atmospheric variation between two winter Sentinel-2 scenes, but — again —
  this is one example, not a validated claim of robustness.

Neither of these constitutes proof that the model reliably makes this distinction in general; both
are documented as single supporting data points, not conclusions.
