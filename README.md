# Satellite Change Detection

Deep Learning-Based Multi-Temporal Satellite Image Change Detection and Analysis System.

> **Status: Phases 0-12 complete.** Dataset acquired and verified, baseline and Siamese U-Net
> (+ Attention) models trained and evaluated on a real held-out benchmark test set, 5 research
> experiments compared, change-region quantification and a Streamlit dashboard built on the real
> trained model, and a real-world Sentinel-2 demonstration run (explicitly distinguished from the
> benchmark evaluation — see `docs/REAL_WORLD_DEMO.md`). Every number in this README is real,
> measured output — see `DEVELOPMENT_LOG.md` for the full phase-by-phase history and
> `docs/LIMITATIONS.md` for what this project does not do. Anything not yet measured is still
> marked `NOT YET MEASURED` per `DEVELOPMENT_RULES.md` Rule 3, rather than estimated.

## Project Overview

This is an academic Deep Learning / Computer Vision project. Given two satellite images of the
same geographic area taken at different times (a BEFORE image and an AFTER image), the system
detects and quantifies genuine geographical change — new/demolished structures, urban expansion,
and similar — while explicitly guarding against apparent-but-not-real differences caused by
clouds, shadows, seasonal vegetation, lighting, or misalignment (see `docs/LIMITATIONS.md` once
written).

The Streamlit dashboard is a demonstration layer on top of the trained model, not the core
contribution. The core contribution is the Siamese U-Net change-detection model and its
experimental evaluation. Full details: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Problem Statement

Manual comparison of multi-temporal satellite imagery to identify geographic change is slow and
error-prone. This project builds a deep-learning pipeline that takes a registered before/after
image pair and produces a pixel-level binary change mask, plus region-level statistics
(count, area, percentage changed).

## Objectives

1. Build a reproducible benchmark pipeline (dataset -> preprocessing -> model -> training ->
   evaluation) around a Siamese U-Net.
2. Achieve and document real, measured IoU/Dice/Precision/Recall/F1 on a held-out benchmark
   test set (LEVIR-CD).
3. Extract and quantify individual change regions from the predicted mask.
4. Provide an interactive dashboard for demonstration, backed by the real trained model.
5. Attempt a real-world demonstration (Sentinel-2) with explicitly documented domain-gap caveats.

## System Architecture

See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md#architecture) for the full baseline U-Net and
Siamese U-Net architecture diagrams, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
(to be written in Phase 4/5) for the as-implemented version.

## Technology Stack

- **Core:** Python, PyTorch, torchvision, NumPy, Pandas, OpenCV, Pillow, scikit-learn, PyYAML
- **Visualization:** Matplotlib, Plotly
- **Geospatial (introduced when required):** Rasterio, GeoPandas, Shapely, Folium
- **Dashboard:** Streamlit

## Dataset

Primary: **LEVIR-CD** benchmark building change-detection dataset (Chen & Shi, 2020) — 637
before/after 1024×1024 image pairs with binary building-change masks, official split: 445 train /
64 val / 128 test. Acquired via a documented Hugging Face mirror (the official Google Drive/Baidu
links are manual-download-only) and fully verified: 100% A/B/label pairing, zero corrupted files,
consistent dimensions across all 637 samples. See [`docs/DATASET.md`](docs/DATASET.md) for full
source documentation, verification methodology, real measured results, and the split/leakage
methodology.

## Installation

```bash
git clone <repo-url>
cd satellite-change-detection
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

(Environment verification and exact pinned versions are finalized in Phase 1 — see
`DEVELOPMENT_LOG.md`.)

## Environment Setup

Implemented (Phase 1). After creating and activating the virtual environment (see Installation
above), verify the environment with:

```bash
python scripts/check_env.py
```

This reports Python/PyTorch/torchvision versions, CUDA availability, GPU name/VRAM, and runs a
real tensor operation on both the selected device and an explicit CPU fallback. Measured on the
development machine (see `DEVELOPMENT_LOG.md` Phase 1 for the full report): PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, CUDA available, GPU = NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
The `requirements.txt` install alone pulls the CPU build of PyTorch; see the note at the top of
`requirements.txt` for the CUDA-wheel install command used to get the GPU build above.

## Training

Baseline U-Net (Phase 4) and Siamese U-Net (Phase 5, primary architecture): **both implemented.**
Since Phase 13, training also supports optional early stopping and an LR scheduler:

```bash
python -m src.training.train --config configs/baseline.yaml            # baseline, fixed 30 epochs
python -m src.training.train --config configs/siamese.yaml             # Siamese (primary), fixed 30 epochs
python -m src.training.train --config configs/siamese_attention_e100.yaml  # best model: up to 100 epochs, early stopping + LR scheduler
```

Trains for the epoch count in the config (`--epochs N` overrides it; with early stopping enabled,
this is a *maximum* budget, not a guarantee — training can stop sooner), logging per-epoch
train/val loss, IoU/Dice/Precision/Recall/F1/Accuracy, and current learning rate to
`outputs/experiments/<experiment_name>/history.csv`, and saving `best.pt` (selected by validation
IoU, never the test set)/`last.pt` checkpoints to `outputs/checkpoints/<experiment_name>/`. Full
methodology, hyperparameter rationale, the Phase 13 early-stopping/LR-scheduler experiments, and
the real reproducibility caveats discovered while building this (GPU training is not bit-exact
reproducible even with a fixed seed): [`docs/TRAINING.md`](docs/TRAINING.md).

## Evaluation

**Implemented (Phase 4-7, extended Phase 15).** Full rigorous writeup — methodology,
class-imbalance-aware metric rationale, quantitative baseline-vs-Siamese comparison, qualitative
results, training curves, explicitly stated evaluation limitations, prediction-probability maps,
validation-only threshold optimization, and controlled robustness testing (brightness/contrast/
noise/misregistration) — in [`docs/EVALUATION.md`](docs/EVALUATION.md). Notable Phase 15 finding:
the model is essentially insensitive to decision threshold (0.30-0.70), but shows a real,
measured sensitivity to reduced contrast/brightness and small misregistration (~0.10-0.12 mean
IoU degradation, worst single case 0.42).

```bash
python -m src.evaluation.evaluate --config configs/baseline.yaml \
    --checkpoint outputs/checkpoints/baseline_unet/best.pt
```

Reports real, measured IoU/Dice/Precision/Recall/F1/Accuracy on the held-out test split and saves
a qualitative prediction grid (before/after/ground-truth/prediction/overlay/diff) to
`outputs/visualizations/`. See Results below for the actual baseline numbers.

## Inference & Change Analysis

**Implemented (Phase 9, extended Phase 16).** `src/inference/predict.py` (`Predictor` class) loads
a config + checkpoint once and predicts a binary change mask (and, since Phase 15, a raw
prediction-probability map) from a before/after image pair. `src/analysis/{regions,area,
statistics}.py` extract connected-component **"Detected Change Regions"** (never labeled with a
semantic category like "Building" — LEVIR-CD's binary labels give no basis for that, see
`docs/DATASET.md`) with full geometry per region (bounding box, centroid, width/height, perimeter,
aspect ratio, change density) plus mean/max prediction probability, and aggregate statistics
(region count, largest/smallest/average region size, total/percent changed pixels) — physical area
in m²/hectares only when a pixel size is explicitly provided, never assumed (LEVIR-CD's documented
0.5 m/pixel, adjusted for the model's resized input resolution: see
`src/analysis/area.py::levir_cd_effective_pixel_size`). `scripts/export_regions.py` saves real
per-region CSV/JSON data (now including severity, see below) and region-ID-labeled overlays to
`outputs/regions/`. **Severity (Phase 17):** `src/analysis/severity.py` computes a transparent,
fully documented 0-100 score per region from measurable model outputs (region size, mean
prediction probability, shape density, relative size) — **explicitly an analytical score, not
ground truth or a physical damage assessment**; real measured result on 258 detected regions:
3 Low / 215 Moderate / 40 High / 0 Very High.

```bash
python scripts/analyze_predictions.py --config configs/siamese_attention_e100.yaml \
    --checkpoint outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt
```

Runs the current best model (Phase 13 Experiment C — see Results) on real test images, saves a region-count/area report
(`outputs/metrics/region_analysis_demo.json`) and per-sample visualizations with region bounding
boxes (`outputs/visualizations/region_analysis/`). Real example measured output: a dense-
subdivision test tile yielded 54 detected regions, ~15% of the tile changed, 3.92 hectares total
changed area; a genuinely no-change tile (despite a strong seasonal lighting/vegetation difference
between before/after — the kind of apparent-but-not-real difference `PROJECT_CONTEXT.md` warns
about) correctly yielded only 3 tiny regions covering 0.05% of the tile.

## Dashboard

**Implemented (Phase 10).**

```bash
python -m streamlit run dashboard/app.py
```

Upload a before/after image pair, select any of the 5 trained models (sidebar shows that model's
real, measured benchmark metrics — not simulated), and click "Detect Changes" to run real
inference: predicted mask, overlay, region count, percent/area changed, and a full per-region
table. Verified end-to-end in a real browser (Playwright): model switching, image upload, live
inference producing results matching `scripts/analyze_predictions.py`'s output exactly for the
same test image, and graceful error handling for an invalid uploaded file. The "Capabilities"
table on the page states plainly what is and isn't implemented (e.g. no change-type
classification, no verified real-world-imagery support) — nothing is faked or implied.

## Results

Real measured test-set results (128 held-out LEVIR-CD test samples, never used in training or
checkpoint selection), same data split, training budget (30 epochs), and checkpoint-selection
rule (best validation IoU) for both models:

| Metric | Baseline U-Net (Phase 4) | Siamese U-Net (Phase 5, primary architecture) |
|--------|-------|-------|
| IoU | 0.6234 | **0.6442** |
| Dice | 0.7680 | **0.7836** |
| Precision | 0.7333 | **0.7982** |
| Recall | **0.8062** | 0.7695 |
| F1 | 0.7680 | **0.7836** |
| Accuracy | 0.9752 | **0.9784** |

The Siamese U-Net (shared encoder, `diff_concat` feature comparison, 14.7M parameters) outperforms
the simple channel-concatenation baseline (7.8M parameters) on IoU/Dice/Precision/F1/Accuracy, but
the baseline has meaningfully higher recall — a real precision/recall tradeoff, not a tie. Both
trained 30 epochs, Adam (lr=1e-4), BCE+Dice loss, batch size 8, image size 256, single NVIDIA RTX
4050 Laptop GPU (~25 min each). Full details, training curves, and known-issue notes (neither
model was trained to full convergence within the fixed 30-epoch budget, and training on this GPU
is not perfectly bit-reproducible even with a fixed seed — see `DEVELOPMENT_LOG.md` Phase 6) in
`DEVELOPMENT_LOG.md` (Phase 4-6 entries) and
`outputs/metrics/{baseline_unet,siamese_unet_diff_concat}_test_metrics.json`.

**Phase 8 research experiments found an even better model.** Adding Attention-U-Net-style
skip-connection gates to the `diff_concat` Siamese architecture improved *every* metric (not just
a tradeoff): **test IoU=0.6560, Dice=0.7922, Precision=0.8018, Recall=0.7829, Accuracy=0.9791** —
the best result across all 5 architectures compared under an identical, controlled 30-epoch
training budget (baseline, 3 Siamese comparison-mode variants, and this attention variant). A
genuinely interesting ablation finding: the `diff`-only comparison mode alone (IoU=0.5569) actually
*underperforms* the simple baseline — raw before/after feature context turns out to matter more
than an explicit difference signal. Full results, interpretation, and the documented reasoning for
deferring a Transformer variant and a formal hyperparameter search (dataset size, no evidence of a
CNN failure mode, scope) are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**Phase 13 found the winning architecture from Phase 8 was itself undertrained.** The 30-epoch
budget used for the controlled architecture comparison above was deliberately equal across all 5
models — but that same equal budget turned out to be too short for the winner. Adding early
stopping (patience=10 on validation IoU) and a `ReduceLROnPlateau` LR scheduler and giving the
Siamese+Attention architecture a larger epoch ceiling, tested experimentally rather than assumed,
produced a real, substantial improvement:

| Experiment | Max epochs | Actual epochs | Best epoch | Early stopped | Test IoU | Test Dice | Test F1 |
|---|---|---|---|---|---|---|---|
| A — original (Phase 8) | 30 | 30 | 26 | No (fixed budget) | 0.6560 | 0.7922 | 0.7922 |
| B — longer training | 60 | 60 | 60 | No (still improving) | 0.7031 | 0.8257 | 0.8257 |
| **C — longer + early stop** | 100 | 78 | 68 | **Yes** (patience=10) | **0.7123** | **0.8320** | **0.8320** |

**`siamese_unet_diff_concat_attention_e100` (Experiment C) is now the best model in this project**,
test IoU=0.7123 — a +0.0563 absolute IoU improvement over the Phase 8 result, from training
strategy alone (identical architecture, data, optimizer, and seed as Experiment A). Full
methodology, the LR-scheduler-triggered improvement bursts visible in the training curves, and
honest limitations (this is one seed, and Experiment B — the uncapped-budget run — was still
improving when its 60-epoch ceiling was hit, so an even longer budget might do better still, untested)
are in [`docs/TRAINING.md`](docs/TRAINING.md) and `DEVELOPMENT_LOG.md` (Phase 13).

## Experiments

**Implemented (Phase 8).** 5 real experiments compared on identical data/training protocol:
baseline U-Net, Siamese U-Net (`diff`/`concat`/`diff_concat` comparison modes), and Siamese +
Attention (`diff_concat`, the winner). Full comparison table and ablation interpretation:
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**Phase 20 built the Transformer variant Phase 8 deferred, and measured it for real.**
`models/transformer_change.py` — a genuine self-attention encoder (`nn.TransformerEncoder`) Siamese
architecture — trained under the identical 30-epoch protocol as the 5 CNN models above, for a fair
comparison. **Result, reported honestly: it loses.** Test IoU=0.3575, well below even the weakest
CNN variant (`diff`, IoU=0.5569) — consistent with Vision Transformers needing more data or
pretraining than this project's 445 training pairs / from-scratch training provide. It does have
the fewest parameters (4.05M) and fastest inference (3.42 ms/pair) of all 6 models, a real but
non-decisive tradeoff. Full result table and interpretation:
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (Phase 20 section).

## Real-World Demonstration

**Implemented (Phase 11) — a demonstration, explicitly not a validated evaluation.** Ran the best
model on real, independently-sourced Sentinel-2 imagery (Earth Search STAC API, AWS-hosted, no
authentication required) for a Pflugerville, TX suburb, 2019-12-06 vs. 2024-12-19:

```bash
python scripts/real_world_demo.py
```

**Critical caveat, stated plainly:** Sentinel-2 is 10 m/pixel — **20x coarser** than the 0.5 m/pixel
LEVIR-CD imagery the model was trained on. The model correctly detected the scene's most obvious
real change (a large new building complex, visually confirmed present in the after-image and
absent in the before-image), but also predicted several smaller regions that could not be
independently verified as real vs. domain-gap artifacts — **no ground truth exists for this
real-world pair, so no IoU/Dice/accuracy is or can be reported here.** Full method, results,
resolution-gap analysis, and an honest discussion of what this does and does not demonstrate:
[`docs/REAL_WORLD_DEMO.md`](docs/REAL_WORLD_DEMO.md). Do not read this section's qualitative
result as equivalent in validity to the benchmark numbers in "Results" above.

## Geospatial Change Intelligence

**Implemented (Phase 18).** Extends the real-world Sentinel-2 demonstration above from
image-space analysis (pixel coordinates only, no CRS — true of every LEVIR-CD PNG) to genuine
geospatial analysis: `src/geospatial/raster.py::fetch_georeferenced_crop()` fetches a real GeoTIFF
that preserves its native CRS/affine transform (read with `rasterio`), and a hard guard
(`has_georeference()`) refuses to run geospatial conversion on any raster that isn't genuinely
georeferenced — no invented coordinates. `src/geospatial/polygons.py` converts detected pixel
regions into real geographic polygons and real-world area in m²/hectares (`polygon_area_m2`, which
raises rather than computing area in a geographic/degree CRS) and reprojects to WGS84 for GeoJSON
export. Outputs: `regions.geojson`, `regions.csv`, `regions.gpkg` (GeoPackage), and an interactive
Folium map (`region_map.html`) with per-region popups.

```bash
python scripts/geospatial_analysis.py
```

Real measured result on the same Pflugerville, TX pair as Phase 11: raster 583x561 px at
10.0 m/pixel (EPSG:32614), **6 regions detected, 30.89 hectares total detected-change area**
(computed from the raster's actual UTM projection). Same caveats as the Phase 11 demonstration
apply in full — no ground truth exists for this pair, so this is a real, unforced measurement, not
a validated accuracy figure. Full results table and pipeline details:
[`docs/EVALUATION.md`](docs/EVALUATION.md) (Phase 18 section).

## Multi-Temporal Analysis

**Implemented (Phase 21).** Extends the geospatial demonstration above from a single before/after
pair to an ordered sequence of real Sentinel-2 acquisitions. `src/temporal/sequence.py` selects N
dates spread evenly across a real, cloud-filtered STAC search (never fabricated dates) and pairs
them into adjacent intervals, each analyzed by the existing two-image pipeline **completely
independently**. **No causal or tracking claims are made:** a region detected in one interval is
never asserted to be the same physical change as a region in another interval — the model has no
cross-image tracking mechanism.

```bash
python scripts/multitemporal_analysis.py
```

Real measured result: searched 2017-2024 for the same Pflugerville, TX area as Phase 11/18, found
**385 real cloud-filtered candidate dates**, selected 5 spread across that span, and computed 4
independent intervals: 1-3 regions and 2.86-7.20 ha detected-change area per interval (full
table and the "not a trend line" caveat: [`docs/EVALUATION.md`](docs/EVALUATION.md), Phase 21
section). The existing two-image mode (dashboard default) is completely unmodified.

## Real-World Pipeline Hardening

**Implemented (Phase 22).** `src/realworld/validation.py` adds real, computed input-validation
checks for arbitrary (non-LEVIR-CD) before/after uploads — surfaced in the dashboard's upload flow
and `scripts/real_world_demo.py`: a registration-offset estimate (`cv2.phaseCorrelate`, flags
>3px estimated shift — a diagnostic, not a correction), a resolution-plausibility check (flags
imagery ≥5x coarser than LEVIR-CD's 0.5 m/pixel training resolution), and an explicitly-heuristic
cloud/bright-region screen (not a validated cloud detector — see
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)). The exact required disclaimer — **"Model trained on
LEVIR-CD imagery. Performance on this imagery has not been independently validated."** — is
displayed wherever real-world imagery is processed. None of this validates prediction accuracy;
it only surfaces honest signals about the input. The offline/local-file upload path is unchanged
and still works without network access.

## Limitations

**Finalized (Phase 12).** Full account in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): dataset
scope (binary building-change only, single Texas region, class imbalance), model/training
caveats (not trained to full convergence, no hyperparameter search, single-seed, GPU
non-determinism), evaluation scope (benchmark-only metrics, documented false positives/negatives),
and — the most consequential — the benchmark-vs-real-world domain gap (Sentinel-2 is 20x coarser
resolution than the training data; no ground truth exists for real-world predictions). Every item
traces to a specific finding from Phases 2-11, not a generic disclaimer.

## Future Scope

Additional datasets (WHU-CD, DSIFN-CD), formal hyperparameter search, multi-seed variance
estimates. **Multi-class change typing (Phase 19) was investigated, not skipped:** three real
candidate datasets (SECOND, HRSCD-Clean, xView2/xBD) were evaluated and none was reliably
obtainable at this session's measured ~18 KB/s network throughput (smallest viable candidate,
`Devansh25/xview2` at 3.85 GB, would take ~2.4 days; full details and the recommended first
candidate for a future session in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)) — LEVIR-CD's binary
labels were never repurposed to fake it.

## Project Structure

```
satellite-change-detection/
├── data/                # raw/processed/train/val/test (gitignored beyond structure)
├── models/              # unet.py, siamese_encoder.py, siamese_unet.py, attention.py, losses.py
├── src/
│   ├── data/            # dataset.py, preprocessing.py, augmentation.py, dataloader.py
│   ├── training/        # train.py, validate.py, trainer.py, checkpoint.py, logger.py
│   ├── evaluation/      # metrics.py, evaluate.py, benchmark.py
│   ├── inference/       # predict.py
│   ├── analysis/        # regions.py, area.py, statistics.py
│   ├── geospatial/      # raster.py, polygons.py, maps.py
│   └── visualization/   # visualize.py, overlays.py, plots.py
├── dashboard/           # app.py, components/, utils/
├── notebooks/           # exploration and analysis notebooks
├── configs/             # config.yaml, baseline.yaml, siamese.yaml
├── outputs/             # checkpoints/, predictions/, visualizations/, metrics/, experiments/
├── tests/
├── docs/                # ARCHITECTURE.md, DATASET.md, TRAINING.md, EVALUATION.md, EXPERIMENTS.md, REAL_WORLD_DEMO.md, LIMITATIONS.md
├── PROJECT_CONTEXT.md
├── DEVELOPMENT_RULES.md
├── DEVELOPMENT_LOG.md
├── README.md
├── requirements.txt
└── .env.example
```

## Development Process

This project is built phase-by-phase per `DEVELOPMENT_RULES.md`, with every phase verified by
actually running the code and recording real results in `DEVELOPMENT_LOG.md` before moving on.
No fabricated metrics, no unsupported capability claims.
