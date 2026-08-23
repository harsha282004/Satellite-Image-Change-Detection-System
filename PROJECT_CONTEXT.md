# PROJECT_CONTEXT.md

## Project Name
Deep Learning-Based Multi-Temporal Satellite Image Change Detection and Analysis System
(repository: `satellite-change-detection`)

## Nature of the Project
This is a **Deep Learning / Computer Vision academic project**, not primarily a web-development
project. The core contribution is the deep-learning-based multi-temporal satellite image change
detection model and its experimental evaluation. The Streamlit dashboard is a demonstration/interface
layer built on top of a working, evaluated model — it must never be built ahead of the model.

## Objective
Given two satellite images of approximately the same geographic region taken at different dates
(a BEFORE image and an AFTER image), produce a pixel-level binary change mask, extract and quantify
changed regions, and visualize the result — first as a rigorously evaluated benchmark model, later
demonstrated on real-world imagery with explicitly documented caveats.

```
BEFORE IMAGE + AFTER IMAGE
        -> IMAGE PREPROCESSING
        -> DEEP LEARNING MODEL (Siamese U-Net)
        -> CHANGE DETECTION (binary mask)
        -> CHANGE SEGMENTATION
        -> CHANGE REGION EXTRACTION
        -> CHANGE ANALYSIS
        -> CHANGE QUANTIFICATION
        -> VISUALIZATION
        -> INTERACTIVE DASHBOARD
```

The initial, scientifically reliable target is:

```
Before Image + After Image -> Siamese U-Net -> Pixel-level Binary Change Mask
```

Change-region analysis, quantification, visualization, dashboard, and real-world demonstration are
added only after this baseline works and is evaluated.

### Capability honesty
Do not claim the system classifies change types (buildings / roads / vegetation / water / etc.)
unless the dataset and trained model actually support that. Distinguish explicitly between
**Implemented**, **Experimentally supported**, and **Planned / Future scope** everywhere in the docs.

## Architecture

### Baseline (Phase 4)
Simple U-Net operating on a single fused/stacked input, used as a reference point before the
Siamese architecture.

### Primary architecture (Phase 5) — Siamese U-Net
```
BEFORE -> Shared Encoder -> Feature A ---\
                                          Feature Fusion -> U-Net Decoder -> Binary Change Mask
AFTER  -> Shared Encoder -> Feature B ---/
```
- Encoder weights are shared between both branches (true Siamese, not two independent encoders).
- Feature comparison is configurable: absolute difference, concatenation, or difference+concatenation.
- Decoder follows standard U-Net upsampling with skip connections.

### Later research variants (Phase 8, only if justified)
- Siamese U-Net + Attention
- Transformer-based change detection
Implemented only when hardware, dataset size, and time make the experiment practical, and only
when it provides measurable value over the baseline/Siamese U-Net.

## Datasets

### Primary
**LEVIR-CD** — benchmark building change-detection dataset. Paired before/after image tiles with
binary building-change ground-truth masks.

### Secondary (optional, only if primary is validated first)
- WHU-CD
- DSIFN-CD

### Real-world demonstration (Phase 11)
**Sentinel-2** imagery, used only after benchmark evaluation is complete, with clearly documented
resolution/sensor/domain differences from the benchmark training data. Real-world results are never
presented as having the same validity as benchmark results unless experimentally demonstrated.

### Dataset acquisition rules
- Never scrape arbitrary satellite images from the internet as training data.
- Document the official/research source of any dataset used.
- If a Kaggle mirror is used, document that it is a mirror and identify the underlying original dataset.
- Verify structure, pairing, dimensions, channels, and label values before use.
- Prevent geographic leakage between train/val/test splits; document the split methodology.
- If a dataset requires manual download (registration/license), STOP and tell the user exactly
  what to download and where to place it — never fabricate a download.

## Preprocessing
Resize, normalization, tensor conversion, mask binarization, and paired spatial augmentation
(flip, rotation, crop, scale, brightness) applied **identically** to before image, after image,
and mask together.

## Evaluation Metrics
IoU, Dice, Precision, Recall, F1, Accuracy. Accuracy alone is never sufficient because change-detection
datasets are typically strongly class-imbalanced (most pixels are "no change").

## Training Strategy
Config-driven (YAML), reproducible: fixed seeds, checkpointing, best-model selection by validation
metric (not just loss), full metric/curve logging per experiment, documented hyperparameters.

## Visualization
Before / After / Ground Truth / Prediction / Overlay / Difference panels; training curves
(loss, IoU, Dice vs. epoch); qualitative result grids (multiple examples, not one cherry-picked case).

## Dashboard (Phase 10, built last)
Streamlit app: upload before/after images -> run actual trained model inference -> show mask,
overlay, changed area, change percentage, region count, region statistics, model info. No simulated
or hardcoded metrics — if a feature isn't implemented, it is marked unavailable, not faked.

## Development Phases
```
PHASE 0  Project Initialization
PHASE 1  Environment & Project Setup
PHASE 2  Dataset Acquisition & Understanding
PHASE 3  Data Preprocessing & DataLoader
PHASE 4  Baseline U-Net
PHASE 5  Siamese U-Net
PHASE 6  Training Pipeline
PHASE 7  Evaluation & Visualization
PHASE 8  Model Improvement & Research Experiments
PHASE 9  Change Analysis & Quantification
PHASE 10 Streamlit Dashboard
PHASE 11 Real-World Satellite Demonstration
PHASE 12 Final Integration & Documentation
```
Phases are implemented sequentially, each verified (tests run, real output inspected, errors fixed,
result recorded in DEVELOPMENT_LOG.md) before moving to the next. The baseline is never skipped, and
advanced architectures are never started before the Siamese U-Net baseline works.

## Known Limitations To Be Documented Throughout
Dataset limitations, image registration errors, satellite resolution, cloud interference, seasonal
change, lighting differences, false positives/negatives, class imbalance, domain shift, limited
geographic generalization, benchmark-vs-real-world performance gap.

## Scientific Principle
The system must distinguish **actual geographical change** from **apparent visual difference**
caused by clouds, shadows, seasonal vegetation, lighting, misalignment, sensor/resolution
differences. This distinction is discussed explicitly in `docs/LIMITATIONS.md`.

## Technology Stack
- Core: Python, PyTorch, torchvision, NumPy, Pandas, OpenCV, Pillow, scikit-learn, PyYAML
- Visualization: Matplotlib, Plotly
- Geospatial (introduced only when required): Rasterio, GeoPandas, Shapely, Folium
- Dashboard: Streamlit
- PyTorch is the committed DL framework; no switch to TensorFlow/Keras without strong documented reason.

## Source of Truth
This document (`PROJECT_CONTEXT.md`), together with `DEVELOPMENT_RULES.md`, is the source of truth
for the project's objective and architecture. Fundamental objectives/architecture are not changed
silently — any deviation is called out explicitly to the user before proceeding.
