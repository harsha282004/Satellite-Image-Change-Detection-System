# Satellite Change Detection

Deep Learning-Based Multi-Temporal Satellite Image Change Detection and Analysis System.

> **Status: Phase 0 (Project Initialization) complete.** No model has been trained yet. Any
> numbers that appear in this README are placeholders and are explicitly marked
> `NOT YET MEASURED` until real experiments are run — see `DEVELOPMENT_RULES.md` (Rule 3).

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

Primary: **LEVIR-CD** benchmark building change-detection dataset. See
[`docs/DATASET.md`](docs/DATASET.md) (written in Phase 2) for source, structure, verification, and
split methodology once acquired. Dataset is **not yet acquired** as of Phase 0.

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

`NOT YET IMPLEMENTED` — Phase 1.

## Training

`NOT YET IMPLEMENTED` — Phase 4 (baseline U-Net), Phase 5 (Siamese U-Net), Phase 6 (training
pipeline/configs).

## Evaluation

`NOT YET IMPLEMENTED` — Phase 7. Metrics reported will be IoU, Dice, Precision, Recall, F1, and
Accuracy on a held-out test set, with real measured numbers only.

## Inference

`NOT YET IMPLEMENTED` — Phase 9 (`src/inference/predict.py`).

## Dashboard

`NOT YET IMPLEMENTED` — Phase 10 (`dashboard/app.py`).

## Results

`NOT YET MEASURED`. No model has been trained. This section will be filled in with real metrics
after Phase 7.

## Experiments

`NOT YET IMPLEMENTED` — Phase 8. Will compare U-Net vs. Siamese U-Net vs. justified variants using
real measured metrics only (see `docs/EXPERIMENTS.md`, written then).

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) (written progressively; finalized in Phase 12).
Key categories to be documented: dataset limitations, registration error, resolution/sensor
differences, cloud interference, seasonal/lighting change, false positives/negatives, class
imbalance, domain shift, geographic generalization, benchmark-vs-real-world gap.

## Future Scope

Additional datasets (WHU-CD, DSIFN-CD), attention/Transformer variants (only if justified by
measured results), multi-class change typing (only if a suitable labeled dataset is identified).

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
├── docs/                # ARCHITECTURE.md, DATASET.md, TRAINING.md, EVALUATION.md, EXPERIMENTS.md, LIMITATIONS.md
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
