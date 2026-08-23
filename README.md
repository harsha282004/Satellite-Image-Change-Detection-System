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

```bash
python -m src.training.train --config configs/baseline.yaml   # baseline
python -m src.training.train --config configs/siamese.yaml    # Siamese (primary)
```

Trains for the epoch count in the config (`--epochs N` overrides it), logging per-epoch
train/val loss and IoU/Dice/Precision/Recall/F1/Accuracy to
`outputs/experiments/<experiment_name>/history.csv`, and saving `best.pt` (selected by validation
IoU)/`last.pt` checkpoints to `outputs/checkpoints/<experiment_name>/`.

## Evaluation

**Implemented (Phase 4-7).** Full rigorous writeup — methodology, class-imbalance-aware metric
rationale, quantitative baseline-vs-Siamese comparison, qualitative results, training curves, and
explicitly stated evaluation limitations — in [`docs/EVALUATION.md`](docs/EVALUATION.md).

```bash
python -m src.evaluation.evaluate --config configs/baseline.yaml \
    --checkpoint outputs/checkpoints/baseline_unet/best.pt
```

Reports real, measured IoU/Dice/Precision/Recall/F1/Accuracy on the held-out test split and saves
a qualitative prediction grid (before/after/ground-truth/prediction/overlay/diff) to
`outputs/visualizations/`. See Results below for the actual baseline numbers.

## Inference

`NOT YET IMPLEMENTED` — Phase 9 (`src/inference/predict.py`).

## Dashboard

`NOT YET IMPLEMENTED` — Phase 10 (`dashboard/app.py`).

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
the best result across all 5 experiments run to date (baseline, 3 Siamese comparison-mode
variants, and this attention variant). A genuinely interesting ablation finding: the `diff`-only
comparison mode alone (IoU=0.5569) actually *underperforms* the simple baseline — raw before/after
feature context turns out to matter more than an explicit difference signal. Full results,
interpretation, and the documented reasoning for deferring a Transformer variant and a formal
hyperparameter search (dataset size, no evidence of a CNN failure mode, scope) are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Experiments

**Implemented (Phase 8).** 5 real experiments compared on identical data/training protocol:
baseline U-Net, Siamese U-Net (`diff`/`concat`/`diff_concat` comparison modes), and Siamese +
Attention (`diff_concat`, the winner). Full comparison table, ablation interpretation, and the
documented justification for deferring a Transformer variant and hyperparameter search:
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) (written progressively; finalized in Phase 12).
Key categories to be documented: dataset limitations, registration error, resolution/sensor
differences, cloud interference, seasonal/lighting change, false positives/negatives, class
imbalance, domain shift, geographic generalization, benchmark-vs-real-world gap.

## Future Scope

Additional datasets (WHU-CD, DSIFN-CD), a Transformer-based variant (deferred in Phase 8 —
justification in `docs/EXPERIMENTS.md`: small dataset, no evidence of a CNN failure mode it would
fix), formal hyperparameter search, multi-seed variance estimates, multi-class change typing (only
if a suitable labeled dataset is identified).

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
