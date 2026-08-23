# REAL_WORLD_DEMO.md

## This is a demonstration, not a validated evaluation

Everything in `docs/EVALUATION.md` and `docs/EXPERIMENTS.md` is measured against **held-out
ground-truth labels** from the LEVIR-CD benchmark — real IoU, Dice, precision, recall, F1,
accuracy. **Nothing in this document has that property.** There is no ground-truth change mask for
the real-world Sentinel-2 imagery used here, so no accuracy metric can be computed, and none is
claimed. This document shows what the trained model *predicts* on real, independently-sourced
satellite imagery and discusses, honestly, why that prediction should not be read with the same
confidence as the benchmark results.

## Data source

**Sentinel-2 L2A**, via the Earth Search STAC API (Element84, hosted on AWS Open Data):
`https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`.

### Why this source, not the official Copernicus portal
The official Copernicus Data Space Ecosystem requires account registration — a manual-download
step, per the same reasoning `docs/DATASET.md` applied to LEVIR-CD's official Google Drive/Baidu
links. Earth Search exposes the same underlying Sentinel-2 L2A data (processed by ESA, mirrored
to AWS Open Data) as plain HTTPS Cloud-Optimized GeoTIFFs with **no authentication required** —
a reliable programmatic source, used here for the same reason the LEVIR-CD Hugging Face mirror was
used instead of Google Drive. This is documented as the acquisition source (not a "mirror" in the
same sense as the LEVIR-CD case — Earth Search re-hosts the original ESA-processed data directly,
it does not repackage a third party's derivative).

### What was fetched
The `visual` asset (TCI — True Color Image, already radiometrically rendered to 8-bit RGB) of two
Sentinel-2 L2A scenes, read as a small windowed crop directly over HTTP (no full ~100+ MB tile
download) via `rasterio`'s `/vsicurl/` support — `src/geospatial/raster.py`.

## Location and dates

**Location:** a suburb of Pflugerville, TX (bounding box `[-97.65, 30.41, -97.59, 30.46]`,
WGS84). Chosen because it sits within LEVIR-CD's own source region (the dataset's 20 regions
include several Austin-area Texas suburbs — `docs/DATASET.md`) — a deliberate choice for
thematic continuity with the training data's geography and development pattern, **not** a claim
that this improves the validity of the result (the sensor/resolution gap discussed below applies
regardless of location).

**Dates:**
- Before: **2019-12-06** (`S2A_14RPU_20191206_1_L2A`, cloud cover 0.001%)
- After: **2024-12-19** (`S2A_14RPU_20241219_0_L2A`, cloud cover 0.003%)

Both winter dates, chosen specifically to reduce seasonal vegetation/lighting differences as a
confound (per `PROJECT_CONTEXT.md`'s "actual change vs. apparent difference" principle) — a ~5
year gap is enough for real suburban development to occur in a fast-growing area, without the
before/after pair also differing in season. Both scenes have near-zero cloud cover.

## The critical domain gap: resolution

| | LEVIR-CD (training data) | Sentinel-2 (this demo) |
|---|---|---|
| Pixel size | **0.5 m/pixel** | **10 m/pixel** |
| Resolution ratio | 1x | **20x coarser** |
| Sensor | Google Earth (aerial/high-res satellite composite) | Sentinel-2 MSI |
| A single building | Tens to hundreds of pixels | Often **a handful of pixels or less** |

This is the single most important caveat in this document. The model was trained exclusively on
0.5 m/pixel imagery where individual buildings are large, well-resolved shapes. At 10 m/pixel, a
typical suburban house occupies a fraction of one pixel to a few pixels — the fine building-outline
detail the model learned to recognize simply does not exist in Sentinel-2 imagery at this
resolution. Per `PROJECT_CONTEXT.md`'s explicit instruction ("If Sentinel-2 data is unsuitable for
the trained model, document the limitation rather than forcing an invalid demonstration"), this
gap is stated here plainly rather than glossed over — the result below should be read in that
light.

Other real, unquantified domain differences: different sensor (Sentinel-2 multispectral instrument
vs. Google Earth composite imagery), different atmospheric/radiometric correction pipeline, and no
guarantee of the same georegistration precision as LEVIR-CD's pre-aligned pairs.

## Method (reproducible)

```bash
python scripts/real_world_demo.py
```

Workflow (matches `PROJECT_CONTEXT.md`'s Phase 11 diagram): select location → fetch date-A image →
fetch date-B image → preprocess (resize to the model's 256px input, same pipeline as
`src/data/preprocessing.py`) → run the Phase 8 best model (Siamese U-Net + Attention) → predicted
change mask → region/statistics summary (no physical-area conversion is applied here, since the
LEVIR-CD-derived pixel-size assumption in `src/analysis/area.py` does not apply to 10 m/pixel
Sentinel-2 imagery — reporting hectares here would misapply that documented assumption).
All parameters (location, item IDs, dates, cloud cover, resolution gap, prediction summary) are
saved to `outputs/real_world_demo/report.json` for reproducibility (Rule 7).

## Result (real, measured prediction — not validated against ground truth)

```
Before: 2019-12-06 (cloud cover 0.001%)
After:  2024-12-19 (cloud cover 0.003%)
Predicted: 1,621 / 65,536 pixels changed (2.47%), 19 regions (>=4px)
```

Full visual output: `outputs/real_world_demo/combined.png` (before / after / predicted mask /
overlay, in that order — same visual layout as `docs/EVALUATION.md`'s benchmark grids, for direct
visual comparison of style, not of validity).

### What the model got right (by visual inspection — genuinely useful, not just a caveat)
A large new commercial/industrial building complex, clearly visible in the after-image (present
in `2024-12-19`, absent in `2019-12-06`) near the top-right of the crop, was correctly detected —
one of the model's largest predicted regions sits exactly on that new building's real footprint.
This is a genuine positive result: despite the 20x resolution gap, the model generalized well
enough to catch the single most obvious real change in the scene.

### What is uncertain (by visual inspection, honestly reported)
The predicted mask also contains several smaller scattered regions elsewhere in the scene. At
this crop's resolution and the reviewer's ability to visually inspect it, it was **not possible to
independently confirm or rule out** whether these correspond to real (but subtle, sub-pixel-scale)
development, image registration/alignment artifacts, or genuine false positives caused by the
resolution/sensor domain gap. This is exactly the situation `PROJECT_CONTEXT.md`'s "actual
geographical change vs. apparent visual difference" principle warns about — without ground truth,
these predictions cannot be trusted at face value, and this document does not claim they represent
real change.

## What this demonstration does and does not show

**Does show:** the trained model can be run end-to-end on real, independently-sourced Sentinel-2
imagery through a fully automated, reproducible pipeline, and its top prediction on this one
example corresponds to a real, visually-confirmable change.

**Does NOT show:** that the model's benchmark performance (`docs/EVALUATION.md`: IoU=0.6560 on
LEVIR-CD) transfers to Sentinel-2 imagery. No quantitative metric was computed here because no
ground truth exists. This is one anecdotal example at one location and one date pair — not a
systematic evaluation, not a statistically meaningful sample, and not evidence of general
real-world reliability. A rigorous real-world evaluation would require independently labeled
Sentinel-2 change masks, which this project does not have and has not attempted to fabricate.

## Status summary

| Item | Status |
|---|---|
| Programmatic, unauthenticated Sentinel-2 access | **Implemented** (`src/geospatial/raster.py`) |
| End-to-end real-world inference pipeline | **Implemented** (`scripts/real_world_demo.py`) |
| Qualitative real-world result (one location/date pair) | **Measured, documented** — see above |
| Domain-gap analysis (resolution/sensor) | **Documented explicitly** |
| Quantitative real-world accuracy (IoU/Dice/etc. on Sentinel-2) | **Not possible** — no ground truth available; not attempted |
| Systematic multi-location/multi-date real-world evaluation | **Future scope** — would require real-world ground-truth labels |
