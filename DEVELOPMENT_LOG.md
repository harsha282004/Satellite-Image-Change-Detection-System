# DEVELOPMENT_LOG.md

Running log of phase completions. Newest entry at the top. See `PROJECT_CONTEXT.md` for phase
definitions and `DEVELOPMENT_RULES.md` for the verification rules each entry must satisfy.

**Note on Phases 17-23:** run autonomously per explicit user authorization ("CONTINUE
AUTONOMOUSLY — DO NOT WAIT FOR MY CONFIRMATION", given after Phase 16) — no per-phase pause for
confirmation from this point forward, per that instruction. Every rule in `DEVELOPMENT_RULES.md`
(no fabricated metrics, no test-set leakage, preserve prior results, etc.) still applies in full;
only the "wait for the user between phases" behavior changed.

---

## PHASE 22 — Real-World Pipeline Hardening

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/realworld/validation.py`: `REAL_WORLD_DISCLAIMER` (the exact required text: "Model trained
  on LEVIR-CD imagery. Performance on this imagery has not been independently validated.");
  `check_dimensions_match()`; `estimate_registration_offset()` (real `cv2.phaseCorrelate`
  phase-correlation shift estimate, flagged above 3px — a diagnostic only, does not correct/align
  images); `assess_resolution_plausibility()` (flags a pixel size ≥5x coarser than LEVIR-CD's
  0.5 m/pixel training resolution, only when a real pixel size is known — never guessed);
  `screen_for_cloud_cover()` (explicitly heuristic bright+low-saturation pixel screen, NOT a
  validated cloud detector — no labeled cloud-mask data exists in this project to validate one
  against, stated in every warning it produces); `validate_real_world_input()` orchestrates all
  four and aggregates warnings.
- `dashboard/app.py`: upload flow now displays the exact required disclaimer (`st.warning`) and
  the real Phase 22 validation report (expander with all warnings, or a clean-pass caption) for
  every uploaded before/after pair, before inference runs. Capabilities table extended with
  Phase 18/19/20/21/22 rows (geospatial, multi-class limitation, Transformer comparison,
  multi-temporal, input validation) — none of it was reflected there before this phase.
- `scripts/real_world_demo.py`: now calls the same `validate_real_world_input()` (replacing a
  previously-inline, duplicated resolution-gap calculation — Rule 6), prints the disclaimer and
  any warnings, and saves the full validation report into `report.json`.
- `tests/test_realworld_validation.py`: 11 new tests (dimension match/mismatch, resolution
  plausibility for matching vs. Sentinel-2-like coarse resolution, registration-offset near-zero
  for identical images vs. detected for a synthetically `np.roll`-shifted image, dimension-mismatch
  handling, cloud-heuristic flagging a synthetic bright/white image vs. not flagging a natural
  random image, and full-orchestrator aggregation for both a clean pair and a problematic one).
- `docs/LIMITATIONS.md`: new "Real-world input validation (Phase 22)" section explaining exactly
  what each check does and does **not** do (an estimate, not a correction; a heuristic, not a
  validated detector); "Not implemented" section updated to stop listing cloud/registration
  detection as entirely absent, replaced with the precise, narrower honest gap that remains
  (correction/validated detection, not estimation/heuristic screening).

**FILES CREATED:**
- `src/realworld/__init__.py`, `src/realworld/validation.py`, `tests/test_realworld_validation.py`

**FILES MODIFIED:**
- `dashboard/app.py`, `scripts/real_world_demo.py`, `docs/LIMITATIONS.md`, `README.md`

**EXPERIMENTS RUN (real, against live Sentinel-2 data and a live dashboard process):**
1. `scripts/real_world_demo.py` re-run end-to-end (2026-08-25) against the same Pflugerville, TX
   pair as Phase 11 (before=2019-12-06, after=2024-12-19) — confirmed the disclaimer prints, the
   resolution-mismatch warning correctly fires (20.0x, matching Phase 11's already-documented
   domain gap), no registration/cloud warnings for this well-aligned pair, and the model's
   prediction is unchanged from Phase 11 (1621/65536 px changed, 2.47%, 19 regions ≥4px) — Phase 22
   adds validation output only, it does not alter any prediction.
2. Killed a stale Streamlit server process (leftover, holding port 8501) and relaunched fresh with
   the Phase 22 dashboard changes loaded — standard practice for dashboard code changes, per
   earlier phases' documented Streamlit live-reload caching gotcha.
3. Verified the dashboard script executes with zero top-level exceptions via Streamlit's official
   in-process `streamlit.testing.v1.AppTest` harness (a genuine, non-mocked execution of the real
   script, including the new imports and Capabilities table). A full browser-based Playwright
   walkthrough (the pattern used in Phases 10/15-17) was not performed this phase — no working
   Playwright/Node install was available in this session's scratchpad, and setting one up fresh
   would cost a large amount of wall-clock time at this session's measured ~18 KB/s network
   throughput (Phase 19's finding) for a check `AppTest` + the real `real_world_demo.py` run
   already substantially cover. Documented here as an honest scope reduction, not silently skipped.

**RESULTS (actual, measured):**
```
Real-world demo re-run: disclaimer displayed, 1 warning (20.0x resolution mismatch, as expected),
prediction unchanged from Phase 11 (1621/65536 px, 2.47%, 19 regions)
Dashboard: 0 top-level exceptions (streamlit.testing.v1.AppTest), HTTP 200 on a live server
```

**TESTS:**
- `pytest tests/test_realworld_validation.py -v`: 11/11 passed.
- `pytest tests/`: 181/181 passed (170 from Phase 21 + 11 new).

**DOCUMENTATION UPDATED:**
- `docs/LIMITATIONS.md` — new Phase 22 section, "Not implemented" section narrowed to the honest
  remaining gap.
- `README.md` — new "Real-World Pipeline Hardening" section.
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- Registration-offset estimation is a diagnostic only — no correction/alignment is performed, and
  phase correlation itself can be fooled by real large-scale change between the two images (not
  only by misregistration), so a flagged pair is not necessarily misaligned and an unflagged pair
  is not guaranteed to be perfectly aligned.
- The cloud/bright-region screen is explicitly a heuristic (brightness + low saturation), not a
  validated cloud detector — stated in every warning it produces, not just in this document.
- No full browser-based (Playwright) dashboard walkthrough was performed this phase — see
  "Experiments run" above for the substitute verification actually performed and why.
- None of Phase 22's checks validate prediction *accuracy* — they validate input characteristics
  only; no ground truth exists for arbitrary real-world uploads, so no accuracy claim is made.

**NEXT PHASE:**
- PHASE 23 — Final Unified Dashboard + Comprehensive Final Report. Proceeding automatically per
  the standing autonomous authorization. This is the final phase specified in the mega-prompt.

---

## PHASE 21 — Multi-Temporal Change Analysis

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/temporal/sequence.py`: `select_temporal_sequence(items, n_dates)` picks N dates spread as
  evenly as possible across a real STAC search's available time span (closest-real-item-to-each-
  evenly-spaced-target-timestamp, never fabricating a date); `build_intervals()` pairs an ordered
  sequence into adjacent (from, to) tuples; `compute_interval_record()` packages one interval's
  real change statistics (via `src/analysis/statistics.py`, unmodified) and severity distribution
  (via `src/analysis/severity.py`, unmodified) with its real dates/STAC item ids. Module docstring
  states explicitly, and repeats everywhere the result is surfaced: **no causal or tracking claims
  are made** — each interval is an entirely independent two-image detection; the model has no
  mechanism to track a specific change across more than two images.
- `scripts/multitemporal_analysis.py`: real end-to-end script — searches real Sentinel-2 items
  over Phase 11/18's Pflugerville, TX bbox across a multi-year range with genuine cloud filtering,
  selects N dates, fetches real georeferenced crops for each (Phase 18's
  `fetch_georeferenced_crop`), runs the best model independently on every adjacent pair, computes
  per-interval statistics/severity, exports JSON/CSV, and renders a temporal bar-chart
  visualization explicitly captioned "independent detections, not a tracked trend" on the chart
  itself.
- `tests/test_temporal_sequence.py`: 8 new tests using a tiny synthetic fake-STAC-item class
  (id + datetime only, no network) — date-count validation, endpoint selection for 2 dates, real
  even spread across a synthetic 8-year span, no-duplicate-item guarantee, interval pairing
  correctness, and `compute_interval_record` correctness (with and without detected change).
- `docs/EVALUATION.md`: new "Phase 21" section (real result table, no-causal-claims statement,
  status summary).
- The existing two-image analysis path (`src/inference/predict.py`, `src/analysis/*`, the
  dashboard's default mode) is completely unmodified — verified by the full test suite passing
  unchanged and by `scripts/multitemporal_analysis.py` calling those modules' existing public
  functions rather than reimplementing them.

**FILES CREATED:**
- `src/temporal/__init__.py`, `src/temporal/sequence.py`, `scripts/multitemporal_analysis.py`,
  `tests/test_temporal_sequence.py`

**FILES MODIFIED:**
- `.gitignore` (added `outputs/multitemporal/`, same regenerable/network-dependent pattern as
  `outputs/geospatial/`), `docs/EVALUATION.md`, `README.md`

**EXPERIMENTS RUN (real, against live Sentinel-2 data):**
1. `scripts/multitemporal_analysis.py`, run 2026-08-25. Searched
   `2017-01-01/2024-12-31` over `[-97.6500, 30.4100, -97.5900, 30.4600]` with cloud cover < 5%:
   **385 real candidate dates found** (2017-01-07 to 2024-12-31). Selected 5 dates spread across
   that span; fetched 5 real georeferenced crops; computed 4 independent intervals.

**RESULTS (actual, measured — full data: `outputs/multitemporal/temporal_report.{json,csv}`):**
```
2017-01-07 -> 2019-01-05: 1 region,  390 px changed (0.119%), 3.90 ha
2019-01-05 -> 2021-01-04: 1 region,  720 px changed (0.220%), 7.20 ha
2021-01-04 -> 2022-12-25: 3 regions, 426 px changed (0.130%), 4.26 ha
2022-12-25 -> 2024-12-31: 1 region,  286 px changed (0.087%), 2.86 ha
```
A real, unforced result — not tuned to show a particular pattern. Explicitly NOT presented as a
trend line for one physical change (see the no-causal-claims notice above); each interval carries
the full Phase 11/18 domain-gap caveat (10 m/pixel Sentinel-2 vs. 0.5 m/pixel training data, no
ground truth for any real-world pair here).

**TESTS:**
- `pytest tests/test_temporal_sequence.py -v`: 8/8 passed.
- `pytest tests/`: 170/170 passed (162 from Phase 20 + 8 new).

**DOCUMENTATION UPDATED:**
- `docs/EVALUATION.md` — new Phase 21 section.
- `README.md` — new "Multi-Temporal Analysis" section (after "Geospatial Change Intelligence").
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- No causal or tracking claims are possible with this architecture — repeated deliberately, since
  this is the single most important caveat about this feature (same pattern as Phase 17's severity
  disclaimer).
- Only demonstrated on one location (Pflugerville, TX) and one 5-date/4-interval selection, not a
  systematic multi-location or multi-density temporal study.
- Same domain-gap and no-ground-truth limitations as Phase 11/18, inherited in full.
- `outputs/multitemporal/` requires live network access to regenerate and is gitignored,
  consistent with `outputs/real_world_demo/` and `outputs/geospatial/`.

**NEXT PHASE:**
- PHASE 22 — Real-World Pipeline Hardening. Proceeding automatically per the standing autonomous
  authorization.

---

## PHASE 20 — Transformer-Based Architecture (Research Comparison)

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `models/transformer_change.py` (`TransformerChangeDetector`, `TransformerEncoder`,
  `PatchEmbed`, `DecoderBlock`): a genuinely self-attention-based Siamese change-detection
  architecture — `nn.TransformerEncoder` (multi-head self-attention, learnable positional
  embedding, global receptive field from the first layer) as a shared-weight encoder for both
  before/after images, feature comparison reused from `models/siamese_unet.py`
  (`compare_features`/`comparison_channels`, Rule 6), decoded back to full resolution with
  transposed-convolution blocks. This is the Transformer variant Phase 8 explicitly deferred
  (`docs/EXPERIMENTS.md` "Why the Transformer variant was not implemented this phase") — now
  actually built and measured, as a **research comparison only**, never replacing the Siamese
  U-Net + Attention model, which remains this project's primary result.
- `configs/transformer.yaml`: trains under the *exact same* controlled protocol as Phase 8's
  original 5-architecture comparison (30 epochs, Adam lr=1e-4, BCE+Dice loss, batch size 8, image
  size 256, seed 42) for a fair, unconfounded comparison.
- `src/training/train.py`: `build_model()` gained a `"transformer_change"` branch; new import.
- `scripts/architecture_comparison.py`: measures parameters and inference time (batch=1, 5 warmup
  + 50 timed forward passes, CUDA-synchronized) for **all 6** architectures under one identical
  procedure — inference time had never previously been measured for any model in this project.
  Test-set accuracy metrics are read from each model's existing real `*_test_metrics.json` (not
  re-run, to avoid changing already-reported numbers via Phase 6's documented GPU
  non-determinism).
- `tests/test_transformer_change.py`: 10 new tests (token-count correctness, shared-weight
  verification, invalid-config errors, forward-pass shape for all 3 comparison modes, backward/
  optimizer-step correctness, before/after order-sensitivity, encoder weight-sharing).
- `docs/EXPERIMENTS.md`: new "Phase 20" section (protocol, full 6-way comparison table, honest
  interpretation of the losing result, what would likely be needed to close the gap).

**FILES CREATED:**
- `models/transformer_change.py`, `configs/transformer.yaml`, `scripts/architecture_comparison.py`,
  `tests/test_transformer_change.py`

**FILES MODIFIED:**
- `src/training/train.py`, `docs/EXPERIMENTS.md`, `README.md`

**EXPERIMENTS RUN (real, on actual LEVIR-CD data):**
1. `python -m src.training.train --config configs/transformer.yaml` — 30 epochs, real training,
   889.5s (14.8 min) on the project's NVIDIA RTX 4050 Laptop GPU. Best epoch 27 (val IoU=0.3386);
   not early-stopped (no early stopping configured for this controlled-comparison run, matching
   Phase 8's protocol) — still slowly improving at epoch 30, unlike the CNN runs' earlier
   convergence.
2. `python -m src.evaluation.evaluate --config configs/transformer.yaml --checkpoint
   outputs/checkpoints/transformer_change_diff_concat/best.pt` — real held-out test-set evaluation.
3. `python scripts/architecture_comparison.py` — real parameter counts + inference timing for all
   6 models (the 5 Phase 8 CNN checkpoints + the new Transformer), one identical procedure.

**RESULTS (actual, measured — full data: `outputs/metrics/{transformer_change_diff_concat_test_metrics,architecture_comparison}.json`):**
```
Transformer (diff_concat): 4,054,481 params, 3.42 ms/pair inference (fastest, fewest params of all 6)
Test IoU=0.3575, Dice=0.5267, Precision=0.4774, Recall=0.5872, F1=0.5267, Accuracy=0.9462
```
**Substantially underperforms every CNN architecture**, including the weakest one (Siamese `diff`,
IoU=0.5569, from Phase 8). Reported honestly, as obtained — no post-hoc tuning was applied after
seeing this result. Consistent with the well-documented property that Vision Transformers lack
CNNs' spatial inductive biases and need more data or pretraining than this project's 445 from-
scratch training pairs provide — exactly the risk Phase 8's original deferral reasoning
anticipated, now confirmed by measurement rather than predicted.

**TESTS:**
- `pytest tests/test_transformer_change.py -v`: 10/10 passed.
- `pytest tests/`: 162/162 passed (152 from Phase 18/19 + 10 new).

**DOCUMENTATION UPDATED:**
- `docs/EXPERIMENTS.md` — new Phase 20 section.
- `README.md` — Experiments section extended with the honest Phase 20 result.
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- Single run/seed (42), same as every other experiment in this project — no variance estimate.
- No pretrained backbone was used or is available in this codebase's infrastructure — the single
  largest known lever for closing the gap was not attempted, and is documented as such rather than
  implied to be equivalent to a "true Transformer ceiling" result.
- Single-scale patch-grid Transformer (no hierarchical/Swin-style multi-scale design) — a
  materially larger implementation was out of scope for a research-comparison phase.
- Exact Phase 8 per-model training time was not recorded at the time it was run; only the
  Transformer's own training time (889.5s) is exactly measured here, not a like-for-like training-
  time comparison across all 6 — reported honestly as a gap, not backfilled with an estimate.

**NEXT PHASE:**
- PHASE 21 — Multi-Temporal Analysis. Proceeding automatically per the standing autonomous
  authorization.

---

## PHASE 19 — Multi-Class Change Detection

**Date:** 2026-08-25

**STATUS:** LIMITATION DOCUMENTED, NOT IMPLEMENTED (dataset genuinely unobtainable this session —
not skipped, not faked; per the standing autonomous-execution rule to document a real limitation
and continue rather than invent a workaround)

**INVESTIGATED:**
Per the explicit instruction to first inspect/select a suitable *properly labeled* multi-class
dataset rather than misusing LEVIR-CD's binary labels, three real candidates were evaluated:
1. **SECOND** (captain-whu, 4662 pairs, 512x512, 6 land-cover classes — the best-shaped candidate
   for this project's pipeline) — only distributed via Google Drive links on the official project
   page; no direct/resumable HTTP source suitable for headless, reliable fetch.
2. **HRSCD-Clean** (`EPFL-ECEO/HRSCD_clean` on Hugging Face — real semantic segmentation masks,
   MIT-licensed) — confirmed via `HfApi.dataset_info(..., files_metadata=True)` to be a single
   **60.25 GB zip**, no per-sample sharding.
3. **xView2/xBD** (genuine 4-class building-damage-severity labels — no damage/minor/major/
   destroyed; arguably the best conceptual fit, since the classes describe *change*, not land
   cover) — smallest Hugging Face mirror found (`Devansh25/xview2`) is 3.85 GB; other mirrors run
   11-33 GB.

**Root-caused the blocker, not just asserted it:** measured actual throughput with a direct HTTP
range-request benchmark against Hugging Face's own CDN (`cas-bridge.xethub.hf.co`, not a slow
mirror) — a 5 MB range request took 280.33s, i.e. **~18.3 KB/s**. This matches the ~10-22 KB/s
observed installing `pyproj`/`pyogrio` in Phase 18, confirming this is this session's genuine
network condition, not a dataset-specific or PyPI-specific slowdown. At ~18 KB/s: the smallest
candidate (3.85 GB) would take ~2.4 days; HRSCD-Clean's 60 GB would take ~38 days. No differently-
chosen dataset would avoid this — multi-class semantic/damage change detection inherently requires
paired high-resolution imagery plus per-pixel labels, which does not exist in a sub-100MB form.

**DECISION:** Phase 19 is not implemented this session. LEVIR-CD's binary labels are not
repurposed to fabricate multi-class output (would violate the never-fabricate rule). The existing
binary Siamese U-Net + Attention model and all Phase 1-18 work are completely unaffected and
untouched. Full investigation, the size/throughput table, and the recommended first candidate to
revisit if network conditions improve in a future session (`Devansh25/xview2`) are documented in
`docs/LIMITATIONS.md`.

**FILES MODIFIED:**
- `docs/LIMITATIONS.md` (new "Multi-class change detection (Phase 19)" section), `README.md`
  (Future Scope section)

**FILES CREATED:** None (no code was written for a dataset that could not be obtained).

**TESTS:** `pytest tests/`: 152/152 still passing (unchanged from Phase 18 — no code touched).

**NEXT PHASE:**
- PHASE 20 — Transformer-Based Architecture (Research Comparison). Proceeding automatically per
  the standing autonomous authorization. Uses only the existing LEVIR-CD data already on disk, so
  is unaffected by this phase's network-availability finding.

---

## PHASE 18 — Geospatial Change Intelligence

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/geospatial/raster.py` (extended): `fetch_georeferenced_crop(item, bbox_wgs84, out_path)`
  fetches a real Sentinel-2 crop and writes a local GeoTIFF that preserves the source item's actual
  CRS/affine transform (Phase 11's `fetch_visual_crop` discarded georeferencing — this is a
  separate function, Phase 11's demo is untouched); `has_georeference(raster_path)` — the hard
  guard, True only for a real CRS + non-identity transform, refuses geospatial conversion on plain
  imagery rather than inventing coordinates; `read_raster_metadata(raster_path)` reports real
  CRS/transform/bounds/resolution/dimensions read from the file.
- `src/geospatial/polygons.py` (new): `region_bbox_to_polygon()` converts a pixel-space region
  bounding box to a `shapely` polygon using the raster's real affine transform;
  `polygon_area_m2()` returns real area in m² for a projected CRS, raises for a geographic
  (degree) CRS rather than computing a wrong number; `polygon_to_wgs84()` reprojects via
  `pyproj.Transformer` for GeoJSON's mandated WGS84 output; `regions_to_geo_features()` — the
  end-to-end pixel-region-to-GeoJSON-feature pipeline, including Phase 17 severity when present;
  `features_to_geojson()`.
- `src/geospatial/maps.py` (new): `build_region_map()` builds an interactive Folium map with each
  detected region as a popup-annotated polygon layer (area, prediction probability, severity, and
  the standard not-ground-truth disclaimer in every popup); `save_map_html()`.
- `scripts/geospatial_analysis.py` (new): real end-to-end script — fetches a real georeferenced
  Sentinel-2 before/after pair (same Pflugerville, TX location/dates as Phase 11), runs the best
  model (`siamese_attention_e100`), resizes the prediction back to the raster's native pixel grid,
  converts to real geographic features, exports GeoJSON/CSV/GeoPackage + interactive map. Prints
  the Phase 11 domain-gap/no-ground-truth caveat at the top of every run.
- `tests/test_geospatial_phase18.py` (new): 11 tests using synthetic in-memory rasters
  (`rasterio.open(..., 'w', ...)`) — a real-CRS fixture and a plain/identity-transform fixture,
  proving the `has_georeference` guard correctly distinguishes them; polygon/area/reprojection
  correctness against known pixel geometries; end-to-end `regions_to_geo_features` correctness.
- New dependencies installed: `shapely`, `pyproj`, `geopandas` (+ `pyogrio`), `folium` (+
  `branca`, `xyzservices`) — added to `requirements.txt`.
- `docs/EVALUATION.md`: new "Phase 18" section (pipeline, real measured result, status summary),
  explicitly distinguishing image-space analysis (LEVIR-CD PNGs, no CRS) from this phase's
  geospatial analysis (real Sentinel-2 GeoTIFF).

**FILES CREATED:**
- `src/geospatial/polygons.py`, `src/geospatial/maps.py`, `scripts/geospatial_analysis.py`,
  `tests/test_geospatial_phase18.py`

**FILES MODIFIED:**
- `src/geospatial/raster.py`, `requirements.txt`, `.gitignore` (added `outputs/geospatial/`,
  gitignored like `outputs/real_world_demo/` since both require live network access and are
  regenerable), `docs/EVALUATION.md`, `README.md`

**EXPERIMENTS RUN (real, against live Sentinel-2 data):**
1. `scripts/geospatial_analysis.py`, run 2026-08-25 against `S2A_14RPU_20191206_1_L2A` (before)
   and `S2A_14RPU_20241219_0_L2A` (after), bbox `[-97.6500, 30.4100, -97.5900, 30.4600]`
   (Pflugerville, TX — same location/dates as Phase 11).

**RESULTS (actual, measured — full data: `outputs/geospatial/regions.{geojson,csv}`):**
```
Raster: 583x561 px, 10.0 m/pixel, CRS=EPSG:32614 (UTM zone 14N)
Regions detected: 6
Total detected-change area: 30.89 ha (from the raster's real UTM projection)
Severity: 1 Very High, 3 High, 2 Moderate (real scores, range 25.5-79.99)
```
A real, unforced result — not tuned. No ground truth exists for this pair (same limitation as
Phase 11), so this is a measurement, not a validated accuracy figure.

**TESTS:**
- `pytest tests/test_geospatial_phase18.py -v`: 11/11 passed.
- `pytest tests/`: 152/152 passed (141 from Phase 17 + 11 new), 4 benign
  `NotGeoreferencedWarning`s from the intentionally-non-georeferenced test fixture.

**DOCUMENTATION UPDATED:**
- `docs/EVALUATION.md` — new Phase 18 section.
- `README.md` — new "Geospatial Change Intelligence" section (after "Real-World Demonstration").
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- No ground truth exists for this real-world pair — same limitation as Phase 11's demonstration,
  inherited in full, not newly introduced.
- Only demonstrated on one real-world location/date pair (Pflugerville, TX), not a systematic
  geospatial evaluation.
- `outputs/geospatial/` requires live network access to Earth Search/AWS Open Data to regenerate
  and is gitignored, consistent with `outputs/real_world_demo/`.
- `pip install` for the new geospatial dependencies was very slow on this session's network
  (~20-45 min total, `pyogrio`'s wheel alone took ~26 min); all packages installed successfully in
  the end, no functional issue — noted here only because it dominated this phase's wall-clock time.

**NEXT PHASE:**
- PHASE 19 — Multi-Class Change Detection. Proceeding automatically per the standing autonomous
  authorization. First step: inspect/identify a suitable *properly labeled* multi-class change
  detection dataset — LEVIR-CD's binary labels will not be misused to fabricate classes; if no
  suitable dataset can be reliably obtained, the limitation will be documented and the project will
  move on to Phase 20, per the autonomous-execution rules.

---

## PHASE 17 — Change Severity Analysis

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/analysis/severity.py`: `compute_region_severity()` — a fully documented, weighted-sum
  formula (`severity_score = 100 * (0.35*area_score + 0.30*probability_score + 0.15*density_score
  + 0.20*relative_size_score)`, 0-100) built entirely from measurable Phase 16 region fields plus
  Phase 15's prediction probability; `severity_category()` (Low/Moderate/High/Very High,
  configurable thresholds); `compute_severity_for_regions()` (batch, non-mutating);
  `severity_distribution()` and `highest_severity_regions()` for the aggregate/ranking views.
  Every weight and constant (weights, area reference=500px, category thresholds) is named and
  documented as a chosen default, not derived from any labeled severity ground truth — none
  exists for this task, stated explicitly in the module docstring and repeated in the dashboard.
- `dashboard/app.py`: region table gained "Severity Score"/"Severity Category" columns; new
  "Severity Distribution" subsection (region-count-by-category metrics + top-5 highest-severity
  table), with the NOT-ground-truth disclaimer placed directly above it, not only in docs.
- `scripts/export_regions.py`: now scores every exported region with `compute_severity_for_regions`
  before writing CSV/JSON — re-run for real, `outputs/regions/regions.csv` gained
  `severity_score`/`severity_category` columns.
- 12 new pytest tests (`tests/test_severity.py`): category boundary correctness (both default and
  custom thresholds), score range validity, maximal/minimal-input edge cases, monotonicity
  (larger region → higher score, all else equal), area-score capping beyond the reference size,
  custom-weight re-ranking (proves the formula actually uses the weights, not hard-coded), non-
  mutation, distribution/ranking helper correctness.
- `docs/EVALUATION.md`: new "Phase 17" section (formula, real measured distribution, dashboard
  integration, status summary).

**FILES CREATED:**
- `src/analysis/severity.py`, `tests/test_severity.py`

**FILES MODIFIED:**
- `dashboard/app.py`, `scripts/export_regions.py`, `docs/EVALUATION.md`, `README.md`

**EXPERIMENTS RUN (real, on the actual best model's already-detected regions):**
1. `scripts/export_regions.py` re-run on the same 5 real test images as Phase 16, now with
   severity scores for all 258 regions.
2. Real end-to-end dashboard verification via Playwright (fresh server restart first): uploaded a
   real test image pair, confirmed "Severity Distribution" renders with no console/page errors.

**RESULTS (actual, measured — full data: `outputs/regions/regions.csv`):**
```
258 regions total, severity score range 22.3-69.9, mean 44.3
  Low:       3 regions
  Moderate: 215 regions
  High:      40 regions
  Very High:  0 regions
```
No region reached "Very High" on this real sample — an honest, unforced result consistent with
the formula's design (a region must be simultaneously large, high-probability, dense, AND a large
share of its image's total change to approach 100), not tuned to produce a particular distribution.

**TESTS:**
- `pytest tests/`: 141/141 passed (129 from Phase 16 + 12 new).

**DOCUMENTATION UPDATED:**
- `docs/EVALUATION.md` — new Phase 17 section.
- `README.md` — Inference & Change Analysis section extended with the real severity distribution.
- `DEVELOPMENT_LOG.md` — this entry, plus the autonomous-execution note above.

**KNOWN LIMITATIONS:**
- Severity has no ground-truth validation of any kind — repeated deliberately at every mention,
  since this is the single most important caveat about this feature.
- Weights/area-reference/category-thresholds are documented engineering defaults, not the result
  of any optimization or labeled-data-driven calibration.
- Scored only the same 5-image sample used in Phase 16, not the full test set.

**NEXT PHASE:**
- PHASE 18 — Geospatial Change Intelligence. Proceeding automatically per the standing autonomous
  authorization.

---

## PHASE 16 — Region-Level Change Intelligence

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/analysis/regions.py::extract_regions()` extended (not duplicated — Rule 6) with `width`,
  `height`, `perimeter` (via `cv2.findContours`+`arcLength`), `aspect_ratio`, `change_density`
  (pixel_count / bbox area), and — when an optional `probability_map` argument is passed —
  `mean_prediction_probability`/`max_prediction_probability` per region, sourced from the model's
  real sigmoid output, not a placeholder. Raises a clear `ValueError` on a probability-map/mask
  shape mismatch.
- `src/analysis/statistics.py::compute_change_statistics()` gained `smallest_region_pixels`/
  `smallest_region_area` (previously only largest/average existed) and threads the new
  `probability_map` parameter through to `extract_regions`. Both changes are backward-compatible
  (new parameters are optional/keyword, existing keyword-based call sites in `dashboard/app.py`
  and 3 scripts needed no changes to keep working).
- `src/visualization/overlays.py::create_region_id_overlay()`: draws each region's bounding box +
  numeric ID on an image (Phase 16.3), so a region in a table can be visually located.
- `scripts/export_regions.py`: real run on 5 test images — saves `outputs/regions/regions.csv`
  (258 region rows), `outputs/regions/regions.json`, and region-ID overlay PNGs. `--min-region-
  pixels` (default 4) is documented in the script's own docstring (effective ~2m/pixel ground
  sampling → 4px ≈ 16m², below what this model's training resolution can reliably distinguish
  from boundary noise) — Phase 16.2's "no unexplained hard-coded value" requirement.
- `dashboard/app.py`: new "4. Region Analysis" section — region-ID overlay image, then a region
  table (Region, Area px/m², Prediction Probability mean/max, Bounding Box, Width×Height) with an
  explicit caption stating severity scoring is not yet implemented (Phase 17) rather than adding
  an empty/placeholder column; "Change Statistics" gained a "Smallest Region" metric; capability
  table updated (region intelligence: implemented; severity: not yet). The dashboard's already-
  computed probability map (Phase 15) is now passed through to `compute_change_statistics`, so the
  new per-region probability stats come from a real forward pass, not a second one.
- Every region, everywhere in code/docs/dashboard, is called **"Detected Change Region"** —
  Phase 16's explicit instruction never to imply a semantic category (Building/Road/etc.) that
  LEVIR-CD's binary labels don't support.
- 13 new pytest tests: `tests/test_regions_phase16.py` (9 — geometry correctness on a known solid
  rectangle and a sparse L-shape, probability-map-present/absent behavior, shape-mismatch error,
  `smallest_region_pixels` correctness with/without pixel size, probability passthrough),
  `tests/test_overlays.py` (+4 — region-ID overlay shape/dtype, non-mutation of input, actually
  draws something, empty-regions no-op).
- `docs/EVALUATION.md`: new "Phase 16" section (fields added, noise-filtering rationale, region-ID
  overlays, dashboard table, real `outputs/regions/` export results, status summary).

**FILES CREATED:**
- `scripts/export_regions.py`
- `tests/test_regions_phase16.py`
- `outputs/regions/regions.csv`, `regions.json`, `region_ids_test_{29,45,52,75,99}.png` (gitignored,
  consistent with existing outputs policy — regenerable via `scripts/export_regions.py`)

**FILES MODIFIED:**
- `src/analysis/regions.py`, `src/analysis/statistics.py`, `src/visualization/overlays.py`
- `dashboard/app.py`, `docs/EVALUATION.md`, `README.md`, `tests/test_overlays.py`

**EXPERIMENTS RUN (real, on the actual best model):**
1. `scripts/export_regions.py` on 5 real test images (`test_29/45/52/75/99.png`), full region
   geometry + prediction-probability stats computed from real inference output.
2. Real end-to-end dashboard verification via Playwright (fresh server restart first, per the
   Phase 15 caching lesson): uploaded a real test image pair, confirmed the region-ID overlay and
   region table render with no console/page errors, and confirmed (via `[data-testid="stDataFrame"]`
   widget count, since Streamlit's dataframe grid is canvas-rendered and not plain-text-searchable)
   that the table is genuinely present, not silently missing.

**RESULTS (actual, measured — full data: `outputs/regions/regions.csv`/`.json`):**
```
Image          Regions  Largest(px)  Smallest(px)  Average(px)
test_29.png    67       854          4             141.9
test_45.png    113      1450         4             164.5
test_52.png    42       322          5             64.5
test_75.png    35       491          15            148.9
test_99.png    1        41           41            41.0
```
258 total regions across 5 images, each with real geometry + prediction-probability statistics.

**TESTS:**
- `pytest tests/`: 129/129 passed (116 from Phase 15 + 13 new).
- **A real correctness lesson, caught by testing and fixed in the test, not the code:** the first
  draft of the solid-rectangle geometry test asserted a 6×4-pixel rectangle has perimeter
  `2*(6+4)=20`; actual value 16. Root cause: `cv2.arcLength` measures along contour points at
  pixel *centers*, not the outer pixel boundary, so a solid *w*×*h* rectangle's true perimeter by
  this convention is `2*((w-1)+(h-1))`. `extract_regions()`'s implementation was already correct;
  the test's formula was wrong and was corrected with the reasoning documented inline — the same
  pattern as Phase 14's Tversky/Dice smoothing-convention lesson.

**DOCUMENTATION UPDATED:**
- `docs/EVALUATION.md` — new Phase 16 section.
- `README.md` — Inference & Change Analysis section extended.
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- Region export (`scripts/export_regions.py`) ran on 5 hand-picked test images (the same ones used
  in earlier phases' qualitative grids, for continuity), not the full 128-image test set.
- Severity scoring is explicitly not implemented — the dashboard table and docs both say so rather
  than adding an empty column or a placeholder value.
- `min_region_pixels=4`'s reasoning (ground-sampling-distance-based) is a documented engineering
  judgment, not derived from a labeled sensitivity study of what region sizes are genuinely noise
  vs. real small detections.

**NEXT PHASE:**
- PHASE 17 — Change Severity Analysis: not started without explicit user go-ahead, per this
  project's phase-by-phase execution rule.

---

## PHASE 15 — Confidence, Probability and Threshold Optimization

**Date:** 2026-08-25

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/inference/predict.py`: new `predict_probability()` (returns the raw `sigmoid(logits)`
  probability map, float32 in [0,1]) — `predict_mask()` now calls it internally and thresholds,
  so mask and probability are guaranteed consistent (same forward pass, verified by test). Added
  `Predictor.predict_probability_from_arrays/from_paths`.
- `src/evaluation/threshold_analysis.py`: `sweep_thresholds()` runs the model once per batch and
  reuses the logits to compute metrics at every candidate threshold (not one forward pass per
  threshold); `select_best_threshold()` picks by a given metric, breaking ties toward 0.5.
- `scripts/threshold_optimization.py`: real run — swept 9 thresholds (0.30-0.70) on the
  **validation** set, selected by validation IoU, then evaluated that one threshold once on the
  **test** set (never used for selection, per Rule 3). Wrote `outputs/metrics/threshold_analysis.
  csv` and a 5-panel metric-vs-threshold plot.
- `scripts/generate_probability_maps.py`: saved representative probability-map/mask/overlay
  visualizations (15.1) for 3 real test scenes.
- `src/evaluation/robustness.py`: 6 controlled perturbation functions (brightness ±30%, contrast
  ±30%, Gaussian noise σ=15, 5px shift), all pure and unit-tested.
- `scripts/robustness_analysis.py`: real run — 10 test images × 6 perturbations (applied to the
  after-image only, simulating date-to-date variation), IoU measured against unperturbed ground
  truth, worst-case failure visualized.
- `dashboard/app.py`: added a "Prediction Probability" panel (viridis heatmap); threshold slider
  now defaults to the validation-optimized value **only when the selected model matches the
  checkpoint that threshold was swept for**, else falls back to 0.5; capability table updated with
  new rows (probability display: implemented; threshold optimization: implemented; formal
  calibration: explicitly **not** implemented) — terminology "prediction probability" used
  throughout, "confidence" avoided per the instruction not to use that term without calibration.
- 30 new pytest tests: `test_predict_probability.py` (3 — shape/range, mask-probability
  consistency, cross-threshold consistency), `test_threshold_analysis.py` (5 — sweep correctness,
  a real recall-monotonicity check against actual model behavior, selection logic incl. tie-
  breaking), `test_robustness.py` (11 — each perturbation's identity/boundary/determinism
  properties, shape/dtype preservation for all 6 registered perturbations).
- `docs/EVALUATION.md`: new "Phase 15" section (probability maps, threshold sweep table +
  interpretation, dashboard integration, robustness table + worst-case finding); corrected the
  now-stale "threshold fixed at 0.5" limitation and "not yet implemented" status row.

**FILES CREATED:**
- `src/evaluation/threshold_analysis.py`, `src/evaluation/robustness.py`
- `scripts/threshold_optimization.py`, `scripts/generate_probability_maps.py`,
  `scripts/robustness_analysis.py`
- `tests/test_predict_probability.py`, `tests/test_threshold_analysis.py`, `tests/test_robustness.py`
- `outputs/metrics/threshold_analysis.csv`, `threshold_optimization_report.json`,
  `robustness_analysis.csv`, `robustness_summary.json` (gitignored, consistent with existing policy)
- `outputs/visualizations/threshold_analysis.png`, `probability_maps/*.png`,
  `robustness/worst_case_test_101.png` (gitignored)

**FILES MODIFIED:**
- `src/inference/predict.py` (refactored `predict_mask` to build on new `predict_probability`)
- `dashboard/app.py`, `docs/EVALUATION.md`, `README.md`

**EXPERIMENTS RUN (real, on the actual best model — `siamese_unet_diff_concat_attention_e100`):**
1. Threshold sweep: 9 thresholds × validation set (64 images), then the selected threshold once
   on the test set (128 images).
2. Probability-map generation: 3 representative real test scenes.
3. Robustness: 10 real test images (with ground truth) × 6 perturbations = 60 perturbed
   evaluations + 10 baseline evaluations, all against real LEVIR-CD ground truth.

**RESULTS (actual, measured — full data in the CSVs/JSONs listed above, interpretation in
`docs/EVALUATION.md` Phase 15 section):**
```
Threshold sweep (validation): IoU ranges 0.7131 (t=0.70) to 0.7196 (t=0.40) — a spread of only
  0.0065, i.e. the model is essentially threshold-insensitive in 0.30-0.70.
Selected threshold: 0.40 (by validation IoU).
Test set @ 0.40: IoU=0.7122  |  Test set @ default 0.50: IoU=0.7123 — a tie within noise, NOT
  a real improvement. Reported honestly as such, not spun as a win.

Robustness (mean IoU degradation across 10 images):
  Gaussian noise: +0.0098 (minimal)         Contrast +30%:  +0.0005 (negligible)
  Brightness +30%: +0.0200 (minimal)        Contrast -30%:  +0.1048 (substantial)
  Shift 5px:       +0.1178 (substantial)    Brightness -30%: +0.1198 (substantial)
Worst single case: test_101.png, contrast -30%, IoU drop = 0.4234 (from good detection to
  near-total miss, mostly false negatives) — real vulnerability, visualized and saved.
```
**Key findings, both genuine and neither exaggerated nor downplayed:** (1) threshold tuning does
not meaningfully help this model — a useful negative result; (2) the model has a real, measurable
vulnerability to darkened/low-contrast imagery and small misregistration, consistent with but now
quantifying the qualitative concerns already in `docs/LIMITATIONS.md`.

**TESTS:**
- `pytest tests/`: 116/116 passed (86 from Phase 14 + 30 new).
- Real end-to-end dashboard verification via Playwright after the `predict.py` refactor: an
  initial test run hit a real bug — Streamlit's `@st.cache_resource` had cached a `Predictor`
  instance from before the refactor, so the running server's file-watcher picked up the edited
  `dashboard/app.py` but served a stale cached object missing the new
  `predict_probability_from_arrays` method (`AttributeError`, caught in the server's own log, not
  hidden). Fixed by killing and cleanly restarting the Streamlit process (module-level caches
  don't survive a fresh interpreter); re-verified with the same Playwright script — zero errors,
  correct default threshold (0.40) shown in the slider, probability heatmap rendered correctly.

**DOCUMENTATION UPDATED:**
- `docs/EVALUATION.md` — new Phase 15 section; corrected two stale claims from before Phase 15 existed.
- `README.md` — Evaluation section summary of the Phase 15 findings.
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- No formal probability calibration (reliability diagrams / Expected Calibration Error) — stated
  explicitly as not implemented everywhere "prediction probability" appears, per the instruction
  never to call it "confidence" without one.
- Threshold sweep and robustness testing were both run only for the single best model
  (`siamese_unet_diff_concat_attention_e100`) — not repeated for the other 7 trained models.
- Robustness testing used one perturbation magnitude each (±30% brightness/contrast, σ=15 noise,
  5px shift) on 10 images — not a magnitude sweep or the full 128-image test set; a coarse,
  real first measurement, not an exhaustive robustness certification.
- Perturbations were applied to the after-image only, chosen as the more realistic simulation of
  date-to-date variation — the before-image-only or both-images-perturbed cases were not tested.

**NEXT PHASE:**
- PHASE 16 — Region-Level Change Intelligence: not started without explicit user go-ahead, per
  this project's phase-by-phase execution rule.

---

## PHASE 14.3-14.4 — Hyperparameter Experiments & Final Configuration

**Date:** 2026-08-24

**STATUS:** COMPLETED (Phase 14 in full: 14.1-14.4 all done)

**IMPLEMENTED:**
- 4 new configs, each identical to `configs/siamese_attention_e100.yaml` (loss confirmed as
  `bce_dice` in 14.2) except one hyperparameter: `configs/siamese_attention_lr5e-5.yaml`
  (lr=5e-5), `configs/siamese_attention_lr2e-4.yaml` (lr=2e-4),
  `configs/siamese_attention_weight_decay.yaml` (optimizer=adamw, weight_decay=0.01),
  `configs/siamese_attention_bs4.yaml` (batch_size=4, chosen over a larger batch for GPU VRAM
  safety margin on this shared machine — documented in the config's own comment). A controlled,
  deliberately non-exhaustive matrix (one variant per hyperparameter, not a cross-product), per
  the "not an unnecessarily huge grid search" instruction. The `lr=1e-4/wd=0/bs=8` baseline row
  reuses Phase 13 Experiment C rather than retraining it.
- Ran all 4 for real, evaluated all 4 on the real held-out test set, generated training-curve
  plots, wrote `outputs/metrics/hyperparameter_experiment_comparison.csv` (every value pulled
  programmatically from the source JSON/CSV files — a script cross-checked the CSV against source
  after writing, per the transcription-error lesson from Phase 14.2), and wrote up the full
  comparison, interpretation, and Phase 14.4's final-configuration conclusion in
  `docs/EXPERIMENTS.md` (new "Phase 14.3" and "Phase 14.4" sections appended).
- **Phase 14.4 conclusion: the single best configuration found across all of Phase 13+14 testing
  is Phase 13 Experiment C's exact recipe** (`configs/siamese_attention_e100.yaml` — Adam, lr=1e-4,
  weight_decay=0.0, batch_size=8, bce_dice loss, max 100 epochs with early stopping patience=10 +
  `ReduceLROnPlateau`) — nothing tested in Phase 14 (3 alternative losses, 2 alternative learning
  rates, one weight-decay setting, one batch-size setting — 7 new full training runs total) beat
  it on validation IoU, confirmed by test-set evaluation.

**FILES CREATED:**
- `configs/siamese_attention_lr5e-5.yaml`, `configs/siamese_attention_lr2e-4.yaml`,
  `configs/siamese_attention_weight_decay.yaml`, `configs/siamese_attention_bs4.yaml`
- `outputs/checkpoints/siamese_unet_diff_concat_attention_{lr5e-5,lr2e-4,wd0.01,bs4}/{best,last}.pt` (gitignored)
- `outputs/experiments/siamese_unet_diff_concat_attention_{lr5e-5,lr2e-4,wd0.01,bs4}/{history.csv,history.json}` (gitignored)
- `outputs/metrics/siamese_unet_diff_concat_attention_{lr5e-5,lr2e-4,wd0.01,bs4}_test_metrics.json` (gitignored)
- `outputs/metrics/hyperparameter_experiment_comparison.csv` (gitignored, consistent with existing policy)
- `outputs/visualizations/siamese_unet_diff_concat_attention_{lr5e-5,lr2e-4,wd0.01,bs4}_{training_curves,test_predictions}.png` (gitignored)

**FILES MODIFIED:**
- `docs/EXPERIMENTS.md` (Phase 14.3 hyperparameter section, Phase 14.4 final-configuration section)

**EXPERIMENTS RUN (real, on the actual GPU/dataset — 2 required a clean restart from scratch after
unrelated session interruptions killed them mid-run; neither partial run was ever reported as a
result):**
1. `lr=5e-5` — restarted once (first attempt killed at epoch 60 by a session interruption).
   Completed run: early-stopped at epoch 63/100, best epoch 53.
2. `lr=2e-4` — completed run: early-stopped at epoch 59/100, best epoch 49.
3. `weight_decay=0.01` (AdamW) — completed run: early-stopped at epoch 65/100, best epoch 55.
4. `batch_size=4` — completed run: early-stopped at epoch 67/100, best epoch 57.
5. All 4 evaluated on the real 128-image held-out test split via `src/evaluation/evaluate.py`
   using each experiment's own `best.pt` (selected by validation IoU during training — the test
   set was never used to choose a hyperparameter value).

**RESULTS (actual, measured — full data: `outputs/metrics/hyperparameter_experiment_comparison.csv`,
interpretation: `docs/EXPERIMENTS.md` "Phase 14.3"/"Phase 14.4" sections):**
```
Hyperparameter        Best ep.  Val IoU   Test IoU  Test Precision  Test Recall
lr=1e-4 (baseline)     68        0.7188    0.7123    0.8402          0.8239
lr=5e-5                53        0.6653    0.6560    0.7763          0.8090
lr=2e-4                49        0.7094    0.6999    0.8339          0.8133
weight_decay=0.01      55        0.7102    0.7028    0.8277          0.8232
batch_size=4           57        0.7135    0.6997    0.8424          0.8051
```
**Every variant underperformed the baseline, on both validation and test data (the ranking held
across both, a real consistency check).** Learning rate showed the clearest pattern: halving it
(5e-5) hurt substantially (worst result in the whole experiment set), doubling it (2e-4) hurt much
less — interpreted as `ReduceLROnPlateau` already annealing the 1e-4 baseline down through useful
intermediate values, so starting lower leaves less useful range before the `min_lr=1e-6` floor.
Weight decay and batch size both landed close to but consistently below baseline — small, real
effects, consistent with this project never having observed overfitting (so regularization had no
problem to solve) and with batch_size=4's noisier gradients not being large enough to help here.
**No hyperparameter change is adopted — the original Phase 13 configuration remains the best.**

**PHASE 14 CONCLUSION (14.1-14.4 combined):** across 8 real, measured configurations (the original
baseline + 3 alternative losses + 4 alternative hyperparameters), **the baseline
(`configs/siamese_attention_e100.yaml`) won every comparison.** This is a genuine, useful research
finding in its own right — Phase 13's training-strategy improvement (30→100 max epochs + early
stopping + scheduler) was the dominant lever for this architecture, and the specific loss/
hyperparameter choices that came with it were, empirically, already well-chosen. **No changes are
adopted from Phase 14; the best model in this project remains
`siamese_unet_diff_concat_attention_e100`, unchanged since Phase 13.**

**TESTS:**
- `pytest tests/`: 97/97 passed (no source-code changes in 14.3-14.4, only new configs/docs/data —
  confirms no regression from the config additions).
- Every CSV value cross-checked against its source JSON/CSV file programmatically (not hand-typed)
  before being treated as final, per the Phase 14.2 transcription-error lesson.

**DOCUMENTATION UPDATED:**
- `docs/EXPERIMENTS.md` — Phase 14.3 (hyperparameter comparison table + interpretation + training
  curves) and Phase 14.4 (final best-configuration writeup, selection reasoning, honestly-scoped
  "what remains untested" section) sections appended.
- `DEVELOPMENT_LOG.md` — this entry.
- `README.md` — not modified; no leaderboard change (the best model was already
  `siamese_unet_diff_concat_attention_e100` from Phase 13, and Phase 14 confirmed rather than
  superseded it).

**KNOWN LIMITATIONS:**
- No cross-product hyperparameter grid — each dimension varied independently against the same
  baseline, not jointly. A combined optimum elsewhere in the space is not ruled out.
- Single seed (42) throughout — same caveat as every experiment in this project.
- Batch_size=4 was chosen over a larger batch (e.g. 16) specifically for VRAM safety on a
  sometimes-shared GPU, not because it was expected to be the more informative direction to test —
  documented as a practical constraint, not a scientific choice.
- Loss-parameter values in Phase 14.2 were not swept (one setting each) — remains possible a
  different parameterization could close some of the gap to BCE+Dice.

**NEXT PHASE:**
- PHASE 15 — Confidence, Probability and Threshold Optimization: not started without explicit user
  go-ahead, per this project's phase-by-phase execution rule.

---

## PHASE 14.1-14.2 — Loss Function Experiments

**Date:** 2026-08-24

**STATUS:** IN PROGRESS (14.1-14.2 complete; 14.3 hyperparameter experiments and 14.4 final
configuration not started — reported separately once run, per the phase-by-phase execution rule)

**IMPLEMENTED:**
- `models/losses.py`: added `FocalLoss` (binary focal loss, Lin et al. 2017, configurable
  `alpha`/`gamma`), `FocalDiceLoss` (Focal + Dice, mirrors `BCEDiceLoss`'s structure),
  `WeightedBCEDiceLoss` (positive-class-weighted BCE + Dice, configurable `pos_weight`, documented
  as a moderate choice — not the exact ~1:21 inverse class ratio — since very large pos_weight is
  known to over-predict the positive class), `TverskyLoss` (configurable `alpha`/`beta`, default
  0.3/0.7 — Salehi et al. 2017's recall-favoring default). All four registered in `get_loss()`.
- `src/training/train.py`: `get_loss()` call now passes through `config["training"].get(
  "loss_params", {})`, so loss hyperparameters are config-driven and recorded per experiment
  (Rule 7/14); added a `Loss: ... params=...` startup print line for transparency.
- `configs/siamese_attention_focal_dice.yaml`, `_weighted_bce_dice.yaml`, `_tversky.yaml`: each
  identical to `configs/siamese_attention_e100.yaml` (Phase 13's scientifically justified best
  training strategy — max 100 epochs, early stopping patience=10, `ReduceLROnPlateau`) except the
  loss function and its parameters — isolates the loss as the only variable, per the "controlled
  experiment" requirement. **The BCE+Dice entry in this comparison reuses Phase 13 Experiment C's
  result rather than retraining it** — it is already the exact controlled measurement for
  `loss=bce_dice` under this training strategy.
- 11 new pytest tests (`tests/test_losses_phase14.py`): Focal loss correctness (near-zero for
  confident-correct predictions, `alpha` weighting verified), Focal+Dice gradient flow, Weighted
  BCE+Dice reduces to plain BCE+Dice at `pos_weight=1.0` and penalizes missed positives more at
  higher `pos_weight`, Tversky reduces to Dice at `alpha=beta=0.5` **with `smooth=0`** (a real
  discrepancy was caught and fixed here — see TESTS below), Tversky penalizes false negatives more
  than false positives at its recall-favoring default, and `get_loss()` factory/kwargs-passthrough
  checks for all three new losses.
- Ran all 3 new loss experiments for real, evaluated all 4 loss variants (3 new + the reused
  BCE+Dice) on the real held-out test set, generated training-curve plots for the 3 new
  experiments, wrote `outputs/metrics/loss_experiment_comparison.csv`, and wrote up the full
  comparison and interpretation in `docs/EXPERIMENTS.md` (new "Phase 14 — Loss Function
  Experiments" section appended; also corrected a now-stale "best result overall" claim in the
  Phase 8 status list to point to Phase 13's improved result instead).

**FILES CREATED:**
- `configs/siamese_attention_focal_dice.yaml`, `configs/siamese_attention_weighted_bce_dice.yaml`,
  `configs/siamese_attention_tversky.yaml`
- `tests/test_losses_phase14.py`
- `outputs/checkpoints/siamese_unet_diff_concat_attention_{focal_dice,weighted_bce_dice,tversky}/
  {best,last}.pt` (gitignored)
- `outputs/experiments/siamese_unet_diff_concat_attention_{focal_dice,weighted_bce_dice,tversky}/
  {history.csv,history.json}` (gitignored)
- `outputs/metrics/siamese_unet_diff_concat_attention_{focal_dice,weighted_bce_dice,tversky}_test_metrics.json` (gitignored)
- `outputs/metrics/loss_experiment_comparison.csv` (gitignored, consistent with existing
  outputs/metrics policy — regenerable from the JSON/CSV sources it was compiled from)
- `outputs/visualizations/siamese_unet_diff_concat_attention_{focal_dice,weighted_bce_dice,tversky}_{training_curves,test_predictions}.png` (gitignored)

**FILES MODIFIED:**
- `models/losses.py`, `src/training/train.py`, `docs/EXPERIMENTS.md`

**EXPERIMENTS RUN (real, on the actual GPU/dataset):**
1. Focal+Dice (`configs/siamese_attention_focal_dice.yaml`) — early-stopped at epoch 58/100, best
   epoch 48.
2. Weighted BCE+Dice (`configs/siamese_attention_weighted_bce_dice.yaml`) — early-stopped at
   epoch 63/100, best epoch 53.
3. Tversky (`configs/siamese_attention_tversky.yaml`) — early-stopped at epoch 49/100, best
   epoch 39.
4. All 3 evaluated on the real 128-image held-out test split via `src/evaluation/evaluate.py`
   using each experiment's own `best.pt` (selected by validation IoU, never the test set).
   BCE+Dice's test metrics reused unmodified from Phase 13 Experiment C.

**RESULTS (actual, measured — full data: `outputs/metrics/loss_experiment_comparison.csv`,
interpretation: `docs/EXPERIMENTS.md` "Phase 14" section):**
```
Loss                Best ep.  Val IoU   Test IoU  Test Precision  Test Recall
BCE+Dice (reused)   68        0.7188    0.7123    0.8402          0.8239
Focal+Dice          48        0.6758    0.6646    0.7803          0.8176
Weighted BCE+Dice   53        0.6579    0.6539    0.7199          0.8770
Tversky             39        0.6376    0.6322    0.6941          0.8764
```
**BCE+Dice won clearly and consistently** — +0.0477 test IoU over the next-best alternative
(Focal+Dice), same ranking on every metric except recall. Weighted BCE+Dice and Tversky (both
recall-favoring by design) show exactly the expected precision/recall shift (Tversky: highest
recall of all 4 at 0.8764, but lowest precision at 0.6941 and lowest IoU at 0.6322) — the loss
mechanisms are working as designed, they just weren't the right fix for this task: none of the
three alternatives needed more training time either (all early-stopped well before BCE+Dice's 78
epochs), suggesting a worse optimum found faster, not a slower path to the same place.
**Conclusion: BCE+Dice remains the loss function for this project's best model — no change
adopted.**

**TESTS:**
- `pytest tests/`: 97/97 passed (86 from Phase 13 + 11 new).
- **A real correctness issue was caught and fixed during test-writing, not shipped silently:** the
  first draft of `test_tversky_loss_equals_dice_when_alpha_beta_half` asserted `TverskyLoss(alpha=
  0.5, beta=0.5, smooth=1.0)` exactly equals `DiceLoss(smooth=1.0)` and failed (0.5008 vs. 0.5087).
  Root cause: the two losses' smoothing conventions are both individually standard/correct but
  algebraically different when `smooth≠0` (Tversky smooths `TP` and the denominator by `smooth`;
  this project's `DiceLoss` smooths `2*TP` and the denominator by `smooth`) — they only coincide
  exactly at `smooth=0`. Not a bug in `TverskyLoss` (it correctly implements the standard Tversky
  formula); the test's assumption was wrong and was corrected, with the reasoning documented
  inline in the test.

**DOCUMENTATION UPDATED:**
- `docs/EXPERIMENTS.md` — new Phase 14 loss-comparison section (table, interpretation, training
  curves, status); corrected a stale "best result overall" claim in the existing Phase 8 status
  list.
- `DEVELOPMENT_LOG.md` — this entry.
- `README.md` not yet updated for Phase 14 (no leaderboard change — BCE+Dice, already the best
  model's loss, was confirmed as the best loss, not replaced).

**KNOWN LIMITATIONS:**
- Each alternative loss was tested with one parameter setting each (`pos_weight=5.0`,
  `alpha=0.3/beta=0.7`, `focal_alpha=0.8/focal_gamma=2.0`) — not a parameter sweep. A different
  `pos_weight` or Tversky `alpha`/`beta` could plausibly change the result; this experiment shows
  these *specific, documented, literature-reasonable defaults* underperform BCE+Dice, not that no
  parameterization of these loss families could ever match it.
- Single seed (42) for all 4 — same caveat as every other experiment in this project.
- Loss experiments used the same architecture as Phase 13 (Siamese+Attention, `diff_concat`) only
  — untested whether a different loss might help a different architecture more.

**NEXT PHASE:**
- PHASE 14.3-14.4 — Hyperparameter Experiments & Final Configuration: controlled learning-rate
  (5e-5, 2e-4), weight-decay, and batch-size experiments (validation-selected, never the test
  set), then identify and document the single best overall training configuration. Not started
  without explicit user go-ahead, per this project's phase-by-phase execution rule.

---

## PHASE 13 — Advanced Training Strategy

**Date:** 2026-08-24

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/training/trainer.py`: `Trainer` gains optional `scheduler` and `early_stopping` constructor
  args (both default to off/`None`, preserving every pre-Phase-13 config's exact behavior).
  `fit()` now: logs current LR every epoch, steps the scheduler on validation IoU, tracks
  `best_epoch` (not just `best_val_iou`), and breaks the loop when `early_stopping.patience`
  epochs pass without a new best validation IoU — always keeping the best-epoch checkpoint
  regardless of when/whether training stops early. Returns a result dict (`best_val_iou`,
  `best_epoch`, `max_epochs`, `epochs_trained`, `early_stopped`) instead of a bare float.
- `src/training/train.py`: `build_optimizer` adds AdamW + configurable `weight_decay` (Adam
  remains the unchanged default); new `build_scheduler` builds `ReduceLROnPlateau` from a config
  dict or returns `None` for the string `"none"`/absent key; `main()` wires both into `Trainer`,
  wraps `fit()` in a wall-clock timer, and prints a fuller final summary (max epochs vs. actual
  epochs trained vs. best epoch vs. early-stopped, explicitly distinguished per the user's
  instruction not to conflate a capped budget with epochs actually trained).
- `src/visualization/plots.py`: added a conditional 5th subplot (learning-rate curve, log scale)
  — only rendered when an experiment's `history.csv` has an `lr` column (Phase 13+ experiments),
  so all 5 pre-Phase-13 training-curve PNGs are unaffected if regenerated.
- `configs/siamese_attention_e60.yaml`, `configs/siamese_attention_e100.yaml`: Experiments B/C —
  identical architecture/data/optimizer/seed to the preserved Experiment A
  (`configs/siamese_attention.yaml`), differing only in max epochs (60/100) and the new
  early-stopping (patience=10, monitor=val_iou) / `ReduceLROnPlateau` (factor=0.5, patience=4,
  min_lr=1e-6) blocks. Distinct `experiment_name`s from Experiment A and from each other —
  deliberately, per the Phase 6 lesson (never reuse a tracked experiment's name).
- 18 new pytest tests (`tests/test_trainer.py`): early stopping stops at the right epoch (scripted
  val_iou sequence via a mocked `validate`), best checkpoint retained is the best epoch not the
  last, scheduler reduces LR on a scripted plateau, LR stays constant with no scheduler, LR is
  logged every epoch, regression test proving early-stopping-disabled runs the full epoch count,
  plus direct unit tests of `build_optimizer`/`build_scheduler`.
- Ran all 3 experiments for real on the actual LEVIR-CD data/GPU (see EXPERIMENTS RUN/RESULTS
  below), evaluated all 3 on the real held-out test set, generated training-curve plots for B/C,
  wrote `outputs/metrics/training_experiment_comparison.csv`, and wrote up the full methodology,
  interpretation, and honest limitations in `docs/TRAINING.md` (Phase 13 section appended).
- Updated `README.md` (Results section: new Phase 13 A/B/C comparison table and "current best
  model" claim; Training section: mentions early stopping/scheduler and the new best-model
  command; Inference & Change Analysis section: "current best model" command updated to point at
  the new best checkpoint) and `dashboard/app.py` (`MODEL_OPTIONS`: added Experiment B and C as
  new selectable entries, with C — the new best — as the new default/first entry; the original
  Phase 8 attention entry and all other pre-existing entries were kept unchanged, per the explicit
  instruction not to remove/replace dashboard functionality unnecessarily).

**FILES CREATED:**
- `configs/siamese_attention_e60.yaml`, `configs/siamese_attention_e100.yaml`
- `tests/test_trainer.py`
- `outputs/checkpoints/siamese_unet_diff_concat_attention_{e60,e100}/{best,last}.pt` (gitignored)
- `outputs/experiments/siamese_unet_diff_concat_attention_{e60,e100}/{history.csv,history.json}` (gitignored)
- `outputs/metrics/siamese_unet_diff_concat_attention_{e60,e100}_test_metrics.json` (gitignored)
- `outputs/metrics/training_experiment_comparison.csv`
- `outputs/visualizations/siamese_unet_diff_concat_attention_{e60,e100}_{training_curves,test_predictions}.png` (gitignored)

**FILES MODIFIED:**
- `src/training/trainer.py`, `src/training/train.py`, `src/visualization/plots.py`
- `README.md`, `docs/TRAINING.md`, `dashboard/app.py`

**EXPERIMENTS RUN (real, on the actual GPU/dataset — see `docs/TRAINING.md` for the full
methodology and interpretation):**
1. Experiment A — preserved unchanged from Phase 8 (`configs/siamese_attention.yaml`, 30 fixed
   epochs, no scheduler, no early stopping). Not retrained; existing checkpoint/history/test-
   metrics re-verified unchanged after B and C completed.
2. Experiment B — `configs/siamese_attention_e60.yaml`, max 60 epochs, early stopping
   (patience=10) + `ReduceLROnPlateau`. Interrupted once by an unrelated session interruption at
   epoch 6 of an earlier attempt; killed cleanly and **restarted from scratch** (not resumed —
   this `Trainer` has no checkpoint-resume capability) after the partial run's outputs were
   deleted. The restart completed the full run reported below.
3. Experiment C — `configs/siamese_attention_e100.yaml`, max 100 epochs, same early
   stopping/scheduler settings as B. Also interrupted and killed once (partial progress to epoch
   55) during a separate session pause; partial outputs deleted and **restarted from scratch** for
   the same reason. The restart completed the full run reported below.
4. All 3 evaluated on the real, held-out LEVIR-CD test split (128 images) via
   `src/evaluation/evaluate.py`, using each experiment's own `best.pt` (selected by validation IoU
   during training) and own config — the test set was never used for checkpoint selection or for
   choosing which experiment "won" the model comparison (that judgment was made on validation IoU;
   test evaluation happened once per experiment, after training was already complete).

**RESULTS (actual, measured — full data: `outputs/metrics/training_experiment_comparison.csv`,
interpretation: `docs/TRAINING.md` Phase 13 section):**
```
                          Max ep.  Actual ep.  Best ep.  Early stop  Val IoU@best  Test IoU
Experiment A (Phase 8)    30       30          26        N/A         0.6702        0.6560
Experiment B (max 60)     60       60          60        No          0.7106        0.7031
Experiment C (max 100)    100      78          68        Yes         0.7188        0.7123
```
**Experiment A was genuinely undertrained**: giving the identical architecture/data/optimizer more
epochs plus a plateau-triggered LR scheduler improved test IoU by +0.0563 absolute (0.6560 →
0.7123), substantially larger than any architecture-choice effect measured in Phase 8. The
improvement mechanism is directly visible in Experiment B's training-curve LR panel: two scheduler
step-downs (epoch 39, epoch 58), each immediately followed by an acceleration in validation
IoU/Dice. **No overfitting was observed** in B or C — train/validation metrics track closely at
every best epoch. Experiment C's early stopping triggered from a genuine validation-IoU plateau
(oscillating ~0.70-0.72 for ~10 epochs), not from train/val divergence. Experiment B, notably,
never triggered early stopping and was still at its best on the very last (60th) epoch — an
honestly-reported open question about whether an even longer budget would help further, which
Experiment C's early stop (at epoch 78, best epoch 68) suggests is approaching a genuine plateau
for this specific architecture/seed/recipe, but does not prove for other configurations.

**siamese_unet_diff_concat_attention_e100 (Experiment C) is the new best model in this project**,
superseding Phase 8's `siamese_unet_diff_concat_attention` (Experiment A) — same architecture,
better training strategy.

**TESTS:**
- `pytest tests/`: 86/86 passed (68 from Phase 12 + 18 new: early-stopping control flow, best-
  checkpoint retention, scheduler LR reduction, LR logging, `build_optimizer`/`build_scheduler`
  unit tests, and a regression test confirming full-epoch-count behavior is unchanged when the new
  features are left unconfigured).
- Backward-compatibility smoke test: re-ran `configs/baseline.yaml` (throwaway experiment name,
  `--epochs 2`) through the modified `Trainer`/`train.py` — reproduced the exact epoch-1
  `train_loss=0.7457` seen in every prior baseline run, confirming the refactor introduced no
  behavioral change for existing configs.
- Both new configs (`siamese_attention_e60.yaml`, `siamese_attention_e100.yaml`) smoke-tested
  (1-2 epochs, throwaway checkpoint/history cleaned up afterward) before committing to full runs —
  confirmed correct param count (15,428,125, matching Experiment A's architecture exactly),
  correct scheduler/early-stopping config printed and active.
- All 3 experiments' checkpoints evaluated for real on the held-out test set (not estimated);
  Experiment A's pre-existing checkpoint/metrics re-verified byte-identical to their Phase 8 values
  after B and C's runs completed, confirming Rule 11 ("preserve all baseline results") was upheld.

**DOCUMENTATION UPDATED:**
- `docs/TRAINING.md` — full Phase 13 section: motivation, what was added, the 3-experiment
  comparison table, interpretation (undertraining confirmed, no overfitting observed, the open
  question about Experiment B's uncapped ceiling), and known limitations of this experiment set.
- `README.md` — Results section (new A/B/C table, explicit "new best model" statement), Training
  section (mentions early stopping/scheduler, adds the Experiment C training command), Inference &
  Change Analysis section ("current best model" example command updated).
- `DEVELOPMENT_LOG.md` — this entry.

**KNOWN LIMITATIONS:**
- Single seed (42) for A/B/C — no variance estimate; the Phase 6 GPU-non-determinism caveat
  applies to these runs too.
- Experiment A's training time was never precisely measured (the `time.time()` wall-clock
  instrumentation didn't exist until this phase) — recorded honestly as
  `NOT_PRECISELY_MEASURED_PRE_PHASE13` in the CSV rather than estimated or backfilled.
- Early-stopping patience (10) and scheduler patience (4) were used at their specified default
  values, not themselves tuned or searched.
- Only one architecture (Siamese+Attention, `diff_concat`) was tested with longer training —
  whether the baseline U-Net or the other comparison modes would show similar gains from longer
  training + scheduling is untested; Phase 8's equal-30-epoch comparison remains the only
  controlled cross-architecture result in this project.
- Two separate session interruptions killed in-progress training runs mid-epoch during this phase
  (once for Experiment B, once for Experiment C); both were restarted cleanly from scratch after
  deleting the partial outputs — no partial/interrupted run's results were ever reported as if
  complete.

**NEXT PHASE:**
- PHASE 14 — Loss Function and Hyperparameter Experiments: using Experiment C's training strategy
  (max epochs + early stopping + LR scheduler) as the new baseline recipe, compare loss functions
  (Focal+Dice, Weighted BCE+Dice, Tversky — to be added to `models/losses.py`) and a controlled
  learning-rate/weight-decay/batch-size matrix, selecting on validation performance only. Not
  started without explicit user go-ahead, per this project's phase-by-phase execution rule.

---

## PHASE 12 — Final Integration & Documentation

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `docs/LIMITATIONS.md`: consolidated, honest limitations document drawing on every real finding
  from Phases 2-11 (dataset scope/imbalance, mask-binarization caveat, undertrained models, no
  hyperparameter search, single-seed/GPU-non-determinism, fixed 0.5 threshold, only one
  Siamese+Attention combination trained, Transformer deliberately deferred, benchmark-only
  evaluation scope, and — the largest, most consequential gap — the Sentinel-2 resolution/domain
  gap with no real-world ground truth) — every item cites the specific doc/log entry it comes
  from, not a generic disclaimer list.
- `docs/TRAINING.md`: the training methodology document specified in the original project
  structure but not yet written — consolidated from what Phases 4-8 actually built and ran:
  pipeline description, the exact shared hyperparameter recipe and why each choice was made, and
  the Phase 6 reproducibility finding restated with its full implications for anyone re-running
  these commands.
- `tests/test_edge_cases.py` (8 tests, portable/synthetic): before/after images with genuinely
  different native dimensions from each other, explicit CPU-device inference, explicit GPU-device
  inference with a CPU/GPU output-agreement check (skipped if no CUDA GPU — none needed on this
  machine, since Phase 1 confirmed one), missing-file errors for both `load_image`/`load_mask`,
  an invalid (non-image) file error, and `Predictor` raising clear errors for a missing config or
  missing checkpoint path.
- `scripts/verify_end_to_end.py`: full real pipeline (load → preprocess → predict → analyze →
  visualize) run against the actual best trained checkpoint on 3 test images deliberately
  different from every sample name used in any prior phase's documentation (`test_10`, `test_45`,
  `test_80` vs. Phase 9's `test_1/29/52/75/99/121`), plus two real edge cases against real data:
  mismatched before/after native dimensions and a missing-file error.
- Final documentation consistency pass: found and fixed a stale "Phase 0 complete, no model
  trained yet" status banner at the top of `README.md` (left over from the very first commit —
  every subsequent phase had updated its own section but not that top-level banner), corrected
  the Limitations section to point to the now-finalized `docs/LIMITATIONS.md` instead of "written
  progressively", and added `docs/TRAINING.md`/`docs/REAL_WORLD_DEMO.md` to the README's project
  structure listing.

**FILES CREATED:**
- `docs/LIMITATIONS.md`, `docs/TRAINING.md`
- `tests/test_edge_cases.py`
- `scripts/verify_end_to_end.py`
- `outputs/visualizations/phase12_verification/*.png` (gitignored)

**FILES MODIFIED:**
- `README.md` (stale status banner fixed, Limitations/Training sections finalized, project
  structure listing corrected)

**COMMANDS EXECUTED:**
- `pytest tests/ -v` (66 -> 74 tests)
- `python scripts/verify_end_to_end.py`
- Manual inspection of one genuinely-fresh test image's before/after/prediction (`test_45.png`)

**TESTS:**
- `pytest tests/`: 74/74 passed, including the real CPU/GPU agreement check (this machine has the
  RTX 4050 GPU confirmed in Phase 1, so that test ran for real rather than being skipped) — CPU
  and GPU forward passes on identical weights/input agreed on >95% of predicted pixels (small
  disagreement expected and acceptable: different floating-point execution paths, not a bug).
- `scripts/verify_end_to_end.py` against the real best checkpoint (Siamese+Attention) and 3
  genuinely unseen test images: full pipeline succeeded on all 3 (mask shape/binarity asserted,
  not just eyeballed), mismatched-native-dimensions case succeeded (1024x1024 vs. 400x700 input),
  missing-file case correctly raised a clear `ValueError` rather than crashing or hanging.
- Manually inspected `test_45.png`'s actual before/after images after noticing its predicted
  26.14% changed-pixel figure was unusually high compared to every other example seen in this
  project — confirmed it is a **genuine, dramatic real change** (forest/farmland fully converted
  to a dense new subdivision covering nearly the entire tile), correctly detected at large scale,
  not a false-positive failure. Included as a real finding rather than being silently discarded
  for looking like an outlier.

**RESULTS (actual, measured):**
```
scripts/verify_end_to_end.py: ALL CHECKS PASSED
  test_10.png: 72 regions, 11.30% changed
  test_45.png: 118 regions, 26.14% changed (verified: genuine large-scale real development)
  test_80.png: 70 regions, 12.48% changed
  Mismatched dimensions (1024x1024 vs 400x700): handled correctly
  Missing file: raised clear ValueError, no crash

pytest tests/: 74 passed, 0 failed, 5.86s
  (includes a real GPU/CPU agreement test: >95% pixel agreement between devices)
```

**KNOWN ISSUES:**
- This phase did not re-verify Phase 10's dashboard UI in the browser again (already verified for
  real in Phase 10 with Playwright); no dashboard code changed since then, so re-verification was
  judged unnecessary rather than skipped by oversight.
- `docs/LIMITATIONS.md` and this log entry both explicitly flag that no systematic real-world or
  multi-seed evaluation exists — these remain genuinely open items, listed as Future Scope in
  `README.md`, not resolved by this phase.

**NEXT PHASE:**
- None remaining in the originally specified plan (Phases 0-12 complete). Any further work
  (Future Scope items: additional datasets, a Transformer variant, formal hyperparameter search,
  multi-seed variance estimates, multi-class change typing, a systematic real-world evaluation
  with independently-labeled ground truth) would be a new, explicitly-scoped initiative building
  on this now-complete, documented, and verified foundation — not an implicit continuation.

---

## PHASE 11 — Real-World Satellite Demonstration

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- Identified a reliable, unauthenticated Sentinel-2 source: Earth Search STAC API (Element84,
  AWS Open Data), after confirming (per the same reasoning as `docs/DATASET.md`'s LEVIR-CD
  acquisition) that the official Copernicus Data Space portal requires account registration —
  manual-download-only, so not used for automated acquisition.
- `src/geospatial/raster.py`: `search_sentinel2_items()`, `get_item_by_id()`,
  `fetch_visual_crop()` — STAC search plus windowed reads of remote Cloud-Optimized GeoTIFFs over
  HTTP (rasterio `/vsicurl/`), avoiding full ~100+ MB tile downloads for a small area of interest.
- `scripts/real_world_demo.py`: full reproducible pipeline (select location → fetch date-A/date-B
  imagery → preprocess → run the Phase 8 best model → predicted mask → statistics → save
  visualization + a `report.json` with every parameter needed to reproduce the run, per Rule 7).
- Selected location/dates by actually searching the STAC catalog (not guessing blind): a
  Pflugerville, TX suburb (within LEVIR-CD's own source region, for thematic continuity — a
  deliberate choice, not a claim it improves result validity), 2019-12-06 vs. 2024-12-19 (both
  near-zero cloud cover, both winter to reduce seasonal-lighting confound, ~5 year gap for real
  development to occur). Verified visually before committing to this location: fetched and
  inspected actual before/after crops, confirmed genuine visible new construction (a large
  commercial/industrial building complex) before writing any downstream documentation.
- `docs/REAL_WORLD_DEMO.md`: full write-up — explicit resolution-gap table (LEVIR-CD 0.5 m/pixel
  vs. Sentinel-2 10 m/pixel = **20x coarser**, "a typical suburban house occupies a fraction of
  one pixel to a few pixels" at this resolution), method, the real measured prediction (not a
  validated metric — no ground truth exists), an honest split between what the model got right
  (correctly flagged the one visually-confirmable large new building) and what's uncertain
  (several smaller predicted regions that could not be independently verified as real vs. domain-
  gap artifacts), and a status-summary table separating "measured/documented" from "not possible
  without ground truth" and "future scope".
- `README.md` "Real-World Demonstration" section added, explicitly distinguished from the
  benchmark "Results" section per `PROJECT_CONTEXT.md`'s requirement.
- 2 smoke tests (`tests/test_geospatial.py`) covering the network-independent constants —
  consistent with how Phase 2's dataset download and Phase 9's real-image script are also outside
  the fast pytest suite for their genuinely network-dependent parts.

**FILES CREATED:**
- `src/geospatial/__init__.py`, `src/geospatial/raster.py`
- `scripts/real_world_demo.py`
- `docs/REAL_WORLD_DEMO.md`
- `tests/test_geospatial.py`
- `outputs/real_world_demo/{before,after,predicted_mask,overlay,combined}.png`, `report.json` (gitignored)

**FILES MODIFIED:**
- `requirements.txt` (added `rasterio==1.5.1`, `pystac-client==0.9.0`; documented that
  geopandas/shapely/folium were deliberately not added since no vector export or interactive map
  was actually built — Rule 5, don't introduce unneeded complexity)
- `README.md` (new "Real-World Demonstration" section)

**COMMANDS EXECUTED:**
- `pip install rasterio pystac-client`
- STAC search queries (inline, then via `src/geospatial/raster.py`) to find low-cloud-cover
  scenes and pick a before/after pair
- Ad hoc exploratory fetch + visual inspection of candidate crops (to confirm genuine visible
  change before committing to a location) — then superseded by the reusable
  `scripts/real_world_demo.py`, and the ad hoc exploration files deleted
- `python scripts/real_world_demo.py`
- `pytest tests/ -q` (64 -> 66 tests)

**TESTS:**
- `pytest tests/`: 66/66 passed.
- Manually inspected the actual before/after Sentinel-2 crops (visual confirmation of genuine new
  construction) before treating the location/dates as final — not assumed to show interesting
  change without looking.
- Ran `scripts/real_world_demo.py` end-to-end for real: confirmed it reproduces the same
  quantitative result (1,621/65,536 pixels changed, 19 regions) as the earlier ad hoc exploration
  of the same crop, confirming the formalized script is correct and reproducible.
- Manually inspected `outputs/real_world_demo/combined.png` — confirmed the largest predicted
  region visually overlaps the real, visually-confirmed new building complex.

**RESULTS (actual, measured — this is a prediction, NOT a validated metric; full report:
`outputs/real_world_demo/report.json`):**
```
Location: Pflugerville, TX suburb (bbox [-97.65, 30.41, -97.59, 30.46])
Before: 2019-12-06 (S2A_14RPU_20191206_1_L2A, cloud cover 0.001%)
After:  2024-12-19 (S2A_14RPU_20241219_0_L2A, cloud cover 0.003%)
Resolution gap: Sentinel-2 10 m/pixel vs. LEVIR-CD training data 0.5 m/pixel = 20x coarser

Predicted: 1,621 / 65,536 pixels changed (2.47%), 19 regions (>=4px)
Ground truth available: No — no IoU/Dice/precision/recall/accuracy computed or claimed.
```
Qualitative finding: the model's largest predicted region correctly corresponds to a real,
visually-confirmed new building complex, despite the 20x resolution gap — a genuine positive
signal. Several smaller predicted regions could not be independently confirmed as real changes at
this resolution — reported honestly as uncertain, not claimed as either true or false positives.

**KNOWN ISSUES:**
- **This is one location, one date pair, zero ground-truth-validated metrics.** It must not be
  read as evidence of general real-world reliability — `docs/REAL_WORLD_DEMO.md` says this
  explicitly, and so does this log entry, to guard against the result being cited out of context
  later in this project's own documentation.
- The `src/analysis/area.py::levir_cd_effective_pixel_size` physical-area assumption does not
  apply to Sentinel-2 imagery (different native resolution) — `scripts/real_world_demo.py`
  deliberately does not report a physical area for this reason, rather than misapplying the
  LEVIR-CD-derived assumption.
- No cloud/shadow masking, no explicit re-registration/alignment step beyond Sentinel-2's own
  standard georeferencing — both are real, undemonstrated risk factors for any real-world change
  detection pipeline, noted but not implemented in this phase.

**NEXT PHASE:**
- PHASE 12 — Final Integration & Documentation: write `docs/LIMITATIONS.md` (drawing on every
  limitation surfaced across Phases 2-11 — class imbalance, GPU non-determinism, single-seed
  experiments, the resolution/domain gap just documented here, etc.), verify the complete
  end-to-end workflow (upload → preprocess → model → mask → regions → visualization → dashboard)
  including edge cases (invalid input, mismatched dimensions, missing files, CPU-only fallback),
  and finalize all documentation to describe only what is actually implemented.

---

## PHASE 10 — Streamlit Dashboard

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/visualization/overlays.py`: `create_overlay()` — factored out once a third real consumer
  needed the same before/after/mask compositing logic (`src/evaluation/evaluate.py` and
  `scripts/analyze_predictions.py` had inline versions; the dashboard is the third). 4 new tests.
- `dashboard/app.py`: full Streamlit dashboard built directly on the Phase 9 `Predictor`/
  `src/analysis` pipeline — no new inference or analysis logic, only UI wiring around what was
  already implemented and tested. Model selector (all 5 trained models from Phase 4/5/8), a
  decision-threshold slider, a min-region-size input, real benchmark metrics loaded from
  `outputs/metrics/*_test_metrics.json` for whichever model is selected, before/after image
  upload, real model inference on click, predicted-mask/overlay display, region-count/percent-
  changed/area/largest-region stat tiles, and a full per-region data table. A prominent, explicit
  capability note states what the model does and does not do (binary building-change only, no
  type classification, area assumption not verified for arbitrary uploads) — matching Rule 4.
  Invalid uploaded files are caught and shown as a clear error message, not a stack trace or crash
  (a real, expected user-input boundary case, not speculative error handling — Rule on validating
  at system boundaries).
- **Actually launched and drove the app in a real browser**, per the UI-verification requirement:
  installed Playwright + Chromium (no project-specific run skill existed yet for this repo, and
  `chromium-cli` was not available in this environment, so used Playwright directly per the `run`
  skill's documented fallback), started the Streamlit server, and drove it through: initial page
  load (no console errors), model-selector dropdown interaction with real per-model metrics
  updating correctly, uploading real LEVIR-CD test images
  (`data/raw/levir_cd/test/{A,B}/test_29.png`), clicking "Detect Changes" and confirming real
  inference ran, the region-table expander, and the invalid-file error path.

**FILES CREATED:**
- `src/visualization/overlays.py`
- `dashboard/app.py`
- `tests/test_overlays.py`

**FILES MODIFIED:**
- `README.md` (Dashboard section)

**COMMANDS EXECUTED:**
- `pytest tests/ -q` (60 -> 64 tests)
- `npx playwright install chromium` (in a scratchpad npm project, not the repo)
- `venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.headless true --server.port 8501` (background)
- `curl` poll until the server was actually serving (not a blind sleep)
- 5 Playwright driver scripts (Node.js): initial load + screenshot, upload+inference+screenshot,
  region-table-expander+screenshot, model-switcher+screenshot, invalid-file+screenshot
- `taskkill //PID <streamlit pid> //F` to stop the server after verification

**TESTS:**
- `pytest tests/`: 64/64 passed (60 from Phase 9 + 4 new overlay tests).
- **Real browser verification (not just import-and-typecheck):**
  - Initial load: page renders, sidebar shows real IoU=0.6560 for the default-selected best model
    (Phase 8's Siamese+Attention) — matches `docs/EXPERIMENTS.md` exactly. Zero console/page errors.
  - Upload + inference: uploaded the real `test_29.png` before/after pair, clicked "Detect
    Changes", got a real predicted mask/overlay and stats — **54 regions, 14.97% changed, 3.9240 ha
    total changed area, 4932 m² largest region** — which is an *exact* match to
    `scripts/analyze_predictions.py`'s independently-run output for the same image in Phase 9,
    confirming the dashboard calls the same real pipeline rather than a separate/faked path.
  - Region table expander: renders a real per-region table (pixel_count, area_m2, centroid) for
    all 54 regions.
  - Model switcher: selecting "Baseline U-Net (Phase 4)" updated the sidebar to that model's real
    metrics (IoU=0.6234, Dice=0.7680, Precision=0.7333, Recall=0.8062) — matches
    `DEVELOPMENT_LOG.md` Phase 6's restored-baseline numbers exactly.
  - Invalid file: uploading a non-image file produced the clear error "Could not read 'fake.png'
    — not a valid image file." — no crash, no stack trace shown to the user.

**RESULTS:**
No new model training or metrics this phase — the dashboard surfaces exactly the real numbers
already measured in Phases 4-9, confirmed identical via the browser test above.

**KNOWN ISSUES:**
- No project-specific "how to run this app" skill existed before this phase; per the `run` skill's
  own guidance, this is worth capturing (`/run-skill-generator`) for faster verification in future
  phases (e.g. Phase 11's real-world demonstration will likely reuse this same dashboard).
- The dashboard's area-assumption caveat (LEVIR-CD's 0.5 m/pixel resolution) is stated in the UI
  but not enforced or checked against uploaded images — a user uploading an arbitrarily-scaled
  image gets area numbers computed under a documented-but-unverified assumption. This is
  intentional (documented, not silently assumed) but is a real limitation, not a false claim.
- `dashboard/components/` and `dashboard/utils/` (scaffolded in Phase 0) remain unused — `app.py`
  was not yet complex enough to justify splitting it, per Rule 5 (no premature abstraction).

**NEXT PHASE:**
- PHASE 11 — Real-World Satellite Demonstration: investigate Sentinel-2 as a real-world imagery
  source, explicitly distinguish this from the LEVIR-CD benchmark evaluation (different sensor/
  resolution/domain), and document rather than force a demonstration if Sentinel-2 imagery proves
  unsuitable for the trained model.

---

## PHASE 9 — Change Region Analysis & Quantification

**Date:** 2026-08-23

**STATUS:** COMPLETED

**IMPLEMENTED:**
- `src/analysis/regions.py`: `extract_regions()` — 8-connected-component labeling
  (`scipy.ndimage.label`) of a binary change mask, returning per-region pixel count, centroid, and
  bounding box, sorted largest-first. `min_region_pixels` allows explicit (opt-in, not silent)
  noise filtering.
- `src/analysis/area.py`: `pixel_count_to_area()` never assumes a physical pixel size — callers
  must always pass one explicitly (`PROJECT_CONTEXT.md`'s stated requirement). The one documented
  default this project actually uses, `levir_cd_effective_pixel_size()`, derives the *effective*
  ground pixel size for this project's resized model inputs: LEVIR-CD tiles are 1024px at a
  documented 0.5 m/pixel (512m x 512m fixed ground footprint, Phase 2), but this project's models
  operate on tiles resized to 256px (`docs/ARCHITECTURE.md`) — reusing the raw 0.5 m/pixel figure
  against the resized mask would silently overstate area by 16x. The function makes this
  derivation explicit and testable rather than leaving it as an easy-to-miss caller error.
- `src/analysis/statistics.py`: `compute_change_statistics()` — aggregates region count, total/
  percent changed pixels, largest/average region size, and (only when `pixel_size_meters` is
  passed) physical-area conversions.
- `src/inference/predict.py`: `predict_mask()` (pure function, any `model(before, after)->logits`)
  and `Predictor` (loads a config+checkpoint once, predicts repeatedly — built with Phase 10's
  dashboard in mind, per Rule 6, so that phase doesn't need to duplicate this logic).
- `scripts/analyze_predictions.py`: end-to-end demonstration — loads the current best model
  (Phase 8's Siamese + Attention), runs real inference on 6 real test images, extracts regions,
  computes statistics with the documented effective pixel size, and saves both a JSON report and
  per-sample region-bounding-box visualizations.
- 19 new pytest tests: `tests/test_analysis.py` (10 — region extraction on synthetic masks with
  known geometry: empty mask, single region with exact expected pixel-count/bbox/centroid, two
  regions sorted by size, 8-connectivity diagonal-merge behavior, min-size filtering, area-
  conversion arithmetic, the LEVIR-CD pixel-size derivation at multiple resize targets,
  full-statistics correctness with and without a pixel size); `tests/test_predict.py` (3 —
  `predict_mask` output shape/dtype/binariness, input-size-independent-of-model-size handling,
  threshold sensitivity).

**FILES CREATED:**
- `src/analysis/__init__.py`, `src/analysis/regions.py`, `src/analysis/area.py`,
  `src/analysis/statistics.py`
- `src/inference/__init__.py`, `src/inference/predict.py`
- `scripts/analyze_predictions.py`
- `tests/test_analysis.py`, `tests/test_predict.py`
- `outputs/metrics/region_analysis_demo.json` (gitignored)
- `outputs/visualizations/region_analysis/*.png` (gitignored, 6 real sample visualizations)

**FILES MODIFIED:**
- `README.md` (Inference & Change Analysis section)

**COMMANDS EXECUTED:**
- `pytest tests/ -q` (before and after new code, 47 -> 60 tests)
- `python scripts/analyze_predictions.py --config configs/siamese_attention.yaml --checkpoint outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt`

**TESTS:**
- `pytest tests/`: 60/60 passed (47 from Phase 8 + 13 new: 10 analysis + 3 predict).
- Region-extraction correctness verified against hand-computed expected values on synthetic masks
  with known geometry (e.g. a 3x4-pixel rectangle at a known offset must report exactly 12 pixels,
  the exact bbox, and centroid (3.0, 4.5) — not just "some plausible-looking output").
- Real end-to-end run of `scripts/analyze_predictions.py` against the actual best trained
  checkpoint on real LEVIR-CD test images (not synthetic) — manually inspected 2 of the 6 output
  visualizations: a dense-subdivision scene (54 regions correctly bounding-boxed around real
  building clusters) and a genuinely no-change forest scene with a strong seasonal lighting/
  vegetation difference between before/after (before: brown/dry; after: green/lush) — the model
  correctly predicted only 3 tiny regions (0.05% of the tile), a reassuring real data point for
  `PROJECT_CONTEXT.md`'s "actual change vs. apparent difference" principle, though only one
  anecdotal example, not a systematic robustness evaluation.

**RESULTS (actual, measured — full report: `outputs/metrics/region_analysis_demo.json`):**
```
Effective pixel size for this project's 256px model inputs: 2.000 m/pixel
  (derivation: 1024px * 0.5 m/px / 256px = 2.0 m/px)

test_1.png:   13 regions, 1.34% changed,  3,504.0 m^2 total changed area
test_121.png: 10 regions, 2.50% changed,  6,544.0 m^2 total changed area
test_29.png:  54 regions, 14.97% changed, 39,240.0 m^2 total changed area (dense subdivision)
test_52.png:  56 regions, 4.37% changed,  11,452.0 m^2 total changed area
test_75.png:  37 regions, 7.39% changed,  19,384.0 m^2 total changed area
test_99.png:  3 regions,  0.05% changed,  136.0 m^2 total changed area (no-change scene)
```

**KNOWN ISSUES:**
- Area figures are only as accurate as (a) the model's predicted mask and (b) the documented
  pixel-size assumption — they are not validated against any independent ground-truth area
  measurement, and should be read as illustrative quantification of the model's own predictions,
  not as an externally verified physical measurement.
- `min_region_pixels=4` default in `scripts/analyze_predictions.py` is a chosen noise filter, not
  a principled threshold derived from data — documented as a CLI default, not hidden.
- The seasonal-lighting robustness observation above is one anecdotal example from manual
  inspection, not a systematic evaluation across many such cases — do not generalize from it.

**NEXT PHASE:**
- PHASE 10 — Streamlit Dashboard: build `dashboard/app.py` on top of the now-complete
  `Predictor`/`src/analysis` pipeline — upload before/after images, run real model inference
  (Phase 8's best checkpoint), display the mask/overlay/region statistics computed by the exact
  same code already verified in this phase. No simulated metrics; anything not wired to a real
  model call will be marked unavailable rather than faked, per `DEVELOPMENT_RULES.md`.

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
