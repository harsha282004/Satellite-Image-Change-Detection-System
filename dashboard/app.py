"""Phase 23: final unified dashboard. Integrates every prior phase's UI-facing capability into
one cohesive Streamlit app: live upload-and-detect (Phases 9, 15-17, 22), a static real-measured
model comparison (Phases 8, 13, 20), the most recent real geospatial/multi-temporal runs (Phases
18, 21), a documented failure-case reference (Phase 8), and a deterministic (disk-loaded, not
live-recomputed) project summary.

Run with: venv/Scripts/python.exe -m streamlit run dashboard/app.py

Per DEVELOPMENT_RULES.md: this dashboard uses only real trained-model output and real measured
metrics loaded from disk — nothing here is simulated. Unavailable capabilities (change TYPE
classification, verified real-world resolution, multi-class detection) are stated explicitly in
the UI, not implied. The geospatial map and multi-temporal panels show the most recent *real* run
of `scripts/geospatial_analysis.py` / `scripts/multitemporal_analysis.py` (both require live
network access and are not re-run on every dashboard load) — not a live, interactive re-analysis.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.analysis.area import levir_cd_effective_pixel_size
from src.analysis.statistics import compute_change_statistics
from src.analysis.severity import compute_severity_for_regions, highest_severity_regions, severity_distribution
from src.inference.predict import Predictor
from src.realworld.validation import REAL_WORLD_DISCLAIMER, validate_real_world_input
from src.visualization.overlays import create_overlay, create_region_id_overlay

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (display name -> (config path, checkpoint path, experiment_name)). Every entry here is a model
# that was actually trained and evaluated — see docs/EXPERIMENTS.md for the Phase 8 architecture
# comparison (equal 30-epoch budget for all 5) and docs/TRAINING.md / DEVELOPMENT_LOG.md Phase 13
# for the training-strategy experiments (A/B/C below, same architecture, different epoch budgets).
MODEL_OPTIONS = {
    "Siamese U-Net + Attention (best overall, Phase 13 Exp. C)": (
        "configs/siamese_attention_e100.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt",
        "siamese_unet_diff_concat_attention_e100",
    ),
    "Siamese U-Net + Attention (Phase 13 Exp. B, max 60 epochs)": (
        "configs/siamese_attention_e60.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention_e60/best.pt",
        "siamese_unet_diff_concat_attention_e60",
    ),
    "Siamese U-Net + Attention (Phase 8 architecture comparison, 30 epochs)": (
        "configs/siamese_attention.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt",
        "siamese_unet_diff_concat_attention",
    ),
    "Siamese U-Net, diff_concat (Phase 5 primary)": (
        "configs/siamese.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat/best.pt",
        "siamese_unet_diff_concat",
    ),
    "Siamese U-Net, concat (Phase 8 ablation)": (
        "configs/siamese_concat.yaml",
        "outputs/checkpoints/siamese_unet_concat/best.pt",
        "siamese_unet_concat",
    ),
    "Siamese U-Net, diff (Phase 8 ablation)": (
        "configs/siamese_diff.yaml",
        "outputs/checkpoints/siamese_unet_diff/best.pt",
        "siamese_unet_diff",
    ),
    "Baseline U-Net (Phase 4)": (
        "configs/baseline.yaml",
        "outputs/checkpoints/baseline_unet/best.pt",
        "baseline_unet",
    ),
    "Transformer, diff_concat (Phase 20, research comparison — underperforms)": (
        "configs/transformer.yaml",
        "outputs/checkpoints/transformer_change_diff_concat/best.pt",
        "transformer_change_diff_concat",
    ),
}


@st.cache_resource
def load_predictor(config_path: str, checkpoint_path: str) -> Predictor:
    return Predictor(str(PROJECT_ROOT / config_path), str(PROJECT_ROOT / checkpoint_path))


def load_test_metrics(experiment_name: str) -> dict | None:
    path = PROJECT_ROOT / "outputs" / "metrics" / f"{experiment_name}_test_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_image_from_upload(uploaded_file) -> np.ndarray | None:
    try:
        return np.array(Image.open(uploaded_file).convert("RGB"))
    except UnidentifiedImageError:
        st.error(f"Could not read '{uploaded_file.name}' — not a valid image file.")
        return None


def load_selected_threshold(experiment_name: str) -> tuple:
    """Returns (threshold, source_note). Only applies the Phase 15.2 validation-optimized
    threshold when the currently selected model matches the checkpoint that threshold was
    actually swept for — every other model falls back to the untuned default 0.5, rather than
    silently applying a threshold optimized for a different model's outputs."""
    report_path = PROJECT_ROOT / "outputs" / "metrics" / "threshold_optimization_report.json"
    if not report_path.exists():
        return 0.5, "default (no threshold optimization run yet)"
    with open(report_path) as f:
        report = json.load(f)
    if experiment_name in report.get("checkpoint", ""):
        return (
            report["selected_threshold"],
            f"selected via validation-set sweep (Phase 15.2), by max validation IoU",
        )
    return 0.5, "default (threshold optimization was only run for the best model)"


def probability_map_to_rgb(prob_map: np.ndarray) -> np.ndarray:
    """Viridis-colored visualization of a [0,1] probability map, for display only."""
    import matplotlib

    colored = matplotlib.colormaps["viridis"](prob_map)  # (H, W, 4) float in [0,1]
    return (colored[:, :, :3] * 255).astype(np.uint8)


st.set_page_config(page_title="Satellite Change Detection", layout="wide")

st.title("Satellite Image Change Detection")
st.caption(
    "Academic Deep Learning project — Siamese U-Net with Attention for LEVIR-CD building-change "
    "detection, extended with research-grade change intelligence (geospatial, multi-temporal, "
    "severity, architecture comparison). See PROJECT_CONTEXT.md, docs/EVALUATION.md, "
    "docs/EXPERIMENTS.md, and docs/LIMITATIONS.md for full detail behind every panel below."
)
st.info(
    "**Scientific disclaimer, applying to this entire dashboard:** all benchmark metrics shown "
    "are measured on the held-out LEVIR-CD test split only. Predictions on any image you upload "
    "are unvalidated — no ground truth exists for arbitrary imagery. Severity scores are "
    "analytical, not ground truth. Multi-temporal intervals are independent detections, never a "
    "tracked trend. See docs/LIMITATIONS.md for the complete, itemized account.",
    icon="🔬",
)

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
    config_path, checkpoint_path, experiment_name = MODEL_OPTIONS[model_name]
    checkpoint_full_path = PROJECT_ROOT / checkpoint_path
    checkpoint_exists = checkpoint_full_path.exists()

    if not checkpoint_exists:
        st.error(
            f"Checkpoint not found at `{checkpoint_path}`. This model has not been trained in "
            f"this environment yet — see README.md 'Training' to train it, or select a different "
            f"model above."
        )

    default_threshold, threshold_source = load_selected_threshold(experiment_name)
    threshold = st.slider(
        "Decision threshold", min_value=0.0, max_value=1.0, value=default_threshold, step=0.05,
        help="Prediction probability above this value is predicted as 'changed'. "
             f"Default for this model: {threshold_source}. See docs/EVALUATION.md Phase 15.2 "
             "for the full validation-set threshold sweep (outputs/metrics/threshold_analysis.csv).",
    )
    min_region_pixels = st.number_input(
        "Minimum region size (pixels)", min_value=1, value=4,
        help="Predicted regions smaller than this are discarded as noise before counting/area "
             "statistics — does not affect the displayed mask itself.",
    )

    st.divider()
    st.subheader("Benchmark performance (real, measured)")
    metrics = load_test_metrics(experiment_name)
    if metrics:
        tm = metrics["test_metrics"]
        st.caption(
            f"Measured on the held-out LEVIR-CD benchmark test set (128 images) — "
            f"NOT on whatever image you upload below. See docs/EVALUATION.md."
        )
        st.metric("IoU", f"{tm['iou']:.4f}")
        c1, c2 = st.columns(2)
        c1.metric("Dice", f"{tm['dice']:.4f}")
        c2.metric("F1", f"{tm['f1']:.4f}")
        c1.metric("Precision", f"{tm['precision']:.4f}")
        c2.metric("Recall", f"{tm['recall']:.4f}")
    else:
        st.info("No benchmark test-metrics file found for this model.")

tab_overview, tab_detect, tab_compare, tab_geo, tab_failures = st.tabs([
    "Project Overview", "Live Detection", "Model Comparison",
    "Geospatial & Multi-Temporal", "Failure Cases & Limitations",
])

# ---------------------------------------------------------------------------
# Tab: Project Overview — a deterministic summary loaded from saved metrics files, not
# live-recomputed on every page load, so it never drifts from what evaluate.py actually measured.
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("Deterministic Project Summary")
    st.caption(
        "\"Deterministic\" here means: every number below is loaded from a saved "
        "`outputs/metrics/*.json` file produced by a real training/evaluation run — nothing on "
        "this tab is recomputed live or subject to per-session randomness."
    )

    best_metrics = load_test_metrics("siamese_unet_diff_concat_attention_e100")
    c1, c2, c3, c4 = st.columns(4)
    if best_metrics:
        tm = best_metrics["test_metrics"]
        c1.metric("Best model test IoU", f"{tm['iou']:.4f}")
        c2.metric("Best model test Dice", f"{tm['dice']:.4f}")
        c3.metric("Best model test F1", f"{tm['f1']:.4f}")
        c4.metric("Best model test Accuracy", f"{tm['accuracy']:.4f}")
    st.caption(
        "Best model: Siamese U-Net + Attention, `diff_concat`, 100-epoch budget with early "
        "stopping + LR scheduler (Phase 13 Experiment C) — checkpoint epoch 68. A +0.0563 "
        "absolute IoU improvement over the original Phase 8 result (0.6560), from training "
        "strategy alone. See docs/TRAINING.md Phase 13."
    )

    st.subheader("Phases completed")
    st.markdown(
        """
This project was built in two mega-phases. **Phases 0-12** (dataset, baseline, Siamese U-Net,
attention, dashboard, real-world demo, documentation) established the core system. **Phases
13-23** (this dashboard is Phase 23) extended it into a research-grade change-intelligence system:

| Phase | What it added | Status |
|---|---|---|
| 13 | Advanced training strategy (early stopping, LR scheduler, longer budget) | Done — new best model |
| 14 | Loss function & hyperparameter experiments | Done — BCE+Dice confirmed best |
| 15 | Prediction probability, threshold optimization, robustness testing | Done |
| 16 | Region-level intelligence (geometry, per-region probability) | Done |
| 17 | Change severity analysis (analytical, not ground truth) | Done |
| 18 | Geospatial intelligence (real polygons/area/GeoJSON/map, real Sentinel-2) | Done |
| 19 | Multi-class change detection | **Not implemented** — no suitable dataset reliably obtainable this session (see docs/LIMITATIONS.md) |
| 20 | Transformer architecture research comparison | Done — reported honestly, underperforms CNN |
| 21 | Multi-temporal (>2 date) analysis | Done — independent intervals, no tracking claims |
| 22 | Real-world pipeline hardening (input validation) | Done |
| 23 | This unified dashboard | Done |
        """
    )
    st.caption("Full detail for every phase: `DEVELOPMENT_LOG.md`.")

# ---------------------------------------------------------------------------
# Tab: Live Detection — upload a pair, run real inference (Phases 9, 15-17, 22).
# ---------------------------------------------------------------------------
with tab_detect:
    st.header("1. Upload Images")
    st.info(
        "**Capability note:** this model was trained only on LEVIR-CD (Google Earth imagery, "
        "1024x1024 tiles, 0.5 m/pixel, USA building-change scenes). It predicts a **binary "
        "building-change mask only** — it does NOT classify change type (road/vegetation/water/etc.), "
        "since the training data has no such labels (see docs/DATASET.md, docs/PROJECT_CONTEXT.md "
        "capability-honesty rules). Area statistics below assume your uploaded images match LEVIR-CD's "
        "resolution; this is **not verified** for arbitrary uploads — treat area figures as "
        "illustrative unless you know your images match that resolution. See docs/LIMITATIONS.md.",
        icon="ℹ️",
    )

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    with col2:
        after_file = st.file_uploader("After image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

    if before_file and after_file:
        before_img = load_image_from_upload(before_file)
        after_img = load_image_from_upload(after_file)

        if before_img is not None and after_img is not None:
            st.warning(f"**{REAL_WORLD_DISCLAIMER}**", icon="⚠️")
            validation_report = validate_real_world_input(before_img, after_img)
            if validation_report["warnings"]:
                with st.expander(
                    f"Input validation: {len(validation_report['warnings'])} warning(s) found "
                    f"(Phase 22 — real, computed checks, not a validated accuracy estimate)",
                    expanded=True,
                ):
                    for w in validation_report["warnings"]:
                        st.write(f"- {w}")
                    reg = validation_report["registration"]
                    if reg["estimated_shift_magnitude_px"] is not None:
                        st.caption(
                            f"Estimated registration offset: {reg['estimated_shift_magnitude_px']} px "
                            f"(phase-correlation estimate, not corrected)."
                        )
            else:
                st.caption(
                    "Input validation (Phase 22): dimensions match, no large estimated registration "
                    "offset, no unusually bright/washed-out regions detected. This does NOT confirm "
                    "prediction accuracy — see the disclaimer above."
                )

            if not checkpoint_exists:
                st.warning("Cannot run inference: the selected model's checkpoint is missing (see sidebar).")
            elif st.button("Detect Changes", type="primary"):
                with st.spinner("Running inference..."):
                    predictor = load_predictor(config_path, checkpoint_path)
                    prob_map = predictor.predict_probability_from_arrays(before_img, after_img)
                    mask = (prob_map > threshold).astype(np.uint8)

                    import cv2
                    size = predictor.image_size
                    before_resized = cv2.resize(before_img, (size, size))
                    after_resized = cv2.resize(after_img, (size, size))
                    overlay = create_overlay(after_resized, mask, color=(1.0, 0.0, 0.0), alpha=0.6)
                    prob_rgb = probability_map_to_rgb(prob_map)

                    pixel_size_m = levir_cd_effective_pixel_size(size)
                    stats = compute_change_statistics(
                        mask, probability_map=prob_map, pixel_size_meters=pixel_size_m,
                        min_region_pixels=min_region_pixels
                    )

                st.header("2. Results")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.image(before_resized, caption="Before")
                c2.image(after_resized, caption="After")
                c3.image(prob_rgb, caption="Prediction Probability")
                c4.image(mask * 255, caption="Predicted Change Mask")
                c5.image(overlay, caption="Overlay (predicted change = red)")
                st.caption(
                    "**Prediction Probability** = `sigmoid(model output)`, per pixel — brighter "
                    "(yellow) means the model output a higher probability of change at that pixel, "
                    "darker (purple) means lower. This is **not** a calibrated confidence score (i.e. "
                    "a pixel at 0.8 is not verified to be correct ~80% of the time) — no calibration "
                    "study has been run on this model. See docs/EVALUATION.md Phase 15."
                )

                st.header("3. Change Statistics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Changed Regions", stats["num_regions"])
                m2.metric("Changed Area (% of tile)", f"{stats['percent_changed']:.2f}%")
                m3.metric("Total Changed Area", f"{stats['total_changed_area']['area_hectares']:.4f} ha")
                m4.metric(
                    "Largest Region",
                    f"{stats['largest_region_area']['area_m2']:.0f} m²" if stats["num_regions"] else "—",
                )
                m5.metric(
                    "Smallest Region",
                    f"{stats['smallest_region_area']['area_m2']:.0f} m²" if stats["num_regions"] else "—",
                )
                st.caption(
                    f"Area computed at an assumed {pixel_size_m:.2f} m/pixel "
                    f"(src/analysis/area.py::levir_cd_effective_pixel_size — see the capability note above). "
                    f"Regions below {min_region_pixels}px are excluded as noise (sidebar setting)."
                )

                st.header("4. Region Analysis")
                st.caption(
                    "Every detected region is labeled **\"Detected Change Region\"** — this model was "
                    "trained only on binary building-change labels, so it has no basis to classify "
                    "regions as e.g. \"Building\" or \"Road\" (see docs/DATASET.md)."
                )
                if stats["regions"]:
                    scored_regions = compute_severity_for_regions(stats["regions"])

                    region_id_overlay = create_region_id_overlay(after_resized, stats["regions"])
                    st.image(region_id_overlay, caption="Regions labeled by ID (matches table below)", width=500)

                    region_table = [
                        {
                            "Region": r["id"],
                            "Area (px)": r["pixel_count"],
                            "Area (m²)": round(r["pixel_count"] * pixel_size_m**2, 1),
                            "Prediction Probability (mean)": round(r.get("mean_prediction_probability", float("nan")), 3),
                            "Prediction Probability (max)": round(r.get("max_prediction_probability", float("nan")), 3),
                            "Bounding Box (row, col)": f"({r['bbox']['min_row']}-{r['bbox']['max_row']}, "
                                                        f"{r['bbox']['min_col']}-{r['bbox']['max_col']})",
                            "Width x Height (px)": f"{r['width']} x {r['height']}",
                            "Severity Score": round(r["severity_score"], 1),
                            "Severity Category": r["severity_category"],
                        }
                        for r in scored_regions
                    ]
                    with st.expander(f"All {stats['num_regions']} detected regions", expanded=True):
                        st.dataframe(region_table, width="stretch")

                    st.subheader("5. Exports")
                    st.caption(
                        "Exports the region table above (image-space pixel coordinates, no CRS — "
                        "these images have no geographic reference). For a real geographic GeoJSON "
                        "export with true coordinates and area, see the 'Geospatial & Multi-Temporal' "
                        "tab, which uses actual georeferenced Sentinel-2 imagery (Phase 18)."
                    )
                    exp_col1, exp_col2 = st.columns(2)
                    exp_col1.download_button(
                        "Download region table (CSV)",
                        data="\n".join(
                            [",".join(region_table[0].keys())]
                            + [",".join(str(v) for v in row.values()) for row in region_table]
                        ) if region_table else "",
                        file_name="detected_regions.csv", mime="text/csv",
                    )
                    exp_col2.download_button(
                        "Download full statistics (JSON)",
                        data=json.dumps({k: v for k, v in stats.items() if k != "regions"} |
                                        {"regions": region_table}, indent=2, default=str),
                        file_name="change_statistics.json", mime="application/json",
                    )

                    st.subheader("Severity Distribution")
                    st.caption(
                        "**Severity is an analytical score derived from measurable model outputs "
                        "(region size, the model's own prediction probability, region density/shape, "
                        "and relative size within this image) — it is NOT ground truth, NOT a "
                        "physical damage assessment, and has not been validated against any labeled "
                        "severity data.** Formula and full disclaimer: `src/analysis/severity.py`, "
                        "docs/EVALUATION.md Phase 17."
                    )
                    dist = severity_distribution(scored_regions)
                    dist_cols = st.columns(4)
                    for i, cat in enumerate(["Low", "Moderate", "High", "Very High"]):
                        dist_cols[i].metric(
                            cat,
                            dist["region_count_by_category"].get(cat, 0),
                            help=f"{dist['changed_pixels_by_category'].get(cat, 0)} changed pixels in this category",
                        )

                    top_regions = highest_severity_regions(scored_regions, n=5)
                    with st.expander(f"Top {len(top_regions)} highest-severity regions"):
                        st.dataframe(
                            [
                                {"Region": r["id"], "Severity Score": round(r["severity_score"], 1),
                                 "Category": r["severity_category"], "Area (px)": r["pixel_count"]}
                                for r in top_regions
                            ],
                            width="stretch",
                        )
                else:
                    st.write("No changed regions detected above the minimum region size.")
    else:
        st.info("Upload both a before and after image to begin.")

# ---------------------------------------------------------------------------
# Tab: Model Comparison — real measured results from Phases 8, 13, 20 (static, disk-loaded).
# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Architecture Comparison (Phase 8 + Phase 20)")
    st.caption(
        "All 6 architectures trained under the identical controlled protocol (30 epochs, Adam "
        "lr=1e-4, BCE+Dice loss, batch size 8, seed 42) for a fair comparison. Parameters and "
        "inference time measured together, one identical procedure, in Phase 20 "
        "(`scripts/architecture_comparison.py`). Full interpretation: docs/EXPERIMENTS.md."
    )
    arch_comparison = load_json(PROJECT_ROOT / "outputs" / "metrics" / "architecture_comparison.json")
    if arch_comparison:
        st.dataframe(
            [
                {
                    "Architecture": r["name"],
                    "Parameters": f"{r['parameters']:,}",
                    "Inference (ms/pair)": r["inference_ms_per_pair"],
                    "Test IoU": r["iou"],
                    "Test Dice": r["dice"],
                    "Test Precision": r["precision"],
                    "Test Recall": r["recall"],
                    "Test F1": r["f1"],
                    "Test Accuracy": r["accuracy"],
                }
                for r in arch_comparison["results"]
            ],
            width="stretch",
        )
        st.caption(
            "**Honest result:** the Transformer variant (Phase 20) has the fewest parameters and "
            "fastest inference, but substantially underperforms every CNN architecture — reported "
            "as measured, not tuned to look better. See docs/EXPERIMENTS.md Phase 20."
        )
    else:
        st.info("`outputs/metrics/architecture_comparison.json` not found — run "
                "`scripts/architecture_comparison.py` to generate it.")

    st.header("Training Strategy Comparison (Phase 13)")
    st.markdown(
        """
| Experiment | Max epochs | Actual epochs | Best epoch | Test IoU | Test Dice |
|---|---|---|---|---|---|
| A — original (Phase 8, fixed 30-epoch budget) | 30 | 30 | 26 | 0.6560 | 0.7922 |
| B — longer training (no early stop) | 60 | 60 | 60 | 0.7031 | 0.8257 |
| **C — longer + early stopping (current best model)** | 100 | 78 | 68 | **0.7123** | **0.8320** |
        """
    )
    st.caption(
        "Same architecture, data, optimizer, and seed as Experiment A — the entire improvement "
        "comes from training strategy (longer budget + early stopping + LR scheduler), not "
        "architecture changes. See docs/TRAINING.md and DEVELOPMENT_LOG.md Phase 13."
    )

# ---------------------------------------------------------------------------
# Tab: Geospatial & Multi-Temporal — most recent real runs (Phases 18, 21), disk-loaded.
# ---------------------------------------------------------------------------
with tab_geo:
    st.header("Geospatial Change Intelligence (Phase 18)")
    st.caption(
        "Shows the most recent real run of `scripts/geospatial_analysis.py` against live "
        "Sentinel-2 imagery (Pflugerville, TX). This is NOT re-run live by this dashboard — it "
        "requires network access to Earth Search/AWS Open Data. Inherits every caveat from "
        "docs/REAL_WORLD_DEMO.md: 10 m/pixel Sentinel-2 vs. 0.5 m/pixel training data, no ground "
        "truth for this real-world pair."
    )
    geojson_path = PROJECT_ROOT / "outputs" / "geospatial" / "regions.geojson"
    map_path = PROJECT_ROOT / "outputs" / "geospatial" / "region_map.html"
    geojson_data = load_json(geojson_path)
    if geojson_data and map_path.exists():
        st.caption(f"{len(geojson_data.get('features', []))} real detected region(s), real "
                   f"geographic polygons in WGS84 (converted from the raster's actual UTM CRS).")
        st.iframe(map_path, height=500)
        st.download_button(
            "Download regions (GeoJSON)", data=json.dumps(geojson_data, indent=2),
            file_name="regions.geojson", mime="application/geo+json",
        )
    else:
        st.info("No geospatial run found. Run `scripts/geospatial_analysis.py` (requires network "
                "access) to generate `outputs/geospatial/regions.geojson` and `region_map.html`.")

    st.divider()
    st.header("Multi-Temporal Change Analysis (Phase 21)")
    st.warning(
        "**No causal or tracking claims:** each interval below is an independent two-image "
        "detection. A region flagged in one interval is never the same tracked event as a region "
        "in another interval — the model has no cross-image tracking mechanism.",
        icon="⚠️",
    )
    temporal_report = load_json(PROJECT_ROOT / "outputs" / "multitemporal" / "temporal_report.json")
    temporal_chart = PROJECT_ROOT / "outputs" / "multitemporal" / "temporal_change_area.png"
    if temporal_report:
        st.caption(
            f"Selected dates: {', '.join(temporal_report['selected_dates'])} "
            f"(Pflugerville, TX, searched {temporal_report['date_range_searched']})"
        )
        st.dataframe(
            [
                {
                    "From": iv["from_date"], "To": iv["to_date"],
                    "Regions": iv["num_regions"],
                    "Changed pixels": iv["total_changed_pixels"],
                    "% changed": round(iv["percent_changed"], 3),
                    "Area (ha)": iv.get("total_changed_area", {}).get("area_hectares"),
                }
                for iv in temporal_report["intervals"]
            ],
            width="stretch",
        )
        if temporal_chart.exists():
            st.image(str(temporal_chart), caption="Per-interval change area and region count "
                                                    "(independent detections, not a tracked trend)")
    else:
        st.info("No multi-temporal run found. Run `scripts/multitemporal_analysis.py` (requires "
                "network access) to generate `outputs/multitemporal/temporal_report.json`.")

# ---------------------------------------------------------------------------
# Tab: Failure Cases & Limitations — a documented real finding, not a live search.
# ---------------------------------------------------------------------------
with tab_failures:
    st.header("Documented Failure Case (Phase 8)")
    st.markdown(
        """
The Phase 8 30-epoch Siamese U-Net + Attention model's real test-set prediction grid showed
generally strong agreement with ground truth, but also a genuine failure case: **one no-change
test scene produced a small cluster of false-positive predictions** with no counterpart in the
ground truth or in any other model's output on that same scene — a concrete example of the
false-positive risk `PROJECT_CONTEXT.md`'s "actual change vs. apparent difference" principle warns
about. This is not hidden despite being an imperfection in the (at the time) best-performing model.
See `docs/EXPERIMENTS.md` "Qualitative note".
        """
    )
    failure_viz = PROJECT_ROOT / "outputs" / "visualizations" / "siamese_unet_diff_concat_attention_e100_test_predictions.png"
    if failure_viz.exists():
        st.image(str(failure_viz), caption="Real prediction grid, current best model "
                                            "(Phase 13 Experiment C) — for visual inspection, not "
                                            "cherry-picked to show only successes.")
        st.caption(
            "Note: the specific false-positive failure case documented in Phase 8 was observed on "
            "the 30-epoch Phase 8 checkpoint. It is not separately re-confirmed here for the "
            "current 100-epoch best model (Experiment C) — shown above for direct visual inspection "
            "instead of an unverified claim of an identical failure."
        )
    else:
        st.info("Prediction grid image not found on disk.")

    st.divider()
    st.header("Full Limitations")
    st.caption(
        "This dashboard surfaces the most relevant caveats inline, next to each result. The "
        "complete, itemized account — dataset, model/training, evaluation, domain-gap, multi-class "
        "(Phase 19), and real-world-validation limitations — is in `docs/LIMITATIONS.md`, which "
        "every quantitative claim in this project traces back to."
    )

st.divider()
st.header("Capabilities")
st.markdown(
    """
| Capability | Status |
|---|---|
| Binary building-change detection (LEVIR-CD-trained) | **Implemented** |
| Change region count / pixel & area quantification | **Implemented** |
| Multiple trained models, selectable (incl. Transformer research comparison) | **Implemented** |
| Real, measured benchmark metrics display | **Implemented** |
| Prediction probability visualization | **Implemented** — raw sigmoid output, not calibrated confidence |
| Threshold optimization (validation-set sweep) | **Implemented** — see `docs/EVALUATION.md` Phase 15.2 |
| Region-level intelligence (geometry, prediction probability per region, region-ID overlay) | **Implemented** — see `docs/EVALUATION.md` Phase 16 |
| Region severity scoring | **Implemented** — analytical score only, NOT ground truth (see `src/analysis/severity.py`, docs/EVALUATION.md Phase 17) |
| Change-type classification (road/vegetation/water/etc.) | **Not implemented** — no such labels in training data |
| Geospatial analysis (real polygons, area, GeoJSON/map export) | **Implemented** — real georeferenced Sentinel-2 imagery only, see `docs/EVALUATION.md` Phase 18 |
| Multi-temporal (>2 date) analysis | **Implemented** — independent per-interval detections, no tracking/causal claims, see `docs/EVALUATION.md` Phase 21 |
| Multi-class / damage-severity change detection | **Not implemented** — no suitable dataset could be reliably obtained this session, see `docs/LIMITATIONS.md` Phase 19 |
| Transformer-based architecture | **Implemented as a research comparison only** — underperforms the CNN model, see `docs/EXPERIMENTS.md` Phase 20 |
| Verified real-world (non-LEVIR-CD) imagery support | **Experimental / not verified** — see `docs/LIMITATIONS.md` and Phase 11 |
| Upload-time input validation (dimension match, estimated registration offset, bright/cloud-like-pixel heuristic) | **Implemented** — real, computed diagnostic checks; NOT a validated accuracy estimate, see Phase 22 |
| Cloud/shadow detection | **Heuristic only** — a simple brightness/saturation screen, not a validated cloud detector (Phase 22) |
| Region-table (CSV/JSON) and geospatial GeoJSON export | **Implemented** — see "Live Detection" tab exports and "Geospatial & Multi-Temporal" tab downloads |
| Formal probability calibration (e.g. temperature scaling) | **Not implemented** — "prediction probability" is a raw sigmoid output only |
"""
)
