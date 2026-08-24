"""Streamlit dashboard: upload a before/after satellite image pair, run the actual trained model,
and display the real predicted change mask, overlay, and region/area statistics.

Run with: venv/Scripts/python.exe -m streamlit run dashboard/app.py

Per DEVELOPMENT_RULES.md: this dashboard uses only real trained-model output and real measured
benchmark metrics loaded from disk — nothing here is simulated. Unavailable capabilities (change
TYPE classification, verified real-world resolution) are stated explicitly in the UI, not implied.
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
    "Academic Deep Learning project (see PROJECT_CONTEXT.md). Demonstration layer on top of a "
    "trained Siamese U-Net — see docs/EVALUATION.md and docs/EXPERIMENTS.md for full benchmark "
    "results, and docs/LIMITATIONS.md for what this system does not do."
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

                with st.expander(f"All {stats['num_regions']} detected regions", expanded=True):
                    st.dataframe(
                        [
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
                        ],
                        width="stretch",
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

st.divider()
st.header("Capabilities")
st.markdown(
    """
| Capability | Status |
|---|---|
| Binary building-change detection (LEVIR-CD-trained) | **Implemented** |
| Change region count / pixel & area quantification | **Implemented** |
| Multiple trained models, selectable | **Implemented** |
| Real, measured benchmark metrics display | **Implemented** |
| Prediction probability visualization | **Implemented** — raw sigmoid output, not calibrated confidence |
| Threshold optimization (validation-set sweep) | **Implemented** — see `docs/EVALUATION.md` Phase 15.2 |
| Region-level intelligence (geometry, prediction probability per region, region-ID overlay) | **Implemented** — see `docs/EVALUATION.md` Phase 16 |
| Region severity scoring | **Implemented** — analytical score only, NOT ground truth (see `src/analysis/severity.py`, docs/EVALUATION.md Phase 17) |
| Change-type classification (road/vegetation/water/etc.) | **Not implemented** — no such labels in training data |
| Verified real-world (non-LEVIR-CD) imagery support | **Experimental / not verified** — see `docs/LIMITATIONS.md` and Phase 11 |
| Cloud/shadow/registration-error detection | **Not implemented** |
| Formal probability calibration (e.g. temperature scaling) | **Not implemented** — "prediction probability" is a raw sigmoid output only |
"""
)
