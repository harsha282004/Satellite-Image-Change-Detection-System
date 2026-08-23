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
from src.inference.predict import Predictor
from src.visualization.overlays import create_overlay

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (display name -> (config path, checkpoint path, experiment_name)). Every entry here is a model
# that was actually trained and evaluated — see docs/EXPERIMENTS.md for the real comparison.
MODEL_OPTIONS = {
    "Siamese U-Net + Attention (best, Phase 8)": (
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

    threshold = st.slider(
        "Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Sigmoid probability above this value is predicted as 'changed'. "
             "docs/EVALUATION.md's reported metrics use the standard 0.5.",
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
                mask = predictor.predict_from_arrays(before_img, after_img, threshold=threshold)

                import cv2
                size = predictor.image_size
                before_resized = cv2.resize(before_img, (size, size))
                after_resized = cv2.resize(after_img, (size, size))
                overlay = create_overlay(after_resized, mask, color=(1.0, 0.0, 0.0), alpha=0.6)

                pixel_size_m = levir_cd_effective_pixel_size(size)
                stats = compute_change_statistics(
                    mask, pixel_size_meters=pixel_size_m, min_region_pixels=min_region_pixels
                )

            st.header("2. Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(before_resized, caption="Before")
            c2.image(after_resized, caption="After")
            c3.image(mask * 255, caption="Predicted Change Mask")
            c4.image(overlay, caption="Overlay (predicted change = red)")

            st.header("3. Change Statistics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Changed Regions", stats["num_regions"])
            m2.metric("Changed Area (% of tile)", f"{stats['percent_changed']:.2f}%")
            m3.metric("Total Changed Area", f"{stats['total_changed_area']['area_hectares']:.4f} ha")
            m4.metric(
                "Largest Region",
                f"{stats['largest_region_area']['area_m2']:.0f} m²" if stats["num_regions"] else "—",
            )
            st.caption(
                f"Area computed at an assumed {pixel_size_m:.2f} m/pixel "
                f"(src/analysis/area.py::levir_cd_effective_pixel_size — see the capability note above)."
            )

            if stats["regions"]:
                with st.expander(f"All {stats['num_regions']} detected regions"):
                    st.dataframe(
                        [
                            {
                                "region": i + 1,
                                "pixel_count": r["pixel_count"],
                                "area_m2": round(r["pixel_count"] * pixel_size_m**2, 1),
                                "centroid_row": round(r["centroid"][0], 1),
                                "centroid_col": round(r["centroid"][1], 1),
                            }
                            for i, r in enumerate(stats["regions"])
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
| Change-type classification (road/vegetation/water/etc.) | **Not implemented** — no such labels in training data |
| Verified real-world (non-LEVIR-CD) imagery support | **Experimental / not verified** — see `docs/LIMITATIONS.md` and Phase 11 |
| Cloud/shadow/registration-error detection | **Not implemented** |
"""
)
