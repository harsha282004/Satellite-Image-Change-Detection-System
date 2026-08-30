"""Change Detection — the main interactive workflow: upload a before/after pair, run the real
trained model, and explore the detected changes. Every number shown here comes from the actual
model output for the uploaded pair — nothing is simulated or precomputed.
"""
import json

import cv2
import numpy as np
import streamlit as st

from data import load_image_from_upload, load_predictor, probability_map_to_rgb
from src.analysis.area import levir_cd_effective_pixel_size
from src.analysis.severity import compute_severity_for_regions, highest_severity_regions, severity_distribution
from src.analysis.statistics import compute_change_statistics
from src.realworld.validation import REAL_WORLD_DISCLAIMER, validate_real_world_input
from src.visualization.overlays import create_overlay, create_region_id_overlay
from theme import ICONS, empty_state, info_banner, kpi_row, section_header, status_badge

section_header("Change detection", "Upload two images of the same area, taken at different times.", ICONS["detect"])

sel = st.session_state.get("model_selection")
threshold = st.session_state.get("detection_threshold", 0.5)
min_region_pixels = st.session_state.get("min_region_pixels", 4)


def _describe(img: np.ndarray, upload) -> str:
    h, w = img.shape[:2]
    size_bytes = getattr(upload, "size", None)
    size_str = f", {size_bytes / 1024:.0f} KB" if size_bytes else ""
    return f"{w} × {h} px{size_str}"


col1, col2 = st.columns(2)
with col1:
    with st.container(border=True, key="card-upload-before"):
        st.markdown(f"**{ICONS['upload']} Before image**")
        before_file = st.file_uploader(
            "Before image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            label_visibility="collapsed", key="before_upload",
        )
        before_img = load_image_from_upload(before_file) if before_file else None
        if before_file and before_img is None:
            st.error("Could not read this file — not a valid image.", icon=ICONS["error"])
        elif before_img is not None:
            st.image(before_img, width="stretch")
            st.caption(f"{before_file.name} · {_describe(before_img, before_file)}")
with col2:
    with st.container(border=True, key="card-upload-after"):
        st.markdown(f"**{ICONS['upload']} After image**")
        after_file = st.file_uploader(
            "After image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            label_visibility="collapsed", key="after_upload",
        )
        after_img = load_image_from_upload(after_file) if after_file else None
        if after_file and after_img is None:
            st.error("Could not read this file — not a valid image.", icon=ICONS["error"])
        elif after_img is not None:
            st.image(after_img, width="stretch")
            st.caption(f"{after_file.name} · {_describe(after_img, after_file)}")

if before_img is None or after_img is None:
    st.space("medium")
    empty_state(
        "🛰️", "No imagery loaded",
        "Upload a before and after satellite image above to begin analysis.",
    )
    st.session_state.pop("detection_result", None)
    st.stop()

current_upload_ids = (before_file.file_id, after_file.file_id)
if st.session_state.get("detection_result", {}).get("source_ids") != current_upload_ids:
    # A new image pair was uploaded — any previously computed result no longer applies to it.
    st.session_state.pop("detection_result", None)

validation_report = validate_real_world_input(before_img, after_img)
dims_ok = validation_report["dimensions"]["dimensions_match"]

st.space("small")
if not dims_ok:
    with st.container(border=True, key="card-compat-error"):
        st.markdown(f"**{ICONS['error']} Image compatibility error**")
        st.write(
            "The uploaded images have incompatible dimensions "
            f"({validation_report['dimensions']['before_shape']} vs. "
            f"{validation_report['dimensions']['after_shape']}). Please upload two images with "
            "matching dimensions."
        )
        with st.expander("Technical details"):
            st.code(validation_report["dimensions"]["warning"])
    st.stop()
else:
    with st.container(horizontal=True):
        status_badge("Dimensions compatible", "success")
        if validation_report["registration"].get("likely_misregistered"):
            status_badge("Possible misalignment detected", "warning")
        if any("bright" in (w or "") for w in validation_report["warnings"]):
            status_badge("Possible cloud/overexposure", "warning")
        if not validation_report["warnings"]:
            status_badge("No input issues detected", "success")

with st.expander("Validation details"):
    for w in validation_report["warnings"] or ["No issues detected."]:
        st.write(f"- {w}")

st.caption(REAL_WORLD_DISCLAIMER)

with st.expander("Advanced settings", icon=":material/tune:"):
    st.caption(f"Model: **{sel['display_name']}** · Threshold: **{threshold}** · "
               f"Minimum region size: **{min_region_pixels} px**")
    st.caption("Change these in the sidebar.")

run_disabled = not sel["checkpoint_exists"]
run = st.button("Run change detection", type="primary", disabled=run_disabled, width="stretch")
if run_disabled:
    st.error("Cannot run inference: the selected model's checkpoint is unavailable.", icon=ICONS["error"])

if not run and "detection_result" not in st.session_state:
    st.stop()

if run:
    try:
        with st.status("Analyzing satellite imagery", expanded=True) as status:
            st.write("Preprocessing imagery")
            predictor = load_predictor(sel["config_path"], sel["checkpoint_path"])
            size = predictor.image_size
            before_resized = cv2.resize(before_img, (size, size))
            after_resized = cv2.resize(after_img, (size, size))

            st.write("Running neural network inference")
            prob_map = predictor.predict_probability_from_arrays(before_img, after_img)

            st.write("Generating change mask")
            mask = (prob_map > threshold).astype(np.uint8)

            st.write("Extracting change regions")
            pixel_size_m = levir_cd_effective_pixel_size(size)
            stats = compute_change_statistics(
                mask, probability_map=prob_map, pixel_size_meters=pixel_size_m,
                min_region_pixels=min_region_pixels,
            )

            st.write("Preparing analysis")
            scored_regions = compute_severity_for_regions(stats["regions"]) if stats["regions"] else []
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.detection_result = {
            "before_resized": before_resized, "after_resized": after_resized,
            "prob_map": prob_map, "mask": mask, "stats": stats,
            "scored_regions": scored_regions, "pixel_size_m": pixel_size_m,
            "source_ids": current_upload_ids,
        }
    except Exception as exc:  # noqa: BLE001 — surfaced as a friendly card, not a raw traceback
        with st.container(border=True, key="card-inference-error"):
            st.markdown(f"**{ICONS['error']} Processing error**")
            st.write("The model could not process this image pair.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
        st.stop()

result = st.session_state.detection_result
before_resized, after_resized = result["before_resized"], result["after_resized"]
prob_map, mask, stats = result["prob_map"], result["mask"], result["stats"]
scored_regions, pixel_size_m = result["scored_regions"], result["pixel_size_m"]

prob_rgb = probability_map_to_rgb(prob_map)

st.space("large")
section_header("Detection results")

view_mode = st.segmented_control(
    "View", ["Side by side", "Overlay", "Probability heatmap"], default="Side by side",
    label_visibility="collapsed",
)
if view_mode == "Overlay":
    opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.6, 0.05)
    overlay = create_overlay(after_resized, mask, color=(1.0, 0.0, 0.0), alpha=opacity)
    st.image(overlay, caption="Detected change overlaid on the after image", width="stretch")
elif view_mode == "Probability heatmap":
    st.image(prob_rgb, caption="Prediction probability (raw model output, not a calibrated confidence score)", width="stretch")
else:
    overlay = create_overlay(after_resized, mask, color=(1.0, 0.0, 0.0), alpha=0.6)
    cols = st.columns(4)
    for col, (img, caption) in zip(cols, [
        (before_resized, "Before"), (after_resized, "After"),
        (mask * 255, "Change mask"), (overlay, "Overlay"),
    ]):
        with col:
            st.image(img, width="stretch")
            st.caption(caption)

st.space("large")
section_header("Detection summary")
avg_prob = (
    round(100 * np.mean([r.get("mean_prediction_probability", 0.0) for r in stats["regions"]]), 1)
    if stats["regions"] else None
)
kpi_row([
    {"label": "Detected regions", "value": stats["num_regions"]},
    {"label": "Changed area", "value": f"{stats['total_changed_pixels']:,} px"},
    {"label": "Change percentage", "value": f"{stats['percent_changed']:.2f}%"},
    {"label": "Avg. prediction probability", "value": f"{avg_prob}%" if avg_prob is not None else "N/A"},
    {"label": "Largest region", "value": f"{stats['largest_region_pixels']:,} px" if stats["num_regions"] else "N/A"},
])
st.caption(
    f"Area figures assume {pixel_size_m:.2f} m/pixel (LEVIR-CD's effective training resolution). "
    "This assumption is not verified for arbitrary uploads — see Diagnostics for details."
)

st.space("large")
section_header("Region analysis", "Every detected region is a \"detected change region\" — this model has no basis to assign a semantic category such as building or road.")
if stats["regions"]:
    region_id_overlay = create_region_id_overlay(after_resized, stats["regions"])
    st.image(region_id_overlay, caption="Regions labeled by ID", width=480)

    region_table = [
        {
            "Region": r["id"],
            "Area (px)": r["pixel_count"],
            "Width": r["width"],
            "Height": r["height"],
            "Perimeter": round(r.get("perimeter", 0), 1),
            "Aspect ratio": round(r.get("aspect_ratio", 0), 2),
            "Change density": round(r.get("change_density", 0), 3),
            "Prediction probability": round(r.get("mean_prediction_probability", float("nan")), 3),
            "Severity": round(r["severity_score"], 1),
            "Category": r["severity_category"],
        }
        for r in scored_regions
    ]

    categories = sorted({row["Category"] for row in region_table})
    selected_categories = st.multiselect("Filter by severity category", categories, default=categories)
    filtered = [row for row in region_table if row["Category"] in selected_categories]

    st.dataframe(
        filtered,
        column_config={
            "Prediction probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Severity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        },
        hide_index=True,
    )

    exp_col1, exp_col2 = st.columns(2)
    exp_col1.download_button(
        f"{ICONS['export']} Download region table (CSV)",
        data="\n".join([",".join(region_table[0].keys())] + [",".join(str(v) for v in row.values()) for row in region_table]) if region_table else "",
        file_name="detected_regions.csv", mime="text/csv", width="stretch",
    )
    exp_col2.download_button(
        f"{ICONS['export']} Download full statistics (JSON)",
        data=json.dumps({k: v for k, v in stats.items() if k != "regions"} | {"regions": region_table}, indent=2, default=str),
        file_name="change_statistics.json", mime="application/json", width="stretch",
    )

    st.space("large")
    section_header("Severity distribution", "An analytical score from measurable model outputs — not ground truth or a physical damage assessment.")
    dist = severity_distribution(scored_regions)
    dist_cols = st.columns(4)
    for i, cat in enumerate(["Low", "Moderate", "High", "Very High"]):
        dist_cols[i].metric(cat, dist["region_count_by_category"].get(cat, 0), border=True)

    top_regions = highest_severity_regions(scored_regions, n=5)
    with st.expander(f"Top {len(top_regions)} highest-severity regions"):
        st.dataframe(
            [{"Region": r["id"], "Severity": round(r["severity_score"], 1), "Category": r["severity_category"], "Area (px)": r["pixel_count"]} for r in top_regions],
            hide_index=True,
        )
else:
    empty_state("✅", "No changes detected", "No regions met the minimum size threshold configured in the sidebar.")

info_banner(
    "Prediction probability is the model's raw sigmoid output, not a calibrated confidence score.",
    link_label="Learn more", link_page="app_pages/diagnostics.py", key="detect-caveat",
)
