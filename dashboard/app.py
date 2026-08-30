"""Satellite Change Intelligence — entry point.

Run with: venv/Scripts/python.exe -m streamlit run dashboard/app.py

This file is the app's router/frame: page config, the dark design system, the global sidebar
(brand identity + model/threshold controls shared across every page), and the page navigation.
Each page lives in `dashboard/app_pages/` and reads its inputs from `st.session_state` (set here)
plus the shared loaders in `dashboard/data.py`. No model, checkpoint, or evaluation logic lives in
this file or was changed by the frontend redesign — see `dashboard/data.py` for the unchanged
data-loading layer.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from data import MODEL_OPTIONS, load_selected_threshold, load_test_metrics
from theme import ICONS, inject_theme

st.set_page_config(
    page_title="Satellite Change Intelligence",
    page_icon=":material/satellite_alt:",
    layout="wide",
)
inject_theme()

pages = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=ICONS["overview"], default=True),
        st.Page("app_pages/change_detection.py", title="Change detection", icon=ICONS["detect"]),
        st.Page("app_pages/model_analysis.py", title="Model analysis", icon=ICONS["models"]),
        st.Page("app_pages/geospatial.py", title="Geospatial intelligence", icon=ICONS["geo"]),
        st.Page("app_pages/temporal.py", title="Temporal analysis", icon=ICONS["temporal"]),
        st.Page("app_pages/diagnostics.py", title="Diagnostics", icon=ICONS["diagnostics"]),
    ],
    position="sidebar",
)

with st.sidebar:
    st.markdown(
        '<div class="sci-brand">SATELLITE<br>CHANGE<br><span>INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-powered Earth observation & change detection")
    st.space("medium")

    model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="selected_model_name")
    config_path, checkpoint_path, experiment_name = MODEL_OPTIONS[model_name]
    checkpoint_exists = (PROJECT_ROOT / checkpoint_path).exists()
    st.session_state.model_selection = {
        "display_name": model_name,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "experiment_name": experiment_name,
        "checkpoint_exists": checkpoint_exists,
    }
    if not checkpoint_exists:
        st.error("Selected model's checkpoint is unavailable in this environment.", icon=ICONS["error"])

    default_threshold, threshold_source = load_selected_threshold(experiment_name)
    threshold = st.slider(
        "Detection threshold", min_value=0.0, max_value=1.0, value=default_threshold, step=0.05,
        help=f"Pixels with predicted change probability above this value are classified as "
             f"changed. Current default: {threshold_source}.",
        key="detection_threshold",
    )
    min_region_pixels = st.number_input(
        "Minimum region size (px)", min_value=1, value=4,
        help="Detected regions smaller than this are treated as noise and excluded from counts "
             "and area statistics.",
        key="min_region_pixels",
    )

    st.space("medium")
    metrics = load_test_metrics(experiment_name)
    if metrics:
        tm = metrics["test_metrics"]
        st.caption("Current model — benchmark performance")
        with st.container(horizontal=True):
            st.metric("IoU", f"{tm['iou']:.4f}", border=True)
            st.metric("Dice", f"{tm['dice']:.4f}", border=True)

    st.space("medium")
    st.page_link("app_pages/diagnostics.py", label="Diagnostics", icon=ICONS["diagnostics"])

pages.run()
