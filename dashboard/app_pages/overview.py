"""Overview — the landing page. Orients a new user in under a few seconds: what this system does,
its real measured performance, what it can do, and how to get started. No development history,
phase numbers, or implementation-status tables — those live in Diagnostics / the repository docs.
"""
import streamlit as st

from data import FLAGSHIP_DISPLAY_NAME, FLAGSHIP_EXPERIMENT, load_test_metrics
from theme import ICONS, capability_card, hero, info_banner, kpi_row, section_header

hero(
    "Satellite Change Intelligence",
    "AI-powered analysis of before/after satellite imagery — detecting, quantifying, and "
    "mapping geographical change with a deep-learning vision model.",
    tagline="Earth observation · computer vision · geospatial analytics",
)

metrics = load_test_metrics(FLAGSHIP_EXPERIMENT)
section_header("Model performance", f"Measured on a held-out benchmark test set — {FLAGSHIP_DISPLAY_NAME}")
if metrics:
    tm = metrics["test_metrics"]
    kpi_row([
        {"label": "IoU", "value": f"{tm['iou']:.4f}", "help": "Intersection-over-Union — the primary accuracy metric for change segmentation."},
        {"label": "Dice", "value": f"{tm['dice']:.4f}", "help": "Dice coefficient (overlap score)."},
        {"label": "F1 score", "value": f"{tm['f1']:.4f}", "help": "Harmonic mean of precision and recall."},
        {"label": "Accuracy", "value": f"{tm['accuracy']:.4f}", "help": "Overall pixel-wise accuracy."},
    ])
else:
    st.caption("Benchmark metrics unavailable in this environment.")

st.space("large")
section_header("System capabilities")
row1 = st.columns(3)
with row1[0]:
    capability_card(ICONS["detect"], "Building change detection",
                     "Detect changes between before/after satellite image pairs using a trained "
                     "Siamese neural network.", key="cap-detect")
with row1[1]:
    capability_card(ICONS["region"], "Change region analysis",
                     "Quantify every detected region — area, shape, geometry, and prediction "
                     "confidence.", key="cap-region")
with row1[2]:
    capability_card(ICONS["map"], "Geospatial analysis",
                     "View detected changes on a real interactive map and export them as "
                     "GeoJSON.", key="cap-geo")
row2 = st.columns(3)
with row2[0]:
    capability_card(ICONS["trend"], "Multi-temporal analysis",
                     "Compare change across more than two observation dates, interval by "
                     "interval.", key="cap-temporal")
with row2[1]:
    capability_card(ICONS["models"], "Model comparison",
                     "Compare multiple trained architectures side by side on real, measured "
                     "results.", key="cap-models")
with row2[2]:
    capability_card(ICONS["severity"], "Change severity scoring",
                     "Rank detected regions by an analytical severity score derived from model "
                     "outputs.", key="cap-severity")

st.space("large")
section_header("Quick start")
steps = st.columns(5)
labels = [
    (ICONS["upload"], "Upload before image"),
    (ICONS["upload"], "Upload after image"),
    (ICONS["detect"], "Run detection"),
    (ICONS["region"], "Analyze changes"),
    (ICONS["export"], "Export results"),
]
for i, (col, (icon, label)) in enumerate(zip(steps, labels), start=1):
    with col:
        with st.container(border=True, key=f"card-step-{i}"):
            st.markdown(f"**{i}.** {icon}")
            st.caption(label)

st.space("large")
info_banner(
    "Research model trained on the LEVIR-CD benchmark. Uploaded imagery should match expected "
    "input characteristics for meaningful results.",
    link_label="View limitations", link_page="app_pages/diagnostics.py", key="overview",
)
