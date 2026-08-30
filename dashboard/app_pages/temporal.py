"""Temporal Analysis — change across more than two observation dates. Shows the most recent real
multi-temporal analysis run. Each interval is an independent detection between two images; no
tracking or causal claim is made across intervals — the model has no mechanism to follow a
specific change over time.
"""
import streamlit as st

from data import PROJECT_ROOT, load_json
from theme import ICONS, info_banner, kpi_row, section_header

section_header("Temporal analysis", "Change measured across a sequence of observation dates.", ICONS["temporal"])

report = load_json(PROJECT_ROOT / "outputs" / "multitemporal" / "temporal_report.json")
chart_path = PROJECT_ROOT / "outputs" / "multitemporal" / "temporal_change_area.png"

if not report:
    st.info(
        "No multi-temporal analysis available in this environment. Run the multi-temporal "
        "analysis pipeline against real satellite imagery to generate results.",
        icon=ICONS["info"],
    )
    st.stop()

info_banner(
    "Each interval below is an independent detection between two images — not a tracked trend "
    "for a single change.",
    link_label="Learn more", link_page="app_pages/diagnostics.py", key="temporal-caveat",
)

intervals = report["intervals"]
kpi_row([
    {"label": "Observation dates", "value": len(report["selected_dates"])},
    {"label": "Intervals analyzed", "value": len(intervals)},
    {"label": "Total detected regions", "value": sum(iv["num_regions"] for iv in intervals)},
])

st.space("large")
section_header("Interval results")
for i, iv in enumerate(intervals, start=1):
    with st.container(border=True, key=f"card-interval-{i}"):
        st.markdown(f"**T{i} → T{i+1}**  ·  {iv['from_date']} → {iv['to_date']}")
        cols = st.columns(3)
        cols[0].metric("Detected regions", iv["num_regions"])
        cols[1].metric("Changed pixels", f"{iv['total_changed_pixels']:,}")
        area_ha = iv.get("total_changed_area", {}).get("area_hectares")
        cols[2].metric("Changed area", f"{area_ha:.2f} ha" if area_ha is not None else "N/A")

if chart_path.exists():
    st.space("large")
    section_header("Temporal trend")
    st.image(str(chart_path), caption="Per-interval detected change area and region count", width="stretch")
