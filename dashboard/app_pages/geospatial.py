"""Geospatial Intelligence — displays the most recent real geospatial analysis run (real
georeferenced Sentinel-2 imagery converted to true geographic polygons). This page does not
trigger a live re-analysis — that requires network access to fetch satellite imagery and is run
separately via the project's geospatial analysis pipeline.
"""
import json

import streamlit as st

from data import PROJECT_ROOT, load_json
from theme import ICONS, kpi_row, section_header

section_header("Geospatial intelligence", "Detected changes mapped to real geographic coordinates.", ICONS["geo"])

geojson_path = PROJECT_ROOT / "outputs" / "geospatial" / "regions.geojson"
map_path = PROJECT_ROOT / "outputs" / "geospatial" / "region_map.html"
geojson_data = load_json(geojson_path)

if not (geojson_data and map_path.exists()):
    st.info(
        "No geospatial analysis available in this environment. Run the geospatial analysis "
        "pipeline against real satellite imagery to generate results.",
        icon=ICONS["info"],
    )
    st.stop()

features = geojson_data.get("features", [])
total_area_ha = sum(f["properties"].get("area_hectares", 0) for f in features)
largest = max((f["properties"].get("area_hectares", 0) for f in features), default=0)

kpi_row([
    {"label": "Detected regions", "value": len(features)},
    {"label": "Total changed area", "value": f"{total_area_ha:.2f} ha"},
    {"label": "Largest region", "value": f"{largest:.2f} ha"},
])
st.caption(
    "Real, unvalidated result from one satellite image pair — no ground truth exists for "
    "real-world imagery. See Diagnostics for details."
)

st.space("large")
st.iframe(map_path, height=520)

st.space("medium")
st.download_button(
    f"{ICONS['export']} Download GeoJSON", data=json.dumps(geojson_data, indent=2),
    file_name="regions.geojson", mime="application/geo+json",
)

with st.expander("Region details"):
    st.dataframe(
        [
            {
                "Region": f["properties"].get("region_id"),
                "Area (m²)": f["properties"].get("area_m2"),
                "Area (ha)": f["properties"].get("area_hectares"),
                "Prediction probability": f["properties"].get("mean_prediction_probability"),
                "Severity": f["properties"].get("severity_score"),
                "Category": f["properties"].get("severity_category"),
            }
            for f in features
        ],
        hide_index=True,
    )
