"""Phase 18.5: interactive Folium map of detected-change-region geographic features.

Only ever built from real GeoJSON features produced by
`src/geospatial/polygons.py::regions_to_geo_features` — i.e. only for imagery with a real,
verified CRS/transform (`src/geospatial/raster.py::has_georeference`). Never called on plain
LEVIR-CD PNGs.
"""
import folium


def build_region_map(geojson_features: list, center_lat: float = None, center_lon: float = None,
                      zoom_start: int = 15) -> folium.Map:
    """Builds a Folium map with each detected region drawn as a polygon layer, popup showing its
    real, measured properties (area, prediction probability, severity if present). If
    `center_lat`/`center_lon` aren't given, centers on the mean centroid of all feature polygons."""
    if center_lat is None or center_lon is None:
        center_lat, center_lon = _mean_centroid(geojson_features)

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    regions_layer = folium.FeatureGroup(name="Detected Change Regions", show=True)
    for feature in geojson_features:
        props = feature["properties"]
        popup_html = _region_popup_html(props)
        folium.GeoJson(
            feature,
            style_function=lambda _f: {"color": "red", "weight": 2, "fillColor": "red", "fillOpacity": 0.35},
            tooltip=f"Region {props.get('region_id', '?')}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(regions_layer)
    regions_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _region_popup_html(props: dict) -> str:
    lines = [f"<b>Detected Change Region {props.get('region_id', '?')}</b><br>"]
    lines.append(f"Area: {props.get('area_m2', '?')} m² ({props.get('area_hectares', '?')} ha)<br>")
    if "mean_prediction_probability" in props:
        lines.append(f"Prediction probability (mean): {props['mean_prediction_probability']}<br>")
    if "severity_score" in props:
        lines.append(f"Severity: {props['severity_score']} ({props.get('severity_category', '?')})<br>")
    lines.append(
        "<i>Not ground truth — a model prediction and an analytical score. "
        "See docs/EVALUATION.md.</i>"
    )
    return "".join(lines)


def _mean_centroid(geojson_features: list) -> tuple:
    if not geojson_features:
        return 0.0, 0.0
    lats, lons = [], []
    for feature in geojson_features:
        coords = feature["geometry"]["coordinates"][0]  # exterior ring of a Polygon
        for lon, lat in coords:
            lats.append(lat)
            lons.append(lon)
    return sum(lats) / len(lats), sum(lons) / len(lons)


def save_map_html(fmap: folium.Map, out_path: str) -> str:
    fmap.save(out_path)
    return out_path
