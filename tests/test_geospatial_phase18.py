"""Phase 18: pixel->geographic conversion, area calculation, and the georeference guard —
using small synthetic in-memory rasters (real rasterio datasets, real CRS/transform) so these
tests are fast and network-independent, unlike Phase 11's real-Sentinel-2-dependent scripts.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.geospatial.polygons import (
    polygon_area_m2,
    polygon_to_wgs84,
    region_bbox_to_polygon,
    regions_to_geo_features,
)
from src.geospatial.raster import has_georeference, read_raster_metadata


UTM_14N = "EPSG:32614"  # matches the Sentinel-2 tile used in Phase 11's real-world demo


@pytest.fixture
def georeferenced_tif(tmp_path):
    """A tiny 20x20 raster at 10 m/pixel in UTM 14N, origin arbitrary but real/consistent —
    exactly the shape of a real Sentinel-2 'visual' crop, just synthetic pixel content."""
    path = tmp_path / "synthetic_geo.tif"
    transform = from_origin(500000, 3300000, 10, 10)  # 10m pixel size, UTM 14N-plausible origin
    data = np.random.default_rng(0).integers(0, 255, (3, 20, 20), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=3, dtype="uint8",
        crs=UTM_14N, transform=transform,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture
def non_georeferenced_tif(tmp_path):
    """A raster with no CRS and an identity transform — simulates a plain PNG/JPEG with no
    geospatial metadata (what LEVIR-CD images and arbitrary uploads actually are)."""
    path = tmp_path / "plain.tif"
    data = np.zeros((3, 10, 10), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=3, dtype="uint8",
    ) as dst:
        dst.write(data)
    return str(path)


def test_has_georeference_true_for_real_geo_raster(georeferenced_tif):
    assert has_georeference(georeferenced_tif) is True


def test_has_georeference_false_for_plain_raster(non_georeferenced_tif):
    assert has_georeference(non_georeferenced_tif) is False


def test_read_raster_metadata_reports_real_values(georeferenced_tif):
    meta = read_raster_metadata(georeferenced_tif)
    assert meta["crs"] == UTM_14N
    assert meta["resolution_x"] == pytest.approx(10.0)
    assert meta["resolution_y"] == pytest.approx(10.0)
    assert meta["width"] == 20
    assert meta["height"] == 20
    assert meta["is_georeferenced"] is True


def test_read_raster_metadata_plain_raster_reports_not_georeferenced(non_georeferenced_tif):
    meta = read_raster_metadata(non_georeferenced_tif)
    assert meta["is_georeferenced"] is False


def test_region_bbox_to_polygon_uses_real_affine_transform(georeferenced_tif):
    with rasterio.open(georeferenced_tif) as src:
        transform = src.transform

    region = {"bbox": {"min_row": 0, "max_row": 5, "min_col": 0, "max_col": 5}}
    polygon = region_bbox_to_polygon(region, transform)

    # top-left pixel (0,0) maps to the raster's origin (500000, 3300000); 5 pixels * 10m = 50m span
    minx, miny, maxx, maxy = polygon.bounds
    assert minx == pytest.approx(500000)
    assert maxy == pytest.approx(3300000)
    assert (maxx - minx) == pytest.approx(50.0)
    assert (maxy - miny) == pytest.approx(50.0)


def test_polygon_area_m2_correct_for_known_pixel_region(georeferenced_tif):
    with rasterio.open(georeferenced_tif) as src:
        transform = src.transform

    # a 5x4 pixel region at 10m/pixel -> 50m x 40m -> 2000 m^2
    region = {"bbox": {"min_row": 0, "max_row": 4, "min_col": 0, "max_col": 5}}
    polygon = region_bbox_to_polygon(region, transform)
    area = polygon_area_m2(polygon, UTM_14N)
    assert area == pytest.approx(2000.0)


def test_polygon_area_m2_raises_for_geographic_crs():
    from shapely.geometry import box
    polygon = box(0, 0, 1, 1)
    with pytest.raises(ValueError, match="projected"):
        polygon_area_m2(polygon, "EPSG:4326")


def test_polygon_to_wgs84_returns_plausible_lon_lat(georeferenced_tif):
    with rasterio.open(georeferenced_tif) as src:
        transform = src.transform
    region = {"bbox": {"min_row": 0, "max_row": 5, "min_col": 0, "max_col": 5}}
    polygon = region_bbox_to_polygon(region, transform)
    polygon_wgs84 = polygon_to_wgs84(polygon, UTM_14N)

    minx, miny, maxx, maxy = polygon_wgs84.bounds
    assert -180 <= minx <= 180 and -180 <= maxx <= 180
    assert -90 <= miny <= 90 and -90 <= maxy <= 90
    # UTM zone 14N is roughly -96 longitude — sanity check it's in a plausible range, not (0,0)
    assert -100 < minx < -90


def test_regions_to_geo_features_produces_valid_geojson_features(georeferenced_tif):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 1  # one region

    features = regions_to_geo_features(mask, georeferenced_tif, min_region_pixels=1)

    assert len(features) == 1
    feature = features[0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert "area_m2" in feature["properties"]
    assert "area_hectares" in feature["properties"]
    assert feature["properties"]["area_m2"] == pytest.approx(6 * 6 * 100)  # 6x6 px * (10m)^2


def test_regions_to_geo_features_includes_probability_when_given(georeferenced_tif):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    probs = np.full((20, 20), 0.75, dtype=np.float32)

    features = regions_to_geo_features(mask, georeferenced_tif, probability_map=probs)
    assert features[0]["properties"]["mean_prediction_probability"] == pytest.approx(0.75)


def test_regions_to_geo_features_empty_mask_returns_no_features(georeferenced_tif):
    mask = np.zeros((20, 20), dtype=np.uint8)
    features = regions_to_geo_features(mask, georeferenced_tif)
    assert features == []
