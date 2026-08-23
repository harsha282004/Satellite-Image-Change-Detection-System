"""Network-independent smoke tests only — src/geospatial/raster.py's actual STAC search and COG
reads require internet access to AWS-hosted Sentinel-2 data, consistent with how the Phase 2
dataset download and Phase 9's real-image demo script are also exercised by their own dedicated
scripts (scripts/real_world_demo.py) rather than the fast pytest suite."""
from src.geospatial.raster import EARTH_SEARCH_URL, SENTINEL2_L2A_COLLECTION


def test_earth_search_url_is_https():
    assert EARTH_SEARCH_URL.startswith("https://")


def test_sentinel2_collection_name():
    assert SENTINEL2_L2A_COLLECTION == "sentinel-2-l2a"
