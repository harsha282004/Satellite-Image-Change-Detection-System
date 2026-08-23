"""Sentinel-2 access via the Earth Search STAC API (Element84, AWS-hosted).

No authentication required — the official Copernicus Data Space portal requires account
registration (a manual-download-only source, per the same reasoning as docs/DATASET.md's LEVIR-CD
acquisition), so this project uses Earth Search's public, unauthenticated STAC index over the
Sentinel-2 Cloud-Optimized GeoTIFFs on AWS Open Data instead — see docs/REAL_WORLD_DEMO.md.

Windowed crops are read directly from the remote COGs over HTTP (rasterio/GDAL's /vsicurl/
support) — no full-tile (~100+ MB) download needed for a small area of interest.
"""
from typing import List

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"


def search_sentinel2_items(bbox_wgs84: List[float], datetime_range: str,
                            max_cloud_cover: float = 10.0, max_items: int = 100) -> list:
    """`bbox_wgs84`: [min_lon, min_lat, max_lon, max_lat]. Returns items sorted oldest-first."""
    catalog = Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[SENTINEL2_L2A_COLLECTION],
        bbox=bbox_wgs84,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )
    items = list(search.items())
    items.sort(key=lambda i: i.datetime)
    return items


def get_item_by_id(item_id: str, bbox_wgs84: List[float]):
    catalog = Client.open(EARTH_SEARCH_URL)
    search = catalog.search(collections=[SENTINEL2_L2A_COLLECTION], ids=[item_id], bbox=bbox_wgs84)
    items = list(search.items())
    if not items:
        raise ValueError(f"No STAC item found for id={item_id!r}")
    return items[0]


def fetch_visual_crop(item, bbox_wgs84: List[float]) -> np.ndarray:
    """Read the true-color 'visual' (TCI, already radiometrically rendered RGB) asset of a STAC
    item, windowed to `bbox_wgs84`. Returns (H, W, 3) uint8 RGB — native Sentinel-2 resolution,
    10 m/pixel (see docs/REAL_WORLD_DEMO.md for why this matters relative to LEVIR-CD's 0.5 m/pixel
    training data)."""
    visual_url = item.assets["visual"].href
    with rasterio.open(visual_url) as src:
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        data = src.read([1, 2, 3], window=window)
    return np.transpose(data, (1, 2, 0))
