"""End-to-end smoke test for the exact real function-call sequence the Change Detection page runs
on an uploaded image pair. `streamlit.testing.v1.AppTest` cannot simulate `st.file_uploader`, so
this test calls the same real pipeline functions directly, with a real image pair and the real
best-model checkpoint, to lock in that the sequence `dashboard/app_pages/change_detection.py` runs
actually works end-to-end (not just that the page renders with nothing uploaded).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from data import MODEL_OPTIONS, load_predictor, probability_map_to_rgb
from src.analysis.area import levir_cd_effective_pixel_size
from src.analysis.severity import compute_severity_for_regions, highest_severity_regions, severity_distribution
from src.analysis.statistics import compute_change_statistics
from src.realworld.validation import validate_real_world_input
from src.visualization.overlays import create_overlay, create_region_id_overlay

TEST_IMAGE_ID = "test_29.png"


def _load_test_pair():
    before = np.array(Image.open(PROJECT_ROOT / "data/raw/levir_cd/test/A" / TEST_IMAGE_ID).convert("RGB"))
    after = np.array(Image.open(PROJECT_ROOT / "data/raw/levir_cd/test/B" / TEST_IMAGE_ID).convert("RGB"))
    return before, after


def test_full_change_detection_pipeline_runs_end_to_end_on_real_data():
    before_img, after_img = _load_test_pair()

    validation_report = validate_real_world_input(before_img, after_img)
    assert validation_report["dimensions"]["dimensions_match"] is True

    config_path, checkpoint_path, _ = MODEL_OPTIONS["Siamese U-Net + Attention — Recommended"]
    predictor = load_predictor(config_path, checkpoint_path)
    size = predictor.image_size
    before_resized = cv2.resize(before_img, (size, size))
    after_resized = cv2.resize(after_img, (size, size))

    prob_map = predictor.predict_probability_from_arrays(before_img, after_img)
    assert prob_map.shape == (size, size)
    assert prob_map.min() >= 0.0 and prob_map.max() <= 1.0

    mask = (prob_map > 0.4).astype(np.uint8)
    pixel_size_m = levir_cd_effective_pixel_size(size)
    stats = compute_change_statistics(
        mask, probability_map=prob_map, pixel_size_meters=pixel_size_m, min_region_pixels=4,
    )
    assert stats["num_regions"] > 0  # test_29 is a known real-change scene

    scored_regions = compute_severity_for_regions(stats["regions"])
    assert len(scored_regions) == stats["num_regions"]
    for r in scored_regions:
        assert "severity_score" in r and "severity_category" in r

    dist = severity_distribution(scored_regions)
    assert sum(dist["region_count_by_category"].values()) == stats["num_regions"]

    top = highest_severity_regions(scored_regions, n=5)
    assert len(top) <= 5

    overlay = create_overlay(after_resized, mask, color=(1.0, 0.0, 0.0), alpha=0.6)
    assert overlay.shape == after_resized.shape

    prob_rgb = probability_map_to_rgb(prob_map)
    assert prob_rgb.shape == (size, size, 3)

    id_overlay = create_region_id_overlay(after_resized, stats["regions"])
    assert id_overlay.shape == after_resized.shape

    avg_prob = np.mean([r.get("mean_prediction_probability", 0.0) for r in stats["regions"]])
    assert 0.0 <= avg_prob <= 1.0


def test_full_change_detection_pipeline_handles_no_detected_change():
    """The same real image compared against itself must yield no real change (defensive check
    that the pipeline degrades gracefully, not that this is a documented expected result)."""
    before_img, _ = _load_test_pair()
    config_path, checkpoint_path, _ = MODEL_OPTIONS["Siamese U-Net + Attention — Recommended"]
    predictor = load_predictor(config_path, checkpoint_path)

    prob_map = predictor.predict_probability_from_arrays(before_img, before_img)
    mask = (prob_map > 0.4).astype(np.uint8)
    stats = compute_change_statistics(mask, probability_map=prob_map, min_region_pixels=4)
    scored_regions = compute_severity_for_regions(stats["regions"]) if stats["regions"] else []
    # Should not raise even with zero (or near-zero) detected regions.
    severity_distribution(scored_regions)
