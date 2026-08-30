"""Tests for dashboard/data.py — the dashboard's shared data-loading layer. Uses a synthetic
threshold-optimization report (no dependency on the real file's presence/content) so this test is
deterministic and fast.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))

import data as dashboard_data


def test_load_selected_threshold_exact_match_only(tmp_path, monkeypatch):
    """Regression test: a shorter experiment name that happens to be a string-prefix of the
    swept checkpoint's experiment directory (e.g. 'siamese_unet_diff' vs.
    'siamese_unet_diff_concat_attention_e100') must NOT be treated as a match — only the exact
    experiment whose checkpoint was actually swept should receive the optimized threshold."""
    metrics_dir = tmp_path / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True)
    report = {
        "checkpoint": "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt",
        "selected_threshold": 0.4,
    }
    with open(metrics_dir / "threshold_optimization_report.json", "w") as f:
        json.dump(report, f)

    monkeypatch.setattr(dashboard_data, "PROJECT_ROOT", tmp_path)

    exact_threshold, exact_source = dashboard_data.load_selected_threshold(
        "siamese_unet_diff_concat_attention_e100"
    )
    assert exact_threshold == 0.4
    assert "optimized" in exact_source

    for prefix_experiment in (
        "siamese_unet_diff", "siamese_unet_diff_concat", "siamese_unet_diff_concat_attention",
    ):
        threshold, source = dashboard_data.load_selected_threshold(prefix_experiment)
        assert threshold == 0.5, f"{prefix_experiment} incorrectly matched the swept checkpoint"
        assert "default" in source


def test_load_selected_threshold_missing_report_returns_default(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_data, "PROJECT_ROOT", tmp_path)
    threshold, source = dashboard_data.load_selected_threshold("any_experiment")
    assert threshold == 0.5
    assert "no threshold sweep" in source


def test_model_options_all_have_three_element_tuples():
    for name, value in dashboard_data.MODEL_OPTIONS.items():
        assert len(value) == 3, f"{name} should map to (config_path, checkpoint_path, experiment_name)"


def test_format_bytes_human_readable():
    assert dashboard_data.format_bytes(500) == "500 B"
    assert dashboard_data.format_bytes(2048) == "2.0 KB"
