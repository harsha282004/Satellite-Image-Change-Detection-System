"""Shared data-loading layer for the dashboard — model registry, checkpoint/metrics loading, image
decoding, threshold selection. Every function here is unchanged in behavior from the pre-redesign
dashboard; only display strings were reworded to drop internal phase/development references. All
config paths, checkpoint paths, and experiment names are byte-identical to before — no backend,
model, or evaluation logic was touched by the frontend redesign.
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.inference.predict import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (display name -> (config path, checkpoint path, experiment_name)). Every entry is a model that
# was actually trained and evaluated on the real LEVIR-CD test set — see docs/EXPERIMENTS.md for
# full methodology. Display names are user-facing and phase-free; the underlying paths are exactly
# what the project's training/evaluation pipeline produced.
MODEL_OPTIONS = {
    "Siamese U-Net + Attention — Recommended": (
        "configs/siamese_attention_e100.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention_e100/best.pt",
        "siamese_unet_diff_concat_attention_e100",
    ),
    "Siamese U-Net + Attention — Extended training": (
        "configs/siamese_attention_e60.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention_e60/best.pt",
        "siamese_unet_diff_concat_attention_e60",
    ),
    "Siamese U-Net + Attention — Standard training": (
        "configs/siamese_attention.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat_attention/best.pt",
        "siamese_unet_diff_concat_attention",
    ),
    "Siamese U-Net — Diff + Concat": (
        "configs/siamese.yaml",
        "outputs/checkpoints/siamese_unet_diff_concat/best.pt",
        "siamese_unet_diff_concat",
    ),
    "Siamese U-Net — Concat": (
        "configs/siamese_concat.yaml",
        "outputs/checkpoints/siamese_unet_concat/best.pt",
        "siamese_unet_concat",
    ),
    "Siamese U-Net — Diff": (
        "configs/siamese_diff.yaml",
        "outputs/checkpoints/siamese_unet_diff/best.pt",
        "siamese_unet_diff",
    ),
    "Baseline U-Net": (
        "configs/baseline.yaml",
        "outputs/checkpoints/baseline_unet/best.pt",
        "baseline_unet",
    ),
    "Transformer — Experimental": (
        "configs/transformer.yaml",
        "outputs/checkpoints/transformer_change_diff_concat/best.pt",
        "transformer_change_diff_concat",
    ),
}

# The flagship model shown on the Overview page's headline KPIs — the best real, measured result
# in the project, independent of whatever model is currently selected in the sidebar.
FLAGSHIP_EXPERIMENT = "siamese_unet_diff_concat_attention_e100"
FLAGSHIP_DISPLAY_NAME = "Siamese U-Net + Attention"


@st.cache_resource
def load_predictor(config_path: str, checkpoint_path: str) -> Predictor:
    return Predictor(str(PROJECT_ROOT / config_path), str(PROJECT_ROOT / checkpoint_path))


def load_test_metrics(experiment_name: str) -> dict | None:
    path = PROJECT_ROOT / "outputs" / "metrics" / f"{experiment_name}_test_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_image_from_upload(uploaded_file) -> np.ndarray | None:
    try:
        return np.array(Image.open(uploaded_file).convert("RGB"))
    except UnidentifiedImageError:
        return None


def load_selected_threshold(experiment_name: str) -> tuple:
    """Returns (threshold, source_note). Only applies the validation-optimized threshold when the
    currently selected model matches the checkpoint that threshold was actually swept for — every
    other model falls back to the untuned default 0.5, rather than silently applying a threshold
    optimized for a different model's outputs."""
    report_path = PROJECT_ROOT / "outputs" / "metrics" / "threshold_optimization_report.json"
    if not report_path.exists():
        return 0.5, "default (no threshold sweep available)"
    with open(report_path) as f:
        report = json.load(f)
    # Exact match against the checkpoint's own experiment directory name — not substring
    # containment, which would incorrectly match a shorter experiment name (e.g.
    # "siamese_unet_diff") against a longer one that happens to start with it (e.g.
    # "siamese_unet_diff_concat_attention_e100").
    swept_experiment = Path(report.get("checkpoint", "")).parent.name
    if experiment_name == swept_experiment:
        return report["selected_threshold"], "optimized via validation-set sweep"
    return 0.5, "default (sweep only available for the recommended model)"


def probability_map_to_rgb(prob_map: np.ndarray) -> np.ndarray:
    """Viridis-colored visualization of a [0,1] probability map, for display only."""
    import matplotlib

    colored = matplotlib.colormaps["viridis"](prob_map)  # (H, W, 4) float in [0,1]
    return (colored[:, :, :3] * 255).astype(np.uint8)


def format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
