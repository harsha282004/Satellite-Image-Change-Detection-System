"""Single before/after inference: load a trained checkpoint and predict a binary change mask.

Reused by src/analysis (Phase 9) and, later, dashboard/app.py (Phase 10) — one implementation,
not duplicated per caller (DEVELOPMENT_RULES.md Rule 6).
"""
import numpy as np
import torch

from src.data.preprocessing import load_image, normalize_image, resize_image, to_tensor_image
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, load_config, resolve_device


@torch.no_grad()
def predict_probability(model, before_rgb: np.ndarray, after_rgb: np.ndarray, image_size: int,
                         device) -> np.ndarray:
    """`before_rgb`/`after_rgb`: (H, W, 3) uint8 RGB arrays, any input size. Returns an
    `(image_size, image_size)` float32 array of per-pixel "changed" probabilities in [0, 1]
    (`sigmoid(logits)`) — the raw model output before any threshold is applied (Phase 15).

    This is a probability in the sigmoid-output sense only — the model has not been evaluated for
    calibration (does a 0.8 output actually correspond to ~80% empirical accuracy?), so it must
    not be described as "confidence" anywhere downstream. See docs/EVALUATION.md Phase 15 section.
    """
    before_r = resize_image(before_rgb, image_size)
    after_r = resize_image(after_rgb, image_size)
    before_t = to_tensor_image(normalize_image(before_r)).unsqueeze(0).to(device)
    after_t = to_tensor_image(normalize_image(after_r)).unsqueeze(0).to(device)

    model.eval()
    logits = model(before_t, after_t)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    return probs


def predict_mask(model, before_rgb: np.ndarray, after_rgb: np.ndarray, image_size: int,
                  device, threshold: float = 0.5) -> np.ndarray:
    """`before_rgb`/`after_rgb`: (H, W, 3) uint8 RGB arrays, any input size. Returns an
    `(image_size, image_size)` uint8 binary mask ({0, 1}), resized to the model's input size."""
    probs = predict_probability(model, before_rgb, after_rgb, image_size, device)
    return (probs > threshold).astype(np.uint8)


class Predictor:
    """Loads a config + checkpoint once, then predicts repeatedly — avoids re-loading the model
    for every image pair (relevant for Phase 10's interactive dashboard)."""

    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = resolve_device(self.config["device"])
        self.image_size = self.config["dataset"]["image_size"]
        self.model = build_model(self.config).to(self.device)
        self.checkpoint_info = load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

    def predict_from_arrays(self, before_rgb: np.ndarray, after_rgb: np.ndarray,
                             threshold: float = 0.5) -> np.ndarray:
        return predict_mask(self.model, before_rgb, after_rgb, self.image_size, self.device, threshold)

    def predict_from_paths(self, before_path: str, after_path: str, threshold: float = 0.5) -> np.ndarray:
        before_rgb = load_image(before_path)
        after_rgb = load_image(after_path)
        return self.predict_from_arrays(before_rgb, after_rgb, threshold=threshold)

    def predict_probability_from_arrays(self, before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
        return predict_probability(self.model, before_rgb, after_rgb, self.image_size, self.device)

    def predict_probability_from_paths(self, before_path: str, after_path: str) -> np.ndarray:
        before_rgb = load_image(before_path)
        after_rgb = load_image(after_path)
        return self.predict_probability_from_arrays(before_rgb, after_rgb)
