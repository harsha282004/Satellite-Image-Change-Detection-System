"""Single before/after inference: load a trained checkpoint and predict a binary change mask.

Reused by src/analysis (Phase 9) and, later, dashboard/app.py (Phase 10) — one implementation,
not duplicated per caller (DEVELOPMENT_RULES.md Rule 6).
"""
import numpy as np
import torch

from src.data.preprocessing import load_image, normalize_image, resize_image, to_tensor_image
from src.evaluation.metrics import logits_to_binary_preds
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, load_config, resolve_device


@torch.no_grad()
def predict_mask(model, before_rgb: np.ndarray, after_rgb: np.ndarray, image_size: int,
                  device, threshold: float = 0.5) -> np.ndarray:
    """`before_rgb`/`after_rgb`: (H, W, 3) uint8 RGB arrays, any input size. Returns an
    `(image_size, image_size)` uint8 binary mask ({0, 1}), resized to the model's input size."""
    before_r = resize_image(before_rgb, image_size)
    after_r = resize_image(after_rgb, image_size)
    before_t = to_tensor_image(normalize_image(before_r)).unsqueeze(0).to(device)
    after_t = to_tensor_image(normalize_image(after_r)).unsqueeze(0).to(device)

    model.eval()
    logits = model(before_t, after_t)
    pred = logits_to_binary_preds(logits, threshold)[0, 0].cpu().numpy().astype(np.uint8)
    return pred


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
