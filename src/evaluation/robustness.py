"""Phase 15.4: controlled input perturbations for robustness testing.

Each function returns a modified copy of an input image; ground truth is never touched, since
these perturbations simulate nuisance imaging variation (lighting, sensor noise, imperfect
registration) that should NOT change the true underlying geographic change — a real IoU drop under
one of these means the model is sensitive to that nuisance factor, which is exactly what this
analysis is meant to surface (PROJECT_CONTEXT.md's "actual change vs. apparent difference"
principle, applied as a controlled test rather than only discussed qualitatively).
"""
import cv2
import numpy as np


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = image.astype(np.float32).mean()
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by (dx, dy) pixels, replicating edge pixels at the border — simulates a small
    misregistration between the before/after capture, not an out-of-frame crop."""
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


PERTURBATIONS = {
    "brightness_+30%": lambda img: adjust_brightness(img, 1.3),
    "brightness_-30%": lambda img: adjust_brightness(img, 0.7),
    "contrast_+30%": lambda img: adjust_contrast(img, 1.3),
    "contrast_-30%": lambda img: adjust_contrast(img, 0.7),
    "gaussian_noise_sigma15": lambda img: add_gaussian_noise(img, 15.0),
    "shift_5px": lambda img: shift_image(img, 5, 5),
}
