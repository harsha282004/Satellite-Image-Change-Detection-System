"""Paired before/after/mask preprocessing for LEVIR-CD-style change-detection samples.

All functions operate on numpy arrays (H, W, C) uint8 for images and (H, W) uint8 for masks,
until explicitly converted to CHW float tensors by `to_tensor`. Keeping the numpy stage separate
from the tensor stage lets `augmentation.py` apply identical spatial transforms to the before
image, after image, and mask before any of them become tensors.
"""
import cv2
import numpy as np
import torch

MASK_BINARIZATION_THRESHOLD = 127


def load_image(path: str) -> np.ndarray:
    """Load an RGB image as (H, W, 3) uint8. Raises if the file can't be decoded."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: str) -> np.ndarray:
    """Load a single-channel mask as (H, W) uint8, raw pixel values (not yet binarized)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask


def binarize_mask(mask: np.ndarray, threshold: int = MASK_BINARIZATION_THRESHOLD) -> np.ndarray:
    """Threshold a raw mask to strictly {0, 1} uint8.

    LEVIR-CD mask files are not always exactly {0, 255} — some contain anti-aliased edge
    values (e.g. 156, 254). See docs/DATASET.md "Mask binarization note" (Phase 2 verification).
    """
    return (mask > threshold).astype(np.uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB image to (size, size) with linear interpolation (smooth, for photos)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize a mask to (size, size) with nearest-neighbor interpolation (preserves {0,1})."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale a uint8 RGB image to float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def to_tensor_image(image: np.ndarray) -> torch.Tensor:
    """(H, W, 3) float32 in [0,1] -> (3, H, W) float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 {0,1} -> (1, H, W) float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)


def preprocess_pair(before: np.ndarray, after: np.ndarray, mask: np.ndarray, image_size: int):
    """Full non-augmented preprocessing pipeline: resize -> (binarize mask) -> normalize -> tensor.

    `mask` must already be binarized (see `binarize_mask`) before calling this, since
    binarization should happen once at raw resolution, not after resizing.
    """
    before_r = resize_image(before, image_size)
    after_r = resize_image(after, image_size)
    mask_r = resize_mask(mask, image_size)

    before_t = to_tensor_image(normalize_image(before_r))
    after_t = to_tensor_image(normalize_image(after_r))
    mask_t = to_tensor_mask(mask_r)

    return before_t, after_t, mask_t
