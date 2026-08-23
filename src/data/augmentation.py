"""Paired spatial/photometric augmentation for before/after/mask triplets.

Applied after resizing to a fixed square size (see preprocessing.resize_image/resize_mask), so
every augmentation here operates on same-sized (image_size, image_size, ...) numpy arrays.

Spatial augmentations (flip, 90-degree rotation, random-crop-then-resize "scale" jitter) are
applied identically to the before image, after image, and mask using one shared random draw per
sample, per PROJECT_CONTEXT.md ("Spatial augmentations must be applied consistently"). Photometric
augmentation (brightness) is applied independently to the before/after images only — never to the
mask, since brightness is not a meaningful concept for a binary label.
"""
import random

import cv2
import numpy as np


class PairedAugmentor:
    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotate90_p: float = 0.5,
        scale_jitter_p: float = 0.3,
        scale_range: tuple = (0.8, 1.0),
        brightness_p: float = 0.3,
        brightness_range: tuple = (0.8, 1.2),
    ):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate90_p = rotate90_p
        self.scale_jitter_p = scale_jitter_p
        self.scale_range = scale_range
        self.brightness_p = brightness_p
        self.brightness_range = brightness_range

    def __call__(self, before: np.ndarray, after: np.ndarray, mask: np.ndarray):
        size = mask.shape[0]

        if random.random() < self.hflip_p:
            before, after, mask = before[:, ::-1], after[:, ::-1], mask[:, ::-1]

        if random.random() < self.vflip_p:
            before, after, mask = before[::-1, :], after[::-1, :], mask[::-1, :]

        if random.random() < self.rotate90_p:
            k = random.choice([1, 2, 3])
            before, after, mask = np.rot90(before, k), np.rot90(after, k), np.rot90(mask, k)

        if random.random() < self.scale_jitter_p:
            scale = random.uniform(*self.scale_range)
            crop_size = max(1, int(size * scale))
            max_offset = size - crop_size
            top = random.randint(0, max_offset) if max_offset > 0 else 0
            left = random.randint(0, max_offset) if max_offset > 0 else 0

            before_c = before[top:top + crop_size, left:left + crop_size]
            after_c = after[top:top + crop_size, left:left + crop_size]
            mask_c = mask[top:top + crop_size, left:left + crop_size]

            before = cv2.resize(before_c, (size, size), interpolation=cv2.INTER_LINEAR)
            after = cv2.resize(after_c, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_c, (size, size), interpolation=cv2.INTER_NEAREST)

        if random.random() < self.brightness_p:
            factor = random.uniform(*self.brightness_range)
            before = np.clip(before.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() < self.brightness_p:
            factor = random.uniform(*self.brightness_range)
            after = np.clip(after.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        before = np.ascontiguousarray(before)
        after = np.ascontiguousarray(after)
        mask = np.ascontiguousarray(mask)
        return before, after, mask
