import numpy as np

from src.data.augmentation import PairedAugmentor


def test_spatial_augmentation_applies_identically_to_before_after_mask():
    """Flip/rotate must move the mask's changed pixels in lockstep with the images, or the
    label would no longer correspond to the image content."""
    size = 16
    before = np.zeros((size, size, 3), dtype=np.uint8)
    after = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Mark a single distinctive pixel in a fixed corner across all three, encoded as a matching
    # bright marker so we can verify they land in the same place after any spatial transform.
    before[2, 3] = [255, 0, 0]
    after[2, 3] = [0, 255, 0]
    mask[2, 3] = 255

    augmentor = PairedAugmentor(
        hflip_p=1.0, vflip_p=0.0, rotate90_p=0.0, scale_jitter_p=0.0, brightness_p=0.0
    )
    aug_before, aug_after, aug_mask = augmentor(before, after, mask)

    before_marker = np.argwhere(aug_before[:, :, 0] == 255)
    mask_marker = np.argwhere(aug_mask == 255)
    after_marker = np.argwhere(aug_after[:, :, 1] == 255)

    assert len(before_marker) == 1 and len(mask_marker) == 1 and len(after_marker) == 1
    assert tuple(before_marker[0]) == tuple(mask_marker[0]) == tuple(after_marker[0])


def test_no_augmentation_when_all_probabilities_zero():
    size = 16
    before = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    after = before.copy()
    mask = np.zeros((size, size), dtype=np.uint8)

    augmentor = PairedAugmentor(
        hflip_p=0.0, vflip_p=0.0, rotate90_p=0.0, scale_jitter_p=0.0, brightness_p=0.0
    )
    aug_before, aug_after, aug_mask = augmentor(before, after, mask)

    assert np.array_equal(before, aug_before)
    assert np.array_equal(mask, aug_mask)


def test_output_shapes_preserved_after_scale_jitter():
    size = 32
    before = np.zeros((size, size, 3), dtype=np.uint8)
    after = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    augmentor = PairedAugmentor(
        hflip_p=0.0, vflip_p=0.0, rotate90_p=0.0, scale_jitter_p=1.0, brightness_p=0.0
    )
    aug_before, aug_after, aug_mask = augmentor(before, after, mask)

    assert aug_before.shape == (size, size, 3)
    assert aug_after.shape == (size, size, 3)
    assert aug_mask.shape == (size, size)
