"""Phase 22: real-world pipeline hardening — image validation, registration/alignment checking,
resolution plausibility, and an explicit cloud-screening heuristic for arbitrary user-uploaded or
externally-sourced before/after image pairs (as opposed to the curated, pre-registered LEVIR-CD
benchmark pairs every other phase's quantitative results are measured on).

None of this makes the model's predictions more accurate on non-LEVIR-CD imagery — it only
surfaces real, computable signals about whether the *input* looks like something the model has any
business being run on, so a user gets an honest warning instead of a silently unreliable result.
**Benchmark accuracy (docs/EVALUATION.md) is never reported as, or implied to be, real-world
accuracy for imagery validated here — no ground truth exists for arbitrary uploads, and this
module reports input-quality signals, not an accuracy estimate.**
"""
import cv2
import numpy as np

from src.analysis.area import LEVIR_CD_ORIGINAL_PIXEL_SIZE_M

REAL_WORLD_DISCLAIMER = (
    "Model trained on LEVIR-CD imagery. Performance on this imagery has not been independently "
    "validated."
)

# A pixel size this many times coarser than the LEVIR-CD training resolution is flagged as a
# resolution-mismatch warning — e.g. Sentinel-2's 10 m/pixel is 20x coarser than LEVIR-CD's
# 0.5 m/pixel (docs/REAL_WORLD_DEMO.md), exactly the domain gap already measured for Phase 11/18.
# This threshold (5x) is a documented, chosen default, not derived from a labeled sensitivity
# study — set loosely to avoid false alarms on imagery that's only modestly coarser.
RESOLUTION_MISMATCH_FACTOR = 5.0

# A phase-correlation-estimated shift beyond this many pixels is flagged as likely misregistered.
# LEVIR-CD's own pairs are pre-registered by the dataset authors; a few pixels of estimated shift
# on real imagery is within phase correlation's own noise floor, so this default is conservative.
REGISTRATION_SHIFT_WARNING_PX = 3.0

# Simple, explicitly-heuristic bright/washed-out pixel screen (near-white AND low-saturation).
# This is NOT a validated cloud detector — no labeled cloud-mask data exists in this project. It
# will miss thin/translucent clouds and can false-positive on bright rooftops, sand, snow, or
# overexposure. It exists to surface an honest "this image looks unusually bright in X% of pixels"
# warning, not to reliably identify clouds — stated explicitly in every warning it produces.
CLOUD_BRIGHTNESS_THRESHOLD = 200  # 0-255 per RGB channel, mean
CLOUD_SATURATION_THRESHOLD = 30   # 0-255, HSV saturation channel
CLOUD_WARNING_PERCENT = 5.0


def check_dimensions_match(before: np.ndarray, after: np.ndarray) -> dict:
    match = before.shape == after.shape
    return {
        "dimensions_match": match,
        "before_shape": tuple(before.shape),
        "after_shape": tuple(after.shape),
        "warning": None if match else (
            f"Before image shape {tuple(before.shape)} does not match after image shape "
            f"{tuple(after.shape)} — a mismatched pair is not a valid before/after comparison."
        ),
    }


def assess_resolution_plausibility(pixel_size_meters: float) -> dict:
    """Compares a claimed/known pixel size against LEVIR-CD's 0.5 m/pixel training resolution.
    Only meaningful when `pixel_size_meters` is independently known (e.g. real raster metadata via
    `src/geospatial/raster.py::read_raster_metadata`) — this function never guesses one."""
    factor = pixel_size_meters / LEVIR_CD_ORIGINAL_PIXEL_SIZE_M
    mismatch = factor >= RESOLUTION_MISMATCH_FACTOR
    return {
        "pixel_size_meters": pixel_size_meters,
        "training_pixel_size_meters": LEVIR_CD_ORIGINAL_PIXEL_SIZE_M,
        "resolution_ratio": factor,
        "resolution_mismatch_warning": mismatch,
        "warning": None if not mismatch else (
            f"This imagery's pixel size ({pixel_size_meters:.2f} m/pixel) is {factor:.1f}x coarser "
            f"than the LEVIR-CD training resolution (0.5 m/pixel) — the same order of domain gap "
            f"documented in docs/REAL_WORLD_DEMO.md for Sentinel-2 imagery. Fine-grained "
            f"building-outline detail the model learned to recognize may not be resolvable here."
        ),
    }


def estimate_registration_offset(before: np.ndarray, after: np.ndarray) -> dict:
    """Estimates the pixel shift between `before` and `after` via phase correlation
    (`cv2.phaseCorrelate`, a standard signal-processing technique) — a diagnostic for whether the
    pair looks pre-registered (LEVIR-CD's pairs are; arbitrary uploads are not guaranteed to be).
    Does NOT correct/align the images — only reports an estimated shift and a threshold-based
    warning; alignment correction is out of scope (see docs/LIMITATIONS.md)."""
    if before.shape[:2] != after.shape[:2]:
        return {
            "estimated_shift_x_px": None, "estimated_shift_y_px": None,
            "estimated_shift_magnitude_px": None, "likely_misregistered": None,
            "warning": "Cannot estimate registration offset: image dimensions do not match.",
        }

    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (shift_x, shift_y), _response = cv2.phaseCorrelate(before_gray, after_gray)

    magnitude = float(np.hypot(shift_x, shift_y))
    likely_misregistered = magnitude >= REGISTRATION_SHIFT_WARNING_PX
    return {
        "estimated_shift_x_px": round(float(shift_x), 2),
        "estimated_shift_y_px": round(float(shift_y), 2),
        "estimated_shift_magnitude_px": round(magnitude, 2),
        "likely_misregistered": likely_misregistered,
        "warning": None if not likely_misregistered else (
            f"Estimated ~{magnitude:.1f}px shift between before/after images (a phase-correlation "
            f"estimate, not a ground-truth measurement) — the pair may not be pixel-aligned, which "
            f"can produce false-positive change detections along real edges/boundaries that simply "
            f"moved between the two images rather than actually changed."
        ),
    }


def screen_for_cloud_cover(image: np.ndarray) -> dict:
    """Heuristic-only bright/washed-out pixel screen — see module docstring for why this is
    explicitly not a validated cloud detector. Reports the percentage of pixels flagged."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    brightness = image.astype(np.float32).mean(axis=2)

    flagged = (brightness > CLOUD_BRIGHTNESS_THRESHOLD) & (saturation < CLOUD_SATURATION_THRESHOLD)
    percent_flagged = float(flagged.mean() * 100.0)
    return {
        "percent_bright_lowsat_pixels": round(percent_flagged, 2),
        "heuristic_only": True,
        "warning": None if percent_flagged < CLOUD_WARNING_PERCENT else (
            f"{percent_flagged:.1f}% of pixels are unusually bright and low-saturation (a heuristic "
            f"proxy, not a validated cloud/shadow detector — see docs/LIMITATIONS.md). This may "
            f"indicate cloud cover, haze, or overexposure; predictions in these areas should be "
            f"treated with extra skepticism."
        ),
    }


def validate_real_world_input(before: np.ndarray, after: np.ndarray,
                               pixel_size_meters: float = None) -> dict:
    """Orchestrates every real-world input check and collects all non-None warnings into one list,
    for a caller (dashboard, script) to surface without re-deriving this logic (Rule 6)."""
    dims = check_dimensions_match(before, after)
    registration = estimate_registration_offset(before, after)
    cloud_before = screen_for_cloud_cover(before)
    cloud_after = screen_for_cloud_cover(after)
    resolution = assess_resolution_plausibility(pixel_size_meters) if pixel_size_meters is not None else None

    warnings = [w for w in (
        dims["warning"], registration["warning"], cloud_before["warning"], cloud_after["warning"],
        resolution["warning"] if resolution else None,
    ) if w]

    return {
        "disclaimer": REAL_WORLD_DISCLAIMER,
        "dimensions": dims,
        "registration": registration,
        "cloud_screen_before": cloud_before,
        "cloud_screen_after": cloud_after,
        "resolution": resolution,
        "warnings": warnings,
    }
