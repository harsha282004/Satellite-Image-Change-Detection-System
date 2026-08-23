"""Pixel-count -> physical-area conversion.

`pixel_count_to_area` never assumes a physical pixel size silently — callers must always pass
`pixel_size_meters` explicitly (PROJECT_CONTEXT.md: "Never assume a physical pixel size without
documenting the assumption"). `levir_cd_effective_pixel_size` computes the one documented,
justified default this project actually uses for LEVIR-CD-derived predictions.
"""

LEVIR_CD_ORIGINAL_TILE_PIXELS = 1024
LEVIR_CD_ORIGINAL_PIXEL_SIZE_M = 0.5  # documented on the official LEVIR-CD project page (docs/DATASET.md, Phase 2)


def pixel_count_to_area(pixel_count: int, pixel_size_meters: float) -> dict:
    pixel_area_m2 = pixel_size_meters ** 2
    area_m2 = pixel_count * pixel_area_m2
    return {
        "area_m2": area_m2,
        "area_hectares": area_m2 / 10_000.0,
        "pixel_size_meters": pixel_size_meters,
    }


def levir_cd_effective_pixel_size(model_image_size: int) -> float:
    """Effective ground pixel size (meters) for a LEVIR-CD tile resized to `model_image_size`.

    LEVIR-CD tiles are 1024x1024 px at a documented 0.5 m/pixel — i.e. each tile covers a fixed
    512m x 512m ground footprint (docs/DATASET.md). This project's models operate on tiles resized
    to a smaller square (`configs/*.yaml` image_size, 256 by default — see docs/ARCHITECTURE.md).
    That resize does not change the ground footprint, only how many pixels it's spread across, so
    the *effective* pixel size at the model's resolution is larger than the original 0.5 m/pixel —
    reusing 0.5 m/pixel directly against a resized mask would silently overstate area by
    (1024 / model_image_size)^2 (16x too high for the default 256px model). This function makes
    that derivation explicit instead of leaving it as an easy-to-miss caller error.
    """
    ground_extent_m = LEVIR_CD_ORIGINAL_TILE_PIXELS * LEVIR_CD_ORIGINAL_PIXEL_SIZE_M
    return ground_extent_m / model_image_size
