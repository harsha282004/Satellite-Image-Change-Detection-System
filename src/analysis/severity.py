"""Phase 17: change severity scoring.

A transparent, fully documented, configurable analytical score derived entirely from measurable
model outputs (region geometry + the model's own prediction probability). It is explicitly
**NOT ground truth** and **NOT a physical/environmental severity assessment** — no field validation,
no damage assessment, no causal claim about what the change is. It is a way to rank/triage the
regions a model already detected, nothing more. This disclaimer is repeated everywhere a severity
score is displayed (dashboard, docs), not stated once and forgotten.

## The formula (every constant is named and documented, not hidden)

    severity_score = 100 * (
        w_area        * min(1, region.pixel_count / AREA_REFERENCE_PIXELS)
      + w_probability * region.mean_prediction_probability
      + w_density     * region.change_density
      + w_relative_size * (region.pixel_count / total_changed_pixels_in_image)
    )

- **area_score**: the region's own size, normalized against `AREA_REFERENCE_PIXELS` (default 500px
  — chosen as a round, documented reference point representing a "large" detected region at this
  project's ~2 m/pixel effective resolution (~2000 m²), NOT derived from any labeled ground-truth
  severity study — capped at 1.0 so no single huge region can exceed the maximum score alone.
- **probability_score**: the model's own mean sigmoid output within the region — higher model
  output is treated as a more severe/salient detection. This inherits every caveat from Phase 15's
  "prediction probability ≠ calibrated confidence" — it is included as a real, measurable model
  signal, not because it has been verified to track true probability-of-correctness.
- **density_score**: `change_density` (Phase 16) — how compact/solid vs. sparse/scattered the
  region's shape is within its bounding box.
- **relative_size_score**: this region's share of all changed pixels detected in the same image —
  contextualizes a region's size against how much change exists in that specific image overall.

Default weights (`DEFAULT_WEIGHTS`) sum to 1.0 and are themselves a documented, adjustable choice
(area and probability weighted most heavily), not derived from any ground-truth severity labels —
none exist for this task.
"""

DEFAULT_WEIGHTS = {"area": 0.35, "probability": 0.30, "density": 0.15, "relative_size": 0.20}
DEFAULT_AREA_REFERENCE_PIXELS = 500

# Ordered low-to-high; a score >= a category's threshold (and below the next category's) falls
# into that category. Configurable — not scientifically derived, documented as such.
DEFAULT_CATEGORY_THRESHOLDS = {"Low": 0.0, "Moderate": 25.0, "High": 50.0, "Very High": 75.0}


def severity_category(score: float, thresholds: dict = None) -> str:
    thresholds = thresholds or DEFAULT_CATEGORY_THRESHOLDS
    ordered = sorted(thresholds.items(), key=lambda kv: kv[1], reverse=True)
    for name, cutoff in ordered:
        if score >= cutoff:
            return name
    return ordered[-1][0]  # fallback: lowest category


def compute_region_severity(region: dict, total_changed_pixels_in_image: int,
                             weights: dict = None,
                             area_reference_pixels: int = DEFAULT_AREA_REFERENCE_PIXELS,
                             category_thresholds: dict = None) -> dict:
    """Returns a dict with `severity_score` (0-100), `severity_category`, and the individual
    component scores (0-1 each) that produced it — full transparency into the formula's inputs,
    not just the final number."""
    weights = weights or DEFAULT_WEIGHTS

    area_score = min(1.0, region["pixel_count"] / area_reference_pixels)
    probability_score = float(region.get("mean_prediction_probability", 0.0))
    density_score = float(region.get("change_density", 0.0))
    relative_size_score = region["pixel_count"] / max(total_changed_pixels_in_image, 1)
    relative_size_score = min(1.0, relative_size_score)

    raw_score = (
        weights["area"] * area_score
        + weights["probability"] * probability_score
        + weights["density"] * density_score
        + weights["relative_size"] * relative_size_score
    )
    severity_score = max(0.0, min(100.0, 100.0 * raw_score))

    return {
        "severity_score": severity_score,
        "severity_category": severity_category(severity_score, category_thresholds),
        "component_scores": {
            "area_score": area_score,
            "probability_score": probability_score,
            "density_score": density_score,
            "relative_size_score": relative_size_score,
        },
    }


def compute_severity_for_regions(regions: list, weights: dict = None,
                                  area_reference_pixels: int = DEFAULT_AREA_REFERENCE_PIXELS,
                                  category_thresholds: dict = None) -> list:
    """Computes severity for every region in `regions` (from
    `src/analysis/regions.py::extract_regions`) and returns new dicts (region data + severity
    fields merged) — does not mutate the input list/dicts."""
    total_changed_pixels = sum(r["pixel_count"] for r in regions)
    result = []
    for region in regions:
        severity = compute_region_severity(
            region, total_changed_pixels, weights=weights,
            area_reference_pixels=area_reference_pixels, category_thresholds=category_thresholds,
        )
        result.append({**region, **severity})
    return result


def severity_distribution(scored_regions: list) -> dict:
    """Count of regions per severity category, and total changed-pixel area per category —
    Phase 17's "severity distribution" / "affected area by severity" dashboard requirement."""
    counts = {}
    area_by_category = {}
    for r in scored_regions:
        cat = r["severity_category"]
        counts[cat] = counts.get(cat, 0) + 1
        area_by_category[cat] = area_by_category.get(cat, 0) + r["pixel_count"]
    return {"region_count_by_category": counts, "changed_pixels_by_category": area_by_category}


def highest_severity_regions(scored_regions: list, n: int = 5) -> list:
    return sorted(scored_regions, key=lambda r: r["severity_score"], reverse=True)[:n]
