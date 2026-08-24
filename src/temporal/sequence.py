"""Phase 21: multi-temporal (more than 2 dates) change analysis — extends this project's core
two-image change detection to an ordered sequence of real Sentinel-2 acquisitions
(T1 -> T2 -> ... -> TN), computing per-interval change statistics/severity for each adjacent pair.

**No causal or tracking claims are made, anywhere in this module or its outputs.** Each interval's
prediction is an entirely independent binary-change-detection run on that one adjacent pair of
images — the underlying model (`models/siamese_unet.py`) has no mechanism to track a specific
physical object or change across more than two images, and this module adds none either. A
structure flagged as "changed" in interval 2 (T2->T3) and again in interval 4 (T4->T5) is NOT
asserted to be the same underlying event continuing, still under construction, or related in any
way — they are two separate, independent detections that happen to occupy similar pixels. This
module only ever reports per-interval, independently-detected change — never a trajectory, a rate
attributed to one tracked object, or any inference about *why* a change happened.

The existing two-image (single before/after pair) analysis path — `src/inference/predict.py`,
`src/analysis/*`, the dashboard's default mode — is completely unmodified by this module; a
multi-temporal sequence is just N-1 independent two-image analyses run back to back.
"""
from typing import List, Tuple

from src.analysis.severity import compute_severity_for_regions, severity_distribution
from src.analysis.statistics import compute_change_statistics


def select_temporal_sequence(items: list, n_dates: int) -> list:
    """Given STAC items sorted oldest-first (as returned by
    `src/geospatial/raster.py::search_sentinel2_items`), selects `n_dates` items spread as evenly
    as possible across the real available time span — not the first N or a random sample, so the
    resulting sequence genuinely covers the search range rather than clustering near one end.
    Deterministic: always picks the real item closest to each evenly-spaced target timestamp;
    never fabricates a date with no corresponding real acquisition."""
    if n_dates < 2:
        raise ValueError(f"n_dates must be >= 2 to form at least one interval, got {n_dates}")
    if len(items) < n_dates:
        raise ValueError(f"Only {len(items)} items available, cannot select {n_dates} distinct dates")

    first_ts = items[0].datetime.timestamp()
    last_ts = items[-1].datetime.timestamp()
    step = (last_ts - first_ts) / (n_dates - 1)
    targets = [first_ts + i * step for i in range(n_dates)]

    selected = []
    used_ids = set()
    for target in targets:
        for candidate in sorted(items, key=lambda it: abs(it.datetime.timestamp() - target)):
            if candidate.id not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate.id)
                break

    selected.sort(key=lambda it: it.datetime)
    return selected


def build_intervals(dated_items: list) -> List[Tuple]:
    """Pairs an ordered sequence into adjacent (from_item, to_item) intervals:
    [t0,t1,t2] -> [(t0,t1), (t1,t2)]. Each pair is later analyzed completely independently."""
    return list(zip(dated_items[:-1], dated_items[1:]))


def compute_interval_record(from_item, to_item, binary_mask, probability_map=None,
                             pixel_size_meters: float = None) -> dict:
    """Real per-interval change statistics + severity for one (from_item, to_item) pair, given
    that pair's already-computed prediction (`binary_mask`, optionally `probability_map`). Reuses
    `src/analysis/statistics.py` and `src/analysis/severity.py` unchanged (Rule 6) — this function
    only packages their real output plus the interval's real dates/item ids, nothing invented."""
    stats = compute_change_statistics(
        binary_mask, probability_map=probability_map, pixel_size_meters=pixel_size_meters,
    )
    regions = stats.pop("regions")
    scored_regions = compute_severity_for_regions(regions) if regions else []
    dist = severity_distribution(scored_regions) if scored_regions else {
        "region_count_by_category": {}, "changed_pixels_by_category": {},
    }

    return {
        "from_date": str(from_item.datetime.date()),
        "to_date": str(to_item.datetime.date()),
        "from_item_id": from_item.id,
        "to_item_id": to_item.id,
        **stats,
        "severity_distribution": dist,
    }
