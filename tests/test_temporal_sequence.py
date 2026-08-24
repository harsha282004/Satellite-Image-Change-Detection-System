"""Phase 21: tests use a tiny synthetic fake-STAC-item class (id + datetime only) so selection
logic is fast and network-independent, unlike the real `scripts/multitemporal_analysis.py`, which
depends on live Sentinel-2 access."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.temporal.sequence import build_intervals, compute_interval_record, select_temporal_sequence


class FakeItem:
    def __init__(self, item_id: str, dt: datetime):
        self.id = item_id
        self.datetime = dt


def make_items(dates):
    return [FakeItem(f"item_{d.isoformat()}", d) for d in dates]


def test_select_temporal_sequence_requires_at_least_two_dates():
    items = make_items([datetime(2020, 1, 1), datetime(2021, 1, 1)])
    with pytest.raises(ValueError):
        select_temporal_sequence(items, n_dates=1)


def test_select_temporal_sequence_requires_enough_items():
    items = make_items([datetime(2020, 1, 1), datetime(2021, 1, 1)])
    with pytest.raises(ValueError):
        select_temporal_sequence(items, n_dates=5)


def test_select_temporal_sequence_picks_endpoints_for_two_dates():
    dates = [datetime(2017, 1, 1) + timedelta(days=30 * i) for i in range(50)]
    items = make_items(dates)
    selected = select_temporal_sequence(items, n_dates=2)
    assert len(selected) == 2
    assert selected[0].datetime == dates[0]
    assert selected[1].datetime == dates[-1]


def test_select_temporal_sequence_spreads_evenly_across_real_span():
    """Not the first N items — genuinely spread across the full available time range."""
    dates = [datetime(2017, 1, 1) + timedelta(days=10 * i) for i in range(300)]  # ~8 years
    items = make_items(dates)
    selected = select_temporal_sequence(items, n_dates=5)
    assert len(selected) == 5
    # chronologically ordered
    for a, b in zip(selected, selected[1:]):
        assert a.datetime < b.datetime
    # span covers most of the real range, not clustered near one end
    total_span_days = (dates[-1] - dates[0]).days
    selected_span_days = (selected[-1].datetime - selected[0].datetime).days
    assert selected_span_days > total_span_days * 0.9


def test_select_temporal_sequence_never_duplicates_an_item():
    dates = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
    items = make_items(dates)
    selected = select_temporal_sequence(items, n_dates=3)
    assert len({it.id for it in selected}) == 3


def test_build_intervals_pairs_adjacent_dates():
    dates = [datetime(2017, 1, 1), datetime(2019, 1, 1), datetime(2021, 1, 1), datetime(2023, 1, 1)]
    items = make_items(dates)
    intervals = build_intervals(items)
    assert len(intervals) == 3
    assert intervals[0] == (items[0], items[1])
    assert intervals[1] == (items[1], items[2])
    assert intervals[2] == (items[2], items[3])


def test_compute_interval_record_reports_real_dates_and_stats():
    from_item = FakeItem("before_id", datetime(2019, 6, 1))
    to_item = FakeItem("after_id", datetime(2021, 6, 1))

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 1  # one 6x6 region
    probs = np.full((20, 20), 0.8, dtype=np.float32)

    record = compute_interval_record(from_item, to_item, mask, probability_map=probs)

    assert record["from_date"] == "2019-06-01"
    assert record["to_date"] == "2021-06-01"
    assert record["from_item_id"] == "before_id"
    assert record["to_item_id"] == "after_id"
    assert record["num_regions"] == 1
    assert record["total_changed_pixels"] == 36
    assert "severity_distribution" in record
    assert sum(record["severity_distribution"]["region_count_by_category"].values()) == 1


def test_compute_interval_record_handles_no_change():
    from_item = FakeItem("before_id", datetime(2019, 6, 1))
    to_item = FakeItem("after_id", datetime(2021, 6, 1))
    mask = np.zeros((10, 10), dtype=np.uint8)

    record = compute_interval_record(from_item, to_item, mask)

    assert record["num_regions"] == 0
    assert record["total_changed_pixels"] == 0
    assert record["severity_distribution"]["region_count_by_category"] == {}
