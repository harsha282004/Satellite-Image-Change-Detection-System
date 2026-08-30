"""Regression tests for the unified, redesigned dashboard, using Streamlit's own official
in-process `AppTest` harness — a genuine execution of the real `dashboard/app.py` and each real
`dashboard/app_pages/*.py` file (not a mock), run without a browser or file-upload simulation
(AppTest does not support simulating `st.file_uploader`, so the upload-dependent branches of
Change Detection are instead covered by a direct pipeline smoke test — see
`test_change_detection_pipeline.py`).
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")

PAGE_PATHS = [
    "app_pages/change_detection.py",
    "app_pages/model_analysis.py",
    "app_pages/geospatial.py",
    "app_pages/temporal.py",
    "app_pages/diagnostics.py",
]


def test_entry_point_and_overview_run_with_no_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception, f"Dashboard raised: {[str(e) for e in at.exception]}"


def test_every_page_runs_with_no_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    for page_path in PAGE_PATHS:
        at.switch_page(page_path)
        at.run(timeout=60)
        assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"


def test_sidebar_model_selectbox_has_all_eight_models():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    selectboxes = at.sidebar.selectbox
    assert len(selectboxes) >= 1
    model_select = selectboxes[0]
    assert len(model_select.options) == 8
    assert any("Transformer" in opt for opt in model_select.options)
    assert any("Recommended" in opt for opt in model_select.options)


def test_no_phase_numbers_leak_into_main_page_text():
    """The redesign's core requirement: internal development-phase references must not appear on
    the user-facing pages (Overview, Change Detection, Model Analysis, Geospatial, Temporal)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    user_facing_pages = ["", *PAGE_PATHS[:-1]]  # everything except Diagnostics
    for page_path in user_facing_pages:
        if page_path:
            at.switch_page(page_path)
            at.run(timeout=60)
        page_text = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
        for phase_marker in ("Phase 1", "Phase 2", "phase-1", "phase-2"):
            assert phase_marker not in page_text, f"{page_path or 'overview'} leaked {phase_marker!r}"


def test_overview_shows_real_flagship_metrics():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    metric_labels = {m.label for m in at.metric}
    assert {"IoU", "Dice", "F1 score", "Accuracy"}.issubset(metric_labels)


def test_diagnostics_page_has_no_upload_prompt():
    """Diagnostics is a read-only technical page — it must not require or offer an upload."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.switch_page("app_pages/diagnostics.py")
    at.run(timeout=60)
    assert not at.exception
    assert len(at.get("file_uploader")) == 0
