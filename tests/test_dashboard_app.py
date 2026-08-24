"""Phase 23: regression test for the unified dashboard, using Streamlit's own official in-process
`AppTest` harness — a genuine execution of the real `dashboard/app.py` script (not a mock), run
without a browser. Catches import errors, top-level exceptions, and Streamlit API misuse (e.g. a
removed/renamed method) that only surface when the script actually runs.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_dashboard_runs_with_no_top_level_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception, f"Dashboard raised: {[str(e) for e in at.exception]}"


def test_dashboard_renders_all_five_tabs():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert len(at.tabs) == 5


def test_dashboard_shows_scientific_disclaimer():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    info_texts = " ".join(i.value for i in at.info)
    assert "scientific disclaimer" in info_texts.lower() or "unvalidated" in info_texts.lower()


def test_dashboard_model_selectbox_has_all_options():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    selectboxes = at.sidebar.selectbox
    assert len(selectboxes) >= 1
    model_select = selectboxes[0]
    assert len(model_select.options) == 8  # 7 CNN variants + the Phase 20 Transformer
    assert any("Transformer" in opt for opt in model_select.options)
