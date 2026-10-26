"""The dashboard renders from results/ without errors."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    assert at.title[0].value.startswith("Cross-asset")
