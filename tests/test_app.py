"""The dashboard renders from results/ without errors, in every tab and widget state."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def _run():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_renders_without_exceptions():
    at = _run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    assert at.title[0].value.startswith("Cross-asset")
    assert len(at.tabs) == 4  # Streamlit renders every tab's body on each run


def test_every_period_and_strategy_selection():
    at = _run()
    for period in ["out-of-sample", "in-sample", "full"]:
        at.radio[0].set_value(period).run()
        assert not at.exception, (period, at.exception)
    all_names = at.multiselect[0].options
    at.multiselect[0].set_value(all_names).run()
    assert not at.exception, at.exception
    at.multiselect[0].set_value([all_names[-1]]).run()   # a single strategy (ML)
    assert not at.exception, at.exception
