"""The committed README must be exactly what analysis/findings.py generates from results/."""
import importlib.util

from xasset import config


def test_readme_is_generated_from_results():
    spec = importlib.util.spec_from_file_location("findings", config.ROOT / "analysis" / "findings.py")
    findings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(findings)
    expected = findings.render(findings.load())
    actual = (config.ROOT / "README.md").read_text(encoding="utf-8")
    assert actual == expected, "README.md is stale or hand-edited: run python analysis/findings.py"
