"""Headless smoke tests for the Streamlit app and the CLI (no browser needed)."""
import subprocess
import sys

import pytest

from src.utils import ROOT, load_config, project_path

cfg = load_config()
RUN = cfg["serve"]["run"]
HAS_MODELS = (project_path(cfg["paths"]["models_dir"]) / RUN).exists()


def test_app_renders_three_tabs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("ThreatWeigh")
    assert [t.label for t in at.tabs] == ["Predict", "Model results", "How it works"]


@pytest.mark.skipif(not HAS_MODELS, reason=f"no trained models in models/{RUN}")
def test_app_predicts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.toggle[0].set_value(False)                  # skip Integrated Gradients to keep it fast
    predict = [b for b in at.button if b.label == "Predict"][0]
    predict.click().run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Base Score" in labels
    score = float([m.value for m in at.metric if m.label == "Base Score"][0])
    assert 0.0 <= score <= 10.0


@pytest.mark.skipif(not HAS_MODELS, reason=f"no trained models in models/{RUN}")
def test_cli_predicts():
    out = subprocess.run([sys.executable, str(ROOT / "predict.py"),
                          "Unauthenticated remote attackers can execute arbitrary code via a crafted HTTP request."],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CVSS:3.1/AV:" in out.stdout and "Base" in out.stdout
